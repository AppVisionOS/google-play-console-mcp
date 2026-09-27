"""Google Play Console MCP Server – reviews, monetization, vitals, financial reports."""

import contextvars
import csv
import inspect
import io
import json
import os
import time
import zipfile
from datetime import datetime, timezone
from functools import wraps
from typing import Any

import httpx
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from google.cloud import storage
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "Google Play Console MCP",
    instructions=(
        "Google Play Console API – reviews, subscriptions, in-app products, "
        "orders, releases, Android Vitals, and financial reports from Claude Code."
    ),
)

# ---------------------------------------------------------------------------
# Per-request credential override
# ---------------------------------------------------------------------------

_gpc_sa_override: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_gpc_sa_override", default=None
)

_original_tool = mcp.tool


def _tool_with_auth(*tool_args, **tool_kwargs):
    """Wrapper that injects an optional ``service_account_json`` parameter into every tool."""
    orig_decorator = _original_tool(*tool_args, **tool_kwargs)

    def wrapper(func):
        sig = inspect.signature(func)
        auth_param = inspect.Parameter(
            "service_account_json", inspect.Parameter.KEYWORD_ONLY, default=None,
            annotation=str | dict | None,
        )
        new_sig = sig.replace(
            parameters=list(sig.parameters.values()) + [auth_param]
        )

        @wraps(func)
        def inner(*args, **kw):
            sa = kw.pop("service_account_json", None)
            token = _gpc_sa_override.set(sa)
            try:
                return func(*args, **kw)
            finally:
                _gpc_sa_override.reset(token)

        inner.__signature__ = new_sig
        inner.__annotations__ = {
            **getattr(func, "__annotations__", {}),
            "service_account_json": str | dict | None,
        }
        return orig_decorator(inner)

    return wrapper


mcp.tool = _tool_with_auth  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Configuration & Auth Helpers
# ---------------------------------------------------------------------------

SCOPES = [
    "https://www.googleapis.com/auth/androidpublisher",
    "https://www.googleapis.com/auth/playdeveloperreporting",
    "https://www.googleapis.com/auth/cloud-platform",
]

REPORTING_BASE = "https://playdeveloperreporting.googleapis.com/v1beta1"
TIMEOUT = 30.0
MAX_RETRIES = 3


def _env(name: str) -> str:
    """Read a required environment variable."""
    val = os.environ.get(name, "")
    if not val:
        raise RuntimeError(
            f"Environment variable {name} is not set. "
            f"Please configure it before using this tool."
        )
    return val


def _pkg(override: str | None = None) -> str:
    """Return the package name – explicit override or env default."""
    if override:
        return override
    return _env("GOOGLE_PLAY_PACKAGE_NAME")


def _get_credentials() -> service_account.Credentials:
    """Build Google credentials from the service account key file or override."""
    override = _gpc_sa_override.get()
    if override:
        info = override if isinstance(override, dict) else json.loads(override)
        return service_account.Credentials.from_service_account_info(
            info, scopes=SCOPES
        )
    key_path = _env("GOOGLE_PLAY_SERVICE_ACCOUNT_KEY")
    creds = service_account.Credentials.from_service_account_file(
        key_path, scopes=SCOPES
    )
    return creds


def _get_publisher_service():
    """Build the androidpublisher v3 service client."""
    creds = _get_credentials()
    return build("androidpublisher", "v3", credentials=creds, cache_discovery=False)


def _commit_edit_with_fallback(svc, pkg: str, edit_id: str) -> dict[str, Any]:
    """Commit an edit, retrying with changesNotSentForReview when Play refuses auto-review.

    Play answers 400 \"Changes cannot be sent for review automatically\" for apps whose
    edits must be submitted by hand. Retrying with changesNotSentForReview=true stores the
    changes as a pending draft instead of failing the whole write.
    """
    try:
        return dict(svc.edits().commit(packageName=pkg, editId=edit_id).execute() or {})
    except HttpError as exc:
        if "changesNotSentForReview" not in str(exc):
            raise
        resp = dict(svc.edits().commit(
            packageName=pkg, editId=edit_id, changesNotSentForReview=True
        ).execute() or {})
        resp["changesNotSentForReview"] = True
        resp["review_note"] = "Committed without automatic review - submit the changes in Play Console."
        return resp

def _get_gcs_client(legacy: bool = False) -> storage.Client:
    """Build a GCS client from the service account key.

    If legacy=True and GOOGLE_PLAY_LEGACY_SERVICE_ACCOUNT_KEY is set,
    use the legacy service account for accessing the old GCS bucket.
    Falls back to the primary service account if no legacy key is configured.
    """
    if legacy:
        key_path = os.environ.get("GOOGLE_PLAY_LEGACY_SERVICE_ACCOUNT_KEY", "")
        if key_path:
            return storage.Client.from_service_account_json(key_path)
    key_path = _env("GOOGLE_PLAY_SERVICE_ACCOUNT_KEY")
    return storage.Client.from_service_account_json(key_path)


def _get_auth_token() -> str:
    """Get a fresh OAuth2 access token for REST calls (Reporting API)."""
    creds = _get_credentials()
    creds.refresh(Request())
    return creds.token


def _reporting_post(path: str, body: dict | None = None) -> dict:
    """POST to the Play Developer Reporting API with retry on 429."""
    token = _get_auth_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    url = f"{REPORTING_BASE}{path}"

    for attempt in range(MAX_RETRIES):
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(url, headers=headers, json=body or {})
        if resp.status_code == 429:
            wait = 2 ** attempt
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()

    raise httpx.HTTPStatusError(
        "Rate limited after retries",
        request=httpx.Request("POST", url),
        response=resp,  # type: ignore[possibly-undefined]
    )


def _reporting_get(path: str, params: dict | None = None) -> dict:
    """GET from the Play Developer Reporting API with retry on 429."""
    token = _get_auth_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    url = f"{REPORTING_BASE}{path}"

    for attempt in range(MAX_RETRIES):
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.get(url, headers=headers, params=params)
        if resp.status_code == 429:
            wait = 2 ** attempt
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()

    raise httpx.HTTPStatusError(
        "Rate limited after retries",
        request=httpx.Request("GET", url),
        response=resp,  # type: ignore[possibly-undefined]
    )


def _fmt(data: object, max_items: int = 500) -> str:
    """Format data as readable JSON string, capping lists at *max_items*."""
    if isinstance(data, list):
        capped = data[:max_items]
        suffix = (
            f"\n... ({len(data) - max_items} more items)"
            if len(data) > max_items
            else ""
        )
        return json.dumps(capped, indent=2, default=str) + suffix
    return json.dumps(data, indent=2, default=str)


def _date_to_reporting(date_str: str) -> dict:
    """Convert 'YYYY-MM-DD' to Reporting API DateTime object (snake_case)."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return {"year": dt.year, "month": dt.month, "day": dt.day, "time_zone": {"id": "America/Los_Angeles"}}


# =========================================================================
# READ TOOLS — App Management
# =========================================================================


@mcp.tool()
def list_apps(package_name: str = "") -> str:
    """Validate and get details for the configured app (or a specific package).

    Returns app title, default language, and basic info from the store listing.

    Args:
        package_name: Android package name (e.g. com.example.app).
                      Uses GOOGLE_PLAY_PACKAGE_NAME env var if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()

        # Get the app edit to read details
        edit = svc.edits().insert(packageName=pkg, body={}).execute()
        edit_id = edit["id"]

        try:
            details = svc.edits().details().get(
                packageName=pkg, editId=edit_id
            ).execute()

            listings_resp = svc.edits().listings().list(
                packageName=pkg, editId=edit_id
            ).execute()
            listings = listings_resp.get("listings", [])

            result = {
                "package_name": pkg,
                "default_language": details.get("defaultLanguage"),
                "contact_email": details.get("contactEmail"),
                "contact_phone": details.get("contactPhone"),
                "contact_website": details.get("contactWebsite"),
                "listings_count": len(listings),
                "languages": [l.get("language") for l in listings],
            }
        finally:
            # Clean up the edit
            try:
                svc.edits().delete(packageName=pkg, editId=edit_id).execute()
            except Exception:
                pass

        return _fmt(result)
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def get_app_details(package_name: str = "", language: str = "en-US") -> str:
    """Get store listing details: title, description, short description, category.

    Args:
        package_name: Android package name. Uses env default if empty.
        language: BCP-47 language code for the listing (default en-US).
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()

        edit = svc.edits().insert(packageName=pkg, body={}).execute()
        edit_id = edit["id"]

        try:
            listing = svc.edits().listings().get(
                packageName=pkg, editId=edit_id, language=language
            ).execute()

            result = {
                "package_name": pkg,
                "language": listing.get("language"),
                "title": listing.get("title"),
                "short_description": listing.get("shortDescription"),
                "full_description": listing.get("fullDescription"),
                "video": listing.get("video"),
            }
        finally:
            try:
                svc.edits().delete(packageName=pkg, editId=edit_id).execute()
            except Exception:
                pass

        return _fmt(result)
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


# =========================================================================
# READ TOOLS — Reviews
# =========================================================================


@mcp.tool()
def list_reviews(
    package_name: str = "",
    max_results: int = 20,
    token: str = "",
    translation_language: str = "",
) -> str:
    """List user reviews for the app.

    Args:
        package_name: Android package name. Uses env default if empty.
        max_results: Number of reviews to return (max 100).
        token: Pagination token from previous response.
        translation_language: BCP-47 code to translate reviews into (e.g. 'en').
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()

        kwargs: dict[str, Any] = {
            "packageName": pkg,
            "maxResults": min(max_results, 100),
        }
        if token:
            kwargs["token"] = token
        if translation_language:
            kwargs["translationLanguage"] = translation_language

        resp = svc.reviews().list(**kwargs).execute()
        reviews = resp.get("reviews", [])
        next_token = resp.get("tokenPagination", {}).get("nextPageToken")

        results = []
        for r in reviews:
            comments = r.get("comments", [])
            user_comment = comments[0].get("userComment", {}) if comments else {}
            dev_comment = comments[0].get("developerComment", {}) if len(comments) > 1 else (
                comments[1].get("developerComment", {}) if len(comments) > 1 else {}
            )

            results.append({
                "review_id": r.get("reviewId"),
                "author": r.get("authorName"),
                "rating": user_comment.get("starRating"),
                "text": user_comment.get("text"),
                "language": user_comment.get("reviewerLanguage"),
                "device": user_comment.get("device"),
                "os_version": user_comment.get("androidOsVersion"),
                "app_version_code": user_comment.get("appVersionCode"),
                "app_version_name": user_comment.get("appVersionName"),
                "last_modified": user_comment.get("lastModified", {}).get("seconds"),
                "thumbs_up": user_comment.get("thumbsUpCount"),
                "thumbs_down": user_comment.get("thumbsDownCount"),
                "has_dev_reply": bool(dev_comment),
            })

        output: dict[str, Any] = {"reviews": results, "total_returned": len(results)}
        if next_token:
            output["next_token"] = next_token

        return _fmt(output)
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def get_review(review_id: str, package_name: str = "", translation_language: str = "") -> str:
    """Get details of a single review.

    Args:
        review_id: The review ID to fetch.
        package_name: Android package name. Uses env default if empty.
        translation_language: BCP-47 code to translate the review into.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()

        kwargs: dict[str, Any] = {"packageName": pkg, "reviewId": review_id}
        if translation_language:
            kwargs["translationLanguage"] = translation_language

        r = svc.reviews().get(**kwargs).execute()
        comments = r.get("comments", [])
        user_comment = comments[0].get("userComment", {}) if comments else {}

        result = {
            "review_id": r.get("reviewId"),
            "author": r.get("authorName"),
            "rating": user_comment.get("starRating"),
            "text": user_comment.get("text"),
            "language": user_comment.get("reviewerLanguage"),
            "device": user_comment.get("device"),
            "os_version": user_comment.get("androidOsVersion"),
            "app_version_code": user_comment.get("appVersionCode"),
            "app_version_name": user_comment.get("appVersionName"),
            "last_modified": user_comment.get("lastModified", {}).get("seconds"),
            "thumbs_up": user_comment.get("thumbsUpCount"),
            "thumbs_down": user_comment.get("thumbsDownCount"),
            "comments": comments,
        }
        return _fmt(result)
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


# =========================================================================
# WRITE TOOLS — Reviews
# =========================================================================


@mcp.tool()
def reply_to_review(review_id: str, reply_text: str, package_name: str = "") -> str:
    """Reply to a user review. This is a WRITE operation visible to the user.

    Args:
        review_id: The review ID to reply to.
        reply_text: The reply text to post.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()

        resp = svc.reviews().reply(
            packageName=pkg,
            reviewId=review_id,
            body={"replyText": reply_text},
        ).execute()

        result_comment = resp.get("result", {}).get("replyComment", {})
        return _fmt({
            "status": "replied",
            "review_id": review_id,
            "reply_text": result_comment.get("text"),
            "last_edited": result_comment.get("lastEdited", {}).get("seconds"),
        })
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


# =========================================================================
# READ TOOLS — Monetization / Products
# =========================================================================


@mcp.tool()
def list_subscriptions(package_name: str = "", page_size: int = 50, page_token: str = "") -> str:
    """List subscription products defined in the app.

    Args:
        package_name: Android package name. Uses env default if empty.
        page_size: Number of results per page (max 100).
        page_token: Pagination token from previous response.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()

        kwargs: dict[str, Any] = {
            "packageName": pkg,
            "pageSize": min(page_size, 100),
        }
        if page_token:
            kwargs["pageToken"] = page_token

        resp = svc.monetization().subscriptions().list(**kwargs).execute()
        subscriptions = resp.get("subscriptions", [])
        next_token = resp.get("nextPageToken")

        results = []
        for sub in subscriptions:
            base_plans = sub.get("basePlans", [])
            results.append({
                "product_id": sub.get("productId"),
                "package_name": sub.get("packageName"),
                "archived": sub.get("archived", False),
                "base_plans_count": len(base_plans),
                "base_plans": [
                    {
                        "base_plan_id": bp.get("basePlanId"),
                        "state": bp.get("state"),
                        "auto_renewing": bp.get("autoRenewingBasePlanType") is not None,
                        "prepaid": bp.get("prepaidBasePlanType") is not None,
                    }
                    for bp in base_plans
                ],
                "listings": [
                    {
                        "language": l.get("languageCode"),
                        "title": l.get("title"),
                    }
                    for l in sub.get("listings", [])
                ],
            })

        output: dict[str, Any] = {"subscriptions": results, "total_returned": len(results)}
        if next_token:
            output["next_page_token"] = next_token

        return _fmt(output)
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def get_subscription(product_id: str, package_name: str = "") -> str:
    """Get details of a specific subscription product including base plans and pricing.

    Args:
        product_id: The subscription product ID.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()

        sub = svc.monetization().subscriptions().get(
            packageName=pkg, productId=product_id
        ).execute()

        base_plans = sub.get("basePlans", [])
        bp_details = []
        for bp in base_plans:
            offers_info = []
            # Try to get offers for this base plan
            try:
                offers_resp = svc.monetization().subscriptions().basePlans().offers().list(
                    packageName=pkg,
                    productId=product_id,
                    basePlanId=bp.get("basePlanId"),
                ).execute()
                for offer in offers_resp.get("subscriptionOffers", []):
                    phases = []
                    for phase in offer.get("phases", []):
                        price = phase.get("regionalConfigs", [{}])[0].get("price", {})
                        phases.append({
                            "duration": phase.get("duration"),
                            "recurrence_count": phase.get("recurrenceCount"),
                            "price_currency": price.get("currencyCode"),
                            "price_units": price.get("units"),
                            "price_nanos": price.get("nanos"),
                        })
                    offers_info.append({
                        "offer_id": offer.get("offerId"),
                        "state": offer.get("state"),
                        "phases": phases,
                    })
            except HttpError:
                pass

            ar = bp.get("autoRenewingBasePlanType", {})
            pp = bp.get("prepaidBasePlanType", {})

            bp_details.append({
                "base_plan_id": bp.get("basePlanId"),
                "state": bp.get("state"),
                "billing_period": ar.get("billingPeriodDuration") if ar else None,
                "grace_period": ar.get("gracePeriodDuration") if ar else None,
                "resubscribe_state": ar.get("resubscribeState") if ar else None,
                "prepaid_duration": pp.get("billingPeriodDuration") if pp else None,
                "regional_configs": bp.get("regionalConfigs", []),
                "offers": offers_info,
            })

        result = {
            "product_id": sub.get("productId"),
            "package_name": sub.get("packageName"),
            "archived": sub.get("archived", False),
            "listings": sub.get("listings", []),
            "base_plans": bp_details,
        }
        return _fmt(result)
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def list_inapp_products(package_name: str = "", max_results: int = 100, token: str = "") -> str:
    """List one-time in-app products (managed products).

    Args:
        package_name: Android package name. Uses env default if empty.
        max_results: Number of results (max 100).
        token: Pagination token from previous response.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()

        kwargs: dict[str, Any] = {
            "packageName": pkg,
            "maxResults": min(max_results, 100),
        }
        if token:
            kwargs["token"] = token

        resp = svc.inappproducts().list(**kwargs).execute()
        products = resp.get("inappproduct", [])
        next_token = resp.get("tokenPagination", {}).get("nextPageToken")

        results = []
        for p in products:
            default_price = p.get("defaultPrice", {})
            results.append({
                "sku": p.get("sku"),
                "status": p.get("status"),
                "purchase_type": p.get("purchaseType"),
                "default_language": p.get("defaultLanguage"),
                "default_price_currency": default_price.get("currency"),
                "default_price_micros": default_price.get("priceMicros"),
                "title": p.get("listings", {}).get("en-US", {}).get("title")
                    or next(iter(p.get("listings", {}).values()), {}).get("title"),
            })

        output: dict[str, Any] = {"products": results, "total_returned": len(results)}
        if next_token:
            output["next_token"] = next_token

        return _fmt(output)
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def get_inapp_product(sku: str, package_name: str = "") -> str:
    """Get details of a specific in-app product.

    Args:
        sku: The product SKU / ID.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()

        p = svc.inappproducts().get(packageName=pkg, sku=sku).execute()

        default_price = p.get("defaultPrice", {})
        prices = p.get("prices", {})

        result = {
            "sku": p.get("sku"),
            "status": p.get("status"),
            "purchase_type": p.get("purchaseType"),
            "default_language": p.get("defaultLanguage"),
            "default_price_currency": default_price.get("currency"),
            "default_price_micros": default_price.get("priceMicros"),
            "listings": p.get("listings", {}),
            "prices_by_region": {
                region: {
                    "currency": price.get("currency"),
                    "price_micros": price.get("priceMicros"),
                }
                for region, price in prices.items()
            },
            "grace_period": p.get("gracePeriod"),
        }
        return _fmt(result)
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


# =========================================================================
# READ TOOLS — Orders & Purchases
# =========================================================================


@mcp.tool()
def list_orders(
    package_name: str = "",
    start_time: str = "",
    end_time: str = "",
    max_results: int = 50,
    token: str = "",
) -> str:
    """List orders (purchases) for the app within a date range.

    Args:
        package_name: Android package name. Uses env default if empty.
        start_time: Start timestamp in ISO 8601 (e.g. 2024-01-01T00:00:00Z).
        end_time: End timestamp in ISO 8601. Defaults to now.
        max_results: Number of results (max 100).
        token: Pagination token from previous response.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()

        kwargs: dict[str, Any] = {
            "packageName": pkg,
            "maxResults": min(max_results, 100),
        }
        if start_time:
            kwargs["startTime"] = start_time
        if end_time:
            kwargs["endTime"] = end_time
        if token:
            kwargs["token"] = token

        resp = svc.orders().list(**kwargs).execute()
        orders = resp.get("orders", [])
        next_token = resp.get("tokenPagination", {}).get("nextPageToken")

        results = []
        for o in orders:
            results.append({
                "order_id": o.get("orderId"),
                "product_id": o.get("productId"),
                "purchase_type": o.get("purchaseType"),
                "order_time": o.get("creationTime"),
                "financial_status": o.get("financialStatus"),
                "order_state": o.get("orderState"),
                "currency": o.get("currency"),
                "amount_micros": o.get("amountMicros"),
                "tax_micros": o.get("taxMicros"),
                "refund_amount_micros": o.get("refundAmountMicros"),
            })

        output: dict[str, Any] = {"orders": results, "total_returned": len(results)}
        if next_token:
            output["next_token"] = next_token

        return _fmt(output)
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def get_purchase_status(purchase_token: str, product_id: str, package_name: str = "") -> str:
    """Verify and get details of an in-app product purchase.

    Args:
        purchase_token: The purchase token from the client.
        product_id: The in-app product ID.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()

        p = svc.purchases().products().get(
            packageName=pkg, productId=product_id, token=purchase_token
        ).execute()

        result = {
            "kind": p.get("kind"),
            "purchase_time_millis": p.get("purchaseTimeMillis"),
            "purchase_state": p.get("purchaseState"),
            "consumption_state": p.get("consumptionState"),
            "developer_payload": p.get("developerPayload"),
            "order_id": p.get("orderId"),
            "acknowledgement_state": p.get("acknowledgementState"),
            "purchase_type": p.get("purchaseType"),
            "quantity": p.get("quantity"),
            "region_code": p.get("regionCode"),
        }
        return _fmt(result)
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def get_subscription_purchase(
    purchase_token: str,
    product_id: str,
    package_name: str = "",
) -> str:
    """Get subscription purchase details (v2 API) – renewal chain, pricing, cancellation info.

    Args:
        purchase_token: The purchase token from the client.
        product_id: The subscription product ID.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()

        p = svc.purchases().subscriptionsv2().get(
            packageName=pkg, token=purchase_token
        ).execute()

        line_items = []
        for item in p.get("lineItems", []):
            auto_renewing = item.get("autoRenewingPlan", {})
            prepaid = item.get("prepaidPlan", {})
            offer = item.get("offerDetails", {})

            line_items.append({
                "product_id": item.get("productId"),
                "expiry_time": item.get("expiryTime"),
                "auto_renewing_plan": {
                    "auto_renew_enabled": auto_renewing.get("autoRenewEnabled"),
                } if auto_renewing else None,
                "prepaid_plan": {
                    "allowExtendAfterTime": prepaid.get("allowExtendAfterTime"),
                } if prepaid else None,
                "offer_details": {
                    "offer_id": offer.get("offerId"),
                    "base_plan_id": offer.get("basePlanId"),
                } if offer else None,
            })

        result = {
            "kind": p.get("kind"),
            "region_code": p.get("regionCode"),
            "start_time": p.get("startTime"),
            "subscription_state": p.get("subscriptionState"),
            "latest_order_id": p.get("latestOrderId"),
            "linked_purchase_token": p.get("linkedPurchaseToken"),
            "acknowledgement_state": p.get("acknowledgementState"),
            "external_account_identifiers": p.get("externalAccountIdentifiers"),
            "line_items": line_items,
            "canceled_state_context": p.get("canceledStateContext"),
            "test_purchase": p.get("testPurchase"),
        }
        return _fmt(result)
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def list_voided_purchases(
    package_name: str = "",
    start_time: str = "",
    end_time: str = "",
    max_results: int = 50,
    token: str = "",
    voided_source: int = 0,
) -> str:
    """List voided (refunded/chargebacked) purchases.

    Args:
        package_name: Android package name. Uses env default if empty.
        start_time: Voided after this time (milliseconds since epoch).
        end_time: Voided before this time (milliseconds since epoch).
        max_results: Number of results (max 100).
        token: Pagination token from previous response.
        voided_source: 0 = all, 1 = developer, 2 = Google.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()

        kwargs: dict[str, Any] = {
            "packageName": pkg,
            "maxResults": min(max_results, 100),
        }
        if start_time:
            kwargs["startTime"] = str(start_time)
        if end_time:
            kwargs["endTime"] = str(end_time)
        if token:
            kwargs["token"] = token
        if voided_source:
            kwargs["type"] = voided_source

        resp = svc.purchases().voidedpurchases().list(**kwargs).execute()
        voided = resp.get("voidedPurchases", [])
        next_token = resp.get("tokenPagination", {}).get("nextPageToken")

        results = []
        for v in voided:
            results.append({
                "order_id": v.get("orderId"),
                "purchase_token": v.get("purchaseToken"),
                "purchase_time_millis": v.get("purchaseTimeMillis"),
                "voided_time_millis": v.get("voidedTimeMillis"),
                "voided_source": v.get("voidedSource"),
                "voided_reason": v.get("voidedReason"),
                "kind": v.get("kind"),
            })

        output: dict[str, Any] = {"voided_purchases": results, "total_returned": len(results)}
        if next_token:
            output["next_token"] = next_token

        return _fmt(output)
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


# =========================================================================
# READ TOOLS — Releases
# =========================================================================


@mcp.tool()
def list_tracks(package_name: str = "") -> str:
    """List release tracks (production, beta, alpha, internal, custom).

    Args:
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()

        edit = svc.edits().insert(packageName=pkg, body={}).execute()
        edit_id = edit["id"]

        try:
            resp = svc.edits().tracks().list(
                packageName=pkg, editId=edit_id
            ).execute()
            tracks = resp.get("tracks", [])

            results = []
            for t in tracks:
                releases = t.get("releases", [])
                results.append({
                    "track": t.get("track"),
                    "releases_count": len(releases),
                    "latest_release": {
                        "name": releases[0].get("name") if releases else None,
                        "status": releases[0].get("status") if releases else None,
                        "version_codes": releases[0].get("versionCodes", []) if releases else [],
                    } if releases else None,
                })
        finally:
            try:
                svc.edits().delete(packageName=pkg, editId=edit_id).execute()
            except Exception:
                pass

        return _fmt(results)
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def get_track_releases(track: str = "production", package_name: str = "") -> str:
    """Get release details for a specific track.

    Args:
        track: Track name (production, beta, alpha, internal, or custom track name).
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()

        edit = svc.edits().insert(packageName=pkg, body={}).execute()
        edit_id = edit["id"]

        try:
            t = svc.edits().tracks().get(
                packageName=pkg, editId=edit_id, track=track
            ).execute()

            releases = t.get("releases", [])
            results = []
            for rel in releases:
                results.append({
                    "name": rel.get("name"),
                    "status": rel.get("status"),
                    "version_codes": rel.get("versionCodes", []),
                    "release_notes": [
                        {
                            "language": note.get("language"),
                            "text": note.get("text"),
                        }
                        for note in rel.get("releaseNotes", [])
                    ],
                    "user_fraction": rel.get("userFraction"),
                    "in_app_update_priority": rel.get("inAppUpdatePriority"),
                })
        finally:
            try:
                svc.edits().delete(packageName=pkg, editId=edit_id).execute()
            except Exception:
                pass

        return _fmt({"track": track, "releases": results})
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


# =========================================================================
# READ TOOLS — Android Vitals (Reporting API)
# =========================================================================


def _build_vitals_query(
    package_name: str,
    metric_set: str,
    metrics: list[str],
    dimensions: list[str],
    start_date: str,
    end_date: str,
    page_size: int = 1000,
) -> tuple[str, dict]:
    """Build the Reporting API query path and body."""
    path = f"/apps/{package_name}/{metric_set}:query"
    body: dict[str, Any] = {
        "metrics": metrics,
        "dimensions": dimensions,
        "timeline_spec": {
            "aggregation_period": "DAILY",
            "start_time": _date_to_reporting(start_date),
            "end_time": _date_to_reporting(end_date),
        },
        "page_size": page_size,
    }
    return path, body


def _parse_vitals_response(resp: dict) -> list[dict]:
    """Parse Reporting API response rows into readable dicts."""
    rows = resp.get("rows", [])
    results = []
    for row in rows:
        entry: dict[str, Any] = {}
        # Dimensions
        for dim in row.get("dimensions", []):
            dim_val = dim.get("stringValue") or dim.get("int64Value")
            entry[dim.get("dimension", "unknown")] = dim_val
        # Metrics
        for met in row.get("metrics", []):
            metric_name = met.get("metric", "unknown")
            dec_val = met.get("decimalValue")
            if dec_val is not None:
                entry[metric_name] = dec_val.get("value")
            else:
                entry[metric_name] = met.get("int64Value")
        # Start time
        start = row.get("startTime", {})
        if start:
            d = start.get("year", "")
            if d:
                entry["date"] = f"{start.get('year')}-{start.get('month', 1):02d}-{start.get('day', 1):02d}"
        results.append(entry)
    return results


@mcp.tool()
def get_crash_rate(
    start_date: str,
    end_date: str,
    package_name: str = "",
    page_size: int = 100,
) -> str:
    """Get crash rate metrics from Android Vitals.

    Args:
        start_date: Start date YYYY-MM-DD.
        end_date: End date YYYY-MM-DD.
        package_name: Android package name. Uses env default if empty.
        page_size: Max rows to return (default 100).
    """
    try:
        pkg = _pkg(package_name or None)
        path, body = _build_vitals_query(
            package_name=pkg,
            metric_set="crashRateMetricSet",
            metrics=["crashRate", "userPerceivedCrashRate", "distinctUsers"],
            dimensions=["versionCode"],
            start_date=start_date,
            end_date=end_date,
            page_size=page_size,
        )
        resp = _reporting_post(path, body)
        results = _parse_vitals_response(resp)
        return _fmt(results) if results else "No crash rate data found."
    except httpx.HTTPStatusError as exc:
        return f"Error: {exc.response.status_code} – {exc.response.text}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def get_anr_rate(
    start_date: str,
    end_date: str,
    package_name: str = "",
    page_size: int = 100,
) -> str:
    """Get ANR (Application Not Responding) rate metrics from Android Vitals.

    Args:
        start_date: Start date YYYY-MM-DD.
        end_date: End date YYYY-MM-DD.
        package_name: Android package name. Uses env default if empty.
        page_size: Max rows to return (default 100).
    """
    try:
        pkg = _pkg(package_name or None)
        path, body = _build_vitals_query(
            package_name=pkg,
            metric_set="anrRateMetricSet",
            metrics=["anrRate", "userPerceivedAnrRate", "distinctUsers"],
            dimensions=["versionCode"],
            start_date=start_date,
            end_date=end_date,
            page_size=page_size,
        )
        resp = _reporting_post(path, body)
        results = _parse_vitals_response(resp)
        return _fmt(results) if results else "No ANR rate data found."
    except httpx.HTTPStatusError as exc:
        return f"Error: {exc.response.status_code} – {exc.response.text}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def get_vitals_overview(
    start_date: str,
    end_date: str,
    package_name: str = "",
) -> str:
    """Get a combined Android Vitals overview – crash, ANR, excessive wakeups,
    stuck wake locks, slow start, and slow rendering rates.

    Queries all six metric sets and returns them in a single response.

    Args:
        start_date: Start date YYYY-MM-DD.
        end_date: End date YYYY-MM-DD.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)

        metric_sets = [
            {
                "name": "crashRateMetricSet",
                "metrics": ["crashRate", "userPerceivedCrashRate", "distinctUsers"],
            },
            {
                "name": "anrRateMetricSet",
                "metrics": ["anrRate", "userPerceivedAnrRate", "distinctUsers"],
            },
            {
                "name": "excessiveWakeupRateMetricSet",
                "metrics": ["excessiveWakeupRate", "distinctUsers"],
            },
            {
                "name": "stuckBackgroundWakelockRateMetricSet",
                "metrics": ["stuckBgWakelockRate", "distinctUsers"],
            },
            {
                "name": "slowStartRateMetricSet",
                "metrics": ["slowStartRate", "distinctUsers"],
            },
            {
                "name": "slowRenderingRateMetricSet",
                "metrics": ["slowRenderingRate20Fps", "slowRenderingRate30Fps", "distinctUsers"],
            },
        ]

        overview: dict[str, Any] = {"period": f"{start_date} to {end_date}", "metrics": {}}

        for ms in metric_sets:
            try:
                path, body = _build_vitals_query(
                    package_name=pkg,
                    metric_set=ms["name"],
                    metrics=ms["metrics"],
                    dimensions=[],
                    start_date=start_date,
                    end_date=end_date,
                    page_size=100,
                )
                resp = _reporting_post(path, body)
                rows = _parse_vitals_response(resp)
                overview["metrics"][ms["name"]] = rows if rows else "no data"
            except (httpx.HTTPStatusError, Exception) as e:
                overview["metrics"][ms["name"]] = f"error: {e}"

        return _fmt(overview)
    except RuntimeError as exc:
        return str(exc)


# =========================================================================
# READ TOOLS — Financial Reports (GCS)
# =========================================================================


def _decode_report(data: bytes) -> str:
    """Decode a Play report: stats exports are UTF-16, financial reports UTF-8."""
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16")
    return data.decode("utf-8-sig")


def _read_gcs_csv(bucket_name: str, blob_path: str, legacy: bool = False) -> list[dict]:
    """Download a report from GCS and parse it into a list of dicts.

    Financial reports are published as zips with a suffixed name
    (earnings/earnings_202608_<id>-N.zip, sales/salesreport_202608.zip), so
    when *blob_path* doesn't exist every blob sharing its stem is read and the
    rows concatenated — a month can have one earnings file per payments profile.
    """
    client = _get_gcs_client(legacy=legacy)
    blob = client.bucket(bucket_name).blob(blob_path)
    if blob.exists():
        blobs = [blob]
    else:
        stem = blob_path.removesuffix(".csv")
        blobs = [
            b for b in client.list_blobs(bucket_name, prefix=stem)
            if b.name.endswith((".csv", ".zip"))
        ]
        if not blobs:
            raise FileNotFoundError(f"gs://{bucket_name}/{stem}[*.csv|*.zip] not found")

    rows: list[dict] = []
    for b in blobs:
        data = b.download_as_bytes()
        if b.name.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                texts = [_decode_report(zf.read(n)) for n in zf.namelist() if n.endswith(".csv")]
        else:
            texts = [_decode_report(data)]
        for text in texts:
            rows.extend(csv.DictReader(io.StringIO(text)))
    return rows


def _row_package(row: dict) -> str:
    """Package name of a financial report row (column name varies by report/era)."""
    return row.get("Package ID") or row.get("Product ID") or row.get("Product id") or ""


def _gcs_bucket_name() -> str:
    """Build the GCS bucket name for Play Console financial reports."""
    dev_id = _env("GOOGLE_PLAY_DEVELOPER_ID")
    return f"pubsite_prod_{dev_id}"


def _gcs_legacy_bucket_name() -> str | None:
    """Build the legacy GCS bucket name if configured."""
    dev_id = os.environ.get("GOOGLE_PLAY_LEGACY_DEVELOPER_ID", "")
    if dev_id:
        return f"pubsite_prod_{dev_id}"
    return None


def _read_gcs_csv_with_fallback(blob_path: str) -> list[dict]:
    """Try primary bucket first, fall back to legacy bucket if not found.

    This supports apps transferred between Google Play Console accounts
    where historical financial data remains in the old account's GCS bucket.
    """
    bucket_name = _gcs_bucket_name()
    try:
        return _read_gcs_csv(bucket_name, blob_path)
    except Exception as primary_exc:
        legacy_bucket = _gcs_legacy_bucket_name()
        if not legacy_bucket:
            raise
        try:
            return _read_gcs_csv(legacy_bucket, blob_path, legacy=True)
        except Exception as legacy_exc:
            # Surface both failures — reporting only the legacy one hides the real cause.
            raise RuntimeError(
                f"primary {bucket_name}: {primary_exc} | legacy {legacy_bucket}: {legacy_exc}"
            ) from legacy_exc


@mcp.tool()
def get_sales_report(year: int, month: int, package_name: str = "") -> str:
    """Get monthly sales report from Google Play financial reports (GCS).

    Returns transaction-level data: orders, refunds, taxes, fees.

    Args:
        year: Report year (e.g. 2024).
        month: Report month (1-12).
        package_name: Filter by package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        blob_path = f"sales/salesreport_{year}{month:02d}.csv"

        rows = _read_gcs_csv_with_fallback(blob_path)

        # Filter by package name if provided
        if pkg:
            rows = [r for r in rows if _row_package(r).startswith(pkg)]

        results = []
        for r in rows:
            results.append({
                "order_id": r.get("Order Number") or r.get("Order Charged Date"),
                "product_id": _row_package(r),
                "sku_id": r.get("SKU ID") or r.get("Sku Id"),
                "product_type": r.get("Product Type"),
                "description": r.get("Description") or r.get("Product Title"),
                "currency": r.get("Currency of Sale") or r.get("Buyer Currency"),
                "amount": r.get("Item Price") or r.get("Charged Amount"),
                "tax": r.get("Taxes Collected"),
                "transaction_type": r.get("Transaction Type") or r.get("Financial Status"),
                "country": r.get("Buyer Country") or r.get("Country of Buyer"),
                "state": r.get("Buyer State") or r.get("State of Buyer") or r.get("Buyer Postal Code"),
            })

        return _fmt({
            "period": f"{year}-{month:02d}",
            "total_rows": len(results),
            "transactions": results,
        })
    except Exception as exc:
        return f"Error reading sales report: {exc}"


@mcp.tool()
def get_earnings_report(year: int, month: int, package_name: str = "") -> str:
    """Get monthly earnings report from Google Play financial reports (GCS).

    Returns what Google actually pays out after fees and taxes.

    Args:
        year: Report year (e.g. 2024).
        month: Report month (1-12).
        package_name: Filter by package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        blob_path = f"earnings/earnings_{year}{month:02d}.csv"

        rows = _read_gcs_csv_with_fallback(blob_path)

        if pkg:
            rows = [r for r in rows if _row_package(r).startswith(pkg)]

        results = []
        for r in rows:
            results.append({
                "product_id": _row_package(r),
                "sku_id": r.get("Sku Id") or r.get("SKU ID"),
                "product_type": r.get("Product Type"),
                "description": r.get("Description"),
                "currency": r.get("Buyer Currency"),
                "amount_buyer": r.get("Amount (Buyer Currency)"),
                "currency_conversion_rate": r.get("Currency Conversion Rate"),
                "amount_merchant": r.get("Amount (Merchant Currency)"),
                "merchant_currency": r.get("Merchant Currency"),
                "transaction_type": r.get("Transaction Type"),
                "country": r.get("Buyer Country"),
            })

        return _fmt({
            "period": f"{year}-{month:02d}",
            "total_rows": len(results),
            "earnings": results,
        })
    except Exception as exc:
        return f"Error reading earnings report: {exc}"


@mcp.tool()
def get_revenue_summary(year: int, month: int, package_name: str = "") -> str:
    """Get revenue summary grouped by product and country from earnings report.

    Aggregates earnings data to show total revenue per product per country.

    Args:
        year: Report year (e.g. 2024).
        month: Report month (1-12).
        package_name: Filter by package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        blob_path = f"earnings/earnings_{year}{month:02d}.csv"

        rows = _read_gcs_csv_with_fallback(blob_path)

        if pkg:
            rows = [r for r in rows if _row_package(r).startswith(pkg)]

        # Aggregate by product + country; total_amount is net payout (charges
        # minus Google fees and refunds), by_transaction_type keeps the gross split.
        summary: dict[str, dict[str, Any]] = {}
        by_type: dict[str, float] = {}
        for r in rows:
            product = _row_package(r) or "unknown"
            country = r.get("Buyer Country") or "unknown"
            key = f"{product}|{country}"

            merchant_amount_str = r.get("Amount (Merchant Currency)", "0")
            try:
                amount = float(str(merchant_amount_str).replace(",", ""))
            except (ValueError, TypeError):
                amount = 0.0

            if key not in summary:
                summary[key] = {
                    "product_id": product,
                    "country": country,
                    "merchant_currency": r.get("Merchant Currency", ""),
                    "total_amount": 0.0,
                    "transaction_count": 0,
                }
            summary[key]["total_amount"] += amount
            summary[key]["transaction_count"] += 1
            tx_type = r.get("Transaction Type") or "unknown"
            by_type[tx_type] = by_type.get(tx_type, 0.0) + amount

        results = sorted(summary.values(), key=lambda x: x["total_amount"], reverse=True)
        for r in results:
            r["total_amount"] = round(r["total_amount"], 2)

        total_revenue = sum(r["total_amount"] for r in results)

        return _fmt({
            "period": f"{year}-{month:02d}",
            "total_revenue": round(total_revenue, 2),
            "by_transaction_type": {k: round(v, 2) for k, v in by_type.items()},
            "currency": results[0]["merchant_currency"] if results else "N/A",
            "products_count": len(set(r["product_id"] for r in results)),
            "countries_count": len(set(r["country"] for r in results)),
            "breakdown": results,
        })
    except Exception as exc:
        return f"Error reading revenue summary: {exc}"


# =========================================================================
# WRITE TOOLS — Purchase Management
# =========================================================================


@mcp.tool()
def acknowledge_product_purchase(
    purchase_token: str,
    product_id: str,
    package_name: str = "",
) -> str:
    """Acknowledge an in-app product purchase. Must be done within 3 days.

    Args:
        purchase_token: The purchase token from the client.
        product_id: The in-app product ID.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        svc.purchases().products().acknowledge(
            packageName=pkg, productId=product_id, token=purchase_token,
            body={},
        ).execute()
        return _fmt({"status": "acknowledged", "product_id": product_id})
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def consume_product_purchase(
    purchase_token: str,
    product_id: str,
    package_name: str = "",
) -> str:
    """Consume a consumable in-app product purchase (mark as used).

    Args:
        purchase_token: The purchase token from the client.
        product_id: The in-app product ID.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        svc.purchases().products().consume(
            packageName=pkg, productId=product_id, token=purchase_token,
        ).execute()
        return _fmt({"status": "consumed", "product_id": product_id})
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def acknowledge_subscription(
    purchase_token: str,
    product_id: str,
    package_name: str = "",
) -> str:
    """Acknowledge a subscription purchase. Must be done within 3 days.

    Args:
        purchase_token: The purchase token from the client.
        product_id: The subscription product ID.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        svc.purchases().subscriptions().acknowledge(
            packageName=pkg, subscriptionId=product_id, token=purchase_token,
            body={},
        ).execute()
        return _fmt({"status": "acknowledged", "subscription_id": product_id})
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def cancel_subscription(
    purchase_token: str,
    product_id: str,
    package_name: str = "",
) -> str:
    """Cancel a subscription. Access continues until end of current billing period.

    Args:
        purchase_token: The purchase token from the client.
        product_id: The subscription product ID.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        svc.purchases().subscriptions().cancel(
            packageName=pkg, subscriptionId=product_id, token=purchase_token,
        ).execute()
        return _fmt({"status": "cancelled", "subscription_id": product_id})
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def refund_subscription(
    purchase_token: str,
    product_id: str,
    package_name: str = "",
) -> str:
    """Refund a subscription. Access continues until end of current billing period.

    Args:
        purchase_token: The purchase token from the client.
        product_id: The subscription product ID.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        svc.purchases().subscriptions().refund(
            packageName=pkg, subscriptionId=product_id, token=purchase_token,
        ).execute()
        return _fmt({"status": "refunded", "subscription_id": product_id})
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def revoke_subscription(
    purchase_token: str,
    product_id: str,
    package_name: str = "",
) -> str:
    """Revoke a subscription – refund AND immediately revoke access.

    Args:
        purchase_token: The purchase token from the client.
        product_id: The subscription product ID.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        svc.purchases().subscriptions().revoke(
            packageName=pkg, subscriptionId=product_id, token=purchase_token,
        ).execute()
        return _fmt({"status": "revoked", "subscription_id": product_id})
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def defer_subscription(
    purchase_token: str,
    product_id: str,
    expected_expiry_time_millis: int,
    desired_expiry_time_millis: int,
    package_name: str = "",
) -> str:
    """Defer a subscription's next billing date.

    Args:
        purchase_token: The purchase token from the client.
        product_id: The subscription product ID.
        expected_expiry_time_millis: Current expected expiry time in millis since epoch.
        desired_expiry_time_millis: New desired expiry time in millis since epoch.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        resp = svc.purchases().subscriptions().defer(
            packageName=pkg, subscriptionId=product_id, token=purchase_token,
            body={
                "deferralInfo": {
                    "expectedExpiryTimeMillis": expected_expiry_time_millis,
                    "desiredExpiryTimeMillis": desired_expiry_time_millis,
                }
            },
        ).execute()
        return _fmt({
            "status": "deferred",
            "subscription_id": product_id,
            "new_expiry_time_millis": resp.get("newExpiryTimeMillis"),
        })
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


# =========================================================================
# WRITE TOOLS — In-App Products CRUD
# =========================================================================


@mcp.tool()
def create_inapp_product(
    sku: str,
    default_price_currency: str,
    default_price_micros: str,
    title: str,
    description: str = "",
    language: str = "en-US",
    purchase_type: str = "managedUser",
    package_name: str = "",
) -> str:
    """Create a new in-app product (managed product).

    Args:
        sku: Product SKU/ID (unique within the app).
        default_price_currency: Currency code (e.g. 'USD', 'TRY').
        default_price_micros: Price in micros (e.g. '990000' for 0.99).
        title: Product title.
        description: Product description.
        language: BCP-47 language code for the listing (default en-US).
        purchase_type: 'managedUser' (non-consumable) or 'subscription'.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()

        body = {
            "sku": sku,
            "status": "active",
            "purchaseType": purchase_type,
            "defaultPrice": {
                "currency": default_price_currency,
                "priceMicros": default_price_micros,
            },
            "listings": {
                language: {
                    "title": title,
                    "description": description or title,
                }
            },
            "defaultLanguage": language,
        }

        result = svc.inappproducts().insert(
            packageName=pkg, body=body
        ).execute()
        return _fmt({"status": "created", "sku": result.get("sku"), "product": result})
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def update_inapp_product(
    sku: str,
    default_price_currency: str = "",
    default_price_micros: str = "",
    title: str = "",
    description: str = "",
    language: str = "en-US",
    package_name: str = "",
) -> str:
    """Update an existing in-app product.

    Args:
        sku: Product SKU/ID to update.
        default_price_currency: New currency code.
        default_price_micros: New price in micros.
        title: New product title.
        description: New product description.
        language: BCP-47 language code for the listing.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()

        # Get current product
        current = svc.inappproducts().get(packageName=pkg, sku=sku).execute()

        # Update fields
        if default_price_currency and default_price_micros:
            current["defaultPrice"] = {
                "currency": default_price_currency,
                "priceMicros": default_price_micros,
            }
        if title:
            listings = current.get("listings", {})
            if language not in listings:
                listings[language] = {}
            listings[language]["title"] = title
            if description:
                listings[language]["description"] = description
            current["listings"] = listings

        result = svc.inappproducts().update(
            packageName=pkg, sku=sku, body=current
        ).execute()
        return _fmt({"status": "updated", "sku": result.get("sku")})
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def delete_inapp_product(sku: str, package_name: str = "") -> str:
    """Delete an in-app product. This is permanent.

    Args:
        sku: Product SKU/ID to delete.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        svc.inappproducts().delete(packageName=pkg, sku=sku).execute()
        return _fmt({"status": "deleted", "sku": sku})
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


# =========================================================================
# WRITE TOOLS — Subscription Products CRUD
# =========================================================================


@mcp.tool()
def create_subscription_product(
    product_id: str,
    listings_json: str,
    package_name: str = "",
) -> str:
    """Create a new subscription product.

    Args:
        product_id: Subscription product ID (unique within the app).
        listings_json: JSON string of listings, e.g. '[{"languageCode":"en-US","title":"Premium Monthly"}]'.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        listings = json.loads(listings_json)

        body = {
            "productId": product_id,
            "listings": listings,
        }

        result = svc.monetization().subscriptions().create(
            packageName=pkg, body=body, productId=product_id,
        ).execute()
        return _fmt({"status": "created", "product_id": result.get("productId"), "subscription": result})
    except json.JSONDecodeError as exc:
        return f"Invalid JSON in listings_json: {exc}"
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def update_subscription_product(
    product_id: str,
    listings_json: str = "",
    package_name: str = "",
) -> str:
    """Update an existing subscription product.

    Args:
        product_id: Subscription product ID.
        listings_json: JSON string of updated listings.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()

        body: dict[str, Any] = {"productId": product_id}
        if listings_json:
            body["listings"] = json.loads(listings_json)

        result = svc.monetization().subscriptions().patch(
            packageName=pkg, productId=product_id, body=body,
        ).execute()
        return _fmt({"status": "updated", "product_id": result.get("productId")})
    except json.JSONDecodeError as exc:
        return f"Invalid JSON in listings_json: {exc}"
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def archive_subscription_product(product_id: str, package_name: str = "") -> str:
    """Archive a subscription product. Archived products can't be purchased.

    Args:
        product_id: Subscription product ID to archive.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        result = svc.monetization().subscriptions().archive(
            packageName=pkg, productId=product_id, body={},
        ).execute()
        return _fmt({"status": "archived", "product_id": product_id, "result": result})
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


# =========================================================================
# READ/WRITE TOOLS — Base Plans & Offers
# =========================================================================


@mcp.tool()
def list_base_plan_offers(
    product_id: str,
    base_plan_id: str,
    package_name: str = "",
) -> str:
    """List all offers for a subscription base plan.

    Args:
        product_id: Subscription product ID.
        base_plan_id: Base plan ID.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()

        resp = svc.monetization().subscriptions().basePlans().offers().list(
            packageName=pkg, productId=product_id, basePlanId=base_plan_id,
        ).execute()

        offers = resp.get("subscriptionOffers", [])
        results = []
        for offer in offers:
            phases = []
            for phase in offer.get("phases", []):
                prices = []
                for rc in phase.get("regionalConfigs", []):
                    price = rc.get("price", {})
                    prices.append({
                        "region": rc.get("regionCode"),
                        "currency": price.get("currencyCode"),
                        "units": price.get("units"),
                        "nanos": price.get("nanos"),
                    })
                phases.append({
                    "duration": phase.get("duration"),
                    "recurrence_count": phase.get("recurrenceCount"),
                    "prices": prices,
                })
            results.append({
                "offer_id": offer.get("offerId"),
                "state": offer.get("state"),
                "phases": phases,
                "targeting": offer.get("targeting"),
            })

        return _fmt({"offers": results, "total": len(results)})
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def activate_base_plan(
    product_id: str,
    base_plan_id: str,
    package_name: str = "",
) -> str:
    """Activate a subscription base plan, making it available for purchase.

    Args:
        product_id: Subscription product ID.
        base_plan_id: Base plan ID to activate.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        result = svc.monetization().subscriptions().basePlans().activate(
            packageName=pkg, productId=product_id, basePlanId=base_plan_id,
            body={},
        ).execute()
        return _fmt({"status": "activated", "base_plan_id": base_plan_id, "result": result})
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def deactivate_base_plan(
    product_id: str,
    base_plan_id: str,
    package_name: str = "",
) -> str:
    """Deactivate a subscription base plan. Existing subscribers keep access.

    Args:
        product_id: Subscription product ID.
        base_plan_id: Base plan ID to deactivate.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        result = svc.monetization().subscriptions().basePlans().deactivate(
            packageName=pkg, productId=product_id, basePlanId=base_plan_id,
            body={},
        ).execute()
        return _fmt({"status": "deactivated", "base_plan_id": base_plan_id, "result": result})
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


# =========================================================================
# READ/WRITE TOOLS — External Transactions
# =========================================================================


@mcp.tool()
def create_external_transaction(
    external_transaction_id: str,
    original_price_currency: str,
    original_price_micros: str,
    transaction_time: str,
    package_name: str = "",
) -> str:
    """Report an external transaction (alternative billing).

    Args:
        external_transaction_id: Unique external transaction ID.
        original_price_currency: Currency code (e.g. 'USD').
        original_price_micros: Price in micros as string.
        transaction_time: ISO 8601 timestamp of the transaction.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        body = {
            "originalPreTaxAmount": {
                "currency": original_price_currency,
                "priceMicros": original_price_micros,
            },
            "originalTaxAmount": {
                "currency": original_price_currency,
                "priceMicros": "0",
            },
            "transactionTime": transaction_time,
            "oneTimeTransaction": {},
        }
        result = svc.externaltransactions().createexternaltransaction(
            packageName=pkg, body=body,
            externalTransactionId=external_transaction_id,
        ).execute()
        return _fmt({"status": "created", "transaction": result})
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def refund_external_transaction(
    external_transaction_id: str,
    package_name: str = "",
) -> str:
    """Refund an external transaction.

    Args:
        external_transaction_id: The external transaction ID to refund.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        result = svc.externaltransactions().refundexternaltransaction(
            packageName=pkg, externalTransactionId=external_transaction_id,
            body={"fullRefund": {}},
        ).execute()
        return _fmt({"status": "refunded", "transaction": result})
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def get_external_transaction(
    external_transaction_id: str,
    package_name: str = "",
) -> str:
    """Get details of an external transaction.

    Args:
        external_transaction_id: The external transaction ID to retrieve.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        result = svc.externaltransactions().getexternaltransaction(
            packageName=pkg, externalTransactionId=external_transaction_id,
        ).execute()
        return _fmt(result)
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


# =========================================================================
# READ TOOLS — Store Listing
# =========================================================================


@mcp.tool()
def get_store_listing(language: str, package_name: str = "") -> str:
    """Get the current store listing for a specific language.

    Read-before-write helper. Creates a transient edit, reads the listing,
    and abandons the edit (no mutation).

    Args:
        language: BCP-47 language code (e.g. 'en-US', 'tr-TR').
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()

        edit = svc.edits().insert(packageName=pkg, body={}).execute()
        edit_id = edit["id"]

        try:
            listing = svc.edits().listings().get(
                packageName=pkg, editId=edit_id, language=language
            ).execute()
            return _fmt(listing)
        finally:
            try:
                svc.edits().delete(packageName=pkg, editId=edit_id).execute()
            except Exception:
                pass

    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def list_store_listings(package_name: str = "") -> str:
    """List all store listings (per-locale) for an app.

    Returns the language codes plus a snippet of each listing.
    Creates a transient edit, lists, and abandons (no mutation).

    Args:
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()

        edit = svc.edits().insert(packageName=pkg, body={}).execute()
        edit_id = edit["id"]

        try:
            resp = svc.edits().listings().list(
                packageName=pkg, editId=edit_id
            ).execute()
            listings = resp.get("listings", [])
            return _fmt({
                "package": pkg,
                "total_locales": len(listings),
                "listings": listings,
            })
        finally:
            try:
                svc.edits().delete(packageName=pkg, editId=edit_id).execute()
            except Exception:
                pass

    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


# =========================================================================
# WRITE TOOLS — Store Listing
# =========================================================================


@mcp.tool()
def create_store_listing(
    language: str,
    title: str,
    short_description: str,
    full_description: str,
    video: str = "",
    package_name: str = "",
) -> str:
    """Create a NEW store listing for a language that doesn't have one yet.

    Errors out if a listing already exists for the language — use
    update_store_listing for existing locales.

    Args:
        language: BCP-47 language code (e.g. 'fr-FR', 'es-419').
        title: App title (max 30 chars).
        short_description: Short description (max 80 chars).
        full_description: Full description (max 4000 chars).
        video: Optional YouTube video URL.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()

        edit = svc.edits().insert(packageName=pkg, body={}).execute()
        edit_id = edit["id"]

        try:
            try:
                existing = svc.edits().listings().get(
                    packageName=pkg, editId=edit_id, language=language
                ).execute()
                if existing:
                    return (
                        f"Listing for '{language}' already exists. "
                        f"Use update_store_listing to modify it."
                    )
            except HttpError as get_err:
                if get_err.status_code != 404:
                    raise

            body: dict[str, Any] = {
                "language": language,
                "title": title,
                "shortDescription": short_description,
                "fullDescription": full_description,
            }
            if video:
                body["video"] = video

            result = svc.edits().listings().update(
                packageName=pkg, editId=edit_id, language=language, body=body
            ).execute()

            _commit_edit_with_fallback(svc, pkg, edit_id)

            return _fmt({
                "status": "created_and_committed",
                "language": language,
                "title": result.get("title"),
            })
        except Exception:
            try:
                svc.edits().delete(packageName=pkg, editId=edit_id).execute()
            except Exception:
                pass
            raise

    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def update_store_listing(
    language: str,
    title: str = "",
    short_description: str = "",
    full_description: str = "",
    video: str = "",
    package_name: str = "",
) -> str:
    """Update app store listing for a specific language.

    This creates an edit, updates the listing, and commits it.

    Args:
        language: BCP-47 language code (e.g. 'en-US', 'tr-TR').
        title: App title (max 30 chars).
        short_description: Short description (max 80 chars).
        full_description: Full description (max 4000 chars).
        video: YouTube video URL for the listing.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()

        edit = svc.edits().insert(packageName=pkg, body={}).execute()
        edit_id = edit["id"]

        try:
            # Get current listing
            try:
                current = svc.edits().listings().get(
                    packageName=pkg, editId=edit_id, language=language
                ).execute()
            except HttpError:
                current = {"language": language}

            body: dict[str, Any] = {"language": language}
            body["title"] = title or current.get("title", "")
            body["shortDescription"] = short_description or current.get("shortDescription", "")
            body["fullDescription"] = full_description or current.get("fullDescription", "")
            if video:
                body["video"] = video

            result = svc.edits().listings().update(
                packageName=pkg, editId=edit_id, language=language, body=body
            ).execute()

            _commit_edit_with_fallback(svc, pkg, edit_id)

            return _fmt({
                "status": "updated_and_committed",
                "language": language,
                "title": result.get("title"),
            })
        except Exception:
            try:
                svc.edits().delete(packageName=pkg, editId=edit_id).execute()
            except Exception:
                pass
            raise

    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


# =========================================================================
# READ TOOLS — App Versions
# =========================================================================


@mcp.tool()
def list_app_versions(package_name: str = "") -> str:
    """List uploaded APK and App Bundle versions.

    Args:
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()

        edit = svc.edits().insert(packageName=pkg, body={}).execute()
        edit_id = edit["id"]

        try:
            bundles = []
            try:
                resp = svc.edits().bundles().list(
                    packageName=pkg, editId=edit_id
                ).execute()
                for b in resp.get("bundles", []):
                    bundles.append({
                        "version_code": b.get("versionCode"),
                        "sha256": b.get("sha256"),
                    })
            except HttpError:
                pass

            apks = []
            try:
                resp = svc.edits().apks().list(
                    packageName=pkg, editId=edit_id
                ).execute()
                for a in resp.get("apks", []):
                    binary = a.get("binary", {})
                    apks.append({
                        "version_code": a.get("versionCode"),
                        "sha256": binary.get("sha256"),
                    })
            except HttpError:
                pass
        finally:
            try:
                svc.edits().delete(packageName=pkg, editId=edit_id).execute()
            except Exception:
                pass

        return _fmt({
            "bundles": bundles,
            "apks": apks,
            "total_bundles": len(bundles),
            "total_apks": len(apks),
        })
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


# =========================================================================
# READ TOOLS — Individual Vitals Metrics
# =========================================================================


@mcp.tool()
def get_slow_start_rate(
    start_date: str,
    end_date: str,
    package_name: str = "",
    page_size: int = 100,
) -> str:
    """Get slow app start rate metrics from Android Vitals.

    The slowStartRateMetricSet requires the startType dimension (HOT/WARM/COLD).

    Args:
        start_date: Start date YYYY-MM-DD.
        end_date: End date YYYY-MM-DD.
        package_name: Android package name. Uses env default if empty.
        page_size: Max rows to return.
    """
    try:
        pkg = _pkg(package_name or None)
        path, body = _build_vitals_query(
            package_name=pkg,
            metric_set="slowStartRateMetricSet",
            metrics=["slowStartRate", "distinctUsers"],
            dimensions=["startType", "versionCode"],
            start_date=start_date,
            end_date=end_date,
            page_size=page_size,
        )
        resp = _reporting_post(path, body)
        results = _parse_vitals_response(resp)
        return _fmt(results) if results else "No slow start rate data found."
    except httpx.HTTPStatusError as exc:
        return f"Error: {exc.response.status_code} – {exc.response.text}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def get_slow_rendering_rate(
    start_date: str,
    end_date: str,
    package_name: str = "",
    page_size: int = 100,
) -> str:
    """Get slow rendering (UI jank) rate metrics from Android Vitals.

    NOTE: This metric is only available for game apps (primary category = GAME).
    Non-game apps will receive a 403 from the Reporting API.

    Args:
        start_date: Start date YYYY-MM-DD.
        end_date: End date YYYY-MM-DD.
        package_name: Android package name. Uses env default if empty.
        page_size: Max rows to return.
    """
    try:
        pkg = _pkg(package_name or None)
        path, body = _build_vitals_query(
            package_name=pkg,
            metric_set="slowRenderingRateMetricSet",
            metrics=["slowRenderingRate20Fps", "slowRenderingRate30Fps", "distinctUsers"],
            dimensions=["versionCode"],
            start_date=start_date,
            end_date=end_date,
            page_size=page_size,
        )
        resp = _reporting_post(path, body)
        results = _parse_vitals_response(resp)
        return _fmt(results) if results else "No slow rendering rate data found."
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 403:
            return (
                "slowRenderingRate is only available for game apps "
                "(primary category = GAME). This package is not a game; "
                "use get_excessive_wakeup_rate or get_anr_rate instead."
            )
        return f"Error: {exc.response.status_code} – {exc.response.text}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def get_excessive_wakeup_rate(
    start_date: str,
    end_date: str,
    package_name: str = "",
    page_size: int = 100,
) -> str:
    """Get excessive wakeup rate metrics from Android Vitals (battery impact).

    Args:
        start_date: Start date YYYY-MM-DD.
        end_date: End date YYYY-MM-DD.
        package_name: Android package name. Uses env default if empty.
        page_size: Max rows to return.
    """
    try:
        pkg = _pkg(package_name or None)
        path, body = _build_vitals_query(
            package_name=pkg,
            metric_set="excessiveWakeupRateMetricSet",
            metrics=["excessiveWakeupRate", "distinctUsers"],
            dimensions=["versionCode"],
            start_date=start_date,
            end_date=end_date,
            page_size=page_size,
        )
        resp = _reporting_post(path, body)
        results = _parse_vitals_response(resp)
        return _fmt(results) if results else "No excessive wakeup rate data found."
    except httpx.HTTPStatusError as exc:
        return f"Error: {exc.response.status_code} – {exc.response.text}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def get_stuck_wakelock_rate(
    start_date: str,
    end_date: str,
    package_name: str = "",
    page_size: int = 100,
) -> str:
    """Get stuck background wakelock rate metrics from Android Vitals.

    Args:
        start_date: Start date YYYY-MM-DD.
        end_date: End date YYYY-MM-DD.
        package_name: Android package name. Uses env default if empty.
        page_size: Max rows to return.
    """
    try:
        pkg = _pkg(package_name or None)
        path, body = _build_vitals_query(
            package_name=pkg,
            metric_set="stuckBackgroundWakelockRateMetricSet",
            metrics=["stuckBgWakelockRate", "distinctUsers"],
            dimensions=["versionCode"],
            start_date=start_date,
            end_date=end_date,
            page_size=page_size,
        )
        resp = _reporting_post(path, body)
        results = _parse_vitals_response(resp)
        return _fmt(results) if results else "No stuck wakelock rate data found."
    except httpx.HTTPStatusError as exc:
        return f"Error: {exc.response.status_code} – {exc.response.text}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def get_error_rate(
    start_date: str,
    end_date: str,
    package_name: str = "",
    page_size: int = 100,
) -> str:
    """Get application error rate (non-crash errors) from Android Vitals.

    Args:
        start_date: Start date YYYY-MM-DD.
        end_date: End date YYYY-MM-DD.
        package_name: Android package name. Uses env default if empty.
        page_size: Max rows to return.
    """
    try:
        pkg = _pkg(package_name or None)
        path, body = _build_vitals_query(
            package_name=pkg,
            metric_set="errorRateMetricSet",
            metrics=["errorRate", "distinctUsers"],
            dimensions=["versionCode"],
            start_date=start_date,
            end_date=end_date,
            page_size=page_size,
        )
        resp = _reporting_post(path, body)
        results = _parse_vitals_response(resp)
        return _fmt(results) if results else "No error rate data found."
    except httpx.HTTPStatusError as exc:
        return f"Error: {exc.response.status_code} – {exc.response.text}"
    except RuntimeError as exc:
        return str(exc)


# =========================================================================
# READ TOOLS — Reporting (Additional)
# =========================================================================


_STORE_ACQUISITION_DIMENSIONS = {"country", "traffic_source"}
_RATINGS_DIMENSIONS = {
    "overview", "country", "app_version", "carrier", "device", "language", "os_version",
}


@mcp.tool()
def get_store_acquisition_report(
    year: int,
    month: int,
    package_name: str = "",
    dimension: str = "country",
) -> str:
    """Get monthly store listing acquisition metrics (visitors, installers, conversion).

    Reads from the Play Console GCS export bucket. Data is monthly granularity.
    Reports for a given month typically appear within a few days after month end.

    Args:
        year: Report year (e.g. 2026).
        month: Report month (1-12).
        package_name: Android package name. Uses env default if empty.
        dimension: 'country' or 'traffic_source'. Defaults to 'country'.
    """
    try:
        pkg = _pkg(package_name or None)
        if dimension not in _STORE_ACQUISITION_DIMENSIONS:
            return (
                f"Invalid dimension '{dimension}'. "
                f"Must be one of: {sorted(_STORE_ACQUISITION_DIMENSIONS)}."
            )
        blob_path = f"stats/store_performance/store_performance_{pkg}_{year}{month:02d}_{dimension}.csv"
        rows = _read_gcs_csv_with_fallback(blob_path)
        return _fmt({
            "package": pkg,
            "period": f"{year}-{month:02d}",
            "dimension": dimension,
            "total_rows": len(rows),
            "rows": rows,
        })
    except Exception as exc:
        return f"Error reading store acquisition report: {exc}"


@mcp.tool()
def get_ratings_overview(
    year: int,
    month: int,
    package_name: str = "",
    dimension: str = "overview",
) -> str:
    """Get monthly app ratings distribution from the Play Console GCS export.

    Data is monthly granularity. Reports for a given month typically appear
    within a few days after month end.

    Args:
        year: Report year (e.g. 2026).
        month: Report month (1-12).
        package_name: Android package name. Uses env default if empty.
        dimension: 'overview' (default), 'country', 'app_version', 'carrier',
            'device', 'language', or 'os_version'.
    """
    try:
        pkg = _pkg(package_name or None)
        if dimension not in _RATINGS_DIMENSIONS:
            return (
                f"Invalid dimension '{dimension}'. "
                f"Must be one of: {sorted(_RATINGS_DIMENSIONS)}."
            )
        blob_path = f"stats/ratings/ratings_{pkg}_{year}{month:02d}_{dimension}.csv"
        rows = _read_gcs_csv_with_fallback(blob_path)
        return _fmt({
            "package": pkg,
            "period": f"{year}-{month:02d}",
            "dimension": dimension,
            "total_rows": len(rows),
            "rows": rows,
        })
    except Exception as exc:
        return f"Error reading ratings overview: {exc}"


# =========================================================================
# WRITE TOOLS — Release Management
# =========================================================================


@mcp.tool()
def update_track(
    track: str,
    version_codes: list[str],
    status: str = "completed",
    user_fraction: float = 0,
    release_name: str = "",
    release_notes_json: str = "",
    package_name: str = "",
) -> str:
    """Update a release track (push a release to production, beta, etc.).

    This creates an edit, updates the track, and commits it.

    Args:
        track: Track name (production, beta, alpha, internal, or custom).
        version_codes: List of version codes to include in the release.
        status: Release status: 'completed' (full rollout), 'halted', 'draft',
                'inProgress' (staged rollout - requires user_fraction).
        user_fraction: For staged rollout (0.0 to 1.0). Required when status='inProgress'.
        release_name: Optional release name (e.g. '1.2.3').
        release_notes_json: JSON array of release notes, e.g.
            '[{"language":"en-US","text":"Bug fixes and improvements"}]'
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()

        edit = svc.edits().insert(packageName=pkg, body={}).execute()
        edit_id = edit["id"]

        try:
            release: dict[str, Any] = {
                "status": status,
                "versionCodes": version_codes,
            }
            if release_name:
                release["name"] = release_name
            if status == "inProgress" and user_fraction > 0:
                release["userFraction"] = user_fraction
            if release_notes_json:
                release["releaseNotes"] = json.loads(release_notes_json)

            body = {
                "track": track,
                "releases": [release],
            }

            result = svc.edits().tracks().update(
                packageName=pkg, editId=edit_id, track=track, body=body
            ).execute()

            _commit_edit_with_fallback(svc, pkg, edit_id)

            return _fmt({
                "status": "updated_and_committed",
                "track": track,
                "release_status": status,
                "version_codes": version_codes,
                "result": result,
            })
        except Exception:
            try:
                svc.edits().delete(packageName=pkg, editId=edit_id).execute()
            except Exception:
                pass
            raise

    except json.JSONDecodeError as exc:
        return f"Invalid JSON in release_notes_json: {exc}"
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


# =========================================================================
# WRITE TOOLS — Subscription Offer CRUD
# =========================================================================


@mcp.tool()
def create_subscription_offer(
    product_id: str,
    base_plan_id: str,
    offer_id: str,
    phases_json: str,
    package_name: str = "",
) -> str:
    """Create a subscription offer for a base plan.

    Args:
        product_id: Subscription product ID.
        base_plan_id: Base plan ID.
        offer_id: Unique offer ID.
        phases_json: JSON array of offer phases, e.g.
            '[{"duration":"P1M","recurrenceCount":1,"regionalConfigs":[{"regionCode":"US","price":{"currencyCode":"USD","units":"0","nanos":990000000}}]}]'
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        phases = json.loads(phases_json)

        body = {
            "productId": product_id,
            "basePlanId": base_plan_id,
            "offerId": offer_id,
            "phases": phases,
        }

        result = svc.monetization().subscriptions().basePlans().offers().create(
            packageName=pkg, productId=product_id, basePlanId=base_plan_id,
            body=body, offerId=offer_id,
        ).execute()
        return _fmt({"status": "created", "offer_id": result.get("offerId"), "offer": result})
    except json.JSONDecodeError as exc:
        return f"Invalid JSON in phases_json: {exc}"
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def get_subscription_offer(
    product_id: str,
    base_plan_id: str,
    offer_id: str,
    package_name: str = "",
) -> str:
    """Get details of a single subscription offer.

    Args:
        product_id: Subscription product ID.
        base_plan_id: Base plan ID.
        offer_id: Offer ID.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        result = svc.monetization().subscriptions().basePlans().offers().get(
            packageName=pkg, productId=product_id,
            basePlanId=base_plan_id, offerId=offer_id,
        ).execute()
        return _fmt(result)
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def update_subscription_offer(
    product_id: str,
    base_plan_id: str,
    offer_id: str,
    phases_json: str = "",
    package_name: str = "",
) -> str:
    """Update an existing subscription offer.

    Args:
        product_id: Subscription product ID.
        base_plan_id: Base plan ID.
        offer_id: Offer ID to update.
        phases_json: JSON array of updated phases.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()

        body: dict[str, Any] = {
            "productId": product_id,
            "basePlanId": base_plan_id,
            "offerId": offer_id,
        }
        if phases_json:
            body["phases"] = json.loads(phases_json)

        result = svc.monetization().subscriptions().basePlans().offers().patch(
            packageName=pkg, productId=product_id,
            basePlanId=base_plan_id, offerId=offer_id, body=body,
        ).execute()
        return _fmt({"status": "updated", "offer_id": result.get("offerId")})
    except json.JSONDecodeError as exc:
        return f"Invalid JSON in phases_json: {exc}"
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def activate_subscription_offer(
    product_id: str,
    base_plan_id: str,
    offer_id: str,
    package_name: str = "",
) -> str:
    """Activate a subscription offer, making it available for purchase.

    Args:
        product_id: Subscription product ID.
        base_plan_id: Base plan ID.
        offer_id: Offer ID to activate.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        result = svc.monetization().subscriptions().basePlans().offers().activate(
            packageName=pkg, productId=product_id,
            basePlanId=base_plan_id, offerId=offer_id, body={},
        ).execute()
        return _fmt({"status": "activated", "offer_id": offer_id, "result": result})
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def deactivate_subscription_offer(
    product_id: str,
    base_plan_id: str,
    offer_id: str,
    package_name: str = "",
) -> str:
    """Deactivate a subscription offer. Existing users keep their offer.

    Args:
        product_id: Subscription product ID.
        base_plan_id: Base plan ID.
        offer_id: Offer ID to deactivate.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        result = svc.monetization().subscriptions().basePlans().offers().deactivate(
            packageName=pkg, productId=product_id,
            basePlanId=base_plan_id, offerId=offer_id, body={},
        ).execute()
        return _fmt({"status": "deactivated", "offer_id": offer_id, "result": result})
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def delete_subscription_offer(
    product_id: str,
    base_plan_id: str,
    offer_id: str,
    package_name: str = "",
) -> str:
    """Delete a subscription offer. Must be deactivated first.

    Args:
        product_id: Subscription product ID.
        base_plan_id: Base plan ID.
        offer_id: Offer ID to delete.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        svc.monetization().subscriptions().basePlans().offers().delete(
            packageName=pkg, productId=product_id,
            basePlanId=base_plan_id, offerId=offer_id,
        ).execute()
        return _fmt({"status": "deleted", "offer_id": offer_id})
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


# =========================================================================
# WRITE/READ TOOLS — Additional Monetization
# =========================================================================


@mcp.tool()
def delete_subscription_product(product_id: str, package_name: str = "") -> str:
    """Delete a subscription product. Must be archived first.

    Args:
        product_id: Subscription product ID to delete.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        svc.monetization().subscriptions().delete(
            packageName=pkg, productId=product_id,
        ).execute()
        return _fmt({"status": "deleted", "product_id": product_id})
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def delete_base_plan(
    product_id: str,
    base_plan_id: str,
    package_name: str = "",
) -> str:
    """Delete a subscription base plan. Must be deactivated first.

    Args:
        product_id: Subscription product ID.
        base_plan_id: Base plan ID to delete.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        svc.monetization().subscriptions().basePlans().delete(
            packageName=pkg, productId=product_id, basePlanId=base_plan_id,
        ).execute()
        return _fmt({"status": "deleted", "base_plan_id": base_plan_id})
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def migrate_base_plan_prices(
    product_id: str,
    base_plan_id: str,
    regional_price_migrations_json: str,
    package_name: str = "",
) -> str:
    """Migrate prices for existing subscribers of a base plan.

    Args:
        product_id: Subscription product ID.
        base_plan_id: Base plan ID.
        regional_price_migrations_json: JSON array of migrations, e.g.
            '[{"regionCode":"US","oldestAllowedPriceVersionTime":"2024-01-01T00:00:00Z","priceIncreaseType":"OPT_IN_PRICE_INCREASE"}]'
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        migrations = json.loads(regional_price_migrations_json)

        result = svc.monetization().subscriptions().basePlans().migratePrices(
            packageName=pkg, productId=product_id, basePlanId=base_plan_id,
            body={"regionalPriceMigrations": migrations},
        ).execute()
        return _fmt({"status": "migrated", "result": result})
    except json.JSONDecodeError as exc:
        return f"Invalid JSON: {exc}"
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


# =========================================================================
# WRITE TOOLS — App Details & Listings
# =========================================================================


@mcp.tool()
def update_app_details(
    contact_email: str = "",
    contact_phone: str = "",
    contact_website: str = "",
    default_language: str = "",
    package_name: str = "",
) -> str:
    """Update app contact info and default language.

    Creates an edit, patches details, and commits.

    Args:
        contact_email: Developer contact email.
        contact_phone: Developer contact phone.
        contact_website: Developer website URL.
        default_language: Default BCP-47 language code.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()

        edit = svc.edits().insert(packageName=pkg, body={}).execute()
        edit_id = edit["id"]

        try:
            current = svc.edits().details().get(
                packageName=pkg, editId=edit_id
            ).execute()

            body: dict[str, Any] = {}
            body["contactEmail"] = contact_email or current.get("contactEmail", "")
            body["contactPhone"] = contact_phone or current.get("contactPhone", "")
            body["contactWebsite"] = contact_website or current.get("contactWebsite", "")
            body["defaultLanguage"] = default_language or current.get("defaultLanguage", "")

            result = svc.edits().details().patch(
                packageName=pkg, editId=edit_id, body=body
            ).execute()

            _commit_edit_with_fallback(svc, pkg, edit_id)

            return _fmt({"status": "updated_and_committed", "details": result})
        except Exception:
            try:
                svc.edits().delete(packageName=pkg, editId=edit_id).execute()
            except Exception:
                pass
            raise

    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def delete_store_listing(language: str, package_name: str = "") -> str:
    """Delete a store listing for a specific language.

    Creates an edit, deletes the listing, and commits.

    Args:
        language: BCP-47 language code to delete (e.g. 'fr-FR').
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()

        edit = svc.edits().insert(packageName=pkg, body={}).execute()
        edit_id = edit["id"]

        try:
            svc.edits().listings().delete(
                packageName=pkg, editId=edit_id, language=language
            ).execute()

            _commit_edit_with_fallback(svc, pkg, edit_id)
            return _fmt({"status": "deleted_and_committed", "language": language})
        except Exception:
            try:
                svc.edits().delete(packageName=pkg, editId=edit_id).execute()
            except Exception:
                pass
            raise

    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def delete_all_store_listings(package_name: str = "", confirm: bool = False) -> str:
    """Delete ALL store listings (every locale) for an app — DESTRUCTIVE.

    Requires confirm=True to proceed. Use with extreme care: this wipes the
    entire localized store presence in a single committed edit.

    Args:
        package_name: Android package name. Uses env default if empty.
        confirm: Must be True to actually run. Defaults to False as a safety guard.
    """
    if not confirm:
        return (
            "Refusing to run without confirm=True. This is destructive — "
            "it will delete every per-locale store listing for the app."
        )
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()

        edit = svc.edits().insert(packageName=pkg, body={}).execute()
        edit_id = edit["id"]

        try:
            svc.edits().listings().deleteall(
                packageName=pkg, editId=edit_id
            ).execute()
            _commit_edit_with_fallback(svc, pkg, edit_id)
            return _fmt({"status": "all_listings_deleted_and_committed", "package": pkg})
        except Exception:
            try:
                svc.edits().delete(packageName=pkg, editId=edit_id).execute()
            except Exception:
                pass
            raise

    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def list_store_images(
    language: str = "en-US",
    image_type: str = "phoneScreenshots",
    package_name: str = "",
) -> str:
    """List store listing images for a language and image type.

    Args:
        language: BCP-47 language code.
        image_type: One of: featureGraphic, icon, phoneScreenshots,
                    sevenInchScreenshots, tenInchScreenshots,
                    tvScreenshots, wearScreenshots, tvBanner.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()

        edit = svc.edits().insert(packageName=pkg, body={}).execute()
        edit_id = edit["id"]

        try:
            resp = svc.edits().images().list(
                packageName=pkg, editId=edit_id,
                language=language, imageType=image_type
            ).execute()

            images = resp.get("images", [])
            results = []
            for img in images:
                results.append({
                    "id": img.get("id"),
                    "url": img.get("url"),
                    "sha1": img.get("sha1"),
                    "sha256": img.get("sha256"),
                })
        finally:
            try:
                svc.edits().delete(packageName=pkg, editId=edit_id).execute()
            except Exception:
                pass

        return _fmt({"images": results, "total": len(results), "image_type": image_type})
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def delete_store_image(
    image_id: str,
    language: str = "en-US",
    image_type: str = "phoneScreenshots",
    package_name: str = "",
) -> str:
    """Delete a store listing image.

    Creates an edit, deletes the image, and commits.

    Args:
        image_id: Image ID to delete.
        language: BCP-47 language code.
        image_type: Image type (phoneScreenshots, featureGraphic, etc.).
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()

        edit = svc.edits().insert(packageName=pkg, body={}).execute()
        edit_id = edit["id"]

        try:
            svc.edits().images().delete(
                packageName=pkg, editId=edit_id,
                language=language, imageType=image_type, imageId=image_id
            ).execute()

            _commit_edit_with_fallback(svc, pkg, edit_id)
            return _fmt({"status": "deleted_and_committed", "image_id": image_id})
        except Exception:
            try:
                svc.edits().delete(packageName=pkg, editId=edit_id).execute()
            except Exception:
                pass
            raise

    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


# =========================================================================
# READ/WRITE TOOLS — Testers & Country Availability
# =========================================================================


@mcp.tool()
def get_testers(track: str = "internal", package_name: str = "") -> str:
    """Get the list of testers for a track.

    Args:
        track: Track name (production, beta, alpha, internal).
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()

        edit = svc.edits().insert(packageName=pkg, body={}).execute()
        edit_id = edit["id"]

        try:
            result = svc.edits().testers().get(
                packageName=pkg, editId=edit_id, track=track
            ).execute()
        finally:
            try:
                svc.edits().delete(packageName=pkg, editId=edit_id).execute()
            except Exception:
                pass

        return _fmt({
            "track": track,
            "google_groups": result.get("googleGroups", []),
        })
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def update_testers(
    track: str,
    google_groups: list[str],
    package_name: str = "",
) -> str:
    """Update the tester list for a track.

    Creates an edit, updates testers, and commits.

    Args:
        track: Track name (production, beta, alpha, internal).
        google_groups: List of Google Group email addresses for testing.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()

        edit = svc.edits().insert(packageName=pkg, body={}).execute()
        edit_id = edit["id"]

        try:
            body = {"googleGroups": google_groups}
            result = svc.edits().testers().patch(
                packageName=pkg, editId=edit_id, track=track, body=body
            ).execute()

            _commit_edit_with_fallback(svc, pkg, edit_id)
            return _fmt({"status": "updated_and_committed", "track": track, "testers": result})
        except Exception:
            try:
                svc.edits().delete(packageName=pkg, editId=edit_id).execute()
            except Exception:
                pass
            raise

    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def get_country_availability(track: str = "production", package_name: str = "") -> str:
    """Get country availability for a track.

    Args:
        track: Track name (production, beta, alpha, internal).
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()

        edit = svc.edits().insert(packageName=pkg, body={}).execute()
        edit_id = edit["id"]

        try:
            result = svc.edits().countryavailability().get(
                packageName=pkg, editId=edit_id, track=track
            ).execute()
        finally:
            try:
                svc.edits().delete(packageName=pkg, editId=edit_id).execute()
            except Exception:
                pass

        countries = result.get("countries", [])
        return _fmt({
            "track": track,
            "total_countries": len(countries),
            "countries": countries,
        })
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


# =========================================================================
# READ TOOLS — Generated APKs
# =========================================================================


@mcp.tool()
def list_generated_apks(version_code: int, package_name: str = "") -> str:
    """List generated APKs for a specific app bundle version.

    Args:
        version_code: The version code of the uploaded app bundle.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()

        resp = svc.generatedapks().list(
            packageName=pkg, versionCode=version_code
        ).execute()

        apks = resp.get("generatedApks", [])
        results = []
        for apk in apks:
            results.append({
                "variant_id": apk.get("variantId"),
                "generated_apks": apk.get("generatedStandaloneApks", []),
                "generated_universal": apk.get("generatedUniversalApk"),
                "target_abi": apk.get("targetingInfo", {}).get("abiTargeting"),
            })

        return _fmt({"version_code": version_code, "apks": results, "total": len(results)})
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


# =========================================================================
# READ TOOLS — Vitals: Anomalies & Error Debugging (Reporting API)
# =========================================================================


@mcp.tool()
def list_anomalies(package_name: str = "") -> str:
    """List detected anomalies (spikes in crashes, ANRs, etc.) from Android Vitals.

    Args:
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        path = f"/apps/{pkg}/anomalies"
        resp = _reporting_get(path)

        anomalies = resp.get("anomalies", [])
        results = []
        for a in anomalies:
            results.append({
                "name": a.get("name"),
                "metric_set": a.get("metricSet"),
                "dimension": a.get("timelineSpec", {}).get("aggregationPeriod"),
                "details": a.get("details"),
            })

        return _fmt(results) if results else "No anomalies detected."
    except httpx.HTTPStatusError as exc:
        return f"Error: {exc.response.status_code} – {exc.response.text}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def search_error_issues(
    package_name: str = "",
    page_size: int = 25,
    page_token: str = "",
    filter_str: str = "",
) -> str:
    """Search grouped error issues (crashes/ANRs) with stack trace signatures.

    Args:
        package_name: Android package name. Uses env default if empty.
        page_size: Max results (default 25).
        page_token: Pagination token.
        filter_str: Optional filter (e.g. 'errorIssueType=CRASH').
    """
    try:
        pkg = _pkg(package_name or None)
        path = f"/apps/{pkg}/errorIssues:search"
        params: dict[str, Any] = {"pageSize": page_size}
        if page_token:
            params["pageToken"] = page_token
        if filter_str:
            params["filter"] = filter_str

        resp = _reporting_get(path, params)

        issues = resp.get("errorIssues", [])
        next_token = resp.get("nextPageToken")

        results = []
        for issue in issues:
            results.append({
                "name": issue.get("name"),
                "type": issue.get("type"),
                "error_report_count": issue.get("errorReportCount"),
                "distinct_users": issue.get("distinctUsers"),
                "first_os_version": issue.get("firstOsVersion"),
                "last_os_version": issue.get("lastOsVersion"),
                "first_app_version": issue.get("firstAppVersion"),
                "last_app_version": issue.get("lastAppVersion"),
                "cause": issue.get("cause"),
                "location": issue.get("location"),
            })

        output: dict[str, Any] = {"issues": results, "total": len(results)}
        if next_token:
            output["next_page_token"] = next_token

        return _fmt(output)
    except httpx.HTTPStatusError as exc:
        return f"Error: {exc.response.status_code} – {exc.response.text}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def search_error_reports(
    package_name: str = "",
    page_size: int = 25,
    page_token: str = "",
    filter_str: str = "",
) -> str:
    """Search individual error reports (crash/ANR instances) with stack traces.

    Args:
        package_name: Android package name. Uses env default if empty.
        page_size: Max results (default 25).
        page_token: Pagination token.
        filter_str: Optional filter (e.g. 'errorIssueType=ANR').
    """
    try:
        pkg = _pkg(package_name or None)
        path = f"/apps/{pkg}/errorReports:search"
        params: dict[str, Any] = {"pageSize": page_size}
        if page_token:
            params["pageToken"] = page_token
        if filter_str:
            params["filter"] = filter_str

        resp = _reporting_get(path, params)

        reports = resp.get("errorReports", [])
        next_token = resp.get("nextPageToken")

        results = []
        for report in reports:
            results.append({
                "name": report.get("name"),
                "type": report.get("type"),
                "os_version": report.get("osVersion"),
                "app_version": report.get("appVersion"),
                "device_model": report.get("deviceModel"),
                "issue": report.get("issue"),
                "report_text": report.get("reportText"),
            })

        output: dict[str, Any] = {"reports": results, "total": len(results)}
        if next_token:
            output["next_page_token"] = next_token

        return _fmt(output)
    except httpx.HTTPStatusError as exc:
        return f"Error: {exc.response.status_code} – {exc.response.text}"
    except RuntimeError as exc:
        return str(exc)


# =========================================================================
# Edits / Publishing System
# =========================================================================


@mcp.tool()
def create_edit(package_name: str = "") -> str:
    """Create a new edit session for staging changes before publishing.

    Args:
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        resp = svc.edits().insert(packageName=pkg, body={}).execute()
        return _fmt(resp)
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def validate_edit(edit_id: str, package_name: str = "") -> str:
    """Validate an edit before committing to check for errors.

    Args:
        edit_id: The edit session ID to validate.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        resp = svc.edits().validate(packageName=pkg, editId=edit_id).execute()
        return _fmt(resp)
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def commit_edit(edit_id: str, package_name: str = "") -> str:
    """Commit (publish) an edit, applying all staged changes.

    Args:
        edit_id: The edit session ID to commit.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        resp = _commit_edit_with_fallback(svc, pkg, edit_id)
        return _fmt(resp)
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def delete_edit(edit_id: str, package_name: str = "") -> str:
    """Delete/cancel an edit session, discarding all staged changes.

    Args:
        edit_id: The edit session ID to delete.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        svc.edits().delete(packageName=pkg, editId=edit_id).execute()
        return "Edit deleted successfully."
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def upload_apk(edit_id: str, apk_path: str, package_name: str = "") -> str:
    """Upload an APK file to an edit session.

    Args:
        edit_id: The edit session ID.
        apk_path: Local filesystem path to the APK file.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        media = MediaFileUpload(apk_path, mimetype="application/vnd.android.package-archive")
        resp = svc.edits().apks().upload(
            packageName=pkg, editId=edit_id, media_body=media
        ).execute()
        return _fmt(resp)
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def upload_bundle(edit_id: str, bundle_path: str, package_name: str = "") -> str:
    """Upload an AAB (Android App Bundle) file to an edit session.

    Args:
        edit_id: The edit session ID.
        bundle_path: Local filesystem path to the AAB file.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        media = MediaFileUpload(bundle_path, mimetype="application/octet-stream")
        resp = svc.edits().bundles().upload(
            packageName=pkg, editId=edit_id, media_body=media
        ).execute()
        return _fmt(resp)
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def list_edit_apks(edit_id: str, package_name: str = "") -> str:
    """List all APKs uploaded in an edit session.

    Args:
        edit_id: The edit session ID.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        resp = svc.edits().apks().list(packageName=pkg, editId=edit_id).execute()
        return _fmt(resp)
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def list_edit_bundles(edit_id: str, package_name: str = "") -> str:
    """List all bundles (AABs) uploaded in an edit session.

    Args:
        edit_id: The edit session ID.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        resp = svc.edits().bundles().list(packageName=pkg, editId=edit_id).execute()
        return _fmt(resp)
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


# =========================================================================
# Purchases v2
# =========================================================================


@mcp.tool()
def get_product_purchase_v2(purchase_token: str, package_name: str = "") -> str:
    """Get one-time product purchase status using the Purchases v2 API.

    Args:
        purchase_token: The purchase token from the client.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        token_val = _get_auth_token()
        url = (
            f"https://androidpublisher.googleapis.com/androidpublisher/v3/"
            f"applications/{pkg}/purchases/productsv2/tokens/{purchase_token}"
        )
        headers = {"Authorization": f"Bearer {token_val}"}
        with httpx.Client(timeout=TIMEOUT) as c:
            resp = c.get(url, headers=headers)
        resp.raise_for_status()
        return _fmt(resp.json())
    except httpx.HTTPStatusError as exc:
        return f"Error: {exc.response.status_code} – {exc.response.text}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def get_subscription_v2(purchase_token: str, package_name: str = "") -> str:
    """Get subscription purchase details using the Purchases v2 API.

    Args:
        purchase_token: The purchase token from the client.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        token_val = _get_auth_token()
        url = (
            f"https://androidpublisher.googleapis.com/androidpublisher/v3/"
            f"applications/{pkg}/purchases/subscriptionsv2/tokens/{purchase_token}"
        )
        headers = {"Authorization": f"Bearer {token_val}"}
        with httpx.Client(timeout=TIMEOUT) as c:
            resp = c.get(url, headers=headers)
        resp.raise_for_status()
        return _fmt(resp.json())
    except httpx.HTTPStatusError as exc:
        return f"Error: {exc.response.status_code} – {exc.response.text}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def cancel_subscription_v2(purchase_token: str, package_name: str = "") -> str:
    """Cancel a subscription using the Purchases v2 API.

    Args:
        purchase_token: The purchase token from the client.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        token_val = _get_auth_token()
        url = (
            f"https://androidpublisher.googleapis.com/androidpublisher/v3/"
            f"applications/{pkg}/purchases/subscriptionsv2/tokens/{purchase_token}:cancel"
        )
        headers = {
            "Authorization": f"Bearer {token_val}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=TIMEOUT) as c:
            resp = c.post(url, headers=headers, json={})
        resp.raise_for_status()
        return "Subscription cancelled successfully."
    except httpx.HTTPStatusError as exc:
        return f"Error: {exc.response.status_code} – {exc.response.text}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def defer_subscription_v2(
    purchase_token: str,
    expected_expiry_time: str,
    desired_expiry_time: str,
    package_name: str = "",
) -> str:
    """Defer a subscription billing period using the Purchases v2 API.

    Args:
        purchase_token: The purchase token from the client.
        expected_expiry_time: Current expected expiry (RFC 3339, e.g. 2025-12-31T23:59:59Z).
        desired_expiry_time: New desired expiry (RFC 3339).
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        token_val = _get_auth_token()
        url = (
            f"https://androidpublisher.googleapis.com/androidpublisher/v3/"
            f"applications/{pkg}/purchases/subscriptionsv2/tokens/{purchase_token}:defer"
        )
        headers = {
            "Authorization": f"Bearer {token_val}",
            "Content-Type": "application/json",
        }
        body = {
            "deferralInfo": {
                "expectedExpiryTime": expected_expiry_time,
                "desiredExpiryTime": desired_expiry_time,
            }
        }
        with httpx.Client(timeout=TIMEOUT) as c:
            resp = c.post(url, headers=headers, json=body)
        resp.raise_for_status()
        return _fmt(resp.json())
    except httpx.HTTPStatusError as exc:
        return f"Error: {exc.response.status_code} – {exc.response.text}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def revoke_subscription_v2(purchase_token: str, package_name: str = "") -> str:
    """Revoke a subscription using the Purchases v2 API.

    Args:
        purchase_token: The purchase token from the client.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        token_val = _get_auth_token()
        url = (
            f"https://androidpublisher.googleapis.com/androidpublisher/v3/"
            f"applications/{pkg}/purchases/subscriptionsv2/tokens/{purchase_token}:revoke"
        )
        headers = {
            "Authorization": f"Bearer {token_val}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=TIMEOUT) as c:
            resp = c.post(url, headers=headers, json={})
        resp.raise_for_status()
        return "Subscription revoked successfully."
    except httpx.HTTPStatusError as exc:
        return f"Error: {exc.response.status_code} – {exc.response.text}"
    except RuntimeError as exc:
        return str(exc)


# =========================================================================
# One-Time Products (Monetization)
# =========================================================================


@mcp.tool()
def list_onetime_products(
    package_name: str = "",
    page_size: int = 100,
    page_token: str = "",
) -> str:
    """List one-time products (managed products) via the monetization API.

    Args:
        package_name: Android package name. Uses env default if empty.
        page_size: Max products per page (default 100).
        page_token: Pagination token for next page.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        req = svc.monetization().onetimeproducts().list(
            packageName=pkg, pageSize=page_size
        )
        if page_token:
            req = svc.monetization().onetimeproducts().list(
                packageName=pkg, pageSize=page_size, pageToken=page_token
            )
        resp = req.execute()
        return _fmt(resp)
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def get_onetime_product(product_id: str, package_name: str = "") -> str:
    """Get details for a single one-time product.

    Args:
        product_id: The product ID (SKU) to retrieve.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        resp = svc.monetization().onetimeproducts().get(
            packageName=pkg, productId=product_id
        ).execute()
        return _fmt(resp)
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def create_onetime_product(
    product_id: str,
    product_config_json: str,
    package_name: str = "",
) -> str:
    """Create a new one-time product.

    Args:
        product_id: The product ID (SKU) for the new product.
        product_config_json: JSON string with product configuration
            (e.g. {"listings": {"en-US": {"title": "...", "description": "..."}},
            "defaultPrice": {"currencyCode": "USD", "units": "1"}}).
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        body = json.loads(product_config_json)
        body["productId"] = product_id
        resp = svc.monetization().onetimeproducts().patch(
            packageName=pkg, productId=product_id, body=body
        ).execute()
        return _fmt(resp)
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except json.JSONDecodeError as exc:
        return f"Error: Invalid JSON – {exc}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def update_onetime_product(
    product_id: str,
    product_config_json: str,
    package_name: str = "",
) -> str:
    """Update an existing one-time product.

    Args:
        product_id: The product ID (SKU) to update.
        product_config_json: JSON string with fields to update.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        body = json.loads(product_config_json)
        resp = svc.monetization().onetimeproducts().patch(
            packageName=pkg, productId=product_id, body=body
        ).execute()
        return _fmt(resp)
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except json.JSONDecodeError as exc:
        return f"Error: Invalid JSON – {exc}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def delete_onetime_product(product_id: str, package_name: str = "") -> str:
    """Delete a one-time product.

    Args:
        product_id: The product ID (SKU) to delete.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        svc.monetization().onetimeproducts().delete(
            packageName=pkg, productId=product_id
        ).execute()
        return f"One-time product '{product_id}' deleted successfully."
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


# =========================================================================
# Users & Grants
# =========================================================================


@mcp.tool()
def list_users() -> str:
    """List users with access to the developer account.

    Requires GOOGLE_PLAY_DEVELOPER_ID environment variable.
    """
    try:
        dev_id = os.environ.get("GOOGLE_PLAY_DEVELOPER_ID", "")
        if not dev_id:
            return "Error: GOOGLE_PLAY_DEVELOPER_ID environment variable is not set."
        token_val = _get_auth_token()
        url = (
            f"https://androidpublisher.googleapis.com/androidpublisher/v3/"
            f"developers/{dev_id}/users"
        )
        headers = {"Authorization": f"Bearer {token_val}"}
        with httpx.Client(timeout=TIMEOUT) as c:
            resp = c.get(url, headers=headers)
        resp.raise_for_status()
        return _fmt(resp.json())
    except httpx.HTTPStatusError as exc:
        return f"Error: {exc.response.status_code} – {exc.response.text}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def create_user(email: str, permissions_json: str = "[]") -> str:
    """Grant a user access to the developer account.

    Args:
        email: Email address of the user to add.
        permissions_json: JSON array of developer account permission strings
            (e.g. '["VIEW_APP_INFORMATION", "MANAGE_ORDERS"]').
    """
    try:
        dev_id = os.environ.get("GOOGLE_PLAY_DEVELOPER_ID", "")
        if not dev_id:
            return "Error: GOOGLE_PLAY_DEVELOPER_ID environment variable is not set."
        token_val = _get_auth_token()
        url = (
            f"https://androidpublisher.googleapis.com/androidpublisher/v3/"
            f"developers/{dev_id}/users"
        )
        permissions = json.loads(permissions_json)
        body = {
            "email": email,
            "developerAccountPermissions": permissions,
        }
        headers = {
            "Authorization": f"Bearer {token_val}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=TIMEOUT) as c:
            resp = c.post(url, headers=headers, json=body)
        resp.raise_for_status()
        return _fmt(resp.json())
    except httpx.HTTPStatusError as exc:
        return f"Error: {exc.response.status_code} – {exc.response.text}"
    except json.JSONDecodeError as exc:
        return f"Error: Invalid JSON – {exc}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def delete_user(email: str) -> str:
    """Remove a user's access from the developer account.

    Args:
        email: Email address of the user to remove.
    """
    try:
        dev_id = os.environ.get("GOOGLE_PLAY_DEVELOPER_ID", "")
        if not dev_id:
            return "Error: GOOGLE_PLAY_DEVELOPER_ID environment variable is not set."
        token_val = _get_auth_token()
        url = (
            f"https://androidpublisher.googleapis.com/androidpublisher/v3/"
            f"developers/{dev_id}/users/{email}"
        )
        headers = {"Authorization": f"Bearer {token_val}"}
        with httpx.Client(timeout=TIMEOUT) as c:
            resp = c.delete(url, headers=headers)
        resp.raise_for_status()
        return f"User '{email}' removed successfully."
    except httpx.HTTPStatusError as exc:
        return f"Error: {exc.response.status_code} – {exc.response.text}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def update_user_grants(
    email: str,
    grants_json: str,
    package_name: str = "",
) -> str:
    """Update app-level permissions (grants) for a user.

    Args:
        email: Email address of the user.
        grants_json: JSON string with grant body
            (e.g. {"appLevelPermissions": ["ACCESS_LEVEL_READ_ONLY"]}).
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        dev_id = os.environ.get("GOOGLE_PLAY_DEVELOPER_ID", "")
        if not dev_id:
            return "Error: GOOGLE_PLAY_DEVELOPER_ID environment variable is not set."
        token_val = _get_auth_token()
        url = (
            f"https://androidpublisher.googleapis.com/androidpublisher/v3/"
            f"developers/{dev_id}/users/{email}/grants/{pkg}"
        )
        body = json.loads(grants_json)
        headers = {
            "Authorization": f"Bearer {token_val}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=TIMEOUT) as c:
            resp = c.patch(url, headers=headers, json=body)
        resp.raise_for_status()
        return _fmt(resp.json())
    except httpx.HTTPStatusError as exc:
        return f"Error: {exc.response.status_code} – {exc.response.text}"
    except json.JSONDecodeError as exc:
        return f"Error: Invalid JSON – {exc}"
    except RuntimeError as exc:
        return str(exc)


# =========================================================================
# Store Images & Deobfuscation
# =========================================================================


@mcp.tool()
def upload_store_image(
    edit_id: str,
    image_path: str,
    image_type: str,
    language: str = "en-US",
    package_name: str = "",
) -> str:
    """Upload a store listing image (screenshot, icon, feature graphic, etc.).

    Args:
        edit_id: The edit session ID.
        image_path: Local filesystem path to the image file (PNG recommended).
        image_type: One of: phoneScreenshots, sevenInchScreenshots,
            tenInchScreenshots, tvScreenshots, wearScreenshots,
            icon, featureGraphic, tvBanner.
        language: BCP-47 language code (default en-US).
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        media = MediaFileUpload(image_path, mimetype="image/png")
        resp = svc.edits().images().upload(
            packageName=pkg,
            editId=edit_id,
            language=language,
            imageType=image_type,
            media_body=media,
        ).execute()
        return _fmt(resp)
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def upload_deobfuscation_file(
    edit_id: str,
    version_code: int,
    file_path: str,
    deobfuscation_file_type: str = "proguard",
    package_name: str = "",
) -> str:
    """Upload a ProGuard/R8 mapping (deobfuscation) file for an APK version.

    Args:
        edit_id: The edit session ID.
        version_code: The APK version code to associate the mapping with.
        file_path: Local filesystem path to the mapping file.
        deobfuscation_file_type: Type of file – 'proguard' (default) or 'nativeCode'.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        media = MediaFileUpload(file_path, mimetype="application/octet-stream")
        resp = svc.edits().deobfuscationfiles().upload(
            packageName=pkg,
            editId=edit_id,
            apkVersionCode=version_code,
            deobfuscationFileType=deobfuscation_file_type,
            media_body=media,
        ).execute()
        return _fmt(resp)
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


# =========================================================================
# Monetization Helpers
# =========================================================================


@mcp.tool()
def convert_region_prices(
    currency: str,
    units: str,
    nanos: int = 0,
    package_name: str = "",
) -> str:
    """Convert a base price to all supported regions/currencies.

    Args:
        currency: ISO 4217 currency code (e.g. 'USD').
        units: Price in major units (e.g. '1' for $1.00).
        nanos: Fractional part in nano-units (e.g. 990000000 for $0.99).
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        svc = _get_publisher_service()
        body = {
            "price": {
                "currencyCode": currency,
                "units": units,
                "nanos": nanos,
            }
        }
        resp = svc.monetization().convertRegionPrices(
            packageName=pkg, body=body
        ).execute()
        return _fmt(resp)
    except HttpError as exc:
        return f"Error: {exc.status_code} – {exc.reason}"
    except RuntimeError as exc:
        return str(exc)


# =========================================================================
# Additional Vitals
# =========================================================================


@mcp.tool()
def get_lmk_rate(
    start_date: str,
    end_date: str,
    package_name: str = "",
    page_size: int = 100,
) -> str:
    """Get Low Memory Kill (LMK) rate metrics from Android Vitals.

    Args:
        start_date: Start date YYYY-MM-DD.
        end_date: End date YYYY-MM-DD.
        package_name: Android package name. Uses env default if empty.
        page_size: Max rows to return (default 100).
    """
    try:
        pkg = _pkg(package_name or None)
        path, body = _build_vitals_query(
            package_name=pkg,
            metric_set="lmkRateMetricSet",
            metrics=["lmkRate", "distinctUsers"],
            dimensions=["versionCode"],
            start_date=start_date,
            end_date=end_date,
            page_size=page_size,
        )
        resp = _reporting_post(path, body)
        results = _parse_vitals_response(resp)
        return _fmt({"metric": "lmk_rate", "rows": results, "total": len(results)})
    except httpx.HTTPStatusError as exc:
        return f"Error: {exc.response.status_code} – {exc.response.text}"
    except RuntimeError as exc:
        return str(exc)


# =========================================================================
# App Recovery
# =========================================================================


@mcp.tool()
def create_app_recovery(
    version_codes_json: str,
    package_name: str = "",
) -> str:
    """Create an app recovery action targeting specific version codes.

    Args:
        version_codes_json: JSON array of version code integers to target
            (e.g. '[100, 101, 102]').
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        token_val = _get_auth_token()
        url = (
            f"https://androidpublisher.googleapis.com/androidpublisher/v3/"
            f"applications/{pkg}/appRecoveries"
        )
        version_codes = json.loads(version_codes_json)
        body = {
            "remoteInAppUpdate": {"isRecoverable": True},
            "targeting": {
                "versionList": {
                    "versionCodes": version_codes,
                }
            },
        }
        headers = {
            "Authorization": f"Bearer {token_val}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=TIMEOUT) as c:
            resp = c.post(url, headers=headers, json=body)
        resp.raise_for_status()
        return _fmt(resp.json())
    except httpx.HTTPStatusError as exc:
        return f"Error: {exc.response.status_code} – {exc.response.text}"
    except json.JSONDecodeError as exc:
        return f"Error: Invalid JSON – {exc}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def deploy_app_recovery(
    app_recovery_id: str,
    package_name: str = "",
) -> str:
    """Deploy (activate) an app recovery action.

    Args:
        app_recovery_id: The ID of the app recovery action to deploy.
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        token_val = _get_auth_token()
        url = (
            f"https://androidpublisher.googleapis.com/androidpublisher/v3/"
            f"applications/{pkg}/appRecoveries/{app_recovery_id}:deploy"
        )
        headers = {
            "Authorization": f"Bearer {token_val}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=TIMEOUT) as c:
            resp = c.post(url, headers=headers, json={})
        resp.raise_for_status()
        return _fmt(resp.json())
    except httpx.HTTPStatusError as exc:
        return f"Error: {exc.response.status_code} – {exc.response.text}"
    except RuntimeError as exc:
        return str(exc)


@mcp.tool()
def list_app_recoveries(package_name: str = "") -> str:
    """List all app recovery actions for the app.

    Args:
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        token_val = _get_auth_token()
        url = (
            f"https://androidpublisher.googleapis.com/androidpublisher/v3/"
            f"applications/{pkg}/appRecoveries"
        )
        headers = {"Authorization": f"Bearer {token_val}"}
        with httpx.Client(timeout=TIMEOUT) as c:
            resp = c.get(url, headers=headers)
        resp.raise_for_status()
        return _fmt(resp.json())
    except httpx.HTTPStatusError as exc:
        return f"Error: {exc.response.status_code} – {exc.response.text}"
    except RuntimeError as exc:
        return str(exc)


# =========================================================================
# Orders
# =========================================================================


@mcp.tool()
def batch_get_orders(order_ids_json: str, package_name: str = "") -> str:
    """Get multiple orders at once using the batch orders API.

    Args:
        order_ids_json: JSON array of order ID strings
            (e.g. '["GPA.1234-5678", "GPA.9012-3456"]').
        package_name: Android package name. Uses env default if empty.
    """
    try:
        pkg = _pkg(package_name or None)
        token_val = _get_auth_token()
        url = (
            f"https://androidpublisher.googleapis.com/androidpublisher/v3/"
            f"applications/{pkg}/orders:batchGet"
        )
        order_ids = json.loads(order_ids_json)
        body = {"order_ids": order_ids}
        headers = {
            "Authorization": f"Bearer {token_val}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=TIMEOUT) as c:
            resp = c.post(url, headers=headers, json=body)
        resp.raise_for_status()
        return _fmt(resp.json())
    except httpx.HTTPStatusError as exc:
        return f"Error: {exc.response.status_code} – {exc.response.text}"
    except json.JSONDecodeError as exc:
        return f"Error: Invalid JSON – {exc}"
    except RuntimeError as exc:
        return str(exc)


# =========================================================================
# Entry point
# =========================================================================


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
