#!/usr/bin/env python3
"""Verify Google Play Console MCP Server setup – auth and API access."""

import json
import os
import sys


def check_env():
    """Check required environment variables."""
    print("=== Environment Variables ===")
    required = [
        "GOOGLE_PLAY_SERVICE_ACCOUNT_KEY",
        "GOOGLE_PLAY_DEVELOPER_ID",
        "GOOGLE_PLAY_PACKAGE_NAME",
    ]
    all_ok = True
    for var in required:
        val = os.environ.get(var, "")
        if val:
            display = val[:30] + "..." if len(val) > 30 else val
            print(f"  [OK] {var} = {display}")
        else:
            print(f"  [MISSING] {var}")
            all_ok = False
    return all_ok


def check_service_account():
    """Validate the service account key file."""
    print("\n=== Service Account Key ===")
    key_path = os.environ.get("GOOGLE_PLAY_SERVICE_ACCOUNT_KEY", "")
    if not key_path:
        print("  [SKIP] GOOGLE_PLAY_SERVICE_ACCOUNT_KEY not set")
        return False

    if not os.path.exists(key_path):
        print(f"  [ERROR] File not found: {key_path}")
        return False

    try:
        with open(key_path) as f:
            data = json.load(f)
        email = data.get("client_email", "N/A")
        project = data.get("project_id", "N/A")
        print(f"  [OK] Service account: {email}")
        print(f"  [OK] Project: {project}")
        return True
    except Exception as e:
        print(f"  [ERROR] Could not parse key file: {e}")
        return False


def check_credentials():
    """Test credential loading."""
    print("\n=== Credential Loading ===")
    try:
        from google.oauth2 import service_account as sa

        key_path = os.environ.get("GOOGLE_PLAY_SERVICE_ACCOUNT_KEY", "")
        scopes = [
            "https://www.googleapis.com/auth/androidpublisher",
            "https://www.googleapis.com/auth/playdeveloperreporting",
            "https://www.googleapis.com/auth/cloud-platform",
        ]
        creds = sa.Credentials.from_service_account_file(key_path, scopes=scopes)
        print(f"  [OK] Credentials loaded for: {creds.service_account_email}")
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def check_publisher_api():
    """Test Android Publisher API access."""
    print("\n=== Android Publisher API ===")
    try:
        from google.oauth2 import service_account as sa
        from googleapiclient.discovery import build

        key_path = os.environ.get("GOOGLE_PLAY_SERVICE_ACCOUNT_KEY", "")
        pkg = os.environ.get("GOOGLE_PLAY_PACKAGE_NAME", "")
        scopes = ["https://www.googleapis.com/auth/androidpublisher"]
        creds = sa.Credentials.from_service_account_file(key_path, scopes=scopes)
        svc = build("androidpublisher", "v3", credentials=creds, cache_discovery=False)

        # Try to create and delete an edit (read-only check)
        edit = svc.edits().insert(packageName=pkg, body={}).execute()
        edit_id = edit["id"]
        svc.edits().delete(packageName=pkg, editId=edit_id).execute()
        print(f"  [OK] Publisher API accessible for {pkg}")
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def check_reviews_api():
    """Test reviews list access."""
    print("\n=== Reviews API ===")
    try:
        from google.oauth2 import service_account as sa
        from googleapiclient.discovery import build

        key_path = os.environ.get("GOOGLE_PLAY_SERVICE_ACCOUNT_KEY", "")
        pkg = os.environ.get("GOOGLE_PLAY_PACKAGE_NAME", "")
        scopes = ["https://www.googleapis.com/auth/androidpublisher"]
        creds = sa.Credentials.from_service_account_file(key_path, scopes=scopes)
        svc = build("androidpublisher", "v3", credentials=creds, cache_discovery=False)

        resp = svc.reviews().list(packageName=pkg, maxResults=1).execute()
        count = len(resp.get("reviews", []))
        print(f"  [OK] Reviews accessible – got {count} review(s)")
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def check_gcs():
    """Test GCS bucket access for financial reports."""
    print("\n=== GCS Financial Reports ===")
    try:
        from google.cloud import storage

        key_path = os.environ.get("GOOGLE_PLAY_SERVICE_ACCOUNT_KEY", "")
        dev_id = os.environ.get("GOOGLE_PLAY_DEVELOPER_ID", "")
        if not dev_id:
            print("  [SKIP] GOOGLE_PLAY_DEVELOPER_ID not set")
            return False

        client = storage.Client.from_service_account_json(key_path)
        bucket_name = f"pubsite_prod_rev_{dev_id}"
        bucket = client.bucket(bucket_name)

        # List first few blobs to verify access
        blobs = list(bucket.list_blobs(max_results=5))
        print(f"  [OK] GCS bucket '{bucket_name}' accessible – {len(blobs)} blob(s) found")
        for b in blobs:
            print(f"       - {b.name}")
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def main():
    print("Google Play Console MCP – Setup Verification")
    print("=" * 50)

    results = {
        "env": check_env(),
        "key": check_service_account(),
        "creds": check_credentials(),
        "publisher": check_publisher_api(),
        "reviews": check_reviews_api(),
        "gcs": check_gcs(),
    }

    print("\n" + "=" * 50)
    print("Summary:")
    all_ok = all(results.values())
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    if all_ok:
        print("\nAll checks passed! MCP server is ready to use.")
    else:
        print("\nSome checks failed. Please fix the issues above.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
