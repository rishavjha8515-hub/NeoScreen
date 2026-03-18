"""
Day 6 — Task 3
Security Audit: scans entire codebase for credentials, PII, and unsafe patterns.

Usage:
    python day6/security_audit.py
"""

import os
import re
import sys


CREDENTIAL_PATTERNS = [
    (r'AC[a-z0-9]{32}',            "Twilio SID hardcoded"),
    (r'SK[a-z0-9]{32}',            "Twilio token hardcoded"),
    (r'AIza[0-9A-Za-z\-_]{35}',    "Google API key hardcoded"),
    (r'AKIA[0-9A-Z]{16}',          "AWS key hardcoded"),
    (r'password\s*=\s*["\'][^"\']+["\']', "Password hardcoded"),
    (r'secret\s*=\s*["\'][^"\']+["\']',   "Secret hardcoded"),
    (r'token\s*=\s*["\'][^"\']+["\']',    "Token hardcoded"),
]

PII_PATTERNS = [
    (r'mother_name',   "Mother name field"),
    (r'patient_name',  "Patient name field"),
    (r'date_of_birth', "Date of birth field"),
    (r'aadhaar',       "Aadhaar number field"),
    (r'photo_data',    "Photo data stored"),
]

SKIP_DIRS = {"venv", ".git", "__pycache__", "build", ".dart_tool", "Pods"}
SCAN_EXTENSIONS = {".py", ".dart", ".js", ".ts", ".json", ".yaml", ".yml", ".env"}


def scan_file(fpath):
    issues = []
    try:
        with open(fpath, "r", errors="ignore") as f:
            content = f.read()
    except Exception:
        return issues

    for pattern, description in CREDENTIAL_PATTERNS + PII_PATTERNS:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            issues.append((description, matches[0][:40]))

    return issues


def run():
    print("=" * 60)
    print("NeoScreen — Day 6 Task 3: Security Audit")
    print("=" * 60)

    all_issues = []
    files_scanned = 0

    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in SCAN_EXTENSIONS:
                continue
            fpath = os.path.join(root, fname)
            issues = scan_file(fpath)
            files_scanned += 1
            if issues:
                all_issues.append((fpath, issues))

    # Check .env is gitignored
    gitignore_ok = False
    if os.path.exists(".gitignore"):
        with open(".gitignore") as f:
            gitignore_ok = ".env" in f.read()

    # Check .env file itself is not in git index
    import subprocess
    result = subprocess.run(["git", "ls-files", ".env"], capture_output=True, text=True)
    env_in_git = bool(result.stdout.strip())

    print(f"\nFiles scanned: {files_scanned}")
    print()

    if all_issues:
        print("⚠  ISSUES FOUND:")
        for fpath, issues in all_issues:
            print(f"\n  {fpath}")
            for desc, sample in issues:
                print(f"    ✗  {desc}: '{sample}...'")
    else:
        print("✓  No hardcoded credentials or PII patterns found.")

    print()
    print(f"{'✓' if gitignore_ok else '✗'}  .env in .gitignore")
    print(f"{'✓' if not env_in_git else '✗ WARNING'}  .env {'not ' if not env_in_git else ''}tracked by git")

    print("\n" + "=" * 60)
    if not all_issues and gitignore_ok and not env_in_git:
        print("✓ Security audit passed. Safe to push to public GitHub.")
    else:
        print("✗ Fix issues above before pushing to public GitHub.")
        if env_in_git:
            print("\n  To untrack .env from git:")
            print("    git rm --cached .env")
            print("    git commit -m 'remove .env from tracking'")
    print("=" * 60)


if __name__ == "__main__":
    run()
