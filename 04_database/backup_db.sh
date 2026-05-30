#!/usr/bin/env bash
# =============================================================================
# backup_db.sh
#
# Master's thesis: Learning the Levers — Which Creator-Controllable
# Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
# from a Swiss German-Language Perspective
# Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026
#
# Purpose: Daily SQLite backup using `.backup` (safe with WAL writes),
#          with rotation of local copies and Telegram alerts on failure.
# Inputs:  04_database/scraper.db, .env (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
# Outputs: 04_database/backups/scraper_<UTC timestamp>.db (rotated to N days)
# Usage:   /bin/bash 04_database/backup_db.sh
#          # or via cron (see 02_scraper/crontab.txt)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Derive repo root from this script's location (04_database/ -> repo root)
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DB_PATH="${REPO_ROOT}/04_database/scraper.db"
LOCAL_BACKUP_DIR="${REPO_ROOT}/04_database/backups"
KEEP_DAYS=7                                   # number of local backups to retain
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
BACKUP_FILE="scraper_${TIMESTAMP}.db"

# Load .env for Telegram credentials (if available)
ENV_FILE="${REPO_ROOT}/.env"
if [ -f "$ENV_FILE" ]; then
    # Source only TELEGRAM_ vars to avoid polluting the environment
    TELEGRAM_BOT_TOKEN=$(grep -oP '^TELEGRAM_BOT_TOKEN=\K.*' "$ENV_FILE" 2>/dev/null || true)
    TELEGRAM_CHAT_ID=$(grep -oP '^TELEGRAM_CHAT_ID=\K.*' "$ENV_FILE" 2>/dev/null || true)
fi

# Optional: Google Cloud Storage bucket (uncomment when ready)
# GCS_BUCKET="gs://master-thesis-backups"

# ---------------------------------------------------------------------------
# Helper: send Telegram alert
# ---------------------------------------------------------------------------
send_telegram() {
    local message="$1"
    if [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ]; then
        curl -s -X POST \
            "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
            -d "chat_id=${TELEGRAM_CHAT_ID}" \
            -d "text=${message}" \
            > /dev/null 2>&1 || true
    fi
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if [ ! -f "$DB_PATH" ]; then
    echo "[ERROR] Database not found at ${DB_PATH}. Exiting."
    send_telegram "BACKUP FAILURE: Database not found at ${DB_PATH}"
    exit 1
fi

mkdir -p "$LOCAL_BACKUP_DIR"

# ---------------------------------------------------------------------------
# 1. Local backup using SQLite .backup (safe with WAL mode)
# ---------------------------------------------------------------------------
echo "[$(date -u +%H:%M:%S)] Starting backup: ${BACKUP_FILE}"

if sqlite3 "$DB_PATH" ".backup '${LOCAL_BACKUP_DIR}/${BACKUP_FILE}'"; then
    echo "[$(date -u +%H:%M:%S)] Local backup complete: ${LOCAL_BACKUP_DIR}/${BACKUP_FILE}"
    # Log the file size for monitoring
    BACKUP_SIZE=$(du -h "${LOCAL_BACKUP_DIR}/${BACKUP_FILE}" | cut -f1)
    echo "[$(date -u +%H:%M:%S)] Backup size: ${BACKUP_SIZE}"
else
    echo "[ERROR] SQLite backup failed."
    send_telegram "BACKUP FAILURE: SQLite .backup command failed for ${DB_PATH}"
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Rotate old local backups (keep last N days)
# ---------------------------------------------------------------------------
DELETED=$(find "$LOCAL_BACKUP_DIR" -name "scraper_*.db" -type f -mtime +${KEEP_DAYS} | wc -l)
find "$LOCAL_BACKUP_DIR" -name "scraper_*.db" -type f -mtime +${KEEP_DAYS} -delete
echo "[$(date -u +%H:%M:%S)] Rotated ${DELETED} old backup(s) (keeping last ${KEEP_DAYS} days)."

# ---------------------------------------------------------------------------
# 3. Optional: Upload to Google Cloud Storage
# ---------------------------------------------------------------------------
# Uncomment the block below once you have created a GCS bucket.
# The first 5 GB of GCS standard storage is always free.
#
# Create bucket (one-time):
#   gsutil mb -l europe-west6 gs://master-thesis-backups
#
# if command -v gsutil &> /dev/null; then
#     gsutil cp "${LOCAL_BACKUP_DIR}/${BACKUP_FILE}" "${GCS_BUCKET}/${BACKUP_FILE}"
#     echo "[$(date -u +%H:%M:%S)] GCS upload complete: ${GCS_BUCKET}/${BACKUP_FILE}"
# else
#     echo "[WARN] gsutil not found. Skipping GCS upload."
# fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo "[$(date -u +%H:%M:%S)] Backup finished successfully."
