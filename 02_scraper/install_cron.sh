#!/usr/bin/env bash
# =============================================================================
# install_cron.sh
#
# Master's thesis: Learning the Levers — Which Creator-Controllable
# Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
# from a Swiss German-Language Perspective
# Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026
#
# Purpose: Install the project crontab (hourly scrapes, revisits, backups,
#          feature batches) and run pre-flight checks for required paths,
#          scripts, Python venv, and the flock binary.
# Inputs:  02_scraper/crontab.txt
# Outputs: User crontab (replaced wholesale); creates /data/logs,
#          /data/backups, /data/fallback if missing.
# Usage:   chmod +x 02_scraper/install_cron.sh
#          ./02_scraper/install_cron.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CRONTAB_FILE="${SCRIPT_DIR}/crontab.txt"

if [ ! -f "$CRONTAB_FILE" ]; then
    echo "[ERROR] crontab.txt not found at ${CRONTAB_FILE}"
    exit 1
fi

echo "=== Current crontab (before) ==="
crontab -l 2>/dev/null || echo "(empty)"
echo ""

echo "=== Installing crontab from ${CRONTAB_FILE} ==="
crontab "$CRONTAB_FILE"

echo ""
echo "=== New crontab (after) ==="
crontab -l

echo ""
echo "=== Pre-flight checks ==="

# Check that required directories exist
for dir in /data/logs /data/backups /data/fallback; do
    if [ -d "$dir" ]; then
        echo "  [OK] $dir exists"
    else
        mkdir -p "$dir"
        echo "  [CREATED] $dir"
    fi
done

# Check that key scripts exist
# REPO is derived from this script's location (two levels up from 02_scraper/).
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
for script in \
    "$REPO/02_scraper/run_hourly.py" \
    "$REPO/02_scraper/run_revisits.py" \
    "$REPO/03_features/run_features_batch.py" \
    "$REPO/04_database/backup_db.sh"; do
    if [ -f "$script" ]; then
        echo "  [OK] $script"
    else
        echo "  [WARN] $script NOT FOUND"
    fi
done

# Check Python venv
PYTHON="$REPO/.venv/bin/python"
if [ -x "$PYTHON" ]; then
    echo "  [OK] Python venv: $($PYTHON --version)"
else
    echo "  [WARN] Python venv not found at $PYTHON"
fi

# Check flock availability
if command -v flock &> /dev/null; then
    echo "  [OK] flock available"
else
    echo "  [WARN] flock not found - install with: sudo apt install util-linux"
fi

echo ""
echo "Done. Cron jobs are now active."
echo "Monitor with: tail -f /data/logs/hourly.log"
echo "Debug cron:   grep CRON /var/log/syslog"
