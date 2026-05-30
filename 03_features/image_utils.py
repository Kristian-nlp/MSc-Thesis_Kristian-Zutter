"""
image_utils.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    Shared image-fetching helper for the visual feature pipeline.
    Downloads thumbnails from platform CDN URLs into memory only and
    returns an OpenCV BGR ndarray. Supports the residential proxy
    (PROXY_HOST / PROXY_PORT / PROXY_USER / PROXY_PASS via .env) so
    requests can avoid CDN blocks on cloud IP ranges.

Inputs:
    .env                              proxy credentials, USER_AGENT
    (none — library module)

Outputs:
    (none — library module; returns numpy ndarrays in memory.
    Imported by the visual feature extractor.)

Usage:
    from image_utils import fetch_image
    img = fetch_image(url)
"""

import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

# Load .env so proxy settings are available when run as a subprocess
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)

# HTTP settings for thumbnail fetching
REQUEST_TIMEOUT = 15          # seconds per request
REQUEST_RETRIES = 2           # retry once on transient failure
RETRY_BACKOFF = 1.0           # seconds between retries

# Minimum image dimensions to consider valid (pixels)
MIN_IMAGE_DIM = 10

# User-Agent: match the scraper's UA from settings.py for consistency.
# Override via USER_AGENT env var if needed.
USER_AGENT = os.getenv(
    "USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/145.0.7632.109 Safari/537.36",
)

# ---------------------------------------------------------------------------
# Proxy configuration (read from env vars, same as settings.py)
# ---------------------------------------------------------------------------
_PROXY_HOST = os.getenv("PROXY_HOST", "")
_PROXY_PORT = os.getenv("PROXY_PORT", "1000")
_PROXY_USER = os.getenv("PROXY_USER", "")
_PROXY_PASS = os.getenv("PROXY_PASS", "")

_PROXY_URL = ""
if _PROXY_USER and _PROXY_PASS and _PROXY_HOST:
    _PROXY_URL = f"http://{_PROXY_USER}:{_PROXY_PASS}@{_PROXY_HOST}:{_PROXY_PORT}"

PROXIES: dict[str, str] | None = (
    {"http": _PROXY_URL, "https": _PROXY_URL} if _PROXY_URL else None
)

_proxy_logged = False


def _log_proxy_status():
    """Log proxy configuration once on first fetch."""
    global _proxy_logged
    if _proxy_logged:
        return
    _proxy_logged = True
    if PROXIES:
        logger.info(
            "Thumbnail fetches will use residential proxy (%s:%s)",
            _PROXY_HOST, _PROXY_PORT,
        )
    else:
        logger.warning(
            "No proxy configured for thumbnail fetches "
            "(PROXY_USER/PROXY_PASS/PROXY_HOST env vars not set). "
            "Requests will use the VM's direct IP -- CDN blocks possible."
        )


def fetch_image_to_array(url: str) -> np.ndarray | None:
    """
    Fetch an image from a URL and decode it into an OpenCV BGR array.

    The image bytes are held in memory only; nothing is written to
    disc. Returns None on any failure (network error, invalid image,
    corrupt data).

    Args:
        url: Thumbnail / cover image URL.

    Returns:
        NumPy array (H x W x 3, dtype uint8, BGR) or None.
    """
    if not url or not url.startswith("http"):
        return None

    _log_proxy_status()

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "de-CH,de;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
    }

    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            resp = requests.get(
                url,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
                stream=False,
                proxies=PROXIES,
            )
            resp.raise_for_status()

            # Decode raw bytes into a NumPy array
            img_array = np.frombuffer(resp.content, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                logger.warning("cv2.imdecode returned None for %s", url[:120])
                return None

            h, w = img.shape[:2]
            if h < MIN_IMAGE_DIM or w < MIN_IMAGE_DIM:
                logger.warning(
                    "Image too small (%dx%d) from %s", w, h, url[:120]
                )
                return None

            return img

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            # 403/404/410 are permanent -- do not retry
            if status in (403, 404, 410):
                logger.debug(
                    "HTTP %s for thumbnail (permanent, skipping): %s",
                    status, url[:120],
                )
                return None
            logger.warning(
                "HTTP %s fetching thumbnail (attempt %d/%d): %s",
                status, attempt, REQUEST_RETRIES, url[:120],
            )

        except requests.exceptions.RequestException as e:
            logger.warning(
                "Network error fetching thumbnail (attempt %d/%d): %s -- %s",
                attempt, REQUEST_RETRIES, url[:120], e,
            )

        # Back off before retry
        if attempt < REQUEST_RETRIES:
            time.sleep(RETRY_BACKOFF * attempt)

    return None
