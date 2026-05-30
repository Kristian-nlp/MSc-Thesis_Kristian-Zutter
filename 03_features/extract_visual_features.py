"""
extract_visual_features.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    Batch visual-feature extraction. Reads thumbnail_url from the
    posts table, fetches each image into memory only (no file
    written), and computes brightness (HSV V-mean), RMS contrast,
    Hasler-Suesstrunk colourfulness, face detection (OpenCV Haar
    cascade by default, MediaPipe optional), and OCR text length via
    pytesseract. Only the character count is stored, not the text
    itself. Incremental: posts already in features_visual are
    skipped.

    Citations:
        Hasler & Suesstrunk (2003), "Measuring Colorfulness in
            Natural Images", Proc. SPIE 5007.
        Viola & Jones (2001), "Rapid Object Detection using a
            Boosted Cascade of Simple Features", CVPR.

Inputs:
    01_config/settings.py             DB_PATH
    04_database/scraper.db            posts.thumbnail_url
    System binary: tesseract-ocr

Outputs:
    04_database/scraper.db            features_visual table

Usage:
    python 03_features/extract_visual_features.py
    python 03_features/extract_visual_features.py --use-mediapipe
    python 03_features/extract_visual_features.py --dry-run
"""

import argparse
import logging
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Resolve numbered-directory imports (04_database/db.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "04_database"))

# ---------------------------------------------------------------------------
# Optional imports with clear error messages
# ---------------------------------------------------------------------------
try:
    import cv2
except ImportError:
    sys.exit(
        "ERROR: 'opencv-python-headless' not installed.\n"
        "  Install with:  pip install opencv-python-headless "
        "--break-system-packages"
    )

try:
    import numpy as np
except ImportError:
    sys.exit(
        "ERROR: 'numpy' not installed.\n"
        "  Install with:  pip install numpy --break-system-packages"
    )

try:
    import pytesseract
except ImportError:
    sys.exit(
        "ERROR: 'pytesseract' not installed.\n"
        "  Install with:  pip install pytesseract --break-system-packages\n"
        "  Also requires:  sudo apt install tesseract-ocr"
    )

try:
    from PIL import Image
except ImportError:
    sys.exit(
        "ERROR: 'Pillow' not installed.\n"
        "  Install with:  pip install Pillow --break-system-packages"
    )

from image_utils import fetch_image_to_array

logger = logging.getLogger(__name__)


# ===================================================================
# Configuration
# ===================================================================

DB_PATH = Path(__file__).resolve().parent.parent / "04_database" / "scraper.db"

# How many posts to process per DB transaction
BATCH_SIZE = 200

# Polite delay between consecutive HTTP fetches (seconds)
INTER_REQUEST_DELAY = 0.25

# Haar cascade configuration
HAAR_SCALE_FACTOR = 1.1
HAAR_MIN_NEIGHBOURS = 5
HAAR_MIN_FACE_SIZE = (20, 20)  # minimum face size in pixels

# MediaPipe model path (optional, for --use-mediapipe)
# Place blaze_face_short_range.tflite in this directory before passing --use-mediapipe.
MEDIAPIPE_MODEL_DIR = Path(__file__).resolve().parent.parent / "04_database" / "models"
MEDIAPIPE_MODEL_FILENAME = "blaze_face_short_range.tflite"
MEDIAPIPE_CONFIDENCE = 0.5


# ===================================================================
# Visual feature computation (brightness, contrast, colourfulness)
# ===================================================================

def compute_brightness(img: np.ndarray) -> float:
    """
    Mean brightness via the V (value) channel of HSV colour space.

    Args:
        img: BGR image array (H x W x 3).

    Returns:
        Mean brightness as a float (0.0 -- 255.0).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    return float(np.mean(v_channel))


def compute_contrast(img: np.ndarray) -> float:
    """
    RMS contrast: standard deviation of greyscale pixel intensities.

    Args:
        img: BGR image array (H x W x 3).

    Returns:
        RMS contrast as a float (0.0 -- ~127.5 theoretical max).
    """
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.std(grey))


def compute_colourfulness(img: np.ndarray) -> float:
    """
    Hasler & Suesstrunk (2003) colourfulness metric.

    Formula:
        rg = R - G
        yb = 0.5 * (R + G) - B
        sigma_rgyb = sqrt(sigma_rg^2 + sigma_yb^2)
        mu_rgyb    = sqrt(mu_rg^2 + mu_yb^2)
        C = sigma_rgyb + 0.3 * mu_rgyb

    Args:
        img: BGR image array (H x W x 3).

    Returns:
        Colourfulness score (unbounded, typically 0 -- ~200 for
        natural images; pure grey = 0).
    """
    # Split channels (OpenCV uses BGR order)
    B, G, R = img[:, :, 0].astype(np.float64), \
              img[:, :, 1].astype(np.float64), \
              img[:, :, 2].astype(np.float64)

    # Opponent colour channels
    rg = R - G
    yb = 0.5 * (R + G) - B

    # Statistics
    sigma_rg = np.std(rg)
    sigma_yb = np.std(yb)
    mu_rg = np.mean(rg)
    mu_yb = np.mean(yb)

    sigma_rgyb = np.sqrt(sigma_rg ** 2 + sigma_yb ** 2)
    mu_rgyb = np.sqrt(mu_rg ** 2 + mu_yb ** 2)

    return float(sigma_rgyb + 0.3 * mu_rgyb)


# ===================================================================
# Face detection: OpenCV Haar cascade (default)
# ===================================================================

_haar_cascade = None  # lazy-loaded singleton


def _get_haar_cascade() -> cv2.CascadeClassifier:
    """Load the Haar cascade classifier (once, cached)."""
    global _haar_cascade
    if _haar_cascade is not None:
        return _haar_cascade

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    _haar_cascade = cv2.CascadeClassifier(cascade_path)

    if _haar_cascade.empty():
        raise RuntimeError(
            f"Failed to load Haar cascade from {cascade_path}. "
            "Ensure opencv-python-headless is installed correctly."
        )

    logger.info("Haar cascade face detector loaded from %s", cascade_path)
    return _haar_cascade


def detect_faces_haar(img: np.ndarray) -> int:
    """
    Detect faces using OpenCV's Haar cascade classifier.

    Args:
        img: BGR image array (H x W x 3).

    Returns:
        Number of faces detected.
    """
    cascade = _get_haar_cascade()

    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grey = cv2.equalizeHist(grey)

    faces = cascade.detectMultiScale(
        grey,
        scaleFactor=HAAR_SCALE_FACTOR,
        minNeighbors=HAAR_MIN_NEIGHBOURS,
        minSize=HAAR_MIN_FACE_SIZE,
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    # detectMultiScale returns a numpy array or an empty tuple
    if isinstance(faces, np.ndarray):
        return len(faces)
    return 0


# ===================================================================
# Face detection: MediaPipe (optional, --use-mediapipe)
# ===================================================================

_mediapipe_detector = None


def _get_mediapipe_detector(model_path: Path) -> object:
    """
    Initialise the MediaPipe FaceDetector (tasks API).

    Requires the blaze_face_short_range.tflite model file to be
    present on disc.
    """
    global _mediapipe_detector
    if _mediapipe_detector is not None:
        return _mediapipe_detector

    try:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceDetector,
            FaceDetectorOptions,
        )
    except ImportError:
        raise ImportError(
            "mediapipe is not installed. Install with:\n"
            "  pip install mediapipe --break-system-packages"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found at {model_path}.\n"
            "Download it with:\n"
            "  wget -P /data/models/ https://storage.googleapis.com/"
            "mediapipe-models/face_detector/blaze_face_short_range/"
            "float16/1/blaze_face_short_range.tflite"
        )

    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        min_detection_confidence=MEDIAPIPE_CONFIDENCE,
    )
    _mediapipe_detector = FaceDetector.create_from_options(options)
    logger.info("MediaPipe face detector loaded from %s", model_path)
    return _mediapipe_detector


def detect_faces_mediapipe(img: np.ndarray, model_path: Path) -> int:
    """
    Detect faces using MediaPipe's FaceDetector (tasks API).

    Args:
        img: BGR image array (H x W x 3).
        model_path: Path to the .tflite model file.

    Returns:
        Number of faces detected.
    """
    import mediapipe as mp

    detector = _get_mediapipe_detector(model_path)

    # MediaPipe expects RGB, not BGR
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb,
    )

    result = detector.detect(mp_image)
    return len(result.detections) if result.detections else 0


# ===================================================================
# OCR: pytesseract
# ===================================================================

def compute_ocr_text_length(img: np.ndarray) -> int:
    """
    Extract text from an image using Tesseract OCR and return
    the stripped character count.

    Pre-processing: greyscale + Otsu binarisation + PSM 6.
    Only the character count is stored (privacy by design).

    Args:
        img: BGR image array (H x W x 3).

    Returns:
        Character count of detected text (stripped of whitespace).
        Returns 0 if no text detected.
    """
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu's binarisation: automatically finds optimal threshold
    _, binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    pil_img = Image.fromarray(binary)

    try:
        text = pytesseract.image_to_string(
            pil_img,
            config="--psm 6",
        )
    except pytesseract.TesseractError as e:
        logger.warning("Tesseract error: %s", e)
        return 0

    stripped = text.strip()
    # Filter out very short noise (single random characters)
    if len(stripped) <= 1:
        return 0

    return len(stripped)


# ===================================================================
# Combined feature computation (single-pass)
# ===================================================================

def compute_all_visual_features(
    img: np.ndarray,
    use_mediapipe: bool = False,
    mediapipe_model_path: Path | None = None,
) -> dict:
    """
    Compute ALL visual features from a single image in one pass.

    This avoids downloading the same thumbnail twice (once for
    brightness/contrast/colourfulness, once for face/OCR).

    Args:
        img: BGR image array (H x W x 3).
        use_mediapipe: If True, use MediaPipe instead of Haar cascade.
        mediapipe_model_path: Path to .tflite model (required if
                              use_mediapipe is True).

    Returns:
        Dict with keys: brightness, contrast, colourfulness,
        face_count, face_flag, ocr_text_len.
    """
    # Pixel-level metrics
    brightness = round(compute_brightness(img), 4)
    contrast = round(compute_contrast(img), 4)
    colourfulness = round(compute_colourfulness(img), 4)

    # Face detection
    if use_mediapipe and mediapipe_model_path:
        face_count = detect_faces_mediapipe(img, mediapipe_model_path)
    else:
        face_count = detect_faces_haar(img)

    # OCR
    ocr_text_len = compute_ocr_text_length(img)

    return {
        "brightness": brightness,
        "contrast": contrast,
        "colourfulness": colourfulness,
        "face_count": face_count,
        "face_flag": 1 if face_count > 0 else 0,
        "ocr_text_len": ocr_text_len,
    }


# ===================================================================
# Database operations
# ===================================================================

def get_unprocessed_posts(
    conn: sqlite3.Connection,
    limit: int | None = None,
) -> list[dict]:
    """
    Fetch posts that have a thumbnail_url but no features_visual row.

    Args:
        conn: Open SQLite connection.
        limit: Max rows to fetch (None = all).

    Returns:
        List of dicts with post_id and thumbnail_url.
    """
    sql = """
        SELECT p.post_id, p.thumbnail_url
        FROM posts p
        LEFT JOIN features_visual fv ON p.post_id = fv.post_id
        WHERE fv.post_id IS NULL
          AND p.thumbnail_url IS NOT NULL
          AND p.thumbnail_url != ''
        ORDER BY p.created_at ASC
    """
    params: list = []
    if limit:
        sql += " LIMIT ?"
        params.append(int(limit))

    rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def count_null_thumbnail_posts(conn: sqlite3.Connection) -> int:
    """Count posts that have no thumbnail_url (cannot be processed)."""
    row = conn.execute(
        """
        SELECT COUNT(*) AS cnt FROM posts
        WHERE (thumbnail_url IS NULL OR thumbnail_url = '')
        """
    ).fetchone()
    return row["cnt"] if row else 0


def write_features_batch(
    conn: sqlite3.Connection,
    features: list[dict],
) -> int:
    """
    Insert a batch of computed visual features into features_visual.

    Writes ALL columns in a single INSERT (brightness, contrast,
    colourfulness, face_count, face_flag, ocr_text_len). Uses
    INSERT OR IGNORE for crash-restart safety.

    Args:
        conn: Open SQLite connection.
        features: List of dicts with all features_visual columns.

    Returns:
        Number of rows actually inserted.
    """
    if not features:
        return 0

    inserted = 0
    with conn:
        for feat in features:
            try:
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO features_visual (
                        post_id, brightness, contrast, colourfulness,
                        face_count, face_flag, ocr_text_len, processed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        feat["post_id"],
                        feat["brightness"],
                        feat["contrast"],
                        feat["colourfulness"],
                        feat["face_count"],
                        feat["face_flag"],
                        feat["ocr_text_len"],
                        feat["processed_at"],
                    ),
                )
                if cursor.rowcount > 0:
                    inserted += 1
            except sqlite3.Error as e:
                logger.error(
                    "Failed to insert visual features for %s: %s",
                    feat["post_id"], e,
                )

    return inserted


# ===================================================================
# Main pipeline
# ===================================================================

def process_post(
    row: dict,
    use_mediapipe: bool = False,
    mediapipe_model_path: Path | None = None,
) -> dict | None:
    """
    Fetch thumbnail and compute ALL visual features for a single post.

    Args:
        row: Dict with post_id and thumbnail_url.
        use_mediapipe: Whether to use MediaPipe for face detection.
        mediapipe_model_path: Path to MediaPipe model file.

    Returns:
        Dict ready for insertion into features_visual, or None if
        the image could not be fetched/decoded.
    """
    post_id = row["post_id"]
    url = row["thumbnail_url"]

    img = fetch_image_to_array(url)
    if img is None:
        return None

    try:
        features = compute_all_visual_features(
            img,
            use_mediapipe=use_mediapipe,
            mediapipe_model_path=mediapipe_model_path,
        )
    except Exception as e:
        logger.error("Feature computation failed for %s: %s", post_id, e)
        return None
    finally:
        # Explicitly release the image array
        del img

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "post_id": post_id,
        "brightness": features["brightness"],
        "contrast": features["contrast"],
        "colourfulness": features["colourfulness"],
        "face_count": features["face_count"],
        "face_flag": features["face_flag"],
        "ocr_text_len": features["ocr_text_len"],
        "processed_at": now,
    }


def run(
    db_path: Path | str | None = None,
    batch_size: int = BATCH_SIZE,
    limit: int | None = None,
    dry_run: bool = False,
    use_mediapipe: bool = False,
) -> dict:
    """
    Run the full visual-feature extraction pipeline.

    Downloads each thumbnail once and computes all six visual features
    (brightness, contrast, colourfulness, face_count, face_flag,
    ocr_text_len) in a single pass per image.

    Args:
        db_path: Path to the SQLite database. Defaults to /data/scraper.db.
        batch_size: Rows per DB transaction.
        limit: Max total posts to process (None = all pending).
        dry_run: If True, fetch and compute but do not write to DB.
        use_mediapipe: Use MediaPipe instead of Haar cascade for faces.

    Returns:
        Dict with summary stats: total, processed, skipped, errors.
    """
    path = Path(db_path) if db_path else DB_PATH
    detector = "mediapipe" if use_mediapipe else "haar"
    logger.info(
        "Visual feature extraction starting (db=%s, face_detector=%s)",
        path, detector,
    )

    # Resolve MediaPipe model path if needed
    mediapipe_model_path = None
    if use_mediapipe:
        mediapipe_model_path = MEDIAPIPE_MODEL_DIR / MEDIAPIPE_MODEL_FILENAME
        # Validate early so we fail fast
        _get_mediapipe_detector(mediapipe_model_path)

    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")

    stats = {
        "total": 0,
        "processed": 0,
        "skipped": 0,
        "errors": 0,
        "no_thumbnail": 0,
        "detector": detector,
    }

    try:
        # Eagerly load the Haar cascade
        if not use_mediapipe:
            _get_haar_cascade()

        # Report coverage
        no_thumb = count_null_thumbnail_posts(conn)
        stats["no_thumbnail"] = no_thumb
        if no_thumb > 0:
            logger.info(
                "%d posts have no thumbnail_url and will be skipped", no_thumb
            )

        # Fetch unprocessed posts
        posts = get_unprocessed_posts(conn, limit=limit)
        stats["total"] = len(posts)
        logger.info("Found %d unprocessed posts with thumbnail URLs", len(posts))

        if not posts:
            logger.info("Nothing to do -- all posts already processed")
            return stats

        # Process in batches
        batch: list[dict] = []

        for i, row in enumerate(posts, 1):
            try:
                features = process_post(
                    row,
                    use_mediapipe=use_mediapipe,
                    mediapipe_model_path=mediapipe_model_path,
                )

                if features is None:
                    stats["skipped"] += 1
                    continue

                batch.append(features)

            except Exception as e:
                stats["errors"] += 1
                logger.error(
                    "Unexpected error processing post %s: %s",
                    row["post_id"], e,
                )
                continue

            # Polite delay between HTTP requests
            if INTER_REQUEST_DELAY > 0:
                time.sleep(INTER_REQUEST_DELAY)

            # Flush batch
            if len(batch) >= batch_size:
                if not dry_run:
                    written = write_features_batch(conn, batch)
                    stats["processed"] += written
                else:
                    stats["processed"] += len(batch)

                logger.info(
                    "Progress: %d / %d posts "
                    "(batch written: %d, skipped so far: %d)",
                    i, len(posts), len(batch), stats["skipped"],
                )
                batch = []

        # Flush remaining
        if batch:
            if not dry_run:
                written = write_features_batch(conn, batch)
                stats["processed"] += written
            else:
                stats["processed"] += len(batch)

        # Log warning about skipped posts (potential URL expiry)
        if stats["skipped"] > 0:
            skip_pct = 100.0 * stats["skipped"] / stats["total"]
            logger.warning(
                "%d posts (%.1f%%) had unfetchable thumbnails "
                "(possible URL expiry or CDN block). Consider running "
                "the visual pipeline more frequently if this exceeds 5%%.",
                stats["skipped"], skip_pct,
            )

        logger.info(
            "Visual feature extraction complete: "
            "total=%d, processed=%d, skipped=%d, errors=%d, "
            "no_thumbnail=%d, detector=%s",
            stats["total"],
            stats["processed"],
            stats["skipped"],
            stats["errors"],
            stats["no_thumbnail"],
            stats["detector"],
        )

    except Exception as e:
        logger.exception("Fatal error in visual feature extraction: %s", e)
        raise

    finally:
        conn.close()

    return stats


# ===================================================================
# CLI entry point
# ===================================================================

def main():
    """
    Command-line entry point.

    Usage:
        python extract_visual_features.py [--db PATH] [--batch N]
                                          [--limit N] [--dry-run]
                                          [--use-mediapipe]
                                          [--log-level LEVEL]
    """
    parser = argparse.ArgumentParser(
        description=(
            "Extract all visual features (brightness, contrast, "
            "colourfulness, face detection, OCR) from post thumbnails. "
            "Single-pass: downloads each thumbnail once."
        ),
    )
    parser.add_argument(
        "--db", type=str, default=str(DB_PATH),
        help=f"Path to SQLite database (default: {DB_PATH})",
    )
    parser.add_argument(
        "--batch", type=int, default=BATCH_SIZE,
        help=f"Batch size for DB writes (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max posts to process (default: all pending)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch and compute features but do not write to the database",
    )
    parser.add_argument(
        "--use-mediapipe", action="store_true",
        help=(
            "Use MediaPipe FaceDetector instead of OpenCV Haar cascade. "
            "Requires blaze_face_short_range.tflite in /data/models/."
        ),
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stats = run(
        db_path=args.db,
        batch_size=args.batch,
        limit=args.limit,
        dry_run=args.dry_run,
        use_mediapipe=args.use_mediapipe,
    )

    # Exit code: 0 if no errors, 1 if some errors occurred
    if stats["errors"] > 0:
        logger.warning(
            "Completed with %d errors out of %d posts",
            stats["errors"], stats["total"],
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
