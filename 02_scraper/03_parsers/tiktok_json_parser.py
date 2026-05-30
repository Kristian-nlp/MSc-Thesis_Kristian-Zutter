"""
tiktok_json_parser.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    Network JSON interceptor for the TikTok Explore page. Captures
    and parses background API responses (XHR/fetch) that TikTok
    issues during page load, exposing metadata not available in the
    DOM (timestamps, follower counts, share/comment counts, audio
    IDs, and engagement data). Uses Selenium's performance logging
    via CDP to avoid adding a proxy layer that would conflict with
    the Evomi residential proxy.

Inputs:
    (none — library module; receives an active Selenium WebDriver)

Outputs:
    (none — library module; returns TikTokJSONPost dataclasses to
    the join logic. Imported by the tiktok module.)

Usage:
    from tiktok_json_parser import intercept_and_parse
    posts = intercept_and_parse(driver, snapshot_id="tiktok_fresh")
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from selenium.webdriver.remote.webdriver import WebDriver

logger = logging.getLogger(__name__)


# =====================================================================
# CONFIGURABLE API PATTERNS
# TikTok's internal API endpoints change. Update these when needed.
# Last verified: 28 February 2026
# =====================================================================

# URL patterns that indicate TikTok Explore API responses.
# These are the XHR/fetch endpoints that return post data as JSON.
TIKTOK_API_PATTERNS = [
    r"/api/explore/item_list",      # Primary Explore feed endpoint
    r"/api/recommend/item_list",    # Alternative recommendation endpoint
    r"/api/post/item_list",         # Post list variant
    r"/node/share/discover",        # Discover/share endpoint
]

# Compile patterns for efficient matching
_API_REGEX = re.compile("|".join(TIKTOK_API_PATTERNS))


@dataclass
class TikTokJSONPost:
    """
    Post data extracted from a TikTok API JSON response.

    This is the JSON-only representation. The join logic merges this
    with DOM data (TikTokTileData) using post_id as the join key.

    Fields are a superset of what the DOM provides, including metadata
    only available from the API: timestamps, follower counts, detailed
    engagement counters, and audio metadata.
    """
    # Identification
    post_id: str = ""                       # Video ID (matches DOM post_id)
    permalink: str = ""                     # Constructed permalink

    # Author
    author_id: str = ""                     # Unique author ID
    author_handle: str = ""                 # @username (uniqueId)
    author_nickname: str = ""               # Display name
    author_verified: bool = False           # Verified badge
    followers: int | None = None            # Author follower count
    following: int | None = None            # Author following count

    # Content
    caption: str = ""                       # Full caption/description
    media_type: str = "video"               # "video", "image", etc.
    hashtags: list[str] = field(default_factory=list)

    # Engagement counters (primary visibility proxies)
    likes: int | None = None                # diggCount
    comments: int | None = None             # commentCount
    shares: int | None = None               # shareCount
    views: int | None = None                # playCount
    saves: int | None = None                # collectCount (bookmarks)

    # Temporal
    posted_at_utc: str = ""                 # ISO timestamp of creation
    posted_at_unix: int | None = None       # Unix timestamp

    # Audio / Sound
    audio_id: str = ""                      # Music/sound ID
    audio_name: str = ""                    # Sound title
    audio_author: str = ""                  # Sound author
    audio_is_original: bool = False         # Original sound flag

    # Video metadata
    duration_seconds: int | None = None     # Video duration
    cover_url: str = ""                     # Cover image URL

    # Raw data for debugging (not stored in DB)
    raw_item: dict = field(default_factory=dict)


def capture_network_responses(driver: WebDriver) -> list[dict]:
    """
    Retrieve all network responses captured via CDP performance logs.

    The browser must have been created with performance logging enabled
    (see browser.py: goog:loggingPrefs -> performance: ALL). This
    function reads those logs and filters for Network.responseReceived
    events, then fetches the response bodies for API endpoints.

    Args:
        driver: Active Chrome WebDriver with CDP performance logging.

    Returns:
        List of dicts, each containing:
            - url: The request URL
            - body: Parsed JSON body (dict) or None if not JSON
            - status: HTTP status code
            - mime_type: Response MIME type
    """
    responses = []

    try:
        # Get all performance log entries
        logs = driver.get_log("performance")
    except Exception as e:
        logger.error("Failed to retrieve performance logs: %s", e)
        return responses

    logger.debug("Processing %d performance log entries", len(logs))

    for entry in logs:
        try:
            log_message = json.loads(entry["message"])
            message = log_message.get("message", {})

            # We only care about completed network responses
            if message.get("method") != "Network.responseReceived":
                continue

            params = message.get("params", {})
            response = params.get("response", {})
            url = response.get("url", "")
            request_id = params.get("requestId", "")
            status = response.get("status", 0)
            mime_type = response.get("mimeType", "")

            # Filter: only JSON responses from TikTok API endpoints
            if not _is_tiktok_api_response(url, mime_type):
                continue

            logger.debug(
                "Found API response: %s (status=%d, type=%s)",
                url[:120],
                status,
                mime_type,
            )

            # Fetch the response body via CDP
            body = _fetch_response_body(driver, request_id)
            if body is not None:
                responses.append({
                    "url": url,
                    "body": body,
                    "status": status,
                    "mime_type": mime_type,
                    "request_id": request_id,
                })

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Malformed log entry, skip
            continue

    logger.info(
        "Captured %d TikTok API responses from %d log entries",
        len(responses),
        len(logs),
    )

    return responses


def _is_tiktok_api_response(url: str, mime_type: str) -> bool:
    """
    Check whether a network response is a TikTok API call with JSON data.

    Args:
        url: The request URL.
        mime_type: The response MIME type.

    Returns:
        True if this looks like a TikTok API response with post data.
    """
    # Must be from tiktok.com
    if "tiktok.com" not in url and "tiktokv.com" not in url:
        return False

    # Must be JSON
    if "json" not in mime_type and "javascript" not in mime_type:
        return False

    # Must match one of our known API patterns
    if _API_REGEX.search(url):
        return True

    # Fallback: check for API path segments that suggest post data.
    # Require at least one positive indicator AND exclude known non-post endpoints.
    api_indicators = ["/api/", "/node/", "/v1/"]
    non_post_indicators = [
        "/analytics/", "/tracking/", "/config/", "/log/",
        "/pixel/", "/abtest/", "/setting/", "/passport/",
        "/captcha/", "/report/", "/search/suggest",
    ]
    has_api_path = any(indicator in url for indicator in api_indicators)
    is_excluded = any(excl in url for excl in non_post_indicators)
    return has_api_path and not is_excluded


def _fetch_response_body(driver: WebDriver, request_id: str) -> dict | None:
    """
    Fetch a response body via CDP using the request ID.

    Args:
        driver: Active Chrome WebDriver.
        request_id: The requestId from Network.responseReceived.

    Returns:
        Parsed JSON dict, or None if fetching/parsing fails.
    """
    try:
        result = driver.execute_cdp_cmd(
            "Network.getResponseBody",
            {"requestId": request_id},
        )
        body_text = result.get("body", "")

        if not body_text:
            return None

        # Some responses may be base64-encoded
        if result.get("base64Encoded", False):
            import base64
            body_text = base64.b64decode(body_text).decode("utf-8", errors="replace")

        return json.loads(body_text)

    except Exception as e:
        # Common: response body no longer available (browser discarded it)
        # This is expected for some requests and not an error
        logger.debug(
            "Could not fetch body for request %s: %s",
            request_id[:20],
            str(e)[:80],
        )
        return None


def extract_posts_from_responses(
    responses: list[dict],
) -> list[TikTokJSONPost]:
    """
    Parse TikTok API responses into structured post data.

    TikTok's API returns posts in various nested structures. This
    function handles the known formats and extracts post data from
    each. Unknown structures are logged for later investigation.

    Args:
        responses: List of response dicts from capture_network_responses().

    Returns:
        List of TikTokJSONPost objects, deduplicated by post_id.
    """
    all_posts: dict[str, TikTokJSONPost] = {}  # Deduplicate by post_id

    for resp in responses:
        body = resp.get("body")
        if not isinstance(body, dict):
            continue

        # Try known response structures
        items = _extract_item_list(body)

        for item in items:
            post = _parse_single_item(item)
            if post.post_id and post.post_id not in all_posts:
                all_posts[post.post_id] = post

    result = list(all_posts.values())
    logger.info(
        "Parsed %d unique posts from %d API responses",
        len(result),
        len(responses),
    )

    return result


def _extract_item_list(body: dict) -> list[dict]:
    """
    Extract the list of post items from various TikTok API response formats.

    TikTok wraps items differently depending on the endpoint. Known
    structures:
        - body["itemList"]                     (explore/item_list)
        - body["data"]["itemList"]             (some variants)
        - body["itemStruct"]                   (single item)
        - body["items"]                        (generic)
        - body["data"]["items"]                (nested variant)
        - body["data"]["data"]                 (double-nested)

    Args:
        body: Parsed JSON response body.

    Returns:
        List of item dicts (may be empty).
    """
    # Try each known path
    paths_to_try = [
        lambda b: b.get("itemList", []),
        lambda b: b.get("data", {}).get("itemList", []),
        lambda b: b.get("items", []),
        lambda b: b.get("data", {}).get("items", []),
        lambda b: b.get("data", {}).get("data", []),
        lambda b: [b["itemStruct"]] if "itemStruct" in b else [],
        lambda b: b.get("data", {}).get("explore_list", []),
    ]

    for path_fn in paths_to_try:
        try:
            items = path_fn(body)
            if isinstance(items, list) and len(items) > 0:
                # Verify items look like posts (have an "id" or "video" field)
                if _looks_like_post_item(items[0]):
                    logger.debug(
                        "Found %d items via %s",
                        len(items),
                        path_fn.__code__.co_consts[1] if hasattr(path_fn.__code__, 'co_consts') else "path",
                    )
                    return items
        except (TypeError, KeyError, AttributeError):
            continue

    # If nothing matched, log the top-level keys for debugging
    top_keys = list(body.keys())[:10] if isinstance(body, dict) else []
    logger.debug(
        "No known item list structure found. Top-level keys: %s",
        top_keys,
    )

    return []


def _looks_like_post_item(item: dict) -> bool:
    """Check if a dict looks like a TikTok post item."""
    if not isinstance(item, dict):
        return False
    # Posts typically have an "id" or "video" key
    return any(k in item for k in ["id", "video", "desc", "author"])


def _parse_single_item(item: dict) -> TikTokJSONPost:
    """
    Parse a single TikTok API item into a TikTokJSONPost.

    Handles the standard TikTok item structure:
        {
            "id": "7608611486151281942",
            "desc": "Caption text #hashtag",
            "createTime": 1740000000,
            "author": {
                "uniqueId": "username",
                "nickname": "Display Name",
                "verified": true,
                "id": "author_id"
            },
            "authorStats": {
                "followerCount": 50000,
                "followingCount": 200,
                ...
            },
            "stats": {
                "diggCount": 5000,
                "commentCount": 200,
                "shareCount": 100,
                "playCount": 500000,
                "collectCount": 300
            },
            "music": {
                "id": "sound_id",
                "title": "Sound Name",
                "authorName": "Sound Author"
            },
            "video": {
                "duration": 30,
                "cover": "https://..."
            }
        }

    Args:
        item: A single item dict from the API response.

    Returns:
        Populated TikTokJSONPost.
    """
    post = TikTokJSONPost()

    # -- Identification --
    post.post_id = str(item.get("id", ""))

    # Construct permalink
    author = item.get("author", {})
    unique_id = author.get("uniqueId", "")
    if post.post_id and unique_id:
        post.permalink = f"https://www.tiktok.com/@{unique_id}/video/{post.post_id}"

    # -- Author --
    post.author_id = str(author.get("id", ""))
    post.author_handle = unique_id
    post.author_nickname = author.get("nickname", "")
    post.author_verified = author.get("verified", False)

    # Author stats (may be in "author" or "authorStats")
    author_stats = item.get("authorStats", {})
    if not author_stats:
        author_stats = author.get("stats", {})
    post.followers = _safe_int(author_stats.get("followerCount"))
    post.following = _safe_int(author_stats.get("followingCount"))

    # -- Content --
    post.caption = item.get("desc", "")
    post.hashtags = _extract_hashtags(post.caption)

    # Determine media type
    if item.get("imagePost"):
        post.media_type = "image"
    elif item.get("video"):
        post.media_type = "video"
    else:
        post.media_type = "unknown"

    # -- Engagement counters --
    stats = item.get("stats", {})
    post.likes = _safe_int(stats.get("diggCount"))
    post.comments = _safe_int(stats.get("commentCount"))
    post.shares = _safe_int(stats.get("shareCount"))
    post.views = _safe_int(stats.get("playCount"))
    post.saves = _safe_int(stats.get("collectCount"))

    # -- Temporal --
    create_time = item.get("createTime")
    if create_time:
        post.posted_at_unix = int(create_time)
        try:
            dt = datetime.fromtimestamp(int(create_time), tz=timezone.utc)
            post.posted_at_utc = dt.isoformat()
        except (ValueError, OSError):
            post.posted_at_utc = ""

    # -- Audio / Sound --
    music = item.get("music", {})
    post.audio_id = str(music.get("id", ""))
    post.audio_name = music.get("title", "")
    post.audio_author = music.get("authorName", "")
    post.audio_is_original = music.get("original", False)

    # -- Video metadata --
    video = item.get("video", {})
    post.duration_seconds = _safe_int(video.get("duration"))
    post.cover_url = video.get("cover", "") or video.get("originCover", "")

    # Store raw item for debugging
    post.raw_item = item

    return post


def _extract_hashtags(caption: str) -> list[str]:
    """Extract hashtags from a caption string."""
    if not caption:
        return []
    return re.findall(r"#(\w+)", caption)


def _safe_int(value) -> int | None:
    """Safely convert a value to int, returning None on failure."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def save_raw_responses(
    responses: list[dict],
    snapshot_id: str,
    output_dir: Path,
) -> Path | None:
    """
    Save raw API responses to a JSON file for debugging.

    Args:
        responses: List of response dicts from capture_network_responses().
        snapshot_id: Identifier for this snapshot.
        output_dir: Directory to save the file in.

    Returns:
        Path to the saved file, or None if saving failed.
    """
    if not responses:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"raw_json_{snapshot_id}.json"

    save_data = []
    for resp in responses:
        save_data.append({
            "url": resp.get("url", ""),
            "status": resp.get("status", 0),
            "mime_type": resp.get("mime_type", ""),
            "body": resp.get("body"),
        })

    try:
        filepath.write_text(
            json.dumps(save_data, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Raw JSON responses saved: %s", filepath)
        return filepath
    except Exception as e:
        logger.warning("Failed to save raw JSON: %s", e)
        return None


def intercept_and_parse(
    driver: WebDriver,
    snapshot_id: str = "",
    debug_dir: Path | None = None,
) -> list[TikTokJSONPost]:
    """
    Convenience function: capture, parse, and optionally save JSON data.

    This is the main entry point for JSON interception. Call it after
    the page has loaded and before extracting the DOM, so both paths
    operate on the same page load.

    Args:
        driver: Active Chrome WebDriver with CDP performance logging.
        snapshot_id: For naming debug files.
        debug_dir: If provided, save raw JSON for debugging.

    Returns:
        List of TikTokJSONPost objects parsed from API responses.
    """
    # Step 1: Capture all API responses from performance logs
    responses = capture_network_responses(driver)

    # Step 2: Save raw responses for debugging (if requested)
    if debug_dir and responses:
        save_raw_responses(responses, snapshot_id, debug_dir)

    # Step 3: Parse items from responses
    posts = extract_posts_from_responses(responses)

    logger.info(
        "JSON interception complete: %d API responses -> %d posts",
        len(responses),
        len(posts),
    )

    return posts
