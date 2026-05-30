"""
instagram_json_parser.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    Network JSON interceptor for Instagram Explore. Captures and
    parses background GraphQL and feed-API responses, which expose
    rich metadata not available in the DOM (author info, engagement
    counters, timestamps, follower counts, media details). Uses
    Selenium's performance logging via CDP, consistent with the
    TikTok JSON parser approach.

Inputs:
    (none — library module; receives an active Selenium WebDriver)

Outputs:
    (none — library module; returns InstagramJSONPost dataclasses
    to the join logic. Imported by the instagram module.)

Usage:
    from instagram_json_parser import intercept_and_parse
    posts = intercept_and_parse(driver, snapshot_id="ig_fresh")
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
# Instagram's internal API endpoints. Update when needed.
# Last verified: 28 February 2026
# =====================================================================

INSTAGRAM_API_PATTERNS = [
    r"/graphql/query",                      # GraphQL endpoint
    r"/api/v1/discover/web/explore_grid",   # Explore grid API
    r"/api/v1/discover/topical_explore",    # Topical explore variant
    r"/api/v1/feed/search",                 # Search/explore feed
    r"/api/v1/feed/timeline",              # Timeline feed (initial grid data)
    r"/api/v1/explore/grid",               # Grid endpoint variant
    r"/web/explore/grid",                   # Web explore grid
    r"/api/v1/media",                      # Single media endpoint
]

_API_REGEX = re.compile("|".join(INSTAGRAM_API_PATTERNS))


@dataclass
class InstagramJSONPost:
    """
    Post data extracted from an Instagram API JSON response.

    This is the JSON-only representation. The join logic merges this
    with DOM data (InstagramTileData) using shortcode as the join key.

    Instagram JSON provides rich metadata not in the DOM: author info,
    engagement counters, timestamps, and media details.
    """
    # Identification
    post_id: str = ""                       # Numeric media ID (pk)
    shortcode: str = ""                     # Post shortcode (join key)
    permalink: str = ""                     # Constructed permalink

    # Author
    author_id: str = ""                     # Numeric user ID (pk)
    author_handle: str = ""                 # Username
    author_nickname: str = ""               # Full name
    author_verified: bool = False           # Verified badge
    followers: int | None = None            # Follower count (NOTE: almost always None
                                            #   from explore API — IG does not include
                                            #   follower_count in explore responses.
                                            #   Requires separate profile API intercept.)
    following: int | None = None            # Following count

    # Content
    caption: str = ""                       # Full caption text
    media_type: str = ""                    # "image", "video", "carousel"
    hashtags: list[str] = field(default_factory=list)

    # Engagement counters
    likes: int | None = None                # Like count
    comments: int | None = None             # Comment count
    shares: int | None = None               # Not always available on IG
    views: int | None = None                # Video view count
    saves: int | None = None                # Save count (if available)

    # Temporal
    posted_at_utc: str = ""                 # ISO timestamp
    posted_at_unix: int | None = None       # Unix timestamp

    # Media metadata
    cover_url: str = ""                     # Display/cover image URL
    video_url: str = ""                     # Video URL (if video)
    duration_seconds: int | None = None     # Video duration
    width: int | None = None                # Media width
    height: int | None = None               # Media height

    # Location (if tagged)
    location_name: str = ""                 # Location name
    location_id: str = ""                   # Location ID

    # Raw data for debugging
    raw_item: dict = field(default_factory=dict)


def capture_network_responses(driver: WebDriver) -> list[dict]:
    """
    Retrieve all network responses captured via CDP performance logs.

    Filters for Instagram API/GraphQL endpoints that return JSON data
    with post information.

    Args:
        driver: Active Chrome WebDriver with CDP performance logging.

    Returns:
        List of dicts with keys: url, body, status, mime_type.
    """
    responses = []

    try:
        logs = driver.get_log("performance")
    except Exception as e:
        logger.error("Failed to retrieve performance logs: %s", e)
        return responses

    logger.debug("Processing %d performance log entries", len(logs))

    for entry in logs:
        try:
            log_message = json.loads(entry["message"])
            message = log_message.get("message", {})

            if message.get("method") != "Network.responseReceived":
                continue

            params = message.get("params", {})
            response = params.get("response", {})
            url = response.get("url", "")
            request_id = params.get("requestId", "")
            status = response.get("status", 0)
            mime_type = response.get("mimeType", "")

            if not _is_instagram_api_response(url, mime_type):
                continue

            logger.debug(
                "Found Instagram API response: %s (status=%d)",
                url[:120],
                status,
            )

            body = _fetch_response_body(driver, request_id)
            if body is not None:
                responses.append({
                    "url": url,
                    "body": body,
                    "status": status,
                    "mime_type": mime_type,
                    "request_id": request_id,
                })

        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    logger.info(
        "Captured %d Instagram API responses from %d log entries",
        len(responses),
        len(logs),
    )

    return responses


def _is_instagram_api_response(url: str, mime_type: str) -> bool:
    """Check whether a network response is an Instagram API call with JSON data."""
    if "instagram.com" not in url:
        return False

    if "json" not in mime_type and "javascript" not in mime_type:
        return False

    if _API_REGEX.search(url):
        return True

    # Fallback: check for common API path segments
    api_indicators = ["/api/v1/", "/graphql/", "/web/"]
    return any(indicator in url for indicator in api_indicators)


def _fetch_response_body(driver: WebDriver, request_id: str) -> dict | None:
    """Fetch a response body via CDP using the request ID."""
    try:
        result = driver.execute_cdp_cmd(
            "Network.getResponseBody",
            {"requestId": request_id},
        )
        body_text = result.get("body", "")

        if not body_text:
            return None

        if result.get("base64Encoded", False):
            import base64
            body_text = base64.b64decode(body_text).decode("utf-8", errors="replace")

        return json.loads(body_text)

    except Exception as e:
        logger.debug(
            "Could not fetch body for request %s: %s",
            request_id[:20],
            str(e)[:80],
        )
        return None


def extract_posts_from_responses(
    responses: list[dict],
) -> list[InstagramJSONPost]:
    """
    Parse Instagram API responses into structured post data.

    Instagram returns posts in various nested GraphQL structures.
    This function handles known formats and deduplicates by shortcode.

    Args:
        responses: List of response dicts from capture_network_responses().

    Returns:
        List of InstagramJSONPost objects, deduplicated by shortcode.
    """
    all_posts: dict[str, InstagramJSONPost] = {}

    for resp in responses:
        body = resp.get("body")
        if not isinstance(body, dict):
            continue

        items = _extract_media_nodes(body)

        for item in items:
            post = _parse_single_item(item)
            if post.shortcode and post.shortcode not in all_posts:
                all_posts[post.shortcode] = post

    result = list(all_posts.values())
    logger.info(
        "Parsed %d unique Instagram posts from %d API responses",
        len(result),
        len(responses),
    )

    # Quantify follower_count gap — Instagram explore API rarely includes it
    with_followers = sum(1 for p in result if p.followers is not None)
    if result and with_followers < len(result):
        logger.info(
            "Instagram follower_count gap: %d/%d posts have follower data "
            "(explore API does not include follower_count; needs separate profile API)",
            with_followers,
            len(result),
        )

    return result


def _extract_section_media(layout: dict) -> list[dict]:
    """
    Extract media items from a single section's layout_content.

    Handles both the old ``medias`` format and the V2 format with
    ``one_by_two_item.clips.items`` + ``fill_items``.

    Args:
        layout: A ``layout_content`` dict from a ``sectional_items`` entry.

    Returns:
        List of media item dicts.
    """
    items: list[dict] = []

    # Old format: layout_content.medias[*].media
    medias = layout.get("medias", [])
    if isinstance(medias, list):
        for mw in medias:
            if isinstance(mw, dict):
                media = mw.get("media", {})
                if _looks_like_media_item(media):
                    items.append(media)

    # V2 format: layout_content.one_by_two_item.clips.items[*].media
    one_by_two = layout.get("one_by_two_item", {})
    if isinstance(one_by_two, dict):
        clips = one_by_two.get("clips", {})
        if isinstance(clips, dict):
            clip_items = clips.get("items", [])
            if isinstance(clip_items, list):
                for ci in clip_items:
                    if isinstance(ci, dict):
                        media = ci.get("media", {})
                        if _looks_like_media_item(media):
                            items.append(media)

    # V2 format: layout_content.fill_items[*].media
    fill_items = layout.get("fill_items", [])
    if isinstance(fill_items, list):
        for fi in fill_items:
            if isinstance(fi, dict):
                media = fi.get("media", {})
                if _looks_like_media_item(media):
                    items.append(media)

    return items


def _extract_media_nodes(body: dict) -> list[dict]:
    """
    Extract the list of media nodes from various Instagram API response formats.

    Instagram wraps media items differently depending on the endpoint:
        - GraphQL: body["data"]["..."]["edge_..."]["edges"][*]["node"]
        - REST API: body["items"] or body["media"]
        - Explore grid (old): body["sectional_items"][*]["layout_content"]["medias"][*]["media"]
        - Explore grid (V2):  body["sectional_items"][*]["layout_content"]["one_by_two_item"]["clips"]["items"][*]["media"]
                              body["sectional_items"][*]["layout_content"]["fill_items"][*]["media"]

    Args:
        body: Parsed JSON response body.

    Returns:
        List of media item dicts (may be empty).
    """
    items = []

    # ---- GraphQL response paths ----

    # Path 1: data > web_discover_media > edges > node
    try:
        edges = (
            body.get("data", {})
            .get("web_discover_media", {})
            .get("edges", [])
        )
        if edges:
            items.extend(e.get("node", {}) for e in edges if isinstance(e, dict))
            if items:
                logger.debug("Found %d items via web_discover_media.edges", len(items))
                return [i for i in items if _looks_like_media_item(i)]
    except (TypeError, KeyError, AttributeError):
        pass

    # Path 2: data > xdt_api__v1__discover__web__explore_grid > ... > medias/fill_items
    try:
        explore_data = body.get("data", {})
        for key in explore_data:
            if "explore" in key.lower() or "discover" in key.lower():
                sectional = explore_data[key].get("sectional_items", [])
                for section in sectional:
                    layout = section.get("layout_content", {})
                    items.extend(_extract_section_media(layout))
                if items:
                    logger.debug("Found %d items via sectional_items", len(items))
                    return items
    except (TypeError, KeyError, AttributeError):
        pass

    # Path 3: Generic edges pattern (various GraphQL queries)
    try:
        data = body.get("data", {})
        for key, value in data.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if "edge" in subkey and isinstance(subvalue, dict):
                        edges = subvalue.get("edges", [])
                        nodes = [e.get("node", {}) for e in edges if isinstance(e, dict)]
                        valid = [n for n in nodes if _looks_like_media_item(n)]
                        if valid:
                            logger.debug(
                                "Found %d items via data.%s.%s.edges",
                                len(valid), key, subkey,
                            )
                            return valid
    except (TypeError, KeyError, AttributeError):
        pass

    # ---- REST API response paths ----

    # Path 4: items list (REST explore endpoint)
    try:
        rest_items = body.get("items", [])
        if isinstance(rest_items, list) and rest_items:
            # Items may be wrapped in a "media" key
            unwrapped = []
            for item in rest_items:
                if isinstance(item, dict):
                    media = item.get("media", item)
                    if _looks_like_media_item(media):
                        unwrapped.append(media)
            if unwrapped:
                logger.debug("Found %d items via items[]", len(unwrapped))
                return unwrapped
    except (TypeError, KeyError, AttributeError):
        pass

    # Path 5: sectional_items (explore grid REST / V2)
    try:
        sections = body.get("sectional_items", [])
        for section in sections:
            layout = section.get("layout_content", {})
            items.extend(_extract_section_media(layout))
        if items:
            logger.debug("Found %d items via sectional_items (top-level)", len(items))
            return items
    except (TypeError, KeyError, AttributeError):
        pass

    # Nothing found: log top-level keys for debugging
    top_keys = list(body.keys())[:10] if isinstance(body, dict) else []
    logger.debug(
        "No known Instagram media structure found. Top-level keys: %s",
        top_keys,
    )

    return []


def _looks_like_media_item(item: dict) -> bool:
    """Check if a dict looks like an Instagram media item."""
    if not isinstance(item, dict):
        return False
    # Instagram media items typically have "code" (shortcode) or "pk" or "id"
    return any(k in item for k in ["code", "shortcode", "pk", "id", "media_type"])


def _parse_single_item(item: dict) -> InstagramJSONPost:
    """
    Parse a single Instagram media item into an InstagramJSONPost.

    Handles both GraphQL node format and REST API format:

    GraphQL node:
        {
            "id": "123456789",
            "shortcode": "DUYjm3sDW63",
            "display_url": "...",
            "edge_media_to_caption": {"edges": [{"node": {"text": "..."}}]},
            "edge_media_preview_like": {"count": 500},
            "edge_media_to_comment": {"count": 20},
            "owner": {"id": "111", "username": "user"},
            "taken_at_timestamp": 1740000000
        }

    REST API item:
        {
            "pk": "123456789",
            "code": "DUYjm3sDW63",
            "caption": {"text": "..."},
            "like_count": 500,
            "comment_count": 20,
            "user": {"pk": "111", "username": "user"},
            "taken_at": 1740000000
        }

    Args:
        item: A single media item dict from the API response.

    Returns:
        Populated InstagramJSONPost.
    """
    post = InstagramJSONPost()

    # -- Identification --
    post.post_id = str(item.get("pk", "") or item.get("id", ""))
    post.shortcode = item.get("code", "") or item.get("shortcode", "")

    if post.shortcode:
        post.permalink = f"https://www.instagram.com/p/{post.shortcode}/"

    # -- Author --
    # GraphQL format: "owner"
    owner = item.get("owner", {})
    # REST format: "user"
    user = item.get("user", {})
    author = owner or user

    post.author_id = str(author.get("pk", "") or author.get("id", ""))
    post.author_handle = author.get("username", "")
    post.author_nickname = author.get("full_name", "")
    post.author_verified = author.get("is_verified", False)

    # Follower counts (may be nested)
    post.followers = _safe_int(
        author.get("edge_followed_by", {}).get("count")
        or author.get("follower_count")
    )
    post.following = _safe_int(
        author.get("edge_follow", {}).get("count")
        or author.get("following_count")
    )

    # -- Caption --
    # GraphQL: edge_media_to_caption > edges > node > text
    caption_edges = item.get("edge_media_to_caption", {}).get("edges", [])
    if caption_edges:
        post.caption = caption_edges[0].get("node", {}).get("text", "")
    # REST: caption > text
    elif isinstance(item.get("caption"), dict):
        post.caption = item["caption"].get("text", "")
    elif isinstance(item.get("caption"), str):
        post.caption = item["caption"]

    post.hashtags = _extract_hashtags(post.caption)

    # -- Media type --
    # GraphQL: __typename or is_video
    typename = item.get("__typename", "")
    if typename == "GraphVideo" or item.get("is_video"):
        post.media_type = "video"
    elif typename == "GraphSidecar" or item.get("carousel_media"):
        post.media_type = "carousel"
    else:
        # REST: media_type (1=image, 2=video, 8=carousel)
        mt = item.get("media_type")
        if mt == 2:
            post.media_type = "video"
        elif mt == 8:
            post.media_type = "carousel"
        else:
            post.media_type = "image"

    # -- Engagement counters --
    # GraphQL format
    post.likes = _safe_int(
        item.get("edge_media_preview_like", {}).get("count")
        or item.get("like_count")
    )
    post.comments = _safe_int(
        item.get("edge_media_to_comment", {}).get("count")
        or item.get("comment_count")
    )
    post.views = _safe_int(
        item.get("video_view_count")
        or item.get("view_count")
        or item.get("play_count")
        or item.get("ig_play_count")
    )
    # reshare_count is only present for video/reel posts; None for images/carousels
    post.shares = _safe_int(item.get("reshare_count"))

    # -- Temporal --
    taken_at = item.get("taken_at_timestamp") or item.get("taken_at")
    if taken_at:
        post.posted_at_unix = int(taken_at)
        try:
            dt = datetime.fromtimestamp(int(taken_at), tz=timezone.utc)
            post.posted_at_utc = dt.isoformat()
        except (ValueError, OSError):
            post.posted_at_utc = ""

    # -- Media metadata --
    # Try display_url first (GraphQL), then image_versions2 candidates (REST)
    post.cover_url = item.get("display_url", "")
    if not post.cover_url:
        candidates = item.get("image_versions2", {}).get("candidates", [])
        if isinstance(candidates, list) and candidates:
            post.cover_url = candidates[0].get("url", "")

    if item.get("video_duration"):
        post.duration_seconds = _safe_int(item["video_duration"])

    dimensions = item.get("dimensions", {})
    post.width = _safe_int(dimensions.get("width") or item.get("original_width"))
    post.height = _safe_int(dimensions.get("height") or item.get("original_height"))

    # -- Location --
    location = item.get("location", {})
    if isinstance(location, dict):
        post.location_name = location.get("name", "")
        post.location_id = str(location.get("pk", "") or location.get("id", ""))

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
        responses: List of response dicts.
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


def extract_posts_from_page_source(
    driver: WebDriver,
    debug_dir: Path | None = None,
) -> list[InstagramJSONPost]:
    """
    Extract posts from SSR-embedded ``<script type="application/json">`` tags.

    Instagram server-side renders the initial Explore grid into the HTML.
    The full post metadata is embedded in JSON script tags before any
    client-side hydration or XHR fires.  This captures the Top-20 grid
    that CDP network interception misses (CDP only sees scroll-triggered
    API responses).

    Args:
        driver: Active WebDriver on an Instagram page.
        debug_dir: If provided, save raw JSON blobs for inspection.

    Returns:
        Deduplicated list of InstagramJSONPost from page source.
    """
    # Pull all <script type="application/json"> contents via JS
    try:
        script_contents: list[str] = driver.execute_script("""
            return Array.from(
                document.querySelectorAll('script[type="application/json"]')
            ).map(el => el.textContent);
        """)
    except Exception as e:
        logger.warning("Failed to extract SSR script tags: %s", e)
        return []

    if not script_contents:
        logger.debug("No <script type='application/json'> tags found")
        return []

    logger.debug("Found %d SSR script tags to inspect", len(script_contents))

    all_posts: dict[str, InstagramJSONPost] = {}
    raw_blobs: list[dict] = []  # for debug saving

    for idx, raw_text in enumerate(script_contents):
        if not raw_text or len(raw_text) < 50:
            continue

        # Quick-reject: must contain a shortcode-like key
        if '"shortcode"' not in raw_text and '"code"' not in raw_text:
            continue

        try:
            blob = json.loads(raw_text)
        except (json.JSONDecodeError, ValueError):
            continue

        if debug_dir:
            raw_blobs.append({"script_index": idx, "blob": blob})

        # Recursively find media nodes
        media_nodes = _find_media_nodes_recursive(blob, depth=0, max_depth=10)

        for node in media_nodes:
            post = _parse_single_item(node)
            if post.shortcode and post.shortcode not in all_posts:
                all_posts[post.shortcode] = post

    # Save debug blobs if requested
    if debug_dir and raw_blobs:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_path = debug_dir / "ssr_script_tags.json"
        try:
            debug_path.write_text(
                json.dumps(raw_blobs, indent=2, default=str, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.debug("SSR debug blobs saved: %s", debug_path)
        except Exception as e:
            logger.debug("Failed to save SSR debug blobs: %s", e)

    result = list(all_posts.values())
    logger.info("SSR extraction: %d posts from page source", len(result))
    return result


def _find_media_nodes_recursive(
    obj: object,
    depth: int,
    max_depth: int,
) -> list[dict]:
    """
    Recursively search a JSON structure for Instagram media nodes.

    A media node is a dict that passes ``_looks_like_media_item()`` AND
    has a ``"user"`` or ``"owner"`` key (to distinguish real posts from
    other objects that happen to have an ``"id"`` field).

    Args:
        obj: Any JSON-decoded object (dict, list, scalar).
        depth: Current recursion depth.
        max_depth: Stop recursing beyond this depth.

    Returns:
        List of media-node dicts found.
    """
    if depth > max_depth:
        return []

    results: list[dict] = []

    if isinstance(obj, dict):
        if _looks_like_media_item(obj) and ("user" in obj or "owner" in obj):
            results.append(obj)
        else:
            for value in obj.values():
                results.extend(
                    _find_media_nodes_recursive(value, depth + 1, max_depth)
                )

    elif isinstance(obj, list):
        for item in obj:
            results.extend(
                _find_media_nodes_recursive(item, depth + 1, max_depth)
            )

    return results


def intercept_and_parse(
    driver: WebDriver,
    snapshot_id: str = "",
    debug_dir: Path | None = None,
) -> list[InstagramJSONPost]:
    """
    Convenience function: capture, parse, and optionally save JSON data.

    Main entry point for Instagram JSON interception. Call after the
    page has loaded and before DOM extraction.

    Args:
        driver: Active Chrome WebDriver with CDP performance logging.
        snapshot_id: For naming debug files.
        debug_dir: If provided, save raw JSON for debugging.

    Returns:
        List of InstagramJSONPost objects parsed from API responses.
    """
    responses = capture_network_responses(driver)

    if debug_dir and responses:
        save_raw_responses(responses, snapshot_id, debug_dir)

    posts = extract_posts_from_responses(responses)

    logger.info(
        "Instagram JSON interception complete: %d API responses -> %d posts",
        len(responses),
        len(posts),
    )

    return posts
