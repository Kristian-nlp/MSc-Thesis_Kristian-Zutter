"""
run_linkedin_api.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    LinkedIn scrape via the Voyager API (no browser). Replaces a
    browser-based runner because LinkedIn's PerimeterX anti-bot
    detects undetected_chromedriver and burns li_at tokens. The
    Voyager feed endpoint returns Top + Baseline posts with authors,
    engagement counters, and activity URNs over plain HTTPS.

Inputs:
    01_config/settings.py             ACCOUNTS, PROXY_URL
    01_config/cookies/linkedin_*.json li_at + JSESSIONID cookies

Outputs:
    04_database/scraper.db            snapshot + post + scrape_log rows
    DATA_DIR/logs/linkedin.log        run log

Usage:
    python 02_scraper/run_linkedin_api.py --account linkedin_seeded
    python 02_scraper/run_linkedin_api.py --account linkedin_fresh --dry-run
    python 02_scraper/run_linkedin_api.py --account linkedin_seeded --log-level DEBUG
"""

import argparse
import json
import logging
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Path setup (numbered folders cannot be imported directly)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "01_config"))
sys.path.insert(0, str(PROJECT_ROOT / "02_scraper" / "01_core"))
sys.path.insert(0, str(PROJECT_ROOT / "02_scraper" / "02_platforms"))
sys.path.insert(0, str(PROJECT_ROOT / "04_database"))

from settings_loader import load_settings
from auth import load_cookies_from_file, check_cookie_expiry
from base import CapturedPost, SnapshotResult
from logging_config import setup_logging, log_snapshot_summary, log_scrape_start
from db import write_snapshot
from alerting import send_alert

logger = logging.getLogger("run_linkedin_api")

# ---------------------------------------------------------------------------
# Voyager API configuration
# ---------------------------------------------------------------------------
VOYAGER_BASE = "https://www.linkedin.com/voyager/api"
FEED_ENDPOINT = "/feed/updatesV2"

# Headers that mimic the LinkedIn SPA's own XHR calls
API_HEADERS = {
    "Accept": "application/vnd.linkedin.normalized+json+2.1",
    "X-Li-Lang": "de_DE",
    "X-Li-Track": '{"clientVersion":"1.13.8878","mpVersion":"1.13.8878","osName":"web","timezoneOffset":1,"deviceFormFactor":"DESKTOP","mpName":"voyager-web"}',
    "X-Restli-Protocol-Version": "2.0.0",
}


# ---------------------------------------------------------------------------
# Session builder
# ---------------------------------------------------------------------------

def build_session(cookies: list[dict], proxy_url: str = "") -> requests.Session:
    """
    Build a requests.Session with LinkedIn auth cookies and proxy.

    Extracts li_at and JSESSIONID from the cookie list, sets the
    csrf-token header from JSESSIONID (required by Voyager API).

    Args:
        cookies: Cookie dicts from auth.load_cookies_from_file().
        proxy_url: Evomi proxy URL (optional).

    Returns:
        Configured requests.Session.
    """
    session = requests.Session()

    # Extract auth cookies only (routing cookies come from warm-up GET)
    li_at = ""
    jsessionid = ""
    for c in cookies:
        name = c.get("name", "")
        if name == "li_at":
            li_at = c["value"]
        elif name == "JSESSIONID":
            jsessionid = c["value"].strip('"')

    if not li_at:
        raise RuntimeError("li_at cookie not found — cannot authenticate")
    if not jsessionid:
        raise RuntimeError("JSESSIONID cookie not found — cannot set csrf-token")

    # Set common headers and proxy BEFORE warm-up
    ua = (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/145.0.7632.109 Safari/537.36"
    )
    session.headers.update({
        **API_HEADERS,
        "csrf-token": jsessionid,
        "User-Agent": ua,
    })

    if proxy_url:
        session.proxies = {"https": proxy_url, "http": proxy_url}
        logger.debug("Proxy configured: %s", proxy_url.split("@")[-1])

    # Warm-up to acquire fresh routing cookies (lidc, bcookie, bscookie).
    # These have 24-72h TTL and cannot be reliably extracted from cookie files.
    #
    # Strategy: authenticated warm-up with increasing redirect tolerance.
    #   Phase 1: Auth warm-up with max_redirects=15 (covers proxy extra hops).
    #   Phase 2: Auth warm-up with allow_redirects=False — just capture
    #            Set-Cookie from the first response (LinkedIn always sets lidc).
    #   NO anonymous fallback — anon warm-up gets wrong-datacenter lidc,
    #   causing redirect loops on the authenticated API call.
    routing_names = {"lidc", "bcookie", "bscookie"}
    acquired = []

    # Phase 1: Authenticated warm-up (follow redirects to accumulate cookies)
    try:
        warmup_session = requests.Session()
        warmup_session.max_redirects = 15
        warmup_session.cookies.set("li_at", li_at, domain=".linkedin.com")
        warmup_session.cookies.set("JSESSIONID", f'"{jsessionid}"', domain=".linkedin.com")
        if proxy_url:
            warmup_session.proxies = {"https": proxy_url, "http": proxy_url}
        try:
            warmup_resp = warmup_session.get(
                "https://www.linkedin.com/feed/",
                timeout=15,
                headers={"User-Agent": ua},
                allow_redirects=True,
            )
            logger.info("Auth warm-up response: %d", warmup_resp.status_code)
        except requests.TooManyRedirects:
            logger.info("Auth warm-up hit redirect limit (15 hops)")
        # Extract routing cookies accumulated during redirect chain
        for cookie in warmup_session.cookies:
            if cookie.name in routing_names:
                session.cookies.set_cookie(cookie)
                acquired.append(cookie.name)
        if acquired:
            logger.info("Auth warm-up routing cookies: %s", acquired)
    except requests.RequestException as e:
        logger.warning("Auth warm-up failed: %s", e)

    # Phase 2: Single-hop auth warm-up (no redirects, just capture Set-Cookie)
    if not acquired:
        logger.info("Phase 1 got no routing cookies, trying no-redirect warm-up")
        try:
            warmup_resp = requests.get(
                "https://www.linkedin.com/feed/",
                timeout=15,
                allow_redirects=False,
                headers={"User-Agent": ua, "Cookie": f"li_at={li_at}; JSESSIONID=\"{jsessionid}\""},
                proxies=session.proxies if proxy_url else None,
            )
            logger.info(
                "No-redirect warm-up response: %d",
                warmup_resp.status_code,
            )
            for cookie in warmup_resp.cookies:
                if cookie.name in routing_names:
                    session.cookies.set(cookie.name, cookie.value, domain=".linkedin.com")
                    acquired.append(cookie.name)
            if acquired:
                logger.info("No-redirect warm-up routing cookies: %s", acquired)
        except requests.RequestException as e:
            logger.warning("No-redirect warm-up failed: %s", e)

    if not acquired:
        logger.warning("No routing cookies acquired — API calls may redirect-loop")

    # NOW set auth cookies (after routing cookies are in place)
    session.cookies.set("li_at", li_at, domain=".linkedin.com")
    session.cookies.set("JSESSIONID", f'"{jsessionid}"', domain=".linkedin.com")

    logger.info("Session cookies: %s", list(session.cookies.keys()))

    return session


# ---------------------------------------------------------------------------
# Voyager API calls
# ---------------------------------------------------------------------------

def fetch_feed(
    session: requests.Session,
    count: int = 20,
    start: int = 0,
) -> dict:
    """
    Fetch LinkedIn feed updates from the Voyager API.

    Args:
        session: Authenticated requests.Session.
        count: Number of feed items to request.
        start: Pagination offset.

    Returns:
        Raw JSON response dict.

    Raises:
        requests.HTTPError: On 4xx/5xx responses.
    """
    url = f"{VOYAGER_BASE}{FEED_ENDPOINT}"
    params = {
        "q": "feed",
        "count": count,
        "start": start,
    }

    logger.debug("GET %s?count=%d&start=%d", url, count, start)
    resp = session.get(url, params=params, timeout=30)

    if resp.status_code == 429:
        raise requests.HTTPError(
            f"429 Too Many Requests — LinkedIn rate limit hit",
            response=resp,
        )
    if resp.status_code in (401, 403):
        raise requests.HTTPError(
            f"{resp.status_code} Auth failure — li_at token may be expired",
            response=resp,
        )

    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Entity resolution from included[] array
# ---------------------------------------------------------------------------

def build_entity_map(included: list[dict]) -> dict:
    """
    Build a URN → entity lookup dict from the Voyager included[] array.

    Entities are typed by $type and linked by entityUrn or their
    special reference keys.

    Args:
        included: The "included" array from Voyager response.

    Returns:
        Dict mapping entityUrn → entity dict.
    """
    entity_map = {}
    for entity in included:
        urn = entity.get("entityUrn") or entity.get("$id")
        if urn:
            entity_map[urn] = entity
    return entity_map


def resolve_author(update: dict, entity_map: dict) -> tuple[str, str, int | None]:
    """
    Resolve the author of a feed update.

    The Voyager feed render includes an inline `actor` component with
    name text and URN. Falls back to entity_map MiniProfile/MiniCompany.

    Returns:
        (author_handle, author_id, follower_count) tuple.
    """
    author_name = ""
    author_urn = ""
    follower_count = None

    # Primary path: inline actor component (always present in UpdateV2)
    actor_obj = update.get("actor", {})
    if isinstance(actor_obj, dict):
        author_urn = actor_obj.get("urn", "")

        # Name from actor.name.text
        name_obj = actor_obj.get("name", {})
        if isinstance(name_obj, dict):
            author_name = name_obj.get("text", "")

        # Follower count from actor.description.text (e.g. "14,581,356 followers")
        desc_obj = actor_obj.get("description", {})
        if isinstance(desc_obj, dict):
            desc_text = desc_obj.get("text", "")
            follower_match = re.search(r"([\d,.']+)\s*(?:followers|Follower)", desc_text, re.IGNORECASE)
            if follower_match:
                try:
                    follower_count = int(follower_match.group(1).replace(",", "").replace(".", "").replace("'", ""))
                except ValueError:
                    pass

    # Fallback: *actor reference → entity_map
    if not author_name:
        actor_ref = update.get("*actor", "")
        if actor_ref and actor_ref in entity_map:
            actor = entity_map[actor_ref]
            etype = actor.get("$type", "")
            if "MiniProfile" in etype or "firstName" in actor:
                author_name = f"{actor.get('firstName', '')} {actor.get('lastName', '')}".strip()
            elif "MiniCompany" in etype or "name" in actor:
                author_name = actor.get("name", "")
            author_urn = author_urn or actor_ref

    return author_name, author_urn, follower_count


def resolve_engagement(update: dict, entity_map: dict) -> dict:
    """
    Resolve engagement counters from a SocialDetail entity.

    Voyager stores counters in the SocialDetail entity:
    - likes.paging.total → like count
    - comments.paging.total → comment count
    - totalShares → share count
    Falls back to *totalSocialActivityCounts reference if available.

    Returns:
        Dict with numLikes, numComments, numShares (all int or None).
    """
    result = {"numLikes": None, "numComments": None, "numShares": None}

    # Resolve *socialDetail reference
    social_urn = update.get("*socialDetail", "")
    social = entity_map.get(social_urn, {}) if social_urn else {}

    if not social:
        return result

    # Primary path: paging totals (Voyager render format)
    likes_obj = social.get("likes", {})
    if isinstance(likes_obj, dict):
        paging = likes_obj.get("paging", {})
        if isinstance(paging, dict) and "total" in paging:
            result["numLikes"] = paging["total"]

    comments_obj = social.get("comments", {})
    if isinstance(comments_obj, dict):
        paging = comments_obj.get("paging", {})
        if isinstance(paging, dict) and "total" in paging:
            result["numComments"] = paging["total"]

    if "totalShares" in social:
        result["numShares"] = social["totalShares"]

    # Fallback: *totalSocialActivityCounts reference
    if result["numLikes"] is None:
        counts_urn = social.get("*totalSocialActivityCounts", "")
        counts = entity_map.get(counts_urn, {}) if counts_urn else {}
        if counts:
            result["numLikes"] = counts.get("numLikes", result["numLikes"])
            result["numComments"] = counts.get("numComments", result["numComments"])
            result["numShares"] = counts.get("numShares", result["numShares"])

    return result


# ---------------------------------------------------------------------------
# Timestamp decoding
# ---------------------------------------------------------------------------

def decode_activity_timestamp(activity_urn: str) -> str | None:
    """
    Decode the creation timestamp from a LinkedIn activity URN.

    LinkedIn activity URNs contain a snowflake ID where bits 22+
    encode the creation timestamp in milliseconds since epoch.

    Args:
        activity_urn: e.g. "urn:li:activity:7308123456789012480"

    Returns:
        ISO 8601 UTC timestamp string, or None if decoding fails.
    """
    match = re.search(r"urn:li:activity:(\d+)", activity_urn)
    if not match:
        return None

    try:
        numeric_id = int(match.group(1))
        timestamp_ms = numeric_id >> 22
        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        return dt.isoformat()
    except (ValueError, OSError, OverflowError):
        return None


# ---------------------------------------------------------------------------
# Feed parsing → CapturedPost objects
# ---------------------------------------------------------------------------

def extract_caption(update: dict, entity_map: dict) -> str:
    """Extract caption text from a feed update."""
    # Direct commentary path
    commentary = update.get("commentary", {}) or {}
    text_obj = commentary.get("text", {}) or {}
    if isinstance(text_obj, dict):
        text = text_obj.get("text", "")
    else:
        text = str(text_obj)

    if text:
        return text

    # Fallback: check for text in content
    content = update.get("content", {}) or {}
    if isinstance(content, dict):
        # Article content may have commentary
        article = content.get("article", {}) or {}
        desc = article.get("description", {}) or {}
        if isinstance(desc, dict):
            return desc.get("text", "")
        if isinstance(desc, str):
            return desc

    return ""


def infer_media_type(update: dict, entity_map: dict) -> str:
    """Infer media type from update content entities."""
    content = update.get("content", {}) or {}
    if not isinstance(content, dict):
        return "text"

    # Check for known content type keys
    if "video" in content or "*video" in content:
        return "video"
    if "images" in content or "*images" in content:
        images = content.get("images", content.get("*images", []))
        if isinstance(images, list) and len(images) > 1:
            return "carousel"
        return "image"
    if "image" in content or "*image" in content:
        return "image"
    if "article" in content or "*article" in content:
        return "article"
    if "document" in content or "*document" in content:
        return "document"
    if "poll" in content or "*poll" in content:
        return "poll"

    # Check $type in content for more specific matching
    ctype = content.get("$type", "")
    if "Video" in ctype:
        return "video"
    if "Image" in ctype:
        return "image"
    if "Article" in ctype:
        return "article"

    return "text"


def extract_thumbnail_url(update: dict, entity_map: dict) -> str:
    """
    Extract a thumbnail/cover image URL from a feed update's content.

    Voyager API nests image URLs in various content structures depending
    on media type. This function checks all known paths and returns the
    first URL found.

    Args:
        update: A single feed update dict.
        entity_map: URN → entity lookup from build_entity_map().

    Returns:
        Image URL string, or "" if none found.
    """
    content = update.get("content", {}) or {}
    if not isinstance(content, dict):
        return ""

    # Helper: dig into a Voyager image object for a URL
    def _resolve_image(img_obj: object) -> str:
        """Extract URL from a Voyager image entity (various nesting levels)."""
        if not isinstance(img_obj, dict):
            # Could be a URN reference string
            if isinstance(img_obj, str) and img_obj in entity_map:
                img_obj = entity_map[img_obj]
            else:
                return ""

        # Direct URL field
        for key in ("url", "externalUrl", "fileIdentifyingUrlPathSegment"):
            if key in img_obj and isinstance(img_obj[key], str):
                url = img_obj[key]
                if url.startswith("http"):
                    return url

        # vectorImage.rootUrl + artifacts (common Voyager pattern)
        vi = img_obj.get("vectorImage", {})
        if isinstance(vi, dict):
            root = vi.get("rootUrl", "")
            artifacts = vi.get("artifacts", [])
            if root and isinstance(artifacts, list) and artifacts:
                # Pick the largest artifact (last in list)
                suffix = artifacts[-1].get("fileIdentifyingUrlPathSegment", "")
                if root and suffix:
                    return f"{root}{suffix}"
                if root:
                    return root

        # Nested in attributes[0].vectorImage or attributes[0].miniImage
        attrs = img_obj.get("attributes", [])
        if isinstance(attrs, list):
            for attr in attrs:
                if not isinstance(attr, dict):
                    continue
                for img_key in ("vectorImage", "miniImage", "imageViewModel"):
                    nested = attr.get(img_key, {})
                    result = _resolve_image(nested) if isinstance(nested, dict) else ""
                    if result:
                        return result

        # data[0].url (REST-style image arrays)
        data = img_obj.get("data", [])
        if isinstance(data, list):
            for d in data:
                if isinstance(d, dict) and "url" in d:
                    return d["url"]

        return ""

    # --- Try each content type ---

    # 1. images[] (image/carousel posts)
    images = content.get("images", content.get("*images", []))
    if isinstance(images, list):
        for img in images:
            url = _resolve_image(img)
            if url:
                return url

    # 2. image (single image)
    image = content.get("image", content.get("*image"))
    if image:
        url = _resolve_image(image)
        if url:
            return url

    # 3. video → posterImage / thumbnail
    video = content.get("video", content.get("*video"))
    if isinstance(video, dict):
        for poster_key in ("posterImage", "thumbnail", "thumbnailImage"):
            poster = video.get(poster_key)
            if poster:
                url = _resolve_image(poster)
                if url:
                    return url
    elif isinstance(video, str) and video in entity_map:
        video_entity = entity_map[video]
        for poster_key in ("posterImage", "thumbnail", "thumbnailImage"):
            poster = video_entity.get(poster_key)
            if poster:
                url = _resolve_image(poster)
                if url:
                    return url

    # 4. article → largeImage / smallImage
    article = content.get("article", content.get("*article"))
    if isinstance(article, dict):
        for img_key in ("largeImage", "smallImage", "heroImage", "image"):
            art_img = article.get(img_key)
            if art_img:
                url = _resolve_image(art_img)
                if url:
                    return url
    elif isinstance(article, str) and article in entity_map:
        article_entity = entity_map[article]
        for img_key in ("largeImage", "smallImage", "heroImage", "image"):
            art_img = article_entity.get(img_key)
            if art_img:
                url = _resolve_image(art_img)
                if url:
                    return url

    # 5. document → coverImage
    document = content.get("document", content.get("*document"))
    if isinstance(document, dict):
        cover = document.get("coverImage", document.get("coverPages", [{}]))
        if isinstance(cover, list) and cover:
            cover = cover[0]
        url = _resolve_image(cover)
        if url:
            return url
    elif isinstance(document, str) and document in entity_map:
        doc_entity = entity_map[document]
        cover = doc_entity.get("coverImage")
        if cover:
            url = _resolve_image(cover)
            if url:
                return url

    return ""


def parse_feed_to_posts(
    raw: dict,
    is_top: bool,
    rank_offset: int = 0,
) -> list[CapturedPost]:
    """
    Parse a Voyager feed response into CapturedPost objects.

    Args:
        raw: Raw JSON response from fetch_feed().
        is_top: Whether these are top posts (rank <= 20).
        rank_offset: Starting rank offset (0 for top, 20 for baseline).

    Returns:
        List of CapturedPost objects with resolved references.
    """
    included = raw.get("included", [])
    entity_map = build_entity_map(included)

    # Elements are referenced by URN in data["*elements"]
    element_urns = raw.get("data", {}).get("*elements", [])
    if not element_urns:
        # Fallback: try data.elements directly
        element_urns = raw.get("data", {}).get("elements", [])

    posts = []
    rank = rank_offset + 1

    for urn in element_urns:
        # Resolve the update element from entity_map
        if isinstance(urn, str):
            update = entity_map.get(urn, {})
        elif isinstance(urn, dict):
            update = urn
        else:
            continue

        if not update:
            continue

        # Skip non-post updates (ads, system notifications, etc.)
        update_type = update.get("$type", "")
        if "AdUpdate" in update_type or "SystemUpdate" in update_type:
            logger.debug("Skipping non-post update: %s", update_type)
            continue

        # Extract activity URN — prefer updateMetadata.urn (clean activity URN),
        # then fall back to extracting from the composite fs_updateV2 URN
        activity_urn = ""
        meta = update.get("updateMetadata", {})
        if isinstance(meta, dict):
            activity_urn = meta.get("urn", "")

        # Fallback: extract from the element URN itself
        # e.g. "urn:li:fs_updateV2:(urn:li:activity:XXXX,...)"
        if not activity_urn and isinstance(urn, str):
            match = re.search(r"(urn:li:activity:\d+)", urn)
            if match:
                activity_urn = match.group(1)

        # Extract numeric ID from activity URN
        activity_match = re.search(r"urn:li:activity:(\d+)", activity_urn)
        if not activity_match:
            # Try ugcPost URN as fallback
            ugc_match = re.search(r"urn:li:ugcPost:(\d+)", str(urn) + str(activity_urn))
            if ugc_match:
                activity_match = ugc_match
                activity_urn = f"urn:li:ugcPost:{ugc_match.group(1)}"

        if not activity_match:
            logger.debug("Skipping element without activity/ugcPost URN: %s", str(urn)[:100])
            continue

        numeric_id = activity_match.group(1)

        # Resolve author (inline actor component)
        author_handle, author_id, follower_count = resolve_author(update, entity_map)

        # Resolve engagement (SocialDetail entity)
        engagement = resolve_engagement(update, entity_map)

        # Extract caption
        caption = extract_caption(update, entity_map)

        # Extract hashtags from caption
        hashtags = re.findall(r"#(\w+)", caption) if caption else []

        # Decode timestamp from URN snowflake
        posted_at = decode_activity_timestamp(activity_urn)
        # Fallback to createdAt field
        if not posted_at:
            created_at = update.get("createdAt")
            if created_at and isinstance(created_at, (int, float)):
                try:
                    dt = datetime.fromtimestamp(created_at / 1000, tz=timezone.utc)
                    posted_at = dt.isoformat()
                except (ValueError, OSError):
                    pass

        # Build permalink from clean activity URN
        permalink = f"https://www.linkedin.com/feed/update/{activity_urn}/"

        # Infer media type
        media_type = infer_media_type(update, entity_map)

        # Extract thumbnail URL from content
        thumbnail_url = extract_thumbnail_url(update, entity_map)

        post = CapturedPost(
            post_id=numeric_id,
            platform="linkedin",
            permalink=permalink,
            rank_observed=rank,
            is_top=is_top,
            media_type=media_type,
            caption=caption,
            author_id=author_id,
            author_handle=author_handle,
            likes=engagement["numLikes"],
            comments=engagement["numComments"],
            shares=engagement["numShares"],
            followers=follower_count,
            posted_at_utc=posted_at or "",
            hashtags=hashtags,
            thumbnail_url=thumbnail_url,
            source="api",
        )

        posts.append(post)
        rank += 1

    return posts


# ---------------------------------------------------------------------------
# Main scrape function
# ---------------------------------------------------------------------------

def run_scrape(account_key: str, dry_run: bool = False) -> dict:
    """
    Execute a single LinkedIn API scrape for the given account.

    Args:
        account_key: Key from settings.ACCOUNTS (e.g. "linkedin_seeded").
        dry_run: If True, skip database write.

    Returns:
        Dictionary summary of the scrape result.
    """
    settings = load_settings()

    # Validate account key
    if account_key not in settings.ACCOUNTS:
        logger.error(
            "Unknown account key: %s. Valid keys: %s",
            account_key, list(settings.ACCOUNTS.keys()),
        )
        return {"success": False, "error": f"Unknown account: {account_key}"}

    account = settings.ACCOUNTS[account_key]
    if account["platform"] != "linkedin":
        logger.error("Account %s is not a LinkedIn account", account_key)
        return {"success": False, "error": "Not a LinkedIn account"}

    log_scrape_start(
        platform="linkedin",
        account_type=account["account_type"],
        account_key=account_key,
    )

    start_time = time.time()
    snapshot_id = f"linkedin_{account['account_type']}_{uuid.uuid4().hex[:8]}"

    result = SnapshotResult(
        snapshot_id=snapshot_id,
        platform="linkedin",
        account_type=account["account_type"],
        surface="top_feed",
        captured_at_utc=datetime.now(timezone.utc).isoformat(),
        timezone="Europe/Zurich",
    )

    try:
        # Step 1: Load cookies
        cookie_path = Path(account["cookie_file"])
        logger.info("Loading cookies from: %s", cookie_path)
        cookies = load_cookies_from_file(cookie_path, platform="linkedin")

        warnings = check_cookie_expiry(cookies, warn_days=3)
        for w in warnings:
            logger.warning("Cookie health: %s", w)

        # Step 2: Build authenticated session
        proxy_url = settings.PROXY_URL if hasattr(settings, "PROXY_URL") else ""
        session = build_session(cookies, proxy_url=proxy_url)
        logger.info("API session built (proxy: %s)", "yes" if proxy_url else "no")

        # Step 3: Fetch Top 20
        logger.info("Fetching Top 20 posts (start=0, count=20)...")
        top_raw = fetch_feed(session, count=20, start=0)
        result.top_posts = parse_feed_to_posts(top_raw, is_top=True, rank_offset=0)
        logger.info("Parsed %d top posts", len(result.top_posts))

        # Small delay between API calls to be gentle
        time.sleep(2)

        # Step 4: Fetch Baseline 50
        logger.info("Fetching Baseline 50 posts (start=20, count=50)...")
        baseline_raw = fetch_feed(session, count=50, start=20)
        result.baseline_posts = parse_feed_to_posts(
            baseline_raw, is_top=False, rank_offset=20,
        )
        logger.info("Parsed %d baseline posts", len(result.baseline_posts))

        # Warn if very few posts returned
        total = len(result.top_posts) + len(result.baseline_posts)
        if total < 5:
            logger.warning(
                "Very few posts returned (%d) — feed may be empty or auth failing",
                total,
            )

        result.success = True

    except requests.HTTPError as e:
        result.error_message = str(e)
        logger.error("API error: %s", e)

        # Send targeted alerts
        status_code = e.response.status_code if e.response is not None else 0
        if status_code == 429:
            send_alert(
                f"LinkedIn API 429 — rate limit hit for {account_key}.\n"
                f"Token may be burned. Consider rotating."
            )
        elif status_code in (401, 403):
            send_alert(
                f"LinkedIn API {status_code} — auth failure for {account_key}.\n"
                f"li_at token may be expired. Re-export cookies."
            )

    except (requests.ConnectionError, requests.TooManyRedirects) as e:
        # One retry after 5s on network/redirect error
        logger.warning("Network error, retrying in 5s: %s", e)
        time.sleep(5)
        try:
            top_raw = fetch_feed(session, count=20, start=0)
            result.top_posts = parse_feed_to_posts(top_raw, is_top=True, rank_offset=0)
            time.sleep(2)
            baseline_raw = fetch_feed(session, count=50, start=20)
            result.baseline_posts = parse_feed_to_posts(
                baseline_raw, is_top=False, rank_offset=20,
            )
            result.success = True
            logger.info("Retry succeeded: %d top, %d baseline",
                        len(result.top_posts), len(result.baseline_posts))
        except Exception as retry_err:
            result.error_message = f"Network retry failed: {retry_err}"
            logger.error(result.error_message)

    except Exception as e:
        result.error_message = f"Unexpected error: {e}"
        logger.exception(result.error_message)

    finally:
        result.duration_seconds = time.time() - start_time

    # Step 5: Write to SQLite
    if dry_run:
        logger.info(
            "DRY RUN: skipping database write. "
            "Top=%d, Baseline=%d, Success=%s",
            len(result.top_posts),
            len(result.baseline_posts),
            result.success,
        )
        db_success = True
    else:
        if result.success:
            db_success = write_snapshot(result)
            if not db_success:
                logger.warning("SQLite write failed, data saved to temp JSON")
        else:
            db_success = True  # No data to write

    # Step 6: Build summary
    summary = {
        "success": result.success,
        "snapshot_id": result.snapshot_id,
        "platform": result.platform,
        "account_type": result.account_type,
        "captured_at": result.captured_at_utc,
        "top_posts_count": len(result.top_posts),
        "baseline_posts_count": len(result.baseline_posts),
        "duration_seconds": round(result.duration_seconds, 1),
        "db_write": db_success,
        "error": result.error_message,
    }

    # Step 7: Log snapshot summary
    log_snapshot_summary(
        snapshot_id=result.snapshot_id,
        platform=result.platform,
        account_type=result.account_type,
        success=result.success,
        top_count=len(result.top_posts),
        baseline_count=len(result.baseline_posts),
        duration=result.duration_seconds,
        error=result.error_message,
    )

    if result.success:
        logger.info("Sample Top posts:")
        for post in result.top_posts[:5]:
            logger.info(
                "  Rank %d: %s (ID: %s, likes: %s, author: %s)",
                post.rank_observed,
                post.permalink[:60] if post.permalink else "no-link",
                post.post_id,
                post.likes,
                post.author_handle[:30] if post.author_handle else "?",
            )

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run a single LinkedIn API scrape (no browser needed)."
    )
    parser.add_argument(
        "--account",
        required=True,
        help="Account key from settings.ACCOUNTS (e.g. linkedin_seeded)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run scraper but skip database write",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    setup_logging(platform="linkedin", level=args.log_level)

    result = run_scrape(
        account_key=args.account,
        dry_run=args.dry_run,
    )

    # Print JSON summary
    print("\n" + "=" * 60)
    print("RESULT SUMMARY:")
    print(json.dumps(result, indent=2))
    print("=" * 60)

    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
