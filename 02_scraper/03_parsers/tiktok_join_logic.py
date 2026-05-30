"""
tiktok_join_logic.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    Join logic for the TikTok dual-path extraction. Matches DOM
    records (TikTokTileData) with JSON records (TikTokJSONPost)
    using post IDs as the canonical join key, then produces enriched
    CapturedPost objects. The DOM supplies rank order and visual
    position; the JSON supplies timestamps, follower counts,
    engagement counters, audio metadata, and video duration. When
    both sources have the same field the JSON value is treated as
    authoritative, with discrepancies logged for review.

Inputs:
    (none — library module)

Outputs:
    (none — library module; returns (CapturedPost list, JoinReport)
    to the TikTok scraper. Imported by the tiktok module.)

Usage:
    from tiktok_join_logic import join_dom_and_json
    posts, report = join_dom_and_json(dom_tiles, json_posts)
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiktok_dom_parser import TikTokTileData
from tiktok_json_parser import TikTokJSONPost
from join_logic import JoinReport

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02_platforms"))
from base import CapturedPost

logger = logging.getLogger(__name__)


def join_dom_and_json(
    dom_tiles: list[TikTokTileData],
    json_posts: list[TikTokJSONPost],
    log_discrepancies: bool = True,
) -> tuple[list[CapturedPost], JoinReport]:
    """
    Merge DOM-extracted tiles with JSON-intercepted posts.

    Join strategy:
        1. Index JSON posts by post_id for O(1) lookup
        2. For each DOM tile, look up the matching JSON post
        3. Merge fields, preferring JSON for metadata, DOM for rank
        4. DOM tiles without JSON match still produce a CapturedPost
           (with source="dom") -- partial data is better than no data
        5. JSON posts without DOM match are returned separately
           (useful for debugging, not typically written to DB)

    Args:
        dom_tiles: Tiles from the DOM parser.
        json_posts: Posts from the JSON interceptor.
        log_discrepancies: Log field-level differences between sources.

    Returns:
        Tuple of (merged CapturedPost list, JoinReport).
    """
    report = JoinReport(
        total_dom=len(dom_tiles),
        total_json=len(json_posts),
    )

    # Index JSON posts by post_id
    json_by_id: dict[str, TikTokJSONPost] = {}
    for jp in json_posts:
        if jp.post_id:
            json_by_id[jp.post_id] = jp

    # Also index by permalink for fallback matching
    json_by_permalink: dict[str, TikTokJSONPost] = {}
    for jp in json_posts:
        if jp.permalink:
            json_by_permalink[jp.permalink] = jp

    merged_posts: list[CapturedPost] = []
    matched_json_ids: set[str] = set()

    for tile in dom_tiles:
        # Try to find matching JSON post
        json_post = _find_json_match(tile, json_by_id, json_by_permalink)

        if json_post:
            # Full match: merge both sources
            post = _merge_tile_and_json(tile, json_post)
            post.source = "joined"
            matched_json_ids.add(json_post.post_id)
            report.matched += 1

            # Check for field discrepancies
            if log_discrepancies:
                discrepancies = _check_discrepancies(tile, json_post)
                if discrepancies:
                    report.field_discrepancies += 1
                    logger.debug(
                        "Post %s: field discrepancies: %s",
                        tile.post_id,
                        discrepancies,
                    )
        else:
            # DOM only: create CapturedPost from tile data alone
            post = _post_from_dom_only(tile)
            report.dom_only += 1

        merged_posts.append(post)

    # Count JSON-only posts (not matched to any DOM tile)
    report.json_only = len(json_by_id) - len(matched_json_ids)

    # Log summary
    logger.info(
        "Join result: %d DOM + %d JSON -> %d matched, %d DOM-only, "
        "%d JSON-only, %d with discrepancies",
        report.total_dom,
        report.total_json,
        report.matched,
        report.dom_only,
        report.json_only,
        report.field_discrepancies,
    )

    if report.total_dom > 0:
        match_rate = report.matched / report.total_dom * 100
        logger.info("Join match rate: %.1f%%", match_rate)

        if match_rate < 50:
            logger.warning(
                "Low join match rate (%.1f%%). JSON interception may "
                "be failing or API endpoints may have changed.",
                match_rate,
            )

    return merged_posts, report


def _find_json_match(
    tile: TikTokTileData,
    json_by_id: dict[str, TikTokJSONPost],
    json_by_permalink: dict[str, TikTokJSONPost],
) -> TikTokJSONPost | None:
    """
    Find the JSON post matching a DOM tile.

    Tries post_id first (canonical), then falls back to permalink.

    Args:
        tile: A DOM-extracted tile.
        json_by_id: JSON posts indexed by post_id.
        json_by_permalink: JSON posts indexed by permalink.

    Returns:
        Matching TikTokJSONPost, or None if no match found.
    """
    # Primary: match by post_id
    if tile.post_id and tile.post_id in json_by_id:
        return json_by_id[tile.post_id]

    # Fallback: match by permalink
    if tile.permalink and tile.permalink in json_by_permalink:
        return json_by_permalink[tile.permalink]

    # Normalised permalink fallback (strip query params, trailing slash)
    if tile.permalink:
        normalised = tile.permalink.split("?")[0].rstrip("/")
        for key, jp in json_by_permalink.items():
            if key.split("?")[0].rstrip("/") == normalised:
                return jp

    logger.debug(
        "No JSON match for DOM tile: post_id=%s, permalink=%s",
        tile.post_id,
        tile.permalink[:60] if tile.permalink else "",
    )
    return None


def _merge_tile_and_json(
    tile: TikTokTileData,
    json_post: TikTokJSONPost,
) -> CapturedPost:
    """
    Create a CapturedPost by merging DOM tile data with JSON post data.

    Strategy:
        - Rank and position: always from DOM (that is the visual truth)
        - Identification: prefer JSON (more complete)
        - Engagement counters: prefer JSON (exact values vs DOM's "349K")
        - Temporal data: only from JSON
        - Audio metadata: prefer JSON
        - Caption: prefer JSON (full text vs DOM's truncated alt text)

    Args:
        tile: DOM-extracted tile (rank, position, on-screen data).
        json_post: JSON-intercepted post (metadata, counters).

    Returns:
        Merged CapturedPost.
    """
    raw_id = json_post.post_id or tile.post_id
    return CapturedPost(
        # Identification (JSON preferred, DOM fallback)
        # Prefix with platform to avoid cross-platform PK collision in posts table
        post_id=f"tiktok_{raw_id}" if raw_id else "",
        permalink=json_post.permalink or tile.permalink,

        # Rank (always from DOM -- this is the visual position)
        rank_observed=tile.visual_rank,
        is_top=True,  # Set by caller based on context

        # Media type (JSON has this, DOM assumes "video")
        media_type=json_post.media_type or "video",

        # Caption (JSON has full text, DOM has truncated alt text)
        caption=json_post.caption or tile.caption,

        # Author (JSON has ID, DOM has handle)
        author_id=json_post.author_id,
        author_handle=json_post.author_handle or tile.author_handle,

        # Engagement (JSON has exact counts, DOM has rounded "349K")
        likes=json_post.likes if json_post.likes is not None else tile.like_count,
        comments=json_post.comments,
        shares=json_post.shares,
        views=json_post.views,
        followers=json_post.followers,

        # Temporal (only from JSON)
        posted_at_utc=json_post.posted_at_utc,

        # Hashtags (JSON has structured data)
        hashtags=json_post.hashtags if json_post.hashtags else [],

        # Audio (JSON has structured metadata)
        audio_present=bool(json_post.audio_id or tile.audio_name),
        audio_id=json_post.audio_id,
        audio_name=json_post.audio_name or tile.audio_name,
        audio_is_original=json_post.audio_is_original,

        # Thumbnail (prefer JSON cover, fall back to DOM thumbnail)
        thumbnail_url=json_post.cover_url or tile.thumbnail_url,

        # Source flag
        source="joined",

        # Raw data combines both sources for debugging
        raw_data={
            "dom": {
                "grid_index": tile.grid_index,
                "visual_rank": tile.visual_rank,
                "rank_match": tile.rank_match,
                "bbox": {
                    "x": tile.bbox_x,
                    "y": tile.bbox_y,
                    "w": tile.bbox_width,
                    "h": tile.bbox_height,
                },
                "dom_likes": tile.like_count,
                "dom_caption_len": len(tile.caption),
            },
            "json": {
                "views": json_post.views,
                "saves": json_post.saves,
                "duration_seconds": json_post.duration_seconds,
                "audio_is_original": json_post.audio_is_original,
                "following": json_post.following,
                "posted_at_unix": json_post.posted_at_unix,
            },
        },
    )


def _post_from_dom_only(tile: TikTokTileData) -> CapturedPost:
    """
    Create a CapturedPost from DOM data only (no JSON match).

    Handles the graceful partial-match case so partial data is
    preserved rather than discarded.

    Args:
        tile: DOM-extracted tile.

    Returns:
        CapturedPost with DOM fields only.
    """
    return CapturedPost(
        post_id=f"tiktok_{tile.post_id}" if tile.post_id else "",
        permalink=tile.permalink,
        rank_observed=tile.visual_rank,
        is_top=True,
        media_type="video",
        caption=tile.caption,
        author_handle=tile.author_handle,
        likes=tile.like_count,
        audio_present=bool(tile.audio_name),
        audio_name=tile.audio_name,
        thumbnail_url=tile.thumbnail_url,
        source="dom",
        raw_data={
            "grid_index": tile.grid_index,
            "visual_rank": tile.visual_rank,
            "rank_match": tile.rank_match,
            "dom_only_reason": "no_json_match",
        },
    )


def _check_discrepancies(
    tile: TikTokTileData,
    json_post: TikTokJSONPost,
) -> list[str]:
    """
    Compare overlapping fields between DOM and JSON data.

    Useful for detecting when one extraction path is stale or broken.

    Args:
        tile: DOM-extracted tile.
        json_post: JSON-intercepted post.

    Returns:
        List of discrepancy descriptions (empty if no issues).
    """
    issues = []

    # Post ID should always match (that is how we joined)
    if tile.post_id and json_post.post_id and tile.post_id != json_post.post_id:
        issues.append(
            f"post_id mismatch: DOM={tile.post_id}, JSON={json_post.post_id}"
        )

    # Username comparison (case-insensitive)
    if (
        tile.author_handle
        and json_post.author_handle
        and tile.author_handle.lower() != json_post.author_handle.lower()
    ):
        issues.append(
            f"author mismatch: DOM={tile.author_handle}, "
            f"JSON={json_post.author_handle}"
        )

    # Like count comparison (DOM is rounded, so allow tolerance)
    if tile.like_count is not None and json_post.likes is not None:
        dom_likes = tile.like_count
        json_likes = json_post.likes
        # Allow 10% tolerance for rounding (e.g. "349K" vs 349,123)
        if dom_likes > 0 and abs(dom_likes - json_likes) / dom_likes > 0.10:
            issues.append(
                f"likes discrepancy: DOM={dom_likes}, JSON={json_likes}"
            )

    return issues
