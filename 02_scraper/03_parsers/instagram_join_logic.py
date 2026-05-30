"""
instagram_join_logic.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    Join logic for the Instagram dual-path extraction. Matches DOM
    records (InstagramTileData) with JSON records (InstagramJSONPost)
    using shortcode as the canonical join key, then produces enriched
    CapturedPost objects. The DOM is the authoritative source for
    rank order; the JSON supplies almost all engagement and author
    metadata because the Explore grid DOM exposes little of it.

Inputs:
    (none — library module)

Outputs:
    (none — library module; returns (CapturedPost list, JoinReport)
    to the Instagram scraper. Imported by the instagram module.)

Usage:
    from instagram_join_logic import join_dom_and_json
    posts, report = join_dom_and_json(dom_tiles, json_posts)
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from instagram_dom_parser import InstagramTileData
from instagram_json_parser import InstagramJSONPost
from join_logic import JoinReport

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02_platforms"))
from base import CapturedPost

logger = logging.getLogger(__name__)


def join_dom_and_json(
    dom_tiles: list[InstagramTileData],
    json_posts: list[InstagramJSONPost],
    log_discrepancies: bool = True,
) -> tuple[list[CapturedPost], JoinReport]:
    """
    Merge DOM-extracted tiles with JSON-intercepted posts for Instagram.

    Join strategy (mirrors TikTok):
        1. Index JSON posts by shortcode for O(1) lookup
        2. For each DOM tile, look up the matching JSON post
        3. Merge fields, preferring JSON for metadata, DOM for rank
        4. DOM tiles without JSON match produce CapturedPost with source="dom"

    The key difference from TikTok: Instagram's join key is the shortcode
    (alphanumeric string) rather than a numeric post ID. The shortcode is
    visible in the DOM href and in the JSON "code"/"shortcode" field.

    Args:
        dom_tiles: Tiles from the Instagram DOM parser.
        json_posts: Posts from the Instagram JSON interceptor.
        log_discrepancies: Log field-level differences between sources.

    Returns:
        Tuple of (merged CapturedPost list, JoinReport).
    """
    report = JoinReport(
        total_dom=len(dom_tiles),
        total_json=len(json_posts),
    )

    # Index JSON posts by shortcode
    json_by_shortcode: dict[str, InstagramJSONPost] = {}
    for jp in json_posts:
        if jp.shortcode:
            json_by_shortcode[jp.shortcode] = jp

    # Also index by post_id for fallback
    json_by_id: dict[str, InstagramJSONPost] = {}
    for jp in json_posts:
        if jp.post_id:
            json_by_id[jp.post_id] = jp

    merged_posts: list[CapturedPost] = []
    matched_shortcodes: set[str] = set()

    for tile in dom_tiles:
        json_post = _find_json_match(tile, json_by_shortcode, json_by_id)

        if json_post:
            post = _merge_tile_and_json(tile, json_post)
            post.source = "joined"
            matched_shortcodes.add(json_post.shortcode)
            report.matched += 1

            if log_discrepancies:
                discrepancies = _check_discrepancies(tile, json_post)
                if discrepancies:
                    report.field_discrepancies += 1
                    logger.debug(
                        "Post %s: field discrepancies: %s",
                        tile.shortcode,
                        discrepancies,
                    )
        else:
            post = _post_from_dom_only(tile)
            report.dom_only += 1

        merged_posts.append(post)

    report.json_only = len(json_by_shortcode) - len(matched_shortcodes)

    logger.info(
        "Instagram join result: %d DOM + %d JSON -> %d matched, "
        "%d DOM-only, %d JSON-only, %d with discrepancies",
        report.total_dom,
        report.total_json,
        report.matched,
        report.dom_only,
        report.json_only,
        report.field_discrepancies,
    )

    if report.total_dom > 0:
        match_rate = report.matched / report.total_dom * 100
        logger.info("Instagram join match rate: %.1f%%", match_rate)

        if match_rate < 50:
            logger.warning(
                "Low join match rate (%.1f%%). JSON interception may "
                "be failing or API endpoints may have changed.",
                match_rate,
            )

    return merged_posts, report


def _find_json_match(
    tile: InstagramTileData,
    json_by_shortcode: dict[str, InstagramJSONPost],
    json_by_id: dict[str, InstagramJSONPost],
) -> InstagramJSONPost | None:
    """
    Find the JSON post matching a DOM tile.

    Primary key: shortcode (always available from the href).
    Fallback: post_id (if the DOM somehow has it).

    Args:
        tile: A DOM-extracted tile.
        json_by_shortcode: JSON posts indexed by shortcode.
        json_by_id: JSON posts indexed by post_id.

    Returns:
        Matching InstagramJSONPost, or None if no match found.
    """
    # Primary: match by shortcode
    if tile.shortcode and tile.shortcode in json_by_shortcode:
        return json_by_shortcode[tile.shortcode]

    # Fallback: match by post_id (unlikely to be in DOM, but just in case)
    if tile.post_id and tile.post_id in json_by_id:
        return json_by_id[tile.post_id]

    logger.debug(
        "No JSON match for Instagram tile: shortcode=%s, permalink=%s",
        tile.shortcode,
        tile.permalink[:60] if tile.permalink else "",
    )
    return None


def _merge_tile_and_json(
    tile: InstagramTileData,
    json_post: InstagramJSONPost,
) -> CapturedPost:
    """
    Create a CapturedPost by merging DOM tile data with JSON post data.

    Strategy (consistent with TikTok):
        - Rank and position: always from DOM
        - Identification: prefer JSON (has numeric ID)
        - Engagement counters: only from JSON (not in DOM)
        - Temporal data: only from JSON
        - Caption: prefer JSON (full text vs DOM alt which may be truncated)
        - Author: only from JSON (not visible in DOM grid)

    Args:
        tile: DOM-extracted tile.
        json_post: JSON-intercepted post.

    Returns:
        Merged CapturedPost.
    """
    raw_id = tile.shortcode or json_post.shortcode or json_post.post_id
    return CapturedPost(
        # Identification
        # Prefix with platform to avoid cross-platform PK collision in posts table
        post_id=f"instagram_{raw_id}" if raw_id else "",
        permalink=json_post.permalink or tile.permalink,

        # Rank (always from DOM)
        rank_observed=tile.visual_rank,
        is_top=True,  # Set by caller

        # Media type (JSON is more reliable, DOM has it too)
        media_type=json_post.media_type or tile.media_type,

        # Caption (JSON has full text)
        caption=json_post.caption or tile.caption,

        # Author (only from JSON for Instagram)
        author_id=json_post.author_id,
        author_handle=json_post.author_handle,

        # Engagement (only from JSON)
        likes=json_post.likes,
        comments=json_post.comments,
        shares=json_post.shares,
        views=json_post.views,
        followers=json_post.followers,

        # Temporal (only from JSON)
        posted_at_utc=json_post.posted_at_utc,

        # Hashtags
        hashtags=json_post.hashtags if json_post.hashtags else [],

        # Audio (Instagram does not surface audio metadata like TikTok)
        audio_present=False,
        audio_id="",
        audio_name="",
        audio_is_original=None,

        # Thumbnail
        thumbnail_url=json_post.cover_url or tile.thumbnail_url,

        # Source flag
        source="joined",

        # Raw data
        raw_data={
            "dom": {
                "dom_index": tile.dom_index,
                "visual_rank": tile.visual_rank,
                "rank_match": tile.rank_match,
                "shortcode": tile.shortcode,
                "media_type_dom": tile.media_type,
                "bbox": {
                    "x": tile.bbox_x,
                    "y": tile.bbox_y,
                    "w": tile.bbox_width,
                    "h": tile.bbox_height,
                },
                "dom_caption_len": len(tile.caption),
            },
            "json": {
                "views": json_post.views,
                "saves": json_post.saves,
                "duration_seconds": json_post.duration_seconds,
                "width": json_post.width,
                "height": json_post.height,
                "location_name": json_post.location_name,
                "posted_at_unix": json_post.posted_at_unix,
            },
        },
    )


def _post_from_dom_only(tile: InstagramTileData) -> CapturedPost:
    """
    Create a CapturedPost from DOM data only (no JSON match).

    Instagram DOM-only posts are sparser than TikTok DOM-only posts:
    no likes, no author, no timestamp. Only rank, shortcode, caption
    (from alt text), media type, and thumbnail.

    Args:
        tile: DOM-extracted tile.

    Returns:
        CapturedPost with DOM fields only.
    """
    return CapturedPost(
        post_id=f"instagram_{tile.shortcode}" if tile.shortcode else "",
        permalink=tile.permalink,
        rank_observed=tile.visual_rank,
        is_top=True,
        media_type=tile.media_type,
        caption=tile.caption,
        audio_present=False,
        thumbnail_url=tile.thumbnail_url,
        source="dom",
        raw_data={
            "dom_index": tile.dom_index,
            "visual_rank": tile.visual_rank,
            "rank_match": tile.rank_match,
            "shortcode": tile.shortcode,
            "dom_only_reason": "no_json_match",
        },
    )


def _check_discrepancies(
    tile: InstagramTileData,
    json_post: InstagramJSONPost,
) -> list[str]:
    """
    Compare overlapping fields between DOM and JSON data.

    For Instagram, the overlap is minimal: only shortcode and media
    type are present in both sources. Caption comparison is also
    useful since the DOM alt text may be truncated.

    Args:
        tile: DOM-extracted tile.
        json_post: JSON-intercepted post.

    Returns:
        List of discrepancy descriptions.
    """
    issues = []

    # Shortcode should always match (that is the join key)
    if (
        tile.shortcode
        and json_post.shortcode
        and tile.shortcode != json_post.shortcode
    ):
        issues.append(
            f"shortcode mismatch: DOM={tile.shortcode}, "
            f"JSON={json_post.shortcode}"
        )

    # Media type comparison
    if tile.media_type and json_post.media_type:
        # Normalise: DOM says "reel" but JSON says "video"
        dom_type = "video" if tile.media_type == "reel" else tile.media_type
        json_type = json_post.media_type
        if dom_type != json_type:
            issues.append(
                f"media_type mismatch: DOM={tile.media_type}, "
                f"JSON={json_post.media_type}"
            )

    return issues
