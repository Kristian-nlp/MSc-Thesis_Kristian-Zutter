"""
tiktok_dom_parser.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    DOM parser for the TikTok Explore grid. Extracts ranked posts
    using two independent rank methods (sequential grid-item-container
    IDs and visual position from getBoundingClientRect). Selectors
    are defined as module-level constants so they can be updated
    in one place when TikTok changes its HTML.

Inputs:
    (none — library module; receives an active Selenium WebDriver)

Outputs:
    (none — library module; returns TikTokTileData dataclasses to
    the TikTok scraper. Imported by the tiktok module.)

Usage:
    from tiktok_dom_parser import extract_tiles, SELECTORS
    tiles = extract_tiles(driver, max_tiles=20)
"""

import logging
import re
from dataclasses import dataclass

from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import (
    NoSuchElementException,
    StaleElementReferenceException,
)

from parser_utils import sort_by_visual_position

logger = logging.getLogger(__name__)


# =====================================================================
# CONFIGURABLE SELECTORS
# Last verified: 28 February 2026
# Update these when TikTok changes its HTML structure.
# =====================================================================

SELECTORS = {
    # The grid container wrapping all tiles
    "grid_container": '[data-e2e="explore-item-list"]',

    # Individual tile wrapper (has sequential ID: grid-item-container-N)
    "tile_wrapper": 'div[id^="grid-item-container-"]',

    # The explore item inside each tile
    "explore_item": '[data-e2e="explore-item"]',

    # Post permalink (contains /@user/video/ID)
    "post_link": 'a[href*="/video/"]',

    # Like count container and its span
    "like_container": '[data-e2e="explore-card-like-container"]',
    "like_count_span": '[data-e2e="explore-card-like-container"] span',

    # Username
    "username": '[data-e2e="explore-card-user-unique-id"]',

    # User profile link
    "user_link": '[data-e2e="explore-card-user-link"]',

    # Verified badge (present only on verified accounts)
    "verified_badge": '[data-e2e="explore-card-user-verified"]',

    # Thumbnail image (inside the video container)
    "thumbnail_img": "picture img",

    # Caption/description area below the tile
    "card_desc": '[data-e2e="explore-card-desc"]',
}


@dataclass
class TikTokTileData:
    """
    Raw data extracted from a single TikTok Explore grid tile.

    This is the DOM-only representation. The join logic merges this
    with JSON data to produce the final CapturedPost.
    """
    # Rank and position
    grid_index: int = -1                # From grid-item-container-N
    visual_rank: int = -1               # From getBoundingClientRect sort
    rank_match: bool = True             # True if grid_index == visual_rank

    # Post identification
    post_id: str = ""                   # Video ID from URL
    permalink: str = ""                 # Full video URL
    author_handle: str = ""             # @username
    author_verified: bool = False       # Has verified badge

    # Content
    caption: str = ""                   # From alt text or card desc
    audio_name: str = ""                # Extracted from alt text pattern
    like_count: int | None = None       # Parsed from "349K" format
    thumbnail_url: str = ""             # Cover image URL

    # Position data (pixels, for verification)
    bbox_x: float = 0.0
    bbox_y: float = 0.0
    bbox_width: float = 0.0
    bbox_height: float = 0.0


def extract_tiles(driver: WebDriver, max_tiles: int = 20) -> list[TikTokTileData]:
    """
    Extract tiles from the TikTok Explore grid.

    Uses two ranking methods:
    1. grid-item-container-{N} sequential IDs (DOM order)
    2. getBoundingClientRect positions (visual order)

    Args:
        driver: Active Selenium WebDriver on the Explore page.
        max_tiles: Maximum number of tiles to extract (default 20).

    Returns:
        List of TikTokTileData sorted by visual rank order.
    """
    # ---- Step 1: Get visual positions via JavaScript ----
    # This runs a single JS call that returns all tile data at once,
    # which is faster and more consistent than querying elements
    # one by one from Python.
    tile_positions = _get_tile_positions_js(driver)

    if not tile_positions:
        logger.warning("No tiles found via JavaScript, falling back to Selenium")
        selenium_tiles = _extract_tiles_selenium(driver, max_tiles)
        if not selenium_tiles:
            # Distinguish empty grid from broken selectors
            _diagnose_empty_grid(driver)
        return selenium_tiles

    # ---- Step 2: Sort by visual position (top-to-bottom, left-to-right) ----
    # Group by row (tiles within ~50px of each other vertically are
    # in the same row), then sort by x within each row.
    sorted_tiles = sort_by_visual_position(tile_positions)

    # ---- Step 3: Extract metadata for each tile ----
    results = []
    for visual_rank, tile_info in enumerate(sorted_tiles[:max_tiles]):
        tile_data = _extract_single_tile(
            driver, tile_info, visual_rank=visual_rank + 1
        )
        results.append(tile_data)

    # ---- Step 4: Verify rank consistency ----
    mismatches = sum(1 for t in results if not t.rank_match)
    if mismatches > 0:
        logger.warning(
            "Rank mismatch: %d/%d tiles have grid_index != visual_rank. "
            "Screenshot review recommended.",
            mismatches,
            len(results),
        )
    else:
        logger.info(
            "Rank verification passed: all %d tiles match grid_index "
            "and visual position.",
            len(results),
        )

    return results


def _get_tile_positions_js(driver: WebDriver) -> list[dict]:
    """
    Get all tile positions and basic data in a single JavaScript call.

    Returns a list of dicts with keys: index, id, x, y, width, height,
    href, likeText, username.
    """
    js_script = """
    const tiles = document.querySelectorAll('div[id^="grid-item-container-"]');
    const results = [];

    tiles.forEach(tile => {
        const rect = tile.getBoundingClientRect();
        const idMatch = tile.id.match(/grid-item-container-(\\d+)/);
        const index = idMatch ? parseInt(idMatch[1]) : -1;

        // Extract href from the video link
        const link = tile.querySelector('a[href*="/video/"]');
        const href = link ? link.getAttribute('href') : '';

        // Extract like count text
        const likeSpan = tile.querySelector(
            '[data-e2e="explore-card-like-container"] span'
        );
        const likeText = likeSpan ? likeSpan.textContent.trim() : '';

        // Extract username
        const userEl = tile.querySelector(
            '[data-e2e="explore-card-user-unique-id"]'
        );
        const username = userEl ? userEl.textContent.trim() : '';

        // Extract verified status
        const verified = tile.querySelector(
            '[data-e2e="explore-card-user-verified"]'
        ) !== null;

        // Extract thumbnail alt text (contains caption + audio info)
        const img = tile.querySelector('picture img');
        const altText = img ? img.getAttribute('alt') || '' : '';
        const thumbUrl = img ? img.getAttribute('src') || '' : '';

        results.push({
            index: index,
            id: tile.id,
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: rect.height,
            href: href,
            likeText: likeText,
            username: username,
            verified: verified,
            altText: altText,
            thumbUrl: thumbUrl
        });
    });

    return results;
    """

    try:
        tiles = driver.execute_script(js_script)
        logger.debug("JavaScript extracted %d tiles", len(tiles))
        return tiles or []
    except Exception as e:
        logger.error("JavaScript tile extraction failed: %s", e)
        return []



def _extract_single_tile(
    driver: WebDriver,
    tile_info: dict,
    visual_rank: int,
) -> TikTokTileData:
    """
    Build a TikTokTileData from the JavaScript-extracted tile info.

    Args:
        driver: WebDriver (for any additional queries if needed).
        tile_info: Dict from _get_tile_positions_js().
        visual_rank: The visual rank (1-based) after position sorting.

    Returns:
        Populated TikTokTileData.
    """
    tile = TikTokTileData(
        grid_index=tile_info.get("index", -1),
        visual_rank=visual_rank,
        permalink=tile_info.get("href", ""),
        author_handle=tile_info.get("username", ""),
        author_verified=tile_info.get("verified", False),
        thumbnail_url=tile_info.get("thumbUrl", ""),
        bbox_x=tile_info.get("x", 0),
        bbox_y=tile_info.get("y", 0),
        bbox_width=tile_info.get("width", 0),
        bbox_height=tile_info.get("height", 0),
    )

    # Check if grid index matches visual rank
    # grid_index is 0-based, visual_rank is 1-based
    tile.rank_match = (tile.grid_index == visual_rank - 1)

    # Extract post ID from permalink
    # Pattern: /@username/video/7608611486151281942
    tile.post_id = _extract_post_id(tile.permalink)

    # Parse like count from text (e.g. "349K" -> 349000)
    tile.like_count = parse_count(tile_info.get("likeText", ""))

    # Parse caption and audio from alt text
    # Pattern: "Caption text... created by Author with SoundName"
    alt_text = tile_info.get("altText", "")
    tile.caption, tile.audio_name = _parse_alt_text(alt_text)

    return tile


def _extract_post_id(permalink: str) -> str:
    """Extract the numeric video ID from a TikTok permalink."""
    if not permalink:
        return ""
    match = re.search(r"/video/(\d+)", permalink)
    return match.group(1) if match else ""


def _parse_alt_text(alt_text: str) -> tuple[str, str]:
    """
    Parse caption and audio name from thumbnail alt text.

    TikTok encodes both in the alt attribute:
    "Caption text  created by Author with SoundName"

    Args:
        alt_text: The img alt attribute value.

    Returns:
        Tuple of (caption, audio_name).
    """
    if not alt_text:
        return ("", "")

    # Try to split on the LAST "created by ... with ..." occurrence.
    # The pattern is: caption  created by AuthorName with SoundName
    # Use greedy (.+) for caption so that if the caption itself contains
    # "created by", the regex matches the rightmost occurrence.
    match = re.search(
        r"^(.+)\s+created by\s+.+?\s+with\s+(.+)$",
        alt_text,
        re.IGNORECASE,
    )

    if match:
        caption = match.group(1).strip()
        audio_name = match.group(2).strip()
        return (caption, audio_name)

    # Fallback: entire alt text is caption, no audio detected
    return (alt_text.strip(), "")


def parse_count(text: str) -> int | None:
    """
    Parse a human-readable count to an integer.

    Handles TikTok display formats: "349K", "3.9M", "5.7M", "1702", "4105"

    Args:
        text: Count string from the DOM.

    Returns:
        Integer count, or None if parsing fails.
    """
    if not text:
        return None

    text = text.strip().upper().replace(",", "")

    try:
        if text.endswith("K"):
            return int(float(text[:-1]) * 1_000)
        elif text.endswith("M"):
            return int(float(text[:-1]) * 1_000_000)
        elif text.endswith("B"):
            return int(float(text[:-1]) * 1_000_000_000)
        else:
            return int(float(text))
    except (ValueError, TypeError):
        logger.debug("Could not parse count: '%s'", text)
        return None


def _extract_tiles_selenium(
    driver: WebDriver,
    max_tiles: int = 20,
) -> list[TikTokTileData]:
    """
    Fallback extraction using pure Selenium (no JavaScript).

    Used if the JS approach fails. Slower but more resilient to
    certain types of page rendering issues.

    Args:
        driver: Active WebDriver.
        max_tiles: Maximum tiles to extract.

    Returns:
        List of TikTokTileData in DOM order.
    """
    logger.info("Using Selenium fallback for tile extraction")
    results = []

    try:
        containers = driver.find_elements(
            By.CSS_SELECTOR, SELECTORS["tile_wrapper"]
        )
    except Exception as e:
        logger.error("Could not find tile containers: %s", e)
        return results

    for i, container in enumerate(containers[:max_tiles]):
        tile = TikTokTileData(grid_index=i, visual_rank=i + 1)

        try:
            # Permalink
            link = container.find_element(
                By.CSS_SELECTOR, SELECTORS["post_link"]
            )
            tile.permalink = link.get_attribute("href") or ""
            tile.post_id = _extract_post_id(tile.permalink)

            # Like count
            try:
                like_el = container.find_element(
                    By.CSS_SELECTOR, SELECTORS["like_count_span"]
                )
                tile.like_count = parse_count(like_el.text)
            except NoSuchElementException:
                pass

            # Username
            try:
                user_el = container.find_element(
                    By.CSS_SELECTOR, SELECTORS["username"]
                )
                tile.author_handle = user_el.text.strip()
            except NoSuchElementException:
                pass

            # Verified
            try:
                container.find_element(
                    By.CSS_SELECTOR, SELECTORS["verified_badge"]
                )
                tile.author_verified = True
            except NoSuchElementException:
                tile.author_verified = False

            # Thumbnail and alt text
            try:
                img = container.find_element(
                    By.CSS_SELECTOR, SELECTORS["thumbnail_img"]
                )
                tile.thumbnail_url = img.get_attribute("src") or ""
                alt = img.get_attribute("alt") or ""
                tile.caption, tile.audio_name = _parse_alt_text(alt)
            except NoSuchElementException:
                pass

        except (NoSuchElementException, StaleElementReferenceException) as e:
            logger.warning("Failed to extract tile %d: %s", i, e)

        results.append(tile)

    logger.info("Selenium fallback extracted %d tiles", len(results))
    return results


def _diagnose_empty_grid(driver: WebDriver) -> None:
    """
    Diagnose why zero tiles were found. Helps distinguish between:
    - Page loaded but grid is truly empty
    - Grid container exists but selectors for tiles broke
    - Page didn't load at all (auth failure, rate limit, etc.)
    """
    try:
        has_grid = driver.execute_script(
            f'return document.querySelector(\'{SELECTORS["grid_container"]}\') !== null;'
        )
        if has_grid:
            logger.error(
                "GRID_EMPTY: Grid container found but zero tiles extracted. "
                "TikTok tile selectors may have changed. Check SELECTORS dict."
            )
        else:
            page_title = driver.title or ""
            logger.error(
                "GRID_MISSING: Grid container not found (page title: '%s'). "
                "Possible auth failure, rate limit, or page structure change.",
                page_title[:80],
            )
    except Exception as e:
        logger.error("GRID_DIAG_FAILED: Could not diagnose empty grid: %s", e)


def verify_extraction_with_screenshot(
    driver: WebDriver,
    tiles: list[TikTokTileData],
    output_path: str,
) -> str:
    """
    Overlay rank numbers on a screenshot for visual verification.

    Draws rank numbers on each tile's position so the extraction
    order can be visually confirmed against the screen.

    Args:
        driver: Active WebDriver.
        tiles: Extracted tiles with bounding box data.
        output_path: Where to save the annotated screenshot.

    Returns:
        Path to the saved screenshot.
    """
    # Inject rank labels via JavaScript overlay
    js_overlay = """
    // Remove any existing rank overlays
    document.querySelectorAll('.rank-overlay').forEach(el => el.remove());

    const tiles = arguments[0];
    tiles.forEach(tile => {
        const overlay = document.createElement('div');
        overlay.className = 'rank-overlay';
        overlay.textContent = tile.rank;
        overlay.style.cssText = `
            position: fixed;
            left: ${tile.x + 5}px;
            top: ${tile.y + 5}px;
            background: rgba(255, 0, 0, 0.85);
            color: white;
            font-size: 18px;
            font-weight: bold;
            padding: 4px 8px;
            border-radius: 4px;
            z-index: 99999;
            pointer-events: none;
            font-family: monospace;
        `;
        document.body.appendChild(overlay);
    });
    """

    overlay_data = [
        {"rank": t.visual_rank, "x": t.bbox_x, "y": t.bbox_y}
        for t in tiles
    ]

    try:
        driver.execute_script(js_overlay, overlay_data)
        driver.save_screenshot(output_path)
        logger.info("Rank-annotated screenshot saved: %s", output_path)

        # Clean up overlays
        driver.execute_script(
            "document.querySelectorAll('.rank-overlay').forEach(el => el.remove());"
        )
    except Exception as e:
        logger.warning("Screenshot verification failed: %s", e)

    return output_path
