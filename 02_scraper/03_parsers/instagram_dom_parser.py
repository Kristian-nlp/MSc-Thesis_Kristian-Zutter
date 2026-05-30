"""
instagram_dom_parser.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    DOM parser for the Instagram Explore grid. Extracts ranked posts
    from a CSS grid layout where each tile is an <a> element wrapping
    a thumbnail with the caption in the img alt text. Posts are
    identified by shortcode rather than numeric ID, and media type
    is inferred from SVG aria-labels (Reel, Carousel, or absent for
    plain image).

Inputs:
    (none — library module; receives an active Selenium WebDriver)

Outputs:
    (none — library module; returns InstagramTileData dataclasses
    to the join logic. Imported by the instagram module.)

Usage:
    from instagram_dom_parser import extract_tiles, SELECTORS
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
# Update these when Instagram changes its HTML structure.
# =====================================================================

SELECTORS = {
    # Grid layout container (CSS grid with tile arrangement)
    "grid_container": "div.xdj266r.x1mpyi22",

    # Post link (<a> tag with shortcode href)
    # Primary: class-based (current). Fallback: structural href-based.
    "post_link": 'a._a6hd[href*="/p/"], a._a6hd[href*="/reel/"]',

    # Tile outer container (inside the <a> tag)
    "tile_container": "._aagu",

    # Image wrapper
    "image_wrapper": "._aagv",

    # Thumbnail image (contains caption in alt text)
    "thumbnail_img": "._aagv img",

    # Hover overlay (empty div, used for hover effects)
    "hover_overlay": "._aagw",

    # Media type indicator (sibling div after the <a> tag)
    "media_indicator": "div.xuk3077",

    # Media type SVG (inside the indicator div)
    "media_svg": "div.xuk3077 svg",

    # Explore page main content area
    "explore_main": 'main[role="main"]',
}

# Structural fallback selectors that don't depend on obfuscated class names.
# Instagram uses Atomic CSS (e.g. _a6hd, xdj266r) that changes on redeployment.
# These fallbacks use href patterns and structural relationships instead.
SELECTORS_FALLBACK = {
    "post_link": 'a[href*="/p/"], a[href*="/reel/"]',
    "thumbnail_img": 'a[href*="/p/"] img, a[href*="/reel/"] img',
    "media_svg": 'svg[aria-label]',
    "explore_main": 'main[role="main"]',
}


@dataclass
class InstagramTileData:
    """
    Raw data extracted from a single Instagram Explore grid tile.

    This is the DOM-only representation. The join logic will merge
    this with JSON data to produce the final CapturedPost.

    Instagram tiles are simpler than TikTok tiles: no like counts
    or usernames are visible in the grid. Those come from JSON.
    """
    # Rank and position
    dom_index: int = -1                 # Sequential position in DOM order
    visual_rank: int = -1               # From getBoundingClientRect sort
    rank_match: bool = True             # True if dom_index == visual_rank

    # Post identification
    shortcode: str = ""                 # Instagram shortcode (e.g. "DUYjm3sDW63")
    post_id: str = ""                   # Numeric post ID (from JSON, empty in DOM)
    permalink: str = ""                 # Full post URL
    media_type: str = ""                # "reel", "carousel", or "image"

    # Content (from img alt text)
    caption: str = ""                   # Caption text from alt attribute
    thumbnail_url: str = ""             # Cover image URL

    # Position data (pixels, for verification)
    bbox_x: float = 0.0
    bbox_y: float = 0.0
    bbox_width: float = 0.0
    bbox_height: float = 0.0


def extract_tiles(driver: WebDriver, max_tiles: int = 20) -> list[InstagramTileData]:
    """
    Extract tiles from the Instagram Explore grid.

    Uses two ranking methods (mirroring the TikTok approach):
    1. DOM order (sequential position of <a> elements)
    2. getBoundingClientRect positions (visual order)

    Args:
        driver: Active Selenium WebDriver on the Explore page.
        max_tiles: Maximum number of tiles to extract (default 20).

    Returns:
        List of InstagramTileData sorted by visual rank order.
    """
    # ---- Step 1: Get all tile data via JavaScript ----
    tile_positions = _get_tile_positions_js(driver)

    if not tile_positions:
        logger.warning("No tiles found via JavaScript, falling back to Selenium")
        selenium_tiles = _extract_tiles_selenium(driver, max_tiles)
        if not selenium_tiles:
            _diagnose_empty_grid(driver)
        return selenium_tiles

    # ---- Step 2: Sort by visual position (top-to-bottom, left-to-right) ----
    sorted_tiles = sort_by_visual_position(tile_positions)

    # ---- Step 3: Build InstagramTileData for each tile ----
    results = []
    for visual_rank, tile_info in enumerate(sorted_tiles[:max_tiles]):
        tile_data = _build_tile_data(tile_info, visual_rank=visual_rank + 1)
        results.append(tile_data)

    # ---- Step 4: Verify rank consistency ----
    mismatches = sum(1 for t in results if not t.rank_match)
    if mismatches > 0:
        logger.warning(
            "Rank mismatch: %d/%d tiles have dom_index != visual_rank. "
            "Screenshot review recommended.",
            mismatches,
            len(results),
        )
    else:
        logger.info(
            "Rank verification passed: all %d tiles match DOM index "
            "and visual position.",
            len(results),
        )

    return results


def _get_tile_positions_js(driver: WebDriver) -> list[dict]:
    """
    Get all tile positions and basic data in a single JavaScript call.

    Instagram Explore tiles are <a> tags with class _a6hd and href
    matching /p/ or /reel/. Each tile wraps a thumbnail <img> with
    caption in the alt attribute. A sibling div may contain an SVG
    indicating media type (Reel or Carousel).

    Returns a list of dicts with keys: index, x, y, width, height,
    href, shortcode, altText, thumbUrl, mediaType.
    """
    js_script = """
    // Find all post link tiles on the Explore page
    // Try class-based selector first, fall back to structural href-based
    let links = document.querySelectorAll('a._a6hd[href*="/p/"], a._a6hd[href*="/reel/"]');
    if (links.length === 0) {
        links = document.querySelectorAll('a[href*="/p/"], a[href*="/reel/"]');
    }
    const results = [];

    links.forEach((link, index) => {
        const rect = link.getBoundingClientRect();
        const href = link.getAttribute('href') || '';

        // Extract shortcode from href (/p/SHORTCODE/ or /reel/SHORTCODE/)
        const shortcodeMatch = href.match(/\\/(?:p|reel)\\/([^/]+)/);
        const shortcode = shortcodeMatch ? shortcodeMatch[1] : '';

        // Extract thumbnail image data
        // Try class-based selector first, fall back to any img inside the link
        let img = link.querySelector('._aagv img');
        if (!img) {
            img = link.querySelector('img');
        }
        const altText = img ? (img.getAttribute('alt') || '') : '';
        const thumbUrl = img ? (img.getAttribute('src') || '') : '';

        // Determine media type from sibling SVG indicator
        // Try class-based selector first, fall back to any SVG with aria-label
        let mediaType = 'image';  // Default: no indicator = static image
        const parent = link.parentElement;
        if (parent) {
            let indicator = parent.querySelector('div.xuk3077 svg');
            if (!indicator) {
                indicator = parent.querySelector('svg[aria-label]');
            }
            if (indicator) {
                const label = indicator.getAttribute('aria-label') || '';
                if (label.toLowerCase().includes('reel')) {
                    mediaType = 'reel';
                } else if (label.toLowerCase().includes('carousel')) {
                    mediaType = 'carousel';
                }
            }
        }

        results.push({
            index: index,
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: rect.height,
            href: href,
            shortcode: shortcode,
            altText: altText,
            thumbUrl: thumbUrl,
            mediaType: mediaType
        });
    });

    return results;
    """

    try:
        tiles = driver.execute_script(js_script)
        logger.debug("JavaScript extracted %d Instagram tiles", len(tiles))
        return tiles or []
    except Exception as e:
        logger.error("JavaScript tile extraction failed: %s", e)
        return []



def _build_tile_data(tile_info: dict, visual_rank: int) -> InstagramTileData:
    """
    Build an InstagramTileData from JavaScript-extracted tile info.

    Args:
        tile_info: Dict from _get_tile_positions_js().
        visual_rank: The visual rank (1-based) after position sorting.

    Returns:
        Populated InstagramTileData.
    """
    shortcode = tile_info.get("shortcode", "")
    href = tile_info.get("href", "")

    # Build full permalink
    permalink = f"https://www.instagram.com{href}" if href else ""

    tile = InstagramTileData(
        dom_index=tile_info.get("index", -1),
        visual_rank=visual_rank,
        shortcode=shortcode,
        permalink=permalink,
        media_type=tile_info.get("mediaType", "image"),
        thumbnail_url=tile_info.get("thumbUrl", ""),
        bbox_x=tile_info.get("x", 0),
        bbox_y=tile_info.get("y", 0),
        bbox_width=tile_info.get("width", 0),
        bbox_height=tile_info.get("height", 0),
    )

    # Check rank consistency (dom_index is 0-based, visual_rank is 1-based)
    tile.rank_match = (tile.dom_index == visual_rank - 1)

    # Extract caption from alt text
    alt_text = tile_info.get("altText", "")
    tile.caption = _parse_caption_from_alt(alt_text)

    return tile


def _parse_caption_from_alt(alt_text: str) -> str:
    """
    Extract the user-written caption from the img alt attribute.

    Instagram's alt text can be:
    1. The actual caption (with hashtags, emojis, etc.)
    2. Auto-generated: "Photo by USERNAME on DATE. May be an image of ..."

    We keep the full alt text as-is. The feature engineering
    pipeline parses hashtags, language, etc. later.

    Args:
        alt_text: The img alt attribute value.

    Returns:
        Caption string (may be empty).
    """
    if not alt_text:
        return ""
    return alt_text.strip()


def _extract_tiles_selenium(
    driver: WebDriver,
    max_tiles: int = 20,
) -> list[InstagramTileData]:
    """
    Fallback extraction using pure Selenium (no JavaScript).

    Used if the JS approach fails. Slower but more resilient.

    Args:
        driver: Active WebDriver.
        max_tiles: Maximum tiles to extract.

    Returns:
        List of InstagramTileData in DOM order.
    """
    logger.info("Using Selenium fallback for Instagram tile extraction")
    results = []

    try:
        links = driver.find_elements(By.CSS_SELECTOR, SELECTORS["post_link"])
        if not links:
            # Fallback to structural selectors (no class dependency)
            links = driver.find_elements(By.CSS_SELECTOR, SELECTORS_FALLBACK["post_link"])
            if links:
                logger.warning(
                    "Primary Instagram selectors failed, structural fallback found %d links. "
                    "Class names may have changed — update SELECTORS.",
                    len(links),
                )
    except Exception as e:
        logger.error("Could not find post links: %s", e)
        return results

    for i, link in enumerate(links[:max_tiles]):
        tile = InstagramTileData(dom_index=i, visual_rank=i + 1)

        try:
            # Permalink and shortcode
            href = link.get_attribute("href") or ""
            tile.permalink = href if href.startswith("http") else f"https://www.instagram.com{href}"
            match = re.search(r"/(?:p|reel)/([^/]+)", href)
            tile.shortcode = match.group(1) if match else ""

            # Thumbnail and caption from alt text
            # Try class-based selector, then fall back to any img
            try:
                img = link.find_element(By.CSS_SELECTOR, "._aagv img")
            except NoSuchElementException:
                try:
                    img = link.find_element(By.CSS_SELECTOR, "img")
                except NoSuchElementException:
                    img = None
            if img:
                tile.thumbnail_url = img.get_attribute("src") or ""
                tile.caption = _parse_caption_from_alt(
                    img.get_attribute("alt") or ""
                )

            # Media type from sibling SVG
            try:
                parent = link.find_element(By.XPATH, "..")
                try:
                    svg = parent.find_element(By.CSS_SELECTOR, "div.xuk3077 svg")
                except NoSuchElementException:
                    svg = parent.find_element(By.CSS_SELECTOR, "svg[aria-label]")
                label = svg.get_attribute("aria-label") or ""
                if "reel" in label.lower():
                    tile.media_type = "reel"
                elif "carousel" in label.lower():
                    tile.media_type = "carousel"
                else:
                    tile.media_type = "image"
            except NoSuchElementException:
                tile.media_type = "image"

        except (NoSuchElementException, StaleElementReferenceException) as e:
            logger.warning("Failed to extract Instagram tile %d: %s", i, e)

        results.append(tile)

    logger.info("Selenium fallback extracted %d Instagram tiles", len(results))
    return results


def _diagnose_empty_grid(driver: WebDriver) -> None:
    """
    Diagnose why zero tiles were found. Helps distinguish between:
    - Page loaded but grid is truly empty
    - Grid container exists but tile selectors broke (Atomic CSS changed)
    - Page didn't load at all (auth failure, rate limit, etc.)
    """
    try:
        has_main = driver.execute_script(
            f'return document.querySelector(\'{SELECTORS["explore_main"]}\') !== null;'
        )
        # Check for any <a> with /p/ or /reel/ href (structural, no class dependency)
        has_any_post_links = driver.execute_script(
            'return document.querySelectorAll(\'a[href*="/p/"], a[href*="/reel/"]\').length;'
        )
        if has_any_post_links and has_any_post_links > 0:
            logger.error(
                "SELECTORS_STALE: Found %d post links via structural query but "
                "class-based selectors matched none. Instagram Atomic CSS classes "
                "have likely changed. Update SELECTORS dict.",
                has_any_post_links,
            )
        elif has_main:
            logger.error(
                "GRID_EMPTY: Main content area found but zero post links. "
                "Possible login wall, rate limit, or empty Explore page."
            )
        else:
            page_title = driver.title or ""
            logger.error(
                "GRID_MISSING: Main content area not found (page title: '%s'). "
                "Possible auth failure or page structure change.",
                page_title[:80],
            )
    except Exception as e:
        logger.error("GRID_DIAG_FAILED: Could not diagnose empty grid: %s", e)


def verify_extraction_with_screenshot(
    driver: WebDriver,
    tiles: list[InstagramTileData],
    output_path: str,
) -> str:
    """
    Overlay rank numbers on a screenshot for visual verification.

    Same approach as the TikTok parser: draws rank labels on each
    tile's bounding box position.

    Args:
        driver: Active WebDriver.
        tiles: Extracted tiles with bounding box data.
        output_path: Where to save the annotated screenshot.

    Returns:
        Path to the saved screenshot.
    """
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
