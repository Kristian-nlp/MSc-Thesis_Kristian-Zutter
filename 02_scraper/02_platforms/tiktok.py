"""
tiktok.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    TikTok Explore scraper. Uses dual-path extraction: Path A parses
    the Explore grid DOM for rank order and visual position; Path B
    intercepts the network JSON for metadata and engagement counters;
    the two paths are joined by post_id to produce a unified
    CapturedPost set. Explore is used instead of For You because the
    desktop For You renders as a single-video swipe feed without
    simultaneous rank positions.

Inputs:
    01_config/settings.py             discovery URL, surface params
    (Selenium WebDriver injected at construction time)

Outputs:
    (none — library module; returns a SnapshotResult to the runner.
    Imported by run_tiktok.py.)

Usage:
    from tiktok import TikTokScraper
    scraper = TikTokScraper(driver, account_type, account_key)
    result = scraper.scrape()
"""

import logging
import random
import time
from pathlib import Path

from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

import sys
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "02_scraper" / "01_core"))
sys.path.insert(0, str(_project_root / "02_scraper" / "02_platforms"))
sys.path.insert(0, str(_project_root / "02_scraper" / "03_parsers"))

from base import BaseScraper, CapturedPost, SnapshotResult

# DOM parser
from tiktok_dom_parser import (
    extract_tiles,
    verify_extraction_with_screenshot,
    TikTokTileData,
    SELECTORS,
)

# JSON interceptor
from tiktok_json_parser import intercept_and_parse, TikTokJSONPost

# Join logic
from tiktok_join_logic import join_dom_and_json, JoinReport

logger = logging.getLogger(__name__)


class TikTokScraper(BaseScraper):
    """
    Scraper for TikTok's Explore discovery surface.

    Uses dual-path extraction:
        1. DOM parsing for rank order and visual position
        2. CDP network interception for metadata and counters
        3. Join by post_id to produce enriched CapturedPost objects
    """

    def __init__(
        self,
        driver: WebDriver,
        account_type: str,
        account_key: str,
    ):
        super().__init__(
            driver=driver,
            platform="tiktok",
            account_type=account_type,
            account_key=account_key,
        )
        # Store JSON posts captured during page load
        self._json_posts: list[TikTokJSONPost] = []
        self._join_report: JoinReport | None = None

    def scrape(self) -> SnapshotResult:
        """
        Override the base scrape to integrate JSON interception.

        The key change: after the page loads (but before DOM extraction),
        we capture the network JSON responses. This ensures both paths
        operate on the same page load. The join happens during post
        extraction.
        """
        result = super().scrape()

        # Attach join report to the result for monitoring
        if self._join_report:
            result.raw_data = {  # type: ignore[attr-defined]
                "join_report": {
                    "total_dom": self._join_report.total_dom,
                    "total_json": self._join_report.total_json,
                    "matched": self._join_report.matched,
                    "dom_only": self._join_report.dom_only,
                    "json_only": self._join_report.json_only,
                    "field_discrepancies": self._join_report.field_discrepancies,
                }
            }

        return result

    def _wait_for_surface_load(self) -> None:
        """
        Wait for the Explore grid to render with video tiles.

        After the grid loads, we capture JSON responses from the
        performance logs. This must happen AFTER the page has loaded
        its API calls but BEFORE we start extracting DOM data, so
        both paths cover the same content.
        """
        self.logger.debug("Waiting for Explore grid to load...")

        # Dismiss cookie banners and popups
        self._dismiss_popups()

        # Wait for the grid container
        try:
            WebDriverWait(self.driver, self.settings.PAGE_LOAD_TIMEOUT).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, SELECTORS["grid_container"])
                )
            )
        except TimeoutException:
            self.logger.warning(
                "Grid container not found, trying tile wrapper..."
            )
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, SELECTORS["tile_wrapper"])
                )
            )

        # Wait for tiles to populate
        try:
            WebDriverWait(self.driver, 10).until(
                lambda d: len(d.find_elements(
                    By.CSS_SELECTOR, SELECTORS["tile_wrapper"]
                )) >= 12  # At least 2 full rows of 6
            )
        except TimeoutException:
            self.logger.warning(
                "Fewer than 12 tiles loaded, proceeding with what we have"
            )

        # Extra render time for images and like counts to populate
        time.sleep(self.settings.RENDER_WAIT)

        # ---- Capture JSON responses AFTER page load ----
        # The API calls have completed by now, so performance logs
        # contain the response data we need.
        self.logger.info("Intercepting network JSON responses...")
        self._json_posts = intercept_and_parse(
            driver=self.driver,
            snapshot_id=f"tiktok_{self.account_type}",
            debug_dir=self.settings.SCREENSHOT_DIR / "json_debug",
        )
        self.logger.info(
            "JSON interception: %d posts captured", len(self._json_posts)
        )

        self.logger.debug("Explore grid loaded")

    def _dismiss_popups(self) -> None:
        """Remove cookie banners and login overlays."""
        dismiss_scripts = [
            # Cookie consent banners — try to reject/decline first, then remove
            """
            (function() {
                const banner = document.querySelector(
                    'tiktok-cookie-banner, [class*="cookie"], [id*="cookie"]'
                );
                if (!banner) return;
                // Try reject/decline button first
                const buttons = banner.querySelectorAll('button');
                for (const btn of buttons) {
                    const text = btn.textContent.toLowerCase();
                    if (text.includes('decline') || text.includes('ablehnen') ||
                        text.includes('reject') || text.includes('nur erforderliche')) {
                        btn.click();
                        return;
                    }
                }
                // Fallback: remove from DOM
                banner.remove();
            })();
            """,
            # Login/signup modals
            """
            document.querySelectorAll(
                '[class*="DivModalContainer"]'
            ).forEach(el => el.style.display = 'none');
            """,
            # Fixed overlays
            """
            document.querySelectorAll('[class*="overlay"]').forEach(el => {
                if (getComputedStyle(el).position === 'fixed' &&
                    el.offsetHeight > window.innerHeight * 0.5) {
                    el.remove();
                }
            });
            """,
        ]

        for script in dismiss_scripts:
            try:
                self.driver.execute_script(script)
            except Exception as e:
                self.logger.debug("Popup dismiss script failed: %s", e)

        time.sleep(0.5)

    def _freeze_view(self) -> None:
        """
        Freeze the Explore grid to prevent content changes.

        Stops video autoplay, disables infinite scroll and lazy loading,
        and prevents dynamic content updates during extraction.
        """
        freeze_script = """
        // 1. Pause all videos
        document.querySelectorAll('video').forEach(v => {
            v.pause();
            v.autoplay = false;
        });

        // 2. Disable IntersectionObserver (used for lazy loading)
        window.__originalIO = window.IntersectionObserver;
        window.IntersectionObserver = class {
            constructor() {}
            observe() {}
            unobserve() {}
            disconnect() {}
        };

        // 3. Disable scroll-triggered loading (store ref for removal)
        window.__scrollBlocker = function(e) { e.stopPropagation(); };
        window.addEventListener('scroll', window.__scrollBlocker, true);

        // 4. Record freeze timestamp
        window.__frozenAt = new Date().toISOString();
        return window.__frozenAt;
        """

        frozen_at = self.driver.execute_script(freeze_script)
        self.logger.info("Explore grid frozen at %s", frozen_at)

    def _extract_top_posts(self) -> list[CapturedPost]:
        """
        Extract Top 20 posts using dual-path extraction + join.

        Flow:
            1. DOM parser extracts tiles with rank order
            2. JSON interceptor already captured metadata
            3. Join logic merges by post_id

        Returns:
            List of up to 20 enriched CapturedPost objects.
        """
        top_n = self.platform_config["top_n"]

        # ---- Path A: DOM extraction ----
        tiles = extract_tiles(self.driver, max_tiles=top_n)

        if not tiles:
            self.logger.error("No tiles extracted from Explore grid")
            return []

        # Save rank-annotated screenshot for verification
        screenshot_path = (
            self.settings.SCREENSHOT_DIR / f"ranks_{self.account_key}.png"
        )
        verify_extraction_with_screenshot(
            self.driver, tiles, str(screenshot_path)
        )

        # ---- Join DOM + JSON ----
        posts, self._join_report = join_dom_and_json(
            dom_tiles=tiles,
            json_posts=self._json_posts,
            log_discrepancies=True,
        )

        # NOTE: rank_observed and is_top are set by BaseScraper.scrape()

        self.logger.info(
            "Extracted %d Top posts (target: %d) -- "
            "%d joined, %d DOM-only",
            len(posts),
            top_n,
            self._join_report.matched if self._join_report else 0,
            self._join_report.dom_only if self._join_report else len(posts),
        )

        return posts

    def _extract_baseline_posts(self) -> list[CapturedPost]:
        """
        Extract baseline posts by scrolling past the Top section.

        Scrolls down to load tiles beyond the initial Top 20,
        targeting ranks 51-100 (50 baseline posts).

        Baseline posts also use dual-path extraction. However,
        scrolling triggers new API calls, so we re-intercept
        JSON after scrolling.
        """
        # Re-enable lazy loading so scrolling loads new tiles
        self._unfreeze_view()

        baseline_start = self.platform_config["baseline_start"]  # 51
        baseline_end = self.platform_config["baseline_end"]       # 100
        target_count = baseline_end - baseline_start + 1          # 50

        self.logger.info(
            "Scrolling to collect baseline (ranks %d-%d)",
            baseline_start,
            baseline_end,
        )

        # Count current tiles
        current_count = self._count_tiles()
        stale_attempts = 0
        total_scrolls = 0
        max_stale = self.settings.PLATFORMS["tiktok"]["max_scroll_attempts"]
        max_total = max_stale * 3  # Hard cap to prevent infinite loops

        while current_count < baseline_end and stale_attempts < max_stale and total_scrolls < max_total:
            scroll_factor = random.uniform(0.7, 1.3)
            self.driver.execute_script(
                f"window.scrollBy(0, window.innerHeight * {scroll_factor});"
            )
            time.sleep(self.settings.SCROLL_WAIT + random.uniform(0, 1.5))
            total_scrolls += 1

            new_count = self._count_tiles()
            if new_count == current_count:
                stale_attempts += 1
            else:
                current_count = new_count
                stale_attempts = 0

            self.logger.debug(
                "Scroll %d/%d: %d tiles loaded (need %d, %d stale)",
                total_scrolls,
                max_total,
                current_count,
                baseline_end,
                stale_attempts,
            )

        if total_scrolls >= max_total:
            self.logger.warning(
                "Hit total scroll cap (%d) with %d tiles (target: %d)",
                max_total, current_count, baseline_end,
            )

        # ---- Re-intercept JSON after scrolling (new API calls) ----
        self.logger.info("Re-intercepting JSON after scroll...")
        scroll_json_posts = intercept_and_parse(
            driver=self.driver,
            snapshot_id=f"tiktok_{self.account_type}_baseline",
            debug_dir=self.settings.SCREENSHOT_DIR / "json_debug",
        )

        # Combine with initial JSON posts (deduplicate by post_id)
        all_json = {jp.post_id: jp for jp in self._json_posts}
        for jp in scroll_json_posts:
            if jp.post_id and jp.post_id not in all_json:
                all_json[jp.post_id] = jp

        combined_json = list(all_json.values())
        self.logger.info(
            "Combined JSON pool: %d posts (initial: %d, scroll: %d new)",
            len(combined_json),
            len(self._json_posts),
            len(combined_json) - len(self._json_posts),
        )

        # Extract ALL tiles then slice for baseline
        all_tiles = extract_tiles(self.driver, max_tiles=baseline_end)
        baseline_tiles = all_tiles[baseline_start - 1 : baseline_end]

        # ---- Join baseline tiles with combined JSON ----
        posts, baseline_report = join_dom_and_json(
            dom_tiles=baseline_tiles,
            json_posts=combined_json,
            log_discrepancies=False,  # Less noisy for baseline
        )

        # NOTE: rank_observed and is_top are set by BaseScraper.scrape()

        self.logger.info(
            "Extracted %d baseline posts (target: %d) -- "
            "%d joined, %d DOM-only",
            len(posts),
            target_count,
            baseline_report.matched,
            baseline_report.dom_only,
        )

        return posts

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _count_tiles(self) -> int:
        """Count tile wrappers currently in the DOM."""
        try:
            return len(self.driver.find_elements(
                By.CSS_SELECTOR, SELECTORS["tile_wrapper"]
            ))
        except Exception:
            return 0
