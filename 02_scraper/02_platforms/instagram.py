"""
instagram.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    Instagram Explore scraper. Uses dual-path extraction: Path A
    parses the Explore grid DOM for rank order, visual position, and
    media type; Path B intercepts the network JSON for metadata,
    engagement counters, and author identity; the two paths are then
    joined by shortcode to produce a unified CapturedPost set.

Inputs:
    01_config/settings.py             discovery URL, surface params
    (Selenium WebDriver injected at construction time)

Outputs:
    (none — library module; returns a SnapshotResult to the runner.
    Imported by run_instagram.py.)

Usage:
    from instagram import InstagramScraper
    scraper = InstagramScraper(driver, account_type, account_key)
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

# Instagram-specific parsers
from instagram_dom_parser import (
    extract_tiles,
    verify_extraction_with_screenshot,
    InstagramTileData,
    SELECTORS,
)
from instagram_json_parser import (
    intercept_and_parse,
    extract_posts_from_page_source,
    InstagramJSONPost,
)
from instagram_join_logic import join_dom_and_json
from instagram_profile_fetcher import fetch_follower_counts

# Re-use JoinReport from shared module
from join_logic import JoinReport

logger = logging.getLogger(__name__)


class InstagramScraper(BaseScraper):
    """
    Scraper for Instagram's Explore discovery surface.

    Uses dual-path extraction:
        1. DOM parsing for rank order and media type
        2. CDP network interception for metadata and counters
        3. Join by shortcode to produce enriched CapturedPost objects

    Instagram-specific considerations:
        - Must be authenticated (cookies required)
        - Aggressive bot detection: login walls, CAPTCHAs, rate limits
        - No like counts in grid DOM (all from JSON)
        - Explore page may redirect to login if session expired
    """

    def __init__(
        self,
        driver: WebDriver,
        account_type: str,
        account_key: str,
    ):
        super().__init__(
            driver=driver,
            platform="instagram",
            account_type=account_type,
            account_key=account_key,
        )
        self._json_posts: list[InstagramJSONPost] = []
        self._join_report: JoinReport | None = None

    def _get_discovery_url(self) -> str:
        """Return the Explore page URL."""
        return "https://www.instagram.com/explore/"

    def scrape(self) -> SnapshotResult:
        """
        Override the base scrape to integrate JSON interception and
        follower count enrichment.

        Same pattern as TikTok: after page load, capture JSON responses
        before DOM extraction so both paths cover the same content.
        After extraction, enrich posts with follower counts from the
        Instagram profile API (explore JSON does not include them).
        """
        result = super().scrape()

        # Enrich with follower counts from profile API
        if result.success:
            self._enrich_follower_counts(result)

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
        Wait for the Explore grid to render with post tiles.

        Instagram may show:
        1. Login wall (session expired) -- detect and raise
        2. Cookie consent banner -- dismiss
        3. App install prompt -- dismiss
        4. The actual Explore grid with tiles

        After the grid loads, we capture JSON responses from the
        performance logs.
        """
        self.logger.debug("Waiting for Instagram Explore grid to load...")

        # Check for login redirect (session expired)
        self._check_login_redirect()

        # Dismiss popups and overlays
        self._dismiss_popups()

        # Wait for post link tiles to appear
        try:
            WebDriverWait(self.driver, self.settings.PAGE_LOAD_TIMEOUT).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, SELECTORS["post_link"])
                )
            )
        except TimeoutException:
            self.logger.warning(
                "No post tiles found. Checking for login wall..."
            )
            self._check_login_redirect()
            # Try once more after dismissing popups
            self._dismiss_popups()
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, SELECTORS["post_link"])
                )
            )

        # Wait for a reasonable number of tiles to load
        try:
            WebDriverWait(self.driver, 10).until(
                lambda d: len(d.find_elements(
                    By.CSS_SELECTOR, SELECTORS["post_link"]
                )) >= 9  # At least 3 rows of 3
            )
        except TimeoutException:
            self.logger.warning(
                "Fewer than 9 tiles loaded, proceeding with what we have"
            )

        # Extra render time for images to load
        time.sleep(self.settings.RENDER_WAIT)

        # ---- Capture JSON responses AFTER page load ----
        self.logger.info("Intercepting Instagram JSON responses...")
        self._json_posts = intercept_and_parse(
            driver=self.driver,
            snapshot_id=f"instagram_{self.account_type}",
            debug_dir=self.settings.SCREENSHOT_DIR / "json_debug",
        )
        self.logger.info(
            "JSON interception: %d posts captured", len(self._json_posts)
        )

        # ---- Fallback: direct fetch() for explore grid API ----
        # CDP network interception may miss the initial page load response.
        # Use the browser's own fetch() to call the explore grid API directly,
        # inheriting the session cookies. This fills the Top-20 metadata gap.
        if len(self._json_posts) < 20:
            self.logger.info(
                "CDP captured only %d posts (< 20). "
                "Attempting direct explore grid fetch...",
                len(self._json_posts),
            )
            try:
                # Use execute_async_script for the fetch() Promise
                self.driver.set_script_timeout(10)
                fetch_result = self.driver.execute_async_script("""
                    const callback = arguments[arguments.length - 1];
                    fetch('/api/v1/discover/web/explore_grid/', {credentials: 'include'})
                        .then(r => r.json())
                        .then(data => callback(JSON.stringify(data)))
                        .catch(() => callback(null));
                """)
                if fetch_result:
                    import json as _json
                    from instagram_json_parser import (
                        _extract_media_nodes,
                        _parse_single_item,
                    )
                    body = _json.loads(fetch_result)
                    media_nodes = _extract_media_nodes(body)
                    existing_codes = {p.shortcode for p in self._json_posts}
                    fetch_new = 0
                    for node in media_nodes:
                        post = _parse_single_item(node)
                        if post.shortcode and post.shortcode not in existing_codes:
                            self._json_posts.append(post)
                            existing_codes.add(post.shortcode)
                            fetch_new += 1
                    self.logger.info(
                        "Direct fetch: %d media nodes, %d new posts "
                        "(total JSON pool: %d)",
                        len(media_nodes),
                        fetch_new,
                        len(self._json_posts),
                    )
            except Exception as e:
                self.logger.warning(
                    "Direct explore grid fetch failed: %s", e,
                )

        # ---- Save page source for debugging (like LinkedIn) ----
        try:
            ps_path = self.settings.SCREENSHOT_DIR / f"pagesource_{self.account_key}.html"
            ps_path.write_text(self.driver.page_source, encoding="utf-8")
            self.logger.debug("Page source saved: %s", ps_path)
        except Exception as e:
            self.logger.warning("Failed to save page source: %s", e)

        # ---- Extract posts from SSR-embedded JSON ----
        ssr_posts = extract_posts_from_page_source(
            driver=self.driver,
            debug_dir=self.settings.SCREENSHOT_DIR / "json_debug",
        )
        if ssr_posts:
            # Merge: CDP wins on duplicates, SSR fills gaps
            existing_codes = {p.shortcode for p in self._json_posts}
            new_from_ssr = [p for p in ssr_posts if p.shortcode not in existing_codes]
            self._json_posts.extend(new_from_ssr)
            self.logger.info(
                "SSR merge: %d SSR posts, %d new (total JSON pool: %d)",
                len(ssr_posts),
                len(new_from_ssr),
                len(self._json_posts),
            )

        self.logger.debug("Instagram Explore grid loaded")

    def _check_login_redirect(self) -> None:
        """
        Detect if Instagram has redirected to a login page.

        This indicates the session cookie has expired and needs
        refreshing. Raises an exception to trigger the error
        handling in BaseScraper.scrape().
        """
        current_url = self.driver.current_url
        if "/accounts/login" in current_url or "/challenge/" in current_url:
            cookie_path = self.settings.ACCOUNTS.get(
                self.account_key, {},
            ).get("cookie_file", "01_config/cookies/")
            raise TimeoutException(
                f"Instagram redirected to login/challenge: {current_url}. "
                "Session expired. To fix: "
                "1) Open instagram.com in your browser and log in, "
                "2) Export cookies with EditThisCookie, "
                f"3) Save to: {cookie_path}"
            )

    def _dismiss_popups(self) -> None:
        """
        Remove cookie banners, login overlays, and app install prompts.

        Instagram commonly shows:
        - Cookie consent dialog
        - "Log in to continue" overlay
        - "Get the app" banner
        - Notification permission request
        """
        dismiss_scripts = [
            # Cookie consent banner — decline optional cookies.
            # Try "Decline optional cookies" / "Optionale Cookies ablehnen" first,
            # then fall back to "Allow essential" / "Nur erforderliche".
            # Last resort: remove the banner from DOM.
            """
            (function() {
                const buttons = document.querySelectorAll('button');
                // Priority: reject/decline buttons
                for (const btn of buttons) {
                    const text = btn.textContent.toLowerCase();
                    if (text.includes('decline') || text.includes('ablehnen') ||
                        text.includes('reject') || text.includes('nur erforderliche') ||
                        text.includes('only essential') || text.includes('nur notwendige')) {
                        btn.click();
                        return;
                    }
                }
                // Fallback: remove cookie banner from DOM entirely
                document.querySelectorAll(
                    '[class*="cookie"], [id*="cookie-banner"], [role="dialog"]'
                ).forEach(el => {
                    if (el.textContent.toLowerCase().includes('cookie')) {
                        el.remove();
                    }
                });
            })();
            """,
            # Login/signup modals and overlays
            """
            document.querySelectorAll(
                '[role="dialog"], [class*="RnEpo"], [class*="LoginOverlay"]'
            ).forEach(el => el.remove());
            """,
            # Fixed position overlays covering more than half the screen
            """
            document.querySelectorAll('div[role="presentation"]').forEach(el => {
                if (getComputedStyle(el).position === 'fixed') {
                    el.remove();
                }
            });
            """,
            # "Not now" buttons (notification prompts, app install)
            """
            document.querySelectorAll('button').forEach(btn => {
                const text = btn.textContent.trim().toLowerCase();
                if (text === 'not now' || text === 'jetzt nicht' ||
                    text === 'nicht jetzt') {
                    btn.click();
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

        Stops video autoplay, disables infinite scroll and lazy
        loading, and prevents dynamic content updates.
        """
        freeze_script = """
        // 1. Pause all videos
        document.querySelectorAll('video').forEach(v => {
            v.pause();
            v.autoplay = false;
        });

        // 2. Disable IntersectionObserver (lazy loading)
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
        self.logger.info("Instagram Explore grid frozen at %s", frozen_at)

    def _extract_top_posts(self) -> list[CapturedPost]:
        """
        Extract Top 20 posts using dual-path extraction + join.

        Flow:
            1. DOM parser extracts tiles with rank order
            2. JSON interceptor already captured metadata
            3. Join by shortcode to produce enriched CapturedPosts

        Returns:
            List of up to 20 enriched CapturedPost objects.
        """
        top_n = self.platform_config["top_n"]

        # ---- Path A: DOM extraction ----
        tiles = extract_tiles(self.driver, max_tiles=top_n)

        if not tiles:
            self.logger.error("No tiles extracted from Instagram Explore grid")
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

        Instagram Explore loads content dynamically via infinite
        scroll, firing new GraphQL/API requests as you scroll.
        We re-intercept JSON after scrolling.
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

        current_count = self._count_tiles()
        stale_attempts = 0
        total_scrolls = 0
        max_stale = self.settings.PLATFORMS["instagram"]["max_scroll_attempts"]
        # Instagram virtualizes DOM (~40-70 tiles visible at a time),
        # so we scroll a fixed number of times rather than chasing an
        # unreachable DOM count.  30 scrolls ≈ 2 min, loads ~100+ unique
        # tiles server-side even if DOM recycles them.
        max_total = 30

        while stale_attempts < max_stale and total_scrolls < max_total:
            scroll_factor = random.uniform(0.7, 1.3)
            self.driver.execute_script(
                f"window.scrollBy(0, window.innerHeight * {scroll_factor});"
            )
            time.sleep(self.settings.SCROLL_WAIT + random.uniform(0, 1.5))
            total_scrolls += 1

            new_count = self._count_tiles()
            if new_count == current_count:
                stale_attempts += 1
                # Instagram sometimes needs a pause before loading more
                if stale_attempts % 3 == 0:
                    time.sleep(2)
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

        # ---- Re-intercept JSON after scrolling ----
        self.logger.info("Re-intercepting JSON after scroll...")
        scroll_json_posts = intercept_and_parse(
            driver=self.driver,
            snapshot_id=f"instagram_{self.account_type}_baseline",
            debug_dir=self.settings.SCREENSHOT_DIR / "json_debug",
        )

        # Combine with initial JSON posts (deduplicate by shortcode)
        all_json = {jp.shortcode: jp for jp in self._json_posts}
        for jp in scroll_json_posts:
            if jp.shortcode and jp.shortcode not in all_json:
                all_json[jp.shortcode] = jp

        combined_json = list(all_json.values())
        self.logger.info(
            "Combined JSON pool: %d posts (initial: %d, scroll: %d new)",
            len(combined_json),
            len(self._json_posts),
            len(combined_json) - len(self._json_posts),
        )

        # Extract all tiles currently in DOM.  Instagram virtualizes
        # the grid so we won't have tiles 1-100 simultaneously; take
        # everything after the Top-N as baseline.
        top_n = self.platform_config["top_n"]  # 20
        all_tiles = extract_tiles(self.driver, max_tiles=baseline_end)
        baseline_tiles = [t for t in all_tiles if t.visual_rank > top_n]

        # ---- Join baseline tiles with combined JSON ----
        posts, baseline_report = join_dom_and_json(
            dom_tiles=baseline_tiles,
            json_posts=combined_json,
            log_discrepancies=False,
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

    def _enrich_follower_counts(self, result: SnapshotResult) -> None:
        """
        Fetch follower counts for unique authors and backfill into posts.

        Instagram explore JSON does not include follower_count (confirmed).
        This queries the profile API for each unique author_pk, then sets
        ``post.followers`` before the DB write.
        """
        all_posts = result.top_posts + result.baseline_posts
        if not all_posts:
            return

        # Collect unique author PKs and map to their posts
        pk_to_posts: dict[str, list[CapturedPost]] = {}
        for post in all_posts:
            if post.author_id:
                pk_to_posts.setdefault(post.author_id, []).append(post)

        if not pk_to_posts:
            self.logger.info("No author PKs to enrich with follower counts")
            return

        self.logger.info(
            "Enriching follower counts for %d unique authors",
            len(pk_to_posts),
        )

        follower_map = fetch_follower_counts(
            driver=self.driver,
            author_pks=list(pk_to_posts.keys()),
            max_authors=50,
            delay_between=0.5,
        )

        # Apply to CapturedPost objects
        enriched_authors = 0
        enriched_posts = 0
        for pk, count in follower_map.items():
            if count is not None and pk in pk_to_posts:
                enriched_authors += 1
                for post in pk_to_posts[pk]:
                    post.followers = count
                    enriched_posts += 1

        self.logger.info(
            "Follower enrichment: %d/%d authors, %d posts updated",
            enriched_authors,
            len(pk_to_posts),
            enriched_posts,
        )

    def _count_tiles(self) -> int:
        """Count post link tiles currently in the DOM."""
        try:
            return len(self.driver.find_elements(
                By.CSS_SELECTOR, SELECTORS["post_link"]
            ))
        except Exception:
            return 0
