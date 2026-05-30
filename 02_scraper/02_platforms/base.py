"""
base.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    Abstract base class and shared dataclasses for the platform
    scrapers. Defines the four-step scraping protocol used by every
    platform: (1) reset browser state, (2) load the discovery
    surface and freeze the view, (3) dual-path extraction via DOM
    and network JSON, and (4) join and output. Subclasses (TikTok,
    Instagram, LinkedIn) implement the platform-specific extraction.

Inputs:
    01_config/settings.py             discovery URLs, surface params
    (none — library module)

Outputs:
    (none — library module; exposes CapturedPost, SnapshotResult,
    and BaseScraper. Imported by the scraper orchestrators (run_*.py)
    and the per-platform scrapers.)

Usage:
    from base import BaseScraper, CapturedPost, SnapshotResult
    class TikTokScraper(BaseScraper):
        ...
"""

import logging
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path

from selenium.webdriver.remote.webdriver import WebDriver
from selenium.common.exceptions import TimeoutException, WebDriverException

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from settings_loader import load_settings

logger = logging.getLogger(__name__)


@dataclass
class CapturedPost:
    """
    A single post captured from a discovery surface.

    This is the intermediate representation before writing to SQLite.
    Fields align with the schema defined in 04_database/schema.sql.
    """
    post_id: str = ""                   # Platform-specific post ID
    platform: str = ""                  # "tiktok", "instagram", or "linkedin"
    permalink: str = ""                 # Full URL to the post
    rank_observed: int = 0              # Position on the surface (1-based)
    is_top: bool = False                # True if in Top 20, False if baseline
    media_type: str = ""                # "video", "image", "carousel", etc.
    caption: str = ""                   # Raw caption text
    author_id: str = ""                 # Author identifier (hashed later)
    author_handle: str = ""             # @handle (pseudonymised later)
    likes: int | None = None            # Public counter
    comments: int | None = None         # Public counter
    shares: int | None = None           # Public counter
    views: int | None = None            # View/play count (TikTok, Instagram)
    followers: int | None = None        # Author follower count
    posted_at_utc: str = ""             # ISO timestamp of original post
    hashtags: list[str] = field(default_factory=list)
    audio_present: bool | None = None    # 0/1 from metadata (has any audio)
    audio_id: str = ""                  # Sound/audio identifier
    audio_name: str = ""                # Sound name
    audio_is_original: bool | None = None  # True if original sound (TikTok)
    thumbnail_url: str = ""             # Cover image URL
    source: str = ""                    # "dom", "json", or "joined"
    raw_data: dict = field(default_factory=dict)  # Raw data for debugging


@dataclass
class SnapshotResult:
    """
    Result of a single scraping snapshot.

    Contains all captured posts plus metadata about the scrape itself.
    This maps to the snapshots + captures tables in SQLite.
    """
    snapshot_id: str = ""
    platform: str = ""
    account_type: str = ""
    surface: str = ""
    captured_at_utc: str = ""
    timezone: str = "Europe/Zurich"
    top_posts: list[CapturedPost] = field(default_factory=list)
    baseline_posts: list[CapturedPost] = field(default_factory=list)
    success: bool = False
    error_message: str = ""
    duration_seconds: float = 0.0
    screenshot_path: str = ""
    raw_data: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)  # Structured quality indicators


class BaseScraper(ABC):
    """
    Abstract base class for platform scrapers.

    Implements the common scraping protocol. Subclasses must implement:
    - _extract_top_posts(): Extract Top N posts from the discovery surface
    - _extract_baseline_posts(): Extract baseline posts
    - _get_discovery_url(): Return the URL to scrape
    - _wait_for_surface_load(): Wait for platform-specific content
    - _freeze_view(): Stop dynamic content from changing

    The scrape() method orchestrates the full protocol.
    """

    def __init__(
        self,
        driver: WebDriver,
        platform: str,
        account_type: str,
        account_key: str,
    ):
        self.driver = driver
        self.platform = platform
        self.account_type = account_type
        self.account_key = account_key
        self.settings = load_settings()
        self.platform_config = self.settings.PLATFORMS[platform]
        self.logger = logging.getLogger(
            f"{__name__}.{platform}.{account_type}"
        )

    def scrape(self) -> SnapshotResult:
        """
        Execute the full scraping protocol for one snapshot.

        Protocol:
            1. Load page and freeze view
            2. Extract Top posts (DOM parse, rank order)
            3. Extract baseline posts (scroll or navigate)
            4. Package results

        Returns:
            SnapshotResult with all captured posts and metadata.
        """
        snapshot_id = f"{self.platform}_{self.account_type}_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        result = SnapshotResult(
            snapshot_id=snapshot_id,
            platform=self.platform,
            account_type=self.account_type,
            surface=self.platform_config["surface"],
            captured_at_utc=datetime.now(timezone.utc).isoformat(),
            timezone="Europe/Zurich",
        )

        try:
            # Step 1: Load the discovery surface
            url = self._get_discovery_url()
            self.logger.info("Loading discovery surface: %s", url)
            # Skip navigation if already on the target URL (e.g. after
            # session validation in the runner script)
            current = self.driver.current_url.split("?")[0].rstrip("/")
            target = url.split("?")[0].rstrip("/")
            if current.lower() != target.lower():
                self.driver.get(url)
            else:
                self.logger.debug("Already on %s, skipping navigation", url)

            # Step 2: Wait for content to render
            self._wait_for_surface_load()

            # Step 3: Freeze the view (stop autoplay, infinite scroll)
            self._freeze_view()

            # Take a debug screenshot
            screenshot_path = self._take_screenshot(snapshot_id)
            result.screenshot_path = str(screenshot_path)

            # Step 4: Extract Top posts
            self.logger.info("Extracting Top %d posts", self.platform_config["top_n"])
            result.top_posts = self._extract_top_posts()
            for i, post in enumerate(result.top_posts):
                post.platform = self.platform
                post.rank_observed = i + 1
                post.is_top = True

            self.logger.info("Captured %d Top posts", len(result.top_posts))

            # Step 5: Extract baseline posts
            self.logger.info("Extracting baseline posts")
            result.baseline_posts = self._extract_baseline_posts()
            for i, post in enumerate(result.baseline_posts):
                post.platform = self.platform
                post.rank_observed = self.platform_config["baseline_start"] + i
                post.is_top = False

            self.logger.info(
                "Captured %d baseline posts", len(result.baseline_posts)
            )

            # Track data quality: how many posts were captured vs target
            top_target = self.platform_config["top_n"]
            baseline_target = (
                self.platform_config["baseline_end"]
                - self.platform_config["baseline_start"]
                + 1
            )
            result.metadata["top_target"] = top_target
            result.metadata["top_actual"] = len(result.top_posts)
            result.metadata["baseline_target"] = baseline_target
            result.metadata["baseline_actual"] = len(result.baseline_posts)

            if len(result.baseline_posts) < baseline_target:
                self.logger.warning(
                    "Incomplete baseline: captured %d/%d posts",
                    len(result.baseline_posts),
                    baseline_target,
                )

            result.success = True

        except TimeoutException as e:
            result.error_message = f"Page load timeout: {e}"
            self.logger.error(result.error_message)

        except WebDriverException as e:
            result.error_message = f"WebDriver error: {e}"
            self.logger.error(result.error_message)

        except Exception as e:
            result.error_message = f"Unexpected error: {e}"
            self.logger.exception(result.error_message)

        finally:
            result.duration_seconds = time.time() - start_time
            self.logger.info(
                "Snapshot %s completed in %.1fs (success=%s, top=%d, baseline=%d)",
                snapshot_id,
                result.duration_seconds,
                result.success,
                len(result.top_posts),
                len(result.baseline_posts),
            )

        return result

    def _take_screenshot(self, snapshot_id: str) -> Path:
        """Save a debug screenshot for verification."""
        path = self.settings.SCREENSHOT_DIR / f"{snapshot_id}.png"
        try:
            self.driver.save_screenshot(str(path))
            self.logger.debug("Screenshot saved: %s", path)
        except Exception as e:
            self.logger.warning("Failed to save screenshot: %s", e)
        return path

    def _get_discovery_url(self) -> str:
        """Return the discovery surface URL for this platform."""
        return self.platform_config["discovery_url"]

    # ----- Abstract methods for platform-specific logic -----

    @abstractmethod
    def _wait_for_surface_load(self) -> None:
        """
        Wait for the discovery surface to fully render.

        Each platform has different loading behaviour. This method
        should block until the main content area is visible and
        populated with posts.
        """
        ...

    @abstractmethod
    def _freeze_view(self) -> None:
        """
        Freeze the discovery surface to prevent content changes.

        Stop autoplay videos, disable infinite scroll, and prevent
        any dynamic content updates that would change the visible
        rank order during extraction.
        """
        ...

    def _unfreeze_view(self) -> None:
        """
        Re-enable lazy loading and scroll-triggered content loading.

        Must be called before baseline extraction on platforms that
        collect baseline by scrolling on the same page (TikTok,
        Instagram). Without this, the freeze from _freeze_view()
        blocks IntersectionObserver and scroll events, preventing
        new tiles from loading during baseline scrolling.

        LinkedIn does not need this — its baseline navigates to a
        new URL (?sortBy=recent) which resets JS state.
        """
        unfreeze_script = """
        // 1. Restore IntersectionObserver if we saved the original
        if (window.__originalIO) {
            window.IntersectionObserver = window.__originalIO;
            delete window.__originalIO;
        }

        // 2. Remove the scroll-blocking listener.
        //    We stored the listener reference during freeze so we can
        //    remove it now. If not stored, this is a no-op.
        if (window.__scrollBlocker) {
            window.removeEventListener('scroll', window.__scrollBlocker, true);
            delete window.__scrollBlocker;
        }

        // 3. Mark unfreeze timestamp
        window.__unfrozenAt = new Date().toISOString();
        return window.__unfrozenAt;
        """

        try:
            unfrozen_at = self.driver.execute_script(unfreeze_script)
            self.logger.info("View unfrozen at %s for baseline scrolling", unfrozen_at)
        except Exception as e:
            self.logger.warning("Failed to unfreeze view: %s", e)

    @abstractmethod
    def _extract_top_posts(self) -> list[CapturedPost]:
        """
        Extract the Top N posts from the discovery surface.

        Posts must be in on-screen order (left to right, top to
        bottom) as defined by the scraping protocol. This is the
        DOM parsing step.

        Returns:
            List of CapturedPost objects in rank order.
        """
        ...

    @abstractmethod
    def _extract_baseline_posts(self) -> list[CapturedPost]:
        """
        Extract baseline posts for comparison.

        For TikTok/Instagram: scroll past Top to ranks 51-100.
        For LinkedIn: navigate to the Recent tab.

        Returns:
            List of CapturedPost objects in rank order.
        """
        ...
