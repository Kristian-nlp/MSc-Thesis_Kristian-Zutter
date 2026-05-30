"""
revisit_scraper.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    Single-post page scrapers for engagement velocity revisits.
    Unlike the discovery surface scrapers (which parse feed grids),
    these navigate to individual post permalinks and extract the
    current engagement counters. Selector sources: TikTok uses
    data-e2e attributes; LinkedIn uses the Voyager API; Instagram
    uses embedded SSR JSON with DOM text patterns as fallback.

Inputs:
    01_config/cookies/*.json          per-platform session cookies
    (Selenium WebDriver or requests.Session passed by the caller)

Outputs:
    (none — library module; returns RevisitCounters dataclasses to
    the revisit orchestrator. Imported by run_revisits.py.)

Usage:
    from revisit_scraper import extract_counters, is_post_missing
    counters = extract_counters(driver, "tiktok", url, timeout=15)
"""

import json
import logging
import re
import sys
import time
import urllib.parse
from pathlib import Path

import requests
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
)

# ---------------------------------------------------------------------------
# Path setup (numbered folders cannot be imported directly)
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "01_config"))
sys.path.insert(0, str(_project_root / "02_scraper" / "01_core"))
sys.path.insert(0, str(_project_root / "02_scraper" / "02_platforms"))
sys.path.insert(0, str(_project_root / "02_scraper" / "03_parsers"))
sys.path.insert(0, str(_project_root / "04_database"))
from revisit_db import RevisitCounters
from auth import load_cookies_from_file

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Number parsing utilities
# ---------------------------------------------------------------------------

def parse_counter(text: str) -> int | None:
    """
    Parse a human-readable counter string into an integer.

    Handles formats seen across TikTok, Instagram, and LinkedIn:
        "621.6K"   -> 621600
        "1.2M"     -> 1200000
        "7.632"    -> 7632     (German thousands separator)
        "10.456.789" -> 10456789 (large German number)
        "1901"     -> 1901
        "29"       -> 29
        "1,234"    -> 1234     (English thousands separator)
        "31 Kommentare" -> 31  (strip trailing text)
        "Keine"    -> 0        (German zero)
        ""         -> None
        None       -> None

    Returns:
        Integer count, or None if parsing fails.
    """
    if not text:
        return None

    # Strip whitespace and any trailing text (e.g. "Kommentare", "Reposts")
    text = text.strip()

    # Handle German/English zero words before stripping trailing text
    first_word = text.split()[0].lower() if text else ""
    if first_word in ("keine", "kein", "no", "none"):
        return 0

    text = re.sub(r"\s+.*$", "", text)

    # Handle German decimal comma: "11.700,0" → "11.700"
    # Strip trailing ",0" or ",00" (display artifacts, not real decimals
    # for engagement counters which are always integers).
    # Limit to 1-2 zeros: ",000" is an English thousands separator (10,000).
    text = re.sub(r",0{1,2}$", "", text)

    if not text:
        return None

    # Handle K/M/B suffixes (TikTok style: "621.6K")
    suffix_match = re.match(
        r"^([\d,.]+)\s*([KkMmBb]?)$", text
    )
    if suffix_match:
        num_str = suffix_match.group(1)
        suffix = suffix_match.group(2).upper()

        # Replace comma with dot for float parsing (handles "1,2K")
        num_str = num_str.replace(",", ".")

        # If there are multiple dots, they are thousands separators
        # (German: "10.456.789" = 10456789)
        parts = num_str.split(".")
        if len(parts) > 2:
            # Multiple dots: ALL are thousands separators, join everything
            num_str = "".join(parts)
        elif len(parts) == 2 and not suffix:
            # Single dot without suffix: could be German thousands sep
            # "7.632" -> 7632 (German) vs "1.5" -> 1.5
            # Heuristic: if decimal part has 3 digits, it is a thousands sep
            if len(parts[1]) == 3:
                num_str = parts[0] + parts[1]

        try:
            value = float(num_str)
        except ValueError:
            logger.debug("Could not parse counter number: %r", text)
            return None

        multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
        value *= multipliers.get(suffix, 1)

        return int(value)

    logger.debug("Could not parse counter: %r", text)
    return None


# ---------------------------------------------------------------------------
# LinkedIn Voyager API revisit (no browser needed)
# ---------------------------------------------------------------------------

# Voyager API configuration (mirrored from run_linkedin_api.py)
_VOYAGER_BASE = "https://www.linkedin.com/voyager/api"
_API_HEADERS = {
    "Accept": "application/vnd.linkedin.normalized+json+2.1",
    "X-Li-Lang": "de_DE",
    "X-Li-Track": '{"clientVersion":"1.13.8878","mpVersion":"1.13.8878","osName":"web","timezoneOffset":1,"deviceFormFactor":"DESKTOP","mpName":"voyager-web"}',
    "X-Restli-Protocol-Version": "2.0.0",
}


def build_linkedin_revisit_session(
    cookie_path: Path,
    proxy_url: str = "",
) -> requests.Session:
    """
    Build a requests.Session for LinkedIn revisit API calls.

    Mirrors build_session() from run_linkedin_api.py but takes a
    cookie file path directly.

    Args:
        cookie_path: Path to the LinkedIn cookie JSON file.
        proxy_url: Evomi proxy URL (optional).

    Returns:
        Configured requests.Session with LinkedIn auth.

    Raises:
        RuntimeError: If required cookies are missing.
    """
    cookies = load_cookies_from_file(cookie_path, platform="linkedin")

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

    session = requests.Session()

    ua = (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/145.0.7632.109 Safari/537.36"
    )
    session.headers.update({
        **_API_HEADERS,
        "csrf-token": jsessionid,
        "User-Agent": ua,
    })

    if proxy_url:
        session.proxies = {"https": proxy_url, "http": proxy_url}

    # Warm-up for routing cookies (see run_linkedin_api.py for details).
    # NO anonymous fallback — anon lidc causes redirect loops.
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
            warmup_session.get(
                "https://www.linkedin.com/feed/",
                timeout=15,
                headers={"User-Agent": ua},
                allow_redirects=True,
            )
        except requests.TooManyRedirects:
            logger.info("Auth warm-up hit redirect limit (15 hops)")
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
            for cookie in warmup_resp.cookies:
                if cookie.name in routing_names:
                    session.cookies.set(cookie.name, cookie.value, domain=".linkedin.com")
                    acquired.append(cookie.name)
            if acquired:
                logger.info("No-redirect warm-up routing cookies: %s", acquired)
        except requests.RequestException as e:
            logger.warning("No-redirect warm-up failed: %s", e)

    # Set auth cookies AFTER routing cookies are in place
    session.cookies.set("li_at", li_at, domain=".linkedin.com")
    session.cookies.set("JSESSIONID", f'"{jsessionid}"', domain=".linkedin.com")

    return session


def extract_linkedin_counters_api(
    session: requests.Session,
    permalink: str,
) -> RevisitCounters:
    """
    Extract engagement counters for a LinkedIn post via Voyager API.

    Extracts the activity URN from the permalink, fetches the update
    via the Voyager API, and resolves engagement counters using the
    same entity resolution logic as the feed scraper.

    Args:
        session: Authenticated requests.Session from build_linkedin_revisit_session().
        permalink: Full LinkedIn post URL, e.g.
                   https://www.linkedin.com/feed/update/urn:li:activity:1234/

    Returns:
        RevisitCounters with likes, comments, shares.
    """
    counters = RevisitCounters()

    # Extract activity URN from permalink
    urn_match = re.search(r"(urn:li:(?:activity|ugcPost):\d+)", permalink)
    if not urn_match:
        logger.error("Could not extract URN from permalink: %s", permalink)
        return counters

    activity_urn = urn_match.group(1)

    # URL-encode the URN for the API path
    encoded_urn = urllib.parse.quote(activity_urn, safe="")

    # Fetch single update via Voyager API (/feed/updates/, not /updatesV2/)
    url = f"{_VOYAGER_BASE}/feed/updates/{encoded_urn}"

    try:
        resp = session.get(url, timeout=15)

        if resp.status_code == 429:
            logger.warning("LinkedIn API 429 — rate limited during revisit")
            return counters
        if resp.status_code in (401, 403):
            raise AuthExpiredError(
                f"LinkedIn API {resp.status_code} — li_at token may be expired"
            )

        resp.raise_for_status()
        data = resp.json()

    except AuthExpiredError:
        raise
    except requests.RequestException as e:
        logger.warning("LinkedIn API request failed for %s: %s", permalink, e)
        return counters
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("LinkedIn API response parse error for %s: %s", permalink, e)
        return counters

    # Extract counters from SocialActivityCounts in included[] array.
    # The /feed/updates/ endpoint returns SocialActivityCounts directly
    # with numLikes, numComments, numShares as top-level fields.
    included = data.get("included", [])

    # Find the post-level SocialActivityCounts (not comment-level).
    # The entityUrn may contain the activity URN or a different ugcPost URN,
    # so we match on type + exclude comment URNs.
    for entity in included:
        etype = entity.get("$type", "")
        entity_urn = entity.get("entityUrn", "")
        if (
            "SocialActivityCounts" in etype
            and "comment" not in entity_urn
            and entity.get("numLikes") is not None
        ):
            counters.likes = entity.get("numLikes")
            counters.comments = entity.get("numComments")
            counters.shares = entity.get("numShares")
            break

    # Try to extract follower count from actor description
    for entity in included:
        etype = entity.get("$type", "")
        if "MiniProfile" in etype:
            occupation = entity.get("occupation", "")
            follower_match = re.search(
                r"([\d,.']+)\s*(?:followers|Follower)", occupation, re.IGNORECASE,
            )
            if follower_match:
                try:
                    counters.follower_count = int(
                        follower_match.group(1)
                        .replace(",", "").replace(".", "").replace("'", "")
                    )
                except ValueError:
                    pass
                break

    logger.info(
        "LinkedIn API counters: likes=%s, comments=%s, shares=%s | %s",
        counters.likes, counters.comments, counters.shares, permalink,
    )
    return counters


# ---------------------------------------------------------------------------
# TikTok single-post page extraction
# ---------------------------------------------------------------------------

def extract_tiktok_counters(
    driver: WebDriver,
    url: str,
    timeout: int = 15,
) -> RevisitCounters:
    """
    Extract engagement counters from a TikTok video permalink.

    URL format: https://www.tiktok.com/@username/video/1234567890

    TikTok uses stable data-e2e attributes for counter elements.
    The standalone permalink page uses "video-*" prefixed attributes:
        - data-e2e="video-like-count"       -> likes
        - data-e2e="video-comment-count"    -> comments
        - data-e2e="video-share-count"      -> shares (shows "Teilen" label, not a number)

    There is no separate views counter on the permalink page.
    Views are available in the JSON API response if needed.

    NOTE: video-share-count contains the German label "Teilen" (not a
    number), so parse_counter returns None for shares.  This is a
    platform limitation — share counts are not exposed on TikTok
    standalone permalink pages.

    Derived from: 260304_TikTok_permalink.txt (verified 2026-03-04)
    """
    counters = RevisitCounters()

    driver.get(url)

    # Wait for the like count to appear (signals page has rendered)
    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR,
                 '[data-e2e="video-like-count"], [data-e2e="browse-like-count"]')
            )
        )
    except TimeoutException:
        logger.warning(
            "TikTok page did not render counters within %ds: %s", timeout, url,
        )
        time.sleep(3)

    # --- Likes ---
    selectors_likes = [
        '[data-e2e="video-like-count"]',     # standalone permalink page
        '[data-e2e="browse-like-count"]',    # fallback: explore overlay
        '[data-e2e="like-count"]',
    ]
    counters.likes = _try_selectors(driver, selectors_likes, "tiktok likes")

    # --- Comments ---
    selectors_comments = [
        '[data-e2e="video-comment-count"]',  # standalone permalink page
        '[data-e2e="browse-comment-count"]', # fallback: explore overlay
        '[data-e2e="comment-count"]',
    ]
    counters.comments = _try_selectors(
        driver, selectors_comments, "tiktok comments",
    )

    # --- Shares ---
    # NOTE: video-share-count on permalink contains "Teilen" (label),
    # not a number. parse_counter will return None. This is expected.
    selectors_shares = [
        '[data-e2e="video-share-count"]',    # standalone permalink page
        '[data-e2e="browse-share-count"]',   # fallback: explore overlay
        '[data-e2e="share-count"]',
    ]
    counters.shares = _try_selectors(
        driver, selectors_shares, "tiktok shares",
    )

    logger.info(
        "TikTok counters: likes=%s, comments=%s, shares=%s | %s",
        counters.likes, counters.comments, counters.shares, url,
    )
    return counters


# ---------------------------------------------------------------------------
# LinkedIn single-post page extraction
# ---------------------------------------------------------------------------

def extract_linkedin_counters(
    driver: WebDriver,
    url: str,
    timeout: int = 15,
) -> RevisitCounters:
    """
    Extract engagement counters from a LinkedIn post permalink.

    URL format: https://www.linkedin.com/feed/update/urn:li:activity:1234

    LinkedIn permalink pages (verified 18 Mar 2026) use BEM-style
    classes that survived the feed rewrite:
        - Reactions: span.social-details-social-counts__reactions-count
        - Comments: button[aria-label*="Kommentare"] → regex
        - Reposts: button[aria-label*="Reposts"] → regex
        - Follower count: .update-components-actor__description text

    The old data-test-id="social-actions__*" + data-num-* attributes
    no longer exist on permalink pages.

    LinkedIn does not expose view counts in the public DOM.
    The UI is in German (Swiss locale).

    Derived from: 260318 LinkedIn permalink HTML (verified 2026-03-18)
    """
    counters = RevisitCounters()

    driver.get(url)

    # Wait for social counts section (BEM classes still present on permalinks)
    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, '.social-details-social-counts')
            )
        )
    except TimeoutException:
        logger.warning(
            "LinkedIn social counts did not appear within %ds: %s",
            timeout, url,
        )
        time.sleep(3)

    # --- Reactions (likes + other reaction types combined) ---
    # Primary: BEM class on permalink
    selectors_reactions = [
        "span.social-details-social-counts__reactions-count",
        ".social-details-social-counts__reactions-count",
    ]
    counters.likes = _try_selectors(
        driver, selectors_reactions, "linkedin reactions",
    )

    # --- Comments ---
    # Primary: button with aria-label containing "Kommentare"
    if counters.comments is None:
        try:
            buttons = driver.find_elements(
                By.CSS_SELECTOR,
                'button[aria-label*="Kommentar"]',
            )
            for btn in buttons:
                aria = btn.get_attribute("aria-label") or ""
                match = re.search(
                    r"(\d[\d.]*)\s*Kommentar", aria, re.IGNORECASE,
                )
                if match:
                    counters.comments = parse_counter(match.group(1))
                    break
        except Exception as e:
            logger.debug("LinkedIn comments extraction error: %s", e)

    # Fallback: BEM class buttons
    if counters.comments is None:
        try:
            buttons = driver.find_elements(
                By.CSS_SELECTOR,
                ".social-details-social-counts__comments button,"
                " button.social-details-social-counts__count-value",
            )
            for btn in buttons:
                text = btn.text.strip()
                match = re.match(
                    r"(\d[\d.,]*)\s+Kommentar", text, re.IGNORECASE,
                )
                if match:
                    counters.comments = parse_counter(match.group(1))
                    break
        except Exception as e:
            logger.debug("LinkedIn comments fallback error: %s", e)

    # --- Reposts ---
    # Primary: button with aria-label containing "Reposts"
    if counters.shares is None:
        try:
            buttons = driver.find_elements(
                By.CSS_SELECTOR,
                'button[aria-label*="Repost"]',
            )
            for btn in buttons:
                aria = btn.get_attribute("aria-label") or ""
                match = re.search(
                    r"(\d[\d.]*)\s*Repost", aria, re.IGNORECASE,
                )
                if match:
                    counters.shares = parse_counter(match.group(1))
                    break
        except Exception as e:
            logger.debug("LinkedIn reposts extraction error: %s", e)

    # Fallback: right-aligned social counts
    if counters.shares is None:
        try:
            elems = driver.find_elements(
                By.CSS_SELECTOR,
                ".social-details-social-counts__item--right-aligned button",
            )
            for elem in elems:
                text = elem.text.strip()
                match = re.match(
                    r"(\d[\d.,]*)\s+Repost", text, re.IGNORECASE,
                )
                if match:
                    counters.shares = parse_counter(match.group(1))
                    break
        except Exception as e:
            logger.debug("LinkedIn reposts fallback error: %s", e)

    # --- Follower count (from permalink author section) ---
    try:
        # Try: .update-components-actor__description
        desc_elems = driver.find_elements(
            By.CSS_SELECTOR,
            ".update-components-actor__description",
        )
        for desc in desc_elems:
            text = desc.text.strip()
            match = re.search(r"([\d.]+)\s*Follower", text, re.IGNORECASE)
            if match:
                counters.follower_count = parse_counter(match.group(1))
                break

        # Fallback: aria-label "Ansehen: {Name} {N} Follower:innen"
        if not hasattr(counters, 'follower_count') or counters.follower_count is None:
            links = driver.find_elements(
                By.CSS_SELECTOR, 'a[aria-label*="Follower"]',
            )
            for link in links:
                label = link.get_attribute("aria-label") or ""
                match = re.search(r"([\d.]+)\s*Follower", label, re.IGNORECASE)
                if match:
                    counters.follower_count = parse_counter(match.group(1))
                    break
    except Exception as e:
        logger.debug("LinkedIn follower count extraction error: %s", e)

    logger.info(
        "LinkedIn counters: reactions=%s, comments=%s, reposts=%s | %s",
        counters.likes, counters.comments, counters.shares, url,
    )
    return counters


# ---------------------------------------------------------------------------
# Instagram single-post page extraction
# ---------------------------------------------------------------------------

def extract_instagram_counters(
    driver: WebDriver,
    url: str,
    timeout: int = 15,
) -> RevisitCounters:
    """
    Extract engagement counters from an Instagram post permalink.

    URL format: https://www.instagram.com/p/SHORTCODE/

    Instagram counter structure (from 260228 screenshot, German locale):
        - Likes: inside a <section>, structure is
          button > div > span.html-span containing the number,
          wrapped by "Gefaellt " and " Mal" text nodes.
          The span has class "html-span" with the raw count (e.g. "7.632").
        - Comments: no dedicated public counter on permalink page;
          count is derived from "Alle X Kommentare anzeigen" link or
          from JSON.

    Instagram uses obfuscated class names (e.g. "x1lliihq x1n2onr6"),
    so we rely on structural patterns and JS text scanning rather than
    exact class names.

    Note: Instagram's DOM uses German thousands separator (dot), so
    "7.632" means 7,632.  parse_counter() handles this.
    """
    counters = RevisitCounters()

    # Extract shortcode from URL (e.g. https://www.instagram.com/p/SHORTCODE/)
    shortcode_match = re.search(r"/p/([A-Za-z0-9_-]+)", url)
    shortcode = shortcode_match.group(1) if shortcode_match else None

    # Navigate with a short timeout — Instagram permalink pages often hang
    # loading heavy resources. We only need the DOM to be interactive enough
    # for fetch() or CDP, not fully loaded.
    original_timeout = driver.timeouts.page_load
    driver.set_page_load_timeout(30)
    try:
        driver.get(url)
    except TimeoutException:
        logger.debug("Instagram page load timed out at 30s (expected): %s", url)
    finally:
        driver.set_page_load_timeout(original_timeout)

    # Wait for the article/post to render
    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "section, article")
            )
        )
        # Extra wait for counters to populate (Instagram loads them async)
        time.sleep(2)
    except TimeoutException:
        logger.warning(
            "Instagram post did not render within %ds: %s", timeout, url,
        )
        time.sleep(3)

    # --- Strategy 0: CDP network interception (most reliable) ---
    # Capture GraphQL/API responses that fired during page load.
    # Same approach the hourly scraper uses successfully for baseline posts.
    cdp_counters = _extract_instagram_cdp_counters(driver, shortcode)
    if cdp_counters:
        counters.likes = cdp_counters.get("likes")
        counters.comments = cdp_counters.get("comments")
        counters.views = cdp_counters.get("views")
        if counters.likes is not None or counters.comments is not None:
            logger.info(
                "Instagram CDP counters: likes=%s, comments=%s, views=%s | %s",
                counters.likes, counters.comments, counters.views, url,
            )
            return counters

    # --- Strategy 0b: direct fetch() to media info API ---
    # If CDP missed the page-load responses, use the browser's own fetch()
    # to call Instagram's media info endpoint directly. The browser has valid
    # cookies, so fetch() with credentials: 'include' is authenticated.
    if shortcode:
        try:
            driver.set_script_timeout(10)
            fetch_result = driver.execute_async_script("""
                const callback = arguments[arguments.length - 1];
                const shortcode = arguments[0];
                fetch(
                    '/api/v1/media/' + shortcode + '/info/',
                    {credentials: 'include'}
                )
                    .then(r => r.json())
                    .then(data => callback(JSON.stringify(data)))
                    .catch(() => callback(null));
            """, shortcode)
            if fetch_result:
                fetch_data = json.loads(fetch_result)
                fetch_counters = _find_counters_in_json(fetch_data, shortcode)
                if fetch_counters:
                    counters.likes = fetch_counters.get("likes")
                    counters.comments = fetch_counters.get("comments")
                    counters.views = fetch_counters.get("views")
                    logger.info(
                        "Instagram fetch() counters: likes=%s, comments=%s, views=%s | %s",
                        counters.likes, counters.comments, counters.views, url,
                    )
                    return counters
                else:
                    logger.debug("Instagram fetch() returned data but no matching counters")
        except Exception as e:
            logger.debug("Instagram fetch() fallback failed: %s", e)

    # --- Likes ---
    # Strategy 1: Embedded JSON (most reliable, same data source as T0)
    try:
        counters.likes = _extract_instagram_json_counter(
            driver, "like_count", shortcode
        )
    except Exception as e:
        logger.debug("Instagram likes JSON extraction failed: %s", e)

    # Strategy 2: JS text pattern "Gefällt X Mal" / "X likes"
    if counters.likes is None:
        try:
            likes_text = driver.execute_script("""
                const buttons = document.querySelectorAll(
                    'button, div[role="button"]'
                );
                for (const btn of buttons) {
                    const text = btn.textContent || '';
                    const match = text.match(/Gef.llt\\s+([\\d.]+)\\s+Mal/);
                    if (match) return match[1];
                }
                for (const btn of buttons) {
                    const text = btn.textContent || '';
                    const match = text.match(/([\\d.,]+)\\s+likes?/i);
                    if (match) return match[1];
                }
                return null;
            """)
            if likes_text:
                counters.likes = parse_counter(likes_text)
        except Exception as e:
            logger.debug("Instagram likes JS extraction failed: %s", e)

    # Strategy 3: aria-label fallback
    if counters.likes is None:
        try:
            elems = driver.find_elements(
                By.CSS_SELECTOR,
                '[aria-label*="Gefällt"], [aria-label*="like"]',
            )
            for elem in elems:
                aria = elem.get_attribute("aria-label") or ""
                match = re.search(r"([\d.]+)", aria)
                if match:
                    counters.likes = parse_counter(match.group(1))
                    break
        except Exception as e:
            logger.debug("Instagram likes aria-label failed: %s", e)

    # REMOVED: blind span.html-span scan — grabbed wrong numbers (Issue 2)

    # --- Comments ---
    # Strategy 1: Embedded JSON (most reliable, same data source as T0)
    try:
        counters.comments = _extract_instagram_json_counter(
            driver, "comment_count", shortcode
        )
    except Exception as e:
        logger.debug("Instagram comments JSON extraction failed: %s", e)

    # Strategy 2: DOM text pattern "Alle X Kommentare" / "View all X comments"
    if counters.comments is None:
        try:
            comments_text = driver.execute_script("""
                const links = document.querySelectorAll('a, button, span');
                for (const el of links) {
                    const text = el.textContent || '';
                    let match = text.match(
                        /(?:Alle|View all)\\s+([\\d.,]+)\\s+(?:Kommentar|comment)/i
                    );
                    if (match) return match[1];
                    match = text.match(/([\\d.,]+)\\s+(?:Kommentar|comment)/i);
                    if (match) return match[1];
                }
                return null;
            """)
            if comments_text:
                counters.comments = parse_counter(comments_text)
        except Exception as e:
            logger.debug("Instagram comments DOM extraction failed: %s", e)

    # --- Views (video posts) ---
    for field in ("play_count", "video_view_count", "view_count"):
        try:
            counters.views = _extract_instagram_json_counter(
                driver, field, shortcode
            )
            if counters.views is not None:
                break
        except Exception:
            continue

    # --- Shares ---
    # Instagram does not expose share counts publicly on permalink pages.
    # counters.shares remains None.

    logger.info(
        "Instagram counters: likes=%s, comments=%s, views=%s | %s",
        counters.likes, counters.comments, counters.views, url,
    )
    return counters


# ---------------------------------------------------------------------------
# Missing post detection
# ---------------------------------------------------------------------------

class AuthExpiredError(Exception):
    """Raised when a revisit page indicates the session cookie has expired."""
    pass


def check_revisit_auth(driver: WebDriver, platform: str) -> None:
    """
    Check if the current revisit page shows a login wall.

    This must be called BEFORE is_post_missing() to distinguish
    between "post deleted" and "session expired". If the session
    has expired, all revisits will show login pages, which must
    not be misclassified as deleted posts.

    Raises:
        AuthExpiredError: If a login redirect or auth wall is detected.
    """
    current_url = driver.current_url.lower()

    auth_signals = {
        "tiktok": ["/login"],
        "instagram": ["/accounts/login", "/challenge/"],
        "linkedin": ["/login", "/authwall", "/checkpoint"],
    }

    for signal in auth_signals.get(platform, []):
        if signal in current_url:
            raise AuthExpiredError(
                f"{platform} session expired: redirected to {driver.current_url}. "
                "Re-export cookies to continue revisits."
            )


def is_post_missing(driver: WebDriver, platform: str) -> bool:
    """
    Check whether the current page indicates the post is unavailable.

    Each platform shows different signals for deleted/private content:
        - TikTok:    "Couldn't find this account" or video removed notice
        - Instagram: "Sorry, this page isn't available" / 404
        - LinkedIn:  "This content isn't available"

    IMPORTANT: Call check_revisit_auth() first to avoid misclassifying
    expired sessions as deleted posts.

    Returns:
        True if the post appears to be deleted or unavailable.
    """
    try:
        title = driver.title.lower()
    except WebDriverException:
        return True  # Cannot even read the page -- treat as missing

    # TikTok: use counter-presence as a positive signal instead of
    # searching page_source for error strings. Error strings can appear
    # in i18n translation bundles on working pages, causing false positives.
    if platform == "tiktok":
        try:
            driver.find_element(
                By.CSS_SELECTOR,
                '[data-e2e="video-like-count"], [data-e2e="browse-like-count"], [data-e2e="like-count"]',
            )
            return False  # Counters visible = post exists
        except NoSuchElementException:
            pass
        # No counter found — check title for error indicators
        tiktok_title_signals = [
            "couldn't find",
            "nicht verfügbar",
            "unavailable",
            "removed",
        ]
        for signal in tiktok_title_signals:
            if signal in title:
                logger.info("TikTok post missing (title signal): %r", signal)
                return True
        # No counter and no title signal — assume missing after timeout
        logger.info("TikTok post missing (no counters found)")
        return True

    # Instagram and LinkedIn: check title and page_source for error strings
    try:
        page_source = driver.page_source.lower()
    except WebDriverException:
        return True

    missing_signals = {
        "instagram": [
            "sorry, this page isn't available",
            "diese seite ist leider nicht verfügbar",
            "content isn't available",
        ],
        "linkedin": [
            "this content isn't available",
            "page not found",
            "dieser inhalt ist nicht verfügbar",
        ],
    }

    signals = missing_signals.get(platform, [])
    for signal in signals:
        if signal in page_source or signal in title:
            logger.info("Post missing signal detected: %r", signal)
            return True

    return False


# ---------------------------------------------------------------------------
# Instagram CDP network interception for revisits
# ---------------------------------------------------------------------------

# API URL patterns for Instagram (same as instagram_json_parser.py)
_INSTAGRAM_API_PATTERNS = re.compile(
    r"/graphql/query|/api/v1/discover|/api/v1/feed|/api/v1/media|/web/explore"
)


def _extract_instagram_cdp_counters(
    driver: WebDriver,
    shortcode: str | None,
) -> dict | None:
    """
    Extract counters from CDP performance log entries after page load.

    Captures Network.responseReceived events, fetches response bodies
    for Instagram API endpoints, and searches for the target shortcode's
    engagement data.

    Args:
        driver: Active WebDriver with CDP performance logging enabled.
        shortcode: Instagram shortcode to match in response data.

    Returns:
        Dict with likes/comments/views keys, or None if nothing found.
    """
    try:
        logs = driver.get_log("performance")
    except Exception as e:
        logger.debug("CDP performance log retrieval failed: %s", e)
        return None

    if not logs:
        logger.debug("No CDP performance log entries for Instagram revisit")
        return None

    logger.debug("Instagram revisit: processing %d performance log entries", len(logs))

    for entry in logs:
        try:
            msg = json.loads(entry["message"]).get("message", {})
            if msg.get("method") != "Network.responseReceived":
                continue

            params = msg.get("params", {})
            response = params.get("response", {})
            url = response.get("url", "")
            request_id = params.get("requestId", "")
            mime_type = response.get("mimeType", "")

            # Filter for Instagram API responses
            if "instagram.com" not in url:
                continue
            if "json" not in mime_type and "javascript" not in mime_type:
                continue
            if not (_INSTAGRAM_API_PATTERNS.search(url) or "/api/v1/" in url or "/graphql/" in url):
                continue

            # Fetch response body via CDP
            try:
                result = driver.execute_cdp_cmd(
                    "Network.getResponseBody",
                    {"requestId": request_id},
                )
                body_text = result.get("body", "")
                if not body_text:
                    continue
                if result.get("base64Encoded", False):
                    import base64
                    body_text = base64.b64decode(body_text).decode("utf-8", errors="replace")
                body = json.loads(body_text)
            except Exception:
                continue

            # Search for counters matching the shortcode
            counters = _find_counters_in_json(body, shortcode)
            if counters:
                return counters

        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    logger.debug("No matching Instagram API responses found in CDP logs")
    return None


def _find_counters_in_json(
    obj: object,
    shortcode: str | None,
    depth: int = 0,
) -> dict | None:
    """
    Recursively search JSON for engagement counters matching a shortcode.

    Looks for objects with like_count/comment_count that also have a
    matching 'code' field (shortcode).

    Returns:
        Dict with likes/comments/views, or None.
    """
    if depth > 12 or not isinstance(obj, (dict, list)):
        return None

    if isinstance(obj, dict):
        # Check if this dict has the target shortcode and counters
        code = obj.get("code") or obj.get("shortcode")
        if shortcode and code == shortcode:
            likes = obj.get("like_count")
            comments = obj.get("comment_count")
            views = (
                obj.get("play_count")
                or obj.get("video_view_count")
                or obj.get("view_count")
            )
            if likes is not None or comments is not None:
                return {
                    "likes": int(likes) if likes is not None else None,
                    "comments": int(comments) if comments is not None else None,
                    "views": int(views) if views is not None else None,
                }

        # Also check nested items/media arrays
        for key in ("items", "media", "edges", "data", "node",
                     "xdt_api__v1__media__shortcode__web_info"):
            if key in obj:
                result = _find_counters_in_json(obj[key], shortcode, depth + 1)
                if result:
                    return result

        # Recurse into all values
        for val in obj.values():
            result = _find_counters_in_json(val, shortcode, depth + 1)
            if result:
                return result

    elif isinstance(obj, list):
        for item in obj:
            result = _find_counters_in_json(item, shortcode, depth + 1)
            if result:
                return result

    return None


# ---------------------------------------------------------------------------
# Instagram embedded JSON extraction
# ---------------------------------------------------------------------------

def _extract_instagram_json_counter(
    driver: WebDriver,
    field_name: str,
    shortcode: str | None = None,
) -> int | None:
    """
    Extract a counter from Instagram's SSR-embedded JSON data.

    Instagram permalink pages embed post data in <script type="application/json">
    tags.  When *shortcode* is provided the function first looks for a JSON object
    whose ``code`` field matches the shortcode and returns the counter from that
    object — this avoids accidentally reading counters from comments, suggested
    posts, or UI-state objects.

    Fallback order:
      1. Object with matching ``code`` (shortcode).
      2. ``xdt_api__v1__media__shortcode__web_info.items[0]`` path.
      3. Blind DFS (original behaviour) — better to return *something*.

    Args:
        driver: Active WebDriver on an Instagram permalink page.
        field_name: JSON field name to extract (e.g. "comment_count").
        shortcode: Instagram shortcode extracted from the permalink URL.

    Returns:
        Integer count, or None if not found.
    """
    result = driver.execute_script("""
        const field = arguments[0];
        const shortcode = arguments[1];
        const scripts = document.querySelectorAll('script[type="application/json"]');

        // --- Strategy 1: find object whose "code" matches the shortcode ---
        if (shortcode) {
            for (const script of scripts) {
                try {
                    const text = script.textContent;
                    if (!text.includes(shortcode)) continue;
                    const data = JSON.parse(text);
                    const hit = findByCode(data, shortcode, 0);
                    if (hit !== null && hit !== undefined) return hit;
                } catch(e) {}
            }
        }

        // --- Strategy 2: xdt_api path ---
        for (const script of scripts) {
            try {
                const text = script.textContent;
                if (!text.includes('xdt_api__v1__media__shortcode__web_info')) continue;
                const data = JSON.parse(text);
                const info = findKey(data, 'xdt_api__v1__media__shortcode__web_info', 0);
                if (info && info.items && info.items[0]) {
                    const val = info.items[0][field];
                    if (typeof val === 'number') return val;
                }
            } catch(e) {}
        }

        // --- Strategy 3: blind DFS (legacy fallback) ---
        for (const script of scripts) {
            try {
                const text = script.textContent;
                if (!text.includes('"' + field + '"')) continue;
                const data = JSON.parse(text);
                const value = findField(data, field, 0);
                if (value !== null && value !== undefined) return value;
            } catch(e) {}
        }
        return null;

        // Search for an object with obj.code === shortcode, then return obj[field].
        function findByCode(obj, code, depth) {
            if (depth > 10 || !obj || typeof obj !== 'object') return null;
            if (obj.code === code && typeof obj[field] === 'number') return obj[field];
            for (const k of Object.keys(obj)) {
                const result = findByCode(obj[k], code, depth + 1);
                if (result !== null) return result;
            }
            return null;
        }

        // Return the value of a specific top-level key anywhere in the tree.
        function findKey(obj, key, depth) {
            if (depth > 10 || !obj || typeof obj !== 'object') return null;
            if (key in obj) return obj[key];
            for (const k of Object.keys(obj)) {
                const result = findKey(obj[k], key, depth + 1);
                if (result !== null) return result;
            }
            return null;
        }

        // Blind DFS for any numeric field with the given name.
        function findField(obj, key, depth) {
            if (depth > 8 || !obj || typeof obj !== 'object') return null;
            if (key in obj && typeof obj[key] === 'number') return obj[key];
            for (const k of Object.keys(obj)) {
                const result = findField(obj[k], key, depth + 1);
                if (result !== null) return result;
            }
            return null;
        }
    """, field_name, shortcode)

    if result is not None:
        logger.debug("Instagram JSON counter %s = %s (shortcode=%s)",
                      field_name, result, shortcode)
        return int(result)
    return None


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

PLATFORM_EXTRACTORS = {
    "tiktok": extract_tiktok_counters,
    "instagram": extract_instagram_counters,
    "linkedin": extract_linkedin_counters,  # Browser-based (kept as fallback, unused)
}

# LinkedIn API extractor is called directly from run_revisits.py,
# not through the PLATFORM_EXTRACTORS dispatcher.


def extract_counters(
    driver: WebDriver,
    platform: str,
    url: str,
    timeout: int = 15,
) -> RevisitCounters:
    """
    Dispatch to the platform-specific counter extractor.

    Args:
        driver: Active Selenium WebDriver.
        platform: "tiktok", "instagram", or "linkedin".
        url: Full permalink URL.
        timeout: Seconds to wait for page elements.

    Returns:
        RevisitCounters with whatever could be extracted.

    Raises:
        ValueError: If platform is not recognised.
    """
    extractor = PLATFORM_EXTRACTORS.get(platform)
    if extractor is None:
        raise ValueError(
            f"Unknown platform: {platform}. "
            f"Expected one of {list(PLATFORM_EXTRACTORS.keys())}"
        )
    return extractor(driver, url, timeout)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _try_selectors(
    driver: WebDriver,
    selectors: list[str],
    label: str,
) -> int | None:
    """
    Try multiple CSS selectors in order and return the first parsed value.

    Args:
        driver: Active WebDriver.
        selectors: List of CSS selector strings.
        label: Human-readable label for logging.

    Returns:
        Parsed integer or None.
    """
    for selector in selectors:
        try:
            elem = driver.find_element(By.CSS_SELECTOR, selector)
            text = elem.text.strip()
            value = parse_counter(text)
            if value is not None:
                logger.debug(
                    "%s: selector %r -> %r -> %d",
                    label, selector, text, value,
                )
                return value
        except NoSuchElementException:
            continue
        except Exception as e:
            logger.debug("%s: selector %r error: %s", label, selector, e)
            continue

    logger.debug("%s: no selector matched", label)
    return None
