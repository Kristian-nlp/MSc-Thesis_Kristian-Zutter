"""
auth.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    Cookie management for session authentication. Loads cookies
    exported from EditThisCookie (JSON array) and injects them into
    a Selenium WebDriver, mapping EditThisCookie fields to Selenium
    fields, applying a LinkedIn-specific whitelist (auth cookies only),
    and exposing expiry checks plus a per-platform session verifier.

Inputs:
    01_config/cookies/*.json          per-account cookie exports
    (none — library module)

Outputs:
    (none — library module; mutates the active Selenium driver and
    returns parsed cookie lists, warning strings, and a verification
    boolean. Imported by the scraper orchestrators (run_*.py).)

Usage:
    from auth import load_cookies_from_file, inject_cookies, verify_session
    cookies = load_cookies_from_file(path, platform="tiktok")
    inject_cookies(driver, cookies, "https://www.tiktok.com")
    assert verify_session(driver, "tiktok")
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from selenium.webdriver.remote.webdriver import WebDriver

logger = logging.getLogger(__name__)

# LinkedIn cookie whitelist: only inject auth-essential cookies.
# PerimeterX cookies (_pxvid, fptctx2, dfpfpt) and Cloudflare tokens
# (__cf_bm) from the export browser actively trigger bot detection
# when injected into a different browser/IP. Let the server regenerate them.
LINKEDIN_ESSENTIAL_COOKIES = {
    "li_at",        # Primary auth token
    "JSESSIONID",   # Session identifier
    "bcookie",      # Browser cookie (long-lived, identifies account)
    "bscookie",     # Secure browser cookie
    "li_rm",        # Remember-me token
    "liap",         # Login app indicator
    "li_gc",        # GDPR consent (prevents consent popups)
    "li_mc",        # Marketing consent (prevents consent popups)
    "lidc",         # Datacenter routing (prevents redirect loops)
}


def load_cookies_from_file(cookie_path: Path, platform: str = "") -> list[dict]:
    """
    Load cookies from an EditThisCookie JSON export file.

    EditThisCookie exports cookies as a JSON array where each cookie
    is an object with fields: name, value, domain, path, expirationDate,
    secure, httpOnly, sameSite, etc.

    Args:
        cookie_path: Path to the JSON cookie file.
        platform: Platform name (e.g. "linkedin"). When set to "linkedin",
                  only auth-essential cookies are kept to avoid injecting
                  stale bot-detection tokens that trigger PerimeterX.

    Returns:
        List of cookie dictionaries ready for Selenium injection.

    Raises:
        FileNotFoundError: If cookie file does not exist.
        json.JSONDecodeError: If cookie file is not valid JSON.
    """
    if not cookie_path.exists():
        raise FileNotFoundError(
            f"Cookie file not found: {cookie_path}. "
            f"Export cookies from your browser using EditThisCookie "
            f"and save them to this path."
        )

    try:
        raw_cookies = json.loads(cookie_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        logger.error(
            "Cookie file %s is corrupted (invalid JSON): %s", cookie_path, e
        )
        raise ValueError(
            f"Cookie file {cookie_path.name} is not valid JSON: {e}"
        ) from e

    if not isinstance(raw_cookies, list):
        raise ValueError(
            f"Expected a JSON array of cookies, got {type(raw_cookies).__name__}"
        )

    # LinkedIn: filter to auth-essential cookies only.
    # Stale PerimeterX/Cloudflare cookies from the export browser actively
    # trigger bot detection when injected into a different browser/IP.
    if platform.lower() == "linkedin":
        before = len(raw_cookies)
        skipped = [c.get("name", "?") for c in raw_cookies
                   if c.get("name", "") not in LINKEDIN_ESSENTIAL_COOKIES]
        raw_cookies = [c for c in raw_cookies
                       if c.get("name", "") in LINKEDIN_ESSENTIAL_COOKIES]
        logger.info(
            "LinkedIn cookie filter: kept %d/%d (skipped: %s)",
            len(raw_cookies), before, ", ".join(skipped) if skipped else "none",
        )

    # Convert EditThisCookie format to Selenium format
    selenium_cookies = []
    for cookie in raw_cookies:
        # Normalize domain: EditThisCookie sometimes exports
        # ".www.linkedin.com" or "www.linkedin.com" instead of
        # ".linkedin.com", which prevents auth cookies from working.
        domain = cookie.get("domain", "")
        if domain.endswith(".www.linkedin.com") or domain == "www.linkedin.com":
            domain = ".linkedin.com"

        sc = {
            "name": cookie["name"],
            "value": cookie["value"],
            "domain": domain,
            "path": cookie.get("path", "/"),
            "secure": cookie.get("secure", False),
        }

        # EditThisCookie uses "expirationDate" (Unix timestamp),
        # Selenium uses "expiry" (integer)
        if "expirationDate" in cookie and cookie["expirationDate"]:
            sc["expiry"] = int(cookie["expirationDate"])

        # httpOnly support
        if "httpOnly" in cookie:
            sc["httpOnly"] = cookie["httpOnly"]

        # sameSite support (Selenium accepts "Strict", "Lax", "None").
        # EditThisCookie exports "no_restriction" and "unspecified"
        # which Selenium rejects, so we map them here.
        raw_same_site = cookie.get("sameSite", "")
        if raw_same_site:
            mapping = {
                "no_restriction": "None",
                "unspecified": "Lax",
                "lax": "Lax",
                "strict": "Strict",
                "none": "None",
            }
            sc["sameSite"] = mapping.get(raw_same_site.lower(), "Lax")

        selenium_cookies.append(sc)

    logger.info(
        "Loaded %d cookies from %s", len(selenium_cookies), cookie_path.name
    )
    return selenium_cookies


def inject_cookies(driver: WebDriver, cookies: list[dict], domain: str) -> int:
    """
    Inject cookies into an active Selenium session.

    The browser must first navigate to the cookie's domain before
    injection (Selenium requirement). This function navigates to
    the domain, clears existing cookies, then injects the new ones.

    Args:
        driver: Active Selenium WebDriver instance.
        cookies: List of cookie dicts from load_cookies_from_file().
        domain: The domain to navigate to before injection
                (e.g. "https://www.tiktok.com").

    Returns:
        Number of cookies successfully injected.
    """
    # Navigate to domain first (required by Selenium for cookie injection).
    # Use /robots.txt to avoid triggering a full page render, login walls,
    # or cookie consent popups on the unauthenticated first load.
    target = domain.rstrip("/") + "/robots.txt"
    logger.debug("Navigating to %s for cookie injection", target)
    driver.get(target)

    # Clear any existing cookies from this domain
    driver.delete_all_cookies()

    # Validate that cookies match the target domain
    domain_clean = domain.replace("https://", "").replace("http://", "").split("/")[0]
    for cookie in cookies:
        cookie_domain = cookie.get("domain", "")
        if cookie_domain and not domain_clean.endswith(cookie_domain.lstrip(".")):
            logger.warning(
                "Cookie '%s' domain '%s' does not match target '%s' — "
                "check that the correct cookie file was loaded",
                cookie.get("name", "unknown"), cookie_domain, domain_clean,
            )
            break  # One warning is enough — don't spam for every cookie

    injected = 0
    failed = 0

    for cookie in cookies:
        try:
            driver.add_cookie(cookie)
            injected += 1
        except Exception as e:
            # Some cookies may fail (wrong domain, invalid format)
            # Log but do not crash - partial injection is acceptable
            logger.warning(
                "Failed to inject cookie '%s': %s",
                cookie.get("name", "unknown"),
                str(e),
            )
            failed += 1

    logger.info(
        "Injected %d/%d cookies for %s (%d failed)",
        injected,
        len(cookies),
        domain,
        failed,
    )
    return injected


def check_cookie_expiry(cookies: list[dict], warn_days: int = 3) -> list[str]:
    """
    Check if any cookies are expired or expiring soon.

    Used by the cookie health check in the hourly orchestrator.

    Args:
        cookies: List of cookie dicts.
        warn_days: Warn if cookie expires within this many days.

    Returns:
        List of warning messages for expired/expiring cookies.
    """
    warnings = []
    now = datetime.now().timestamp()
    warn_threshold = now + (warn_days * 86400)

    for cookie in cookies:
        expiry = cookie.get("expiry") or cookie.get("expirationDate")
        name = cookie.get("name", "unknown")

        if not expiry:
            # Session cookies have no expiry — flag them since they are
            # often the critical auth cookies (sessionid, li_at, etc.)
            warnings.append(
                f"Cookie '{name}' has no expiry (session cookie) — "
                f"verify it is still valid"
            )
            continue

        if expiry < now:
            warnings.append(f"Cookie '{name}' has EXPIRED")
        elif expiry < warn_threshold:
            days_left = (expiry - now) / 86400
            warnings.append(
                f"Cookie '{name}' expires in {days_left:.1f} days"
            )

    return warnings


def verify_session(driver: WebDriver, platform: str) -> bool:
    """
    Check if the browser session is authenticated after cookie injection.

    Looks for platform-specific indicators of a logged-in state. Call this
    after inject_cookies() + navigating to the discovery surface.

    Args:
        driver: Active Selenium WebDriver after cookie injection and page load.
        platform: One of "tiktok", "instagram", "linkedin".

    Returns:
        True if the session appears authenticated, False otherwise.
    """
    page_source = driver.page_source.lower()
    current_url = driver.current_url.lower()

    # Check for login redirects (all platforms)
    login_indicators = [
        "/login", "/accounts/login", "/authwall",
        "login_required", "sign-in", "signin",
    ]
    for indicator in login_indicators:
        if indicator in current_url:
            logger.warning(
                "Session validation FAILED for %s: redirected to %s",
                platform, driver.current_url,
            )
            return False

    # Platform-specific checks
    if platform == "tiktok":
        # TikTok shows a login modal or redirects unauthenticated users
        if "log in to tiktok" in page_source or "login-modal" in page_source:
            logger.warning("Session validation FAILED for TikTok: login modal detected")
            return False

    elif platform == "instagram":
        # Instagram redirects to /accounts/login/ or shows a login wall
        if "log in" in page_source and "instagram" in page_source:
            # Check for the login form specifically
            if 'id="loginform"' in page_source or "not-logged-in" in page_source:
                logger.warning("Session validation FAILED for Instagram: login form detected")
                return False

    elif platform == "linkedin":
        # LinkedIn shows an authwall for unauthenticated users
        if "authwall" in current_url or "join now" in page_source:
            logger.warning("Session validation FAILED for LinkedIn: authwall detected")
            _save_debug_screenshot(driver, platform)
            return False

        # Detect error pages (429, redirect loops, empty pages)
        error_indicators = [
            "http error", "err_too_many_redirects", "too many redirects",
            "this page isn't working", "diese website ist nicht erreichbar",
        ]
        for indicator in error_indicators:
            if indicator in page_source:
                logger.warning(
                    "Session validation FAILED for LinkedIn: error page detected (%s)",
                    indicator,
                )
                _save_debug_screenshot(driver, platform)
                return False

        # Check page title — LinkedIn feed shows "Feed | LinkedIn" or
        # "(X) Feed | LinkedIn"; error/blank pages show just "www.linkedin.com"
        # or have an empty title
        try:
            title = driver.title.strip()
        except Exception:
            title = ""
        if title in ("", "www.linkedin.com", "LinkedIn"):
            logger.warning(
                "Session validation FAILED for LinkedIn: suspicious page title '%s' "
                "(expected 'Feed | LinkedIn' or similar)",
                title,
            )
            _save_debug_screenshot(driver, platform)
            return False

        # Require a positive indicator: feed content or compose box.
        # LinkedIn is an SPA — the shell loads first, feed content renders
        # asynchronously.  Retry a few times before declaring failure.
        positive_indicators = [
            "urn:li:activity",       # Post URNs in hrefs or data attributes
            'data-testid="mainfeed"',  # New feed container (post-rewrite)
            "beitrag beginnen",      # German "Start a post"
            "start a post",          # English "Start a post"
            "share-box",             # Compose box class
        ]
        has_positive = any(ind in page_source for ind in positive_indicators)
        if not has_positive:
            # SPA may not have rendered yet — retry up to 3 times (15s total)
            import time as _time
            for attempt in range(3):
                _time.sleep(5)
                page_source = driver.page_source.lower()
                has_positive = any(ind in page_source for ind in positive_indicators)
                if has_positive:
                    logger.debug(
                        "LinkedIn positive indicator found after %d extra wait(s)",
                        attempt + 1,
                    )
                    break
        if not has_positive:
            logger.warning(
                "Session validation FAILED for LinkedIn: no positive feed indicators "
                "found (no post URNs, no compose box). Page may be an error or empty page."
            )
            _save_debug_screenshot(driver, platform)
            return False

    logger.debug("Session validation passed for %s", platform)
    return True


def _save_debug_screenshot(driver: WebDriver, platform: str) -> None:
    """Save a debug screenshot on verification failure."""
    try:
        screenshot_dir = Path("/data/screenshots")
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        path = screenshot_dir / f"verify_failed_{platform}.png"
        driver.save_screenshot(str(path))
        logger.info("Debug screenshot saved: %s", path)
    except Exception as e:
        logger.debug("Could not save debug screenshot: %s", e)
