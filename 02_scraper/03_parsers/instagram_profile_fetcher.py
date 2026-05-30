"""
instagram_profile_fetcher.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    Profile-level follower count fetcher for Instagram. The Explore
    API does not include follower_count in its responses, so this
    module queries Instagram's internal profile endpoint from the
    authenticated browser session and matches the response back to
    discovered authors by user PK. Follower counts are required as
    the denominator for engagement velocity features.

Inputs:
    (none — library module; receives an active Selenium WebDriver
    and the list of author PKs collected during a snapshot)

Outputs:
    (none — library module; returns a {pk: follower_count} dict to
    the Instagram scraper. Imported by the instagram module.)

Usage:
    from instagram_profile_fetcher import fetch_follower_counts
    counts = fetch_follower_counts(driver, author_pks)
"""

import logging
import time
from typing import Optional

from selenium.webdriver.remote.webdriver import WebDriver

logger = logging.getLogger(__name__)


def fetch_follower_counts(
    driver: WebDriver,
    author_pks: list[str],
    max_authors: int = 50,
    delay_between: float = 0.5,
) -> dict[str, int]:
    """
    Fetch follower counts for Instagram authors via the internal API.

    Uses ``driver.execute_async_script()`` to run ``fetch()`` calls from
    the authenticated browser context. Each call hits Instagram's
    internal user info endpoint.

    Args:
        driver: Authenticated Chrome WebDriver on instagram.com.
        author_pks: List of numeric user PKs from explore posts.
        max_authors: Cap on unique authors to query (anti-detection).
        delay_between: Seconds between requests (rate limiting).

    Returns:
        Dict mapping ``author_pk`` -> ``follower_count``.
        Authors that failed are omitted from the dict.
    """
    if not author_pks:
        return {}

    # Deduplicate and cap
    unique_pks = list(dict.fromkeys(author_pks))  # preserve order, remove dupes
    if len(unique_pks) > max_authors:
        logger.info(
            "Capping profile lookups: %d unique authors, fetching %d",
            len(unique_pks),
            max_authors,
        )
        unique_pks = unique_pks[:max_authors]

    logger.info(
        "Fetching follower counts for %d unique Instagram authors",
        len(unique_pks),
    )

    # Increase script timeout for the batch
    original_timeout_sec = driver.timeouts.script or 30
    driver.set_script_timeout(15)

    results: dict[str, int] = {}
    failures = 0

    for i, pk in enumerate(unique_pks):
        count = _fetch_single_follower_count(driver, pk)
        if count is not None:
            results[pk] = count
        else:
            failures += 1

        # Rate limit (skip delay after last request)
        if i < len(unique_pks) - 1 and delay_between > 0:
            time.sleep(delay_between)

    # Restore original timeout
    try:
        driver.set_script_timeout(original_timeout_sec)
    except Exception:
        pass

    logger.info(
        "Follower fetch complete: %d/%d succeeded, %d failed",
        len(results),
        len(unique_pks),
        failures,
    )

    return results


def _fetch_single_follower_count(
    driver: WebDriver,
    pk: str,
) -> Optional[int]:
    """
    Fetch follower_count for a single Instagram user via internal API.

    Uses the browser's authenticated session to call Instagram's
    user info endpoint. Returns None on any failure.

    Args:
        driver: Authenticated WebDriver on instagram.com.
        pk: Numeric user PK (string).

    Returns:
        Follower count as int, or None on failure.
    """
    # JavaScript fetch using the browser's authenticated session.
    # The CSRF token and app ID are required headers for Instagram's API.
    script = """
    var callback = arguments[arguments.length - 1];
    var pk = arguments[0];
    (async function() {
        try {
            var csrfToken = '';
            var cookies = document.cookie.split(';');
            for (var i = 0; i < cookies.length; i++) {
                var c = cookies[i].trim();
                if (c.startsWith('csrftoken=')) {
                    csrfToken = c.substring('csrftoken='.length);
                    break;
                }
            }
            var resp = await fetch(
                '/api/v1/users/' + pk + '/info/',
                {
                    method: 'GET',
                    headers: {
                        'X-CSRFToken': csrfToken,
                        'X-IG-App-ID': '936619743392459',
                        'X-Requested-With': 'XMLHttpRequest'
                    },
                    credentials: 'include'
                }
            );
            if (!resp.ok) {
                callback(null);
                return;
            }
            var data = await resp.json();
            var count = (data.user && data.user.follower_count) || null;
            callback(count);
        } catch (e) {
            callback(null);
        }
    })();
    """

    try:
        result = driver.execute_async_script(script, pk)
        if result is not None:
            count = int(result)
            logger.debug("Author pk=%s: follower_count=%d", pk, count)
            return count
        else:
            logger.debug("Author pk=%s: follower_count not available", pk)
            return None
    except Exception as e:
        logger.debug("Author pk=%s: fetch failed: %s", pk, e)
        return None
