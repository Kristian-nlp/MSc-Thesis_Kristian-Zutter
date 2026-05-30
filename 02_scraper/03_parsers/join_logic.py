"""
join_logic.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    Shared dataclass for join operations between DOM and JSON
    extraction paths. Used by the platform-specific join modules
    (tiktok_join_logic, instagram_join_logic) to track join
    quality across snapshots.

Inputs:
    (none — library module)

Outputs:
    (none — library module; exposes the JoinReport dataclass.
    Imported by the per-platform join modules.)

Usage:
    from join_logic import JoinReport
    report = JoinReport(total_dom=20, total_json=18, matched=18)
"""

from dataclasses import dataclass


@dataclass
class JoinReport:
    """
    Summary of a join operation for monitoring and debugging.

    Track this across snapshots to detect systematic issues
    (e.g. JSON interception suddenly returning no data).
    """
    total_dom: int = 0              # Tiles from DOM parser
    total_json: int = 0             # Posts from JSON interceptor
    matched: int = 0                # Posts found in both sources
    dom_only: int = 0               # Posts only in DOM (no JSON match)
    json_only: int = 0              # Posts only in JSON (not in DOM Top/Baseline)
    field_discrepancies: int = 0    # Matched posts with differing field values
