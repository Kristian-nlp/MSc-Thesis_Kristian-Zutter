"""
parser_utils.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    Shared parser helpers used across all platform DOM parsers.
    Currently exposes one function for sorting tiles by visual
    position (top-to-bottom, left-to-right) given their pixel
    coordinates, with a configurable row tolerance.

Inputs:
    (none — library module)

Outputs:
    (none — library module; returns sorted lists. Imported by the
    per-platform DOM parsers.)

Usage:
    from parser_utils import sort_by_visual_position
    ordered = sort_by_visual_position(tiles, row_tolerance=50.0)
"""


def sort_by_visual_position(
    tiles: list[dict],
    row_tolerance: float = 50.0,
) -> list[dict]:
    """
    Sort tiles by visual position: top-to-bottom, left-to-right.

    Tiles are grouped into rows based on their y-coordinate. Two
    tiles are in the same row if their y values differ by less
    than row_tolerance pixels. Within each row, tiles are sorted
    by x-coordinate (left to right).

    Args:
        tiles: List of tile dicts with x, y coordinates.
        row_tolerance: Max pixel difference to consider same row.

    Returns:
        Tiles sorted in visual rank order.
    """
    if not tiles:
        return []

    # Sort by y first, then x
    tiles_sorted = sorted(tiles, key=lambda t: (t["y"], t["x"]))

    # Group into rows
    rows = []
    current_row = [tiles_sorted[0]]

    for tile in tiles_sorted[1:]:
        if abs(tile["y"] - current_row[0]["y"]) <= row_tolerance:
            current_row.append(tile)
        else:
            rows.append(sorted(current_row, key=lambda t: t["x"]))
            current_row = [tile]
    rows.append(sorted(current_row, key=lambda t: t["x"]))

    # Flatten rows into a single list
    result = []
    for row in rows:
        result.extend(row)

    return result
