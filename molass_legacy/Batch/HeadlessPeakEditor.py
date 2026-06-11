"""
Batch.HeadlessPeakEditor — re-exports draw_scores_headless from molass-library.

The canonical implementation lives in molass.Testing.HeadlessPeakEditor.
This module exists for backward compatibility with scripts that import from
molass_legacy.Batch.HeadlessPeakEditor.

Usage (preferred)::

    from molass.Testing.HeadlessPeakEditor import draw_scores_headless

Usage (legacy path)::

    from molass_legacy.Batch.HeadlessPeakEditor import draw_scores_headless

See molass-library/molass/Testing/HeadlessPeakEditor.py for full documentation.
"""
from molass.Testing.HeadlessPeakEditor import draw_scores_headless  # noqa: F401

__all__ = ["draw_scores_headless"]