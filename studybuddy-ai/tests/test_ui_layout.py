"""Regression test: the tabs must stay at the same page position across reruns.

If a progress box or banner above the tabs appears on one run and disappears on the next, Streamlit
treats the tabs as a new element and jumps back to the first tab ("Ask") on the user's first click.
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _layout(at):
    """Everything in the main area except the two fixed placeholder slots (indexes 0 and 1)."""
    return [(i, type(e).__name__) for i, e in at.main.children.items() if i > 1]


def test_tabs_do_not_move_after_processing():
    at = AppTest.from_file(APP, default_timeout=90).run()
    next(c for c in at.sidebar.checkbox if c.label.startswith("Offline")).check().run()
    next(b for b in at.sidebar.button if b.label == "Try sample").click().run()   # the run that shows progress
    after_processing = _layout(at)
    assert any(t == "Block" for _, t in after_processing)                           # the tabs exist

    next(b for b in at.button if b.label == "Generate quiz").click().run()          # the user's first decision
    first_decision = _layout(at)
    next(b for b in at.button if b.label == "Generate summarized notes").click().run()
    later = _layout(at)

    assert after_processing == first_decision == later


def test_tabs_do_not_move_when_demo_banner_toggles():
    at = AppTest.from_file(APP, default_timeout=90).run()
    next(b for b in at.sidebar.button if b.label == "Try sample").click().run()
    before = _layout(at)
    next(c for c in at.sidebar.checkbox if c.label.startswith("Offline")).check().run()
    assert _layout(at) == before
