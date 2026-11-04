"""The terminal progress line uses the same episode-level denominators as reports."""

from dagger.runner import format_collection_progress


def test_empty_progress_has_no_fictitious_takeover_rate():
    line = format_collection_progress([], 0, 20, 100)
    assert "[--------------------] 0/20" in line
    assert "takeover rate=n/a" in line


def test_progress_counts_requests_and_actual_interventions_separately():
    records = [
        {"policy_steps": 12, "manual_takeover_requests": 1, "intervention_count": 1},
        {"policy_steps": 15, "manual_takeover_requests": 1, "intervention_count": 0},
        {"policy_steps": 20, "manual_takeover_requests": 0, "intervention_count": 0},
        {"policy_steps": 0, "manual_takeover_requests": 0, "intervention_count": 0},
    ]
    line = format_collection_progress(records, 5, 20, 100)
    assert "[#####---------------] 5/20" in line
    assert "rollouts=4/100" in line
    assert "requested=2/3 (66.7%)" in line
    assert "triggered=1/3 (33.3%)" in line


def test_progress_bar_caps_at_full_width():
    assert "[####################] 21/20" in format_collection_progress([], 21, 20, 100)
