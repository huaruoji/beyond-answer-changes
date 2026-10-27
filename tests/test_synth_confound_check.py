"""An item-level OR must not be mistaken for the manuscript's view-level gap."""

import pytest

from scripts.synth_confound_check import change_rates


def test_change_rates_keep_both_option_orders():
    changed = {
        "one": {"target_replaced_diff": [True, True], "target_replaced_same": [True, False]},
        "two": {"target_replaced_diff": [True, False], "target_replaced_same": [False, False]},
    }
    assert change_rates(list(changed), changed, "view") == (0.75, 0.25, 2)
    assert change_rates(list(changed), changed, "any-order") == (1.0, 0.5, 2)
    assert change_rates(["one"], changed, "view") == (1.0, 0.5, 1)


def test_change_rates_reject_incomplete_view_pair():
    changed = {"one": {"target_replaced_diff": [True], "target_replaced_same": [False, False]}}
    with pytest.raises(ValueError, match="both option orders"):
        change_rates(["one"], changed, "view")
