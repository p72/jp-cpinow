"""Offline checks (no network): partition, policy look-ahead guard, release calendar."""
import pandas as pd

from scripts import config, policy, release


def test_leaves_partition_total():
    tot = sum(config.leaf_weight(l) for l in config.LEAVES)
    assert tot == 10000
    # five targets ≈ official weights (rounding of class weights: ±1)
    official = {"headline": 10000, "core": 9621, "corecore": 8872, "western": 6609, "services": 4964}
    for k, w in official.items():
        assert abs(sum(config.leaf_weight(l) for l in config.leaves_of(k)) - w) <= 1


def test_policy_not_visible_before_announcement():
    idx = pd.date_range("2026-06-01", "2026-12-01", freq="MS")
    before = policy.monthly_value("subsidy_elec", idx, asof="2026-06-01")
    after = policy.monthly_value("subsidy_elec", idx, asof="2026-06-30")
    assert before.loc["2026-08-01"] == 0          # announced 2026-06-12
    assert after.loc["2026-08-01"] == 3.5
    assert after.loc["2026-11-01"] == 0           # October usage: no subsidy announced


def test_release_calendar():
    o = release.origins(pd.Timestamp("2026-09-01"))
    assert release.last_nat(o["a"]) == pd.Timestamp("2026-09-01")
    assert release.last_nat(o["b"]) == pd.Timestamp("2026-09-01")
    assert release.last_tky(o["a"]) == pd.Timestamp("2026-09-01")
    assert release.last_tky(o["c"]) == pd.Timestamp("2026-10-01")
    assert release.last_boj(o["a"]) == pd.Timestamp("2026-09-01")
    assert release.last_boj(o["b"]) == pd.Timestamp("2026-10-01")
