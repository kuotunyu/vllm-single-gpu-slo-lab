import numpy as np
import pytest

from slo_lab.stats import CI, bootstrap_ci, paired_diff_ci, percentile, percentile_ci, wilson_ci


def test_percentile_linear_interpolation():
    assert percentile([1, 2, 3, 4, 5], 50) == 3.0
    assert percentile([1, 2, 3, 4, 5], 95) == pytest.approx(4.8)


def test_percentile_rejects_empty_and_bad_q():
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_wilson_hand_computed():
    # 5/10 at 95%: centre 0.5, half-width 1.96/1.38416 * sqrt(0.025 + 0.009604) = 0.26341
    ci = wilson_ci(5, 10)
    assert ci.point == 0.5
    assert ci.lo == pytest.approx(0.2366, abs=1e-3)
    assert ci.hi == pytest.approx(0.7634, abs=1e-3)
    # 0/10: lo clamps to 0, hi = z^2 / (n + z^2) = 3.8416 / 13.8416
    ci0 = wilson_ci(0, 10)
    assert ci0.lo == 0.0
    assert ci0.hi == pytest.approx(0.27754, abs=1e-4)
    # 10/10 mirrors it
    ci10 = wilson_ci(10, 10)
    assert ci10.lo == pytest.approx(0.72246, abs=1e-4)
    assert ci10.hi == pytest.approx(1.0)


def test_wilson_rejects_bad_inputs():
    with pytest.raises(ValueError):
        wilson_ci(0, 0)
    with pytest.raises(ValueError):
        wilson_ci(11, 10)


def test_bootstrap_constant_sample_is_degenerate():
    assert bootstrap_ci([2.0] * 20, np.mean) == CI(2.0, 2.0, 2.0)


def test_bootstrap_percentile_brackets_point_and_is_deterministic():
    values = list(range(1, 101))
    a = percentile_ci(values, 50, seed=1)
    b = percentile_ci(values, 50, seed=1)
    assert a == b
    assert a.point == 50.5
    assert a.lo <= a.point <= a.hi
    assert 35 < a.lo < a.hi < 66
    assert percentile_ci(values, 50, seed=2) != a


def test_bootstrap_rejects_bad_inputs():
    with pytest.raises(ValueError):
        bootstrap_ci([], np.mean)
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0], np.mean, n_boot=0)
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0], np.mean, alpha=1.5)


def test_paired_diff_identical_is_zero_and_shift_is_exact():
    a = np.arange(10, dtype=float)
    assert paired_diff_ci(a, a, np.mean) == CI(0.0, 0.0, 0.0)
    shifted = paired_diff_ci(a, a + 1.0, np.mean)
    assert shifted.point == pytest.approx(-1.0)
    assert shifted.lo == pytest.approx(-1.0)
    assert shifted.hi == pytest.approx(-1.0)


def test_paired_diff_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_diff_ci([1.0, 2.0], [1.0], np.mean)
