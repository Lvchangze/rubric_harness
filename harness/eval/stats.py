"""Statistics for the rubric comparison report — pure numpy, no scipy/sklearn.

The experiment is a *paired* design: every rubric source sees the same questions,
so the interesting quantity is almost always a within-question difference
(``agentic - baseline`` on question *q*) rather than a difference of marginal
means. Everything here is built around that:

* :func:`paired_table` aligns per-question rows into a ``(question, source)``
  matrix and reports how many questions had to be dropped for incompleteness.
* :func:`paired_diff` gives a bootstrap CI on the mean paired difference and a
  p-value from a two-sided sign-flip permutation test, which is the right null
  for "the label on the rubric source is arbitrary".
* :func:`summarize_metric` packages both for one metric across all sources.

Design rules obeyed by every function here:

* **Never raise.** Empty input, a single observation, all-NaN input and constant
  input all return NaN-filled results instead of exceptions, because the report
  generator runs over dozens of metrics and one degenerate column must not kill
  the run.
* **Deterministic.** Every resampling routine takes an explicit ``seed``.
* **NaN means "not available".** Non-finite observations are dropped pairwise,
  and the surviving count is always reported as ``n``.

Run ``python -m harness.eval.stats`` to execute the self-test, which checks each
routine against synthetic data with a known answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "PairedTable",
    "auc",
    "bh_fdr",
    "cohens_kappa",
    "mean_ci",
    "paired_diff",
    "paired_table",
    "rankdata",
    "safe_mean",
    "summarize_metric",
    "wilcoxon_signed_rank",
]

NAN = float("nan")

# Enumerating 2**n sign vectors is exact; beyond this the permutation test falls
# back to Monte-Carlo sampling.
EXACT_PERMUTATION_MAX_N = 20
MIN_PERMUTATION_DRAWS = 10_000
# Cap on floats materialised per resampling chunk (~32 MB at float64).
_CHUNK_CELLS = 4_000_000


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def _as_float_array(values: Iterable[Any] | None) -> np.ndarray:
    """Coerce anything iterable to ``float64``; unconvertible entries become NaN."""
    if values is None:
        return np.empty(0, dtype=float)
    if isinstance(values, np.ndarray):
        try:
            return values.astype(float, copy=False).ravel()
        except (TypeError, ValueError):
            values = values.tolist()
    out: list[float] = []
    for value in values:
        if isinstance(value, bool):
            out.append(float(value))
            continue
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            out.append(NAN)
    return np.asarray(out, dtype=float)


def _finite(arr: np.ndarray) -> np.ndarray:
    return arr[np.isfinite(arr)]


def safe_mean(values: Iterable[Any] | None) -> float:
    """Mean of the finite entries; NaN when there are none. Never warns."""
    finite = _finite(_as_float_array(values))
    return float(finite.mean()) if finite.size else NAN


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average ("fractional") ranks, 1-based — the tie handling scipy calls ``average``."""
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 0:
        return np.empty(0, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ordered = arr[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i + 1
        while j < n and ordered[j] == ordered[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j + 1)  # mean of 1-based ranks i+1..j
        i = j
    return ranks


def _normal_two_sided_p(z: float) -> float:
    """Two-sided p-value of a standard normal deviate, via ``math.erfc``."""
    if not math.isfinite(z):
        return NAN
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def _bootstrap_means(x: np.ndarray, *, iters: int, seed: int) -> np.ndarray:
    """``iters`` bootstrap resample means of ``x``, computed in memory-safe chunks."""
    n = x.size
    rng = np.random.default_rng(seed)
    per_chunk = max(1, min(int(iters), max(1, _CHUNK_CELLS // max(1, n))))
    means = np.empty(int(iters), dtype=float)
    done = 0
    while done < iters:
        take = min(per_chunk, iters - done)
        idx = rng.integers(0, n, size=(take, n))
        means[done : done + take] = x[idx].mean(axis=1)
        done += take
    return means


def mean_ci(
    values: Iterable[Any] | None,
    *,
    iters: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float | int]:
    """Bootstrap percentile confidence interval for a mean.

    Returns
    -------
    dict with keys

    ``mean``
        Mean of the finite observations; NaN if there are none.
    ``se``
        Bootstrap standard error (standard deviation of the resample means);
        NaN when ``n < 2``.
    ``ci_low`` / ``ci_high``
        ``alpha/2`` and ``1 - alpha/2`` percentiles of the resample means;
        NaN when ``n < 2`` (a one-point CI would overstate precision).
    ``n``
        Number of finite observations actually used.
    """
    x = _finite(_as_float_array(values))
    n = int(x.size)
    if n == 0:
        return {"mean": NAN, "se": NAN, "ci_low": NAN, "ci_high": NAN, "n": 0}
    mean = float(x.mean())
    if n == 1:
        return {"mean": mean, "se": NAN, "ci_low": NAN, "ci_high": NAN, "n": 1}
    means = _bootstrap_means(x, iters=max(1, int(iters)), seed=seed)
    low, high = np.percentile(means, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return {
        "mean": mean,
        "se": float(means.std(ddof=1)),
        "ci_low": float(low),
        "ci_high": float(high),
        "n": n,
    }


# ---------------------------------------------------------------------------
# Paired difference: bootstrap CI + sign-flip permutation p-value
# ---------------------------------------------------------------------------


def _sign_flip_p(d: np.ndarray, *, iters: int, seed: int) -> tuple[float, bool]:
    """Two-sided paired permutation p-value; returns ``(p, was_exact)``.

    The null is that the sign of each within-question difference is arbitrary,
    so the reference distribution is the mean of ``d`` under all (or many)
    sign vectors. Exact enumeration for ``n <= EXACT_PERMUTATION_MAX_N``,
    Monte-Carlo otherwise.
    """
    n = int(d.size)
    if n == 0:
        return NAN, False
    observed = abs(float(d.mean()))
    tol = 1e-12 * max(1.0, observed)

    if n <= EXACT_PERMUTATION_MAX_N:
        total = 1 << n
        bits = np.arange(n, dtype=np.int64)
        per_chunk = max(1, min(total, max(1, _CHUNK_CELLS // n)))
        count = 0
        for start in range(0, total, per_chunk):
            codes = np.arange(start, min(start + per_chunk, total), dtype=np.int64)[:, None]
            signs = 1.0 - 2.0 * ((codes >> bits) & 1).astype(float)
            count += int(np.count_nonzero(np.abs(signs @ d) / n >= observed - tol))
        return count / total, True

    draws = max(int(iters), MIN_PERMUTATION_DRAWS)
    rng = np.random.default_rng(seed)
    per_chunk = max(1, min(draws, max(1, _CHUNK_CELLS // n)))
    count = 0
    done = 0
    while done < draws:
        take = min(per_chunk, draws - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(take, n))
        count += int(np.count_nonzero(np.abs(signs @ d) / n >= observed - tol))
        done += take
    # +1 in both terms keeps the p-value from ever being exactly 0.
    return (count + 1) / (draws + 1), False


def paired_diff(
    a_values: Iterable[Any] | None,
    b_values: Iterable[Any] | None,
    *,
    iters: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired comparison of ``a`` against ``b`` (positive means ``a`` is larger).

    Inputs are aligned by position; a pair is dropped if either side is
    non-finite, or if the two sequences differ in length beyond the common
    prefix.

    Returns
    -------
    dict with keys

    ``mean_diff``
        Mean of ``a_i - b_i`` over surviving pairs; NaN if none survive.
    ``se``
        Bootstrap standard error of ``mean_diff``; NaN when ``n < 2``.
    ``ci_low`` / ``ci_high``
        Bootstrap percentile CI of ``mean_diff``; NaN when ``n < 2``.
    ``n``
        Number of surviving pairs.
    ``p_value``
        Two-sided sign-flip permutation p-value; NaN when ``n == 0``.
    ``n_dropped``
        Pairs discarded for non-finite values or length mismatch.
    ``exact``
        True when ``p_value`` came from full enumeration rather than sampling.
    """
    a = _as_float_array(a_values)
    b = _as_float_array(b_values)
    m = min(a.size, b.size)
    n_len_mismatch = int(max(a.size, b.size) - m)
    a, b = a[:m], b[:m]
    keep = np.isfinite(a) & np.isfinite(b)
    d = a[keep] - b[keep]
    n = int(d.size)
    n_dropped = int(m - n) + n_len_mismatch

    out: dict[str, Any] = {
        "mean_diff": NAN,
        "se": NAN,
        "ci_low": NAN,
        "ci_high": NAN,
        "n": n,
        "p_value": NAN,
        "n_dropped": n_dropped,
        "exact": False,
    }
    if n == 0:
        return out
    out["mean_diff"] = float(d.mean())
    p, exact = _sign_flip_p(d, iters=max(1, int(iters)), seed=seed)
    out["p_value"] = p
    out["exact"] = exact
    if n == 1:
        return out
    means = _bootstrap_means(d, iters=max(1, int(iters)), seed=seed + 1)
    low, high = np.percentile(means, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    out["se"] = float(means.std(ddof=1))
    out["ci_low"] = float(low)
    out["ci_high"] = float(high)
    return out


def wilcoxon_signed_rank(
    a: Iterable[Any] | None, b: Iterable[Any] | None
) -> dict[str, Any]:
    """Wilcoxon signed-rank test, normal approximation with tie correction.

    This is an *approximation*: the p-value comes from the asymptotic normal
    distribution of ``W+`` with a continuity correction and the standard tie
    adjustment to the variance, not from the exact rank distribution. It is
    unreliable below roughly ``n = 10`` non-zero differences; treat
    :func:`paired_diff`'s permutation p-value as primary and this as a
    distribution-free cross-check.

    Returns
    -------
    dict with keys

    ``statistic``
        ``W+``, the sum of ranks of the positive differences.
    ``w_minus``
        Sum of ranks of the negative differences.
    ``z``
        Tie-corrected, continuity-corrected normal deviate; NaN if undefined.
    ``p_value``
        Two-sided p-value; 1.0 when every difference is zero.
    ``n``
        Number of non-zero differences entering the statistic.
    ``n_zero``
        Number of exactly-zero differences, which are discarded.
    ``n_dropped``
        Pairs discarded for non-finite values or length mismatch.
    """
    x = _as_float_array(a)
    y = _as_float_array(b)
    m = min(x.size, y.size)
    n_len_mismatch = int(max(x.size, y.size) - m)
    x, y = x[:m], y[:m]
    keep = np.isfinite(x) & np.isfinite(y)
    d = x[keep] - y[keep]
    n_dropped = int(m - d.size) + n_len_mismatch

    nonzero = d[d != 0.0]
    n_zero = int(d.size - nonzero.size)
    n = int(nonzero.size)
    out: dict[str, Any] = {
        "statistic": NAN,
        "w_minus": NAN,
        "z": NAN,
        "p_value": NAN,
        "n": n,
        "n_zero": n_zero,
        "n_dropped": n_dropped,
    }
    if n == 0:
        out.update({"statistic": 0.0, "w_minus": 0.0, "p_value": 1.0})
        return out

    ranks = rankdata(np.abs(nonzero))
    w_plus = float(ranks[nonzero > 0].sum())
    w_minus = float(ranks[nonzero < 0].sum())
    out["statistic"] = w_plus
    out["w_minus"] = w_minus

    mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    _, tie_counts = np.unique(np.abs(nonzero), return_counts=True)
    var -= float(((tie_counts**3 - tie_counts).sum())) / 48.0
    if var <= 0:
        return out
    correction = 0.5 * np.sign(w_plus - mu)
    z = (w_plus - mu - correction) / math.sqrt(var)
    out["z"] = float(z)
    out["p_value"] = _normal_two_sided_p(float(z))
    return out


# ---------------------------------------------------------------------------
# Pairing per-question rows across rubric sources
# ---------------------------------------------------------------------------


@dataclass
class PairedTable:
    """A ``(question, source)`` matrix of one metric, restricted to complete rows.

    Attributes
    ----------
    keys:
        Question ids retained, in first-seen order.
    groups:
        Group labels (rubric sources), in first-seen order.
    matrix:
        ``float64`` array of shape ``(len(keys), len(groups))``; all entries finite.
    dropped_keys:
        Question ids that were missing at least one group or carried a
        non-finite value.
    n_dropped:
        ``len(dropped_keys)``.
    n_rows_in:
        Number of input rows inspected.
    n_duplicates:
        Rows silently ignored because a ``(key, group)`` cell was already filled;
        the first occurrence wins.
    """

    keys: list[str] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)
    matrix: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
    dropped_keys: list[str] = field(default_factory=list)
    n_dropped: int = 0
    n_rows_in: int = 0
    n_duplicates: int = 0

    def column(self, group: str) -> np.ndarray:
        """The metric values for one group, aligned with :attr:`keys`."""
        if group not in self.groups:
            return np.empty(0, dtype=float)
        return self.matrix[:, self.groups.index(group)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "keys": list(self.keys),
            "groups": list(self.groups),
            "matrix": [[float(v) for v in row] for row in self.matrix],
            "n_paired": len(self.keys),
            "n_dropped": self.n_dropped,
            "dropped_keys": list(self.dropped_keys),
            "n_rows_in": self.n_rows_in,
            "n_duplicates": self.n_duplicates,
        }


def paired_table(
    rows: Sequence[Mapping[str, Any]] | None,
    *,
    key: str = "uid",
    group: str = "rubric_source",
    value: str = "value",
) -> PairedTable:
    """Align per-question rows into a complete ``(key, group)`` matrix.

    Keys missing from any group — or carrying a non-finite value in any group —
    are dropped, because an unbalanced paired test is not a paired test. The
    count and identity of the dropped keys are reported so the caller can
    disclose them.
    """
    table = PairedTable()
    if not rows:
        return table

    cells: dict[str, dict[str, float]] = {}
    for row in rows:
        table.n_rows_in += 1
        if not isinstance(row, Mapping) or key not in row or group not in row:
            continue
        k, g = str(row[key]), str(row[group])
        if g not in table.groups:
            table.groups.append(g)
        bucket = cells.setdefault(k, {})
        if g in bucket:
            table.n_duplicates += 1
            continue
        bucket[g] = _as_float_array([row.get(value)])[0]

    groups = table.groups
    kept_keys: list[str] = []
    kept_rows: list[list[float]] = []
    for k, bucket in cells.items():
        values = [bucket.get(g, NAN) for g in groups]
        if all(np.isfinite(v) for v in values) and len(bucket) == len(groups):
            kept_keys.append(k)
            kept_rows.append(values)
        else:
            table.dropped_keys.append(k)

    table.keys = kept_keys
    table.n_dropped = len(table.dropped_keys)
    table.matrix = (
        np.asarray(kept_rows, dtype=float)
        if kept_rows
        else np.empty((0, len(groups)), dtype=float)
    )
    return table


# ---------------------------------------------------------------------------
# Agreement, ranking, multiplicity
# ---------------------------------------------------------------------------


def cohens_kappa(rater_a: Iterable[Any] | None, rater_b: Iterable[Any] | None) -> float:
    """Cohen's kappa for two binary raters; NaN when undefined.

    Kappa is ``(p_o - p_e) / (1 - p_e)``. The denominator vanishes when the
    expected agreement is already 1 — in practice when both raters are constant
    and agree, or when one is constant and the other is constant on the same
    label. In that case chance agreement is total, kappa carries no information,
    and NaN is returned rather than a misleading 0 or 1. Note that a *single*
    constant rater facing a varying one legitimately yields kappa 0: perfect
    chance-level agreement.

    Non-finite / unconvertible entries are dropped pairwise. Values are
    thresholded at ``> 0.5`` so booleans, 0/1 ints and floats all work.
    """
    a = _as_float_array(rater_a)
    b = _as_float_array(rater_b)
    m = min(a.size, b.size)
    a, b = a[:m], b[:m]
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep] > 0.5, b[keep] > 0.5
    n = int(a.size)
    if n == 0:
        return NAN
    p_o = float(np.mean(a == b))
    pa1, pb1 = float(a.mean()), float(b.mean())
    p_e = pa1 * pb1 + (1.0 - pa1) * (1.0 - pb1)
    if 1.0 - p_e <= 1e-12:
        return NAN
    return (p_o - p_e) / (1.0 - p_e)


def auc(scores: Iterable[Any] | None, labels: Iterable[Any] | None) -> float:
    """ROC AUC via the Mann-Whitney rank statistic; NaN when either class is empty.

    ``labels`` are truthy for the positive class. Ties in ``scores`` receive
    average ranks, so a completely uninformative score gives exactly 0.5.
    """
    s = _as_float_array(scores)
    y = _as_float_array(labels)
    m = min(s.size, y.size)
    s, y = s[:m], y[:m]
    keep = np.isfinite(s) & np.isfinite(y)
    s, y = s[keep], y[keep] > 0.5
    n_pos = int(np.count_nonzero(y))
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return NAN
    ranks = rankdata(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bh_fdr(pvalues: Iterable[Any] | None) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (q-values), input order preserved.

    NaN inputs stay NaN and are excluded from the multiplicity count, so a
    metric that could not be computed does not inflate the correction. Output
    values are clipped to ``[0, 1]`` and enforced monotone in the sorted p-value
    sequence.
    """
    p = _as_float_array(pvalues)
    out = np.full(p.size, NAN, dtype=float)
    finite_idx = np.flatnonzero(np.isfinite(p))
    m = finite_idx.size
    if m == 0:
        return out.tolist()
    vals = np.clip(p[finite_idx], 0.0, 1.0)
    order = np.argsort(vals, kind="mergesort")
    ranked = vals[order] * m / np.arange(1, m + 1, dtype=float)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = np.clip(ranked, 0.0, 1.0)
    out[finite_idx] = adjusted
    return out.tolist()


# ---------------------------------------------------------------------------
# One-call convenience for the report generator
# ---------------------------------------------------------------------------


def summarize_metric(
    rows: Sequence[Mapping[str, Any]] | None,
    *,
    value: str,
    group: str = "rubric_source",
    key: str = "uid",
    baseline_group: str = "baseline",
    iters: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Marginal means plus paired contrasts against ``baseline_group``.

    Parameters mirror :func:`paired_table`: ``rows`` are per-question records,
    ``key`` identifies the question, ``group`` the rubric source, ``value`` the
    metric column.

    Returns
    -------
    dict with keys

    ``metric`` / ``baseline_group`` / ``groups``
        Echo of the request: metric column name, the reference group, and the
        group labels found (first-seen order).
    ``n_paired``
        Questions present with a finite value in *every* group.
    ``n_dropped`` / ``dropped_keys``
        Questions excluded from the paired analysis, and their ids.
    ``baseline_present``
        False when ``baseline_group`` never appears; all ``vs_baseline`` entries
        are then None.
    ``per_group``
        One entry per group:

        * ``n``, ``mean``, ``se``, ``ci_low``, ``ci_high`` — marginal statistics
          over *all* rows of that group (see :func:`mean_ci`).
        * ``paired_mean`` — mean over the ``n_paired`` complete questions only;
          this is what the contrasts are computed from.
        * ``vs_baseline`` — :func:`paired_diff` output (``mean_diff`` positive
          means the group beats the baseline), or None for the baseline itself
          and when pairing is impossible.
        * ``wilcoxon`` — :func:`wilcoxon_signed_rank` cross-check, or None.
    ``p_values``
        ``{group: permutation p-value}`` for the non-baseline groups, so the
        caller can feed several metrics at once to :func:`bh_fdr`.
    """
    rows = list(rows or [])
    table = paired_table(rows, key=key, group=group, value=value)

    by_group_all: dict[str, list[Any]] = {}
    for row in rows:
        if isinstance(row, Mapping) and group in row:
            by_group_all.setdefault(str(row[group]), []).append(row.get(value))
    for g in table.groups:
        by_group_all.setdefault(g, [])

    groups = list(table.groups) or sorted(by_group_all)
    baseline_present = baseline_group in groups
    baseline_col = (
        table.column(baseline_group)
        if baseline_present and baseline_group in table.groups
        else np.empty(0, dtype=float)
    )

    per_group: dict[str, Any] = {}
    p_values: dict[str, float] = {}
    for g in groups:
        entry: dict[str, Any] = dict(
            mean_ci(by_group_all.get(g, []), iters=iters, alpha=alpha, seed=seed)
        )
        col = table.column(g)
        entry["paired_mean"] = float(col.mean()) if col.size else NAN
        entry["vs_baseline"] = None
        entry["wilcoxon"] = None
        if baseline_present and g != baseline_group and col.size and baseline_col.size:
            diff = paired_diff(col, baseline_col, iters=iters, alpha=alpha, seed=seed)
            entry["vs_baseline"] = diff
            entry["wilcoxon"] = wilcoxon_signed_rank(col, baseline_col)
            p_values[g] = float(diff["p_value"])
        per_group[g] = entry

    return {
        "metric": value,
        "baseline_group": baseline_group,
        "baseline_present": baseline_present,
        "groups": groups,
        "n_paired": len(table.keys),
        "n_dropped": table.n_dropped,
        "dropped_keys": list(table.dropped_keys),
        "per_group": per_group,
        "p_values": p_values,
    }


# ---------------------------------------------------------------------------
# Self-test — synthetic data with known answers
# ---------------------------------------------------------------------------


def _selftest() -> int:  # pragma: no cover - developer entry point
    failures: list[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
        if not condition:
            failures.append(name)

    rng = np.random.default_rng(0)

    # rankdata: ties get the average rank.
    r = rankdata(np.array([10.0, 20.0, 20.0, 30.0]))
    check("rankdata ties averaged", np.allclose(r, [1.0, 2.5, 2.5, 4.0]), str(r.tolist()))

    # mean_ci: CI must bracket the true mean of a tight sample.
    x = rng.normal(5.0, 1.0, size=200)
    ci = mean_ci(x, iters=2000, seed=1)
    check(
        "mean_ci brackets truth",
        ci["ci_low"] < 5.0 < ci["ci_high"] and abs(ci["mean"] - x.mean()) < 1e-12,
        f"mean={ci['mean']:.3f} ci=[{ci['ci_low']:.3f},{ci['ci_high']:.3f}] n={ci['n']}",
    )
    check("mean_ci n=0", mean_ci([])["n"] == 0 and math.isnan(mean_ci([])["mean"]))
    one = mean_ci([3.0])
    check("mean_ci n=1", one["mean"] == 3.0 and math.isnan(one["ci_low"]))
    const = mean_ci([2.0] * 8, iters=500, seed=0)
    check("mean_ci constant", const["se"] == 0.0 and const["ci_low"] == 2.0)
    check("mean_ci all-NaN", math.isnan(mean_ci([NAN, NAN])["mean"]))

    # paired_diff: recover a known +0.30 shift; CI must exclude 0, p must be small.
    base = rng.normal(0.5, 0.1, size=40)
    shifted = base + 0.30
    pd_res = paired_diff(shifted, base, iters=4000, seed=2)
    check(
        "paired_diff recovers +0.30 shift",
        abs(pd_res["mean_diff"] - 0.30) < 1e-9 and pd_res["ci_low"] > 0.0,
        f"diff={pd_res['mean_diff']:.4f} ci=[{pd_res['ci_low']:.4f},{pd_res['ci_high']:.4f}] "
        f"p={pd_res['p_value']:.2e} exact={pd_res['exact']}",
    )
    check("paired_diff constant shift p tiny", pd_res["p_value"] < 0.001)

    # Same shift with noise, so the CI has non-zero width and must still cover 0.30.
    noisy = base + 0.30 + rng.normal(0.0, 0.05, size=40)
    pd_noisy = paired_diff(noisy, base, iters=4000, seed=6)
    check(
        "paired_diff noisy shift CI covers 0.30",
        pd_noisy["ci_low"] < 0.30 < pd_noisy["ci_high"] and pd_noisy["se"] > 0.0,
        f"diff={pd_noisy['mean_diff']:.4f} ci=[{pd_noisy['ci_low']:.4f},"
        f"{pd_noisy['ci_high']:.4f}] se={pd_noisy['se']:.4f} p={pd_noisy['p_value']:.2e}",
    )

    # Exact enumeration: n=5, all differences positive => p = 2/2**5.
    exact = paired_diff([1.0, 2.0, 3.0, 4.0, 5.0], [0.0] * 5, seed=3)
    check(
        "paired_diff exact enumeration",
        exact["exact"] and abs(exact["p_value"] - 2.0 / 32.0) < 1e-12,
        f"p={exact['p_value']:.6f} expected={2/32:.6f}",
    )
    null = paired_diff([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], seed=4)
    check("paired_diff zero difference", null["mean_diff"] == 0.0 and null["p_value"] == 1.0)
    check("paired_diff n=0", paired_diff([], [])["n"] == 0)
    check("paired_diff n=1", paired_diff([1.0], [0.0])["n"] == 1)
    check(
        "paired_diff drops non-finite pairwise",
        paired_diff([1.0, NAN, 3.0], [0.0, 0.0, 0.0])["n"] == 2,
    )
    big = paired_diff(base + 0.3, base, iters=10_000, seed=5)
    check("paired_diff monte-carlo path", not big["exact"] and big["n"] == 40)

    # Wilcoxon: textbook one-sided-shift example, and the all-ties guard.
    w = wilcoxon_signed_rank([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], [0.0] * 10)
    check(
        "wilcoxon all-positive => W+ = n(n+1)/2",
        w["statistic"] == 55.0 and w["p_value"] < 0.01,
        f"W+={w['statistic']} z={w['z']:.3f} p={w['p_value']:.4f}",
    )
    wt = wilcoxon_signed_rank([1.0, 1.0], [1.0, 1.0])
    check("wilcoxon all ties", wt["p_value"] == 1.0 and wt["n"] == 0)
    check("wilcoxon n=0", math.isnan(wilcoxon_signed_rank([], [])["z"]))

    # Cohen's kappa.
    labels = [1, 0, 1, 1, 0, 0, 1, 0]
    check("kappa identical raters == 1", cohens_kappa(labels, labels) == 1.0)
    flipped = [1 - v for v in labels]
    check(
        "kappa opposite raters == -1",
        abs(cohens_kappa(labels, flipped) + 1.0) < 1e-12,
        f"{cohens_kappa(labels, flipped):.3f}",
    )
    check("kappa both constant == NaN", math.isnan(cohens_kappa([1, 1, 1], [1, 1, 1])))
    check(
        "kappa one constant == 0",
        cohens_kappa([1, 1, 1, 1], [1, 0, 1, 0]) == 0.0,
        f"{cohens_kappa([1, 1, 1, 1], [1, 0, 1, 0])}",
    )
    check("kappa n=0 == NaN", math.isnan(cohens_kappa([], [])))

    # AUC.
    check("auc perfectly separable == 1", auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0)
    check("auc perfectly inverted == 0", auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0)
    check("auc all tied == 0.5", auc([0.5] * 6, [0, 1, 0, 1, 0, 1]) == 0.5)
    check("auc single class == NaN", math.isnan(auc([0.1, 0.2], [1, 1])))

    # Benjamini-Hochberg.
    q = bh_fdr([0.01, 0.02, 0.03, 0.04, 0.05])
    check(
        "bh_fdr monotone and >= p",
        all(qi >= pi - 1e-12 for qi, pi in zip(q, [0.01, 0.02, 0.03, 0.04, 0.05]))
        and q == sorted(q),
        str([round(v, 4) for v in q]),
    )
    check("bh_fdr smallest of 5 equal-spaced", abs(q[0] - 0.05) < 1e-12, f"{q[0]:.4f}")
    qn = bh_fdr([0.01, NAN, 0.5])
    check("bh_fdr passes NaN through", math.isnan(qn[1]) and abs(qn[0] - 0.02) < 1e-12)
    check("bh_fdr empty", bh_fdr([]) == [])

    # paired_table drops incomplete questions.
    rows = [
        {"uid": "q1", "rubric_source": "baseline", "v": 0.1},
        {"uid": "q1", "rubric_source": "agentic", "v": 0.4},
        {"uid": "q2", "rubric_source": "baseline", "v": 0.2},
        {"uid": "q2", "rubric_source": "agentic", "v": 0.5},
        {"uid": "q3", "rubric_source": "baseline", "v": 0.3},  # agentic missing
        {"uid": "q4", "rubric_source": "baseline", "v": NAN},
        {"uid": "q4", "rubric_source": "agentic", "v": 0.9},
    ]
    tab = paired_table(rows, key="uid", group="rubric_source", value="v")
    check(
        "paired_table keeps complete keys only",
        tab.keys == ["q1", "q2"] and tab.n_dropped == 2 and tab.matrix.shape == (2, 2),
        f"keys={tab.keys} dropped={tab.dropped_keys}",
    )
    check(
        "paired_table column alignment",
        np.allclose(tab.column("agentic"), [0.4, 0.5]),
        str(tab.column("agentic").tolist()),
    )

    # summarize_metric end-to-end on a known +0.25 shift.
    synth: list[dict[str, Any]] = []
    for i in range(24):
        b = float(rng.uniform(0.2, 0.6))
        synth.append({"uid": f"q{i}", "rubric_source": "baseline", "v": b})
        synth.append({"uid": f"q{i}", "rubric_source": "agentic", "v": b + 0.25})
        synth.append({"uid": f"q{i}", "rubric_source": "shipped", "v": b - 0.05})
    summary = summarize_metric(
        synth, value="v", baseline_group="baseline", iters=3000, seed=7
    )
    ag = summary["per_group"]["agentic"]["vs_baseline"]
    check(
        "summarize_metric paired diff",
        abs(ag["mean_diff"] - 0.25) < 1e-9 and ag["ci_low"] > 0 and ag["p_value"] < 0.001,
        f"diff={ag['mean_diff']:.4f} p={ag['p_value']:.2e} n_paired={summary['n_paired']}",
    )
    check(
        "summarize_metric baseline has no contrast",
        summary["per_group"]["baseline"]["vs_baseline"] is None
        and set(summary["p_values"]) == {"agentic", "shipped"},
    )
    empty = summarize_metric([], value="v")
    check("summarize_metric empty input", empty["n_paired"] == 0 and empty["per_group"] == {})
    missing_base = summarize_metric(
        [{"uid": "q1", "rubric_source": "agentic", "v": 1.0}], value="v"
    )
    check(
        "summarize_metric absent baseline",
        missing_base["baseline_present"] is False
        and missing_base["per_group"]["agentic"]["vs_baseline"] is None,
    )

    check("safe_mean all-NaN", math.isnan(safe_mean([NAN, NAN])))
    check("safe_mean mixed", safe_mean([1.0, NAN, 3.0]) == 2.0)

    print()
    if failures:
        print(f"{len(failures)} FAILURES: {failures}")
        return 1
    print("all stats self-tests passed")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_selftest())
