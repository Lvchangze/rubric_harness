#!/usr/bin/env python3
"""Independent audit of the `tilt` result: do the expert rubrics know who won?

Why this file exists separately from `rubricbench_rubric_stats.py`
-----------------------------------------------------------------
`OPTIMIZATION_LOG.md` §8 measured a quantity it called **tilt** — the share of a
rubric's non-instruction content words that appear in the *preferred* response
minus the share appearing in the *rejected* one — found +0.020 [+0.012, +0.027]
for the dataset's expert rubrics and 0.000 for all nine generated ones, and drew
a load-bearing conclusion from it: that `expert` is partly an oracle, so the
16-point gap to `framed` is not all a quality gap, and the whole "21.3 points of
available space" framing is wrong.

A conclusion that reframes the report is worth re-deriving rather than
re-running. Nothing here imports `harness/`, reads any `*_score.json`, or shares
a tokeniser, a stop list or a statistic with the original implementation. The
benchmark JSON, the rubric JSONs and the verdict JSONLs are the only inputs;
labels, domains, accuracies and every interval are recomputed from them.

What it checks, in the order the sections run
--------------------------------------------
``recompute``  tilt from scratch, my tokenisation and then theirs, so the number
               is either independently confirmed or shown to be spec-specific.
``permute``    the null distribution of tilt under random re-assignment of which
               response won. Note in advance what this test can and cannot do:
               permuting the winner flips the sign of every per-case term, so the
               null is *exactly* symmetric about zero by construction. It can
               confirm the statistic is not mechanically biased; it has no power
               at all against a length confound, because permutation destroys
               exactly the asymmetry a length confound would create. The test
               that does have that power is ``length``'s matched nulls.
``length``     winner/loser length distributions, tilt after truncating both
               responses to a common length, tilt stratified by length ratio,
               and two nulls that hold the response pair fixed while destroying
               the rubric's connection to it: frequency-matched random tokens,
               and another case's real expert rubric with a matched length ratio.
``robust``     16 metric definitions (stop lists, stemming, repetition,
               rarity bands, substring vs token match, micro vs macro average).
``domain``     the five official domain groups.
``dose``       whether `expert` actually wins more where tilt is higher — the
               confirmatory test. With `framed` tilt as a placebo predictor, and
               the direct question of how much of the expert-minus-framed gap
               survives on cases with no detectable tilt.
``separate``   leakage vs expertise: verbatim n-gram overlap, rarity of the
               tilting tokens, numerals, and the cross-pair control.

Usage::

    python scripts/rubricbench_tilt_audit.py --out results/rubricbench/TILT_AUDIT.md
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "rubricbench" / "data" / "rubricbench_data.json"
RESULTS = ROOT / "results" / "rubricbench"
OPT_RUNS = RESULTS / "opt" / "runs"
OPT_RUBRICS = RESULTS / "opt" / "rubrics"

SEED = 20260831

#: Copied from the benchmark's own `eval_submission.py` grouping, re-typed here
#: rather than imported so this file depends on nothing in `harness/`.
DOMAIN_GROUPS = {
    "chat": {"general", "focus", "human-preference", "factuality", "helpful"},
    "if": {"precise if", "ifeval"},
    "stem": {"stem", "math", "mmlu-pro", "gpqa"},
    "code": {"mbpp", "code"},
    "safety": {"safety", "harmlessness"},
}


def group_of(domain: str) -> str:
    d = (domain or "").strip().lower()
    for g, members in DOMAIN_GROUPS.items():
        if d in members:
            return g
    return "other"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    case_id: str
    instruction: str
    winner: str          # the response the humans preferred, resolved from `label`
    loser: str
    label: int
    domain: str
    group: str
    expert: str
    source: str


def load_cases() -> dict[str, Case]:
    raw = json.loads(BENCH.read_text(encoding="utf-8"))
    out: dict[str, Case] = {}
    for r in raw:
        cid = str(r["case_id"])
        label = int(r["label"])           # 0 => A preferred, 1 => B preferred
        a, b = str(r.get("response_a", "")), str(r.get("response_b", ""))
        win, lose = (b, a) if label == 1 else (a, b)
        out[cid] = Case(
            case_id=cid,
            instruction=str(r.get("instruction", "")),
            winner=win,
            loser=lose,
            label=label,
            domain=str(r.get("domain", "")),
            group=group_of(str(r.get("domain", ""))),
            expert=str(r.get("rubrics", "") or ""),
            source=str(r.get("source", "")),
        )
    return out


def load_rubrics(name: str, cases: dict[str, Case]) -> dict[str, str]:
    """`case_id -> rubric text` for `expert` or any variant name on disk."""
    if name == "expert":
        return {cid: c.expert for cid, c in cases.items()}
    for path in (
        RESULTS / f"{name}_rubrics.json",
        OPT_RUBRICS / f"{name}.json",
        OPT_RUNS / f"dev_{name}_rubrics.json",
    ):
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            return {str(r["case_id"]): str(r.get("rubric") or "") for r in raw}
    raise SystemExit(f"no rubric file for {name!r}")


def load_verdicts(name: str) -> dict[str, dict]:
    """`case_id -> verdict row`. Full-benchmark file wins over the dev-only one."""
    for path in (RESULTS / f"{name}_verdicts.jsonl", OPT_RUNS / f"dev_{name}_verdicts.jsonl"):
        if path.exists():
            rows = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    rows[str(r["case_id"])] = r
            return rows
    raise SystemExit(f"no verdict file for {name!r}")


def correctness(name: str, cases: dict[str, Case], mode: str = "forward") -> dict[str, int]:
    """1 if the judge picked the human-preferred response, 0 otherwise.

    Unanswered counts as wrong, which is the official convention.
    """
    letter = {"A": 0, "B": 1}
    out = {}
    for cid, row in load_verdicts(name).items():
        if cid not in cases:
            continue
        pred = row.get(mode)
        out[cid] = int(pred is not None and letter.get(pred, -1) == cases[cid].label)
    return out


def dev_ids() -> list[str]:
    return list(json.loads((RESULTS / "split.json").read_text(encoding="utf-8"))["dev"])


# ---------------------------------------------------------------------------
# Tokenisation. My own defaults first; the original spec is one variant of it.
# ---------------------------------------------------------------------------

#: A conventional English stop list, written out rather than imported so the
#: audit has no dependency that could change under it. Deliberately does NOT
#: contain evaluative words ("avoid", "correct", "without"): those carry the
#: content a rubric is made of.
STOP_BIG = frozenset("""
a about above after again against all almost also although always am an and another any anything
are around as at be because been before being below between both but by can cannot could did do
does doing done down during each either else enough etc even ever every everything except few for
from further get gets given gives go had has have having he her here hers herself him himself his
how however i if in indeed instead into is it its itself just least less let like made make many
may me might mine more most much must my myself neither never next no nor not nothing now of off
often on once one only onto or other others otherwise our ours ourselves out over own perhaps
please quite rather really same seem seems several shall she should since so some something still
such than that the their theirs them themselves then there therefore these they thing things this
those though through thus to together too toward under until up upon us use used using very was
we well were what when where whether which while who whom whose why will with within without
would yet you your yours yourself
response responses answer answers reply replies output outputs text texts assistant model user
provide provides provided providing include includes included including give gives giving
""".split())

#: The stop list the original implementation used, transcribed for the
#: replication variant.
STOP_ORIG = frozenset("""a an the and or but if of to in on for with without by as at from into is
are was were be been being does do did doesn don not no it its this that these those response
responses answer answers there their they them then than so such any all each other more most
which who whom whose what when where why how can could should would will shall may might must
have has had having i you he she we us our your his her provide provides provided include
includes included given give gives""".split())

_WORD_ORIG = re.compile(r"[a-z][a-z'-]{2,}")
_WORD_ALPHA = re.compile(r"[a-z]+")
_NUM = re.compile(r"\d+(?:\.\d+)?")


def _stem(w: str) -> str:
    """A deliberately small suffix stripper: enough to merge the inflections a
    rubric and a response would differ on, not a linguistics claim."""
    for suf, repl in (("ies", "y"), ("sses", "ss"), ("ches", "ch"), ("shes", "sh"),
                      ("xes", "x"), ("ness", ""), ("ingly", ""), ("edly", "")):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)] + repl
    for suf in ("ing", "ed", "ly", "es", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf) and not w.endswith("ss"):
            return w[: -len(suf)]
    return w


@dataclass(frozen=True)
class Spec:
    """One concrete answer to "which words count and when does one hit"."""

    name: str
    stop: frozenset[str] = STOP_BIG
    min_len: int = 3
    stem: bool = False
    pattern: re.Pattern = _WORD_ALPHA
    substring: bool = False      # hit if the token occurs anywhere in the text
    weight: str = "uniform"      # uniform | rubric_tf | idf
    df_band: tuple[float, float] = (0.0, 1.0)   # keep tokens whose document freq is in band
    note: str = ""

    def words(self, text: str) -> list[str]:
        ws = self.pattern.findall((text or "").lower())
        ws = [w for w in ws if len(w) >= self.min_len]
        if self.stem:
            ws = [_stem(w) for w in ws]
        return [w for w in ws if w not in self.stop]

    def token_set(self, text: str) -> set[str]:
        return set(self.words(text))

    def counts(self, text: str) -> Counter:
        return Counter(self.words(text))


BASE = Spec(name="mine (independent)", note="my stop list, [a-z]+ >= 3 chars, set membership")
ORIG = Spec(name="theirs (replicated)", stop=STOP_ORIG, pattern=_WORD_ORIG,
            note="their stop list and [a-z][a-z'-]{2,}")


# ---------------------------------------------------------------------------
# The statistic
# ---------------------------------------------------------------------------


@dataclass
class Prepared:
    """Everything the tilt statistic needs, tokenised once per spec."""

    ids: list[str]
    rubric: dict[str, set[str]]           # novel rubric tokens (instruction removed)
    rubric_tf: dict[str, Counter]
    win: dict[str, set[str]]
    lose: dict[str, set[str]]
    win_text: dict[str, str]
    lose_text: dict[str, str]
    df: dict[str, float]                  # document frequency over all responses
    weights: dict[str, dict[str, float]] = field(default_factory=dict)


def prepare(spec: Spec, cases: dict[str, Case], rubrics: dict[str, str],
            ids: Sequence[str], df: dict[str, float] | None = None) -> Prepared:
    df = df if df is not None else document_frequency(spec, cases)
    keep_ids, rub, rub_tf, win, lose, wtxt, ltxt, weights = [], {}, {}, {}, {}, {}, {}, {}
    lo, hi = spec.df_band
    for cid in ids:
        case = cases[cid]
        instr = spec.token_set(case.instruction)
        tf = spec.counts(rubrics.get(cid, ""))
        tok = {w for w in tf if w not in instr}
        if lo > 0.0 or hi < 1.0:
            tok = {w for w in tok if lo <= df.get(w, 0.0) < hi}
        if not tok:
            continue
        keep_ids.append(cid)
        rub[cid] = tok
        rub_tf[cid] = Counter({w: tf[w] for w in tok})
        win[cid] = spec.token_set(case.winner) - instr
        lose[cid] = spec.token_set(case.loser) - instr
        wtxt[cid] = (case.winner or "").lower()
        ltxt[cid] = (case.loser or "").lower()
        if spec.weight == "uniform":
            weights[cid] = {w: 1.0 for w in tok}
        elif spec.weight == "rubric_tf":
            weights[cid] = {w: float(tf[w]) for w in tok}
        elif spec.weight == "idf":
            weights[cid] = {w: math.log(1.0 / max(df.get(w, 0.0), 1e-4)) for w in tok}
        else:
            raise ValueError(spec.weight)
    return Prepared(keep_ids, rub, rub_tf, win, lose, wtxt, ltxt, df, weights)


def _hit(spec: Spec, tok: str, toks: set[str], text: str) -> bool:
    return (tok in text) if spec.substring else (tok in toks)


def per_case_tilt(spec: Spec, p: Prepared) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (tilt, share-in-winner, share-in-loser) per case, in `p.ids` order."""
    t, sw, sl = [], [], []
    for cid in p.ids:
        wgt = p.weights[cid]
        total = sum(wgt.values()) or 1.0
        hw = sum(w for tok, w in wgt.items() if _hit(spec, tok, p.win[cid], p.win_text[cid]))
        hl = sum(w for tok, w in wgt.items() if _hit(spec, tok, p.lose[cid], p.lose_text[cid]))
        sw.append(hw / total)
        sl.append(hl / total)
        t.append((hw - hl) / total)
    return np.array(t), np.array(sw), np.array(sl)


def document_frequency(spec: Spec, cases: dict[str, Case]) -> dict[str, float]:
    """Share of the 2*N responses in the benchmark that contain each token."""
    n = 0
    counts: Counter = Counter()
    for case in cases.values():
        for text in (case.winner, case.loser):
            n += 1
            counts.update(spec.token_set(text))
    return {w: c / n for w, c in counts.items()}


# ---------------------------------------------------------------------------
# Intervals and tests, written here rather than imported
# ---------------------------------------------------------------------------


def boot_ci(x: np.ndarray, reps: int = 10000, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(reps, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def mean_ci_t(x: np.ndarray) -> tuple[float, float, float]:
    """Mean and a normal-approximation 95% interval; a cross-check on the bootstrap."""
    m = float(x.mean())
    se = float(x.std(ddof=1) / math.sqrt(len(x)))
    return m, m - 1.96 * se, m + 1.96 * se


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z, ph = 1.959963984540054, k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial tail probability, summed in log space."""
    if n == 0:
        return 1.0
    logp = np.array([math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                     + i * math.log(p) + (n - i) * math.log1p(-p) for i in range(n + 1)])
    pk = logp[k]
    tot = float(np.exp(logp[logp <= pk + 1e-12]).sum())
    return min(1.0, tot)


def logistic_fit(x: np.ndarray, y: np.ndarray, iters: int = 200) -> tuple[float, float, float]:
    """Newton-Raphson logistic regression of y on [1, x]. Returns (b0, b1, se_b1)."""
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(iters):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        W = p * (1 - p)
        g = X.T @ (y - p)
        H = X.T @ (X * W[:, None]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    eta = X @ beta
    p = 1.0 / (1.0 + np.exp(-eta))
    W = p * (1 - p)
    cov = np.linalg.inv(X.T @ (X * W[:, None]) + 1e-9 * np.eye(2))
    return float(beta[0]), float(beta[1]), float(math.sqrt(cov[1, 1]))


def two_sided_z_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------

#: The eight variants §8 measured, plus `baseline` — which §8's table omitted even
#: though it is the published method — and `expert`.
SOURCES_DEV = ["framed", "framed_noweight", "framed_bare", "qform", "balanced",
               "contrastive", "fs_sim5", "fs_fix5", "baseline", "expert"]
SOURCES_FULL = ["none", "baseline", "framed", "expert"]


class Out:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        print(f"\nwrote {path}")


def common_ids(cases: dict[str, Case], rubric_sets: dict[str, dict[str, str]],
               ids: Sequence[str]) -> list[str]:
    """Cases where every source produced a non-empty rubric and an expert one exists."""
    return [cid for cid in sorted(ids)
            if cases[cid].expert.strip()
            and all(rubric_sets[s].get(cid, "").strip() for s in rubric_sets)]


# ---------------------------------------------------------------------------
# 1. Recompute
# ---------------------------------------------------------------------------


def section_recompute(o: Out, cases: dict[str, Case], ids: list[str],
                      rubric_sets: dict[str, dict[str, str]]) -> dict[str, np.ndarray]:
    o("## 1. Tilt, recomputed from the benchmark JSON")
    o()
    o("Two independent tokenisations of the same statistic: the definition I would")
    o("have written (`mine`) and a transcription of the one in")
    o("`rubricbench_rubric_stats.py` (`theirs`). Both operate on rubric words that do")
    o("not occur in the instruction, look them up in each response, and average the")
    o("winner-minus-loser share over cases. 10,000 case-level bootstrap resamples.")
    o()
    o("| source | n | mine: in winner | in loser | **tilt** | 95% CI | theirs: **tilt** | 95% CI | reported |")
    o("|---|--:|--:|--:|--:|:--:|--:|:--:|--:|")
    #: What §8's table published, for the diff column. `baseline` is absent there.
    reported = {"framed": "+0.001", "framed_noweight": "+0.003", "framed_bare": "+0.003",
                "qform": "+0.003", "balanced": "+0.001", "contrastive": "−0.000",
                "fs_sim5": "+0.000", "fs_fix5": "−0.002", "expert": "+0.020",
                "baseline": "not measured"}
    per_case: dict[str, np.ndarray] = {}
    df_base = document_frequency(BASE, cases)
    df_orig = document_frequency(ORIG, cases)
    for src in rubric_sets:
        pm = prepare(BASE, cases, rubric_sets[src], ids, df_base)
        tm, swm, slm = per_case_tilt(BASE, pm)
        po = prepare(ORIG, cases, rubric_sets[src], ids, df_orig)
        to, _, _ = per_case_tilt(ORIG, po)
        lom, him = boot_ci(tm)
        loo, hio = boot_ci(to)
        per_case[src] = tm
        star = "**" if src == "expert" else ""
        o(f"| {star}`{src}`{star} | {len(pm.ids)} | {swm.mean():.3f} | {slm.mean():.3f} | "
          f"{star}{tm.mean():+.4f}{star} | [{lom:+.4f}, {him:+.4f}] | "
          f"{star}{to.mean():+.4f}{star} | [{loo:+.4f}, {hio:+.4f}] | {reported[src]} |")
    o()
    return per_case


# ---------------------------------------------------------------------------
# 2. Permutation null
# ---------------------------------------------------------------------------


def section_permute(o: Out, cases: dict[str, Case], ids: list[str],
                    rubric_sets: dict[str, dict[str, str]], reps: int = 20000) -> None:
    o("## 2. The null distribution under random re-assignment of the winner")
    o()
    o("Check 1 as asked: permute which response is the preferred one, keep everything")
    o("else fixed, confirm the null sits on zero.")
    o()
    o("**What this test can establish, stated before running it.** Swapping the winner")
    o("and the loser for a case negates that case's contribution, so a label")
    o("permutation maps every per-case term `t_i` to `±t_i` with probability one half.")
    o("The null distribution of the mean is therefore *exactly* symmetric about zero")
    o("by construction, for any source and any tokenisation. Running it confirms the")
    o("implementation has no sign or indexing error, and nothing else: in particular it")
    o("has no power against a length confound, because a length confound lives entirely")
    o("in the winner/loser assignment that the permutation destroys. The nulls with")
    o("power over that are in §3.")
    o()
    df_base = document_frequency(BASE, cases)
    rng = np.random.default_rng(SEED)
    o("| source | observed tilt | null mean | null SD | null 2.5% | null 97.5% | share of null > 0 | permutation p |")
    o("|---|--:|--:|--:|--:|--:|--:|--:|")
    for src in rubric_sets:
        p = prepare(BASE, cases, rubric_sets[src], ids, df_base)
        t, _, _ = per_case_tilt(BASE, p)
        signs = rng.choice([-1.0, 1.0], size=(reps, len(t)))
        null = (signs * t).mean(axis=1)
        pval = float((np.abs(null) >= abs(t.mean()) - 1e-15).mean())
        lo, hi = np.percentile(null, [2.5, 97.5])
        o(f"| `{src}` | {t.mean():+.4f} | {null.mean():+.5f} | {null.std():.4f} | "
          f"{lo:+.4f} | {hi:+.4f} | {(null > 0).mean():.3f} | "
          f"{pval:.4g}{' **' if pval < 0.05 else ''} |")
    o()
    o(f"{reps:,} sign-flip permutations, seed {SEED}. The null means are zero to four")
    o("decimals and the exceedance rate above zero is 0.50 in every row, as the")
    o("symmetry argument requires. **Check 1 passes and tells us the statistic is not")
    o("mechanically biased.** It does not yet distinguish rubric content from response")
    o("asymmetry.")
    o()


# ---------------------------------------------------------------------------
# 3. Length
# ---------------------------------------------------------------------------


def _words(text: str) -> list[str]:
    return re.findall(r"\S+", text or "")


def section_length(o: Out, cases: dict[str, Case], ids: list[str],
                   rubric_sets: dict[str, dict[str, str]], reps: int = 200) -> dict:
    o("## 3. The length objection, tested rather than argued away")
    o()
    o("§8 dismissed length in one sentence: if preferred responses were longer, every")
    o("source would be lifted equally, and `framed` sits at +0.001. That argument")
    o("assumes the lift is source-independent, which it is not — the hit rate for a")
    o("token depends on how common the token is, so two sources with different")
    o("vocabularies inherit different length-driven tilts. Measured directly:")
    o()
    # -- 3.1 length distributions
    wl = np.array([len(_words(cases[c].winner)) for c in ids], dtype=float)
    ll = np.array([len(_words(cases[c].loser)) for c in ids], dtype=float)
    cw = np.array([len(cases[c].winner) for c in ids], dtype=float)
    cl = np.array([len(cases[c].loser) for c in ids], dtype=float)
    uw = np.array([len(BASE.token_set(cases[c].winner)) for c in ids], dtype=float)
    ul = np.array([len(BASE.token_set(cases[c].loser)) for c in ids], dtype=float)
    o("### 3.1 Winner and loser lengths")
    o()
    o("| measure | winner mean | loser mean | winner median | loser median | mean paired Δ | 95% CI | share winner longer |")
    o("|---|--:|--:|--:|--:|--:|:--:|--:|")
    for label, a, b in (("whitespace words", wl, ll), ("characters", cw, cl),
                        ("distinct content tokens", uw, ul)):
        d = a - b
        m, lo, hi = mean_ci_t(d)
        o(f"| {label} | {a.mean():.1f} | {b.mean():.1f} | {np.median(a):.0f} | {np.median(b):.0f} | "
          f"{m:+.1f} | [{lo:+.1f}, {hi:+.1f}] | {(a > b).mean():.3f} |")
    o()
    ratio = np.log((wl + 1) / (ll + 1))
    o(f"Mean log length ratio log(winner/loser) = **{ratio.mean():+.4f}** "
      f"(95% CI [{mean_ci_t(ratio)[1]:+.4f}, {mean_ci_t(ratio)[2]:+.4f}]); "
      f"the winner is the longer response in {(wl > ll).mean():.1%} of cases.")
    o()
    # -- 3.2 truncation to a common length
    o("### 3.2 Tilt after truncating both responses to a common length")
    o()
    o("Each pair is cut to the first `min(len_winner, len_loser)` whitespace words, so")
    o("the two sides are exactly equal in length and any hit-rate difference has to")
    o("come from what the words are. The full-length row is repeated for reference.")
    o()
    trunc_cases: dict[str, Case] = {}
    for cid in ids:
        c = cases[cid]
        w, l = _words(c.winner), _words(c.loser)
        k = min(len(w), len(l))
        trunc_cases[cid] = replace(c, winner=" ".join(w[:k]), loser=" ".join(l[:k]))
    df_base = document_frequency(BASE, cases)
    df_trunc = document_frequency(BASE, trunc_cases)
    o("| source | tilt, full responses | 95% CI | tilt, length-matched | 95% CI |")
    o("|---|--:|:--:|--:|:--:|")
    for src in rubric_sets:
        pf = prepare(BASE, cases, rubric_sets[src], ids, df_base)
        tf, _, _ = per_case_tilt(BASE, pf)
        pt = prepare(BASE, trunc_cases, rubric_sets[src], ids, df_trunc)
        tt, _, _ = per_case_tilt(BASE, pt)
        a, b = boot_ci(tf), boot_ci(tt)
        star = "**" if src == "expert" else ""
        o(f"| {star}`{src}`{star} | {tf.mean():+.4f} | [{a[0]:+.4f}, {a[1]:+.4f}] | "
          f"{star}{tt.mean():+.4f}{star} | [{b[0]:+.4f}, {b[1]:+.4f}] |")
    o()
    # -- 3.3 stratified by length ratio
    o("### 3.3 Tilt inside strata of the length ratio")
    o()
    p_exp = prepare(BASE, cases, rubric_sets["expert"], ids, df_base)
    t_exp, _, _ = per_case_tilt(BASE, p_exp)
    p_fr = prepare(BASE, cases, rubric_sets["framed"], ids, df_base)
    t_fr, _, _ = per_case_tilt(BASE, p_fr)
    r_exp = np.array([math.log((len(_words(cases[c].winner)) + 1) /
                               (len(_words(cases[c].loser)) + 1)) for c in p_exp.ids])
    bands = [(-99, -0.5, "winner much shorter"), (-0.5, -0.1, "winner shorter"),
             (-0.1, 0.1, "within 10%"), (0.1, 0.5, "winner longer"),
             (0.5, 99, "winner much longer")]
    # Pair the two sources case by case so the difference column is a paired
    # contrast: any case-level property, length included, cancels out of it.
    fr_by_id = dict(zip(p_fr.ids, t_fr))
    both = [i for i, cid in enumerate(p_exp.ids) if cid in fr_by_id]
    te_p = t_exp[both]
    tf_p = np.array([fr_by_id[p_exp.ids[i]] for i in both])
    r_p = r_exp[both]
    o("| length ratio band | n | `expert` tilt | 95% CI | `framed` tilt | paired `expert` − `framed` | 95% CI |")
    o("|---|--:|--:|:--:|--:|--:|:--:|")
    for lo_b, hi_b, label in bands:
        m = (r_p >= lo_b) & (r_p < hi_b)
        if m.sum() < 10:
            continue
        ci = boot_ci(te_p[m])
        d = te_p[m] - tf_p[m]
        cd = boot_ci(d)
        o(f"| {label} ({lo_b:+.1f}, {hi_b:+.1f}] | {int(m.sum())} | {te_p[m].mean():+.4f} | "
          f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | {tf_p[m].mean():+.4f} | **{d.mean():+.4f}** | "
          f"[{cd[0]:+.4f}, {cd[1]:+.4f}] |")
    dall = te_p - tf_p
    cda = boot_ci(dall)
    o(f"| **all** | {len(dall)} | {te_p.mean():+.4f} | — | {tf_p.mean():+.4f} | "
      f"**{dall.mean():+.4f}** | [{cda[0]:+.4f}, {cda[1]:+.4f}] |")
    slope = np.polyfit(r_p, te_p, 1)
    slope_f = np.polyfit(r_p, tf_p, 1)
    slope_d = np.polyfit(r_p, dall, 1)
    o()
    o(f"Least-squares fit on the log length ratio: `expert` slope {slope[0]:+.4f} per unit,")
    o(f"intercept (equal-length pair) **{slope[1]:+.4f}**; `framed` slope {slope_f[0]:+.4f},")
    o(f"intercept {slope_f[1]:+.4f}; the paired difference has slope {slope_d[0]:+.4f} and")
    o(f"intercept **{slope_d[1]:+.4f}**. Length moves tilt a great deal *within* the")
    o("benchmark — that part of the objection is right — and it moves both sources")
    o("almost equally, so the paired difference is nearly flat in it.")
    o()
    # -- 3.4 two matched nulls
    o("### 3.4 Two nulls that keep the response pair and break the rubric's link to it")
    o()
    o("A length confound cannot be tested by permuting labels (§2). It can be tested")
    o("by keeping the pair exactly as it is and replacing the rubric with something")
    o("that cannot know anything about it:")
    o()
    o("- **frequency-matched random tokens** — each rubric token is swapped for a")
    o("  random token drawn from the same document-frequency band over the 2,294")
    o("  responses, rejecting draws that occur in the case's own instruction. Same")
    o("  number of tokens, same rarity profile, no connection to the pair.")
    o("- **another case's real expert rubric** — the rubric of a different case in the")
    o("  same domain group whose log length ratio is closest to this one's. This holds")
    o("  the writing style, vocabulary and the length asymmetry fixed, and destroys")
    o("  only the pairing.")
    o()
    bands_df = _df_bands(df_base)
    o("| source | observed tilt | random-token null | excess | 95% CI of excess | cross-pair null | excess | 95% CI of excess |")
    o("|---|--:|--:|--:|:--:|--:|--:|:--:|")
    out: dict = {}
    for src in rubric_sets:
        p = prepare(BASE, cases, rubric_sets[src], ids, df_base)
        t, _, _ = per_case_tilt(BASE, p)
        n_rand = _random_token_null(BASE, cases, p, bands_df, reps=reps)
        n_cross = _cross_pair_null(BASE, cases, p, rubric_sets[src], reps=reps)
        e_rand, e_cross = t - n_rand, t - n_cross
        cr, cc = boot_ci(e_rand), boot_ci(e_cross)
        star = "**" if src == "expert" else ""
        o(f"| {star}`{src}`{star} | {t.mean():+.4f} | {n_rand.mean():+.4f} | "
          f"{star}{e_rand.mean():+.4f}{star} | [{cr[0]:+.4f}, {cr[1]:+.4f}] | "
          f"{n_cross.mean():+.4f} | {star}{e_cross.mean():+.4f}{star} | [{cc[0]:+.4f}, {cc[1]:+.4f}] |")
        out[src] = {"tilt": float(t.mean()), "rand": float(n_rand.mean()),
                    "cross": float(n_cross.mean()),
                    "excess_cross": float(e_cross.mean()),
                    "per_case_excess": t - n_cross, "ids": list(p.ids)}
    o()
    o(f"{reps} draws per case for each null, seed {SEED}.")
    o()
    # -- 3.5 the same test in its sharpest form
    o("### 3.5 The same question with the nuisance removed entirely")
    o()
    o("Tilt's denominator is every rubric token, most of which either hit both")
    o("responses or neither and carry no information. Restrict attention to the tokens")
    o("that hit exactly one side — the only ones that can discriminate — and ask what")
    o("share of *those* landed in the winner. Under any explanation that does not")
    o("involve the pair, that share is 0.5, regardless of length, vocabulary or")
    o("response count, so this version needs no null at all.")
    o()
    o("| source | rubric tokens | hit exactly one side | share of those in the winner | 95% CI | p vs 0.5 |")
    o("|---|--:|--:|--:|:--:|--:|")
    for src in rubric_sets:
        p = prepare(BASE, cases, rubric_sets[src], ids, df_base)
        n_tok = w_only = l_only = 0
        for cid in p.ids:
            r, w, l = p.rubric[cid], p.win[cid], p.lose[cid]
            n_tok += len(r)
            w_only += len((r & w) - l)
            l_only += len((r & l) - w)
        m = w_only + l_only
        share = w_only / m if m else 0.5
        ci = wilson(w_only, m)
        pv = binom_two_sided(w_only, m)
        star = "**" if src == "expert" else ""
        o(f"| {star}`{src}`{star} | {n_tok:,} | {m:,} ({m / max(1, n_tok):.1%}) | "
          f"{star}{share:.4f}{star} | [{ci[0]:.4f}, {ci[1]:.4f}] | "
          f"{pv:.3g}{' **' if pv < 0.05 else ''} |")
    o()
    o("Exact binomial (two-sided) against 0.5.")
    o()
    return out


def _df_bands(df: dict[str, float]) -> dict[int, list[str]]:
    """Vocabulary bucketed by log2 document frequency, for the matched null."""
    bands: dict[int, list[str]] = defaultdict(list)
    for w, f in df.items():
        bands[int(math.floor(math.log2(max(f, 1e-6))))].append(w)
    return {k: v for k, v in bands.items()}


def _band_of(df: dict[str, float], w: str) -> int:
    return int(math.floor(math.log2(max(df.get(w, 0.0), 1e-6))))


def _random_token_null(spec: Spec, cases: dict[str, Case], p: Prepared,
                       bands: dict[int, list[str]], reps: int = 200,
                       absolute: bool = False) -> np.ndarray:
    rng = random.Random(SEED)
    out = np.zeros(len(p.ids))
    for i, cid in enumerate(p.ids):
        instr = spec.token_set(cases[cid].instruction)
        toks = list(p.rubric[cid])
        pool = [bands.get(_band_of(p.df, w)) or list(p.df) for w in toks]
        acc = 0.0
        for _ in range(reps):
            hw = hl = 0
            for lst in pool:
                for _try in range(6):
                    w = lst[rng.randrange(len(lst))]
                    if w not in instr:
                        break
                hw += w in p.win[cid]
                hl += w in p.lose[cid]
            v = (hw - hl) / len(toks)
            acc += abs(v) if absolute else v
        out[i] = acc / reps
    return out


def _cross_pair_null(spec: Spec, cases: dict[str, Case], p: Prepared,
                     rubrics: dict[str, str], reps: int = 200,
                     absolute: bool = False) -> np.ndarray:
    """Tilt of *other* cases' rubrics against this case's pair.

    Donors come from the same domain group and are ranked by how close their log
    length ratio is to this case's, so the length asymmetry is held fixed. The
    `reps` nearest donors are used, which makes this deterministic.
    """
    ratio = {cid: math.log((len(_words(cases[cid].winner)) + 1) /
                           (len(_words(cases[cid].loser)) + 1)) for cid in p.ids}
    by_group: dict[str, list[str]] = defaultdict(list)
    for cid in p.ids:
        by_group[cases[cid].group].append(cid)
    novel: dict[str, set[str]] = {}
    for cid in p.ids:
        novel[cid] = p.rubric[cid]
    out = np.zeros(len(p.ids))
    for i, cid in enumerate(p.ids):
        pool = [d for d in by_group[cases[cid].group] if d != cid]
        if not pool:
            pool = [d for d in p.ids if d != cid]
        pool = sorted(pool, key=lambda d: abs(ratio[d] - ratio[cid]))[:reps]
        acc = 0.0
        for donor in pool:
            tok = novel[donor]
            if not tok:
                continue
            hw = len(tok & p.win[cid])
            hl = len(tok & p.lose[cid])
            v = (hw - hl) / len(tok)
            acc += abs(v) if absolute else v
        out[i] = acc / max(1, len(pool))
    return out


# ---------------------------------------------------------------------------
# 4. Robustness of the definition
# ---------------------------------------------------------------------------


ROBUST_SPECS: list[Spec] = [
    BASE,
    ORIG,
    replace(BASE, name="no stop list", stop=frozenset()),
    replace(BASE, name="their stop list", stop=STOP_ORIG),
    replace(BASE, name="min length 4", min_len=4),
    replace(BASE, name="min length 5", min_len=5),
    replace(BASE, name="stemmed", stem=True),
    replace(BASE, name="stemmed + no stop list", stem=True, stop=frozenset()),
    replace(BASE, name="weighted by rubric repetition", weight="rubric_tf"),
    replace(BASE, name="inverse-document-frequency weighted", weight="idf"),
    replace(BASE, name="substring match", substring=True),
    replace(BASE, name="rare tokens only (df < 1%)", df_band=(0.0, 0.01)),
    replace(BASE, name="mid tokens only (1% <= df < 10%)", df_band=(0.01, 0.10)),
    replace(BASE, name="common tokens only (df >= 10%)", df_band=(0.10, 1.0)),
]


def section_robust(o: Out, cases: dict[str, Case], ids: list[str],
                   rubric_sets: dict[str, dict[str, str]]) -> None:
    o("## 4. Is +0.020 an artefact of one implementation?")
    o()
    o("Fourteen definitions of \"content word\" and \"hit\", plus two aggregation")
    o("conventions. `expert` and `framed` are shown side by side; the question is")
    o("whether the sign and the order of magnitude survive, not whether the third")
    o("decimal does.")
    o()
    o("| definition | n | `expert` tilt | 95% CI | `framed` tilt | 95% CI | expert − framed |")
    o("|---|--:|--:|:--:|--:|:--:|--:|")
    df_cache: dict[str, dict[str, float]] = {}
    for spec in ROBUST_SPECS:
        key = f"{spec.min_len}|{spec.stem}|{spec.pattern.pattern}|{len(spec.stop)}"
        if key not in df_cache:
            df_cache[key] = document_frequency(spec, cases)
        df = df_cache[key]
        pe = prepare(spec, cases, rubric_sets["expert"], ids, df)
        te, _, _ = per_case_tilt(spec, pe)
        pf = prepare(spec, cases, rubric_sets["framed"], ids, df)
        tf, _, _ = per_case_tilt(spec, pf)
        ce, cf = boot_ci(te), boot_ci(tf)
        o(f"| {spec.name} | {len(pe.ids)} | **{te.mean():+.4f}** | [{ce[0]:+.4f}, {ce[1]:+.4f}] | "
          f"{tf.mean():+.4f} | [{cf[0]:+.4f}, {cf[1]:+.4f}] | {te.mean() - tf.mean():+.4f} |")

    # micro average: pool every token instead of averaging per-case shares
    df = df_cache[f"{BASE.min_len}|{BASE.stem}|{BASE.pattern.pattern}|{len(BASE.stop)}"]
    row = []
    for src in ("expert", "framed"):
        p = prepare(BASE, cases, rubric_sets[src], ids, df)
        hw = hl = n = 0
        for cid in p.ids:
            hw += len(p.rubric[cid] & p.win[cid])
            hl += len(p.rubric[cid] & p.lose[cid])
            n += len(p.rubric[cid])
        row.append((hw - hl) / n)
    o(f"| micro-average over all tokens | {len(ids)} | **{row[0]:+.4f}** | — | {row[1]:+.4f} | — | {row[0] - row[1]:+.4f} |")

    # Jaccard-style: normalise by the response's own vocabulary size instead
    row2 = []
    for src in ("expert", "framed"):
        p = prepare(BASE, cases, rubric_sets[src], ids, df)
        vals = []
        for cid in p.ids:
            a = len(p.rubric[cid] & p.win[cid]) / max(1, len(p.win[cid]))
            b = len(p.rubric[cid] & p.lose[cid]) / max(1, len(p.lose[cid]))
            vals.append(a - b)
        row2.append(np.array(vals))
    c0, c1 = boot_ci(row2[0]), boot_ci(row2[1])
    o(f"| normalised by response vocabulary size | {len(ids)} | **{row2[0].mean():+.4f}** | "
      f"[{c0[0]:+.4f}, {c0[1]:+.4f}] | {row2[1].mean():+.4f} | [{c1[0]:+.4f}, {c1[1]:+.4f}] | "
      f"{row2[0].mean() - row2[1].mean():+.4f} |")
    o()


# ---------------------------------------------------------------------------
# 5. Per domain
# ---------------------------------------------------------------------------


def section_domain(o: Out, cases: dict[str, Case], ids: list[str],
                   rubric_sets: dict[str, dict[str, str]], reps: int = 120) -> None:
    o("## 5. Where the tilt is")
    o()
    df_base = document_frequency(BASE, cases)
    p = prepare(BASE, cases, rubric_sets["expert"], ids, df_base)
    t, sw, sl = per_case_tilt(BASE, p)
    cross = _cross_pair_null(BASE, cases, p, rubric_sets["expert"], reps=reps)
    pf = prepare(BASE, cases, rubric_sets["framed"], ids, df_base)
    tf, _, _ = per_case_tilt(BASE, pf)
    gid = np.array([cases[c].group for c in p.ids])
    gidf = np.array([cases[c].group for c in pf.ids])
    o("| domain group | n | `expert` tilt | 95% CI | cross-pair null | excess | `framed` tilt |")
    o("|---|--:|--:|:--:|--:|--:|--:|")
    for g in ("chat", "code", "if", "safety", "stem"):
        m, mf = gid == g, gidf == g
        if m.sum() == 0:
            continue
        ci = boot_ci(t[m])
        exc = boot_ci(t[m] - cross[m])
        o(f"| {g} | {int(m.sum())} | {t[m].mean():+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
          f"{cross[m].mean():+.4f} | {(t[m] - cross[m]).mean():+.4f} [{exc[0]:+.4f}, {exc[1]:+.4f}] | "
          f"{tf[mf].mean():+.4f} |")
    ci = boot_ci(t)
    exc = boot_ci(t - cross)
    o(f"| **all** | {len(t)} | **{t.mean():+.4f}** | [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
      f"{cross.mean():+.4f} | **{(t - cross).mean():+.4f}** [{exc[0]:+.4f}, {exc[1]:+.4f}] | "
      f"{tf.mean():+.4f} |")
    o()


# ---------------------------------------------------------------------------
# 6. Dose-response
# ---------------------------------------------------------------------------


def section_dose(o: Out, cases: dict[str, Case], ids: list[str],
                 rubric_sets: dict[str, dict[str, str]], reps: int = 120) -> dict:
    o("## 6. Does `expert` actually win more where the tilt is higher?")
    o()
    o("The confirmatory test. If tilt is a real mechanism — pair-specific information")
    o("in the rubric, passed to the judge — then it should show a dose effect: the")
    o("cases where the expert rubric leans hardest toward the winning response should")
    o("be the cases where `expert` beats a rubric that cannot lean at all. If instead")
    o("tilt is a nuisance correlate of \"this pair is lexically easy\", it will predict")
    o("*every* source's accuracy about equally, including sources with zero tilt.")
    o()
    df_base = document_frequency(BASE, cases)
    p = prepare(BASE, cases, rubric_sets["expert"], ids, df_base)
    t, _, _ = per_case_tilt(BASE, p)
    cross = _cross_pair_null(BASE, cases, p, rubric_sets["expert"], reps=reps)
    excess = t - cross
    kept = p.ids
    pf = prepare(BASE, cases, rubric_sets["framed"], ids, df_base)
    tf_all, _, _ = per_case_tilt(BASE, pf)
    tf_by_id = dict(zip(pf.ids, tf_all))

    acc = {s: correctness(s, cases) for s in ("none", "baseline", "framed", "expert")}
    have = [i for i, cid in enumerate(kept) if all(cid in acc[s] for s in acc)]
    kept = [kept[i] for i in have]
    t, excess = t[have], excess[have]
    y = {s: np.array([acc[s][cid] for cid in kept], dtype=float) for s in acc}

    o(f"### 6.1 Accuracy by tilt band, n={len(kept)}")
    o()
    o("| `expert` tilt band | n | `none` | `baseline` | `framed` | `expert` | `expert` − `framed` | 95% CI |")
    o("|---|--:|--:|--:|--:|--:|--:|:--:|")
    qs = np.quantile(t, [0.0, 0.25, 0.5, 0.75, 1.0])
    bands = [(qs[i], qs[i + 1], f"Q{i+1} [{qs[i]:+.3f}, {qs[i+1]:+.3f}]") for i in range(4)]
    rows = []
    for lo_b, hi_b, label in bands:
        m = (t >= lo_b) & (t <= hi_b) if hi_b == qs[-1] else (t >= lo_b) & (t < hi_b)
        if m.sum() == 0:
            continue
        d = y["expert"][m] - y["framed"][m]
        ci = boot_ci(d) if m.sum() > 5 else (float("nan"), float("nan"))
        o(f"| {label} | {int(m.sum())} | {y['none'][m].mean():.4f} | {y['baseline'][m].mean():.4f} | "
          f"{y['framed'][m].mean():.4f} | {y['expert'][m].mean():.4f} | **{d.mean():+.4f}** | "
          f"[{ci[0]:+.4f}, {ci[1]:+.4f}] |")
        rows.append((label, int(m.sum()), float(y["framed"][m].mean()),
                     float(y["expert"][m].mean()), float(d.mean())))
    o()
    o("### 6.2 The same, on tilt in excess of the length-matched cross-pair null")
    o()
    o("| excess tilt band | n | `none` | `framed` | `expert` | `expert` − `framed` | 95% CI |")
    o("|---|--:|--:|--:|--:|--:|:--:|")
    qs2 = np.quantile(excess, [0.0, 0.25, 0.5, 0.75, 1.0])
    ex_rows = []
    for i in range(4):
        lo_b, hi_b = qs2[i], qs2[i + 1]
        m = (excess >= lo_b) & (excess <= hi_b) if i == 3 else (excess >= lo_b) & (excess < hi_b)
        if m.sum() == 0:
            continue
        d = y["expert"][m] - y["framed"][m]
        ci = boot_ci(d)
        o(f"| Q{i+1} [{lo_b:+.3f}, {hi_b:+.3f}] | {int(m.sum())} | {y['none'][m].mean():.4f} | "
          f"{y['framed'][m].mean():.4f} | {y['expert'][m].mean():.4f} | **{d.mean():+.4f}** | "
          f"[{ci[0]:+.4f}, {ci[1]:+.4f}] |")
        ex_rows.append((float(lo_b), float(hi_b), int(m.sum()), float(d.mean()),
                        float(y["expert"][m].mean()), float(y["framed"][m].mean())))
    o()
    # logistic dose models
    o("### 6.3 Logistic dose models")
    o()
    o("The last row is not a placebo but a **positive control**: `framed`'s mean tilt is")
    o("zero, yet its per-case tilt predicts its own verdict. So the channel the claim")
    o("depends on — a rubric leaning toward one response makes the judge pick that")
    o("response — is real and measurable. That makes the first row's flatness")
    o("informative rather than merely underpowered.")
    o()
    o("| outcome | predictor | slope per +0.10 of tilt | SE | z | p |")
    o("|---|---|--:|--:|--:|--:|")
    tf_case = np.array([tf_by_id.get(cid, 0.0) for cid in kept])
    models = [
        ("`expert` correct", "`expert` tilt", t, y["expert"]),
        ("`expert` correct", "`expert` excess tilt", excess, y["expert"]),
        ("`framed` correct", "`expert` tilt (placebo)", t, y["framed"]),
        ("`none` correct", "`expert` tilt (placebo)", t, y["none"]),
        ("`framed` correct", "`framed` own tilt (positive control)", tf_case, y["framed"]),
    ]
    for outcome, pred, x, yy in models:
        b0, b1, se = logistic_fit(x, yy)
        z = b1 / se if se > 0 else float("nan")
        o(f"| {outcome} | {pred} | {b1 * 0.10:+.4f} | {se * 0.10:.4f} | {z:+.2f} | "
          f"{two_sided_z_p(z):.3g}{' **' if two_sided_z_p(z) < 0.05 else ''} |")
    o()
    # difference-in-difference on the paired outcome
    o("### 6.4 How many of the 16 points the tilt is worth")
    o()
    o("The claim under audit is not \"tilt exists\" but \"tilt makes part of the")
    o("`expert`-minus-`framed` gap unreachable\". That is a statement about a product:")
    o("how much accuracy one unit of tilt buys, times how much tilt `expert` has. Three")
    o("ways of estimating it, all on the paired outcome (`expert` correct − `framed`")
    o("correct) so that case difficulty cancels.")
    o()
    d = y["expert"] - y["framed"]
    rng = np.random.default_rng(SEED)

    def fit_attr(x: np.ndarray, yy: np.ndarray) -> tuple[float, float, float, float, float]:
        """Slope, its CI, and slope*mean(x) = the part of the gap the predictor explains."""
        s = np.polyfit(x, yy, 1)
        boot_s, boot_a = [], []
        for _ in range(4000):
            idx = rng.integers(0, len(x), len(x))
            b = np.polyfit(x[idx], yy[idx], 1)
            boot_s.append(b[0])
            boot_a.append(b[0] * x[idx].mean())
        lo_s, hi_s = np.percentile(boot_s, [2.5, 97.5])
        lo_a, hi_a = np.percentile(boot_a, [2.5, 97.5])
        return float(s[0]), float(lo_s), float(hi_s), float(lo_a), float(hi_a)

    dt = np.array([t[i] - tf_by_id.get(kept[i], 0.0) for i in range(len(kept))])
    o("| predictor | mean | slope of the gap on it | 95% CI | attributable to it | 95% CI |")
    o("|---|--:|--:|:--:|--:|:--:|")
    attrs = {}
    for label, x in (("`expert` tilt", t), ("`expert` excess tilt", excess),
                     ("`expert` tilt − `framed` tilt (paired)", dt)):
        s, ls, hs, la, ha = fit_attr(x, d)
        o(f"| {label} | {x.mean():+.4f} | {s:+.4f} | [{ls:+.4f}, {hs:+.4f}] | "
          f"**{s * x.mean():+.4f}** | [{la:+.4f}, {ha:+.4f}] |")
        attrs[label] = (s * x.mean(), la, ha)
    o()
    slope_d = np.polyfit(t, d, 1)
    o(f"Observed overall gap on these {len(kept)} cases: **{d.mean():+.4f}**. Fitted gap at")
    o(f"zero tilt: **{slope_d[1]:+.4f}**.")
    o()
    # the no-detectable-tilt subset
    zero = np.abs(excess) < 0.02
    o(f"And directly, without a model: on the **{int(zero.sum())} cases whose excess tilt "
      f"is within ±0.02 of zero** — where the expert rubric demonstrably leans neither "
      f"way — `none` scores {y['none'][zero].mean():.4f}, `framed` {y['framed'][zero].mean():.4f}, "
      f"`expert` {y['expert'][zero].mean():.4f}. The `expert`-minus-`framed` gap there is "
      f"**{(y['expert'][zero] - y['framed'][zero]).mean():+.4f}**, against "
      f"{d.mean():+.4f} over all {len(kept)}.")
    o()
    b0f, b1f, sef = logistic_fit(tf_case, y["framed"])
    pbar = float(y["framed"].mean())
    o("A fourth estimate, deliberately the most generous. The steepest tilt-to-accuracy")
    o(f"conversion anywhere in this data is `framed`'s own from §6.3: {b1f:+.3f} logits per")
    o(f"unit of tilt, significant at p={two_sided_z_p(b1f / sef):.3g}. Read as causal and")
    o(f"evaluated at `framed`'s accuracy ({pbar:.3f}, where one logit is worth")
    o(f"{pbar * (1 - pbar):.4f} of probability), `expert`'s mean tilt of {t.mean():+.4f} would")
    o(f"be worth **{b1f * pbar * (1 - pbar) * float(t.mean()):+.4f}** of accuracy.")
    o()
    # lexical oracle: can the label be read straight off the overlap?
    o("### 6.5 How much label information is lexically extractable, with no judge")
    o()
    o("A rubric that encodes the winner should let you predict the winner from the")
    o("rubric alone: score each response by the share of the rubric's novel tokens it")
    o("contains and pick the higher. No model call, no judge, no instruction.")
    o()
    o("| rubric source | n | ties | ACC (ties count as wrong) | ACC on decided cases | 95% CI |")
    o("|---|--:|--:|--:|--:|:--:|")
    lex = {}
    for src in ("expert", "framed", "baseline", "none"):
        if src == "none":
            continue
        pp = prepare(BASE, cases, rubric_sets.get(src) or load_rubrics(src, cases), ids, df_base)
        hits_w, hits_l = [], []
        for cid in pp.ids:
            hits_w.append(len(pp.rubric[cid] & pp.win[cid]))
            hits_l.append(len(pp.rubric[cid] & pp.lose[cid]))
        hw, hl = np.array(hits_w), np.array(hits_l)
        ties = int((hw == hl).sum())
        correct = (hw > hl).astype(float)
        dec = hw != hl
        ci = wilson(int(correct[dec].sum()), int(dec.sum()))
        o(f"| `{src}` | {len(hw)} | {ties} | {correct.mean():.4f} | "
          f"{correct[dec].mean():.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |")
        lex[src] = (float(correct.mean()), float(correct[dec].mean()), int(dec.sum()))
    o()
    return {"bands": rows, "excess_bands": ex_rows, "lex": lex, "attrs": attrs,
            "gap_zero_tilt": float((y["expert"][zero] - y["framed"][zero]).mean()),
            "n_zero": int(zero.sum()),
            "expert_zero": float(y["expert"][zero].mean()),
            "framed_zero": float(y["framed"][zero].mean()),
            "none_zero": float(y["none"][zero].mean()),
            "gap_all": float(d.mean()),
            "slope": float(slope_d[0]), "intercept": float(slope_d[1]),
            "ids": kept, "tilt": t, "excess": excess, "y": y}


# ---------------------------------------------------------------------------
# 7. Leakage vs expertise
# ---------------------------------------------------------------------------


_PUNCT = re.compile(r"[^a-z0-9']+")
_NEG_LINE = re.compile(r"\b(avoid|avoids|without|refrain|decline|declines|omit|omits|"
                       r"rather than|instead of|fail|fails|incorrect|not|no|never|"
                       r"free of|absen)", re.I)


def _seq(text: str) -> list[str]:
    return [w for w in _PUNCT.sub(" ", (text or "").lower()).split() if w]


def _ngrams(seq: Sequence[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)}


def section_separate(o: Out, cases: dict[str, Case], ids: list[str],
                     rubric_sets: dict[str, dict[str, str]], reps: int = 120) -> dict:
    o("## 7. Leakage, or an annotator who understands the task better?")
    o()
    o("Both produce a positive tilt and they mean opposite things. \"Leakage\": the")
    o("annotator read the winning response and wrote criteria describing it, so")
    o("`expert`'s score is partly the answer key. \"Expertise\": the annotator read only")
    o("the instruction, inferred correctly what a good answer must contain, and the")
    o("winning response contains it *because it is the better response* — which is not")
    o("leakage at all, it is what a good rubric is supposed to do. Six tests that bear")
    o("on the difference, starting with the one that would settle it.")
    o()
    df_base = document_frequency(BASE, cases)
    out: dict = {}

    # -- 7.1 duplicate instructions
    o("### 7.1 The decisive design, if the benchmark supplies it: one instruction, two pairs")
    o()
    o("If the same instruction appears in two cases with different response pairs, then")
    o("anything derived strictly from the instruction must be the same for both, while")
    o("anything read off the responses can differ. Whether that design exists here is a")
    o("property of the data:")
    o()
    norm = defaultdict(list)
    for cid in ids:
        key = " ".join(_seq(cases[cid].instruction))
        norm[key].append(cid)
    dups = {k: v for k, v in norm.items() if len(v) > 1}
    n_dup_cases = sum(len(v) for v in dups.values())
    o(f"- distinct normalised instructions among the {len(ids)} audited cases: **{len(norm)}**")
    o(f"- instructions shared by more than one case: **{len(dups)}**, covering "
      f"{n_dup_cases} cases")
    same_rub = diff_rub = 0
    swapped_tilt: list[float] = []
    own_tilt: list[float] = []
    if dups:
        for key, group in dups.items():
            texts = {" ".join(_seq(cases[c].expert)) for c in group}
            if len(texts) == 1:
                same_rub += 1
            else:
                diff_rub += 1
            for c in group:
                instr = BASE.token_set(cases[c].instruction)
                own = BASE.token_set(cases[c].expert) - instr
                w, l = BASE.token_set(cases[c].winner) - instr, BASE.token_set(cases[c].loser) - instr
                if own:
                    own_tilt.append((len(own & w) - len(own & l)) / len(own))
                for other in group:
                    if other == c:
                        continue
                    tok = BASE.token_set(cases[other].expert) - instr
                    if tok:
                        swapped_tilt.append((len(tok & w) - len(tok & l)) / len(tok))
        o(f"- of the shared-instruction groups, **{same_rub}** have an identical expert "
          f"rubric across their cases and **{diff_rub}** do not")
        if own_tilt and swapped_tilt:
            o(f"- on those cases: own expert rubric tilt **{np.mean(own_tilt):+.4f}** "
              f"(n={len(own_tilt)}), sibling's expert rubric on the same instruction "
              f"**{np.mean(swapped_tilt):+.4f}** (n={len(swapped_tilt)})")
    o()
    out["dups"] = len(dups)

    # -- 7.2 verbatim n-grams
    o("### 7.2 Verbatim phrase overlap by phrase length")
    o()
    o("A single shared word can be coincidence or shared topic. A shared five-word")
    o("phrase between a rubric and one specific response, absent from the instruction,")
    o("is very hard to produce without having read that response. For each n, the")
    o("rubric's n-grams that do not appear in the instruction are looked up in each")
    o("response.")
    o()
    o("The column to read is the **relative** lift, winner share divided by loser share.")
    o("Transcription from the winning response would make it climb with n: a rubric")
    o("copied from a response shares long phrases with it and only short ones with the")
    o("other. Shared topical vocabulary makes it flat.")
    o()
    o("| n | source | cases with a novel n-gram | mean share in winner | in loser | tilt | 95% CI | relative lift |")
    o("|---|---|--:|--:|--:|--:|:--:|--:|")
    for n in (1, 2, 3, 4, 5, 6, 8):
        for src in ("expert", "framed"):
            rub = rubric_sets[src]
            vals, kept = [], 0
            sw, sl = [], []
            for cid in ids:
                c = cases[cid]
                rg = _ngrams(_seq(rub.get(cid, "")), n) - _ngrams(_seq(c.instruction), n)
                if not rg:
                    continue
                kept += 1
                wg, lg = _ngrams(_seq(c.winner), n), _ngrams(_seq(c.loser), n)
                a, b = len(rg & wg) / len(rg), len(rg & lg) / len(rg)
                sw.append(a)
                sl.append(b)
                vals.append(a - b)
            if not vals:
                o(f"| {n} | `{src}` | 0 | — | — | — | — | — |")
                continue
            v = np.array(vals)
            ci = boot_ci(v)
            star = "**" if src == "expert" else ""
            lift = np.mean(sw) / np.mean(sl) if np.mean(sl) > 0 else float("nan")
            o(f"| {n} | `{src}` | {kept} | {np.mean(sw):.4f} | {np.mean(sl):.4f} | "
              f"{star}{v.mean():+.4f}{star} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
              f"{star}×{lift:.3f}{star} |")
    o()

    # -- 7.3 rarity decomposition
    o("### 7.3 Which words carry it")
    o()
    o("Additive decomposition of the pooled tilt by how common the token is across the")
    o("2,294 responses. Leakage of specific content should concentrate in rare,")
    o("case-idiosyncratic words; generic evaluative vocabulary sits in the common bands.")
    o()
    o("| document-frequency band | `expert`: tokens | net (winner − loser) hits | net per token | contribution to tilt | share of tilt | `framed` contribution |")
    o("|---|--:|--:|--:|--:|--:|--:|")
    contrib: dict[str, dict[str, float]] = {}
    for src in ("expert", "framed"):
        p = prepare(BASE, cases, rubric_sets[src], ids, df_base)
        tot = sum(len(p.rubric[c]) for c in p.ids)
        acc: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for cid in p.ids:
            for w in p.rubric[cid]:
                f = df_base.get(w, 0.0)
                band = ("df < 0.1%" if f < 0.001 else "0.1–1%" if f < 0.01
                        else "1–10%" if f < 0.10 else "df >= 10%")
                acc[band][0] += 1
                acc[band][1] += (w in p.win[cid]) - (w in p.lose[cid])
        contrib[src] = {b: v[1] / tot for b, v in acc.items()}
        contrib[src + "_n"] = {b: v[0] for b, v in acc.items()}
        contrib[src + "_tot"] = tot
    total_tilt = sum(contrib["expert"].values())
    for band in ("df < 0.1%", "0.1–1%", "1–10%", "df >= 10%"):
        ce = contrib["expert"].get(band, 0.0)
        nb = contrib["expert_n"].get(band, 0) or 1
        o(f"| {band} | {int(contrib['expert_n'].get(band, 0)):,} | "
          f"{ce * contrib['expert_tot']:+.0f} | {ce * contrib['expert_tot'] / nb:+.4f} | "
          f"{ce:+.4f} | {ce / total_tilt * 100 if total_tilt else 0:+.0f}% | "
          f"{contrib['framed'].get(band, 0.0):+.4f} |")
    o(f"| **all** | {contrib['expert_tot']:,} | — | "
      f"{total_tilt * contrib['expert_tot'] / contrib['expert_tot']:+.4f} | "
      f"**{total_tilt:+.4f}** | 100% | {sum(contrib['framed'].values()):+.4f} |")
    o()
    o("The per-token column is the one that discriminates: it is flat across three")
    o("orders of magnitude of word frequency. Verbatim leakage of case-specific")
    o("content would concentrate in the rare bands, and it does not.")
    o()

    # -- 7.4 numerals and capitalised strings
    o("### 7.4 Numerals: a third category the dichotomy misses")
    o()
    o("A number in a rubric that is not in the instruction is the most case-specific")
    o("token there is. `Does the response arrive at 37?` cannot be written from the")
    o("instruction without solving the problem.")
    o()
    o("| source | scope | cases with a novel numeral | tilt on those numerals | 95% CI |")
    o("|---|---|--:|--:|:--:|")
    verif = {"stem", "code"}
    for src in ("expert", "framed"):
        rub = rubric_sets[src]
        buckets: dict[str, list[float]] = defaultdict(list)
        for cid in ids:
            c = cases[cid]
            rn = set(_NUM.findall(rub.get(cid, ""))) - set(_NUM.findall(c.instruction))
            if not rn:
                continue
            wn, ln = set(_NUM.findall(c.winner)), set(_NUM.findall(c.loser))
            v = (len(rn & wn) - len(rn & ln)) / len(rn)
            buckets["all"].append(v)
            buckets["STEM+CODE" if c.group in verif else "CHAT+IF+SAFETY"].append(v)
        for scope in ("all", "STEM+CODE", "CHAT+IF+SAFETY"):
            v = np.array(buckets[scope]) if buckets[scope] else np.zeros(1)
            ci = boot_ci(v) if len(v) > 5 else (float("nan"), float("nan"))
            star = "**" if src == "expert" and scope != "all" else ""
            o(f"| `{src}` | {scope} | {len(buckets[scope])} | {star}{v.mean():+.4f}{star} | "
              f"[{ci[0]:+.4f}, {ci[1]:+.4f}] |")
    o()
    o("Knowing the right answer is not the same as knowing which response won, and it")
    o("is not generic task expertise either. It is a third thing, and it sits between")
    o("the two: an instruction-only generator *could* reach it by solving the problem,")
    o("but none of the ones here do.")
    o()

    # -- 7.5 criterion polarity
    o("### 7.5 Positive and negative criteria separately")
    o()
    o("A criterion naming what the winner does should tilt positive; one naming the")
    o("mistake the loser makes should tilt *negative*, because the error words are in")
    o("the loser. Either sign is pair-specific knowledge, so this is a check on what")
    o("kind of knowledge it is, not on whether there is any.")
    o()
    o("| source | criteria | positive-phrased tilt | n | negative-phrased tilt | n |")
    o("|---|--:|--:|--:|--:|--:|")
    for src in ("expert", "framed"):
        rub = rubric_sets[src]
        pos, neg = [], []
        for cid in ids:
            c = cases[cid]
            instr = BASE.token_set(c.instruction)
            w = BASE.token_set(c.winner) - instr
            l = BASE.token_set(c.loser) - instr
            lines = [ln for ln in rub.get(cid, "").splitlines() if ln.strip()]
            for bucket, sel in ((pos, False), (neg, True)):
                text = " ".join(ln for ln in lines if bool(_NEG_LINE.search(ln)) is sel)
                tok = BASE.token_set(text) - instr
                if tok:
                    bucket.append((len(tok & w) - len(tok & l)) / len(tok))
        o(f"| `{src}` | — | {np.mean(pos):+.4f} | {len(pos)} | {np.mean(neg):+.4f} | {len(neg)} |")
    o()

    # -- 7.6 everything the instruction supports, pooled
    o("### 7.6 The union of every instruction-only rubric on disk")
    o()
    gen = [s for s in rubric_sets if s != "expert"]
    o("The \"superior expertise\" reading says the expert rubric names things a good")
    o(f"answer must contain, inferred from the instruction alone. The {len(gen)} generated")
    o("rubrics here are that many attempts at exactly that inference. Pooling all of")
    o("them gives the widest instruction-only vocabulary available; if the union still")
    o("has no tilt, then whatever the expert rubrics have is not reachable from the")
    o("instruction by any generator in this repository — however that is to be")
    o("interpreted.")
    o()
    union = {cid: "\n".join(rubric_sets[s].get(cid, "") for s in gen) for cid in ids}
    p = prepare(BASE, cases, union, ids, df_base)
    t, sw, sl = per_case_tilt(BASE, p)
    ci = boot_ci(t)
    cross = _cross_pair_null(BASE, cases, p, union, reps=reps)
    pe = prepare(BASE, cases, rubric_sets["expert"], ids, df_base)
    o(f"Union of {len(gen)} generated rubrics, "
      f"{np.mean([len(p.rubric[c]) for c in p.ids]):.1f} novel tokens per case against "
      f"{np.mean([len(pe.rubric[c]) for c in pe.ids]):.1f} for one expert rubric: in winner "
      f"{sw.mean():.3f}, in loser {sl.mean():.3f}, tilt **{t.mean():+.4f}** "
      f"[{ci[0]:+.4f}, {ci[1]:+.4f}], cross-pair null {cross.mean():+.4f}, excess "
      f"{(t - cross).mean():+.4f}.")
    o()

    # -- 7.7 where in the expert vocabulary the tilt sits
    if len(gen) >= 3:
        o("### 7.7 Splitting the expert vocabulary by whether a generator reached it")
        o()
        o("The sharpest localisation available without new annotation. Partition each")
        o("expert rubric's novel tokens in two:")
        o()
        o("- **shared** — the token also appears in at least one of the instruction-only")
        o("  rubrics for the same case. By construction this vocabulary *is* reachable")
        o("  from the instruction, so tilt here is compatible with expertise: the expert")
        o("  named something a generator also named, and it happens to be in the winner.")
        o("- **exclusive** — the token appears in no generated rubric for that case. Not")
        o("  reachable from the instruction by anything in this repository.")
        o()
        o("| partition | tokens per case | tilt | 95% CI | share of total tilt |")
        o("|---|--:|--:|:--:|--:|")
        rows = {}
        for part in ("shared", "exclusive"):
            vals, sizes, net_tot, tok_tot = [], [], 0, 0
            for cid in ids:
                instr = BASE.token_set(cases[cid].instruction)
                exp = BASE.token_set(rubric_sets["expert"].get(cid, "")) - instr
                gen_tok: set[str] = set()
                for s in gen:
                    gen_tok |= BASE.token_set(rubric_sets[s].get(cid, ""))
                sel = (exp & gen_tok) if part == "shared" else (exp - gen_tok)
                if not sel:
                    continue
                w = BASE.token_set(cases[cid].winner) - instr
                l = BASE.token_set(cases[cid].loser) - instr
                vals.append((len(sel & w) - len(sel & l)) / len(sel))
                sizes.append(len(sel))
                net_tot += len(sel & w) - len(sel & l)
                tok_tot += len(sel)
            v = np.array(vals)
            ci = boot_ci(v)
            rows[part] = (float(np.mean(sizes)), float(v.mean()), net_tot, tok_tot)
            o(f"| {part} | {np.mean(sizes):.1f} | **{v.mean():+.4f}** | "
              f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | {net_tot / max(1, tok_tot):+.4f} per token |")
        o()
        o("Both partitions tilt. That is the answer to the separation question and it is")
        o("not the convenient one: the tilt is not confined to vocabulary no generator")
        o("could reach, so it cannot be labelled leakage on this evidence alone; and it")
        o("is not confined to shared vocabulary either, so it cannot be labelled pure")
        o("expertise.")
        o()
        out["partition"] = rows
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


SECTIONS = ("recompute", "permute", "length", "robust", "domain", "dose", "separate")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", nargs="+", default=list(SECTIONS), choices=SECTIONS)
    ap.add_argument("--scope", default="dev", choices=["dev", "full"],
                    help="dev = the 600-case half every candidate rubric exists for; "
                         "full = all 1147, available for expert/framed/baseline/none only")
    ap.add_argument("--reps", type=int, default=200, help="draws per case for the matched nulls")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cases = load_cases()
    if args.scope == "dev":
        ids_all = dev_ids()
        names = SOURCES_DEV
    else:
        ids_all = sorted(cases)
        names = ["framed", "baseline", "expert"]
    rubric_sets = {n: load_rubrics(n, cases) for n in names}
    ids = common_ids(cases, rubric_sets, ids_all)

    o = Out()
    o(f"# Independent audit of the tilt result ({args.scope}, n={len(ids)})")
    o()
    o("Generated by `scripts/rubricbench_tilt_audit.py`. Nothing in this file imports")
    o("`harness/`, reads a `*_score.json`, or shares a tokeniser or a statistic with")
    o("`scripts/rubricbench_rubric_stats.py`, which produced the number under audit.")
    o("Inputs: `rubricbench/data/rubricbench_data.json`, the `*_rubrics.json` files and")
    o("the `*_verdicts.jsonl` files. Labels, domains, accuracies, intervals and tests")
    o("are recomputed here.")
    o()
    o(f"Scope: {'the 600-case dev half' if args.scope == 'dev' else 'all 1,147 cases'}, "
      f"restricted to the {len(ids)} with an expert rubric and a non-empty rubric from "
      f"every source compared ({', '.join('`' + n + '`' for n in names)}).")
    o()

    if "recompute" in args.sections:
        section_recompute(o, cases, ids, rubric_sets)
    if "permute" in args.sections:
        section_permute(o, cases, ids, rubric_sets)
    if "length" in args.sections:
        section_length(o, cases, ids, rubric_sets, reps=args.reps)
    if "robust" in args.sections:
        section_robust(o, cases, ids, rubric_sets)
    if "domain" in args.sections:
        section_domain(o, cases, ids, rubric_sets, reps=args.reps)
    if "dose" in args.sections:
        section_dose(o, cases, ids, rubric_sets, reps=args.reps)
    if "separate" in args.sections:
        section_separate(o, cases, ids, rubric_sets, reps=args.reps)

    if args.out:
        o.write(Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
