#!/usr/bin/env python3
"""Power of the paired exact McNemar test for a given accuracy gap on RubricBench.

Simulates the comparison the reports run: each case is discordant with
probability `rate` (measured on dev), a discordant case favours the better
source with probability (1 + gap / rate) / 2, and the two-sided exact McNemar
p-value is compared with alpha. Also reports how many cases would be needed for
80% power.

    python scripts/rubricbench_power.py --rate 0.2178 --gaps 0.02 0.03 0.04
"""
from __future__ import annotations

import argparse
import math

import numpy as np
from scipy.stats import binom, norm


def power(n_cases: int, rate: float, gap: float, alpha: float, sims: int, rng: np.random.Generator) -> float:
    p_better = (1 + gap / rate) / 2
    n_disc = rng.binomial(n_cases, rate, sims)
    b = rng.binomial(n_disc, p_better)
    c = n_disc - b
    p = np.minimum(1.0, 2 * binom.cdf(np.minimum(b, c), n_disc, 0.5))
    return float(np.mean((p <= alpha) & (b > c)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rate", type=float, default=0.2178, help="share of cases the two sources disagree on")
    ap.add_argument("--gaps", type=float, nargs="+", default=[0.02, 0.03, 0.04])
    ap.add_argument("--sizes", type=int, nargs="+", default=[547, 600, 1147])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.05 / 3, 0.05])
    ap.add_argument("--sims", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=20261003)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"discordant rate {args.rate}, {args.sims} simulations per cell\n")
    print("| gap | cases | alpha | power |")
    print("|--:|--:|--:|--:|")
    for gap in args.gaps:
        for n in args.sizes:
            for alpha in args.alphas:
                print(f"| {gap:+.2f} | {n} | {alpha:.4f} | {power(n, args.rate, gap, alpha, args.sims, rng):.2f} |")

    print("\ncases for 80% power (normal approximation):")
    for gap in args.gaps:
        for alpha in args.alphas:
            n_disc = ((norm.ppf(1 - alpha / 2) + norm.ppf(0.8)) / (gap / args.rate)) ** 2
            print(f"  gap {gap:+.2f}, alpha {alpha:.4f}: ~{math.ceil(n_disc / args.rate)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
