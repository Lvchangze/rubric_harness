#!/usr/bin/env python3
"""Check that the load-bearing numbers in REPORT.md still match the artefacts.

The report is 100KB and cites several hundred numbers. Most were written by
hand from tables that have since been regenerated, and a column can shift under
a sentence without anything failing. This checks the numbers the argument
actually rests on, plus the claims about what was and was not finished.

It is not a full trace of every figure — that would need the report to be
generated rather than written. It is the subset where being wrong would change
a reader's decision.

    python3 scripts/check_report_facts.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REPORT = (REPO / "results" / "REPORT.md").read_text(encoding="utf-8")
HANDOFF = (REPO / "HANDOFF.md").read_text(encoding="utf-8")
README = (REPO / "README.md").read_text(encoding="utf-8")

problems: list[str] = []
checked = 0


def want(label: str, condition: bool, detail: str = "") -> None:
    global checked
    checked += 1
    if not condition:
        problems.append(f"{label}{(' — ' + detail) if detail else ''}")


def cites(text: str, *needles: str) -> bool:
    return all(n in text for n in needles)


def check_screening() -> None:
    """The screening figures in §4.6.3 must match screening_summary.json."""
    path = REPO / "runs" / "rollout_v2" / "screening_summary.json"
    if not path.exists():
        problems.append("screening_summary.json missing")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    stats = data["screen_stats"]
    oracle = data["oracle"]

    want("§4.6.3 screened total", cites(REPORT, "780"), "expected 780")
    want("§4.6.3 selected total", cites(REPORT, "128"), "expected 128")
    want("§4.6.3 mixed rate", cites(REPORT, "16.4%"),
         f"actual {oracle['informative_fraction']:.3f}")
    want("§4.6.3 all-correct count", cites(REPORT, "579"), "expected 579")
    want("§4.6.3 all-wrong count", cites(REPORT, "73"), "expected 73")
    want("§4.6.3 rollout correct rate", cites(REPORT, "84.5%"),
         f"actual {oracle['rollout_correct_rate']:.3f}")
    want("§4.6.3 oracle self-agreement", cites(REPORT, "95.2%"),
         f"actual {oracle['oracle_self_agreement']:.3f}")

    # And the underlying values really are what the report says.
    want("artefact: 780 screened", data["n_screened"] == 780, str(data["n_screened"]))
    want("artefact: 128 selected", data["n_selected"] == 128, str(data["n_selected"]))
    want("artefact: mixed fraction 16.4%",
         abs(oracle["informative_fraction"] - 0.164) < 0.001)
    want("artefact: science/medicine split",
         stats["rar_science/mixed"] == 63 and stats["rar_medicine/mixed"] == 65)


def _live_counts() -> dict[str, int]:
    run = REPO / "runs" / "rollout_v2"
    selected = {
        json.loads(line)["uid"]
        for line in (run / "examples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    counts = {}
    for source in ["shipped", "baseline", "agentic", "agentic-goldonly", "agentic-negonly"]:
        uids = set()
        for line in (run / f"rubrics_{source}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                uids.add(json.loads(line)["uid"])
        counts[source] = len(uids & selected)
    return counts


def check_generation_state() -> None:
    """§4.6.5 describes the outage cutoff, which is a moment in the past.

    Checking it against the live directory was wrong: the endpoint recovered and
    the watchdog resumed the run, so the live counts climb and a correct report
    would start "failing". What the report asserts is the *cutoff*, so that is
    what gets checked, against the snapshot recorded from git.
    """
    snapshot = json.loads(
        (REPO / "runs" / "rollout_v2" / "cutoff_state.json").read_text(encoding="utf-8")
    )
    at_cutoff = snapshot["rubrics_at_cutoff"]
    want("cutoff: shipped complete", at_cutoff["shipped"] == 128, str(at_cutoff["shipped"]))
    want("cutoff: baseline complete", at_cutoff["baseline"] == 128, str(at_cutoff["baseline"]))
    for source in ["agentic", "agentic-goldonly", "agentic-negonly"]:
        want(f"cutoff: {source} at 62", at_cutoff[source] == 62, str(at_cutoff[source]))
    want("§4.6.5 states baseline 128/128", "`rubrics_baseline` **128/128**" in REPORT)
    want("§4.6.5 states agentic 62/128", "各 **62/128**" in REPORT)

    live = _live_counts()
    if live != at_cutoff:
        print("  note: the run has progressed since the cutoff the report describes")
        for source, count in live.items():
            if count != at_cutoff[source]:
                print(f"        {source}: {at_cutoff[source]} -> {count} / {snapshot['n_selected']}")
        print("        re-run scripts/eval_rubrics.py + aggregate_results.py when it finishes,")
        print("        then replace the n=62 preview in REPORT.md 4.6.5b with the n=128 result.")


def check_preview_overlap() -> None:
    """The preview must be declared non-independent, because it is."""
    run = REPO / "runs" / "rollout_v2"
    preview = {
        json.loads(line)["uid"]
        for line in (run / "rubrics_agentic.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    pilot = {
        json.loads(line)["uid"]
        for line in (REPO / "runs" / "pilot_v2" / "examples.jsonl")
        .read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    # The preview cohort is the 62 that were complete at the cutoff, not whatever
    # is on disk now that the run has resumed.
    at_cutoff = json.loads(
        (run / "cutoff_state.json").read_text(encoding="utf-8")
    )["rubrics_at_cutoff"]["agentic"]
    overlap = len(preview & pilot)
    want("preview overlap is total", overlap >= at_cutoff,
         f"{overlap} of the {at_cutoff} preview questions are inside the pilot")
    for name, text in (("REPORT", REPORT), ("HANDOFF", HANDOFF)):
        want(f"{name} declares 100% overlap", "62/62 = 100%" in text)
        want(f"{name} says not independent evidence", "不是独立证据" in text)


def check_unfinished_is_not_a_conclusion() -> None:
    """The n=61 null must never be presented as the answer."""
    # Every place that leans on protocol 2 has to carry the power caveat.
    want("summary flags §4.6 unfinished", "评测阶段因服务端点故障中断" in REPORT)
    want("summary refuses to answer the powered question",
         "本报告没有答案" in REPORT or "本报告无法回答" in REPORT)
    want("limitation 20 exists", "§4.6 的高功效 rollout 评测没有做完" in REPORT)
    want("conclusion marks it unanswered", "评测未开始" in REPORT)
    want("README carries the same caveat", "本仓库没有答案" in README)
    want("HANDOFF calls it a power failure", "功效失败，不是发现" in HANDOFF)
    # And it must not be dressed up as a positive finding anywhere.
    for bad in ["协议二证明", "rollout 协议证实", "已经证明 agentic 更好"]:
        want(f"no overclaim: {bad}", bad not in REPORT)


def check_judgeswap_section() -> None:
    """§4.5 is transcribed from results/judgeswap/; spot-check the key figures."""
    summary_path = REPO / "results" / "judgeswap" / "SUMMARY.md"
    if not summary_path.exists():
        problems.append("results/judgeswap/SUMMARY.md missing")
        return
    source = summary_path.read_text(encoding="utf-8")

    # Each pair is (value as written in REPORT, value as written in the source).
    for value in ["+0.254", "+0.252", "4788", "n=109", "0.491"]:
        want(f"§4.5 figure {value} present in source", value in source, "not in SUMMARY.md")
        want(f"§4.5 figure {value} present in report", value in REPORT)

    for did in ["−0.002", "+0.009", "−0.015", "+0.022"]:
        want(f"§4.5 DiD {did} in report", did in REPORT)
    want("§4.5 names the swapped judge", "api_azure_openai_gpt-5.1" in REPORT)
    want("§4.5 reports zero failures", "0 失败" in REPORT)
    want("§4.5 states no self-preference", "没有自我偏好" in REPORT)
    want("§4.5 keeps head-to-head unflipped", "未翻转" in REPORT)
    want("§4.5 flags the noval flip", "翻转" in REPORT)
    want("§4.5.7 states the two votes are not independent",
         "不是独立的第三票" in REPORT or "不是独立证据" in REPORT)


def check_reading_guide() -> None:
    """The guide must route to what is finished, not to what was planned."""
    guide = REPORT.split("## 1. 动机与定位")[0]
    want("guide points at HANDOFF", "HANDOFF.md" in guide)
    want("guide marks judgeswap done", "§4.5" in guide and "已完成" in guide)
    want("guide marks §4.6 unfinished", "评测未跑完" in guide)
    want("guide flags protocol 2 n=61", "n=61" in guide or "61 题" in guide)


def check_cross_document() -> None:
    """README and HANDOFF must point at each other and agree on the headline."""
    want("README links HANDOFF", "HANDOFF.md" in README)
    want("HANDOFF links README", "README.md" in HANDOFF)
    want("HANDOFF links REPORT", "results/REPORT.md" in HANDOFF)
    for text, name in ((README, "README"), (HANDOFF, "HANDOFF")):
        want(f"{name} carries the 16.4% figure", "16.4%" in text)
    want("HANDOFF states best-of-n direction", "0.816" in HANDOFF and "0.803" in HANDOFF)
    want("README states best-of-n direction", "0.816" in README)


def check_no_stale_source_counts() -> None:
    """rollout_v2 uses five sources; the config and docs must not claim seven."""
    config = (REPO / "configs" / "rollout_v2.yaml").read_text(encoding="utf-8")
    listed = re.findall(r"^\s*-\s*(shipped|baseline|agentic[\w-]*)\s*(?:#.*)?$",
                        config, re.MULTILINE)
    want("rollout_v2 config lists 5 generators", len(listed) == 5, f"{listed}")
    want("report says sources trimmed to 5", "5（砍掉" in REPORT or "5 个" in REPORT
         or "砍掉 `agentic-noval`" in REPORT)


def main() -> int:
    check_screening()
    check_generation_state()
    check_preview_overlap()
    check_unfinished_is_not_a_conclusion()
    check_judgeswap_section()
    check_reading_guide()
    check_cross_document()
    check_no_stale_source_counts()

    print(f"checked {checked} facts across REPORT.md, README.md, HANDOFF.md")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for item in problems:
            print(f"  - {item}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
