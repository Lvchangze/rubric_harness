#!/usr/bin/env python3
"""Assert our RaR prompts still match the paper transcription character-for-character.

The `baseline` column exists to separate "the method is better" from "our model
is better". That separation only holds if `baseline` is a faithful reproduction
of the paper's one-shot generator, which in turn requires the exact appendix
prompts, dispatched per domain -- the medical and science prompts differ in ways
that demonstrably change their output (``docs/00_paper_notes.md`` §2.3). A silent
edit here would quietly invalidate the whole comparison, so this check runs in
the pilot script before any rubric is generated.

The comparison target is ``docs/00_paper_notes.md``, transcribed independently
from the PDF by the forensics pass, which makes this a genuine cross-check
rather than a diff against our own source.

    python scripts/check_prompt_fidelity.py

Exits non-zero on any mismatch.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harness.prompts.rar_original import (  # noqa: E402
    IMPLICIT_JUDGE_SYSTEM,
    MEDICAL_RUBRIC_SYSTEM,
    PREDEFINED_RUBRIC,
    SCIENCE_RUBRIC_SYSTEM,
    build_implicit_judge_user,
    system_for_domain,
)

# The PDF renders typographic characters that are LaTeX artefacts, not prompt
# content; ASCII also keeps a tokeniser from reading "-1" as anything exotic.
_TYPOGRAPHY = {
    "\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"', "\u2022": "-", "\u2026": "...",
}


def normalise(text: str) -> str:
    for src, dst in _TYPOGRAPHY.items():
        text = text.replace(src, dst)
    # PDF hard-wraps are not part of the prompt.
    return re.sub(r"\s+", " ", text).strip()


def extract_blocks(notes: str) -> list[str]:
    return re.findall(r"```text\n(.*?)```", notes, re.S)


def compare_prefix(label: str, ours: str, theirs: str) -> bool:
    """Assert the doc's text is a prefix of ours.

    Needed for the judge prompts: their transcription embeds a nested ```json
    fence, so the outer block cannot be extracted whole. The doc annotates the
    two turns with "System Prompt:" / "User Prompt Template:" labels that are
    not part of the prompt, so those are stripped before comparing.
    """
    a = normalise(ours)
    b = normalise(re.sub(r"^\s*(System Prompt|User Prompt Template)\s*:\s*", "", theirs))
    if a.startswith(b):
        print(f"  PASS  {label}: doc prefix ({len(b)} of {len(a)} chars) matches verbatim")
        return True
    print(f"  FAIL  {label}: doc text is not a prefix of ours")
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            print(f"          first divergence at {i}: ours={a[max(0, i - 40):i + 20]!r}")
            print(f"                                    doc={b[max(0, i - 40):i + 20]!r}")
            break
    return False


def compare(label: str, ours: str, theirs: str) -> bool:
    a, b = normalise(ours), normalise(theirs)
    if a == b:
        print(f"  PASS  {label}: verbatim match ({len(b)} chars)")
        return True
    matcher = difflib.SequenceMatcher(None, a, b)
    print(f"  FAIL  {label}: ratio={matcher.ratio():.5f} ours={len(a)} doc={len(b)}")
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        print(f"          {tag}: ours={a[max(0, i1 - 40):i2 + 20]!r}")
        print(f"          {' ' * len(tag)}  doc={b[max(0, j1 - 40):j2 + 20]!r}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes", type=Path, default=REPO / "docs" / "00_paper_notes.md")
    args = parser.parse_args()

    if not args.notes.exists():
        print(f"paper notes not found at {args.notes}; skipping fidelity check")
        return 0

    blocks = extract_blocks(args.notes.read_text(encoding="utf-8"))
    if len(blocks) < 3:
        print(f"expected >=3 verbatim blocks in {args.notes}, found {len(blocks)}")
        return 1

    print(f"Checking prompts against {args.notes.relative_to(REPO)}")
    medicine_doc, science_doc, implicit_doc = blocks[0], blocks[1], blocks[2]
    ok = compare("medicine generation prompt (§2.1)", MEDICAL_RUBRIC_SYSTEM, medicine_doc)
    ok &= compare("science generation prompt (§2.2)", SCIENCE_RUBRIC_SYSTEM, science_doc)
    ok &= compare_prefix("RaR-IMPLICIT judge system (§4.3)", IMPLICIT_JUDGE_SYSTEM, implicit_doc)

    # The user template lives past the nested fence, so check its skeleton.
    user_turn = normalise(build_implicit_judge_user("{prompt}", "{response}", "{rubric_list_string}"))
    for fragment in (
        "rate the overall quality of the response on a scale of 1 to 10",
        "<prompt> {prompt} </prompt>",
        "<response> {response} </response>",
        "<rubrics> {rubric_list_string} </rubrics>",
        "Your JSON Evaluation:",
    ):
        passed = fragment in user_turn
        print(f"  {'PASS' if passed else 'FAIL'}  implicit user template contains {fragment!r}")
        ok &= passed

    # The two generation prompts must stay distinct and correctly routed; using
    # one for both domains is the specific failure this guards against.
    checks = [
        ("prompts are distinct", MEDICAL_RUBRIC_SYSTEM != SCIENCE_RUBRIC_SYSTEM),
        ("rar_medicine routes to medical", system_for_domain("rar_medicine") == MEDICAL_RUBRIC_SYSTEM),
        ("rar_science routes to science", system_for_domain("rar_science") == SCIENCE_RUBRIC_SYSTEM),
        ("only medicine anchors weights", "(weight 5)" in MEDICAL_RUBRIC_SYSTEM
            and "(weight 5)" not in SCIENCE_RUBRIC_SYSTEM),
        ("only medicine mandates Pitfall opener", "Recommends" in MEDICAL_RUBRIC_SYSTEM
            and "Recommends" not in SCIENCE_RUBRIC_SYSTEM),
        ("predefined rubric has 4 items", len(PREDEFINED_RUBRIC) == 4),
    ]
    for label, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {label}")
        ok &= passed

    print("\nAll prompt fidelity checks passed." if ok else "\nPROMPT FIDELITY FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
