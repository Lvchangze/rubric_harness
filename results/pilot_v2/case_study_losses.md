### Case `sci-a951f8c8c6bb` — rar_science / INFLYTECH/SCP-116K

**题目**

```text
Which salt can furnish H+ in its aqueous solution?
(a) NaH2PO2  (b) Na2HPO3  
(c) Na2HPO4   (d) All of these
```

**agentic 中间证据摘要**

```json
{
  "stages_failed": []
}
```

**`shipped` rubric（7 条）**

1. **[Essential w=5] Acidic Salt Definition** — The response must define an acid salt as a salt capable of furnishing H+ in its aqueous solution, establishing the fundamental concept for the answer.
2. **[Important w=5] Option Analysis** — The answer must analyze each provided salt and distinguish between normal salts and an acid salt based on the number of hydrogen ions in their formulas.
3. **[Essential w=5] Final Answer Identification** — The response must explicitly identify (c) as the correct answer, clearly stating that Na2HPO4 is an acid salt because it contains an acidic proton.
4. **[Important w=4] Chemical Composition Reasoning** — The answer should break down the reasoning by explaining that NaH2PO2 and Na2HPO3 are normal salts derived from mono- and dibasic acids, while Na2HPO4 retains one ionizable hydrogen.
5. **[Optional w=3] Detail on Acid Strength** — The response could include a discussion of the acidic properties of H3PO2 and H3PO3 to illustrate why the other salts do not furnish H+ in solution.
6. **[Optional w=2] Concise Clarity** — The answer should be concise and clear, avoiding unnecessary explanation and focusing directly on how the chemical compositions lead to the correct answer.
7. **[Pitfall w=-1] Pitfall Overgeneralization** — The response must avoid the mistake of overgeneralizing that all options are acid salts without providing specific chemical justifications for the distinctions made.

**`baseline` rubric（7 条）**

1. **[Essential w=5] Correct answer choice** — Identifies (c) Na2HPO4 as the salt that can furnish H+ in its aqueous solution.
2. **[Important w=4] H3PO2 basicity analysis** — Explains that H3PO2 is a monobasic acid, making NaH2PO2 a normal salt that cannot furnish H+.
3. **[Important w=4] H3PO3 basicity analysis** — Explains that H3PO3 is a dibasic acid, making Na2HPO3 a normal salt that cannot furnish H+.
4. **[Important w=4] H3PO4 basicity analysis** — Explains that H3PO4 is a tribasic acid and Na2HPO4 is an acidic salt because it retains one ionizable hydrogen attached to oxygen.
5. **[Optional w=2] Structural distinction** — Mentions that hydrogens directly bonded to phosphorus are non-acidic, whereas hydrogens bonded to oxygen are acidic and can be donated as H+.
6. **[Optional w=2] Clear comparison** — Clearly contrasts why the salts in options (a) and (b) fail to donate H+ while the salt in option (c) succeeds.
7. **[Pitfall w=-2] Assuming all H+ acidic** — Incorrectly assumes that any hydrogen atom present in a chemical formula can be donated as an H+ ion in aqueous solution.

**`agentic` rubric（8 条）**

1. **[Essential w=5] Selects option (c)** — The response selects option (c) Na2HPO4 as the salt that can furnish H+ in aqueous solution.
   - <sub>来源证据: `pitfall` · 验证: gold_pass=True, negatives_failed=2/2, discrimination=1.0</sub>
2. **[Important w=4] H3PO2 is monobasic** — The response identifies H3PO2 as a monobasic acid, meaning only one of its hydrogens is ionizable.
   - <sub>来源证据: `collective_error` · 验证: gold_pass=True, negatives_failed=2/2, discrimination=1.0</sub>
3. **[Important w=4] H3PO3 is dibasic** — The response identifies H3PO3 as a dibasic acid, meaning only two of its hydrogens are ionizable.
   - <sub>来源证据: `collective_error` · 验证: gold_pass=True, negatives_failed=2/2, discrimination=1.0</sub>
4. **[Important w=4] NaH2PO2 is normal salt** — The response states that NaH2PO2 is a normal salt with no ionizable proton remaining.
   - <sub>来源证据: `reference_only` · 验证: gold_pass=True, negatives_failed=2/2, discrimination=1.0</sub>
5. **[Important w=4] Na2HPO3 is normal salt** — The response states that Na2HPO3 is a normal salt with no ionizable proton remaining, since its hydrogen is P-H not O-H.
   - <sub>来源证据: `pitfall` · 验证: gold_pass=True, negatives_failed=2/2, discrimination=1.0</sub>
6. **[Important w=2] Na2HPO4 is acid-salt** — The response states that Na2HPO4 is an acid-salt.
   - <sub>来源证据: `reference_only` · 验证: gold_pass=True, negatives_failed=0/2, discrimination=0.0</sub>
7. **[Important w=3] Na2HPO4 retains one proton** — The response states that Na2HPO4 retains one ionizable proton in the compound.
   - <sub>来源证据: `reference_only` · 验证: gold_pass=True, negatives_failed=0/2, discrimination=0.0</sub>
8. **[Important w=3] P-H non-ionizable in Na2HPO3** — The response explains that the hydrogen in Na2HPO3 is P-H and therefore non-ionizable, distinguishing it from the O-H in Na2HPO4.
   - <sub>来源证据: `pitfall` · 验证: gold_pass=True, negatives_failed=2/2, discrimination=1.0</sub>
