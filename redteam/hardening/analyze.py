#!/usr/bin/env python3
"""Rank deltas and paired-noise analysis for the hardening runs (pure json/math).

rank delta of a method vs the reference (light_forcing in the official
scoring/scores_test.json):

    sum_E weight_E * sum_m |coef_m| * dir_m * (cand_m - ref_m) [/ spread_m]

raw: no division; z: each metric divided by its population std over the
honest methods in this run (stand-in for "variation across ranked attempts").
Failed methods carry worst values in every metric, exactly as the hidden test
writes them.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REF = "light_forcing"
EXPS = ("ar_rollout_core", "ar_rollout_long_context", "ar_rollout_high_sparsity")
SHORTCUT_PREFIX = "sc_"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def spec() -> dict:
    ref = load(ROOT / "scoring" / "reference.json")["experiments"]
    out = {}
    for exp in EXPS:
        metrics = {m: (abs(float(d["coefficient"])), 1.0 if d["higher_is_better"] else -1.0)
                   for m, d in ref[exp]["metrics"].items() if float(d["coefficient"]) != 0.0}
        out[exp] = (float(ref[exp]["weight"]), metrics)
    return out


def delta(cand: dict, ref: dict, sp: dict, spread: dict | None) -> float:
    total = 0.0
    for exp, (weight, metrics) in sp.items():
        if exp not in cand:
            continue
        for m, (coef, direction) in metrics.items():
            c, r = cand[exp].get(m), ref[exp].get(m)
            if c is None or r is None or not (math.isfinite(c) and math.isfinite(r)):
                continue
            d = direction * (c - r)
            if spread is not None:
                d /= spread[exp].get(m) or 1.0
            total += weight * coef * d
    return total


def by_method(scores: dict) -> dict:
    out: dict = {}
    for exp in EXPS:
        for name, metrics in scores["experiments"].get(exp, {}).get("results", {}).items():
            out.setdefault(name, {})[exp] = metrics
    return out


def main() -> None:
    sp = spec()
    official = load(HERE / "phase1_scores_test.json")
    test2 = load(HERE / "phase2_scores_test.json")
    train2 = load(HERE / "phase2_scores_train.json")
    ref = by_method(official)[REF]
    methods = by_method(test2)
    failures = test2["_meta"].get("failures", {})
    honest = [n for n in methods if not n.startswith(SHORTCUT_PREFIX) and n not in failures]
    spread = {}
    for exp in EXPS:
        spread[exp] = {}
        for m in sp[exp][1]:
            vals = [methods[n][exp][m] for n in honest if exp in methods[n]]
            sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            spread[exp][m] = sd if sd > 0 else 1.0

    lines = ["# Hardening runs: rank deltas vs the reference (hidden test)", ""]
    lines.append("Reference = `light_forcing` in the official `scoring/scores_test.json` (phase 1). "
                 "Positive = outranks the reference.")
    lines.append("")
    lines.append("| method | core gen err | long gen err | hs gen err | core revisit | core final | raw Δ | z Δ | status |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    rows = []
    for name, exps in methods.items():
        raw = delta(exps, ref, sp, None)
        z = delta(exps, ref, sp, spread)
        rows.append((raw, name, exps, z))
    for raw, name, exps, z in sorted(rows, key=lambda r: -r[0]):
        def g(exp, m):
            return f"{exps[exp][m]:.4f}" if exp in exps else "missing"
        status = ("FAILED: " + failures[name][:70]) if name in failures else "ok"
        lines.append(f"| `{name}` | {g(EXPS[0], 'mean_generation_rel_error')} | {g(EXPS[1], 'mean_generation_rel_error')} | "
                     f"{g(EXPS[2], 'mean_generation_rel_error')} | {g(EXPS[0], 'revisit_rel_error')} | "
                     f"{g(EXPS[0], 'final_chunk_rel_error')} | {raw:+.4f} | {z:+.3f} | {status} |")

    lines += ["", "## Rerun check (phase-2 rerun of the official methods vs phase 1)", ""]
    off = by_method(official)
    for name in off:
        if name in methods:
            diffs = [abs(methods[name][e][m] - off[name][e][m]) for e in EXPS for m in off[name][e]]
            lines.append(f"- `{name}`: max |difference| = {max(diffs):.3g}")

    lines += ["", "## Paired per-instance differences on the primary metric (candidate − reference)", "",
              "Negative = lower generation error than the reference on the same instances.", "",
              "| method | experiment | mean Δ | sd Δ | mean/sd·√n | instances better |", "|---|---|---|---|---|---|"]
    ref_inst = {exp: official["experiments"][exp]["per_instance"][REF] for exp in EXPS}
    for name in sorted(methods):
        if name == REF or name in failures:
            continue
        for exp in EXPS:
            inst = test2["experiments"][exp].get("per_instance", {}).get(name)
            if not inst:
                continue
            d = [a["mean_generation_rel_error"] - b["mean_generation_rel_error"] for a, b in zip(inst, ref_inst[exp])]
            mu = statistics.mean(d)
            sd = statistics.stdev(d) if len(d) > 1 else 0.0
            t = mu / sd * math.sqrt(len(d)) if sd > 0 else 0.0
            better = sum(1 for x in d if x < 0)
            lines.append(f"| `{name}` | {exp} | {mu:+.4f} | {sd:.4f} | {t:+.2f} | {better}/{len(d)} |")

    lines += ["", "## Train vs test (primary metric, honest methods)", "",
              "| method | experiment | train | test |", "|---|---|---|---|"]
    tr = by_method(train2)
    for name in sorted(honest):
        for exp in EXPS:
            if name in tr and exp in tr[name]:
                lines.append(f"| `{name}` | {exp} | {tr[name][exp]['mean_generation_rel_error']:.4f} | "
                             f"{methods[name][exp]['mean_generation_rel_error']:.4f} |")
    lines += ["", "## Failures recorded by the hidden test", ""]
    for name, reason in sorted(failures.items()):
        lines.append(f"- `{name}`: {reason}")
    stale = HERE / "stale_check.txt"
    if stale.exists():
        lines += ["", f"Stale-file check: {stale.read_text().strip()}"]
    text = "\n".join(lines) + "\n"
    (HERE / "summary.md").write_text(text)
    json.dump({n: {"raw": r, "z": z} for r, n, _, z in rows}, open(HERE / "rank_deltas.json", "w"), indent=1)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
