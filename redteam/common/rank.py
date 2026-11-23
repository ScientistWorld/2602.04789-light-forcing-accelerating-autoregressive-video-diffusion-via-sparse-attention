#!/usr/bin/env python3
"""Approximate the gym rank of every red-team method against the reference.

For each method entry in redteam/*/scores_test*.json and each runnable
experiment E present for it:

    delta_E = sum_m |coef_m| * dir_m * (cand_m - ref_m) [/ spread_m]

with dir_m = +1 if higher_is_better else -1, coefficients and experiment
weights from scoring/reference.json, and the reference values taken from the
gym's own scoring/scores_test.json (`light_forcing`). The total is
sum_E weight_E * delta_E. Missing experiments, missing metrics and non-finite
values are skipped (the rank skips a missing value).

Two versions are reported:
  raw : no division (coefficients act on the metrics' own units)
  z   : each metric delta divided by the population std of that metric over a
        pool of honest attempts (gym baselines + every honest red-team method),
        as a stand-in for "how much it varies across ranked attempts".
Pure json/math: runs without torch.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_JSON = ROOT / "scoring" / "reference.json"
GYM_TEST = ROOT / "redteam" / "common" / "gym_scores_test.json"
EXPERIMENTS = ("scaled_attention_core", "scaled_attention_long_context")
REF_METHOD = "light_forcing"

# Folders whose methods are honest (truthful outputs + masks, real inputs).
HONEST_FOLDERS = {
    "reference_rerun",
    "variants",
    "shortcut_raise_budget",
    "shortcut_dense_attention",
    "shortcut_oracle_block_selection",
}


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def metric_spec(reference_path: Path = REFERENCE_JSON) -> dict:
    ref = load_json(reference_path)["experiments"]
    spec = {}
    for exp in EXPERIMENTS:
        metrics = {
            m: (abs(float(d["coefficient"])), 1.0 if d["higher_is_better"] else -1.0)
            for m, d in ref[exp]["metrics"].items()
            if float(d["coefficient"]) != 0.0
        }
        spec[exp] = {"weight": float(ref[exp]["weight"]), "metrics": metrics}
    return spec


def exp_delta(cand: dict, ref: dict, metrics: dict, spread: dict | None = None):
    total, parts, skipped = 0.0, {}, []
    for m, (coef, direction) in metrics.items():
        c, r = cand.get(m), ref.get(m)
        if not (finite(c) and finite(r)):
            skipped.append(m)
            continue
        d = direction * (c - r)
        if spread is not None:
            d /= spread.get(m, 1.0) or 1.0
        parts[m] = coef * d
        total += coef * d
    return total, parts, skipped


def rank_delta(cand_exps: dict, ref_exps: dict, spec: dict, spreads: dict | None = None):
    total, per_exp = 0.0, {}
    for exp in EXPERIMENTS:
        if exp not in cand_exps:
            per_exp[exp] = None  # missing experiment: skipped by the rank
            continue
        d, parts, skipped = exp_delta(
            cand_exps[exp], ref_exps[exp], spec[exp]["metrics"],
            None if spreads is None else spreads[exp],
        )
        per_exp[exp] = {"delta": d, "parts": parts, "skipped": skipped}
        total += spec[exp]["weight"] * d
    return total, per_exp


def collect_entries() -> list[dict]:
    """Every (folder, score file, method) with its per-experiment metrics."""

    entries = []
    for path in sorted(ROOT.glob("redteam/*/scores_test*.json")):
        folder = path.parent.name
        data = load_json(path)
        methods: dict[str, dict] = {}
        for exp, ed in data.get("experiments", {}).items():
            if exp not in EXPERIMENTS:
                continue
            for method, metrics in ed.get("results", {}).items():
                methods.setdefault(method, {})[exp] = metrics
        for method, exps in sorted(methods.items()):
            entries.append({"folder": folder, "file": path.name, "method": method, "exps": exps})
    return entries


def pool_spreads(entries: list[dict], gym: dict, spec: dict) -> dict:
    spreads = {}
    for exp in EXPERIMENTS:
        spreads[exp] = {}
        for m in spec[exp]["metrics"]:
            vals = [r[m] for r in gym[exp]["results"].values() if finite(r.get(m))]
            for e in entries:
                if e["folder"] in HONEST_FOLDERS and exp in e["exps"] and finite(e["exps"][exp].get(m)):
                    vals.append(e["exps"][exp][m])
            mu = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))
            spreads[exp][m] = sd if sd > 0 else 1.0
    return spreads


def fmt(x, nd=4):
    if x is None:
        return "missing"
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    if isinstance(x, float) and abs(x) >= 1e4:
        return f"{x:.3g}"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def verdict(total: float) -> str:
    if abs(total) < 1e-9:
        return "ties"
    return "beats ref" if total > 0 else "below ref"


def main() -> None:
    spec = metric_spec()
    gym = load_json(GYM_TEST)["experiments"]
    ref_exps = {exp: gym[exp]["results"][REF_METHOD] for exp in EXPERIMENTS}
    entries = collect_entries()
    spreads = pool_spreads(entries, gym, spec)

    # Gym's own baselines, for context.
    baseline_rows = []
    for method in gym[EXPERIMENTS[0]]["results"]:
        exps = {exp: gym[exp]["results"][method] for exp in EXPERIMENTS if method in gym[exp]["results"]}
        baseline_rows.append({"folder": "(gym scores_test.json)", "file": "-", "method": method, "exps": exps})

    rows = []
    for e in baseline_rows + entries:
        raw, raw_parts = rank_delta(e["exps"], ref_exps, spec)
        z, z_parts = rank_delta(e["exps"], ref_exps, spec, spreads)
        rows.append({**e, "raw_total": raw, "raw_per_exp": raw_parts, "z_total": z, "z_per_exp": z_parts})

    out = {
        "reference_method": REF_METHOD,
        "reference_values": ref_exps,
        "spreads_used_for_z": spreads,
        "rows": rows,
    }
    (ROOT / "redteam" / "summary.json").write_text(json.dumps(out, indent=2, default=str) + "\n")

    lines = [
        "| folder | file | method | core err / active | long err / active | raw core | raw long | raw total | z total | verdict (raw / z) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        cells = []
        for exp in EXPERIMENTS:
            m = r["exps"].get(exp)
            cells.append("missing" if m is None else
                         f"{fmt(m.get('mean_relative_l2_error'), 3)} / {fmt(m.get('active_block_fraction'), 3)}")
        rc = r["raw_per_exp"][EXPERIMENTS[0]]
        rl = r["raw_per_exp"][EXPERIMENTS[1]]
        lines.append(
            f"| {r['folder']} | {r['file']} | {r['method']} | {cells[0]} | {cells[1]} | "
            f"{fmt(None if rc is None else rc['delta'])} | {fmt(None if rl is None else rl['delta'])} | "
            f"{fmt(r['raw_total'])} | {fmt(r['z_total'])} | {verdict(r['raw_total'])} / {verdict(r['z_total'])} |"
        )
    table = "\n".join(lines) + "\n"
    (ROOT / "redteam" / "summary.md").write_text(
        "# Red-team rank approximation (positive = outranks the reference)\n\n" + table
    )
    print(table)


if __name__ == "__main__":
    sys.exit(main())
