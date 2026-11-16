"""Make the markdown tables of the measurement report from the results of phase A and phase B."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE))
from jobs import CMU_LABELS, FREY_SEEDS, OIVAE_SEQUENCES, PERCENT_DIMS, PERCENT_SAMPLES, SYNTHETIC_SEEDS  # noqa: E402

VARIANTS = ["baseline", "fix1", "fix2", "fix12", "fix7", "fix5", "all"]
HEADERS = ["Baseline", "Item 1", "Item 2", "Items 1+2", "Item 7", "Item 5", "All fixes"]
LABELS = {"bgplvm": "Bayesian GP-LVM", "dp_gp_lvm": "DP-GP-LVM", "mrd": "MRD"}


def load(name: str, tag: str) -> dict | None:
    path = RESULTS / name / f"{tag}.json"
    return json.loads(path.read_text()) if path.exists() else None


def mean_se(values, digits: int = 1) -> str:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return "n/a"
    se = a.std(ddof=1) / np.sqrt(a.size) if a.size > 1 else 0.0
    return f"{a.mean():.{digits}f} ± {se:.{digits}f}".replace("-", "−")


NOT_DP = {"fix1", "fix7", "fix12"}


def cells(label: str, data: dict) -> list[str]:
    return ["—" if label != "dp_gp_lvm" and k in NOT_DP else mean_se(data[k]) for k in VARIANTS]


def values(name: str, key: str) -> dict[str, float] | None:
    base, fixes = load(name, "baseline"), load(name, "fixes")
    if base is None or fixes is None:
        return None
    out = dict(fixes["tests"][key]["values"])
    assert abs(out["baseline"] - base["tests"][key]["value"]) < 1e-9 * max(1.0, abs(out["baseline"])), name
    return out


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def synthetic_summary() -> tuple[list[list[str]], dict]:
    """Rows of the per-fix table: the sum over the six (N, D) settings for each seed, mean ± SE over seeds."""
    rows, per_config = [], {}
    for variant in ("two-func", "four-func"):
        for label in ("bgplvm", "dp_gp_lvm"):
            sums = defaultdict(list)
            for seed in SYNTHETIC_SEEDS:
                total = defaultdict(float)
                complete = True
                for n in PERCENT_SAMPLES:
                    for d in PERCENT_DIMS:
                        v = values(f"synthetic_{variant}_{label}_seed{seed}_n{n}", f"d{d}")
                        if v is None:
                            complete = False
                            continue
                        per_config.setdefault((variant, label, n, d), []).append(v)
                        for k in VARIANTS:
                            total[k] += v[k]
                if complete:
                    for k in VARIANTS:
                        sums[k].append(total[k])
            count = len(sums["baseline"])
            rows.append([f"Synthetic {variant}, {LABELS[label]}", *cells(label, sums)])
    return rows, per_config


def frey_rows() -> list[list[str]]:
    rows = []
    for variant, labels in (("prediction", ("bgplvm", "dp_gp_lvm")), ("split", ("mrd", "dp_gp_lvm"))):
        for label in labels:
            got = [v for s in FREY_SEEDS if (v := values(f"frey_{variant}_{label}_seed{s}", "")) is not None]
            rows.append([f"Frey faces {variant}, {LABELS[label]}", *cells(label, {k: [g[k] for g in got] for k in VARIANTS})])
    return rows


def advantage_table(per_config: dict) -> str:
    """The DP-GP-LVM advantage for each synthetic setting: baseline and all fixes, and the paired difference."""
    rows = []
    for variant in ("two-func", "four-func"):
        for n in PERCENT_SAMPLES:
            for d in PERCENT_DIMS:
                b = per_config.get((variant, "bgplvm", n, d), [])
                p = per_config.get((variant, "dp_gp_lvm", n, d), [])
                if len(b) != len(p) or not b:
                    continue
                row = [f"{variant}", f"{1 - n:.0%}", f"{1 - d:.0%}"]
                for k in ("baseline", "all"):
                    row += [mean_se([x[k] for x in b]), mean_se([x[k] for x in p])]
                    row.append(mean_se([y[k] - x[k] for x, y in zip(b, p)]))
                rows.append(row)
    header = ["Data", "Ñ missing", "D̃ missing"]
    header += ["BGP-LVM (baseline)", "DP-GP-LVM (baseline)", "DP − BGP (baseline)"]
    header += ["BGP-LVM (all fixes)", "DP-GP-LVM (all fixes)", "DP − BGP (all fixes)"]
    return table(header, rows)


def frey_advantage() -> str:
    rows = []
    for variant, other in (("prediction", "bgplvm"), ("split", "mrd")):
        pairs = []
        for s in FREY_SEEDS:
            a, b = values(f"frey_{variant}_{other}_seed{s}", ""), values(f"frey_{variant}_dp_gp_lvm_seed{s}", "")
            if a is not None and b is not None:
                pairs.append((a, b))
        if pairs:
            rows.append(
                [
                    f"Frey faces {variant}",
                    LABELS[other],
                    mean_se([b["baseline"] - a["baseline"] for a, b in pairs]),
                    mean_se([b["all"] - a["all"] for a, b in pairs]),
                ]
            )
    return table(["Data", "Other model", "DP − other (baseline)", "DP − other (all fixes)"], rows)


def oivae_table() -> str:
    rows = []
    for key in ("seq11", "seq12"):
        cells = defaultdict(list)
        for sequence in OIVAE_SEQUENCES:
            base, fixes = load(f"oivae_dp_gp_lvm_seq{sequence}", "baseline"), load(f"oivae_dp_gp_lvm_seq{sequence}", "fixes")
            if base is None or fixes is None:
                continue
            cells["baseline"].append(base["tests"][key]["value"])
            for k, v in fixes["tests"][key]["values"].items():
                cells[k].append(v)
        n = len(cells["baseline"])
        rows.append([f"Sequence {key[3:]}", mean_se(cells["baseline"], 0), mean_se(cells["fix7"], 0), mean_se(cells["all"], 0)])
    return table(["Test sequence", "Baseline", "Item 7", "All fixes (items 3, 5, 7)"], rows)


def adjusted_rand(a: list[int], b: list[int]) -> float:
    from scipy.special import comb

    a, b = np.asarray(a), np.asarray(b)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table_ = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(table_, (ai, bi), 1)
    sum_comb = comb(table_, 2).sum()
    rows, cols = comb(table_.sum(1), 2).sum(), comb(table_.sum(0), 2).sum()
    expected = rows * cols / comb(len(a), 2)
    maximum = 0.5 * (rows + cols)
    return float((sum_comb - expected) / (maximum - expected)) if maximum != expected else 1.0


def cmu_table() -> str:
    rows = []
    for label in CMU_LABELS:
        info = load(f"cmu_{label}", "fixes")
        if info is None:
            continue
        mask = 3 if label == "dp_gp_lvm" else 93
        for key, name in (("baseline", "Baseline (legacy DP terms)"), ("fix4", "Item 4 (fixed DP terms)")):
            r = info[key]
            groups = np.asarray(r["groups"])[::mask]  # One assignment for each block.
            sizes = sorted(np.unique(groups, return_counts=True)[1].tolist(), reverse=True)
            rows.append(
                [
                    f"mask_size={mask}",
                    name,
                    f"{r['objective_legacy']:.1f}".replace("-", "−"),
                    f"{r['objective_fixed']:.1f}".replace("-", "−"),
                    f"{r['num_groups']}",
                    ", ".join(map(str, sizes)),
                ]
            )
        base_groups = np.asarray(info["baseline"]["groups"])[::mask].tolist()
        fix_groups = np.asarray(info["fix4"]["groups"])[::mask].tolist()
        rows.append([f"mask_size={mask}", "Adjusted Rand index, baseline vs item 4", "", "", "", f"{adjusted_rand(base_groups, fix_groups):.2f}"])
    header = ["Model", "Training", "Objective (legacy DP terms)", "Objective (fixed DP terms)", "Groups", "Blocks in each group"]
    return table(header, rows)


def rmse_table() -> str:
    """The error and the spread of the predictive distribution at the baseline q(X*) and at the fixed q(X*)."""
    from jobs import all_jobs

    acc: dict = defaultdict(lambda: defaultdict(list))
    for job in all_jobs():
        if job.kind != "missing":
            continue
        for key, (_, unobserved) in job.tests.items():
            for tag in ("baseline", "all"):
                z = np.load(RESULTS / job.name / f"{tag}_{key or 'test'}.npz")
                acc[(job.group, job.label)][f"rmse_{tag}"].append(np.sqrt(np.mean((z["mean"] - unobserved) ** 2)))
                variance = np.diagonal(z["covariance"], axis1=-2, axis2=-1)
                acc[(job.group, job.label)][f"sd_{tag}"].append(np.sqrt(variance.mean()))
    rows = []
    for (group, label), d in acc.items():
        name = group[0].upper() + group[1:]
        rows.append([f"{name}, {LABELS[label]}", *(f"{np.mean(d[k]):.2f}" for k in ("rmse_baseline", "rmse_all", "sd_baseline", "sd_all"))])
    return table(["Experiment, model", "RMSE (baseline)", "RMSE (all fixes)", "Predictive SD (baseline)", "Predictive SD (all fixes)"], rows)


def main() -> None:
    synthetic_rows, per_config = synthetic_summary()
    print("## Each fix alone and all fixes\n")
    print(table(["Experiment, model", *HEADERS], synthetic_rows + frey_rows()))
    print("\n## RMSE and predictive SD\n")
    print(rmse_table())
    print("\n## DP-GP-LVM advantage, synthetic\n")
    print(advantage_table(per_config))
    print("\n## DP-GP-LVM advantage, Frey faces\n")
    print(frey_advantage())
    print("\n## oi-VAE test log-likelihood\n")
    print(oivae_table())
    print("\n## CMU swapped legs\n")
    print(cmu_table())


if __name__ == "__main__":
    main()
