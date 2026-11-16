"""Job definitions for the measurement runs: data, model builder, and settings of each trained model.

Each job trains one model once. The training does not depend on the prediction fixes, so all prediction variants
reuse the same trained model. The data and the seeds are the same as in the experiment command-line interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from dp_gp_lvm import DPGPLVM, BayesianGPLVM
from experiments import cmu_swapped_legs, frey_faces, oivae_cmu, synthetic
from experiments._common import block_permute, split_missing, standardise
from experiments._data import cmu_subject7
from experiments._data import cmu_swapped_legs as load_cmu_legs
from experiments._data import frey_faces as load_frey

SYNTHETIC_SEEDS = (10, 11, 12, 13, 14)
PERCENT_SAMPLES = (0.8, 0.9)
PERCENT_DIMS = (0.7, 0.8, 0.9)
FREY_SEEDS = (0, 1, 2, 3, 4)
OIVAE_SEQUENCES = oivae_cmu.TRAINING_SEQUENCES
CMU_LABELS = ("dp_gp_lvm", "dp_gp_lvm_mask_93")


@dataclass
class Job:
    name: str
    group: str
    label: str
    kind: str  # "missing", "new", or "train"
    build: Callable[[], Any]
    iterations: int
    predict_iterations: int
    learning_rate: float
    tests: dict[str, Any] = field(default_factory=dict)
    cost: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)


def synthetic_jobs() -> list[Job]:
    jobs = []
    for variant in ("two-func", "four-func"):
        v = synthetic.VARIANTS[variant]
        d = v.defaults
        y, _ = synthetic.make_data(v)
        for seed in SYNTHETIC_SEEDS:
            permuted = block_permute(y, seed)
            for n in PERCENT_SAMPLES:
                y_train = split_missing(permuted, n, PERCENT_DIMS[0])[0]
                tests = {}
                for dd in PERCENT_DIMS:
                    train_dd, observed, unobserved = split_missing(permuted, n, dd)
                    assert np.array_equal(train_dd, y_train)
                    tests[f"d{dd}"] = (observed, unobserved)
                for label in ("bgplvm", "dp_gp_lvm"):

                    def build(label=label, y_train=y_train, d=d, seed=seed):
                        if label == "bgplvm":
                            return BayesianGPLVM(y_train, d.latent_dims, d.inducing_points, seed=seed)
                        return DPGPLVM(y_train, d.latent_dims, d.inducing_points, d.truncation_level, seed=seed)

                    cost = (10.0 if variant == "four-func" else 2.0) * (5.0 if label == "dp_gp_lvm" else 1.0)
                    jobs.append(
                        Job(
                            f"synthetic_{variant}_{label}_seed{seed}_n{n}",
                            f"synthetic {variant}",
                            label,
                            "missing",
                            build,
                            d.iterations,
                            d.predict_iterations,
                            d.learning_rate,
                            tests,
                            cost,
                            {"seed": seed, "n": n},
                        )
                    )
    return jobs


def frey_jobs() -> list[Job]:
    jobs = []
    faces = load_frey()
    for variant in ("prediction", "split"):
        v = frey_faces.VARIANTS[variant]
        d = v.defaults
        for seed in FREY_SEEDS:
            y_train, observed, unobserved = frey_faces.prepare(faces, v, seed, v.num_train, 10, 0.5)
            available = frey_faces.builders(
                y_train, observed.shape[1], d.latent_dims, d.inducing_points, d.truncation_level
            )
            for label in v.models:
                jobs.append(
                    Job(
                        f"frey_{variant}_{label}_seed{seed}",
                        f"frey {variant}",
                        label,
                        "missing",
                        available[label],
                        d.iterations,
                        d.predict_iterations,
                        d.learning_rate,
                        {"": (observed, unobserved)},
                        30.0 if label == "dp_gp_lvm" else 2.0,
                        {"seed": seed},
                    )
                )
    return jobs


def oivae_jobs() -> list[Job]:
    jobs = []
    d = oivae_cmu.MODES["test-loglikelihood"]
    for sequence in OIVAE_SEQUENCES:
        train_frames = cmu_subject7(sequence)[::3]
        test_frames = [cmu_subject7(s)[::3] for s in oivae_cmu.TEST_SEQUENCES]
        y_train, *y_tests = standardise(train_frames, *test_frames)

        def build(y_train=y_train):
            return DPGPLVM(y_train, d.latent_dims, d.inducing_points, d.truncation_level, seed=1)

        tests = {f"seq{s}": y for s, y in zip(oivae_cmu.TEST_SEQUENCES, y_tests, strict=True)}
        jobs.append(
            Job(
                f"oivae_dp_gp_lvm_seq{sequence}",
                "oivae test-loglikelihood",
                "dp_gp_lvm",
                "new",
                build,
                d.iterations,
                d.predict_iterations,
                d.learning_rate,
                tests,
                25.0,
                {"sequence": sequence},
            )
        )
    return jobs


def cmu_jobs() -> list[Job]:
    (y,) = standardise(load_cmu_legs()[::4])
    d = cmu_swapped_legs.DEFAULTS
    available = cmu_swapped_legs.builders(y, d.latent_dims, d.inducing_points, d.truncation_level)
    return [
        Job(
            f"cmu_{label}",
            "cmu swapped legs",
            label,
            "train",
            available[label],
            d.iterations,
            0,
            d.learning_rate,
            {},
            60.0,
        )
        for label in CMU_LABELS
    ]


def all_jobs() -> list[Job]:
    return synthetic_jobs() + frey_jobs() + oivae_jobs() + cmu_jobs()
