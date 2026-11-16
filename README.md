# Measurements of the legacy fixes

These scripts made the measurements in `docs/legacy-quirks.md` of
[PR #6](https://github.com/AndrewRLawrence/dp-gp-lvm/pull/6). This tag keeps them as they ran. The tag has no parent
commit, and the scripts are not part of the `dp_gp_lvm` package.

The scripts use private functions of `dp_gp_lvm` and `tests/legacy_reference.py`. They work only with the two code
versions in "Code versions". Do not expect them to work with a later version of the library.

## Files

| File | Purpose |
|---|---|
| `jobs.py` | The data, the model, and the settings of each trained model. There are 71 jobs. The data and the seeds are the same as in the experiment commands. |
| `phase_a.py` | Trains each job once with the code on `PYTHONPATH` and scores the baseline. It saves the trained state and the random state after the training. |
| `phase_b.py` | Loads each trained model and scores each fix alone and all fixes. The CMU jobs train again with the fixed DP terms (item 4). |
| `driver.py` | Runs the jobs in parallel subprocesses, the most expensive job first. Each job writes a log to `logs/<tag>/`. |
| `patch.py`, `fast_kernels.py` | Replace `ARDRBFKernel.psi_1` and `ARDRBFKernel.psi_2` with the memory-efficient versions of PR #4. `fast_kernels.py` is the `kernels.py` of PR #4. The maths is the same, and the values differ by about 1e-13 (relative). Without them, `cmu_swapped_legs` needs 12 to 22 GB of memory. |
| `report.py` | Makes the markdown tables from `results/`. |
| `report.md` | The output of `report.py` for the measurements in the document. |

The results (36 MB) are not in the tag.

## Code versions

- **Legacy code (phase A):** `e494060`, version 1.0.0 of the PyTorch port.
- **Fixed code (phase B):** `a50872c`, the item 4 commit of PR #6. After this commit, PR #6 does not change `src` or
  `experiments`, so the squash-merge commit of PR #6 on `main` has the same library. To get `a50872c` after the merge,
  fetch `pull/6/head` from GitHub.

The measurements used the same commits before a rebase onto PR #4 and PR #5. The rebase changed only `kernels.py` (now
the same as `fast_kernels.py`) and added the `restarts` module, which the scripts do not use. A 30-step check gave the
same values on the rebased commits.

## Environment

Python 3.15.2, torch 2.15.0, numpy 2.5.4, and scipy 1.18.1 from `uv.lock`, on the CPU (macOS, arm64). `e494060` and
PR #6 have the same `uv.lock`. Each job uses two threads.

## Commands

Set `MEASURE` to the directory of these scripts. Make one checkout for each code version.

```bash
git fetch origin pull/6/head
git worktree add --detach ../dp-gp-lvm-legacy e494060
git worktree add --detach ../dp-gp-lvm-fixed a50872c
uv sync --project ../dp-gp-lvm-fixed
```

Phase A trains all models with the legacy code. With six workers, it took about 3 hours.

```bash
cd ../dp-gp-lvm-legacy
PYTHONPATH="$PWD/src:$PWD" uv run --project ../dp-gp-lvm-fixed python "$MEASURE/driver.py" --script phase_a.py --workers 6 --tag baseline
```

Phase B scores the fixes on the trained models. With six workers, it took about 2.5 hours.

```bash
cd ../dp-gp-lvm-fixed
PYTHONPATH="$PWD/src:$PWD:$PWD/tests" uv run --project . python "$MEASURE/driver.py" --script phase_b.py --workers 6 --tag fixes
```

The tables:

```bash
PYTHONPATH="$PWD/src:$PWD:$PWD/tests" uv run --project . python "$MEASURE/report.py"
```

`PYTHONPATH` selects the code version, because `uv run` installs the library of the fixed checkout. Each job prints the
path of the `dp_gp_lvm` module that it uses, and each result file records it in `source`. Each job skips its work when
its result file exists. Thus, you can stop a run and start it again.

## Method

`docs/legacy-quirks.md` ("Measurements") describes the method. In short:

- Each model trains once with the legacy code. All variants use the same trained model, because only item 4 changes
  the training.
- Items 1 and 2 change only the covariance, so they use the q(X\*) of the baseline. Items 5 and 7 change the test
  bound, so phase B fits q(X\*) again from the same initial values and random state as the baseline fit.
- Phase B makes the legacy variants from `tests/legacy_reference.py`, and the fixed variants from the library.
