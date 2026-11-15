# Legacy TensorFlow 1.15 golden values

These files contain reference values from the legacy TensorFlow 1.15 code.
The parity tests compare the PyTorch code with these values.

## Source

- Legacy code: git tag `legacy-tf1`. `meta.json` gives the exact commit.
- Environment: `scripts/legacy_golden/Dockerfile`. `meta.json` gives the base image digest.
- Generator: `scripts/legacy_golden/generate.py`.

## Content

Each `.npz` file holds one case. The keys use these prefixes:

| Prefix | Content |
|---|---|
| `in.` | Inputs, for example `y_train` and `y_test`. |
| `var.` | Raw variable values at initialisation. Raw values are before the softplus transform. |
| `out.` | The objective and its parts at initialisation. |
| `prop.` | Model properties at initialisation, for example ARD weights. |
| `grad.` | Gradients of the objective with respect to each raw variable. |
| `predict_new.`, `predict_missing.` | Prediction outputs, gradients, and a 10-step Adam trajectory for q(X*). |
| `train.adam.` | A 20-step Adam trajectory of the training objective. |

The case `dp_gp_lvm_long_training` holds a 1000-step training run on two-group data.
It records the objective every 50 steps and the final assignments.

TensorFlow 1.15 uses multithreaded reductions. Two runs of the generator can differ by about 1e-14.

## Regenerate the files

Do not change these files by hand. To regenerate them, start Docker and run this command from the repository root:

```bash
./scripts/legacy_golden/run.sh
```

The script needs Docker and git. The image is `linux/amd64`. On Apple silicon, Docker runs it with emulation.
