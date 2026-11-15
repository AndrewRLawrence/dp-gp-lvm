#!/usr/bin/env bash
# Generate golden values from the legacy TensorFlow 1.15 code.
# Run this script from the repository root. It needs Docker and git.
# The image is linux/amd64. On Apple silicon, Docker runs it with emulation.
set -euo pipefail

legacy_ref="${LEGACY_REF:-legacy-tf1}"
out_dir="${OUT_DIR:-tests/fixtures/legacy_tf1}"
image="dp-gp-lvm-legacy:tf1.15.5"

work_dir="$(mktemp -d)"
trap 'rm -rf "${work_dir}"' EXIT

# The legacy code finds its root by a sys.path entry that contains "dp_gp_lvm".
mkdir -p "${work_dir}/dp_gp_lvm"
git archive "${legacy_ref}" src | tar -x -C "${work_dir}/dp_gp_lvm"

docker build --platform linux/amd64 -t "${image}" scripts/legacy_golden
mkdir -p "${out_dir}"

docker run --rm --platform linux/amd64 \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e LEGACY_COMMIT="$(git rev-parse "${legacy_ref}^{commit}")" \
  -e BASE_IMAGE="$(sed -n 's/^FROM //p' scripts/legacy_golden/Dockerfile)" \
  -v "${work_dir}/dp_gp_lvm:/work/dp_gp_lvm:ro" \
  -v "${PWD}/scripts/legacy_golden:/scripts:ro" \
  -v "${PWD}/${out_dir}:/out" \
  "${image}" python /scripts/generate.py
