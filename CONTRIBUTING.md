# Contributing

This file gives the workflow and the writing rules for this repository.

## Workflow

1. Install [uv](https://docs.astral.sh/uv/).
2. Install Python and the dependencies:

   ```bash
   uv python install 3.15
   ```

   ```bash
   uv sync
   ```

3. Install the git hooks:

   ```bash
   uv run pre-commit install
   ```

4. Make a branch from `main`.
5. Make your change. Add tests for each change in behaviour.
6. Run the checks:

   ```bash
   uv run pre-commit run --all-files
   ```

   ```bash
   uv run pytest --cov
   ```

7. Open a pull request. Use the sections in [Pull requests](#pull-requests).

## Rules for the maths

- Do not change the maths of a model without a review. The parity tests compare the models with the legacy code.
- If you fix a known issue from `CHANGELOG.md`, make one commit for each fix. Add a test and a changelog entry.
- Keep float64 as the default type.

## Dependencies

- Add runtime dependencies to `[project.dependencies]` only if the models need them.
- Add tools to a dependency group: `test`, `lint`, or `experiments`.
- Give each dependency a range: a lower bound that CI tests and an upper bound at the next major version.
- Run `uv lock` after each change to `pyproject.toml`.

## Writing style

The README, the docstrings, and the pull request descriptions follow
[ASD-STE100 Simplified Technical English](https://www.asd-ste100.org/) to a reasonable level. Use these rules:

- Write short sentences. Use not more than 20 words in an instruction and 25 words in a description.
- Write one instruction in each sentence. Put the instructions in the order that the reader does them.
- Use the active voice. For example, write "The model learns the groups", not "The groups are learned".
- Use the simple present tense for descriptions. Use the imperative for instructions.
- Use one word for one concept. For example, always write "observed dimension", not sometimes "output".
- Do not use phrasal verbs when a simple verb is available. For example, write "remove", not "take out".
- Write lists for sequences and for sets of items.
- Write docstrings in the numpy style. Give the shape of each array argument, for example `(N, Q)`.

## Pull requests

Write the pull request description with these sections:

- **Summary**: one or two sentences about the change.
- **Changes**: a list of the changes.
- **Verification**: the commands that you ran and their results.
- **Known issues**: the problems that remain, if any.

## Releases

### Versions

The versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and
[PEP 440](https://peps.python.org/pep-0440/). The `version` in `pyproject.toml` is the only source of the version.
`dp_gp_lvm.__version__` reads it from the package metadata.

- Increase the major version for a change to the public API that is not compatible.
- Increase the minor version for a new feature, or for a new default that changes the results. In the changelog,
  tell the user how to get the old behaviour.
- Increase the patch version for a bug fix. In the changelog, give the effect of the fix on the results.

### Dev builds

`.github/workflows/release.yml` builds and checks the distributions for each pull request. Each push to `main` also
uploads a dev build to [TestPyPI](https://test.pypi.org/project/dp-gp-lvm/):

- The version is `X.Y.Z.devN`. `N` is the number of commits on `main`, so each merged pull request has a different
  `N`.
- `X.Y.Z` is the version in `pyproject.toml`. If that version has a release tag, `X.Y.Z` is the next patch version.
- After a pre-release tag, for example `v1.0.0rc0`, the dev version is `1.0.0rc1.devN`, until the tag `v1.0.0`. Thus
  each dev version sorts after the last tag and before the next release.

To test a dev build, install it into an environment that already has the dependencies, for example the development
environment. Do not get the dependencies from TestPyPI, because TestPyPI does not check its packages.

```bash
uv pip install --no-deps --default-index https://test.pypi.org/simple/ "dp-gp-lvm==1.0.1.dev15"
```

### Release checklist

1. Make a branch from `main`.
2. Set the new version. Use `major`, `minor`, or `patch`, as [Versions](#versions) tells. The command also updates
   `uv.lock`.

   ```bash
   uv version --bump minor
   ```

   For a pre-release, set the version directly. Use the PEP 440 form for the version and the tag, for example
   `1.0.0rc0` and `v1.0.0rc0`. The workflow rejects other forms, such as `v1.0.0-rc.0`. It marks the GitHub release
   as a pre-release.

   ```bash
   uv version 1.0.0rc0
   ```

3. In `CHANGELOG.md`, change `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`. Add a new empty `## [Unreleased]`
   section above it. Update the links at the end of the file.
4. In `CITATION.cff`, set `version: X.Y.Z` and `date-released: "YYYY-MM-DD"`. Use the date from the changelog.
5. Run the checks. `tests/test_ci_scripts.py` checks that `pyproject.toml`, `CHANGELOG.md`, and `CITATION.cff` agree.
6. Open a pull request with the title `🔖 Release X.Y.Z`. Merge it when CI passes.
7. Wait until CI passes on `main`.
8. Make an annotated tag on the merge commit. Then push the tag. Only a repository admin can push a `v*` tag.

   ```bash
   git switch main && git pull --ff-only
   ```

   ```bash
   git tag -a vX.Y.Z -m "dp-gp-lvm X.Y.Z"
   ```

   ```bash
   git push origin vX.Y.Z
   ```

9. Open the run of the "Release" workflow in the Actions tab. Approve the `pypi` deployment. The workflow uploads the
   distributions to PyPI. Then it makes the GitHub release with the notes from `CHANGELOG.md`.
10. Check the release on [PyPI](https://pypi.org/project/dp-gp-lvm/) and on GitHub.

The workflow checks that the tag is on `main`, and that the tag agrees with `pyproject.toml`, `CHANGELOG.md`, and
`CITATION.cff`. If a check fails, the workflow uploads nothing. Correct the cause in a new pull request. Then delete
the tag, and tag the new commit of `main`.

If the upload fails, for example because of the trusted publisher settings on PyPI, correct the cause. Then re-run
the failed jobs. The publish jobs upload the same files again. You cannot replace a file on PyPI, even after you
delete it. To withdraw a bad release, yank it on PyPI and release a new patch version.

The workflow uses Trusted Publishing, so you do not need a PyPI token. Do not put a token in a repository secret or
in `.env`.

## Golden values

The files in `tests/fixtures/legacy_tf1` come from the legacy TensorFlow 1.15 code. Do not change them by hand.
To make them again, start Docker and run this command from the repository root:

```bash
./scripts/legacy_golden/run.sh
```
