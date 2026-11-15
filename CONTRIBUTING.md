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

## Golden values

The files in `tests/fixtures/legacy_tf1` come from the legacy TensorFlow 1.15 code. Do not change them by hand.
To make them again, start Docker and run this command from the repository root:

```bash
./scripts/legacy_golden/run.sh
```
