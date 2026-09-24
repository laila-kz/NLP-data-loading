# Contributing

Thanks for taking the time to contribute! This project is small and learner-focused, so a few simple conventions keep it that way.

## Setup

```bash
pip install -e ".[dev]"
```

## Checks to run before submitting

```bash
ruff check .              # lint
ruff format --check .     # formatting (or `ruff format .` to autofix)
pytest                    # tests (network tests opt-in via -m network)
```

Everything must pass on the GitHub Actions CI for merge.

## Code conventions

- Style 120 columns, `ruff format`, default ruff rule set (see `pyproject.toml`).
- Public functions need a one-para docstring; keep signatures fully annotated.
- **No `globals()`-based state.** Everything a function needs is an explicit argument — that is the single most important design rule in this repo.
- Tutorial scripts are just that: runnable, heavily-commented walkthroughs that **import** from `nlp_loaders`. Please don't inline the package logic in them.
- Tests must not require network access by default; mark download-heavy ones with `@pytest.mark.network`.

## Commit style

Short imperative subject lines, e.g. `Fix padding value in make_tokenized_collate`.

## Reporting issues

Open an issue with a minimal repro. If the bug is in a tutorial or example, say which script and Python version you're on.