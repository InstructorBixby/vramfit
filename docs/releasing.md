# releasing

A note to myself, and a statement of what a tag means.

1. `CHANGELOG.md`: rename `Unreleased` to the version.
2. Bump `vramfit/__init__.py` and `pyproject.toml`. **Both.**
3. `scripts/release.sh`, which refuses if 1 or 2 is not done.
4. Push the tag.

A tag means: the suite passed on this tree, the docs built strict, and
a wheel built from it installs into an empty venv and can still find
`data/gpus.csv`. That last one is in there because I once built a wheel
whose `package-data` left the GPU table out, and the failure was
invisible until I installed it on the GPU box a week later and every
card came back with no spec row. The tests import from the source tree,
where the CSV is always present, so nothing in the suite can catch it.

There is nowhere this gets published to. Installing it means
`uv pip install -e .` from a checkout, or pointing pip at the wheel in
`dist/`, which is what I do on a machine I am only borrowing. The
version number exists so a ledger row can say which arithmetic produced
it -- that is the whole job it has.
