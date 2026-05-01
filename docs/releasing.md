# releasing

Mostly a note to myself, and a statement of what a tag means.

1. `CHANGELOG.md`: rename `Unreleased` to the version.
2. Bump `vramfit/__init__.py` and `pyproject.toml`.
3. `pytest`, `ruff check .`, `mypy vramfit`, `mkdocs build --strict`.
4. Commit, tag `vX.Y.Z`, push the tag.
5. The release workflow builds a wheel and an sdist, installs the wheel
   and runs `vramfit --version` against it, then publishes through
   PyPI's trusted publisher for this repository.

No API token exists for this project. The workflow mints a short-lived
OIDC credential at publish time, which means there is no secret in the
repository to leak and nothing to rotate when somebody leaves -- and for
a one-maintainer package, no step I can forget.

The install-and-run step in the build job is there because I once
shipped a sdist whose `package-data` left the GPU table out, and the
failure was invisible until somebody installed it from the index.
