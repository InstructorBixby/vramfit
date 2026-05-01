#!/usr/bin/env bash
# Cut a version. Run from a clean tree on main.
#
# A script and not a pipeline job, because everything worth checking
# has to happen BEFORE the tag, and the tag is the only thing the
# pipeline sees. Step two is the one I forget.
set -euo pipefail

cd "$(dirname "$0")/.."

test -z "$(git status --porcelain)" || { echo "tree is dirty"; exit 1; }

ver=$(python -c "import vramfit;print(vramfit.__version__)")
proj=$(python -c \
    "import tomllib;print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")
test "$ver" = "$proj" || { echo "__init__ $ver != pyproject $proj"; exit 1; }
grep -q "## \[$ver\]" CHANGELOG.md \
    || { echo "no changelog section for $ver -- rename Unreleased"; exit 1; }

pytest
ruff check .
mypy vramfit
mkdocs build --strict --site-dir /tmp/vramfit-docs

# Build it, install the wheel into a throwaway venv and run it. This
# step exists because I once built a wheel whose package-data left
# data/gpus.csv out, and the failure was invisible until I installed it
# on the GPU box a week later and every card came back with no spec
# row. Nothing in the test suite catches that: the tests import from
# the source tree, where the CSV is always there.
rm -rf dist
python -m build
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
python -m venv "$tmp/venv"
"$tmp/venv/bin/pip" install -q dist/*.whl
"$tmp/venv/bin/vramfit" --version
"$tmp/venv/bin/python" -c \
    "from vramfit import specs; assert len(specs.load()) > 5, 'gpu table missing from the wheel'"

git tag -a "v$ver" -m "vramfit $ver"
echo "tagged v$ver -- push it when you mean it"
