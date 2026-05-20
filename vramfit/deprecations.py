"""Keeping an old name working while telling you it is old.

The policy is in docs/stability.md: an old flag keeps working, warns
with the replacement in the message, gets a changelog line naming the
release that will remove it, and is only removed in a major.

This exists because `--utilisation` was the wrong name for a flag that
sets vLLM's `gpu_memory_utilization`. I spelled it the way I spell it,
which meant every command copied out of vLLM's own documentation needed
one word translated, and twice I did not notice I had not translated
it. Renaming it was right. Deleting the old spelling was not: it is in
the wrapper scripts on the GPU box, so the only person a hard break
would have cost an evening is me, on a machine I was in the middle of
using.
"""
from __future__ import annotations

import argparse
import warnings


class DeprecatedAlias(argparse.Action):
    """An argparse action that sets the new name and complains.

    `dest` is the new option's destination, so a parser can accept both
    spellings and the rest of the code only ever sees one.
    """

    def __init__(self, option_strings: list[str], dest: str,
                 replacement: str = "", removed_in: str = "2.0",
                 **kw: object) -> None:
        self.replacement = replacement
        self.removed_in = removed_in
        super().__init__(option_strings, dest, **kw)  # type: ignore[arg-type]

    def __call__(self, parser: argparse.ArgumentParser,
                 namespace: argparse.Namespace, values: object,
                 option_string: str | None = None) -> None:
        warn(option_string or self.option_strings[0],
             self.replacement, self.removed_in)
        setattr(namespace, self.dest, values)


def warn(old: str, new: str, removed_in: str = "2.0") -> None:
    warnings.warn(
        f"{old} is deprecated and will be removed in {removed_in}; "
        f"use {new}",
        DeprecationWarning, stacklevel=3)
