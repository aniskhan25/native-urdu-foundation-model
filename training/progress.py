"""Training progress helpers."""

from __future__ import annotations


def resume_progress(loaded_step: int, loaded_tokens: int, *, reset: bool) -> tuple[int, int]:
    if reset:
        return 0, 0
    return loaded_step, loaded_tokens
