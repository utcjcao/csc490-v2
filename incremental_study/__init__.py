"""Standalone paired intrinsic verification study harness.

This package is intentionally isolated from the existing abcrown and IVAN
pipelines. It uses auto_LiRPA as a backend through a thin adapter, but keeps
all orchestration, perturbation handling, and logging within this folder.
"""

