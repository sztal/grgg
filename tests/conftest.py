# ruff: noqa: ARG001
import os
import shutil


def pytest_sessionstart(session):
    for root, dirs, files in os.walk("."):
        for d in dirs:
            if d == "__pycache__":
                shutil.rmtree(root / d)
        for f in files:
            if f.endswith(".pyc"):
                (root / f).unlink()
