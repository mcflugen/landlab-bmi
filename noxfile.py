from __future__ import annotations

import os

import nox

ROOT = os.path.dirname(os.path.abspath(__file__))


@nox.session
def build(session: nox.Session) -> None:
    """Build sdist and wheel dists."""
    session.install("pip", "build")
    session.install("setuptools")
    session.run("python", "--version")
    session.run("pip", "--version")
    session.run("python", "-m", "build")


@nox.session
def install(session: nox.Session) -> None:
    first_arg = session.posargs[0] if session.posargs else None

    if first_arg:
        if os.path.isfile(first_arg):
            session.install(first_arg)
        else:
            session.error("path must be a source distribution")
    else:
        session.install(".")


@nox.session
def test(session: nox.Session) -> None:
    """Run the tests."""
    session.install("pytest", "bmi_wavewatch3", "dask")
    install(session)

    session.run("pytest", "-vvv")


@nox.session
def coverage(session: nox.Session) -> None:
    session.install("coverage", "pytest", "bmi_wavewatch3", "dask")
    session.install("-e", ".")

    session.run("coverage", "erase")
    session.run(
        "coverage",
        "run",
        "--source=landlab_bmi",
        "--module",
        "pytest",
        "-vvv",
        env={"COVERAGE_CORE": "sysmon"},
    )

    if "CI" in os.environ:
        session.run("coverage", "xml", "-o", os.path.join(ROOT, "coverage.xml"))
    else:
        session.run("coverage", "report", "--ignore-errors", "--show-missing")


@nox.session
def lint(session: nox.Session) -> None:
    """Look for lint."""
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files")
