# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Worker function for the ``solve-benchmark`` command.

This lives in the importable ``spack_solver_benchmark`` package (rather than in the
Spack extension command module) so it can be re-imported by name in worker
processes. ``multiprocessing`` pickles the pool target by reference, and under the
``forkserver``/``spawn`` start methods each worker must import the target's module
from scratch. The command module is loaded by Spack under a synthetic
``spack.extensions.*`` name that no fresh interpreter can import, so the target
cannot live there.
"""
import sys
from typing import List, Tuple

import spack.solver.asp as asp
import spack.spec
import spack.util.timer

#: (spec, hash, iteration, setup, load, ground, solve, total, deps)
Record = Tuple[str, str, int, float, float, float, float, float, int]


def _clear_repo_modules() -> None:
    """Clear all spack_repo.* modules from sys.modules to force reimport."""
    to_delete = [name for name in sys.modules if name.startswith("spack_repo.")]
    for name in to_delete:
        del sys.modules[name]


def run_single_solve(inputs: Tuple[List[spack.spec.Spec], int, bool]) -> Record:
    specs, i, clear_repo_modules = inputs
    if clear_repo_modules:
        _clear_repo_modules()
    solver = asp.Solver()
    result, timer, _ = solver.driver.solve(
        asp.SpackSolverSetup(),
        specs,
        reuse=solver.selector.reusable_specs(specs),
    )
    assert isinstance(timer, spack.util.timer.Timer)
    timer.stop()
    spec_hash = result.specs[0].dag_hash() if result.specs else ""
    return (
        str(specs[0]),
        spec_hash,
        i,
        timer.duration("setup"),
        timer.duration("load"),
        timer.duration("ground"),
        timer.duration("solve"),
        timer.duration(),
        len(result.possible_dependencies),
    )
