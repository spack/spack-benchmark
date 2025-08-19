# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
import argparse
import os
import pathlib
import random
import sys
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spack.cmd
import spack.solver.asp as asp
import spack.spec
import spack.util.parallel
import spack.util.timer
from scipy.stats import wilcoxon
from spack.cmd.common.arguments import add_concretizer_args
from spack.llnl.util import tty

SOLUTION_PHASES = "setup", "load", "ground", "solve"
TIMING_COLS = [*SOLUTION_PHASES, "total"]
COLUMNS = ["spec", "iteration", *TIMING_COLS, "deps"]


level = "long"
section = "developer"
description = "benchmark concretization speed"


def setup_parser(subparser: argparse.ArgumentParser):
    sp = subparser.add_subparsers(metavar="SUBCOMMAND", dest="subcommand")

    run_parser = sp.add_parser("run", help=run.__doc__)
    run_parser.add_argument(
        "-r",
        "--repetitions",
        type=int,
        help="number of repetitions for each spec",
        default=1,
    )
    run_parser.add_argument("-o", "--output", help="CSV output file", required=True)
    run_parser.add_argument(
        "-n",
        "--nprocess",
        help="number of processes to use to produce the results",
        default=os.cpu_count(),
        type=int,
    )
    run_parser.add_argument(
        "-s",
        "--shuffle",
        help="shuffle the list of concretizations to be done",
        action="store_true",
    )
    add_concretizer_args(run_parser)
    run_parser.add_argument(
        "specfile",
        help="text file with one spec per line, can be one of the predefined benchmarks",
    )

    compare_parser = sp.add_parser("compare", help=compare.__doc__)
    compare_parser.add_argument(
        "before",
        help="first CSV file to compare (e.g., develop.csv)",
    )
    compare_parser.add_argument(
        "after",
        help="second CSV file to compare (e.g., pr.csv)",
    )
    compare_parser.add_argument(
        "-o",
        "--output",
        help="output plot file (default: comparison.png)",
        default="comparison.png",
    )


Record = Tuple[str, int, float, float, float, float, float, int]


def _run_single_solve(
    inputs: Tuple[List[spack.spec.Spec], int],
) -> Record:
    specs, i = inputs
    solver = asp.Solver()
    result, timer, _ = solver.driver.solve(
        asp.SpackSolverSetup(),
        specs,
        reuse=solver.selector.reusable_specs(specs),
    )
    assert isinstance(timer, spack.util.timer.Timer)
    timer.stop()
    return (
        str(specs[0]),
        i,
        timer.duration("setup"),
        timer.duration("load"),
        timer.duration("ground"),
        timer.duration("solve"),
        timer.duration(),
        len(result.possible_dependencies),
    )


def _warmup():
    specs = spack.cmd.parse_specs("hdf5")
    solver = asp.Solver()
    solver.driver.solve(asp.SpackSolverSetup(), specs, reuse=solver.selector.reusable_specs(specs))


def _validate_and_load_csv_files(
    before_file: str, after_file: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate two CSV files for comparison/plotting."""
    # Load the data using the header row
    try:
        before_df = pd.read_csv(before_file)
        after_df = pd.read_csv(after_file)
    except FileNotFoundError as e:
        raise RuntimeError(f"Could not read CSV file: {e}") from e

    # Verify the expected columns exist
    for df, name in [(before_df, before_file), (after_df, after_file)]:
        if list(df.columns) != COLUMNS:
            raise RuntimeError(f"Unexpected CSV format in {name}. Expected columns: {COLUMNS}")

    # Check that both files have the same specs (validate once upfront)
    before_specs = set(before_df["spec"].unique())
    after_specs = set(after_df["spec"].unique())

    if before_specs != after_specs:
        raise RuntimeError(
            f"Specs in {before_file} and {after_file} do not match: "
            f"{before_specs.symmetric_difference(after_specs)}"
        )

    return before_df, after_df


def run(args):
    """run benchmarks and produce a CSV file of timing results"""
    input_file = pathlib.Path(args.specfile)
    if not input_file.exists():
        current_dir = pathlib.Path(__file__).parent
        input_file = current_dir / "data" / f"{args.specfile}.txt"

    try:
        spec_strs = [line.strip() for line in input_file.read_text().split("\n") if line.strip()]
    except OSError as e:
        raise RuntimeError(f"Could not read the input spec file: {e}") from e

    tty.info("Warm up...")
    _warmup()

    input_list = [
        (spack.cmd.parse_specs(spec_str), i)
        for spec_str in spec_strs
        for i in range(args.repetitions)
    ]

    if args.shuffle:
        random.shuffle(input_list)

    start = time.time()
    pkg_stats: List[Record] = []

    if args.nprocess > 1:
        record_iterator = spack.util.parallel.imap_unordered(
            _run_single_solve,
            input_list,
            processes=args.nprocess,
            debug=tty.is_debug(),
            maxtaskperchild=1,
        )
    else:
        record_iterator = map(_run_single_solve, input_list)

    # Process records with unified progress reporting
    tty.info("Benchmarking...")

    for idx, record in enumerate(record_iterator):
        pkg_stats.append(record)
        tty.msg(f"{record[6]:6.1f}s [{(idx + 1)/len(input_list)*100:3.0f}%] {record[0]}")
        sys.stdout.flush()

    finish = time.time()
    tty.msg(f"Total elapsed time: {finish - start:.2f} seconds")

    # Create DataFrame and write to CSV
    pd.DataFrame(pkg_stats, columns=COLUMNS).to_csv(args.output, index=False)


def compare(args):
    """Compare two CSV files to see whether one is faster than the other and generate a plot."""
    before_df, after_df = _validate_and_load_csv_files(args.before, args.after)
    significant_fields = []
    before = before_df.groupby("spec")[TIMING_COLS].median()
    after = after_df.groupby("spec")[TIMING_COLS].median()

    print("## Performance comparison\n")

    for field in TIMING_COLS:
        # Calculate change in median time
        comparison = pd.DataFrame({"median_before": before[field], "median_after": after[field]})
        comparison["ratio"] = comparison["median_after"] / comparison["median_before"]
        comparison["change_percent"] = (comparison["ratio"] - 1) * 100

        # Statistical Testing using Wilcoxon signed-rank test
        # Null hypothesis: median of log of ratios is 0 (i.e., median of ratios is 1, no change)
        # alternative="less" tests if things got faster (ratios < 1, log of ratios < 0)
        log_ratios = np.log(comparison["ratio"].to_numpy())
        test_result = wilcoxon(log_ratios, alternative="less")
        alpha = 0.05
        p_value: float = test_result.pvalue
        is_significant = p_value < alpha

        print(
            f"**{field}**: {'significant' if is_significant else 'not significant'} "
            f"({p_value:.4f} {'<' if is_significant else '>='} {alpha})\n"
        )
        print(comparison.round(2).to_markdown())
        print()

        if is_significant:
            significant_fields.append(
                {
                    "field": field,
                    "p_value": p_value,
                }
            )

    print(
        "## Summary\n"
        f"Statistically significant improvements ({len(significant_fields)} fields):"
    )
    for result in significant_fields:
        print(f"* {result['field']}: p = {result['p_value']:.4f}")

    # Generate plot
    print(f"\n<!-- generating plot: {args.output} -->")

    # Add source column and combine dataframes
    before_df["source"] = 0
    after_df["source"] = 1
    combined = pd.concat([before_df, after_df])

    # Group by spec and source, calculate statistics
    df = combined.groupby(["spec", "source"])[TIMING_COLS].describe()

    # Set up matplotlib configuration
    plt.rcParams.update({"font.size": 48})

    # Create subplots
    _, axes = plt.subplots(1, 5, sharex=True, sharey=True, figsize=(160, 32), layout="constrained")

    for ax, col in zip(axes, TIMING_COLS):
        col_stats = df.xs(col, level=0, axis=1)
        medians = col_stats["50%"].unstack(level="source")
        mins = col_stats["min"].unstack(level="source")
        maxs = col_stats["max"].unstack(level="source")
        error_bars = np.stack(((medians - mins).T, (maxs - medians).T), axis=1)
        medians.plot(
            ax=ax,
            kind="bar",
            width=0.9,
            title=col.capitalize(),
            grid=True,
            yerr=error_bars,
            capsize=20,
            error_kw={"capthick": 4, "elinewidth": 2},
            alpha=0.7,
        )

        ax.set(xlabel=None, ylabel="Time [sec.]")
        ax.legend(["before", "after"])
        plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment="right")

    plt.savefig(args.output)


def solve_benchmark(parser, args):
    action = {"run": run, "compare": compare}
    return action[args.subcommand](args)
