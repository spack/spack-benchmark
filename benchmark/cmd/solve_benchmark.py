# Copyright 2013-2022 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
import csv
import glob
import os
import re
import pathlib
import sys
import time
import warnings
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import spack.cmd
import spack.solver.asp as asp
import spack.util.parallel

from llnl.util import tty

SOLUTION_PHASES = "setup", "load", "ground", "solve"
VALID_CONFIGURATIONS = "tweety", "handy", "trendy", "many"


level = "long"
section = "developer"
description = "benchmark concretization speed"

def setup_parser(subparser):
    sp = subparser.add_subparsers(metavar="SUBCOMMAND", dest="subcommand")

    run = sp.add_parser("run", help="run benchmarks and produce a CSV file of timing results")

    run.add_argument(
        "-r",
        "--repetitions",
        type=int,
        help="number of repetitions for each spec",
        default=1,
    )
    run.add_argument("-o", "--output", help="CSV output file", required=True)
    run.add_argument(
        "--reuse",
        help="maximum reuse of buildcaches and installations",
        action="store_true",
    )
    run.add_argument("--configs", help="comma separated clingo configurations", default="tweety")
    run.add_argument(
        "-n",
        "--nprocess",
        help="number of processes to use to produce the results",
        default=os.cpu_count(),
        type=int,
    )
    run.add_argument(
        "-s",
        "--shuffle",
        help="shuffle the list of concretizations to be done",
        action="store_true"
    )
    run.add_argument("specfile", help="text file with one spec per line, can be one of the predefined benchmarks")

    plot = sp.add_parser("plot", help="plot results recorded in a CSV file")
    plot.add_argument(
        "--reference-csv",
        type=str,
        required=True,
        help="Path to the reference CSV file.",
    )
    plot.add_argument(
        "--candidate-csv",
        type=str,
        required=True,
        help="Path to the candidate CSV file.",
    )
    plot.add_argument("-o", "--output", help="output image file", required=True)


def process_single_item(inputs):
    args, specs, idx, cf, i = inputs
    control = asp.default_clingo_control()
    # control.configuration.configuration = cf
    solver = spack.solver.asp.Solver()
    setup = spack.solver.asp.SpackSolverSetup()
    reusable_specs = solver.selector.reusable_specs(specs)
    try:
        sol_res, timer, solve_stat = solver.driver.solve(
            setup, specs, reuse=reusable_specs, control=control
        )
        possible_deps = sol_res.possible_dependencies
        timer.stop()
        time_by_phase = tuple(timer.duration(ph) for ph in SOLUTION_PHASES)
    except Exception as e:
        warnings.warn(str(e))
        return None

    total = sum(time_by_phase)
    return (str(specs[0]), cf, i) + time_by_phase + (total, len(possible_deps))


def run(args):
    configs = args.configs.split(",")
    if any(x not in VALID_CONFIGURATIONS for x in configs):
        print(
            "Invalid configuration. Valid options are {0}".format(", ".join(VALID_CONFIGURATIONS))
        )

    # Warmup spack to ensure caches have been written, and clingo is ready
    # (we don't want to measure bootstrapping time)
    specs = spack.cmd.parse_specs("hdf5")
    solver = spack.solver.asp.Solver()
    setup = spack.solver.asp.SpackSolverSetup()
    result, _, _ = solver.driver.solve(setup, specs, reuse=[])
    reusable_specs = solver.selector.reusable_specs(specs)

    # Read the list of specs to be analyzed
    current_file = pathlib.Path(args.specfile)
    if not current_file.exists():
        current_dir = pathlib.Path(__file__ ).parent
        current_file = current_dir / "data" / f"{args.specfile}.txt"
        # TODO: error handling

    lines = current_file.read_text().split("\n")
    pkg_ls = [l.strip() for l in lines if l.strip()]

    # Perform the concretization tests
    input_list = []
    for idx, pkg in enumerate(pkg_ls):
        specs = spack.cmd.parse_specs(pkg)
        for cf in configs:
            for i in range(args.repetitions):
                item = (args, specs, idx, cf, i)
                input_list.append(item)

    if args.shuffle:
        random.shuffle(input_list)

    start = time.time()
    pkg_stats = []

    if args.nprocess > 1:
        for idx, record in enumerate(
            spack.util.parallel.imap_unordered(
                process_single_item,
                input_list,
                processes=args.nprocess,
                debug=tty.is_debug(),
                maxtaskperchild=1,
            )
        ):
            duration = record[-2]
            pkg_stats.append(record)
            percentage = (idx + 1) / len(input_list) * 100
            tty.msg(f"{duration:6.1f}s [{percentage:3.0f}%] {record[0]}")
            sys.stdout.flush()
    else:
        for idx, input in enumerate(input_list):
            record = process_single_item(input)
            duration = record[-2]
            pkg_stats.append(record)
            percentage = (idx + 1) / len(input_list) * 100
            tty.msg(f"{duration:6.1f}s [{percentage:3.0f}%] {record[0]}")
    
    finish = time.time()
    tty.msg(f"Total elapsed time: {finish - start:.2f} seconds")
    
    # Write results to CSV file
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(pkg_stats)


def plot(args):
    plt.rcParams.update({'font.size': 48})

    base_df = pd.read_csv(
        str(args.reference_csv),
        header=None,
        names=[
            "pkg",
            "cfg",
            "iter",
            "setup",
            "load",
            "ground",
            "solve",
            "total",
            "dep_len",
        ],
    )
    base_df["source"] = 0

    target_df = pd.read_csv(
        str(args.candidate_csv),
        header=None,
        names=[
            "pkg",
            "cfg",
            "iter",
            "setup",
            "load",
            "ground",
            "solve",
            "total",
            "dep_len",
        ],
    )
    target_df["source"] = 1

    combined = pd.concat([base_df, target_df])
    df = combined.groupby(["pkg", "source"])[["setup", "load", "ground", "solve", "total"]].describe()

    cols = ["setup", "load", "ground", "solve", "total"]
    titles = [
        "Setup", "Load", "Ground", "Solve", "Total"
    ]
    fig, axes = plt.subplots(1, 5, sharex=True, sharey=True, figsize=(160, 32), layout="constrained")
    for ax, level, title in zip(axes, cols, titles):
        current = df.unstack(level="source").loc[:, (level, "mean", 0):(level, "mean", 1)]

        negvals  = current.to_numpy().T - df.unstack(level="source").loc[:, (level, "min", 0):(level, "min", 1)].to_numpy().T
        posvals  = df.unstack(level="source").loc[:, (level, "max", 0):(level, "max", 1)].to_numpy().T - current.to_numpy().T

        yerr = np.stack((negvals, posvals), axis=1)
        current.plot(
            ax=ax,
            kind="bar",
            width=.9,
            title=title,
            grid=True,
            yerr=yerr,
            capsize=20,
            error_kw={"capthick":4, "elinewidth": 2},
            alpha=0.7
        )
        ax.set(xlabel=None, ylabel="Time [sec.]", ylim=[0, 51])
        ax.legend(["develop", "PR"])
        plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

    plt.savefig(args.output)


def solve_benchmark(parser, args):
    action = {"run": run, "plot": plot}
    return action[args.subcommand](args)
