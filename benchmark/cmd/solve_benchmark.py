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
    plot_type = plot.add_mutually_exclusive_group()
    plot_type.add_argument(
        "--cdf",
        action="store_true",
        help="CDF plot (number of packages vs. execution time)",
    )
    plot_type.add_argument(
        "--scatter",
        action="store_true",
        help="scatter plot (execution time vs. possible dependencies)",
    )
    plot_type.add_argument(
        "--histogram",
        action="store_true",
        help="histogram plot (execution time vs. number of packages)",
    )
    plot.add_argument("csvfile", help="CSV file with timing data")
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
    if args.cdf:
        _plot_cdf(args)
    elif args.scatter:
        _plot_scatter(args)
    elif args.histogram:
        _plot_histogram(args)


def _plot_cdf(args):
    df = pd.read_csv(
        args.csvfile,
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
    print(df.head())

    cfg_ls = list(sorted(set(df["cfg"])))
    pkg_ls = list(sorted(set(df["pkg"])))

    fig, axs = plt.subplots(figsize=(6, 6), dpi=150)

    for cfg in cfg_ls:
        df_by_config = df[df["cfg"] == cfg]
        times = df_by_config["total"]
        times.hist(cumulative=True, density=1, bins=100, ax=axs, label=cfg, histtype="step")

    axs.set_xlabel("Total Time [sec.]", fontsize=20)
    axs.set_ylabel("Percentage of package", fontsize=20)
    axs.legend(loc="upper left")
    fig.savefig(args.output)


def _plot_scatter(args):
    df = pd.read_csv(
        args.csvfile,
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
    df_full = df
    print(df_full.head())

    cfg_ls = list(sorted(set(df["cfg"])))
    pkg_ls = list(sorted(set(df["pkg"])))

    timings = {}
    deps = []

    df_deps = df_full

    fig, axs = plt.subplots(figsize=(6, 6), dpi=150)

    df.plot.scatter(x="dep_len", y="total", ax=axs)
    axs.set_xlabel("Number of possible dependencies", fontsize=20)
    axs.set_ylabel("Total time [s]", fontsize=20)
    fig.savefig(args.output)


def _plot_histogram(args):
    # Data analysis
    df = pd.read_csv(
        args.csvfile,
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
            "ndeps",
        ],
    )
    print(df.head())

    cfg_ls = list(sorted(set(df["cfg"])))
    pkg_ls = list(sorted(set(df["pkg"])))
    timings = {}
    for cf in cfg_ls:
        timings[cf] = {}
        for ph in SOLUTION_PHASES:
            timings[cf][ph] = []
        for pk in pkg_ls:
            tmp_df = df[df["pkg"] == pk]
            tdf = tmp_df[tmp_df["cfg"] == cf]
            for ph in SOLUTION_PHASES:
                timings[cf][ph].append(tdf[ph].median())

    for cf in cfg_ls:
        fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True, figsize=(20, 20), dpi=100)
        axes = list(axs.flatten())
        n_bins = 150

        fig.suptitle(cf, fontsize=24)
        for i, ph in enumerate(SOLUTION_PHASES):
            solve_times = sorted(zip(pkg_ls, timings[cf][ph]), key=lambda x: x[1], reverse=True)
            tab_data = [[p, "{:.3f}".format(t)] for p, t in solve_times[0:5]]

            axes[i].hist(sorted(timings[cf][ph], reverse=True), n_bins, label=ph)
            axes[i].set_title(ph, fontsize=18)
            tab = axes[i].table(cellText=tab_data, bbox=[0.1, -0.5, 0.75, 0.4])
            tab.auto_set_font_size(False)
            tab.auto_set_column_width(col=[0, 1])
            tab.set_fontsize(12)
    plt.savefig(args.output, dpi=150)


def solve_benchmark(parser, args):
    action = {"run": run, "plot": plot}
    return action[args.subcommand](args)
