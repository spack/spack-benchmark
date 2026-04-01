# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
import argparse
import json
import os
import pathlib
import subprocess
import sys

import spack.error

level = "long"
section = "developer"
description = "run micro-benchmarks for Spack internals"

THRESHOLD = 0.10  # flag changes beyond 10%

BENCHMARKS_DIR = pathlib.Path(__file__).parent.parent / "benchmarks"


def _available_suites():
    return sorted(p.stem[len("test_"):] for p in BENCHMARKS_DIR.glob("test_*.py"))


def setup_parser(subparser: argparse.ArgumentParser):
    sp = subparser.add_subparsers(metavar="SUBCOMMAND", dest="subcommand")

    sp.add_parser("list", help=list_suites.__doc__)

    run_parser = sp.add_parser("run", help=run.__doc__)
    run_parser.add_argument(
        "-o",
        "--output",
        help="JSON output file (default: results.json)",
        default="results.json",
    )
    run_parser.add_argument(
        "--suite",
        help="comma-separated list of benchmark suites to run (default: all); "
        "each name maps to a test_<name>.py file in the benchmarks directory",
        metavar="SUITE,...",
    )
    run_parser.add_argument(
        "--filter",
        help="only run benchmarks matching this substring",
        metavar="PATTERN",
    )

    compare_parser = sp.add_parser("compare", help=compare.__doc__)
    compare_parser.add_argument(
        "files", nargs="+", metavar="JSON_FILE", help="two or more benchmark JSON files to compare"
    )
    compare_parser.add_argument(
        "-o",
        "--output",
        help="output PNG file (default: comparison.png)",
        default="comparison.png",
    )
    compare_parser.add_argument(
        "--labels",
        help="comma-separated labels for each file (default: file stem)",
        metavar="LABEL,...",
    )
    compare_parser.add_argument(
        "--log-scale",
        help="use log scale on the y axis",
        action="store_true",
    )
    compare_parser.add_argument(
        "--markdown",
        help="print a markdown table instead of generating a plot",
        action="store_true",
    )


def list_suites(args):
    """list available benchmark suites"""
    for suite in _available_suites():
        print(suite)


def run(args):
    """run micro-benchmarks and save results to a JSON file"""
    import spack

    if args.suite:
        suites = [s.strip() for s in args.suite.split(",")]
        available = _available_suites()
        invalid = [s for s in suites if s not in available]
        if invalid:
            raise spack.error.SpackError(
                f"Unknown suite(s): {', '.join(invalid)}. "
                f"Available: {', '.join(available)}."
            )
        paths = [str(BENCHMARKS_DIR / f"test_{s}.py") for s in suites]
    else:
        paths = [str(BENCHMARKS_DIR)]
    # Spack's lib directory must be on PYTHONPATH so the subprocess can import spack
    spack_lib = str(pathlib.Path(spack.__file__).parent.parent)
    env = os.environ.copy()
    env["PYTHONPATH"] = spack_lib + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *paths,
        "--benchmark-only",
        f"--benchmark-json={args.output}",
    ]
    if args.filter:
        cmd += ["-k", args.filter]
    result = subprocess.run(cmd, env=env)
    # pytest exit codes: 0=passed, 1=failed, 2=interrupted, 3=internal error,
    # 4=usage error, 5=no tests collected
    if result.returncode == 5:
        filter_msg = f" matching '{args.filter}'" if args.filter else ""
        raise spack.error.SpackError(f"No benchmarks found{filter_msg}.")
    elif result.returncode != 0:
        raise spack.error.SpackError(f"Benchmarks failed (pytest exit code {result.returncode}).")


def _load(path):
    with open(path) as f:
        data = json.load(f)
    return {b["name"]: b["stats"] for b in data["benchmarks"]}


def _fmt_us(seconds):
    return f"{seconds * 1e6:.1f} µs"


def _fmt_pct(ratio):
    sign = "+" if ratio >= 0 else ""
    flag = " ⚠️" if ratio > THRESHOLD else (" ✅" if ratio < -THRESHOLD else "")
    return f"{sign}{ratio * 100:.1f}%{flag}"


def _print_markdown(runs):
    # runs: list of (label, {name: stats})
    all_names = sorted(set(name for _, data in runs for name in data))
    baseline_label, baseline_data = runs[0]
    header = "| Test | " + " | ".join(f"{label} (median)" for label, _ in runs)
    if len(runs) > 1:
        header += " | " + " | ".join(f"vs {baseline_label}" for _, _ in runs[1:])
    header += " |"
    sep = "|------|" + "-----------------|" * len(runs)
    if len(runs) > 1:
        sep += "--------|" * (len(runs) - 1)
    print(header)
    print(sep)
    for name in all_names:
        row = f"| {name} |"
        for _, data in runs:
            row += f" {_fmt_us(data[name]['median']) if name in data else '—'} |"
        if len(runs) > 1 and name in baseline_data:
            b = baseline_data[name]["median"]
            for _, data in runs[1:]:
                row += f" {_fmt_pct((data[name]['median'] - b) / b) if name in data else '—'} |"
        print(row)


def _shorten(name, max_len=30):
    if len(name) <= max_len:
        return name
    half = (max_len - 3) // 2
    return name[:half] + "..." + name[len(name) - (max_len - 3 - half):]


def _plot(runs, output, log_scale=False):
    import matplotlib.pyplot as plt
    import numpy as np

    # runs: list of (label, {name: stats})
    all_names = sorted(set(name for _, data in runs for name in data))
    labels = [_shorten(n.split("::")[-1]) for n in all_names]

    n_runs = len(runs)
    width = 0.8 / n_runs
    x = np.arange(len(all_names))
    offsets = np.linspace(-(n_runs - 1) / 2, (n_runs - 1) / 2, n_runs) * width

    fig, ax = plt.subplots(figsize=(max(12, len(all_names) * 0.6), 6), layout="constrained")
    for (label, data), offset in zip(runs, offsets):
        medians = [data[n]["median"] * 1e6 if n in data else 0 for n in all_names]
        ax.bar(x + offset, medians, width, label=label, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    if log_scale:
        ax.set_yscale("log")
        ax.grid(True, axis="y", which="both")
        ax.grid(True, axis="y", which="minor", linestyle=":", linewidth=0.5)
    else:
        ax.grid(True, axis="y")
    ax.set_ylabel("Median time (µs)")
    ax.set_title("Micro-benchmark comparison")
    ax.legend()

    plt.savefig(output)
    print(f"Plot saved to {output}")


def compare(args):
    """compare two or more benchmark JSON files produced by 'run'"""
    if len(args.files) < 2:
        raise spack.error.SpackError("At least two JSON files are required for comparison.")
    labels = args.labels.split(",") if args.labels else [pathlib.Path(f).stem for f in args.files]
    if len(labels) != len(args.files):
        raise spack.error.SpackError(
            f"--labels has {len(labels)} entries but {len(args.files)} files were given."
        )
    runs = [(label, _load(f)) for label, f in zip(labels, args.files)]

    if args.markdown:
        _print_markdown(runs)
    else:
        _plot(runs, args.output, log_scale=args.log_scale)


def microbenchmark(parser, args):
    action = {"list": list_suites, "run": run, "compare": compare}
    return action[args.subcommand](args)
