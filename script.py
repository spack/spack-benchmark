import csv
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

working_directory = pathlib.Path("/tmp/pandas")

reference_csv = working_directory / "propagate.develop.csv"
candidate_csv = working_directory / "propagate.pr.csv"

figure = "radiuss.png"

plt.rcParams.update({'font.size': 48})

base_df = pd.read_csv(
    str(reference_csv),
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

print(base_df.head())
target_df = pd.read_csv(
    str(candidate_csv),
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
print(target_df.head())

combined = pd.concat([base_df, target_df])
df = combined.groupby(["pkg", "source"])[["setup", "load", "ground", "solve", "total"]].describe()
print(df)

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

#plt.tight_layout()
plt.savefig(figure)
#plt.show()
