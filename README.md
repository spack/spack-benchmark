## Installation

```console
$ unset SPACK_PYTHON  # ensure Spack uses the Python from the virtual environment
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install "git+https://github.com/spack/spack-benchmark#egg=spack_solver_benchmark"
```

## Usage

To benchmark concretization speed, and record results in a CSV file:

```console
$ spack list -t radiuss > radiuss.txt
$ spack solve-benchmark run --fresh --repetitions 5 --nprocess 4 -o radiuss.csv radiuss.txt
```

The first command simply creates a text file where each line is an input to the concretization algorithm.
The second command goes over all the inputs and records concretization time for each of them.

## Comparing two benchmarks

The following command compares two benchmark results in CSV format using a statistical test and generates a plot.

```console
$ spack solve-benchmark compare before.csv after.csv -o plot.png
```
