#!/bin/bash

# This script runs a series of Kavier simulations with traces containing
# different percentage of patched jobs. Results for each execution are
# stored in separate folders.
#
# This script accepts a path to the traces, each trace filename follows
# the following pattern: "kavier-X<percentage>-s<iteration>.csv".
#
# Generate the traces with "generate-experiment-traces.sh" script from
# the main internship repository.
#
# You can set "RESULTS_DIR" env variable to override the default results
# directory (which is /tmp/kavier-experiment-results).
#
# Notation - "s" stands for "seed" (number of iteration), and "X" tells
# the percentage of the patched jobs in the trace.

DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

TRACES_DIR_PATH=$1
if [ -z "$TRACES_DIR_PATH" ]; then
    echo "Argument missing - path to the directory with traces."
    exit 1
fi

if [ -z "$RESULTS_DIR" ]; then
    RESULTS_DIR="/tmp/kavier-experiment-results"
fi

NUM_NODES=8
NODE_GPUS=8
POLICY="consolidated-backfill"
PLACEMENT_POLICY="spread"  # Default for Kubernetes (LeastAllocated).

MAX_ITER=100
PERCENTAGES=(0 10 20 30 40 50 60 70 80 90 100)
ITERATIONS=($(seq 1 "$MAX_ITER"))

rm -rf $RESULTS_DIR
mkdir -p $RESULTS_DIR

source $DIR/.venv/bin/activate

function run_simulator {
    trace_path=$1

    trace_filename=$(basename -- "$trace_path")
    trace_name="${trace_filename%.*}"

    result_prefix="$RESULTS_DIR/$trace_name"

    echo "Running $trace_name"

    uv run kavier cluster --jobs $trace_path --policy $POLICY  --placement $PLACEMENT_POLICY --oversized strict --num-nodes $NUM_NODES --node-gpus $NODE_GPUS --out "$result_prefix"_per_jobs.csv --out-nodes "$result_prefix"_per_nodes.csv --plot "$result_prefix"_timeline.pdf > "$result_prefix"_per_cluster.json

    echo ""
}


for x in "${PERCENTAGES[@]}"; do
    for seed in "${ITERATIONS[@]}"; do
        run_simulator "$TRACES_DIR_PATH"/kavier-X${x}-s${seed}.csv
    done
done
