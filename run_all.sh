#!/bin/bash
set -e

# Usage:
# run_for_probilem "ft" "06" "taillard"
run_for_problem() {
	ARGS="--problem $1 --id $2 --format $3"
	for module in baselines.mealpy.demo_mealpy_bv benchmark.ortools_benchmark; do
		OUTPUT="results/${module}_$1_$2_$3.json"
		if ! [ -f $OUTPUT ]; then
			echo python -m $module $ARGS --store $OUTPUT
		fi
	done
}

mkdir -p results
#for i in $(seq 41 50); do
#	run_for_problem "ta" $i "taillard"
#done

run_for_problem "abz" "5" "taillard"

run_for_problem "ft" "10" "taillard"

