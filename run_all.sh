#!/bin/bash
set -e

ALL_MODULES="
baselines.mealpy.demo_mealpy_bv
baselines.mealpy.demo_mealpy_mpv
baselines.LRU.main
benchmark.ortools_benchmark
run_jsp
"
# Usage:
# run_for_probilem "ft" "06" "taillard"
run_for_problem() {
  ARGS="--problem $1 --id $2 --format $3"
  for module in $ALL_MODULES; do
    OUTPUT="results/${module}_$1_$2_$3.json"
    if ! [ -f $OUTPUT ]; then
      python -m $module $ARGS --store $OUTPUT
    fi
  done

  # Special case: ACO + LS
  OUTPUT="results/jsp_ls_$1_$2_$3.json"
  if ! [ -f $OUTPUT ]; then
    python -m run_jsp --enable-ls $ARGS --store $OUTPUT
  fi
}

# Usage:
# run_jss_for_problem "ft" "06"
run_jss_for_problem() {
  OUTPUT="$PWD/results/jss_$1_$2.json"
  if ! [ -f "$OUTPUT" ]; then
    pushd baselines/RL-Job-Shop-Scheduling
	source setup.sh
	pushd JSS/JSS
    WANDB_MODE=offline python main.py --store $OUTPUT --instance-path benchmark/$1/$1$2.txt
    popd
  fi
}

#mkdir -p results
#for i in $(seq 41 50); do
#	run_for_problem "ta" $i "taillard"
#done

run_for_problem "abz" "5" "taillard"

run_for_problem "ft" "10" "taillard"

run_jss_for_problem "abz" "5"

