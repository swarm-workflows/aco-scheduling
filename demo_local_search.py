#!/usr/bin/env python
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mealpy import ACOR
from demo_mealpy_bv import DGProblem, convert_to_nx, draw_networks, ortools_api
from copy import deepcopy

from benchmark.utils import read_file
from glob import glob
from time import time


class ACORLocalSearch(ACOR.OriginalACOR):
    def __init__(self, negate_var, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.negate_var = negate_var
        self.counter = 0

    def evolve(self, epoch):
        super().evolve(epoch)

        pop_new = []
        route_to_improve = []
        # self.pop.sort(key=lambda agent: self.problem.obj_func(agent.solution))
        route_to_improve.append(self.pop[0])
        """
        for i in range(len(self.pop)):
            print(f'pop[{i}] makespan: {self.problem.obj_func(self.pop[i].solution)}')

        for i in range(len(pop_to_improve)):
            print(f'pop_to_improve[{i}] makespan: {self.problem.obj_func(pop_to_improve[i].solution)}')
        """
        # if self.counter % 5 == 0:
        for i in range(len(route_to_improve)):
            pop_new.append(self.improve(route_to_improve[i]))

        pop_new = self.update_target_for_population(pop_new)
        self.pop = self.get_sorted_and_trimmed_population(self.pop + pop_new, self.pop_size, self.problem.minmax)
        self.counter += 1
        print('iter', epoch)

    def improve(self, agent):
        best_makespan = self.problem.obj_func(agent.solution)
        best = agent.solution
        changed = False

        decoded = self.problem.decode_solution(agent.solution)
        print(f"decoded: {decoded}")

        for j in range(len(decoded[self.negate_var])):
            candidate = deepcopy(decoded)
            candidate[self.negate_var][j] = 1 - candidate[self.negate_var][j]
            print(f'candidate: {candidate}')
            print(f'candidate indexed: {candidate[self.negate_var][j]}')
            encoded = self.problem.encode_solution([candidate[v.name] for v in self.problem.bounds])
            print(f'encoded: {encoded}')
            new_makespan = self.problem.obj_func(encoded)
            if new_makespan < best_makespan:
                best_makespan = new_makespan
                best = encoded
                changed = True
        if changed:
            print('imp')
            return self.generate_empty_agent(best)

        return agent


def run_models(g1, g2, times, machines):
    p = DGProblem(g1, g2)
    model = ACOR.OriginalACOR(epoch=5, pop_size=10)
    model.solve(p, mode="swarm")
    print("ACO - best solution", model.g_best.target.fitness)

    p = DGProblem(g1, g2)
    ls_model = ACORLocalSearch(epoch=5, pop_size=100, negate_var='dir_var')
    ls_model.solve(p, mode="swarm")
    print("ACO + LS - best solution", ls_model.g_best.target.fitness)

    # solve the problem in ortools
    solver = ortools_api(times, machines - 1)

    aco_best = []
    for i in range(len(model.history.list_global_best)):
        aco_best.append(model.history.list_global_best[i].target.fitness)
    ls_aco_best = [best.target.fitness for best in ls_model.history.list_global_best]

    ortools_best = solver.ObjectiveValue()
    n, m = len(times), len(times[0])

    fig, axs = plt.subplots(figsize=(4, 4), tight_layout=True)
    axs.plot(np.arange(len(aco_best)), aco_best, label='ACO')
    axs.plot(np.arange(len(ls_aco_best)), ls_aco_best, label='ACO + LS')
    axs.axhline(y=ortools_best, xmin=0, xmax=len(aco_best), color='r', linestyle='--', label="OR-Tools")
    axs.set_xlabel('Epoch')
    axs.set_ylabel('Makespan')
    axs.legend()
    plt.savefig(f"ACO+LS_vs_ORTools_{n}_{m}.png")
    plt.show()

    res_dag = p.build_dag([int(x) for x in model.g_best.solution])
    draw_networks(res_dag)


def main():

    args = argparse.ArgumentParser()
    args.add_argument("--problem", type=str, default="abz")
    args.add_argument("--id", type=str, default="5")
    args.add_argument("--format", type=str, default="standard", choices=["standard", "taillard"])
    args.add_argument("--all", action="store_true")
    args = args.parse_args()

    if args.all:
        if args.format == "taillard":
            files = glob("./benchmark/*/Taillard_specification/*.txt")
        else:
            files = glob("./benchmark/*/*.txt")

        for fn in sorted(files[:]):
            durations, machines = read_file(fn, problem=args.problem, id=args.id, format=args.format)
            print(f'duration type: {type(durations)}, duration matrix: {durations}')
            n_jobs = len(durations)
            n_machines = len(durations[0])
            (g1, g2) = convert_to_nx(durations, machines, n_jobs, n_machines)
            print(f"Solving {fn.split('/')[-1].split('.')[0]} {n_jobs} {n_machines}", end="\t")
            tic = time()
            run_models(g1, g2, durations, machines)
            toc = time()
            print(f"Time: {toc - tic:.2f}")
    else:
        if args.format == "standard":
            fn = f"./benchmark/{args.problem}/{args.problem}{args.id}.txt"
        else:
            fn = f"./benchmark/{args.problem}/Taillard_specification/{args.problem}{args.id}.txt"
        durations, machines = read_file(fn, problem=args.problem, id=args.id, format=args.format)
        print(f'duration type: {type(durations)}, duration matrix: {durations}')
        durations = np.array(durations)
        machines = np.array(machines)

        print(f'duration type: {type(durations)}, duration matrix: {durations}')
        print(f'machine type: {type(machines)}, machine matrix: {machines}')

        n_jobs = len(durations)
        n_machines = len(durations[0])
        (g1, g2) = convert_to_nx(durations, machines, n_jobs, n_machines)
        print(f"Solving {fn.split('/')[-1].split('.')[0]} {n_jobs} {n_machines}", end="\t")
        tic = time()
        run_models(g1, g2, durations, machines)
        toc = time()
        print(f"Time: {toc - tic:.2f}")


if __name__ == '__main__':
    main()
