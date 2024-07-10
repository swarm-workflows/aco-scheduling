#!/usr/bin/env python
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mealpy import ACOR
from demo_mealpy_bv import DGProblem, load_network_fn, draw_networks, ortools_api
from copy import deepcopy

class ACORLocalSearch(ACOR.OriginalACOR):
    def __init__(self, negate_var, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.negate_var = negate_var
    def evolve(self, epoch):
        super().evolve(epoch)

        pop_new = []
        for i in range(len(self.pop)):
            pop_new.append(self.improve(self.pop[i]))
        pop_new = self.update_target_for_population(pop_new)
        self.pop = self.get_sorted_and_trimmed_population(self.pop + pop_new, self.pop_size, self.problem.minmax)
        print('iter', epoch)

    def improve(self, agent):
        best_makespan = self.problem.obj_func(agent.solution)
        best = agent.solution
        changed = False

        decoded = self.problem.decode_solution(agent.solution)
        for j in range(len(decoded[self.negate_var])):
            candidate = deepcopy(decoded)
            candidate[self.negate_var][j] = 1 - candidate[self.negate_var][j]
            encoded = self.problem.encode_solution([candidate[v.name] for v in self.problem.bounds])
            new_makespan = self.problem.obj_func(encoded)
            if new_makespan < best_makespan:
                best_makespan = new_makespan
                best = encoded
                changed = True
        if changed:
            print('imp')
            return self.generate_empty_agent(best)

        return agent




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--pos', type=int, default=0)
    args = parser.parse_args()

    times, machines, (g1, g2) = load_network_fn(args.data, args.pos)
    n, m = times.shape
    
    p = DGProblem(g1, g2)
    model = ACOR.OriginalACOR(epoch=50, pop_size=10)
    model.solve(p, mode="swarm")
    print("ACO + LS - best solution", model.g_best.target.fitness)

    p = DGProblem(g1, g2)
    ls_model = ACORLocalSearch(epoch=50, pop_size=10, negate_var='dir_var')
    ls_model.solve(p, mode="swarm")
    print("ACO + LS - best solution", ls_model.g_best.target.fitness)

    #res_dag = p.build_dag([int(x) for x in model.g_best.solution])
    #draw_networks(res_dag)

    # solve the problem in ortools
    solver = ortools_api(times, machines - 1)

    aco_best = []
    for i in range(len(model.history.list_global_best)):
        aco_best.append(model.history.list_global_best[i].target.fitness)
    ls_aco_best = [best.target.fitness for best in ls_model.history.list_global_best]

    ortools_best = solver.ObjectiveValue()

    fig, axs = plt.subplots(figsize=(4, 4), tight_layout=True)
    axs.plot(np.arange(len(aco_best)), aco_best, label='ACO')
    axs.plot(np.arange(len(ls_aco_best)), ls_aco_best, label='ACO + LS')
    axs.axhline(y=ortools_best, xmin=0, xmax=len(aco_best), color='r', linestyle='--', label="OR-Tools")
    axs.set_xlabel('Epoch')
    axs.set_ylabel('Makespan')
    axs.legend()
    plt.savefig(f"ACO_vs_ORTools_{n}_{m}.png")
    # plt.show()


if __name__ == '__main__':
    main()
