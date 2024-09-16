#!/usr/bin/env python
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mealpy import ACOR
from demo_mealpy_bv import DGProblem, load_network_fn, draw_networks, ortools_api
from mpi4py import MPI


def local_aco(p, epoch, pop_size, starting_solutions=None):
    model = ACOR.OriginalACOR(epoch=epoch, pop_size=pop_size)
    model.solve(p, mode="swarm", starting_solutions=starting_solutions)
    agents = model.get_sorted_population(model.pop)
    return agents[0].target.fitness, agents[-1].target.fitness, [agent.solution for agent in agents]

def compare_with_ortools(aco_best, times, machines, save_fig):
    solver = ortools_api(times, machines - 1)
    ortools_best = solver.objective_value

    fig, axs = plt.subplots(figsize=(4, 4), tight_layout=True)
    epochs = len(aco_best)
    print(f'Epochs {epochs}')
    for i in range(len(aco_best[0])):
        print(i)
        axs.plot(np.arange(epochs), [aco_best[j][i] for j in range(epochs)], label=f'ACO {i}')
    axs.axhline(y=ortools_best, xmin=0, xmax=len(aco_best), color='r', linestyle='--', label="OR-Tools")
    axs.set_xlabel('Epoch')
    axs.set_ylabel('Makespan')
    axs.legend()
    plt.savefig(save_fig)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    np.random.seed(42 + rank)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--pos', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--u-epoch', type=int, default=10)
    parser.add_argument('--pop-size', type=int, default=10)
    args = parser.parse_args()

    times, machines, (g1, g2) = load_network_fn(args.data, args.pos)
    p = DGProblem(g1, g2)

    best_score, worst_score, solutions = local_aco(p, args.u_epoch, args.pop_size)
    print(f'My rank {rank}: best={best_score} worst={worst_score}')

    aco_best = []
    for i in range(1, args.epoch):
        all_best_scores = comm.allgather(best_score)
        if rank == 0:
            print(f'Received scores: {all_best_scores}')
            aco_best.append(all_best_scores)

        overall_best_score = min(all_best_scores)
        best_rank = all_best_scores.index(overall_best_score)
        best_solution = comm.bcast(solutions[0], best_rank)
        if rank != best_rank and overall_best_score < worst_score:
            solutions = solutions[:-1] + [best_solution]
        best_score, worst_score, solutions = local_aco(p, args.u_epoch, args.pop_size, solutions)

    if rank == 0:
        print(aco_best)
        n, m = times.shape
        compare_with_ortools(aco_best, times, machines, f"ACO_vs_ORTools_{n}_{m}.png")
    
if __name__ == '__main__':
    main()
