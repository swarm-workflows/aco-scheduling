
import argparse
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from mealpy import ACOR
from demo_mealpy_mpv import DGProblem, load_network_fn, draw_networks, ortools_api
from demo_local_search import ACORLocalSearch
from mpi4py import MPI


@dataclass
class ParallelSolver:
    u_epoch: int
    pop_size: int
    epoch: int
    comm: MPI.Comm

    def __post_init__(self):
        self.rank = self.comm.Get_rank()

    def get_model(self):
        return ACOR.OriginalACOR(epoch=self.u_epoch, pop_size=self.pop_size)

    def local_aco(self, p, starting_solutions=None):
        model = self.get_model()
        model.solve(p, mode="swarm", starting_solutions=starting_solutions)
        agents = model.get_sorted_population(model.pop)
        return agents[0].target.fitness, agents[-1].target.fitness, [agent.solution for agent in agents]

    def solve(self, p):
        best_score, worst_score, solutions = self.local_aco(p)
        print(f'My rank {self.rank}: best={best_score} worst={worst_score}')

        aco_best = []
        for i in range(1, self.epoch):
            all_best_scores = self.comm.allgather(best_score)
            if self.rank == 0:
                print(f'Received scores: {all_best_scores}')
            aco_best.append(all_best_scores)

            overall_best_score = min(all_best_scores)
            best_rank = all_best_scores.index(overall_best_score)
            best_solution = self.comm.bcast(solutions[0], best_rank)
            if self.rank != best_rank and overall_best_score < worst_score:
                solutions = solutions[:-1] + [best_solution]
            best_score, worst_score, solutions = self.local_aco(p, solutions)
        return aco_best


class ParallelSolverLS(ParallelSolver):
    def get_model(self):
        return ACORLocalSearch(epoch=self.u_epoch, pop_size=self.pop_size, negate_var='dir_var')


def compare_with_ortools(aco_best, times, machines, save_fig):
    solver = ortools_api(times, machines - 1)
    ortools_best = solver.ObjectiveValue()

    fig, axs = plt.subplots(figsize=(4, 4), tight_layout=True)
    for k, v in aco_best.items():
        axs.plot(np.arange(len(v)), v, label=k)
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
    ps = ParallelSolver(args.u_epoch, args.pop_size, args.epoch, MPI.COMM_WORLD)
    aco_best = ps.solve(p)

    p = DGProblem(g1, g2)
    ps = ParallelSolverLS(args.u_epoch, args.pop_size, args.epoch, MPI.COMM_WORLD)
    ls_aco_best = ps.solve(p)

    if rank == 0:
        print(aco_best)
        n, m = times.shape
        compare_with_ortools({'ACO': [e[0] for e in aco_best],
                              'ACO + LS': [e[0] for e in ls_aco_best]}, times, machines, f"ACO_vs_ORTools_{n}_{m}.png")


if __name__ == '__main__':
    main()
