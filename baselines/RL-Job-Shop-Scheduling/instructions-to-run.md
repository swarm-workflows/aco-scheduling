README for use in SWARM Project/aco-scheduling

This code is used as a benchmark comparison for ACO Scheduling algorithm.

Code source: RL-Job-Shop-Scheduling https://github.com/prosysscience/RL-Job-Shop-Scheduling/tree/master


**Begin by setting up your environment:**
1. Ensure you are in the RL-Job-Shop-Scheduling directory
2. run 'source setup.sh'


**Instructions to run code:**
Reinforcement Learning:
You can change the benchmark problem to be solved by editing the instance_path in the main function of main.py
1. run 'cd JSS/JSS'
2. run 'python main.py'

Instructions to run heuristics (FIFO and Most Time Remaining)
You can change the benchmark problem to be solved by editing the instance_path in 'default_config.py'

1. Ensure you are in the RL-Job-Shop-Scheduling directory
2. run 'source setup.sh'
3. run 'python -m JSS.dispatching_rules.FIFO'
