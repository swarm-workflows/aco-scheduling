README for use in SWARM Project/aco-scheduling

This code is used as a benchmark comparison for ACO Scheduling algorithm.

Code source: RL-Job-Shop-Scheduling https://github.com/prosysscience/RL-Job-Shop-Scheduling/tree/master


Instructions to run code:
1. Ensure you are in the RL-Job-Shop-Scheduling directory
2. run 'source setup.sh'
3. run 'cd JSS'
4. run 'python -m JSS.dispatching_rules.FIFO'

Edit 'default_config.py' variable 'instance_path' in order to change the benchmark problem to solve.

If you would like to train the model:
1. cd 'JSS/JSS'
2. python main.py