import argparse
import multiprocessing as mp

# from ray.rllib.agents.ppo import ppo
from ray.rllib.algorithms.ppo import ppo

default_config = {
    'env': 'JSSEnv:jss-v1',
    'seed': 0,
    'framework': 'torch',
    'log_level': 'WARN',
    'num_gpus': 0,
    'instance_path': './JSS/instances/ft06',
    'num_envs_per_worker': 2,
    'rollout_fragment_length': 1024,
    'num_workers': mp.cpu_count() - 1,
    'sgd_minibatch_size': 256,
    'evaluation_interval': None,
    'metrics_smoothing_episodes': 1000,
    'gamma': 1.0,
    'layer_size': 1024,
    'layer_nb': 2,
    'lr': 7e-5,
    "lr_schedule": [[0, 7e-5], [4000000, 3e-5]],
    'clip_param': 0.3,
    'vf_clip_param': 10.0,
    'kl_target': 0.01,
    'num_sgd_iter': 25,
    'lambda': 1.0,
    "use_critic": True,
    "use_gae": True,
    "kl_coeff": 0.2,
    "shuffle_sequences": True,
    "vf_share_layers": False,
    "vf_loss_coeff": 1.0,
    "entropy_coeff": 5e-4,
    "entropy_coeff_schedule": None,
    "grad_clip": None,
    "batch_mode": "truncate_episodes",
    "observation_filter": "NoFilter",
    "simple_optimizer": False,
    "_fake_gpus": False,
}


def parse_config():
    config = default_config.copy()
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance-path')
    parser.add_argument('--store')
    args = parser.parse_args()

    if args.instance_path:
        config['instance_path'] = args.instance_path

    if args.store:
        config['store'] = args.store
    return config
