"""WRITING OUR CODE FOR TRAINING THE MODEL BASED ON PRESET LAYERS"""

import os
import argparse
import torch
from src.env import create_train_env
from src.model import ActorCritic
from src.optimizer import GlobalAdam
from src.process import local_train, local_test
import torch.multiprocessing as _mp
import shutil
os.environ['OMP_NUM_THREADS'] = '1'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="complex")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument("--num_local_steps", type=int, default=50)
    parser.add_argument("--num_global_steps", type=int, default=5e6)
    parser.add_argument("--num_processes", type=int, default=6)
    parser.add_argument("--save_interval", type=int, default=500, help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str, default="tensorboard/a3c_super_mario_bros")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--load_from_previous_stage", type=bool, default=False,
                        help="Load weight from previous trained stage")
    parser.add_argument("--use_gpu", type=bool, default=True)
    args = parser.parse_args()
    return args


def train(_args_):
    torch.manual_seed(123)
    if os.path.isdir(_args_.log_path):
        shutil.rmtree(_args_.log_path)
    os.makedirs(_args_.log_path)
    if not os.path.isdir(_args_.saved_path):
        os.makedirs(_args_.saved_path)
    mp = _mp.get_context("spawn")
    env, num_states, num_actions = create_train_env(_args_.world, _args_.stage, _args_.action_type)
    global_model = ActorCritic(num_states, num_actions)
    if _args_.use_gpu:
        global_model.cuda()
    global_model.share_memory()
    if _args_.load_from_previous_stage:
        if _args_.stage == 1:
            previous_world = _args_.world - 1
            previous_stage = 4
        else:
            previous_world = _args_.world
            previous_stage = _args_.stage - 1
        file_ = "{}/a3c_super_mario_bros_{}_{}".format(_args_.saved_path, previous_world, previous_stage)
        if os.path.isfile(file_):
            global_model.load_state_dict(torch.load(file_))

    optimizer = GlobalAdam(global_model.parameters(), lr=_args_.lr)
    processes = []
    for index in range(_args_.num_processes):
        if index == 0:
            process = mp.Process(target=local_train, args=(index, _args_, global_model, optimizer, True))
        else:
            process = mp.Process(target=local_train, args=(index, _args_, global_model, optimizer))
        process.start()
        processes.append(process)
    process = mp.Process(target=local_test, args=(_args_.num_processes, _args_, global_model))
    process.start()
    processes.append(process)
    for process in processes:
        process.join()


if __name__ == "__main__":
    _args_ = get_args()
    train(_args_)
