""" WRITING OUR TEST FUNCTIONS FOR THE AGENT TO WORK ON THE ENVIRONMENT SETUP"""

import os
import time
import argparse
import torch
from src.env import create_train_env
from src.model import ActorCritic
import torch.nn.functional as f

os.environ['OMP_NUM_THREADS'] = '1'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="complex")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--output_path", type=str, default="output")
    args = parser.parse_args()
    return args


def test(_args_):
    torch.manual_seed(123)
    env, num_states, num_actions = create_train_env(_args_.world, _args_.stage, _args_.action_type,
                                                    "{}/video_{}_{}.mp4".format(_args_.output_path,
                                                                                _args_.world, _args_.stage))
    model = ActorCritic(num_states, num_actions)
    if torch.cuda.is_available():
        model.load_state_dict(
            torch.load("{}/a3c_super_mario_bros_{}_{}".format(_args_.saved_path, _args_.world, _args_.stage)))
        model.cuda()
    else:
        model.load_state_dict(
            torch.load("{}/a3c_super_mario_bros_{}_{}".format(_args_.saved_path, _args_.world, _args_.stage),
                       map_location=lambda storage, loc: storage))
    model.eval()
    state = torch.from_numpy(env.reset())
    done = True
    while True:
        if done:
            tensor_h = torch.zeros((1, 512), dtype=torch.float)
            tensor_c = torch.zeros((1, 512), dtype=torch.float)
            env.reset()
        else:
            tensor_h = tensor_h.detach()
            tensor_c = tensor_c.detach()
        if torch.cuda.is_available():
            tensor_h = tensor_h.cuda()
            tensor_c = tensor_c.cuda()
            state = state.cuda()

        logit_vals, value, tensor_h, tensor_c = model(state, tensor_h, tensor_c)
        policy = f.softmax(logit_vals, dim=1)
        action = torch.argmax(policy).item()
        action = int(action)
        state, reward, done, info = env.step(action)
        state = torch.from_numpy(state)
        env.render()  # This induces a delay which makes the rendering viewable
        time.sleep(0.1)
        if info["flag_get"]:
            print("World {} stage {} completed".format(_args_.world, _args_.stage))
            break


if __name__ == "__main__":
    _args_ = get_args()
    test(_args_)
