"""PROCESSING OUR DATA AND DEFINING THE FUNCTION TO TRAIN IT IN"""

import torch
from src.env import create_train_env
from src.model import ActorCritic
import torch.nn.functional as f
from torch.distributions import Categorical
from collections import deque
from tensorboardX import SummaryWriter
import timeit


def local_train(index, _args_, global_model, optimizer, save=False):
    torch.manual_seed(123 + index)
    if save:
        start_time = timeit.default_timer()
    writer = SummaryWriter(_args_.log_path)
    env, num_states, num_actions = create_train_env(_args_.world, _args_.stage, _args_.action_type)
    local_model = ActorCritic(num_states, num_actions)
    if _args_.use_gpu:
        local_model.cuda()
    local_model.train()
    state = torch.from_numpy(env.reset())
    if _args_.use_gpu:
        state = state.cuda()
    done = True
    curr_step = 0
    curr_episode = 0
    while True:
        if save:
            if curr_episode % _args_.save_interval == 0 and curr_episode > 0:
                torch.save(global_model.state_dict(),
                           "{}/a3c_super_mario_bros_{}_{}".format(_args_.saved_path, _args_.world, _args_.stage))
            print("Process {}. Episode {}".format(index, curr_episode))
        curr_episode += 1
        local_model.load_state_dict(global_model.state_dict())
        if done:
            tensor_h = torch.zeros((1, 512), dtype=torch.float)
            tensor_c = torch.zeros((1, 512), dtype=torch.float)
        else:
            tensor_h = tensor_h.detach()
            tensor_c = tensor_c.detach()
        if _args_.use_gpu:
            tensor_h = tensor_h.cuda()
            tensor_c = tensor_c.cuda()

        log_policies = []
        values = []
        rewards = []
        entropies = []

        for _ in range(_args_.num_local_steps):
            curr_step += 1
            logit_vals, value, tensor_h, tensor_c = local_model(state, tensor_h, tensor_c)
            policy = f.softmax(logit_vals, dim=1)
            log_policy = f.log_softmax(logit_vals, dim=1)
            entropy = -(policy * log_policy).sum(1, keepdim=True)

            m = Categorical(policy)
            action = m.sample().item()

            state, reward, done, _ = env.step(action)
            state = torch.from_numpy(state)
            if _args_.use_gpu:
                state = state.cuda()
            if curr_step > _args_.num_global_steps:
                done = True

            if done:
                curr_step = 0
                state = torch.from_numpy(env.reset())
                if _args_.use_gpu:
                    state = state.cuda()

            values.append(value)
            log_policies.append(log_policy[0, action])
            rewards.append(reward)
            entropies.append(entropy)

            if done:
                break

        r = torch.zeros((1, 1), dtype=torch.float)
        if _args_.use_gpu:
            r = r.cuda()
        if not done:
            _, r, _, _ = local_model(state, tensor_h, tensor_c)

        gae = torch.zeros((1, 1), dtype=torch.float)
        if _args_.use_gpu:
            gae = gae.cuda()
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        next_value = r

        for value, log_policy, reward, entropy in list(zip(values, log_policies, rewards, entropies))[::-1]:
            gae = gae * _args_.gamma * _args_.tau
            gae = gae + reward + _args_.gamma * next_value.detach() - value.detach()
            next_value = value
            actor_loss = actor_loss + log_policy * gae
            r = r * _args_.gamma + reward
            critic_loss = critic_loss + (r - value) ** 2 / 2
            entropy_loss = entropy_loss + entropy

        total_loss = -actor_loss + critic_loss - _args_.beta * entropy_loss
        writer.add_scalar("Train_{}/Loss".format(index), total_loss, curr_episode)
        optimizer.zero_grad()
        total_loss.backward()

        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            if global_param.grad is not None:
                break
            global_param._grad = local_param.grad

        optimizer.step()

        if curr_episode == int(_args_.num_global_steps / _args_.num_local_steps):
            print("Training process {} terminated".format(index))
            if save:
                end_time = timeit.default_timer()
                print('The code runs for %.2f s ' % (end_time - start_time))
            return


def local_test(index, _args_, global_model):
    torch.manual_seed(123 + index)
    env, num_states, num_actions = create_train_env(_args_.world, _args_.stage, _args_.action_type)
    local_model = ActorCritic(num_states, num_actions)
    local_model.eval()
    state = torch.from_numpy(env.reset())
    done = True
    curr_step = 0
    actions = deque(maxlen=_args_.max_actions)
    while True:
        curr_step += 1
        if done:
            local_model.load_state_dict(global_model.state_dict())
        with torch.no_grad():
            if done:
                tensor_h = torch.zeros((1, 512), dtype=torch.float)
                tensor_c = torch.zeros((1, 512), dtype=torch.float)
            else:
                tensor_h = tensor_h.detach()
                tensor_c = tensor_c.detach()

        logit_vals, value, tensor_h, tensor_c = local_model(state, tensor_h, tensor_c)
        policy = f.softmax(logit_vals, dim=1)
        action = torch.argmax(policy).item()
        state, reward, done, _ = env.step(action)
        env.render()
        actions.append(action)
        if curr_step > _args_.num_global_steps or actions.count(actions[0]) == actions.maxlen:
            done = True
        if done:
            curr_step = 0
            actions.clear()
            state = env.reset()
        state = torch.from_numpy(state)
