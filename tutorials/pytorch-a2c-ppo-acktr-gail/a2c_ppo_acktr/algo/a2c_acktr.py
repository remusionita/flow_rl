import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from a2c_ppo_acktr.algo.kfac import KFACOptimizer


class A2C_ACKTR():
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 acktr=False):

        self.actor_critic = actor_critic
        self.acktr = acktr

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        if acktr:
            self.optimizer = KFACOptimizer(actor_critic)
        else:
            self.optimizer = optim.RMSprop(
                actor_critic.parameters(), lr, eps=eps, alpha=alpha)

    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values, action_log_probs, dist_entropy, _, direction_out = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(
                -1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))

        ######## my code here - direction target is given by comparing the
        # current reward with an running average on previous episodes.
        # if the current reward is much lower than probably the target direction
        # should specify a lane change
        _direction_target  = torch.tensor([1], dtype=torch.long)
        _current_reward = rollouts.rewards[0][0][0]
        _avg_prev_reward = torch.mean(rollouts.rewards[1:])
        if (_current_reward - _avg_prev_reward) < -0.75:
            # current reward is much lower than previous. maybe an accident
            # change the lane
            rand = np.random.randint(0,2)
            if rand == 0:
                _direction_target[0] = 0
            else:
                _direction_target[0] = 2

        _direction_target = _direction_target.expand(5,)
        # print(_current_reward, _avg_prev_reward, _direction_target)
        direction_loss = F.nll_loss(direction_out, _direction_target)

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
            # Sampled fisher, see Martens 2014
            self.actor_critic.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()

            value_noise = torch.randn(values.size())
            if values.is_cuda:
                value_noise = value_noise.cuda()

            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

            fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False

        self.optimizer.zero_grad()
        ##### my code here - add direction loss
        (value_loss * self.value_loss_coef + action_loss -
         dist_entropy * self.entropy_coef + direction_loss).backward()

        if self.acktr == False:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)

        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()
