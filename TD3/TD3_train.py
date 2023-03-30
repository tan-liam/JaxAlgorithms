import numpy as np
import torch
import jax
import optax
import flax
import jax.numpy as jnp
import jax.random as jrandom
from flax.training import train_state
from flax import linen as nn
import math
import random
import copy

from .train_utils import train_policy_step, get_action, get_target, update_Q_step
from ..utils import JaxRNG, soft_update_params

import functools.partial


class TD3:
    def __init__(
        self, Q1_state, Q2_state, policy_state, tau, gamma, a_low, a_high, c, seed
    ):
        self.Q1_state = Q1_state
        self.Q2_state = Q2_state
        self.Q1_target_state = copy.deepcopy(Q1_state)
        self.Q2_target_state = copy.deepcopy(Q2_state)
        self.policy_state = policy_state
        self.policy_target_state = copy.deepcopy(policy_state)
        self.tau = tau
        self.gamma = gamma
        self.a_low = a_low
        self.a_high = a_high
        self.c = c
        self.num_steps = 0
        self.random = JaxRNG(seed)

    def train_step(self, states, actions, next_states, rewards, last_transition):
        key = self.random.get_key()
        predicted_actions = get_action(
            self.policy_state, next_states, key, self.c, self.a_low, self.a_high
        )
        target = get_target(
            rewards,
            self.gamma,
            last_transition,
            self.Q1_target_state,
            self.Q2_target_state,
            next_states,
            predicted_actions,
        )
        Q1_state_next, Q2_state_next, Q1_loss, Q2_loss = update_Q_step(
            self.Q1_state, self.Q2_state, states, actions, target
        )
        if self.num_steps % 2 == 0:
            policy_state_next, policy_loss = train_policy_step(
                self.policy_state, self.Q1_state, states
            )
            Q1_target_next = soft_update_params(
                Q1_state_next, self.Q1_target_state, self.tau
            )
            Q2_target_next = soft_update_params(
                Q2_state_next, self.Q2_target_state, self.tau
            )
            policy_target_next = soft_update_params(
                policy_state_next, self.policy_target_state, self.tau
            )
        else:
            policy_loss = None
            policy_state_next = self.policy_state
            Q1_target_next = self.Q1_target_state
            Q2_target_next = self.Q2_target_state
            policy_target_next = self.policy_target_state
        networks = {
            "Q1_state": Q1_state_next,
            "Q2_state": Q2_state_next,
            "Q1_target": Q1_target_next,
            "Q2_target": Q2_target_next,
            "policy_state": policy_state_next,
            "policy_target": policy_target_next,
        }
        losses = {"Q1_loss": Q1_loss, "Q2_loss": Q2_loss, "policy_loss": policy_loss}
        return networks, losses

    @partial(jax.jit, static_argnums=(0,))
    def forward_policy(self, states):
        return self.policy_state.apply_func(
            {"params": self.policy_state.params}, states
        )
