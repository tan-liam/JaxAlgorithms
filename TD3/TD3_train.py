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

import functools.partial as partial


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
        self.Q1_state, self.Q2_state, Q1_loss, Q2_loss = update_Q_step(
            self.Q1_state, self.Q2_state, states, actions, target
        )
        if self.num_steps % 2 == 0:
            self.policy_state, policy_loss = train_policy_step(
                self.policy_state, self.Q1_state, states
            )
            self.Q1_target_state = soft_update_params(
                self.Q1_state, self.Q1_target_state, self.tau
            )
            self.Q2_target_state = soft_update_params(
                self.Q2_state, self.Q2_target_state, self.tau
            )
            self.policy_target_state = soft_update_params(
                self.policy_state, self.policy_target_state, self.tau
            )

        losses = {"Q1_loss": Q1_loss, "Q2_loss": Q2_loss, "policy_loss": policy_loss}
        return losses

    @partial(jax.jit, static_argnums=(0,))
    def forward_policy(self, states):
        return self.policy_state.apply_func(
            {"params": self.policy_state.params}, states
        )
