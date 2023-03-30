import numpy as np
import torch
import copy
import jax
import optax
import flax
import jax.numpy as jnp
import jax.random as jrandom
from flax.training import train_state
from flax import linen as nn
import math
import random

from .train_utils import (
    train_policy_step,
    update_Q_step,
    get_predicted_actions,
    get_target,
)
from ..utils import JaxRNG, soft_update_params

import functools.partial as partial


class DDPG:
    def __init__(self, Q_state, policy_state, gamma, tau, seed):
        self.Q_state = Q_state
        self.Q_target_state = copy.deepcopy(Q_state)
        self.policy_state = policy_state
        self.policy_target_state = copy.deepcopy(policy_state)
        self.gamma = gamma
        self.tau = tau
        self.random = JaxRNG(seed)

    def train_step(self, states, actions, next_states, rewards, last_transition):
        predicted_actions = get_predicted_actions(self.policy_state, next_states)
        target = get_target(
            rewards,
            self.gamma,
            last_transition,
            self.Q_target_state,
            next_states,
            predicted_actions,
        )
        self.Q_state, Q_loss = update_Q_step(self.Q_state, states, actions, target)
        self.Q_target_state = soft_update_params(
            self.Q_state, self.Q_target_state, self.tau
        )
        self.policy_state, policy_loss = train_policy_step(
            self.policy_state, self.Q_state, states
        )
        self.policy_target_state = soft_update_params(
            self.policy_state, self.policy_target_state, self.tau
        )
        losses = {"Q_loss": Q_loss, "policy_loss": policy_loss}
        return losses

    @partial(jax.jit, static_argnums=(0,))
    def forward_policy(self, states):
        return self.policy_state.apply_func(
            {"params": self.policy_state.params}, states
        )
