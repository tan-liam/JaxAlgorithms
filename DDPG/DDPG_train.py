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

from .train_utils import train_policy_step, train_Q_step, soft_update_params


def train_step(target_state, Q_state, policy_state, policy_target, tau, states, actions, next_states, rewards, last_transition):
    Q_state_next, Q_loss = train_Q_step(Q_state, target_state, policy_state, states, actions, rewards, next_states, last_transition)
    policy_state_next, policy_loss = train_policy_step(policy_state, Q_state, states)
    Q_target_next = soft_update_params(Q_state_next, target_state, tau)
    policy_target_next = soft_update_params(policy_state, policy_target, tau)
    networks = {'Q_state': Q_state_next, 'Q_target': Q_target_next, "policy_state": policy_state_next, "policy_target": policy_target_next}
    losses = {'Q_loss': Q_loss, 'policy_loss':policy_loss}
    return networks, losses


