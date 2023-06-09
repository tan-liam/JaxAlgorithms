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


def create_train_state(module, rng, sample_object, learning_rate, decay_steps):
    rng, one_time_key = jrandom.split(rng, 2)
    params = module.init(rng, sample_object, one_time_key)["params"]
    learning_rate = optax.cosine_decay_schedule(learning_rate, decay_steps)
    tx = optax.adam(learning_rate=learning_rate)
    return train_state.TrainState.create(apply_fn=module.apply, params=params, tx=tx)


@jax.jit
def get_predicted_actions(policy_state, next_state):
    actions = policy_state.apply_fn({"params": policy_state.params}, next_state)
    return actions


@jax.jit
def get_target(
    rewards, gamma, last_transition, Q_target_state, next_state, predicted_actions
):
    target = rewards + gamma * last_transition * Q_target_state.apply_fn(
        {"params": Q_target_state.params}, next_state, predicted_actions
    )
    return target


@jax.jit
def update_Q_step(Q_state, states, actions, target):
    def loss_fn(params):
        prediction = Q_state.apply_fn({"params": params}, states, actions)
        loss = optax.l2_loss(prediction, target).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(Q_state.params)
    Q_state = Q_state.apply_gradients(grads=grads)
    return Q_state, loss


@jax.jit
def train_policy_step(policy_state, Q_state, states):
    def loss_fn(params):
        actions = policy_state.apply_fn({"params": params}, states)
        loss = Q_state.apply({"params", Q_state.params}, states, actions)
        loss = -loss.mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(policy_state.params)
    policy_state = policy_state.apply_gradients(grads=grads)
    return policy_state, loss
