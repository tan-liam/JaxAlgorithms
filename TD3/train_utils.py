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
def get_action(policy_state, next_state, rng, c, a_low, a_high):
    actions = policy_state.apply_fn({"params": policy_state.params}, next_state)
    noise = jnp.clip(jrandom.normal(rng, actions.shape), -c, c)
    actions = jnp.clip(actions + noise, a_low, a_high)
    return actions


@jax.jit
def get_target(
    rewards,
    gamma,
    last_transition,
    Q1_target_state,
    Q2_target_state,
    next_state,
    actions,
):
    target = rewards + gamma * last_transition * np.min(
        Q1_target_state.apply_fn(
            {"params": Q1_target_state.params}, next_state, actions
        ),
        Q2_target_state.apply_fn(
            {"params": Q2_target_state.params}, next_state, actions
        ),
    )
    return target


@jax.jit
def update_Q_step(Q1_state, Q2_state, states, actions, target):
    def loss_fn_1(params):
        prediction = Q1_state.apply_fn({"params": params}, states, actions)
        loss = optax.l2_loss(prediction, target).mean()
        return loss

    def loss_fn_2(params):
        prediction = Q2_state.apply_fn({"params": params}, states, actions)
        loss = optax.l2_loss(prediction, target).mean()
        return loss

    grad_fn_1 = jax.value_and_grad(loss_fn_1)
    grad_fn_2 = jax.value_and_grad(loss_fn_2)
    loss_1, grads_1 = grad_fn_1(Q1_state.params)
    loss_2, grads_2 = grad_fn_2(Q2_state.params)
    Q1_state = Q1_state.apply_gradients(grads=grads_1)
    Q2_state = Q2_state.apply_gradients(grads=grads_2)
    return Q1_state, Q2_state, loss_1, loss_2


@jax.jit
def train_policy_step(policy_state, Q1_state, states):
    def loss_fn(params):
        actions = policy_state.apply_fn({"params": params}, states)
        loss = Q1_state.apply({"params", Q1_state.params}, states, actions)
        loss = -loss.mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(policy_state.params)
    policy_state = policy_state.apply_gradients(grads=grads)
    return policy_state, loss


@jax.jit
def soft_update_params(current_state, target_state, tau):
    target_state.params = jax.tree_map(
        lambda x, y: tau * y + x * (1 - tau), current_state.params, target_state.params
    )
    return target_state
