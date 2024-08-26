from typing import List
import wandb
import einops
import jax
import jax.numpy as jnp
import jaxlib
import chex
import tqdm
import numpy as np
import tensorflow as tf

from liren.utils.utils import multi_vmap

device_list = jax.devices()
num_devices = len(device_list)

def masked_mean(x: np.ndarray | jax.Array, mask: np.ndarray | jax.Array, axis=None):
    chex.assert_equal_shape([x, mask])
    return x.mean(axis=axis)

def compute_best_modes(
    predicted_action_modes: jax.Array,
    target_actions: jax.Array,
    action_mask: jax.Array,
):
    *batch_dims, predict_horizon, action_dim = target_actions.shape
    num_modes = predicted_action_modes.shape[-3]
    chex.assert_shape(
        action_mask,
        [
            *batch_dims,
            predict_horizon,
        ],
    )
    chex.assert_shape(
        predicted_action_modes,
        [
            *batch_dims,
            num_modes,
            predict_horizon,
            action_dim,
        ],
    )

    predicted_action_modes_mode_first = einops.rearrange(
        predicted_action_modes,
        "... m p a -> m ... p a",
    )

    mode_mse_by_action = jnp.square(predicted_action_modes_mode_first - target_actions)
    mode_mse = masked_mean(
        jnp.sum(mode_mse_by_action, axis=-1),
        einops.repeat(action_mask, "... -> m ...", m=num_modes),
        axis=-1,
    )
    chex.assert_shape(mode_mse, [num_modes, *batch_dims])

    best_mode_idx = jnp.argmin(mode_mse, axis=0)
    chex.assert_shape(best_mode_idx, [*batch_dims])

    best_mode = multi_vmap(lambda m, i: m[i], len(batch_dims))(
        predicted_action_modes, best_mode_idx
    )
    chex.assert_shape(
        best_mode,
        [
            *batch_dims,
            predict_horizon,
            action_dim,
        ],
    )

    return best_mode


def mode_mse(
    predicted_action_modes: jax.Array,
    target_actions: jax.Array,
    action_mask: jax.Array,
    *,
    breakdown_labels: List[str] = None,
    breakdown_timesteps: bool = True,
    breakdown_history: bool = True,
) -> jax.Array:
    *batch_dims, predict_horizon, action_dim = target_actions.shape
    num_modes = predicted_action_modes.shape[-3]
    chex.assert_shape(
        action_mask,
        [
            *batch_dims,
            predict_horizon,
        ],
    )
    chex.assert_shape(
        predicted_action_modes,
        [
            *batch_dims,
            num_modes,
            predict_horizon,
            action_dim,
        ],
    )

    best_mode = compute_best_modes(predicted_action_modes, target_actions, action_mask)
    chex.assert_shape(
        best_mode,
        [
            *batch_dims,
            predict_horizon,
            action_dim,
        ],
    )
    best_mode_mse_per_action = jnp.square(best_mode - target_actions)
    best_mode_mse = best_mode_mse_per_action.sum(axis=-1)
    chex.assert_shape(best_mode_mse, [*batch_dims, predict_horizon])

    info = {
        "best_mode_mse": masked_mean(best_mode_mse, action_mask),
    }
    verbose = {}

    if breakdown_labels is not None:
        for i, label in enumerate(breakdown_labels):
            action_min_mse = best_mode_mse_per_action[..., i]
            chex.assert_shape(
                [action_min_mse, action_mask], [*batch_dims, predict_horizon]
            )
            verbose[f"best_mode_mse_{label}"] = masked_mean(action_min_mse, action_mask)

    if breakdown_timesteps:
        for t in range(predict_horizon):
            timestep_min_mse = best_mode_mse[..., t]
            timestep_action_mask = action_mask[..., t]
            chex.assert_shape(
                [timestep_min_mse, timestep_action_mask], [*batch_dims]
            )
            verbose["best_mode_mse_t{t}"] = masked_mean(
                timestep_min_mse, timestep_action_mask
            )

    if breakdown_history:
        *batch_dims_without_seq, seq_len = batch_dims
        for t in range(seq_len):
            history_min_mse = best_mode_mse[:, t]
            history_action_mask = action_mask[:, t]
            chex.assert_shape(
                [history_min_mse, history_action_mask], [*batch_dims_without_seq, predict_horizon]
            )
            verbose["best_mode_mse_hist{t}"] = masked_mean(
                history_min_mse, history_action_mask
            )

    return info, verbose


def explained_variance(y_true, y_pred):
    chex.assert_equal_shape([y_true, y_pred])
    numerator = jnp.var(y_true - y_pred)
    denominator = jnp.var(y_true)
    return jnp.clip(1 - numerator / denominator, 0, 1)


def explained_variance_report(
    target_action, predicted_action, labels=["x", "y", "cos_theta", "sin_theta"]
):
    return {
        f"explained_variance_{label}": explained_variance(
            target_action[..., i], predicted_action[..., i]
        )
        for i, label in enumerate(labels)
    }
