import copy
from functools import partial
from typing import Any, Optional

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from liren.model.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from liren.model.common.encoding import GCEncodingWrapperBC, LCEncodingWrapper
from liren.model.common.typing import Batch, PRNGKey
from liren.model.networks.actor_critic_nets import Policy
from liren.model.networks.mlp import MLP


class GCBCAgent(flax.struct.PyTreeNode):
    state: JaxRLTrainState
    lr_schedule: Any = nonpytree_field()

    @partial(jax.jit, static_argnames="pmap_axis")
    def update(self, batch: Batch, pmap_axis: str = None):
        def loss_fn(params, rng):
            rng, key = jax.random.split(rng)
            dist = self.state.apply_fn(
                {"params": params},
                (batch["observations"], batch["goals"]),
                temperature=1.0,
                train=True,
                rngs={"dropout": key},
                name="actor",
            )
            pi_actions = dist.mode()
            log_probs = dist.log_prob(batch["actions"])
            mse = ((pi_actions - batch["actions"]) ** 2).sum(-1)
            actor_loss = -(log_probs).mean()
            actor_std = dist.stddev().mean(axis=1)

            explained_variance = 1 - jnp.var(
                batch["actions"] - pi_actions, axis=0
            ) / jnp.var(batch["actions"], axis=0)

            explained_variance = {
                f"explained_variance_{i}": explained_variance[i]
                for i in range(explained_variance.shape[0])
            }

            return actor_loss, {
                "actor_loss": actor_loss,
                "mse": mse.mean(),
                "log_probs": log_probs.mean(),
                "pi_actions": pi_actions.mean(),
                "mean_std": actor_std.mean(),
                "max_std": actor_std.max(),
                **explained_variance,
            }

        # compute gradients and update params
        new_state, info = self.state.apply_loss_fns(
            loss_fn, pmap_axis=pmap_axis, has_aux=True
        )

        # log learning rates
        info["lr"] = self.lr_schedule(self.state.step)

        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames="argmax")
    def sample_actions(
        self,
        observations: np.ndarray,
        goals: np.ndarray,
        *,
        seed: Optional[PRNGKey] = None,
        temperature: float = 1.0,
        argmax=False
    ) -> jnp.ndarray:
        dist = self.state.apply_fn(
            {"params": self.state.params},
            (observations, goals),
            temperature=temperature,
            name="actor",
        )
        if argmax:
            assert seed is None, "Cannot specify seed when sampling deterministically"
            actions = dist.mode()
        else:
            actions = dist.sample(seed=seed)
        return actions

    @jax.jit
    def get_debug_metrics(self, batch, **kwargs):
        dist = self.state.apply_fn(
            {"params": self.state.params},
            (batch["observations"], batch["goals"]),
            temperature=1.0,
            name="actor",
        )
        pi_actions = dist.mode()
        log_probs = dist.log_prob(batch["actions"])
        mse = ((pi_actions - batch["actions"]) ** 2).sum(-1)

        return {
            "mse": mse,
            "log_probs": log_probs,
            "pi_actions": pi_actions,
        }

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        # example arrays for model init
        observations: FrozenDict,
        actions: jnp.ndarray,
        goals: FrozenDict,
        # agent config
        encoder_def: nn.Module,
        language_conditioned: bool = False,
        # should only be set if not language conditioned
        shared_goal_encoder: Optional[bool] = None,
        early_goal_concat: Optional[bool] = None,
        # other shared network config
        use_proprio: bool = False,
        network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": False,
            "std_parameterization": "exp",
        },
        # optimizer config
        learning_rate: float = 3e-4,
        warmup_steps: int = 1000,
        decay_steps: int = 1000000,
    ):
        if not language_conditioned:
            if shared_goal_encoder is None or early_goal_concat is None:
                raise ValueError(
                    "If not language conditioned, shared_goal_encoder and early_goal_concat must be set"
                )

            if early_goal_concat:
                # passing None as the goal encoder causes early goal concat
                goal_encoder_def = None
            else:
                if shared_goal_encoder:
                    goal_encoder_def = encoder_def
                else:
                    goal_encoder_def = copy.deepcopy(encoder_def)

            encoder_def = GCEncodingWrapperBC(
                encoder=encoder_def,
                goal_encoder=goal_encoder_def,
                use_proprio=use_proprio,
                stop_gradient=False,
            )
        else:
            if shared_goal_encoder is not None or early_goal_concat is not None:
                raise ValueError(
                    "If language conditioned, shared_goal_encoder and early_goal_concat must not be set"
                )
            encoder_def = LCEncodingWrapper(
                encoder=encoder_def,
                use_proprio=use_proprio,
                stop_gradient=False,
            )

        network_kwargs["activate_final"] = True
        networks = {
            "actor": Policy(
                encoder_def,
                MLP(**network_kwargs),
                action_dim=actions.shape[-1],
                **policy_kwargs
            )
        }

        model_def = ModuleDict(networks)

        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=0.0,
        )
        tx = optax.adam(lr_schedule)

        rng, init_rng = jax.random.split(rng)
        params = jax.jit(model_def.init)(init_rng, actor=[(observations, goals)])[
            "params"
        ]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=tx,
            rng=create_rng,
        )

        return cls(state, lr_schedule)
