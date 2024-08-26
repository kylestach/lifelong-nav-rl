# generic imports
from typing import Optional
import numpy as np

# jax imports
from liren.model.agents.continuous.cql import get_default_config as cql_config
import liren.model.vision.resnet_v1 as resnets
from liren.model.agents.continuous.cql import ContinuousCQLAgent
import jax
import flax
from flax.core import frozen_dict
import jax.numpy as jnp

from orbax.checkpoint import (
    CheckpointManager,
    CheckpointManagerOptions,
    PyTreeCheckpointer,
)

import optax


# implemented for gc_cql, gc_bc, gc_cql_disc, stable_contrastive_rl, [IQL], ## RANDOM! just for deployment
class Agent:
    def __init__(self, config, seed):  # basically make_model from before
        self.rng = jax.random.PRNGKey(seed=seed)
        self.config = config
        self.agent_type = self.config.agent_name
        self.actor = self.make_model()

    def make_model(self):
        encoder = resnets.ResNetEncoder(
            stage_sizes=(2, 2, 2, 2), block_cls=resnets.ResNetBlock, pooling_method=self.config.pooling, num_spatial_blocks=8
        )  # spatial softmax pooling could be nice here

        if hasattr(self.config.agent_config, 'history_len') and self.config.agent_config.history_len:
            # batch history height width colors
            obs_shape = jax.numpy.zeros(
                (1, self.config.agent_config.history_len, self.config.image_size, self.config.image_size, 3))
            goal_shape = jax.numpy.zeros(
                (1, self.config.image_size, self.config.image_size, 3))
        else:
            obs_shape = jax.numpy.zeros(
                (1, 1, self.config.image_size, self.config.image_size, 3))
            goal_shape = obs_shape[:, -1]

        if self.agent_type == "gc_cql":
            return self.config.agent_cls.create(
                rng=self.rng,
                observations={"image": obs_shape, "proprio": jax.numpy.zeros(
                    (1, 5))},  # proprio & prev action
                actions=jax.numpy.zeros((1, 2)),
                goals={"image": goal_shape},
                encoder_def=encoder,
                **self.config.agent_config,
            )
        elif self.agent_type == "gc_bc":
            return self.config.agent_cls.create(
                rng=self.rng,
                observations={"image": obs_shape},
                actions=jax.numpy.zeros(2),
                goals={"image": obs_shape[:, -1]},
                encoder_def=encoder,
                **self.config.agent_config,
            )


        else:
            raise ValueError(f"Unknown agent {self.agent_type}")

    def load_checkpoint(self, checkpoint_dir, checkpoint_step):
        self.actor = CheckpointManager(
            directory=checkpoint_dir,
            checkpointers=PyTreeCheckpointer(),
        ).restore(step=checkpoint_step, items=self.actor)
        params = frozen_dict.unfreeze(self.actor.state.params)
        if self.agent_type == "gc_cql":
            params["modules_temperature"]["lagrange"] = jax.numpy.array(-2.2)
        self.actor = self.actor.replace(
            state=self.actor.state.replace(
                params=frozen_dict.freeze(params)
            )
        )

    def update_params(self, params, target_params=None):
        self.actor = self.actor.replace(
            state=self.actor.state.replace(params=params))

        if target_params is not None:
            self.actor = self.actor.replace(target_params=target_params)

    def replicate(self):
        self.actor = flax.jax_utils.replicate(self.actor)
        self.actor = self.actor.replace(
            state=self.actor.state.replace(
                rng=jax.device_put_sharded(
                    list(jax.random.split(
                        self.actor.state.rng[0], self.actor.state.rng.shape[0])),
                    jax.local_devices(),
                )
            )
        )

    # for ONE image / goal pair
    def predict(
            self,
            *,
            obs_image: Optional[np.ndarray] = None,
            obs_state: Optional[np.ndarray] = None,
            goal_image: Optional[np.ndarray] = None,
            goal_state: Optional[np.ndarray] = None,
            random=True,
    ) -> np.ndarray:
        self.rng, key = jax.random.split(self.rng)
        if self.agent_type == "gc_cql":
            if random:
                return jnp.squeeze(
                    self.actor.sample_actions(
                        # expects batch dim
                        observations={"image": obs_image[None, :]},
                        goals={"image": goal_image[None, :]},
                        rng=self.rng,
                        seed=key,
                    ),
                    axis=0,
                )
            else:
                return jnp.squeeze(
                    self.actor.sample_actions(
                        # expects batch
                        observations={"image": obs_image[None, :]},
                        goals={"image": goal_image[None, :]},
                        argmax=True,
                    ),
                    axis=0,
                )

        elif self.agent_type == "gc_bc":
            return jnp.squeeze(
                self.actor.sample_actions(
                    observations={"image": obs_image[None, :]},
                    goals={"image": goal_image[None, :]},
                    seed=key,
                ),
                axis=0,
            )

class RandomAgent():
    def __init__(self):
        self.prev_vel = self.rand_vel()

    def rand_vel(self):
        x = max(0.05, min(0.4, np.random.normal(0.20, 0.05)))
        stdev = 1.8 / 3
        yaw = max(-1.82, min(1.82, np.random.normal(0, stdev)))
        return [x, yaw]

    def rand_action(self):
        next_vel = self.rand_vel()
        self.prev_vel[0] = max(
            0, min(0.4, 0.7 * self.prev_vel[0] + 0.23 * next_vel[0]))
        self.prev_vel[1] = max(-1.82, min(1.82, 0.9 *
                               self.prev_vel[1] + 0.3 * next_vel[1]))
        return self.prev_vel

    def reset_vel(self):
        self.prev_vel = self.rand_vel()

