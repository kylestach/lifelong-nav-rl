import copy
from functools import partial
from typing import Optional, Tuple, Union

import chex
import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from liren.model.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from liren.model.common.encoding import EncodingWrapper, GCEncodingWrapper
from liren.model.common.optimizers import make_optimizer
from liren.model.common.typing import Batch, Data, Params, PRNGKey
from liren.model.networks.actor_critic_nets import Critic, Policy, ensemblize
from liren.model.networks.lagrange import GeqLagrangeMultiplier
from liren.model.networks.mlp import MLP


class SACAgent(flax.struct.PyTreeNode):
    """
    Online actor-critic supporting several different algorithms depending on configuration:
     - SAC (default)
     - TD3 (policy_kwargs={"std_parameterization": "fixed", "fixed_std": 0.1})
     - REDQ (critic_ensemble_size=10, critic_subsample_size=2)
     - SAC-ensemble (critic_ensemble_size>>1)
    """

    state: JaxRLTrainState
    step: jax.Array
    config: dict = nonpytree_field()

    # def _sample_negative_goals(self, batch, rng):
    #     """
    #     for goal/reward relabeling
    #     sample negative goals and change the batch rewards
    #     """
    #     batch_size = batch["rewards"].shape[0]
    #     neg_goal_indices = jnp.roll(jnp.arange(batch_size, dtype=jnp.int32), -1)

    #     neg_goal_mask = (jnp.arange(batch_size) / batch_size) < self.config["gc_kwargs"]["negative_proportion"]
    #     goal_indices = jnp.where(
    #         neg_goal_mask, neg_goal_indices, jnp.arange(batch_size)
    #     )

    #     new_goals = jax.tree_map(lambda x: x[goal_indices], batch["goals"])
    #     if "goal_vec" in batch["goals"].keys():
    #         # new_goals["goal_vec"] = new_goals["goal_vec"].at[neg_goal_mask, 2].set(0)
    #         zeroed_goals = jnp.concatenate([new_goals["goal_vec"][:, :2], 
    #                                         jnp.zeros((new_goals["goal_vec"].shape[0], 1))], axis = 1)
            
    #         # includes dropout: some *real* ones get set to 0, as well
    #         pos_goal_mask = (jnp.arange(batch_size) / batch_size) > (1- self.config["gc_kwargs"]["negative_dropout_proportion"]) 
    #         flagged_mask = pos_goal_mask + neg_goal_mask
    #         new_goal_vec = jnp.where(jnp.repeat(flagged_mask[:, None], repeats=3, axis=1),
    #                                   zeroed_goals, new_goals["goal_vec"])
    #         new_goals["goal_vec"] = new_goal_vec
    #     new_rewards = jnp.where(neg_goal_mask, -1, batch["rewards"])
    #     new_masks = jnp.where(neg_goal_mask, 1, batch["masks"])
        
    #     return {
    #         "goals": new_goals,
    #         "rewards": new_rewards,
    #         "masks": new_masks,
    #     }, neg_goal_mask

    def _include_goals_in_obs(self, batch, which_obs: str):
        assert which_obs in ("observations", "next_observations")
        obs = batch[which_obs]
        if self.config["goal_conditioned"]:
            obs = (obs, batch["goals"])
        return obs

    @partial(jax.jit, static_argnames=("train",))
    def forward_critic(
        self,
        observations: Union[Data, Tuple[Data, Data]],
        actions: jax.Array,
        rng: PRNGKey,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
        # get_features: bool = False,
    ) -> jax.Array:
        """
        Forward pass for critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        if jnp.ndim(actions) == 3:
            # forward the q function with multiple actions on each state
            return jax.vmap(
                lambda a: self.state.apply_fn(
                    {"params": grad_params or self.state.params},
                    observations,
                    a,
                    name="critic",
                    rngs={"dropout": rng} if train else {},
                    train=train,
                    # get_features=get_features,
                ),
                in_axes=1,
                out_axes=-1,
            )(actions)
        else:
            # forward the q function on 1 action on each state
            return self.state.apply_fn(
                {"params": grad_params or self.state.params},
                observations,
                actions,
                name="critic",
                rngs={"dropout": rng} if train else {},
                train=train,
                # get_features=get_features,
            )

    def forward_target_critic(
        self,
        observations: Union[Data, Tuple[Data, Data]],
        actions: jax.Array,
        rng: PRNGKey,
    ) -> jax.Array:
        """
        Forward pass for target critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.forward_critic(
            observations, actions, rng=rng, grad_params=self.state.target_params
        )

    def forward_policy(
        self,
        observations: Union[Data, Tuple[Data, Data]],
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> distrax.Distribution:
        """
        Forward pass for policy network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            name="actor",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_temperature(
        self, *, grad_params: Optional[Params] = None
    ) -> distrax.Distribution:
        """
        Forward pass for temperature Lagrange multiplier.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.state.apply_fn(
            {"params": grad_params or self.state.params}, name="temperature"
        )

    def temperature_lagrange_penalty(
        self, entropy: jnp.ndarray, *, grad_params: Optional[Params] = None
    ) -> distrax.Distribution:
        """
        Forward pass for Lagrange penalty for temperature.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            lhs=entropy,
            rhs=self.config["target_entropy"],
            name="temperature",
        )

    def _compute_next_actions(self, batch, rng):
        """shared computation between loss functions"""
        batch_size = batch["rewards"].shape[0]

        next_action_distributions = self.forward_policy(
            self._include_goals_in_obs(batch, "next_observations"), rng=rng
        )
        (
            next_actions,
            next_actions_log_probs,
        ) = next_action_distributions.sample_and_log_prob(seed=rng)
        chex.assert_equal_shape([batch["actions"], next_actions])
        chex.assert_shape(next_actions_log_probs, (batch_size,))

        return next_actions, next_actions_log_probs

    def _process_target_next_qs(self, target_next_qs, next_actions_log_probs):
        """classes that inherit this class can add to this function
        e.g. CQL will add the cql_max_target_backup option
        """
        if self.config["backup_entropy"]:
            temperature = self.forward_temperature()
            target_next_qs = target_next_qs - temperature * next_actions_log_probs

        return target_next_qs

    def critic_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """classes that inherit this class can change this function"""
        batch_size = batch["rewards"].shape[0]
        rng, next_action_sample_key = jax.random.split(rng)
        next_actions, next_actions_log_probs = self._compute_next_actions(
            batch, next_action_sample_key
        )
        # (batch_size, ) for sac, (batch_size, cql_n_actions) for cql

        # Evaluate next Qs for all ensemble members (cheap because we're only doing the forward pass)
        target_next_qs = self.forward_target_critic(
            self._include_goals_in_obs(batch, "next_observations"),
            next_actions,
            rng=rng,
        )  # (critic_ensemble_size, batch_size)

        # Subsample if requested
        if self.config["critic_subsample_size"] is not None:
            rng, subsample_key = jax.random.split(rng)
            subsample_idcs = jax.random.randint(
                subsample_key,
                (self.config["critic_subsample_size"],),
                0,
                self.config["critic_ensemble_size"],
            )
            target_next_qs = target_next_qs[subsample_idcs]

        # Minimum Q across (subsampled) ensemble members
        target_next_min_q = target_next_qs.min(axis=0)
        chex.assert_equal_shape([target_next_min_q, next_actions_log_probs])
        # (batch_size,) for sac, (batch_size, cql_n_actions) for cql

        target_next_min_q = self._process_target_next_qs(
            target_next_min_q,
            next_actions_log_probs,
        )

        target_q = (
            batch["rewards"]
            + self.config["discount"] * batch["masks"] * target_next_min_q
        )
        chex.assert_shape(target_q, (batch_size,))

        predicted_qs = self.forward_critic(
            self._include_goals_in_obs(batch, "observations"),
            batch["actions"],
            rng=rng,
            grad_params=params,
        )

        chex.assert_shape(
            predicted_qs, (self.config["critic_ensemble_size"], batch_size)
        )
        target_qs = target_q[None].repeat(self.config["critic_ensemble_size"], axis=0)
        chex.assert_equal_shape([predicted_qs, target_qs])
        critic_loss = jnp.mean((predicted_qs - target_qs) ** 2)

        info = {
            "critic_loss": critic_loss,
            "predicted_qs": jnp.mean(predicted_qs),
            "target_qs": jnp.mean(target_qs),
        }
        
        if self.config["goal_conditioned"] and "resample_type" in batch:
            # num_negatives = int(self.config["gc_kwargs"]["negative_proportion"] * batch_size)
            # info["negative_qs"] = jnp.mean(predicted_qs, axis=0)[:num_negatives].mean()
            # info["positive_qs"] = jnp.mean(predicted_qs, axis=0)[num_negatives:].mean()
            original_goals = batch["resample_type"] == 0
            positive_goals = batch["resample_type"] == 1
            negative_goals = batch["resample_type"] == 2

            q_mean = jnp.mean(predicted_qs, axis=0)
            info["original_qs"] = jnp.mean(q_mean * original_goals) / jnp.mean(original_goals)
            info["positive_qs"] = jnp.mean(q_mean * positive_goals) / jnp.mean(positive_goals)
            info["negative_qs"] = jnp.mean(q_mean * negative_goals) / jnp.mean(negative_goals)

        return critic_loss, info

    def policy_loss_fn(self, batch, params: Params, rng: PRNGKey):
        batch_size = batch["rewards"].shape[0]
        temperature = self.forward_temperature()

        rng, policy_rng, sample_rng, critic_rng = jax.random.split(rng, 4)
        action_distributions = self.forward_policy(
            self._include_goals_in_obs(batch, "observations"),
            rng=policy_rng,
            grad_params=params,
        )
        actions, log_probs = action_distributions.sample_and_log_prob(seed=sample_rng)

        predicted_qs = self.forward_critic(
            self._include_goals_in_obs(batch, "observations"),
            actions,
            rng=critic_rng,
        )
        predicted_q = predicted_qs.min(axis=0)
        chex.assert_shape(predicted_q, (batch_size,))
        chex.assert_shape(log_probs, (batch_size,))

        nll_objective = -jnp.mean(action_distributions.log_prob(jnp.clip(batch["actions"], -0.99, 0.99)))
        actor_objective = predicted_q
        actor_loss = -jnp.mean(actor_objective) + jnp.mean(temperature * log_probs) #+ (self.step < 10000) * nll_objective

        action_mode = action_distributions.mode()
        info = {
            "actor_loss": actor_loss,
            "actor_nll": nll_objective,
            "temperature": temperature,
            "entropy": -log_probs.mean(),
            "log_probs": log_probs,
            "actions_mse": ((action_mode - batch["actions"]) ** 2).sum(axis=-1).mean(),
            "dataset_rewards": batch["rewards"],
            "mc_returns": batch.get("mc_returns", None),
        } | {
            f"actions_explained_variance_{i}": 1 - jnp.var(action_mode[..., i] - batch["actions"][..., i]) / jnp.var(batch["actions"][..., i])
            for i in range(actions.shape[-1])
        }

        return actor_loss, info

    def temperature_loss_fn(self, batch, params: Params, rng: PRNGKey):
        rng, next_action_sample_key = jax.random.split(rng)
        next_actions, next_actions_log_probs = self._compute_next_actions(
            batch, next_action_sample_key
        )

        entropy = -next_actions_log_probs.mean()
        temperature_loss = self.temperature_lagrange_penalty(
            entropy,
            grad_params=params,
        )
        return temperature_loss, {"temperature_loss": temperature_loss}

    def loss_fns(self, batch):
        return {
            "critic": partial(self.critic_loss_fn, batch),
            "actor": partial(self.policy_loss_fn, batch),
            "temperature": partial(self.temperature_loss_fn, batch),
        }

    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def update(
        self,
        batch: Batch,
        *,
        pmap_axis: str = None,
        networks_to_update: frozenset[str] = frozenset(
            {"actor", "critic", "temperature"}
        ),
    ) -> Tuple["SACAgent", dict]:
        """
        Take one gradient step on all (or a subset) of the networks in the agent.

        Parameters:
            batch: Batch of data to use for the update. Should have keys:
                "observations", "actions", "next_observations", "rewards", "masks".
            pmap_axis: Axis to use for pmap (if None, no pmap is used).
            networks_to_update: Names of networks to update (default: all networks).
                For example, in high-UTD settings it's common to update the critic
                many times and only update the actor (and other networks) once.
        Returns:
            Tuple of (new agent, info dict).
        """
        batch_size = batch["rewards"].shape[0]
        chex.assert_tree_shape_prefix(batch, (batch_size,))

        rng, goal_rng = jax.random.split(self.state.rng)
        # Negative goal sampling
        # if self.config["goal_conditioned"]:
            # new_stats, _neg_goal_masks = self._sample_negative_goals(batch, goal_rng)
            # save the new goals and rewards
            # for k, v in new_stats.items():
            #     batch[k] = v

        # Compute gradients and update params
        loss_fns = self.loss_fns(batch)

        # Only compute gradients for specified steps
        assert networks_to_update.issubset(
            loss_fns.keys()
        ), f"Invalid gradient steps: {networks_to_update}"
        for key in loss_fns.keys() - networks_to_update:
            loss_fns[key] = lambda params, rng: (0.0, {})

        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        # Update target network (if requested)
        if "critic" in networks_to_update:
            new_state = new_state.target_update(self.config["soft_target_update_rate"])

        # Update RNG
        new_state = new_state.replace(rng=rng)

        # Log learning rates
        for name, opt_state in new_state.opt_states.items():
            if (
                hasattr(opt_state, "hyperparams")
                and "learning_rate" in opt_state.hyperparams.keys()
            ):
                info[f"{name}_lr"] = opt_state.hyperparams["learning_rate"]

        return self.replace(state=new_state, step=self.step+1), info

    @partial(jax.jit, static_argnames=("argmax",))
    def sample_actions(
        self,
        observations: Data,
        goals: Optional[Data] = None,
        *,
        seed: Optional[PRNGKey] = None,
        argmax: bool = False,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Sample actions from the policy network, **using an external RNG** (or approximating the argmax by the mode).
        The internal RNG will not be updated.
        """
        if self.config["goal_conditioned"]:
            assert goals is not None
            obs = (observations, goals)
        else:
            obs = observations
        dist = self.forward_policy(obs, rng=seed, train=False)
        if argmax:
            assert seed is None, "Cannot specify seed when sampling deterministically"
            return dist.mode()
        else:
            return dist.sample(seed=seed)

    @jax.jit
    def get_debug_metrics(self, batch, **kwargs):
        """for compatibility with other agents and training scripts"""
        return {}

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Models
        actor_def: nn.Module,
        critic_def: nn.Module,
        temperature_def: nn.Module,
        # goal conditioned (zhouzypaul: added but not tested)
        goal_conditioned: bool = False,
        goals: Optional[Data] = None,
        early_goal_concat: bool = False,
        shared_goal_encoder: bool = True,
        gc_kwargs: dict = {
            "negative_proportion": 0.0,
            "negative_dropout_proportion": 0.0,
        },
        # Optimizer
        actor_optimizer_kwargs={
            "learning_rate": 3e-4,
            "warmup_steps": 2000,
        },
        critic_optimizer_kwargs={
            "learning_rate": 3e-4,
            "warmup_steps": 2000,
        },
        temperature_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        # Algorithm config
        discount: float = 0.95,
        soft_target_update_rate: float = 0.005,
        target_entropy: Optional[float] = None,
        entropy_per_dim: bool = False,
        backup_entropy: bool = False,
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
    ):
        networks = {
            "actor": actor_def,
            "critic": critic_def,
            "temperature": temperature_def,
        }

        model_def = ModuleDict(networks)

        # Define optimizers
        txs = {
            "actor": make_optimizer(**actor_optimizer_kwargs),
            "critic": make_optimizer(**critic_optimizer_kwargs),
            "temperature": make_optimizer(**temperature_optimizer_kwargs),
        }

        rng, init_rng = jax.random.split(rng)
        network_input = (observations, goals) if goal_conditioned else observations
        params = model_def.init(
            init_rng,
            actor=[network_input],
            critic=[network_input, actions],
            temperature=[],
        )["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        # Config
        assert not entropy_per_dim, "Not implemented"
        if target_entropy is None:
            target_entropy = -actions.shape[-1] / 2

        return cls(
            state=state,
            step=jnp.zeros((), dtype=jnp.int32),
            config=dict(
                critic_ensemble_size=critic_ensemble_size,
                critic_subsample_size=critic_subsample_size,
                discount=discount,
                soft_target_update_rate=soft_target_update_rate,
                target_entropy=target_entropy,
                backup_entropy=backup_entropy,
                goal_conditioned=goal_conditioned,
                early_goal_concat=early_goal_concat,
                shared_goal_encoder=shared_goal_encoder,
                gc_kwargs=gc_kwargs,
            ),
        )

    @classmethod
    def _create_encoder_def(
        cls,
        encoder_def: nn.Module,
        actor_use_proprio: bool,
        critic_use_proprio: bool,
        enable_stacking: bool,
        goal_conditioned: bool,
        early_goal_concat: bool,
        shared_goal_encoder: bool,
        shared_actor_encoder: bool,
        history_len: int,
    ):
        if goal_conditioned:
            if early_goal_concat:
                goal_encoder_def = None
            else:
                goal_encoder_def = (
                    encoder_def if shared_goal_encoder else copy.deepcopy(encoder_def)
                )

            actor_def = GCEncodingWrapper(
                encoder=encoder_def if shared_actor_encoder else copy.deepcopy(encoder_def),
                goal_encoder=goal_encoder_def,
                use_proprio=actor_use_proprio,
                history_len=history_len,
                stop_gradient=False,
            )
            critic_def = GCEncodingWrapper(
                encoder=encoder_def,
                goal_encoder=goal_encoder_def,
                use_proprio=critic_use_proprio,
                history_len=history_len,
                stop_gradient=False,
            )

        else:
            actor_def = EncodingWrapper(
                encoder_def if shared_actor_encoder else copy.deepcopy(encoder_def),
                use_proprio=actor_use_proprio,
                stop_gradient=False,
                enable_stacking=enable_stacking,
            )
            critic_def = EncodingWrapper(
                encoder_def,
                use_proprio=critic_use_proprio,
                stop_gradient=False,
                enable_stacking=enable_stacking,
            )

        return actor_def, critic_def

    @classmethod
    def create_pixels(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Model architecture
        encoder_def: nn.Module,
        shared_encoder: bool = True,
        use_proprio: bool = False,
        critic_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": True,
            "std_parameterization": "uniform",
        },
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        temperature_init: float = 0.1,
        # goal conditioned (zhouzypaul: added but not tested)
        goal_conditioned: bool = False,
        goals: Optional[Data] = None,
        early_goal_concat: bool = False,
        shared_goal_encoder: bool = True,
        **kwargs,
    ):
        """
        Create a new pixel-based agent, with no encoders.
        """

        policy_network_kwargs["activate_final"] = True
        critic_network_kwargs["activate_final"] = True

        encoder_def = cls._create_encoder_def(
            encoder_def,
            use_proprio,
            enable_stacking=True,
            goal_conditioned=goal_conditioned,
            early_goal_concat=early_goal_concat,
            shared_goal_encoder=shared_goal_encoder,
        )

        if shared_encoder:
            encoders = {
                "actor": encoder_def,
                "critic": encoder_def,
            }
        else:
            encoders = {
                "actor": encoder_def,
                "critic": copy.deepcopy(encoder_def),
            }

        # Define networks
        policy_def = Policy(
            encoder=encoders["actor"],
            network=MLP(**policy_network_kwargs),
            action_dim=actions.shape[-1],
            **policy_kwargs,
            name="actor",
        )
        critic_backbone = partial(MLP, **critic_network_kwargs)
        critic_backbone = ensemblize(critic_backbone, critic_ensemble_size)(
            name="critic_ensemble"
        )
        critic_def = partial(
            Critic, encoder=encoders["critic"], network=critic_backbone
        )(name="critic")
        temperature_def = GeqLagrangeMultiplier(
            init_value=temperature_init,
            constraint_shape=(),
            constraint_type="geq",
            name="temperature",
        )

        return cls.create(
            rng,
            observations,
            actions,
            actor_def=policy_def,
            critic_def=critic_def,
            temperature_def=temperature_def,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            goals=goals,
            goal_conditioned=goal_conditioned,
            early_goal_concat=early_goal_concat,
            shared_goal_encoder=shared_goal_encoder,
            **kwargs,
        )

    @classmethod
    def create_states(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Model architecture
        critic_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        policy_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": True,
            "std_parameterization": "uniform",
        },
        temperature_init: float = 0.1,
        # goal conditioned (zhouzypaul: added but not tested)
        goal_conditioned: bool = False,
        goals: Optional[Data] = None,
        early_goal_concat: bool = False,
        shared_goal_encoder: bool = True,
        **kwargs,
    ):
        """
        Create a new stage-based agent, with no encoders.
        """

        policy_network_kwargs["activate_final"] = True
        critic_network_kwargs["activate_final"] = True

        # Define networks
        policy_def = Policy(
            encoder=None,
            network=MLP(**policy_network_kwargs),
            action_dim=actions.shape[-1],
            **policy_kwargs,
            name="actor",
        )
        critic_cls = partial(Critic, encoder=None, network=MLP(**critic_network_kwargs))
        critic_def = ensemblize(critic_cls, critic_ensemble_size)(name="critic")
        temperature_def = GeqLagrangeMultiplier(
            init_value=temperature_init,
            constraint_shape=(),
            constraint_type="geq",
            name="temperature",
        )

        return cls.create(
            rng,
            observations,
            actions,
            actor_def=policy_def,
            critic_def=critic_def,
            temperature_def=temperature_def,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            goals=goals,
            goal_conditioned=goal_conditioned,
            early_goal_concat=early_goal_concat,
            shared_goal_encoder=shared_goal_encoder,
            **kwargs,
        )

    @partial(jax.jit, static_argnames=("utd_ratio", "pmap_axis"))
    def update_high_utd(
        self,
        batch: Batch,
        *,
        utd_ratio: int,
        pmap_axis: Optional[str] = None,
    ) -> Tuple["SACAgent", dict]:
        """
        Fast JITted high-UTD version of `.update`.

        Splits the batch into minibatches, performs `utd_ratio` critic
        (and target) updates, and then one actor/temperature update.

        Batch dimension must be divisible by `utd_ratio`.
        """
        batch_size = batch["rewards"].shape[0]
        assert (
            batch_size % utd_ratio == 0
        ), f"Batch size {batch_size} must be divisible by UTD ratio {utd_ratio}"
        minibatch_size = batch_size // utd_ratio
        chex.assert_tree_shape_prefix(batch, (batch_size,))

        def scan_body(carry: Tuple[SACAgent], data: Tuple[Batch]):
            (agent,) = carry
            (minibatch,) = data
            agent, info = agent.update(
                minibatch, pmap_axis=pmap_axis, networks_to_update=frozenset({"critic"})
            )
            return (agent,), info

        def make_minibatch(data: jnp.ndarray):
            return jnp.reshape(data, (utd_ratio, minibatch_size) + data.shape[1:])

        minibatches = jax.tree_map(make_minibatch, batch)

        (agent,), critic_infos = jax.lax.scan(scan_body, (self,), (minibatches,))

        critic_infos = jax.tree_map(lambda x: jnp.mean(x, axis=0), critic_infos)
        del critic_infos["actor"]
        del critic_infos["temperature"]

        # Take one gradient descent step on the actor and temperature
        agent, actor_temp_infos = agent.update(
            batch,
            pmap_axis=pmap_axis,
            networks_to_update=frozenset({"actor", "temperature"}),
        )
        del actor_temp_infos["critic"]

        infos = {**critic_infos, **actor_temp_infos}

        return agent, infos
