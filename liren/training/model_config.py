from ml_collections import ConfigDict, FieldReference, config_dict
from liren.model.agents.continuous.cql import (
    ContinuousCQLAgent,
    get_default_config as get_default_cql_config,
)
from liren.model.agents.continuous.gc_bc import GCBCAgent
import os

def get_config(config_str: str) -> ConfigDict:
    if config_str == "gc_cql":
        return ConfigDict(
            {
                "agent_config": get_default_cql_config(
                    dict(
                        goal_conditioned=True,
                        early_goal_concat=True,
                        cql_autotune_alpha=False,
                        cql_alpha=30,  # was 30 in my setup
                        cql_temp=1.0,
                        discount=0.97,
                        cql_importance_sample=False,
                        use_calql=False,
                        actor_use_proprio=False,
                        critic_use_proprio=True,
                        history_len=1,
                        target_entropy=-1.0,
                        temperature_init=0.1,
                        critic_feed_actions = True,
                        use_dr3_reg = True,
                        gamma = 10e-4, # dr3 weighting
                    )
                ),
                "agent_name": "gc_cql",
                "agent_cls": ContinuousCQLAgent,
                "discount": 0.97,
                "wandb_proj": "cql_models_varied",
                "batch_size": 64, # 1024,
                "validate": 0.05, # None, # 0.05
                "val_steps": 3000,
                "train_steps": 500_000,
                "image_size": 64,
                "pooling": "spatial_softmax",
                "train_buffer_size": 2500, 
            }
        )
    elif config_str == "gc_bc":
        return ConfigDict(
            {
                "agent_config": ConfigDict(
                    {
                        "shared_goal_encoder": False,
                        "early_goal_concat": True,
                        "learning_rate": 3e-4,
                        "policy_kwargs": {
                            "fixed_std": [0.1, 0.1],
                            "std_parameterization": "fixed",
                            "tanh_squash_distribution": False,
                        },
                    }
                ),
                "agent_name": "gc_bc",
                "agent_cls": GCBCAgent,
                "discount": 0.97,
                "wandb_proj": "tpu_finetuning",
                "batch_size": 1024,
                "validate": 0.05,
                "val_steps": 3000,
                "train_steps": 500_000,
                "image_size": 64,
                "pooling": "avg",
            }
        )
    else:
        raise ValueError(f"Unknown config {config_str}")
