from ml_collections import ConfigDict, config_dict


def get_config(config_str: str) -> ConfigDict:
    if config_str == "gnm":
        return ConfigDict(
            {
                "reward_type": "dense",
                "skip_crash": False,
                "discrete": False,
                "truncate_goal": False,  # don't let end of traj sample beginning as goal
                "negative_probability": 0.25, # 0.25,
                "action_type": "twist",
                "num_frame_skip": 1,
            }
        )
    
    elif config_str == "create":
        return ConfigDict(
            {
                "reward_type": "dense",
                "skip_crash": False,
                "discrete": False,
                "truncate_goal": False,  # don't let end of traj sample beginning as goal
                "negative_probability": 0.12,
                "action_type": "twist",
                "num_frame_skip": 1,
            }
        )

    else:
        raise ValueError(f"Unknown config {config_str}")
