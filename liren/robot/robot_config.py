from ml_collections import ConfigDict, config_dict

def get_config(config_str: str) -> ConfigDict:
    if config_str == "create":
        return ConfigDict(
            {
                "waypoint_spacing": 0.15,
                "angle_scale": 1,
                "x_offset": -1,

                "min_linear_vel": -0.25,
                "max_linear_vel": 0.25,
                "min_angular_vel": -1.25,
                "max_angular_vel": 1.25,
            }
        )
    
    elif config_str == "jackal":
        return ConfigDict(
            {
                "waypoint_spacing": 0.3,
                "angle_scale": 2,
                "x_offset": -1,

                "min_linear_vel": -0.8,
                "max_linear_vel": 0.8,
                "min_angular_vel": -1.25,
                "max_angular_vel": 1.25,
            }
        )

    else:
        raise ValueError(f"Unknown config {config_str}")
