# Lifelong Autonomous Improvement of Navigation Foundation Models in the Wild


The LiReN model uses reinforcement learning and is capable of general goal-conditioned visual navigation. It is similar to existing [General Navigation Models](https://github.com/robodhruv/visualnav-transformer), but it leverages the conservative q-learning algorithm instead of behavior cloning, which makes it more optimal for fine-tuning. 

Furthermore, this project introduces an autonomous improvement framework which has the flexibility to be adapted for use with other machine learning models and robot emobodiments. This allows for quick customization, whether that involves deploying a policy on a new robot, seeing if a model architecture can fine-tune effectively online, or fine-tuning to a new environment. At the moment, we have capabilities defined for fine-tuning on the iRobot Create, and we hope that this codebase is general enough to allow for simple adaptation to other ROS-based robots. 

## Overview

This repository contains code for training, deploying, and fine-tuning the LiReN model. It is organized as follows.
- `./data/`: contains methods and classes for processing data (`load_data.py`, `dataset_transforms.py`, `data_config.py`), deploying or evaluating models (`manual_goal_loop.py`, `tfds_to_npz.py`), and collecting robot data (`recorder_remote.py`). 
- `./model/`: contains model files implementing the soft actor critic algorithm, conservative Q-learning, and behavior cloning in JAX.
- `./robot/actor.py`: main script that runs a policy on a robot (for either deploying or fine-tuning) and can save robot data either locally or to a remote server.
- `./training/agent.py`: defines an abstraction that makes it easy to switch between model architectures.
- `./training/train.py`: main script that trains or fine-tunes a policy on either offline data, online data from a robot, or a mixture of the two. 
- `./utils/`: contains helper functions used in training, fine-tuning, deploying, and assessing models. 


Besides that, this repository provides helpful abstractions for interacting with a robot.
- `./robot/create/robot_action_server.py`: script that handles all the ROS interfacing with the robot and allows training and deployment scripts to not rely on ROS. The iRobot Create's robot action server is quite involved, handling multiple types of actions, docking, goal visualization, and more. It is able to handle autonomous fine-tuning. 
- `./robot/generic/robot_action_server.py`: script that handles the bare minimum interaction with the robot for deployment, meaning it can send image observations to the policy and take twist actions. 
- `./robot/create/state_machine.py`: class that the robot action server uses in order to keep track of what the robot is doing and switch between different actions. The iRobot Create's state machine handles a variety of states, including Teleop, Docking, and Nav2 actions. 
- `./robot/generic/state_machine.py`: barebones class that the minimal robot action server uses. It handles when the robot is idling or taking an action. 

Finally, this repository contains code for deploying the full autonomous system on an iRobot Create.
- `./robot/create/deployment` contains scripts to accomplish this and files to add to a ROS2 package. 


## Model Deployment
The LiReN model shows good goal-reaching and collision avoidance behavior on multiple robots. The robot action server makes it simple to deploy the model on any robot which uses ROS, has a camera which publishes RGB images, and can operate based on twist commands.

### Setup 
First, you need to set up an environment. This guide assumes using conda as your environment manager, but can be adjusted to others. Make sure to run these commands from the topmost directory, `lifelong-nav-rl`

1. Create a new conda environment: 
```
conda env create -f liren/utils/deploy_environment.yml
```
2. Source the conda environment, so the rest of the steps occur in it:
```
conda activate liren-dep
```
3. If you are using a GPU or TPU, install the correct version of jax and jaxlib by following along with the [JAX installation instructions](https://jax.readthedocs.io/en/latest/installation.html). 
4. Install the liren package:
```
pip install -e . 
```
5. Install the `agentlace` package from [this repo](https://github.com/youliangtan/agentlace):
```
git clone https://github.com/youliangtan/agentlace.git
pip install -e agentlace/
```

### Robot 
In order to deploy the model, the robot must use ROS2. Set the image and command topics in `./robot/generic/robot_action_server.py` to match what the robot expects. We recommend using compressed images, if possible. Launch the robot, and then run `python ./robot/generic/robot_action_server.py` on the same machine.

A great way to check if the robot action server is set up correctly is to connect a client to it and make sure it can a) recieve observations and b) take actions. Here's an example of how to do that:

```
from agentlace.action import ActionServer, ActionConfig, ActionClient
from liren.utils.trainer_bridge_common import (
    make_action_config,
)
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# Connect to robot action server 
action_config = make_action_config("generic")
action_client = ActionClient(
        "ROBOT_IP", # Enter your robot IP address here 
        action_config,
    )

# Turn counterclockwise 
action_client.act("action_vw", np.array([0, 1]))

# Visualize image observation
obs = action_client.obs()
pic = obs["image]
if isinstance(obs["image"], (bytes, str)):
    pic = tf.io.decode_image(pic, expand_animations=False)
plt.imshow(pic)
```

### Goal Trajectory
Since this model is goal-conditioned, a goal loop must be collected before it can be deployed. For simple model deployment,the fastest way to get a working goal trajectory is by running `./liren/data/manual_goal_loop.py`, as shown below. 
```
python ./liren/data/manual_goal_loop.py \
    --robot_ip localhost \
    --img_type numpy \
    --img_size 64 \
    --save_path /goal_traj_dir/goal_traj.npz
```

Make sure to adjust the robot ip address, image type (numpy or str, depending on if you're using raw or compressed images), image size (it must match what your model expects, our default is 64), and save path as appropriate. This script will prompt you to indicate when to capture images or save the final trajectory. Please note the first image in the trajectory should match the initial position, and will not be used as a goal. 


The method above only saves an image at each location, which is sufficient to run the model. There is a more thorough way of collecting goals including not only the image but also the ground truth position. This is necessary for fine-tuning and will be discussed in that section. 

### Running the Model 
To deploy the model, run `./liren/robot/actor.py` with the appropriate flags set. 

If you deploy locally on the robot and don't save data, you would use the following command:

```
python ./liren/robot/actor.py \
    --robot_ip localhost \
    --obs_type generic \
    --robot_type jackal \
    --data_save_location none \
    --action_type gc_cql_local \
    --checkpoint_load_dir /checkpoint_dir/generic_liren_495k \
    --checkpoint_load_step 495000 \
    --deterministic \
    --goal_npz /goal_traj_dir/goal_traj.npz \
    --step_by_one \ 
    --manually_advance 
```


If you want to use our checkpoints, you can download them [here](https://drive.google.com/drive/folders/18NhpmJq7fnMppVPpADdE7vvYjScmgYrs?usp=drive_link). 


If you would like to run the model on a separate computer than the one the robot action server is running on, we recommend [TailScale VPN](https://tailscale.com/) to simplify connections between machines on different networks. 

There are a number of additional flags that can be added to `actor.py` for versatile functionality. There will be additional ones discussed in the fine-tuning section. For simplicity here, we assume the desired outcome is to deploy the model on a trajectory and manually assess goals as "reached" to advance to the next goal, so there is no need to have access to the ground truth positions.

- `--robot_ip` must correspond to the ip address of the machine which is running the robot action server. 
- `--obs_type` must correspond to the observation type used by the robot action server, corresponding to one of the configurations defined in `./utils/trainer_bridge_common.py`. 
- `--robot_type` must correspond to a robot emobiement which you have defined in `./robot/robot_config.py` in order to properly scale and trim the actions output by the model. 

- `--max_time` sets how long this script will run for at a maximum. It is helpful if you want to assess how many goals a robot can successfully complete in a specified time frame. By default, it is set to None and the program will run until it is killed or all the goals are reached. 

- `--data_save_location` determines where to save data. The options are "none", "local", and "remote":
    - `none` corresponds to not saving data anywhere. 
    - `local` corresponds to saving data locally. In this case, you must also specify the `--data_save_dir` flag to indicate where collected trajectories should be saved. Your environment must also include the (dlimp package)[https://github.com/kvablack/dlimp]. 
    - `remote` corresponds to saving data to another machine. In this case, you must also specify the flag `--saver_ip` to indicate the ip address of the machine running `./data/recorder_remote.py`. 
- `--action_type` determines how to take actions. The options are "gc_cql_local", "gc_cql_remote", "gc_bc_local", "gc_bc_remote", "random", "inplace", "teleop", and "forward_jerk". 
    - `[model_type]_local` corresponds to loading a policy from a local checkpoint. In this case, you must also specify the `--checkpoint_load_dir` and `-checkpoint_load_step` flags to indicate where to find the orbax checkpoint being used. You also may optionally include the `--deterministic` flag to take deterministic actions instead of sampling them from the computed distribution.
    - `[model_type]_remote` is only expected to be used for fine-tuning.
    - `random` takes random actions, with means and standard deviations for the linear and angular components set in `./training/agent.py`. 
    - `inplace` takes a random angular action. 
    - `teleop` takes no action, instead allowing for the robot to be controlled in another way.
    - `forward_jerk` moves forward at a regular interval set at `action_step` in `./robot/actor.py`. This is helpful for ensuring that data collection isn't suffering from latency issues. 
    - More action types could be added based on other models being deployed. Currently, the supported options are `gc_cql` and `gc_bc`, but more could be added to `./training/agent.py`. 
- `--goal_npz` determines where to load goals from. This should correspond to a `.npz` file, like one collected by `./data/manual_goal_loop.py`. 
    - `--step_by_one` advances the goals deterministically to the next one instead of sampling randomly along the path. 
    - `--manually_advance` lets the user make a goal as "reached" manually by pressing `n`. This is required if goal positions are not specified and the robot is not providing its position from the robot action server. 


## Model Training 
The base model used in this project was trained using a conservative q-learning algorithm on [GNM dataset](https://github.com/robodhruv/drive-any-robot), which is comprised of a mix of datasets from different robot embodiements. 

### Setup
Setup for model training is quite similar to model deployment, but involves a few more packages for handling data. Make sure to run these commands from the topmost directory, `lifelong-nav-rl`

1. Create a new conda environment: 
```
conda env create -f ./liren/utils/train_environment.yml
```
2. Source the conda environment, so the rest of the steps occur in it:
```
conda activate liren-train
```
3. If you are using a GPU or TPU, install the correct version of jax and jaxlib by following along with the [JAX installation instructions](https://jax.readthedocs.io/en/latest/installation.html). 
4. Install the liren package:
```
pip install -e . 
```
5. Install the `agentlace` package from [this repo](https://github.com/youliangtan/agentlace):
```
git clone https://github.com/youliangtan/agentlace
pip install -e agentlace/
```
6. Install the `dlimp` package from [this repo](https://github.com/kvablack/dlimp):
```
git clone https://github.com/kvablack/dlimp
pip install -e dlimp/
```

### Datasets
For GNM dataset structure and information, please reference the data wrangling section of the [General Navigation Models repository](https://github.com/robodhruv/visualnav-transformer). 

In order to convert datasets into the RLDS format, please follow along with the instructions in the [GNM RLDS Dataset Builder repository](https://github.com/kylestach/gnm_rlds_dataset_builder). 

The datasets can be loaded directly and inspected at the trajectory level by using tfds functions alongside the DLataset wrapper from the dlimp package. 
```
import tensorflow_datasets as tfds
import tensorflow as tf 
from dlimp.dataset import DLataset
import matplotlib.pyplot as plt

# Load Dataset
dataset_name = "dataset_name:0.0.1"
dataset_dir = "~/dataset_dir/" 
dataset_builder = tfds.builder(dataset_name, data_dir=dataset_dir)
dataset = DLataset.from_rlds(dataset_builder)
data_iter = dataset.iterator()

# Visualize First Observation
first_traj = next(data_iter)
plt.imshow(tf.io.decode_image(first_traj["observation]["image"][0], expand_animations=False))
```

We found proper dataset normalization to be very important in order for the model to effectively train on data from multiple sources. In particular, each dataset has a particular waypoint spacing and x offset which transform the linear actions to be between -1 and 1 and an angle scaling factor that transforms the angular action to be between -1 and 1. 

In order to integrate another dataset, you will need to modify `./liren/data/load_data.py` to add the additional dataset's configuration to the `DATASETS` dictionary and update the `DATA_MIXES` dictionary with a blend of datasets you would like to use.

### Training Procedure 
Once you have data, you can train the policy on it by running `./liren/training/train.py`. This is a versatile script comptabile with fine-tuning, so for pre-training the base model you would use a command like:

```
python ./liren/training/train.py \
    --data_mix gnm \
    --data_dir /data_load_dir/ \
    --offline_data_config ./liren/data/data_config.py:gnm \
    --offline_data_config.reward_type dense \
    --offline_data_config.action_type twist \

    --checkpoint_save_dir /checkpoint_save_dir/ \
    --checkpoint_interval 5000 \
    --wandb_name wandb_run_name \

    --model_config ./liren/training/model_config.py:gc_cql \
    --model_config.agent_config.cql_alpha 1 \
    --model_config.agent_config.history_len 1 \
    --model_config.image_size 64 \
    --model_config.agent_config.gamma 1e-3 \
    --model_config.agent_config.critic_use_proprio \
    --model_config.agent_config.critic_feed_actions
```

Similarly to `actor.py`, there are a number of additional flags that can be added when running `train.py` for added functionality. There will be additional ones discussed in the fine-tuning section. For pre-training, here are the relevant ones:

- `--data_mix` determines what data mix to use for training, corresponding to the options defined in `./data/load_data.py`. The only option at the moment is `gnm`, but you can easily add more configurations. 
- `--data_dir` determines the path to the folder containing all of the datasets used in the chosen data mixture. 
- `--offline_data_config` determines how to process the data, as defined in `./data/data_config.py`. This includes options such as the negative goal sampling probability, the reward type, and more. 

- `--model_config` determines which training configuration to use, as defined in `./training/model_config.py`. You can change model and training parameters such as wandb project, batch size, agent type, cql alpha, history length, proprio, and more either with command-line arguments or directly in `./training/model_config.py`. 

- `--checkpoint_save_dir` determines where to save model checkpoints.
- `--checkpoint_interval` specifies after how many training steps to save the checkpoint. 


## Autonomous Fine Tuning
In order to make full use of the autonomous supervisor used for fine-tuning, there are additional requirements that must be met. Namely,
- LiDAR for localization to assess goal-reaching by absolute position 
    - The (Nav2 package)[https://docs.nav2.org/]
- Self-docking mechanism so the robot can charge when it is low on battery
- Joystick to control the robot to collect the map and goal loops
- Slack channel so the robot can alert you when it is stuck 

The existing code currently only allows for fine-tuning on an iRobot Create, but the custom robot action server provided at `./liren/robot/create/nav_robot_action_server.py`, state machine provided at `./liren/robot/create/state_machine.py`, and observation and action types in `./liren/utils/trainer_bridge_common.py` should make it straight-forward to adapt to another ROS-based robot. 

For now, we assume you have access to an iRobot Create running ROS2 with a mounted external camera and LiDAR, and have updated the topics which `./liren/robot/create/nav_robot_action_server.py` subscribes to and publishes to.

In order to be able to use the scripts below without modification, you must do the following:

1. Set up a Deployment ROS Package
    1. Create a ROS package called `deployment` in a workspace on the iRobot Create, or add it to an existing workspace.
    2. Copy in `./liren/robot/create/deployment/` to the deployment package and build it. 
    3. Make sure to source this workspace before use in every window by adding it to your ~/.bashrc.
2. Update `LIREN_DIR` path in all of the bash scripts in `./liren/robot/create/deployment/` to correspond to the location you have cloned this repo.
3. Set up a new conda environment: 
```
conda env create -f ./liren/utils/finetune_environment.yml
```
4. Source the conda environment, so the rest of the steps occur in it:
```
conda activate liren-fine
```
5. If you are using a GPU or TPU, install the correct version of jax and jaxlib by following along with the [JAX installation instructions](https://jax.readthedocs.io/en/latest/installation.html). 
6. Install the liren package:
```
pip install -e . 
```
7. Install the `agentlace` package from [this repo](https://github.com/youliangtan/agentlace):
```
git clone https://github.com/youliangtan/agentlace
pip install -e agentlace/
```
8. Install the `dlimp` package from [this repo](https://github.com/kvablack/dlimp):
```
git clone https://github.com/kvablack/dlimp
pip install -e dlimp/
```


### Map Collection & Keepout Zones
As a first step, the environment you will be fine-tuning in must be mapped, so that you can record absolute positions. 

1. On the iRobot Create, launch the map collection script. Make sure there are no errors in any of the panes. 
```
./liren/robot/create/deployment/create_map.sh
```
2. Open a foxglove session and monitor it while you perform the mapping by viewing the 'map' and 'scan' topics.
3. Using the joystick, teleop the robot around the environment. 
4. When the map looks good in foxglove, save it by moving to the adjacent pane, entering a map name, and pressing enter. 
5. Convert the data file and posegraph file saved to ./liren/robot/create/deployment/maps/$1 to a .pgm and .yaml files.
```
ros2 run nav2_map_server map_saver_cli -f ./liren/robot/create/deployment/maps/$1
```
6. Update the map referenced in `./liren/robot/create/deployment/launch_robot.sh` to match the collected map .yaml file. 

### Robot Launch

Now, you can launch the robot and keepout zones in two steps. 

First, run
```
./liren/robot/create/deployment/launch_robot.sh
```
This command opens 5 windows:

1. Disables the iRobot Create's reflexes and launches the robot with `./liren/robot/create/deployment/launch/robot_launch.py`, which includes starting the robot, joystick, LiDAR, and camera. 
2. Runs the navigation stack with `ros2 launch nav2_bringup bringup_launch.py`. Make sure that the `params_file` and `map` locations are set correctly. 
3. Runs an image compressing node `./liren/robot/create/deployment/deployment/image_compressor.py` in to take raw RGB images and republish them as byte strings. 
4. Runs the robot action server `./liren/robot/create/robot_action_server.py`
5. Runs foxglove with `ros2 launch foxglove_bridge foxglove_bridge_launch.xml` so it's easy to keep track of the robot as it fine-tunes. 

Give it some time to start everything up. Once the robot action server complains about not having a keepout zone to reference, run
```
./liren/robot/create/deployment/launch_keepout.sh
```
This command opens 3 windows, corresponding to the Nav2 map server, costmap filter server, and the lifecycle node making sure both of the former are still running. 

### Goal Loop Collection
Now that the robot can localize on the map, you can collect a goal loop with absolute positions.

1. Launch the robot as described above. 
2. Launch the recorder. This will get observations from the robot action server and save them to the specified directory in the TFDS trajectory format. By default, it records 3 observations per second. 
```
python ./liren/robot/recorder.py \
    --data_save_dir  data_save_dir \
    --max_time max_recording_time \
    --server_ip robot_action_server_ip \
    --handle_crash False
```
3. Drive around until you have a goal loop completed, at which point you should kill the recorder python script. Please note that the start and end should be in approximately the same position and orientation, because goal sampling wraps around the end. 
4. Convert the collected tfds folder to a useable .npz goal loop. Specify how dense you want the goal loop to be by changing the step size. 
```
python ./liren/data/tfds_to_npz.py \
    --dataset_name  recorded_dataset_name:0.0.1 \
    --dataset_dir /recorded_dataset_dir/ \
    --save_name /goal_loop_dir/goal_loop_name.npz \
    --step 5
```

#### Aside: Precise Deployment & Evaluation 
A goal loop is necessary for fine-tuning. If, instead, you would like to evaluate the model precisely with absolute positions on an iRobot Create, you can collect a shorter loop with `python ./liren/robot/recorder.py` and extract out the desired goal positions and images by following along with the "Goal Loop Creation" section of `./liren/robot/create/deployment/visualization.ipynb`. Then, to deploy the model, you can run `./liren/robot/actor.py`. Make sure to *not* use the `--manually_advance` flag so that the program assesses if the current position is close enough to the goal before advancing to the next goal. 
```
python ./liren/robot/actor.py \
    --robot_ip localhost \
    --obs_type generic \
    --robot_type create \
    --data_save_location local \
    --data_save_dir /data_save_dir \
    --action_type gc_cql_local \
    --checkpoint_load_dir /checkpoint_dir/generic_liren_495k \
    --checkpoint_load_step 495000 \
    --goal_npz /goal_traj_dir/goal_traj.npz \
    --deterministic \
    --step_by_one 
```
In order to evaluate performance, follow along with the "Goal Assessment" section of `./liren/robot/create/deployment/visualization.ipynb` to load the intended goal trajectory and the collected deployed trajectory and see how much progress was made. 


### Self Docking
You will need to do one last preparation steps to allow for self-docking. If you don't want to be able to dock when your robot is low and resume exploration after, you can avoid this by removing the `--check_battery` flag when running `./liren/robot/actor.py`


If you want to self-dock, you will need to specify two additional flags:
- `--dock_pose` is a tuple of a pose in map coordinates from which the iRobot Create is expected to successfully dock when calling the built in `/dock` action. 
- `--dock_dir` provides the path to a .pkl file containing a dictionary with intermediate poses on the way to the dock_pose. 

Self docking falls back on classical Nav2 navigation through poses. Especially in a large environment, it makes sense to collect a couple of known safe poses from which the robot can find its way back.

The dictionary stored at `--dock_dir` should have the following format: each key is a tuple of a pose in map coordinates, and the value is the next tuple of a pose in map coordinates. Each value should also be present as a key. The final poses should have values of "done", so that the script triggers the `/dock` action. 

When the robot recognizes its battery has gotten too low and it should go to charge, it will compute the closest starting point from the key tuples and go from there. You likely know safe poses that are most reasonable for the robot to navigate to on the way back from any given position in the environment, which you'll want to include in the dictionary. These poses can be collected from a TFDS record or by echoing positions throughout the environment. 


A robust example of generating this based on the original goal loop traversing the entire floor is demonstrated in the "Docking Dictionary Creation" section of `./liren/robot/create/deployment/visualization.ipynb`, though less intermediate points may be sufficient.


### Slack Help

Sometimes, the robot may get stuck. If the latest 10 trajectories end with a crash or in a keepout zone, a slack message will be sent to ask for help, and the actor script will wait until "Enter" is pressed to resume operation. 

You will need to update your Slack Bot Token and Channel ID in `./liren/robot/actor.py`. If you are not familiar with how to set up a SlackBot or find the appropriate Channel ID, read through this tutorial on [having a bot send Slack messages](https://api.slack.com/messaging/sending).  


### Fine-Tuning
Now, to put it all together! 

1. Launch robot. This should be run locally on the iRobot Create. 

```
./liren/robot/create/deployment/launch_robot.sh
### Wait ~1 minute 
./liren/robot/create/deployment/launch_keepout.sh
```

2. Launch the training script. We do not recommend doing this on the robot locally, since it likely doesn't have enough compute to train effectively. We recommend [TailScale VPN](https://tailscale.com/) to get around machines being on different networks. You will need to be using the liren-train conda environment, as outlined in the training section. Then, run 
```
python ./liren/training/train.py \
    --online_training \
    --wait_data 1500 \
    --data_mix gnm \
    --data_dir /data_load_dir/ \

    --data_save_dir /data_save_dir/ \
    --wandb_name wandb_run_name \

    --checkpoint_load_dir /checkpoint_load_dir/generic_liren_495k \
    --checkpoint_load_step 495000 \
    --checkpoint_save_dir /checkpoint_save_dir/ \
    --checkpoint_interval 5000 \

    --offline_data_config.reward_type dense \
    --online_data_config.reward_type dense \
    
    --model_config ./training/model_config.py:gc_cql \
    --model_config.agent_config.cql_alpha 0.3 \
    --model_config.train_buffer_size 10000 \
    --model_config.agent_config.critic_use_proprio \
    --model_config.agent_config.critic_feed_actions \
```


3. Launch the actor. This can be done either on the robot, or on a separate machine. We recommend deploying on the robot to minimize the latency of taking actions. 

```
python ./liren/robot/actor.py \
    --robot_ip localhost \
    --obs_type create \
    --robot_type create \
    --action_type gc_cql_remote \
    --goal_npz /goal_traj_dir/goal_traj.npz \
    --data_save_location remote \
    --saver_ip saver.ip.address \
    --check_battery \
    --check_stuck \
    --handle_keepouts \
    --dock_pose (x,y,z,x,y,z,w) \
    --dock_dir /dock_dir/pose_dict.pkl 
```

Here are a few more details about flags for the training script which are relevant for fine-tuning:
- `--checkpoint_load_dir` says where to load the initial model checkpoint from.
- `--checkpoint_load_step` says what step of the model to load. 

- `--offline_data_config` determines how to process the offline data, as defined in `./data/data_config.py`. This includes options such as the negative goal sampling probability, the reward type, and more. This will be applied to the data loaded by `--data_mix`. 
- `--online_data_config` determines how to process the online data, as defined in `./data/data_config.py`. This includes options such as the negative goal sampling probability, the reward type, and more. This will be applied to the data collected online by the robot.

- `--wait_data` determines how many observations must be collected by the robot before the trainer begins fine-tuning the model. 

- `--taper_data` if enabled, gradually increases the amount of online data being used in the data sampling. This is generally not recommended, as it takes up a lot of space.
- `--taper_step` specifies after how many training steps to bump up the percentage of online data (10% -> 20% -> 30% -> 40% -> 50%).


Here are a few more details about flags for the actor script which are relevant for fine-tuning:
- `--check_battery` determines if the robot should check its battery status and return back to dock if it falls below a certain threshold, currently 20%. 
    - If this is enabled, `--dock_pose` and `--dock_dir` must also be specified. 
- `--check_stuck` determines if the robot should call for help if it encounters at least 10 crashes or keepouts in a row. 
- `--handle_keepouts` determines if the robot should care about keepout zones, or only crashes.


