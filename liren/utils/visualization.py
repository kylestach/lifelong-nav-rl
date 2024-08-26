from abc import ABC, abstractmethod
import io
from typing import List, Optional, Union
import chex

from flax.struct import PyTreeNode, field
import jax
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import wandb
from PIL import Image
import pandas as pd
from collections import deque

import tensorflow as tf

IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406])
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225])

# SUCCESSES / PLOTS


def wandb_successes_avgd(wandb_csv_dir, avg_step=1000, clip_jumps=False):
    # Load Information
    info = pd.read_csv(wandb_csv_dir)
    info = info.dropna()
    info_np = info.values
    end_counts = {  # ONLY WORKS WHEN EXPORTED IN PROPER ORDER!
        "reach":  info_np[:, 1],
        "crash": info_np[:, 4],
        "timeout": info_np[:, 7],
    }

    num_samples = end_counts["reach"] + end_counts["crash"] + end_counts["timeout"]

    # Remove instances where too many new samples were added in a given step
    if clip_jumps:
        step_sample_increases = num_samples[1:] - num_samples[:-1]
        network_issues = np.arange(len(step_sample_increases))[
            step_sample_increases > 10]
        corrected_end_counts = {
            "reach":  info_np[:, 1],
            "crash": info_np[:, 4],
            "timeout": info_np[:, 7],
        }
        for broken in network_issues:
            for key in corrected_end_counts.keys():
                corrected_end_counts[key][broken:] -= (
                    corrected_end_counts[key][broken+1] - corrected_end_counts[key][broken])

        end_counts = corrected_end_counts

    # Average Out
    avgd_data = {
        "reach": [],
        "crash": [],
        "timeout": [],
    }

    end_counts_diff = {}
    for key in end_counts.keys():
        end_counts_diff[key] = [end_counts[key][i + avg_step] - end_counts[key][i]
                                for i in range(len(end_counts[key]) - avg_step)]

    num_intervals = 0
    for i in range(len(end_counts["reach"]) - avg_step):
        total_ends = np.sum([end_counts_diff[key][i:i+avg_step]
                            for key in end_counts.keys()])
        for end in end_counts.keys():
            avgd_data[end].append(np.sum(end_counts_diff[end][i:i+avg_step]) / total_ends)
        num_intervals += 1

    # Visualize
    plt.figure(figsize=(10, 3))
    for end in end_counts.keys():
        plt.plot(np.arange(num_intervals),
                 avgd_data[f"{end}"], label=f"Last {avg_step * 10} {end}")

    plt.legend()
    plt.show()


def traj_dataset_successes_avgd(dataset, avg_step):
    data_iter = dataset.iterator()
    save = avg_step
    ends = deque(maxlen=save)
    percents_reached = [0]
    percents_crashed = [0]
    percents_timeout = [0]
    percents_keepout = [0]

    for traj in data_iter:
        if traj["observation"]["goal"]["reached"][-1]:
            ends.append("reach")
        elif traj["observation"]["crash"][-1]:
            ends.append("crash")
        elif "keepout" in traj["observation"].keys() and traj["observation"]["keepout"][-1]:
            ends.append("keepout")
        else:
            ends.append("timeout")
        percents_timeout.append(ends.count("timeout") / len(ends))
        percents_reached.append(ends.count("reach") / len(ends))
        percents_crashed.append(ends.count("crash") / len(ends))
        percents_keepout.append(ends.count("keepout") / len(ends))

    plt.figure(figsize=(10, 3))
    x_axes = np.arange(len(percents_reached))
    plt.plot(x_axes, percents_reached, label=f"Last {save} Reached")
    plt.plot(x_axes, percents_crashed, label=f"Last {save} Crashed")
    plt.plot(x_axes, percents_keepout, label=f"Last {save} Keepout")
    plt.plot(x_axes, percents_timeout, label=f"Last {save} Timeout")
    plt.legend()
    plt.show()


def traj_dataset_successes_print(dataset):
    dataset_iter = dataset.iterator()

    total_len = 0
    num_trajs = 0

    num_ends_expained = {"reach": 0, "crash": 0, "keepout": 0}
    num_ends_tfds = {'is_terminal': 0, 'is_last': 0, 'is_first': 0}

    lin_vel_0 = 0

    for traj in dataset_iter:
        if traj["observation"]["goal"]["reached"][-1]:
            num_ends_expained["reach"] += 1
        if traj["observation"]["crash"][-1]:
            num_ends_expained["crash"] += 1
        elif "keepout" in traj["observation"].keys() and traj["observation"]["keepout"][-1]:
            num_ends_expained["keepout"] += 1

        if max(np.array(traj["observation"]["linear_velocity"]).flatten()) <= 0:
            lin_vel_0 += 1

        for key in num_ends_tfds.keys():
            num_ends_tfds[key] += int(traj[key][-1])

        total_len += traj["_len"][-1]
        num_trajs += 1

    print(f"{num_trajs} trajectories with total len {total_len} (avg len {total_len / num_trajs:.2f})")
    print(f"{num_ends_expained['reach']} reached ({num_ends_expained['reach'] / num_trajs :.2f}), {num_ends_expained['crash']} crashed, {num_ends_expained['keepout']} keepouts, {num_trajs - sum(num_ends_expained.values())} timed out ")


def view_img_on_axes(img, axes, scaling = False, ):
    if isinstance(img, bytes) or isinstance(img, str):
        img = tf.io.decode_image(img, expand_animations=False)
        img = np.array(img)
        # img = np.array(img, dtype = float)
    
    if scaling:
        print("Scaling image")
        img = img * IMAGENET_STD + IMAGENET_MEAN

    axes.imshow(img)

def view_img(img, scaling = False):
    if isinstance(img, bytes) or isinstance(img, str):
        img = tf.io.decode_image(img, expand_animations=False)
        img = np.array(img) # , dtype = float)
    
    if scaling:
        print("Scaling image")
        img = img * IMAGENET_STD + IMAGENET_MEAN

    plt.imshow(img)
    plt.show()


def traj_view_imgs(data, iter=False, string_imgs=True, scaling=False, num_show=10):
    if not iter:
        data = data.iterator()

    traj = next(data)
    fig, axes = plt.subplots(
        1, min(num_show, traj['_len'][0]), figsize=(12, 5))

    for i in range(min(num_show, traj['_len'][0])):
        if string_imgs:
            img = tf.io.decode_image(
                traj["observation"]["image"][i], expand_animations=False)
        else:
            img = traj["observation"]["image"][i]

        if scaling:
            img = img * IMAGENET_STD + IMAGENET_MEAN

        axes[i].imshow(img)
    plt.show()


def traj_pos_heatmap(data, title="heatmap"):
    positions = []
    data_iter = iter(data.flatten())
    for step in data_iter:
        positions.append(np.array(step["observation"]["position"]))

    positions = np.array(positions)
    plt.title(title)
    plt.scatter(positions[:, 0], positions[:, 1])
    plt.axis("equal")
    plt.show()


def distance_travelled(traj, idx_start, idx_end):
    total_dist = 0
    for i in range(idx_start, idx_end - 1):
        total_dist += np.linalg.norm(traj["observation"]["position"]
                                     [i+1][:2] - traj["observation"]["position"][i][:2])
    return total_dist


def plot_continuous_action_histogram(dataset_name, dataset_iter, samples=3000):
    actions = []

    for i in range(samples):  # 6 seconds for 3000
        one_step = next(dataset_iter)
        actions.append(one_step["actions"])

    actions = np.array(actions)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = ["X Action", "Angle"]
    for i, label in enumerate(labels):
        axes[i % 2].hist(actions[:, i])
        axes[i % 2].set_title(dataset_name + ' ' + label)

    plt.tight_layout
    plt.show()


def plot_discrete_action_histogram(dataset_name, dataset_iter, samples=3000):
    actions = []

    for i in range(samples):
        one_step = next(dataset_iter)
        actions.append(one_step["actions"])
    actions = np.array(actions)

    plt.hist(actions)
    plt.title(f"{dataset_name} Actions")
    print("Unique actions:", np.unique(actions))

    plt.show()


def plot_sample_images(dataset_name, dataset_iter):
    one_step = next(dataset_iter)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(one_step["observations"]["image"]
                   * IMAGENET_STD + IMAGENET_MEAN)
    axes[0].set_title(f"{dataset_name} observation image")

    axes[1].imshow(one_step["goals"]["image"] * IMAGENET_STD + IMAGENET_MEAN)
    axes[1].set_title(f"{dataset_name} goal image")

    plt.tight_layout
    plt.show()


def plot_image_histogram(dataset_name, dataset_iter, samples=1000, bins=10):
    img_entries = {"r": [], "g": [], "b": []}

    for i in range(samples):  # approx 6 seconds for 3000
        one_step = next(dataset_iter)
        for j, color in enumerate(img_entries.keys()):
            img_entries[color].extend(
                np.array(one_step["observations"]["image"])[:, :, j].flatten())

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for i, color in enumerate(img_entries.keys()):
        axes[i // 2, i % 2].hist(img_entries[color], bins=bins)
        axes[i // 2, i % 2].set_title(f"{color} dist for {dataset_name}")

    axes[1, 1].hist(img_entries["r"] + img_entries["b"] +
                    img_entries["g"], bins=bins)
    axes[1, 1].set_title(f"Combined colors for {dataset_name}")

    plt.tight_layout
    plt.show()

# CLASSES BELOW
class Visualization(PyTreeNode, ABC):
    @abstractmethod
    def visualize(self): ...


class RolloutGroup(PyTreeNode):
    rollouts: Union[jax.Array, np.ndarray]
    color: Optional[str] = field(pytree_node=False, default=None)
    label: Optional[str] = field(pytree_node=False, default=None)
    linestyle: Optional[str] = field(pytree_node=False, default=None)
    alpha: float = field(pytree_node=False, default=1.0)

    def __getitem__(self, i: int):
        return RolloutGroup(
            self.rollouts[i],
            self.color,
            self.label,
        )

    def to_numpy(self):
        return RolloutGroup(np.asarray(self.rollouts), self.color, self.label)

    def plot(self, ax: Axes):
        positions = self.rollouts[..., :2]
        directions = self.rollouts[..., 2:]

        if positions.ndim == 2:
            positions = positions[None]
            directions = directions[None]

        init_pos = np.broadcast_to(np.array([0, 0]), positions[:, :1].shape)
        init_dir = np.broadcast_to(np.array([1, 0]), positions[:, :1].shape)
        positions = np.concatenate([init_pos, positions], axis=1)
        directions = np.concatenate([init_dir, directions], axis=1)

        # Plot
        px, py, dx, dy = (
            positions[..., 0],
            positions[..., 1],
            directions[..., 0],
            directions[..., 1],
        )
        ax.plot(px.T, py.T, color=self.color,
                label=self.label, linestyle=self.linestyle)
        ax.quiver(
            px.flatten(),
            py.flatten(),
            dx.flatten(),
            dy.flatten(),
            color=self.color,
            alpha=self.alpha,
            angles="xy",
            scale_units="xy",
            zorder=2,
        )


def visualize_rollout(
    image: np.ndarray,
    goal: np.ndarray,
    rollout_groups: List[RolloutGroup],
    time_to_goal: np.ndarray,
):
    # Make a figure
    fig, axs = plt.subplots(1, 3, figsize=(10, 5), dpi=300)

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])
    axs[0].imshow(np.clip(image * IMAGENET_STD + IMAGENET_MEAN, 0, 1))
    axs[1].imshow(np.clip(goal * IMAGENET_STD + IMAGENET_MEAN, 0, 1))

    # Rollout actions
    for group in rollout_groups:
        group.plot(axs[2])

    axs[2].set_aspect("equal")
    axs[2].legend()

    fig.suptitle(f"Time to goal: {time_to_goal.item():.2f}")

    data_out = io.BytesIO()
    fig.savefig(data_out)
    return wandb.Image(Image.open(data_out))


class RolloutVisualization(Visualization):
    image: jax.Array
    goal: jax.Array
    rollout_groups: list[RolloutGroup]
    time_to_goal: jax.Array

    def visualize(self):
        batch_size = self.image.shape[0]

        image = np.asarray(self.image)
        goal = np.asarray(self.goal)
        rollout_groups = [g.to_numpy() for g in self.rollout_groups]
        time_to_goal = np.asarray(self.time_to_goal)

        return [
            visualize_rollout(
                image[i],
                goal[i],
                [group[i] for group in rollout_groups],
                time_to_goal[i],
            )
            for i in range(batch_size)
        ]
