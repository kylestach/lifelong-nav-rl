import time
import chex
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
import sys, select
# tf.config.set_visible_devices([], 'GPU') 

def pose_distance(
    position: np.ndarray,
    quaternion: np.ndarray,
    goal_position: np.ndarray,
    goal_quaternion: np.ndarray,
    orientation_weight: float = 1.0,
):
    # Compute quaternion distance
    q1 = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)
    q2 = goal_quaternion / np.linalg.norm(goal_quaternion, axis=-1, keepdims=True)
    d_quat = 2 * np.arccos(np.abs(np.sum(q1 * q2, axis=-1)))

    # Compute position distance
    d_pos = np.linalg.norm(position - goal_position, axis=-1)
    return d_pos + orientation_weight * d_quat

def close_enough(
    position: np.ndarray,
    quaternion: np.ndarray,
    goal_position: np.ndarray,
    goal_quaternion: np.ndarray,
    orientation_weight: float = 1.0,
):
    # check position
    if (np.abs(position - goal_position) > 0.75).any(): 
        return False
    
    # check quaternion
    q1 = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)
    q2 = goal_quaternion / np.linalg.norm(goal_quaternion, axis=-1, keepdims=True)
    d_quat = 2 * np.arccos(np.abs(np.sum(q1 * q2, axis=-1)))
    if np.abs(d_quat) > np.pi/ 4: # more than 45 degrees off 
        return False
    return True
    

class TrainingTask:
    def __init__(self, goal_file: str, step_by_one: bool, advance_manually: bool):
        # Load goal file as npz
        self.goal_data = np.load(goal_file)
        self.goal_idx = None
        self.last_reset_time = None

        self.timeout = 300.0
        self.threshold = 0.2
        self.is_first = True
        self.step_by_one = step_by_one
        self.advance_manually = advance_manually 

    def update(self, position: np.ndarray = None, quaternion: np.ndarray = None, crashed: bool = False):
        
        if position is None or quaternion is None:
            assert self.step_by_one, "Must be stepping by one to update without specified position"

        if self.goal_idx is None:
            self.select_goal_idx(position, quaternion, False)

        current_goal = self.get_goal() # original goal we were at 
        goal_position = current_goal["position"]
        goal_quaternion = current_goal["orientation"]
        timeout = time.time() - self.last_reset_time > self.timeout
        was_first = self.is_first
        self.is_first = False

        if self.advance_manually: 
            reached = False
            i, o, e = select.select([sys.stdin], [], [], 0.001)
            if i:
                received_input = sys.stdin.readline().strip()
                if 'r' in received_input or 'n' in received_input or 's' in received_input:
                    reached = True
                if 'c' in received_input:
                    crashed = True

        else: 
            reached =  close_enough(position, quaternion, goal_position, goal_quaternion)
            
        
        return {
            "goal": current_goal,
            "reached_goal": reached,
            "is_first": was_first,
            "is_terminal": (reached or crashed) and not timeout, # effictively is_last and not reached or crashed
            "is_last": reached or crashed or timeout, # this would mean we also need to reset environment! because it was just reset ! 
            "timeout": timeout,
            "crash": crashed,
        }

    def select_goal_idx(self, position: np.ndarray = None, quaternion: np.ndarray = None, reached = False):
        
        if position is None or quaternion is None:
            assert self.step_by_one, "Must be stepping by one to update without specified position"

        if self.step_by_one:
            if self.goal_idx is None:
                self.goal_idx = 1
                self._goal_base_idx = 0
            else:
                self._goal_base_idx = self.goal_idx
                self.goal_idx += 1

            print("Goal IDX is now", self.goal_idx)
            if self.goal_idx >= self.goal_data["data/position"].shape[0]:
                print("Out of goals")

        else: # Find the distance to each point in the dataset, and sample randomly from the top 25
            if reached:
                self._goal_base_idx = self.goal_idx # use current goal as starting sampling location 
            else: 
                topk = 1
                goal_positions = self.goal_data["data/position"]
                goal_quaternions = self.goal_data["data/orientation"]

                try:
                    distances = pose_distance(
                        position, quaternion, goal_positions, goal_quaternions
                    )
                    best_idcs = np.argpartition(distances, topk)[:topk]
                    if topk == 1:
                        self._goal_base_idx = int(best_idcs[0])
                    else:
                        logits = -distances[best_idcs]
                        logits -= logits.max()
                        probs = np.exp(logits)
                        probs = np.nan_to_num(probs, 1e-6)
                        probs /= np.sum(probs)
                        probs = np.nan_to_num(probs, 1 / len(probs))
                        probs /= np.sum(probs)

                        chex.assert_shape(best_idcs, [topk])
                        self._goal_base_idx = int(np.random.choice(best_idcs, p=probs))
                except:
                    breakpoint()

            self.goal_idx = (
                self._goal_base_idx + int(np.random.exponential() * 5) + 5
            ) % len(self.goal_data["data/position"])
            assert isinstance(self.goal_idx, int), f"goal_idx is {self.goal_idx} ({type(self.goal_idx)})"

        self.last_reset_time = time.time()

    def reset(self, position = None, quaternion = None, reached = False):
        self.is_first = True

        if position is None or quaternion is None:
            assert self.step_by_one, "Must be stepping by one to reset without specified position"

        if position is not None and len(position) == 0:
            start_idx = np.random.randint(0, len(self.goal_data["data/position"]))
            position = self.goal_data["data/position"][start_idx]
            quaternion = self.goal_data["data/orientation"][start_idx]
            
        self.select_goal_idx(position, quaternion, reached)
        return position, quaternion

    def get_goal(self):
        if self.goal_idx is None:
            raise ValueError("Goal not selected yet!")

        # Return the goal image and state
        if self.goal_idx >= len(self.goal_data["data/position"]) and self.step_by_one:
            # Completed loop already 
            return None
        else: 
            position = self.goal_data["data/position"][self.goal_idx]
            sample_info = {
                "position": self.goal_data["data/position"][self._goal_base_idx],
                "orientation": self.goal_data["data/orientation"][self._goal_base_idx],
                "offset": np.float32(self.goal_idx - self._goal_base_idx),
            }
            
            return {
                "image": self.goal_data["data/image"][self.goal_idx],
                "position": position,
                "orientation": self.goal_data["data/orientation"][self.goal_idx],
                "sample_info": sample_info,
            }

    def reset_timer(self):
        self.last_reset_time = time.time()

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    task = TrainingTask(
        os.path.join(os.path.dirname(__file__), "../../data/goal_loop.pkl.npz")
    )

    robot_position = np.array([-4.0, 4.0, 0.0])
    robot_quaternion = np.array([0.0, 0.0, 0.0, 1.0])

    # Get yaw from quat
    def _yaw(quat):
        return np.arctan2(
            2.0 * (quat[3] * quat[2] + quat[0] * quat[1]),
            1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2),
        )

    robot_yaw = _yaw(robot_quaternion)

    for _ in range(10):
        task.select_goal_idx(robot_position, robot_quaternion)

        goal_position = task.get_goal()["position"]
        goal_quaternion = task.get_goal()["orientation"]
        goal_yaw = _yaw(goal_quaternion)

        fig, axs = plt.subplot_mosaic([
            ["A", "A"],
            ["B", "C"],
        ])
        axs["A"].axis("equal")
        axs["A"].plot(
            task.goal_data["data/position"][:, 0],
            task.goal_data["data/position"][:, 1],
            ".",
            label="path",
        )
        axs["A"].scatter(
            task.goal_data["data/position"][0, 0],
            task.goal_data["data/position"][0, 1],
            marker="o",
            c="yellow",
            s=100,
            label="begin",
        )
        axs["A"].scatter(
            task.goal_data["data/position"][-1, 0],
            task.goal_data["data/position"][-1, 1],
            marker="o",
            c="pink",
            s=100,
            label="end",
        )
        axs["A"].scatter(
            robot_position[0],
            robot_position[1],
            marker="o",
            c="r",
            s=100,
            zorder=10,
            label="robot",
        )
        axs["A"].quiver(
            robot_position[0],
            robot_position[1],
            np.cos(robot_yaw),
            np.sin(robot_yaw),
            color="r",
        )
        axs["A"].scatter(
            task.goal_data["data/position"][task._goal_base_idx, 0],
            task.goal_data["data/position"][task._goal_base_idx, 1],
            marker="o",
            c="g",
            s=100,
            zorder=0,
            label="goal base",
        )
        axs["A"].scatter(
            goal_position[0],
            goal_position[1],
            marker="x",
            c="g",
            s=100,
            zorder=10,
            label="goal",
        )
        axs["A"].quiver(
            goal_position[0],
            goal_position[1],
            np.cos(goal_yaw),
            np.sin(goal_yaw),
            color="g",
        )
        axs["A"].legend()
        axs["B"].imshow(task.goal_data["data/image"][task._goal_base_idx])
        axs["C"].imshow(task.get_goal()["image"])
        plt.show()