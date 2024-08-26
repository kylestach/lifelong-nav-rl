from abc import ABC
import logging
from typing import List, Tuple, TypeVar, Union
from asyncio import Future
from typing import Optional, Type, Dict

import numpy as np

from rclpy.node import Node
from rclpy.time import Time as RosTime
from rclpy.action import ActionClient as RosActionClient
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
import geometry_msgs.msg as gm
from nav2_msgs.action import NavigateToPose
from tf2_ros import TransformBroadcaster, TransformException
from tf2_ros.buffer import Buffer as TransformBuffer

TwistType = Union[np.ndarray, gm.Twist, gm.TwistStamped]

def _unstamp(
    msg: Optional[TwistType], current_time: RosTime
) -> Tuple[np.ndarray, RosTime]:
    if isinstance(msg, gm.TwistStamped):
        return np.array([msg.twist.linear.x, msg.twist.angular.z]), msg.header.stamp
    elif isinstance(msg, gm.Twist):
        return np.array([msg.linear.x, msg.angular.z]), current_time
    elif isinstance(msg, np.ndarray):
        return msg, current_time
    elif msg is None:
        return np.zeros((2,)), current_time
    else:
        raise TypeError(f"Unknown type {type(msg)}")

class BaseState(ABC):
    start_time: RosTime
    last_updated_time: RosTime
    twist: np.ndarray

    def __init__(self, start_time: RosTime, twist: Optional[TwistType] = None):
        self.twist, self.start_time = _unstamp(twist, start_time)
        self.last_updated_time = self.start_time

    @property
    def priority(self) -> int: ...

    @property
    def timeout(self) -> Optional[float]: ...

    def update(self, *, current_time: RosTime, twist: TwistType):
        self.twist, self.last_updated_time = _unstamp(twist, current_time)

    def expired(self, current_time: RosTime):
        if self.timeout is None:
            return False
        else:
            return (current_time - self.last_updated_time).nanoseconds > self.timeout * 1e9

    def tick(self, current_time: RosTime, current_obs: Dict):
        if self.expired(current_time):
            return IdleState(current_time)
        else:
            return self

    def cancel(self):
        pass


class IdleState(BaseState):
    @property
    def priority(self):
        return 0

    @property
    def timeout(self):
        return None


class EstopState(BaseState):
    @property
    def priority(self):
        return 100

    @property
    def timeout(self):
        return None


class TeleopState(BaseState):
    @property
    def priority(self):
        return 75

    @property
    def timeout(self):
        return 0.25


class TwistTaskState(BaseState):
    @property
    def priority(self):
        return 25

    @property
    def timeout(self):
        return 1.2


T = TypeVar("T", bound=BaseState)


class StateMachine:
    current_state: BaseState

    def __init__(self, node: Node):
        self.clock = node.get_clock()
        self.current_state = IdleState(self.clock.now())
    
    def _change_state(self, new_state: BaseState):
        print(f"STATE CHANGE {self.current_state} -> {new_state}")
        if self.current_state is not None:
            self.current_state.cancel()
        self.current_state = new_state

    def accept_state(self, new_state: BaseState):
        # Make sure the new and previous states aren't the same type
        if isinstance(self.current_state, type(new_state)):
            logging.warning(
                f"Attempted to set state to {type(new_state)} but it is already in that state."
            )
            return False

        should_accept = (
            self.current_state.expired(new_state.start_time)
            or new_state.priority > self.current_state.priority
        )

        if should_accept:
            self._change_state(new_state)

        return should_accept

    def current_state_matches(self, state_type: Type[T]) -> Optional[T]:
        if isinstance(self.current_state, state_type):
            return self.current_state
        return None

    def try_update(self, state_type: Type[T], **kwargs) -> bool:
        if isinstance(self.current_state, state_type):
            self.current_state.update(current_time=self.clock.now(), **kwargs)
            return True
        else:
            return False

    def tick(self, current_obs: Dict):
        now = self.clock.now()
        new_state = self.current_state.tick(now, current_obs)
        if self.current_state != new_state:
            self._change_state(new_state)

        if self.current_state is None or self.current_state.expired(now):
            self._change_state(IdleState(self.clock.now()))
