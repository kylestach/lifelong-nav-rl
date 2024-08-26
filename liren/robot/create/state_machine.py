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
import irobot_create_msgs.action as create_action


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

class RosActionState(BaseState):
    """
    Base class for states that start and end based on the result of a ROS action.

    The ROS action should be configured to send velocities to a particular topic.
    """

    send_goal_future: Future
    done: bool

    def __init__(
        self,
        start_time: RosTime,
        clock,
        action_client: RosActionClient,
        goal,
        twist: np.ndarray = np.zeros((2,)),
    ):
        super().__init__(start_time, twist=twist)

        self.clock = clock
        self.done = False

        # Kick off a ROS action asynchronously
        print("Goal type", type(goal))
        self.send_goal_future = action_client.send_goal_async(
            goal, feedback_callback=self.feedback_callback
        )
        self.send_goal_future.add_done_callback(self.send_goal_callback)

    def cancel(self):
        self.goal_handle.cancel_goal_async()

    def send_goal_callback(self, future: Future):
        self.goal_handle = future.result()

        if not self.goal_handle.accepted:
            self.done = True
            print("GOAL REJECTED:", self.goal_handle)
        else:
            print("GOAL ACCEPTED:", self.goal_handle)

        goal_result_future: Future = self.goal_handle.get_result_async()
        goal_result_future.add_done_callback(self.goal_result_callback)

    def feedback_callback(self, feedback):
        # print("FEEDBACK:", feedback)
        if not self.done:
            self.last_updated_time = self.clock.now()

    def goal_result_callback(self, future: Future):
        print("COMPLETED WITH", future.result)
        self.done = True

    def expired(self, current_time: RosTime):
        if self.done:
            print("EXPIRE DONE")
        if super().expired(current_time):
            print("EXPIRE TIME")
        return self.done or super().expired(current_time)

class RosParamState(BaseState):

    send_param_future: Future
    done: bool

    def __init__(
        self,
        start_time: RosTime,
        clock,
        param_client,
        param_name,
        param_value,
        param_type,
    ):
        super().__init__(start_time)

        self.clock = clock
        self.done = False

        param = Parameter()
        param.name = param_name
        param.value = ParameterValue()
        
        # is instance wasn't behaving properly for bools, because they can also be ints! 
        if param_type is int: 
            param.value.type = ParameterType.PARAMETER_INTEGER
            param.value.integer_value = param_value
        elif param_type is float:
            param.value.type = ParameterType.PARAMETER_DOUBLE
            param.value.double_value = param_value
        elif param_type is str:
            param.value.type = ParameterType.PARAMETER_STRING
            param.value.string_value = param_value
        elif param_type is bool:
            param.value.type = ParameterType.PARAMETER_BOOL
            param.value.bool_value = param_value
        else:
            raise ValueError(f"Unsupported parameter type {param_type}")
        
        request = SetParameters.Request()
        request.parameters = [param]

        self.send_param_future = param_client.call_async(request)
        self.send_param_future.add_done_callback(self.send_param_callback)

    @property
    def priority(self):
        return 50

    @property
    def timeout(self):
        return 2.5

    def send_param_callback(self, future: Future):
        self.param_handle = future.result()

        all_successful = all(result.successful for result in self.param_handle.results)
        if all_successful:
            print("successfully set")
            self.done = True
        else:
            print("failed to set because", self.param_handle.results[0].reason)


        self.done = True

    def expired(self, current_time: RosTime):
        return self.done or super().expired(current_time)

class IRobotActionState(RosActionState):
    """
    State for handling IRobot actions.
    """

    @property
    def priority(self):
        return 75

    @property
    def timeout(self):
        return 10


class Nav2ActionState(RosActionState):
    """
    State for handling IRobot actions.
    """

    @property
    def priority(self):
        return 25

    @property
    def timeout(self):
        return 0.25

# Just Back UP!
class DoResetState(BaseState):
    def __init__(
        self,
        start_time: RosTime,
        twists: List[TwistType],
        time_per_twist: List[float],
    ):
        super().__init__(start_time, twists[0])
        self.twists = twists
        self.time_per_twist = time_per_twist
        # self.done = True

    @property
    def priority(self):
        return 30

    def tick(self, current_time: RosTime, current_obs: Dict):
        time_into_reset = (current_time - self.start_time).nanoseconds / 1e9
        for i, (twist, time) in enumerate(zip(self.twists, self.time_per_twist)):
            time_into_reset -= time
            if time_into_reset < 0:
                self.update(current_time=current_time, twist=twist)
                break

        return super().tick(current_time, current_obs)

    @property
    def timeout(self):
        return sum(self.time_per_twist)

# Navigate to given goal pose 
class IRobotNavState(Nav2ActionState):
    def __init__(
        self,
        start_time: RosTime,
        action_client: RosActionClient,
        goal_pose, 
        clock,
    ):

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = "map"

        goal_msg.pose.pose.position.x = goal_pose[0]
        goal_msg.pose.pose.position.y = goal_pose[1]
        goal_msg.pose.pose.position.z = goal_pose[2]

        goal_msg.pose.pose.orientation.x = goal_pose[3]
        goal_msg.pose.pose.orientation.x = goal_pose[4]
        goal_msg.pose.pose.orientation.x = goal_pose[5]
        goal_msg.pose.pose.orientation.x = goal_pose[6]
        
        super().__init__(start_time, clock, action_client, goal_msg)

    @property
    def priority(self):
        return 30 # Higher than IRobot, this lets us preempt

    @property
    def timeout(self):
        return 10


class IRobotDockState(IRobotActionState):
    def __init__(self, start_time: RosTime, action_client: RosActionClient, clock):
        # Issue a dock command
        super().__init__(start_time, clock, action_client, create_action.Dock.Goal())

    @property
    def priority(self):
        return 50
    
    @property
    def timeout(self):
        return 10


class IRobotUndockState(IRobotActionState):
    def __init__(self, start_time: RosTime, action_client: RosActionClient, clock):
        # Issue a dock command
        super().__init__(start_time, clock, action_client, create_action.Undock.Goal())

    @property
    def priority(self):
        return 50
    
    @property
    def timeout(self):
        return 10


class Nav2DockState(RosActionState):
    def tick(self, current_time: RosTime, current_obs: Dict):
        if self.done:
            return IRobotDockState()
        elif self.expired(current_time):
            return IdleState(current_time)
        else:
            return self


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
