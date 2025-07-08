""" Messages for effects """

import time
import rospy
import numpy as np
from copy import deepcopy
from moveit_commander.robot import RobotCommander
from typing import Iterable, List, Optional, Union
from expressive_motion_generation.trajectory import Trajectory
from expressive_motion_generation.animation import Animation

# Stack of Tasks imports
from stack_of_tasks.tasks.Eq_Tasks import *
from stack_of_tasks.config import Configuration
from stack_of_tasks.controller import Controller
from stack_of_tasks.ref_frame import Origin, Offset
from stack_of_tasks.solver.OSQPSolver import OSQPSolver
from stack_of_tasks.ref_frame.frames import RobotRefFrame
from stack_of_tasks.tasks.base import TaskSoftnessType, RelativeType
from stack_of_tasks.robot_model.actuators import DummyActuator

class Effect:
    """ Base class for all effects """

    def __init__(self, start_index=0, stop_index=-1):
        """
        Parameters:
        - start_index: Keyframe index where the effect should begin.
        - stop_index: Keyframe index where the effect should end. If this is -1, the last keyframe will be used. 
        """
        self.start_index: int = start_index
        self.stop_index: int = stop_index
    
    def get_indices(self, target: Union[Trajectory, Animation]):
        """
        Compute the start and stop indices based on the trajectory planner
        that they should be applied on. This makes sure that negative
        indices get handled correctly.

        Parameters:
        - target: Trajectory Planner or animation that the effect should be applied on

        Returns:
        - start_index: New start index
        - stop_index: New stop index
        """
        if not target.original_indices:
            target.original_indices = range(len(target.times))
            
        print(target.original_indices, self.start_index, self.stop_index)
        
        start_index = target.original_indices[self.start_index]
        if start_index < 0:
            start_index += len(target.times)
        stop_index = target.original_indices[self.stop_index]
        if stop_index < 0:
            stop_index += len(target.times)

        return start_index, stop_index
    
    def apply(self, trajectory_planner: Trajectory, animation: Optional[Animation] = None):
        """ Apply this effect to a given trajectory planner.
         If the task type is animation, providing the animation can add
          metadata for application. """
        pass

class JitterEffect(Effect):
    """ Jitter effect: Adds shakiness to the motion. Amount indicates the maximum modification value
        that is added or subtracted from the joint values in radians. The randomness is scaled down at 
        the beginning and the end to avoid conflicts with the functional objective."""

    def __init__(self, amount: float = 0.01, fill: int = 0,
                 start_index:int = 0, stop_index:int = -1):
        """
        Parameters:
        - amount: Maximum radian value that is added or subtracted.
        - fill: The rate that the trajectory planner is filled up to to make the effect applicable.
        - start_index: Keyframe index where the effect should begin.
        - stop_index: Keyframe index where the effect should end. If this is -1, the last keyframe will be used. 
        """
        super().__init__(start_index, stop_index)
        self.amount = amount
        self.fill = fill
        self.start_index = start_index
        self.stop_index = stop_index
    
    def apply(self, trajectory_planner: Trajectory, animation: Optional[Animation] = None):
        
        # fill up if needed
        if self.fill > 0:
            trajectory_planner.fill_up(self.fill)

        start, stop = self.get_indices(trajectory_planner)

        if animation is not None and animation.original_indices:
            start, stop = self.get_indices(animation)
            # start = animation.original_indices[start]
            # stop = animation.original_indices[stop]

        # first, generate random summands for every position
        summands = np.random.normal(0.0, self.amount, trajectory_planner.positions[start:stop+1].shape)

        # make the effect fade in and out at the beginning and end of the motion
        # to ensure that the start and end point of the motion stay the same
        parabola = -0.05 * (trajectory_planner.times[stop] - trajectory_planner.times[start]) \
            * trajectory_planner.times[start:stop+1] \
            * (trajectory_planner.times[start:stop+1] - (trajectory_planner.times[stop] - trajectory_planner.times[start]))
        cut = np.min([np.ones(parabola.shape), parabola], axis=0)
        
        # apply to positions
        trajectory_planner.positions[start:stop+1] += (summands.T * cut).T

        # finally, scale down global scale a little
        trajectory_planner.scale_global_speed(1.0 + min(self.amount / 2, 0.3))
        return summands, parabola, cut


class GazeEffect(Effect):
    """ Gaze effect: Lets a link orient a given axis through the link at a specific point. This can be used
    to have the robot point at something or 'look' at something. """

    def __init__(self, point, link: str, move_group: str, 
                 axis=[0,0,1], movable=[], start_index:int = 0, stop_index:int = -1):
        """
        Parameters:
        - point: Coordinates of point to look at
        - link: Name of link to point towards point
        - move_group: MoveIt move group that uses the current joint group
        - axis: Direction vector of axis to point towards point
        - movable: List of joint names to use for the action. If none are specified, use all joints.
        - start_index: Keyframe index where the effect should begin.
        - stop_index: Keyframe index where the effect should end. If this is -1, the last keyframe will be used. 
        """
        super().__init__(start_index, stop_index)
        self.point = np.array(point)
        self.link = link
        self.move_group = move_group
        self.axis = np.array(axis)
        self.movable = movable
    
    def apply(self, trajectory_planner: Trajectory, animation: Optional[Animation] = None):

        # first get indices
        start_index, stop_index = self.get_indices(trajectory_planner)

        # if animation, only apply on keyframes and then fill up again
        if not animation is None:

            start_index, stop_index = self.get_indices(animation)

            new_trajectory_planner = Trajectory(animation.times, animation.positions, animation.joint_names)
            new_trajectory_planner.add_gaze(self.point, self.link, self.move_group, 
                                            self.axis, self.movable, start_index, 
                                            stop_index)

            # apply to old animation
            animation.positions = new_trajectory_planner.positions
            animation._reload_trajectory()

            # apply to trajectory planner
            trajectory_planner.positions = animation.trajectory_planner.positions
            trajectory_planner.times = animation.trajectory_planner.times
                
        # if not, just apply
        else:
            trajectory_planner.add_gaze(self.point, self.link, self.move_group, 
                                        self.axis, self.movable, start_index, 
                                        stop_index)
    
    def _apply(self, point, link, move_group, axis=[0,0,1], movable_joints=None, from_index=0, to_index=-1):
        """
        Makes the specified link point at the given point throughout the trajectory
        
        Parameters:
        - point: Point to look at
        - link: Name of the link that should look at the point. If None is given, use the end effector.
        - move_group: Name of the move_group to use for the IK computation
        - axis: The axis that should be pointed to the point
        - movable_joints: Names of the joints that should be moved to achieve pose
        - from_index: Gaze will be applied on keyframes from this index onwards.
        - to_index: If not -1, Gaze will be applied from from_index to this index.
        """
        
        # prepare moveit robot interface
        robot = RobotCommander()
        end_index = len(self.original_indices) - 1 if to_index == -1 else to_index

        full_time = 0

        # go through every keyframe
        for i in range(from_index, end_index + 1):
            t = time.time()

            # apply pointing pose
            joint_state = self._get_pointing_joint_state(move_group, robot, self.original_indices[i], link, point, axis, 
                                                             movable_joints)
            
            print(f"[Gaze Effect] Converged for {i}/{end_index} in {time.time() - t}", end='\r')
            full_time += time.time() - t

            self.positions[self.original_indices[i]] = joint_state
        
        print(f"[Gaze Effect] Finished gaze calculation in {round(full_time, 3)}s")
    
    def _get_pointing_joint_state(self, move_group: str, robot: RobotCommander, time, link, point, 
                                  axis=[0,0,1], movable_joints=None):
        """
        Uses Stack of Tasks framework to generate a joint state that lets the given link point
        the axis at the given point.

        Parameters:
        - move_group: Name of the move group
        - robot: RobotCommander with the commanded robot
        - time: Index of the base position
        - link: Name of the pointer link
        - point: 3D Coordinates of the point to point at
        - axis: Direction vector of the axis to point at the point
        - movable_joints: Joints that can be moved from the original state in order to reach pointing state
        """

        # build current robot state
        config = Configuration(OSQPSolver, DummyActuator)
        controller = Controller(config=config)
        controller.robot_state.incoming_joint_values = np.zeros(len(controller.robot_state.joint_values))

        skipped = 0
        for i in range(len(robot.get_group(move_group).get_active_joints())):
            if controller.robot_state.robot_model.active_joints[i].name in robot.get_group(move_group).get_active_joints():
                    controller.robot_state.incoming_joint_values[i - skipped] = self.positions[time][i]
            else:
                skipped += 1

        # build joint constraint as rho weight vector 
        rho = np.ones(controller.robot_state.robot_model.N)
        if movable_joints is not None:
            for i in range(len(rho)):
                if i in movable_joints:
                    rho[i] = 1
                else:
                    rho[i] = 500
        
        # call stack of tasks and compute joint positions
        joint_states = self._fake_pointer_control_loop(controller, point, link, rho, axis)

        # build list with old joints
        final_list = np.zeros(len(robot.get_group(move_group).get_active_joints()))
        for i in range(len(robot.get_group(move_group).get_active_joints())):
            for j in range(len(joint_states)):
                if controller.robot_state.robot_model.active_joints[j].name == robot.get_group(move_group).get_active_joints()[i]:
                    final_list[i] = joint_states[j]

        return final_list
        

    def _fake_pointer_control_loop(self, controller: Controller, point: np.ndarray, link_name: str, rho: np.ndarray, axis=[0,0,1]):
        """
        Use Stack of Tasks framework and simulate a dummy robot state to iteratively compute a pointing pose that
        respects the current joint values.

        Parameters:
        - controller: Stack of Tasks Controller to use
        - point: Coordinates of point to look at
        - link_name: Link that should point towards the point
        - rho: Weight vector that defines which joints should be used for the task
        - axis: Axis in the link that should be pointed at the point.
        """

        # prepare controller and tasks
        # define point transformation matrix
        point_frame = np.array([[1, 0, 0, point[0]],
                                [0, 1, 0, point[1]],
                                [0, 0, 1, point[2]],
                                [0, 0, 0, 1  ]])

        # define first level task at pointing at the point
        with controller.task_hierarchy.new_level() as level:
            level.append(RayTask(refA=RobotRefFrame(controller.robot_state, link_name=link_name), 
                                            refB=Offset(Origin(), point_frame), 
                                            softness_type=TaskSoftnessType.linear,
                                            refA_axis=np.array(axis),
                                            relType=RelativeType.RELATIVE, weight=1.0,
                                            mode="axis + dp",
                                            rho=rho))
        
        # now start a custom control loop
        warmstart_dq = None
        controller.solver.tasks_changed()

        convergence = 0
        last_dq = 100

        while not rospy.is_shutdown_requested() and convergence < 10:
            warmstart_dq = controller.control_step(warmstart_dq)
            dq = np.linalg.norm(warmstart_dq)
            if abs(dq - last_dq) <= 0.000001:
                convergence += 1
            else:
                convergence = 0
            last_dq = dq
        
        # converged!

        return controller.robot_state.joint_values

class ExtentEffect(Effect):
    """ Extent effect: Modifies the trajectory to follow a wider or narrower path. """
    
    def __init__(self, amount: float, mode_configuration: Iterable[float], 
                 upper_joint_limits: Iterable[float], lower_joint_limits: Iterable[float]):
        """
        Initialize and calculate transform vector for extent effect.
        
        Parameters:
        - amount: Effect scalar that determines the size and direction of the modifications. Should be in the range [-1, 1]. \
            Negative values will make the trajectory narrower, positive values will make it wider.
        - mode_configuration: List of mode for every joint. Modes can be: <br>\
            - <b>'p'</b> <i>(positive extent)</i>: Adding positive values to this joint will make the robot pose wider <br>\
            - <b>'n'</b> <i>(negative extent)</i>: Adding negative values to this joint will make the robot pose narrower <br>\
            - <b>'g'</b> <i>(general extent)</i>: Multiplying a value > 1 with this joint's value will make the robot pose wider, < 1 narrower <br>\
            - <b>'i'</b> <i>(independent)</i>: This joint's value has no influence on the wideness of the robot pose.
        - upper_joint_limits: Upper joint limits, where upper_joint_limits[i] contains the limit for joint i.
        - lower_joint_limits: Upper joint limits, where lower_joint_limits[i] contains the limit for joint i.
        """
        self.configuration = mode_configuration
        self.amount = amount
        self.limit_upper = upper_joint_limits
        self.limit_lower = lower_joint_limits
    
    def _gcd(self, a: float, b: float, limit: float = 0.0001) :
        """
        Compute greatest common divisor for float values.
        https://www.geeksforgeeks.org/program-find-gcd-floating-point-numbers/

        Parameters:
        - a: First number
        - b: Second number
        - limit: Search precision
        """

        # b should always be the smaller value, so switch a and b if this is
        # not the case
        if (a < b) :
            return self._gcd(b, a, limit)
        
        # is the difference of the result < than the threshold?
        if (abs(b) < limit) :
            return a
        
        # otherwise, recursively search for smaller value
        return (self._gcd(b, a - np.floor(a / b) * b, limit))

    def _gcd_array(self, array: Iterable[float]):
        """ Calculates the greatest common divisor for an array of floats.
        
        Parameters:
        - array: Array to find the GCD for.
        
        Returns:
        - gcd: Greatest common divisor
        """
        gcd = array[0]
        for i in range(1, len(array)):
            gcd = self._gcd(array[i], gcd)
        return gcd

    def _get_regular_spaced_trajectory(self, trajectory_planner: Trajectory):
        """
        Compute the biggest frequency that can be used to describe the trajectory
        without loss of quality and return the same trajectory filled up to that
        frequency.

        Parameters:
        - trajectory_planner: Trajectory planner that holds the relevant trajectory

        Returns:
        - (times_n, positions_n) - new times and positions array. The times array will be regular spaced.
        """
        # 1. find greatest common divisor of times
        divisor = self._gcd_array(trajectory_planner.times) # np.gcd.reduce(trajectory_planner.times)

        # 2. build new positions
        new_times = np.arange(trajectory_planner.times[0], trajectory_planner.times[-1], divisor)
        new_positions = []
        for i in new_times:
            new_positions.append(trajectory_planner.get_position_at(i))
        
        # 3. return!
        return (np.array(new_times), np.array(new_positions))

        
    def apply(self, trajectory_planner: Trajectory, animation: Optional[Animation] = None):

        # get evenly spaced trajectory
        new_times, new_positions = self._get_regular_spaced_trajectory(trajectory_planner)
        old_positions = deepcopy(new_positions)
        
        # for each joint
        for i in range(len(trajectory_planner.positions[0])):

            # get fourier transform
            fourier = np.fft.fft(new_positions.T[i])
            frequencies = np.fft.fftfreq(len(new_times), new_times[1] - new_times[0])

            low_freqs = np.abs(frequencies) < np.mean(np.abs(frequencies))

            if self.configuration[i] == 'p':
                fourier[low_freqs] += (abs(fourier[low_freqs]) * self.amount)
            elif self.configuration[i] == 'n':
                fourier[low_freqs] -= (abs(fourier[low_freqs]) * self.amount)
            elif self.configuration[i] == 'm':
                fourier[low_freqs] *= 1 + self.amount
            
            # recompute positions
            new_positions.T[i] = np.fft.ifft(fourier)
        
        difference = new_positions - old_positions

        # make the effect fade in and out at the beginning and end of the motion
        # to ensure that the start and end point of the motion stay the same
        parabola = -0.05 * new_times[-1] * new_times * (new_times - new_times[-1])
        cut = np.min([np.ones(parabola.shape), parabola], axis=0)

        trajectory_planner.positions = (old_positions.T + cut * difference.T).T
        trajectory_planner.times = new_times

class BezierCurveEffect(Effect):
    
    def __init__(self, start_index, stop_index, control_point0, control_point1):
        """
        Scales the velocity of the motion between two keyframes according to
        a Bézier curve. The curve assumes the position at index0 to be (0,0)
        and the position at index1 to be (1,1) to unify the choice of the
        control points cp0 and cp1 across different joint states.

        Parameters:
        - start_index: Index of the first keyframe of the interval
        - stop_index: Index of the second keyframe of the interval
        - control_point0: The first control point in the interval
        - control_point1: The second control point in the interval
        """
        super().__init__(self, start_index, stop_index)
        self.cp0 = control_point0
        self.cp1 = control_point1
    
    def apply(self, trajectory_planner: Trajectory, animation: Optional[Animation] = None):
        
        index0, index1 = self.get_indices(trajectory_planner)

        if animation is not None and animation.original_indices:
            index0, index1 = self.get_indices(animation)
        
        # check if the two points are the same
        if (self.times[index0] == self.times[index1]):
            return

        # first, calculate the bezier curve with n points
        n = 20

        bezier_fn = lambda t: (1 - t)**3 * np.array([0,0]) + 3*(1 - t)**2 * t * self.cp0 + 3 * (1 - t) * t**2 * self.cp1 + t**3 * np.array([1,1])
        curve = bezier_fn(np.linspace([0,0], [1,1], n)).T

        # now interpolate exact position for every timestamp
        for idx in range(index1 - index0):
            
            # get x coordinate on the curve
            # for that normalize the times-intervall that this curve operates on
            x = (self.times[index0 + idx] - self.times[index0]) / (self.times[index1] - self.times[index0])
            time = 0.0
            if x == 0:
                continue
            for i in range(len(curve[0])):
                if x <= curve[0][i]:
                    diff = curve[0, i] - curve[0, i - 1]
                    prog = x - curve[0, i - 1]
                    scal = prog / diff
                    time = curve[1, i - 1] + (curve[1, i] - curve[1, i - 1]) * scal
                    break
            
            # finally apply calculated time back to times array
            self.times[index0 + idx] = time * (self.times[index1] - self.times[index0]) + self.times[index0]
        
