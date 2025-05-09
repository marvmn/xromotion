import copy
import rospy
import numpy as np
import tf.transformations as tf
from sensor_msgs.msg import JointState
from moveit_commander.robot import RobotCommander
from moveit_msgs.msg import PositionIKRequest, Constraints, JointConstraint
from moveit_msgs.srv import GetPositionIKResponse, GetPositionFKRequest, GetPositionFK, GetPositionIK

# Stack of Tasks imports
from stack_of_tasks.tasks.Eq_Tasks import *
from stack_of_tasks.config import Configuration
from stack_of_tasks.controller import Controller
from stack_of_tasks.ref_frame import Origin, Offset
from stack_of_tasks.solver.OSQPSolver import OSQPSolver
from stack_of_tasks.ref_frame.frames import RobotRefFrame
from stack_of_tasks.tasks.base import TaskSoftnessType, RelativeType
from stack_of_tasks.robot_model.actuators import DummyActuator

class TrajectoryPlanner:

    def __init__(self, times, positions):
        """
        Initialize trajectory planner.
        times -- Array with timestamps for the keyframes
        positions -- Array with joint positions for the keyframes
        """
        # set true, unmodified times
        self.true_times = times

        # set array with timestamps for the keyframes
        self.times = copy.deepcopy(times)

        # set true, unmodified positions
        self.true_positions = positions

        # set positions that will be modified through expression transforms
        self.positions = copy.deepcopy(positions)

        # indices of original positions in case the trajectory was filled up
        # with interpolated values
        self.original_indices = range(len(times))

        # When the last point of the trajectory is reached, done becomes True
        self.done = False
    
    def scale_global_speed(self, scalar):
        """
        Scales the speed of the trajectory globally, meaning that every
        keyframe's time gets scaled by the same scalar.
        When scalar > 1 the movement is slower, when 0 < scalar < 1 it
        gets faster.
        """
        for i in range(len(self.times)):
            self.times[i] = self.times[i] * scalar
    
    def add_jitter(self, amount=0.05):
        """
        Applies a certain amount of randomness to a motion to make it seem
        less confident. The randomness is scaled down at the beginning and
        the end to avoid conflicts with the functional objective.
        """

        # first, generate random summands for every position
        summands = np.random.normal(0.0, amount, self.positions.shape)

        # make the effect fade in and out at the beginning and end of the motion
        # to ensure that the start and end point of the motion stay the same
        parabola = -0.05 * self.times[-1] * self.times * (self.times - self.times[-1])
        cut = np.min([np.ones(parabola.shape), parabola], axis=0)
        
        # apply to positions
        self.positions += (summands.T * cut).T

        # finally, scale down global scale a little
        self.scale_global_speed(1.0 + max(amount / 2, 0.3))

    def add_gaze(self, point, link, move_group, axis=[0,0,1], movable_joints=None, from_index=0, to_index=-1):
        """
        Makes the specified link point at the given point throughout the trajectory
        
        Parameters:
        - point: Point to look at
        - link: Name of the link that should look at the point. If None is given, use the end effector.
        - move_group: Name of the move_group to use for the IK computation
        - axis: The axis that should be pointed to the point
        - from_index: Gaze will be applied on keyframes from this index onwards.
        - to_index: If not -1, Gaze will be applied from from_index to this index.
        """
        
        # prepare moveit robot interface
        robot = RobotCommander()
        end_index = len(self.original_indices) - 1 if to_index == -1 else to_index

        # go through every keyframe
        for i in range(from_index, len(self.original_indices) + 1):

            # apply pointing pose
            joint_state = self._get_pointing_joint_state(move_group, robot, self.original_indices[i], link, point, axis, 
                                                             movable_joints)

            self.positions[self.original_indices[i]] = joint_state

    def get_position_at(self, timestamp, original=False):
        """
        Interpolates and returns the point at the specified timestamp (in seconds).
        If original is True, use the original trajectory instead of the modified one.
        """

        # initialize timestamp and position variables

        # timestamps to interpolate between
        ts0 = 0.0
        ts1 = 0.0

        # positions to interpolate between
        pos0 = np.zeros((3))
        pos1 = np.zeros((3))

        # choose positions and times arrays based on parameter
        positions = self.true_positions if original else self.positions
        times = self.true_times if original else self.times

        # now interpolate:
        # if the first timestamp is > 0.0, interpolate from 0!
        if timestamp < times[0]:
            ts1 = times[0]
            pos0 = np.zeros((len(positions[0])))
            pos1 = positions[0]
        
        # if the last timestamp is over, remain in end position and set done = True
        elif timestamp > times[len(times) - 1]:
            self.done = True
            return positions[len(positions) - 1]
        
        # otherwise find the two timestamps that the current time is between
        else:
            for i in range (len(times)):
                if times[i] >= timestamp:
                    ts0 = times[i - 1]
                    ts1 = times[i]
                    pos0 = positions[i - 1]
                    pos1 = positions[i]
                    break
        
        # interpolate time
        diff_points = ts1 - ts0
        prog = timestamp - ts0
        scalar = prog / diff_points

        # interpolate position
        return pos0 + (pos1 - pos0) * scalar

    def apply_bezier_at(self, index0, index1, cp0, cp1):
        """
        Scales the velocity of the motion between two keyframes according to
        a Bézier curve. The curve assumes the position at index0 to be (0,0)
        and the position at index1 to be (1,1) to unify the choice of the
        control points cp0 and cp1 across different joint states.
        """

        # check if the two points are the same
        if (self.times[index0] == self.times[index1]):
            return

        # first, calculate the bezier curve with n points
        n = 20

        bezier_fn = lambda t: (1 - t)**3 * np.array([0,0]) + 3*(1 - t)**2 * t * cp0 + 3 * (1 - t) * t**2 * cp1 + t**3 * np.array([1,1])
        curve = bezier_fn(np.linspace([0,0], [1,1], n)).T
        #print("CURVE:\n", curve)

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

    def fill_up(self, frequency):
        """
        Fills up the times and positions arrays with interpolated values so that the
        resulting times are filled with a frequency of >frequency per second.
        """
        new_positions = copy.deepcopy(self.positions)
        new_times = copy.deepcopy(self.times)

        # go through every intervall and check if it satisfies the rate
        added = 0
        self.original_indices = []

        for i in range(len(self.times) - 1):

            self.original_indices.append(i + added)

            intervall = self.times[i + 1] - self.times[i]

            # if the intervall between these two timestamps it too big, it doesn't satisfy the rate
            # that means new keyframes need to be inserted.
            if intervall > 1/frequency:
                
                j_idx = 0
                added_now = 0
                while (j_idx + 1) * 1/frequency < intervall:
                    new_times = np.insert(new_times, i + j_idx + added + 1, self.times[i] + 1/frequency * (j_idx + 1))
                    new_positions = np.insert(new_positions, i + j_idx + added + 1, self.get_position_at(new_times[i+j_idx + added + 1]), axis=0)
                    j_idx += 1
                    added_now += 1
                added += added_now
        
        # finally save new times and positions
        self.positions = new_positions
        self.times = new_times

        # return the indices of the original keyframes
        self.original_indices.append(added)
        return self.original_indices

    def _get_pointing_joint_state(self, move_group: str, robot: RobotCommander, time, link, point, 
                                  axis=[0,0,1], movable_joints=None):
        """
        Uses MoveIt's inverse kinematics to generate a joint state that lets the given link point
        the axis at the given point.

        Parameters:
        - move_group: Name of the move group
        - robot: RobotCommander with the commanded robot
        - time: Index of the base position
        - link: Name of the pointer link
        - point: 3D Coordinates of the point to point at
        - axis: Direction vector of the axis to point at the point
        - up_vector: When pointing at the point, this axis will point as far upwards as possible.
        - movable_joints: Joints that can be moved from the original state in order to reach pointing state
        """

        # build current robot state
        config = Configuration(OSQPSolver, DummyActuator)
        controller = Controller(config=config)
        controller.robot_state.incoming_joint_values = np.zeros(len(controller.robot_state.joint_values))

        skipped = 0
        for i in range(len(robot.get_group(move_group).get_active_joints())):
            if controller.robot_state.robot_model.active_joints[i].name in robot.get_group(move_group).get_active_joints():
                    controller.robot_state.incoming_joint_values[i - skipped] = self.positions[time,i]
            else:
                skipped += 1

        # build joint constraint as rho weight vector TODO adapt to SoT
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
        

    def _fake_pointer_control_loop(self, controller, point: np.ndarray, link_name: str, rho: np.ndarray, axis=[0,0,1]):
        """
        Use Stack of Tasks framework and simulate a dummy robot state to iteratively compute a pointing pose that
        respects the current joint values.

        Parameters:
        - joint_states: Array of preferred joint positions
        - point: Coordinates of point to look at
        - link_name: Link that should point towards the point
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
        rate = rospy.Rate(40)
        warmstart_dq = None
        controller.solver.tasks_changed()

        convergence = 0
        last_dq = 100

        while not rospy.is_shutdown_requested() and convergence < 10:
            warmstart_dq = controller.control_step(warmstart_dq)
            dq = np.linalg.norm(warmstart_dq)
            if abs(dq - last_dq) <= 0.005:
                convergence += 1
            else:
                convergence = 0
            last_dq = dq
            rate.sleep()
        
        # converged!

        return controller.robot_state.joint_values

        
