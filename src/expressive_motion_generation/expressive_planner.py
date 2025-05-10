import moveit_commander
import numpy as np
import rospy
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState, DisplayRobotState
import time
from expressive_motion_generation.trajectory_planner import TrajectoryPlanner
from expressive_motion_generation.animation_execution import Animation
from expressive_motion_generation.effects import *
from typing import List, Union, Optional


class TargetPlan:

    def __init__(self, target, move_group, target_type='pose', velocity_scaling=None, acceleration_scaling = None):
        """
        Saves a pose goal or joint goal and effects that should be applied.
        """
        self.target = target
        self.target_type = target_type
        self.move_group = move_group
        self.velocity_scaling = velocity_scaling
        self.acceleration_scaling = acceleration_scaling

class Task:
    """ Defines a task to be executed in a expressive motion plan """

    def __init__(self, target: Union[Animation, TargetPlan], trajectory_planner: Optional[TrajectoryPlanner]=None):
        """
        Target is either an Animation or a TargetPlan defining the planned motion.
        If the trajectory planner is already ready, it defines this task as baked.
        """
        self.target = target
        self.trajectory_planner = trajectory_planner
        self.effects = []
    
    def bake(self, trajectory_planner: Optional[TrajectoryPlanner] = None):
        """
        Set trajectory planner and apply effects.

        Parameters:
        - trajectory_planner: If this task does not contain an animation, trajectory \
            needs to be computed and given as argument.
        """
        # set trajectory planner
        if self.is_animation():
            self.trajectory_planner = self.target.trajectory_planner
        else:
            if trajectory_planner is None:
                raise ValueError("Trajectory planner must not be None if this is not an animation.")
            self.trajectory_planner = trajectory_planner
        
        # apply effects
        for effect in self.effects:
            self._apply_effect(effect)

    def add_effects(self, *effects: Effect):
        """ Add effects to this task """
        for effect in effects:

            # special case: If this is a gaze effect, put it in the beginning
            # this is because this effect reinitalizes the trajectory if this is an animation,
            # so it overwrites all effects applied before.
            if type(effect) == GazeEffect:
                self.effects.insert(0, effect)
            else:
                self.effects.append(effect)

        # if already baked, apply the new effects as well
        if self.is_baked():
            for effect in effects:
                self._apply_effect(effect)
    
    def _apply_effect(self, effect: Effect):
        """ Apply the given effect to the trajectory. May only be called
         if trajectory is already baked (trajectory planner exists). """
        animation = self.target if self.is_animation() else None
        effect.apply(self.trajectory_planner, animation)

    def is_baked(self):
        """ Is this task already baked/ready to be executed?"""
        return not self.trajectory_planner is None

    def is_animation(self):
        """ Is this task an animation? """
        return type(self.target) == Animation

    def get_last_joint_state(self):
        """ Returns last joint state for this task """
        if self.is_baked():
            return self.trajectory_planner.positions[-1]
        elif self.is_animation():
            return self.target.positions[-1]
        return None

    def __str__(self):
        target_type = f"Animation {self.target.name}" if self.is_animation() else f"Target ({self.target.target_type}) to {self.target.target}"
        baked = "Baked" if self.is_baked() else "Not Baked"
        return f'Task: {target_type} - {baked}'

class ExpressivePlanner:

    def __init__(self, robot: moveit_commander.RobotCommander, publish_topic='joint_command', fake_display=False):
        """
        Expressive planning framework for functional and expressive animated robot motion.
        Requires robot commander for the robot that should be controlled.
        publish_topic specifies the ROS topic the joint states should be published to.
        If fake_display is True, instead of joint states DisplayRobotState messages get published.
        """

        self.plan: List[Task] = []

        self.robot = robot
        self.frame_id = robot.get_planning_frame()
        self.joint_names = robot.get_active_joint_names()
        self.fake_display = fake_display
        self._last_trajectory_planner = None

        # initialize ros and moveit
        rospy.init_node("expressive_planner", anonymous=True)

        # initialize publisher
        if fake_display:
            self.publisher = rospy.Publisher(publish_topic, DisplayRobotState, queue_size=10)
        else:
            self.publisher = rospy.Publisher(publish_topic, JointState, queue_size=10)
    
    def new_plan(self):
        """
        Initialize a new plan for the given robot.
        """
        self.plan = []
        self._last_trajectory_planner = None
    
    def plan_animation(self, path: str) -> bool:
        """
        Load an animation from a file and append it to plan.

        Parameters:
        - path: Filepath relative to the current working directory
        
        Returns:
        - True if successful, otherwise False.
        """
        # try loading animation and check if file exists
        try:
            animation = Animation(path)
        except FileNotFoundError:
            return False
        
        task = Task(animation)

        # check if animation was successfully loaded
        self.plan.append(task)
    
    def plan_target(self, target, move_group=None, velocity_scaling = None, acceleration_scaling = None, target_type='pose'):
        """
        Add planning a motion to a target pose or joint goal. If move_group is None,
        check if there exists another element in the plan. If so, choose the last used
        move_group. Otherwise choose the first move_group that is returned by the RobotCommander.
        """
        if move_group is None:
            
            # check if there exists an element in the plan so far
            if self.plan:

                # choose the move group from the most recent element
                # since both Animation and TargetPose have the move_group attribute,
                # no distinction needs to be made.
                move_group = self.plan[-1].target.move_group

            else:

                # otherwise get the available move groups and choose the first one
                move_group = self.robot.get_group_names()[0]

        # now that the move_group is set, add target to plan!
        target_plan = TargetPlan(target, move_group, target_type=target_type, 
                                 velocity_scaling=velocity_scaling, acceleration_scaling=acceleration_scaling)
        task = Task(target_plan)
        self.plan.append(task)

    def add_task(self, task: Task, index: int = -1):
        """
        Add a custom task to this planner.

        Parameters:
        - task: Task to be added
        - index: Index where it should be inserted in the plan. If -1, then it will be appended.
        """
        if index == -1:
            self.plan.append(task)
        elif index >= 0 and index < len(self.plan):
            self.plan.insert(index, task)
        else:
            raise IndexError(f"Task could not be inserted at {index}. The plan has length {len(self.plan)}.")

    def at(self, index: int) -> Optional[Task]:
        """
        Returns the task at the given index.

        Parameters:
        - index: Index of the task
        """
        # if index is not valid, return None
        if len(self.plan) <= index:
            return None
        return self.plan[index]
                
    def bake(self):
        """
        Pre-compute all trajectories and prepare trajectory planners for execution.
        """
        last_position = None

        # go through each element
        for task in self.plan:
            
            # if element is animation instance, the trajectory planner is ready to use
            if task.is_animation():
                # TODO transition from last position?
                task.bake()
            
            # otherwise, the motion needs to be computed
            else:
                # if this is the first element, the current robot pose suffices as start pose.
                # otherwise take the last pose of the previous element as starting pose
                trajectory_planner = self.plan_trajectory(task.target.target, task.target.move_group, task.target.target_type, last_position,
                                                          task.target.velocity_scaling, task.target.acceleration_scaling)

                # apply effects!
                task.bake(trajectory_planner)

            # compute last position
            last_position = RobotState()
            last_position.joint_state.position = task.trajectory_planner.positions[-1]
            last_position.joint_state.name = self.robot.get_group(task.target.move_group).get_active_joints()

    def plan_trajectory(self, target, move_group, target_type, start_state=None, velocity_scaling=None, acceleration_scaling=None):
        """
        Uses MoveIt to compute a motion plan to a given pose target, then
        initialize a trajectory planner. If no custom start state is specified,
        the current robot state is used as start state.
        """

        if start_state is not None:
            self.robot.get_group(move_group).set_start_state(start_state)
        else:
            self.robot.get_group(move_group).set_start_state(self.robot.get_group(move_group).get_current_state())

        if velocity_scaling is not None:
            self.robot.get_group(move_group).set_max_velocity_scaling_factor(velocity_scaling)
        
        if acceleration_scaling is not None:
            self.robot.get_group(move_group).set_max_acceleration_scaling_factor(acceleration_scaling)
        
        # set target
        if target_type == 'pose':
            self.robot.get_group(move_group).set_pose_target(target)
        elif target_type == 'joint':
            self.robot.get_group(move_group).set_joint_value_target(target)
        elif target_type == 'orientation':
            self.robot.get_group(move_group).set_orientation_target(target)
        else:
            self.robot.get_group(move_group).set_position_target(target)

        # compute!
        plan_success, mplan, _, error_code = self.robot.get_group(move_group).plan()

        # if planning failed, return error code
        if not plan_success:
            return error_code

        # extract plan and initialize new trajectory planner
        points = mplan.joint_trajectory.points

        # get times
        times = np.zeros(len(points))
        for i in range(len(points)):
            times[i] = points[i].time_from_start.secs + (points[i].time_from_start.nsecs / 1000000000)

        # get positions
        positions = np.zeros((len(points), len(points[0].positions)))
        for i in range(len(points)):
            for j in range(len(points[0].positions)):
                positions[i, j] = points[i].positions[j]
        
        # initialize trajectory planner and append it to list
        trajectory_planner = TrajectoryPlanner(times, positions)
        return trajectory_planner

    def execute(self, original=False):
        """
        Execute the current plan through the given JointState publisher.
        """

        # go through each element, plan and execute it!
        for task in self.plan:

            # if this task is not baked, calculate motion and apply effects
            if not task.is_baked():
                # if element is animation instance, the trajectory planner is ready to use
                if task.is_animation():
                    # TODO transition from last position?
                    task.bake()
                
                # otherwise, the motion needs to be computed
                else:
                    # if this is the first element, the current robot pose suffices as start pose.
                    # otherwise take the last pose of the previous element as starting pose
                    trajectory_planner = self.plan_trajectory(task.target.target, task.target.move_group, task.target.target_type, last_position,
                                                            task.target.velocity_scaling, task.target.acceleration_scaling)

                    # apply effects!
                    task.bake(trajectory_planner)

            # compute last position
            last_position = RobotState()
            last_position.joint_state.position = task.get_last_joint_state()
            last_position.joint_state.name = self.robot.get_group(task.target.move_group).get_active_joints()

            # execute
            self._execute_trajectory(task.trajectory_planner)
            self._last_trajectory_planner = task.trajectory_planner

        # when everything is done, return True
        return True

    def _execute_trajectory(self, trajectory_planner, original=False):
        """
        Execute the given trajectory planner
        """
        
        # define rate and start time
        rate = rospy.Rate(30)
        time_start = time.time()

        # initialize joint state
        joint_state = JointState()
        joint_state.name = self.joint_names
        joint_state.header.frame_id = self.frame_id
        
        if self.fake_display:
            display = DisplayRobotState()
            display.state.joint_state = joint_state

        # if not all joint names are used, remove them from the message


        # start publishing!
        while not rospy.is_shutdown() and not trajectory_planner.done:
            
            # set timestamp
            joint_state.header.stamp = rospy.get_rostime()

            # set joint position
            joint_state.position = trajectory_planner.get_position_at(time.time() - time_start, 
                                                                      original=original).tolist()
            
            if len(joint_state.position) < len(joint_state.name):
                joint_state.position += [0] * (len(joint_state.name) - len(joint_state.position))
                
            # publish and wait for next tick
            if self.fake_display:
                self.publisher.publish(display)
            else:
                self.publisher.publish(joint_state)
            rate.sleep()
