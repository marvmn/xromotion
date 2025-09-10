import moveit_commander
import numpy as np
import threading
import rclpy
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import JointState
import moveit_msgs.msg # import RobotState, DisplayRobotState
import time
from expressive_motion_generation.trajectory import Trajectory
from expressive_motion_generation.animation import Animation
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

    def __init__(self, target: Union[Animation, TargetPlan, float], trajectory_planner: Optional[Trajectory]=None):
        """
        Target is either an Animation, a TargetPlan defining the planned motion, or a float
        defining a wait interval.
        If the trajectory planner is already ready, it defines this task as baked.
        """
        self.target = target
        
        if type(self.target) == int:
            self.target = float(self.target)
        
        self.trajectory_planner = trajectory_planner
        self.custom_effect_order = False
        self.effects = []
    
    def set_custom_effect_order(self, value: bool):
        """
        Defines if the effects will be ordered to comfort to the default order before
        applyling.

        Parameters:
        - value: If True, effects will not be reordered to fit the recommended order
        """
        self.custom_effect_order = value
    
    def bake(self, trajectory_planner: Optional[Trajectory] = None):
        """
        Set trajectory planner and apply effects.

        Parameters:
        - trajectory_planner: If this task does not contain an animation, trajectory \
            needs to be computed and given as argument.
        """
        # set trajectory planner
        if self.is_animation():
            self.trajectory_planner = self.target.trajectory_planner
        elif self.is_wait():
            self.trajectory_planner = Trajectory([0, self.target],
                                                        np.array([[], []]))
            return
        else:
            if trajectory_planner is None:
                raise ValueError("Trajectory planner must not be None if this is not an animation.")
            self.trajectory_planner = trajectory_planner
        
        # reorder effects if needed
        if not self.custom_effect_order:
            self._sort_effects()

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
    
    def _sort_effects(self):
        """ Sorts effects to be in the default order. This guarantees two
        things:
        - GazeEffect will be applied first, which is recommended, since it \
            can overwrite all other effects
        - ExtentEffect will be applied last, since it increases the amount \
            of keyframes by a large amount."""
        new_effects = []
        end = []
        for i in range(len(self.effects)):
            if type(self.effects[i]) == GazeEffect:
                new_effects.insert(0, self.effects[i])
            elif type(self.effects[i]) == ExtentEffect:
                end.append(self.effects[i])
            else:
                new_effects.append(self.effects[i])
        self.effects = new_effects + end

    def is_baked(self):
        """ Is this task already baked/ready to be executed?"""
        return not self.trajectory_planner is None

    def is_animation(self):
        """ Is this task an animation? """
        return type(self.target) == Animation
    
    def is_wait(self):
        """ Is this task a pause interval? """
        return type(self.target) == float

    def get_last_joint_state(self):
        """ Returns last joint state for this task """
        if self.is_wait():
            return None
        if self.is_baked() :
            return self.trajectory_planner.positions[-1]
        elif self.is_animation():
            return self.target.positions[-1]
        elif self.target.target_type == 'joint':
            return self.target.target
        else:
            return None
            

    def __str__(self):
        if self.is_animation():
            target_type = f"Animation {self.target.name}"
        elif self.is_wait():
            target_type = f"Wait {self.target}s"
        else:
            target_type = f"Target ({self.target.target_type}) to {self.target.target}"
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
        rclpy.init()
        self.t = threading.Thread(target=self._spin_in_background)
        self.t.start()
        self.node = rclpy.create_node("expressive_planner")
        rclpy.get_global_executor().add_node(self.node)

        # initialize publisher
        if fake_display:
            self.publisher = self.node.create_publisher(moveit_msgs.msg.DisplayRobotState, publish_topic, queue_size=10)
        else:
            self.publisher = self.node.create_publisher(JointState, publish_topic, queue_size=10)
    
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
        return True
    
    def plan_pause(self, time: float) -> bool:
        """
        Plan a pause interval.

        Parameters:
        - time: Time to wait in seconds
        
        Returns:
        - True if successful, otherwise False.
        """
        task = Task(time)
        self.plan.append(task)
        return True

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

    def get_task_at(self, index: int) -> Optional[Task]:
        """
        Returns the task at the given index.

        Parameters:
        - index: Index of the task
        """
        # if index is not valid, return None
        if len(self.plan) <= index:
            return None
        return self.plan[index]

    def get_last_joint_state(self, index=-1):
        """ If there are any elements in the task, return the joint state that the current plan will end in """
        if not self.plan or abs(index) > len(self.plan):
            return None
        else:
            if self.plan[index].is_wait():
                return self.get_last_joint_state(index - 1)
            state = self.plan[index].get_last_joint_state()
            if state is None:
                self.plan[index].bake()
                state = self.plan[index].get_last_joint_state()
                if state == None:
                    print("[ExpressivePlanner] Last joint state could not be computed. Prebaking plan.")
                    self.bake()
                    state = self.plan[index].get_last_joint_state()
                    if state == None:
                        raise RuntimeError("[ExpressivePlanner] Last joint state could not be computed.")
                return state
            else:
                return state

    def bake(self):
        """
        Pre-compute all trajectories and prepare trajectory planners for execution.
        """
        last_position = None

        # go through each element
        for task in self.plan:
            
            # if element is animation instance or pause, the trajectory planner is ready to use
            if task.is_animation() or task.is_wait():
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
            if not task.is_wait():
                last_position = moveit_msgs.msg.RobotState()
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
        elif target_type == 'name':
            self.robot.get_group(move_group).set_named_target(target)
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
        trajectory_planner = Trajectory(times, positions,
                                               self.robot.get_group(move_group).get_active_joints())
        return trajectory_planner

    def execute(self, original=False, debug_output=False):
        """
        Execute the current plan through the given JointState publisher.
        """
        last_position = None

        # go through each element, plan and execute it!
        for task in self.plan:

            if debug_output:
                print("Executing task", task)

            # if this task is not baked, calculate motion and apply effects
            if not task.is_baked():
                # if element is animation instance, the trajectory planner is ready to use
                if task.is_animation() or task.is_wait():
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
            if not task.is_wait():
                last_position = moveit_msgs.msg.RobotState()
                last_position.joint_state.position = task.get_last_joint_state()
                last_position.joint_state.name = self.robot.get_group(task.target.move_group).get_active_joints()

            # execute
            self._execute_trajectory(task.trajectory_planner)
            self._last_trajectory_planner = task.trajectory_planner

        # when everything is done, return True
        return True

    def _spin_in_background():
        """
        Called by thread to let callbacks run in the background.
        """
        executor = rclpy.get_global_executor()
        try:
            executor.spin()
        except ExternalShutdownException:
            pass

    def _execute_trajectory(self, trajectory_planner: Trajectory, original=False):
        """
        Execute the given trajectory planner
        """
        
        # define rate and start time
        rate = rclpy.create_rate(30)
        time_start = time.time()

        # initialize joint state
        joint_state = JointState()
        joint_state.name = trajectory_planner.joint_names
        joint_state.header.frame_id = self.frame_id
        
        if self.fake_display:
            display = moveit_msgs.msg.DisplayRobotState()
            display.state.joint_state = joint_state

        # if not all joint names are used, remove them from the message


        # start publishing!
        while rclpy.ok() and not trajectory_planner.done:
            
            # set timestamp
            joint_state.header.stamp = self.node.get_clock().now()

            # set joint position
            joint_state.position = trajectory_planner.get_position_at(time.time() - time_start, 
                                                                      original=original).tolist()
                
            # publish and wait for next tick
            if self.fake_display:
                self.publisher.publish(display)
            else:
                self.publisher.publish(joint_state)
            rate.sleep()
        
        # reset trajectory
        trajectory_planner.done = False

    def __del__(self):
        if self.t:
            self.t.join()
