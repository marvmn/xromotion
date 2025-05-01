import moveit_commander
import numpy as np
import rospy
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState
import time
from expressive_motion_generation.trajectory_planner import TrajectoryPlanner
from expressive_motion_generation.animation_execution import Animation

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
        self.effects = {}

class ExpressivePlanner:

    def __init__(self, robot: moveit_commander.RobotCommander, publish_topic='joint_command'):
        """
        Expressive planning framework for functional and expressive animated robot motion.
        """

        self.plan = []
        self.baked = []

        self.robot = robot
        self.frame_id = robot.get_planning_frame()
        self.joint_names = robot.get_active_joint_names()

        # initialize ros and moveit
        rospy.init_node("expressive_planner", anonymous=True)

        # initialize publisher
        self.publisher = rospy.Publisher(publish_topic, JointState, queue_size=10)
    
    def new_plan(self):
        """
        Initialize a new plan for the given robot.
        """
        self.plan = []
        self.baked = []
    
    def plan_animation(self, path):
        """
        Load an animation from a file and append it to plan
        """
        animation = Animation(path)
        self.plan.append(animation)
    
    def plan_target(self, target, move_group=None, velocity_scaling = None, acceleration_scaling = None):
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
                move_group = self.plan[-1].move_group

            else:

                # otherwise get the available move groups and choose the first one
                move_group = self.robot.get_group_names()[0]

        # now that the move_group is set, add target to plan!
        target_plan = TargetPlan(target, move_group, velocity_scaling=velocity_scaling, acceleration_scaling=acceleration_scaling)
        self.plan.append(target_plan)

    
    def apply_effects(self, index=-1, **kwargs):
        """
        Apply given effects to the plan part at the given index.
        If no index is given, the last plan element is chosen.
        Instead of an index, a trajectory planner can also be given.
        
        Possible effects:
        jitter=<jitter amount>
        """
        if type(index) == int:
            # if the chosen element is an Animation, use the embedded trajectory planner and apply effect
            if type(self.plan[index]) == Animation:
                trajectory_planner = self.plan[index].trajectory_planner

            # otherwise this is a TargetPlan, so add this effect for future computing
            else:
                self.plan[index].effects = kwargs
                return
        else:
            trajectory_planner = index

        # apply jitter if given
        if 'jitter' in kwargs.keys():
            trajectory_planner.add_jitter(kwargs['jitter'])

                
    def bake(self):
        """
        Pre-compute all trajectories and prepare trajectory planners for execution.
        """
        self.baked = []

        # go through each element
        for element in self.plan:
            
            # if element is animation instance, the trajectory planner is ready to use
            if type(element) == Animation:
                self.baked.append(element.trajectory_planner)
            
            # otherwise, the motion needs to be computed
            else:
                # if this is the first element, the current robot pose suffices as start pose.
                # otherwise take the last pose of the previous element as starting pose
                start_state = None

                if self.baked:
                    last_position = self.baked[-1].positions[-1]
                    start_state = RobotState()
                    start_state.joint_state.position = last_position
                    start_state.joint_state.name = self.robot.get_group(element.move_group).get_active_joints()
                
                trajectory_planner = self.plan_trajectory(element.target, element.move_group, element.target_type, start_state,
                                                          element.velocity_scaling, element.acceleration_scaling)

                # apply effects!
                self.apply_effects(trajectory_planner, **element.effects)

                self.baked.append(trajectory_planner)

    def plan_trajectory(self, target, move_group, target_type, start_state=None, velocity_scaling=None, acceleration_scaling=None):
        """
        Uses MoveIt to compute a motion plan to a given pose target, then
        initialize a trajectory planner. If no custom start state is specified,
        the current robot state is used as start state.
        """

        if start_state is not None:
            self.robot.get_group(move_group).set_start_state(start_state)

        if velocity_scaling is not None:
            self.robot.get_group(move_group).set_max_velocity_scaling_factor(velocity_scaling)
        
        if acceleration_scaling is not None:
            self.robot.get_group(move_group).set_max_acceleration_scaling_factor(acceleration_scaling)
        
        # set target
        if target_type == 'pose':
            self.robot.get_group(move_group).set_pose_target(target)
        elif target_type == 'joint':
            self.robot.get_group(move_group).set_joint_value_target(target)
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

        # check if the plan has been baked
        if self.baked:
            
            # if so, execute the baked trajectory planners
            # go through each element and execute it
            for element in self.baked:
                self._execute_trajectory(element)
            
        else:

            # go through each element, plan and execute it!
            for element in self.plan:

                # if this element is an animation instance, use the provided trajectory planner
                if type(element) == Animation:
                    self._execute_trajectory(element.trajectory_planner)
                
                # otherwise plan the motion with moveit and execute it
                else:
                    trajectory_planner = self.plan_trajectory(element.target, element.move_group,
                                                              element.target_type,
                                                              velocity_scaling=element.velocity_scaling,
                                                              acceleration_scaling=element.acceleration_scaling)
                    
                    self.apply_effects(trajectory_planner, **element.effects)

                    self._execute_trajectory(trajectory_planner)

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
            self.publisher.publish(joint_state)
            rate.sleep()
