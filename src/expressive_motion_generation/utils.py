import numpy as np
from moveit_commander.robot import RobotCommander
from expressive_motion_generation.animation_execution import Animation
from expressive_motion_generation.expressive_planner import TargetPlan, Task
from expressive_motion_generation.trajectory_planner import TrajectoryPlanner

def make_point_at_task(robot: RobotCommander, move_group: str, point: np.ndarray, link: str, axis=[0,0,1]):
        """
        Create a task that looks at a given point in space. If a time is specified,
        the planner will make the trajectory take that time in seconds.

        Parameters:
        - robot: RobotCommander for the robot
        - move_group: Name of the used move group
        - point: Coordinates of the point in space
        - link: Name of the link that should be pointed
        - axis: Axis inside the link that should be pointed at the point.
        """
        # create targetplan to wrap this goal
        # find pose
        trajectory_planner = TrajectoryPlanner([0.0], [robot.get_group(move_group).get_current_joint_values()],
                                               robot.get_group(move_group).get_active_joints())
        positions = trajectory_planner._get_pointing_joint_state(move_group, robot, 0, link,
                                                                point, axis)
        
        # check if any joint state is close to the limit, because in that case MoveIt will mark
        # the joint state as invalid.
        for i in range(len(positions)):
            joint = robot.get_joint(robot.get_active_joint_names(move_group)[i])
            if positions[i] >= joint.max_bound():
                positions[i] = joint.max_bound() - 0.001
            elif positions[i] <= joint.min_bound():
                positions[i] = joint.min_bound() + 0.001

        # wrap in task and finish!
        target_plan = TargetPlan(positions, move_group, 'joint')
        return Task(target_plan)

def make_point_at_task_from(robot: RobotCommander, move_group: str, point: np.ndarray, link: str, before_state: np.ndarray, 
                            time: float = 3.0, axis=[0,0,1]):
    """
    Create a task that looks at a given point in space. If a time is specified,
        the planner will make the trajectory take that time in seconds.

        Parameters:
        - robot: RobotCommander for the robot
        - move_group: Name of the used move group
        - point: Coordinates of the point in space
        - link: Name of the link that should be pointed
        - before_state: Joint state that should be the beginning of the motion
        - time: Time that the trajectory should take to get from before_state to the pointing pose
        - axis: Axis inside the link that should be pointed at the point.
    """
    # create animation to wrap in
    animation = Animation(None)
    animation.times = np.array([0.0])
    animation.positions = [before_state]
    animation.move_group = move_group
    animation.joint_names = robot.get_group(move_group).get_active_joints()
    animation.name = f"PointAtTask-({point})"

    # find pointing pose
    positions = TrajectoryPlanner(animation.times, animation.positions,
                                  animation.joint_names)._get_pointing_joint_state(move_group, robot, 0, link,
                                                                                                  point, axis)
    
    # append to animation at specified time and return!
    animation.times = np.append(animation.times, time)
    animation.positions.append(positions)
    animation.positions = np.array(animation.positions)
    animation._reload_trajectory()
    return Task(animation)


def convert_animation_to_relative(animation: Animation):
    """
    Convert an absolute animation to a relative animation, based on the first keyframe.

    Parameters:
    - animation: Absolute animation to convert

    Returns:
    - success: True, if conversion successful, otherwise false.
    """

    # check if animation is already relative
    if animation.relative:
        print(f"Animation {animation.name} is already relative! Conversion failed.")
        return False
    
    # extract first position
    base_position = animation.positions[0]

    # subtract this position from all other positions
    matrix = np.tile(base_position, len(animation.times))
    animation.positions -= matrix

    # success, return True!
    return True
