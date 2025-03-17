import expressive_motion_generation.trajectory_planner
from expressive_motion_generation.expressive_planner import ExpressivePlanner
import rospy
import geometry_msgs
import moveit_commander
import moveit_msgs.msg
import sys

# initialize moveit 
moveit_commander.roscpp_initialize(sys.argv)

# instantiate Robot Commander
robot = moveit_commander.RobotCommander()

# instantiate planning scene interface
scene = moveit_commander.PlanningSceneInterface()

# instantiate movegroupcommander
group_name = "panda_arm"
group = moveit_commander.MoveGroupCommander(group_name)


# create expressive planner
planner = ExpressivePlanner(movegroup_commander=group)

# try out a pose
pose_goal = geometry_msgs.msg.Pose()
pose_goal.orientation.w = 1.0
pose_goal.position.x = 0.4
pose_goal.position.y = 0.1
pose_goal.position.z = 0.4

planner.plan_trajectory(pose_target=pose_goal)

# add modifications
planner.trajectory_planner.add_uncertainty(0.08)
planner.trajectory_planner.scale_global_speed(0.2)

# go!
planner.execute(original=False)
