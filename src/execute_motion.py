import expressive_motion_generation.trajectory_planner
from expressive_motion_generation.expressive_planner import ExpressivePlanner
from expressive_motion_generation.animation_execution import Animation
import rospy
import geometry_msgs
import moveit_commander
import moveit_msgs.msg
import sys
import numpy as np
import matplotlib.pyplot as plt

# initialize moveit 
moveit_commander.roscpp_initialize(sys.argv)

# instantiate Robot Commander
robot = moveit_commander.RobotCommander()

# # instantiate planning scene interface
# scene = moveit_commander.PlanningSceneInterface()

# # instantiate movegroupcommander
# group_name = "panda_arm"
# group = moveit_commander.MoveGroupCommander(group_name)


# create expressive planner
planner = ExpressivePlanner(robot=robot, publish_topic='display_robot_state', fake_display=True)

# try out a pose
pose_goal = geometry_msgs.msg.Pose()
# pose_goal.orientation.w = 0.0
# pose_goal.orientation.x = 0.7
# pose_goal.orientation.y = 0.1
# pose_goal.orientation.z = 0.1
pose_goal.position.x = 0.5
pose_goal.position.y = 0.01
pose_goal.position.z = 0.6

"""
planner.plan_trajectory(pose_target=pose_goal)

# add modifications
#planner.trajectory_planner.add_uncertainty(0.08)
planner.trajectory_planner.scale_global_speed(0.2)
planner.trajectory_planner.apply_bezier_at(0, len(planner.trajectory_planner.times) - 1, np.array([0.5, 0]), np.array([0, 1]))

# plot times
plt.figure()
plt.plot(planner.trajectory_planner.times)
plt.savefig("plot_tp_times")

# go!
#planner.execute(original=False)
"""

# try out animation


# anim = Animation("/home/mwiebe/noetic_ws/IsaacSim-ros_workspaces/noetic_ws/panda_animations/animation_happy2.yaml")

# planner.trajectory_planner = anim.trajectory_planner
# planner.trajectory_planner.add_jitter(0.03)
# #print(planner.trajectory_planner.times)
# planner.execute()

planner.new_plan()
planner.plan_animation("/home/mwiebe/noetic_ws/IsaacSim-ros_workspaces/noetic_ws/panda_animations/animation_happy2.yaml")
planner.plan_target(pose_goal, 'panda_arm', 1.0, 1.0)
planner.apply_effects(index=1, jitter=0.01)
planner.bake()
planner.execute()
