import sys
import expressive_motion_generation.trajectory_planner
from expressive_motion_generation.expressive_planner import ExpressivePlanner
from expressive_motion_generation.animation_execution import Animation
from expressive_motion_generation.effects import *
import rospy
import geometry_msgs
import moveit_commander
import moveit_msgs.msg
import numpy as np
import matplotlib.pyplot as plt
# initialize moveit 
moveit_commander.roscpp_initialize(sys.argv)

# instantiate Robot Commander
robot = moveit_commander.RobotCommander()
active_joints = robot.get_active_joint_names()

# create expressive planner
planner = ExpressivePlanner(robot=robot, publish_topic='joint_command', fake_display=False)
robot.get_group('panda_arm').set_end_effector_link('panda_hand')

# try out a pose
pose_goal = geometry_msgs.msg.Pose()
pose_goal.orientation.w = -0.2
pose_goal.orientation.x = -0.6
pose_goal.orientation.y = -0.2
pose_goal.orientation.z = -0.6
pose_goal.position.x = 0.4
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
from expressive_motion_generation.utils import make_point_at_task, make_point_at_task_from
# try out animation
active_joints.pop(6)
active_joints.pop(4)
active_joints.pop(2)

# planner.new_plan()
# planner.plan_animation("/home/mwiebe/noetic_ws/IsaacSim-ros_workspaces/noetic_ws/panda_animations/animation_happy2.yaml")
# gaze = {'point':[1.6, 0.0, 0.6], 'move_group':'panda_arm', 'link':'panda_hand', 'axis':[0,0,1], 'up':[1,0,0],
#         'movable': [4,5], 'from':2, 'to':18}
# planner.apply_effects(index=0, gaze=gaze)
# planner.plan_animation("/home/mwiebe/noetic_ws/IsaacSim-ros_workspaces/noetic_ws/panda_animations/animation_happy2.yaml")
# #planner.plan_target(pose_goal, 'panda_arm', 1.0, 1.0, 'pose')
# #planner.apply_effects(index=1, jitter=0.01)
# planner.bake()
# planner.execute()

planner.new_plan()
planner.plan_animation("/home/mwiebe/noetic_ws/IsaacSim-ros_workspaces/noetic_ws/panda_animations/animation_happy2.yaml")
planner.at(0).add_effects(GazeEffect([0.6, 0, 0.6], 'panda_hand', 'panda_arm', start_index=3, stop_index=17))
planner.plan_target(pose_goal, 'panda_arm', 1.0, 1.0, 'pose')
planner.at(1).add_effects(JitterEffect(0.02))
planner.bake()
planner.add_task(make_point_at_task_from(robot, 'panda_arm', [1.6, 0, 0.6], 'panda_hand', planner.get_last_joint_state()))
planner.execute()
