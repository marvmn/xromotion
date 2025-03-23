import geometry_msgs
import moveit_commander
import numpy as np
import sys
import os

print(sys.argv)

# first check if a animation name was given as argument
if not len(sys.argv) == 2:
    print("USAGE: save_keyframe.py <animation name>")
    sys.exit(1)

# check if animation directory exists
if not os.path.isdir(os.path.join(os.getcwd(), "panda_animations")):
    os.makedirs(os.path.join(os.getcwd(), "panda_animations"))

# get animation name and open file
animation_name = sys.argv[1]
animation_file = open("panda_animations/" + animation_name, "a")

# initialize moveit 
moveit_commander.roscpp_initialize([])

# instantiate Robot Commander
robot = moveit_commander.RobotCommander()

# instantiate planning scene interface
scene = moveit_commander.PlanningSceneInterface()

# instantiate movegroupcommander
group_name = "panda_arm"
group = moveit_commander.MoveGroupCommander(group_name)

# get joint positions
values = group.get_current_joint_values()

# write the joint positions to file
print(values, file=animation_file, end="")

# also write default timing and bezier information
print("#0.000#[0.5, 0.0]#[0.5, 1.0]", file=animation_file)

# close file
animation_file.close()

print("Successfully wrote keyframe to file!")