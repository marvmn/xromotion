import moveit_commander
import numpy as np
import rospy
from sensor_msgs.msg import JointState
import time
from expressive_motion_generation.trajectory_planner import TrajectoryPlanner

class ExpressivePlanner:

    def __init__(self, movegroup_commander: moveit_commander.MoveGroupCommander):
        self.group = movegroup_commander
        self.trajectory_planner = None

        # initialize ros and moveit
        rospy.init_node("expressive_planner", anonymous=True)

        # initialize publisher
        self.publisher = rospy.Publisher("/joint_command", JointState, queue_size=10)
        self.joint_state = JointState()
        self.joint_state.name = [
            "panda_finger_joint1",
            "panda_finger_joint2",
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]
        self.joint_state.header.frame_id = "panda_link0"
    
    def plan_trajectory(self, pose_target):
        """
        Uses MoveIt to compute a motion plan to a given pose target, then
        initialize a trajectory planner.
        """

        # command moveit to plan trajectory
        self.group.set_pose_target(pose_target)
        plan_success, plan, planning_time, error_code = self.group.plan()

        # extract plan and initialize new trajectory planner
        points = plan.joint_trajectory.points

        # get times
        times = np.zeros(len(points))
        for i in range(len(points)):
            times[i] = points[i].time_from_start.secs + (points[i].time_from_start.nsecs / 1000000000)

        # get positions
        positions = np.zeros((len(points), len(points[0].positions)))
        for i in range(len(points)):
            for j in range(len(points[0].positions)):
                positions[i, j] = points[i].positions[j]
        
        self.trajectory_planner = TrajectoryPlanner(times, positions)
        return self.trajectory_planner

    def execute(self, original=False):
        """
        Execute the current trajectory planner's plan through the given JointState publisher.
        """

        # check if a trajectory planner is initialized
        if self.trajectory_planner == None:
            print("No plan to be executed.")
            return
        
        # TODO: check if motion plan is valid (starts at current pos)

        # execute motion
        rate = rospy.Rate(30)
        time_start = time.time()

        while not rospy.is_shutdown() and not self.trajectory_planner.done:
            self.joint_state.header.stamp = rospy.get_rostime()
            self.joint_state.position = [0, 0] + self.trajectory_planner.get_position_at(
                time.time() - time_start, original=original).tolist()
            #print(joint_state.position)
            self.publisher.publish(self.joint_state)
            rate.sleep()

        return True
