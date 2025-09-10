import copy
import numpy as np
from typing import Optional
import tf.transformations as tf
from sensor_msgs.msg import JointState
from moveit_commander.robot import RobotCommander
from moveit_msgs.msg import PositionIKRequest, Constraints, JointConstraint
from moveit_msgs.srv import GetPositionIKResponse, GetPositionFKRequest, GetPositionFK, GetPositionIK

class Trajectory:

    def __init__(self, times, positions, joint_names=[]):
        """
        Initialize trajectory planner.

        Parameters:
        - times: Array with timestamps for the keyframes
        - positions: Array with joint positions for the keyframes
        - joint_names: List of joint names
        """
        # set true, unmodified times
        self.true_times = np.array(times, dtype=float)

        # set array with timestamps for the keyframes
        self.times = copy.deepcopy(self.true_times)

        # set true, unmodified positions
        self.true_positions = np.array(positions, dtype=float)

        # set positions that will be modified through expression transforms
        self.positions = copy.deepcopy(self.true_positions)

        # indices of original positions in case the trajectory was filled up
        # with interpolated values
        self.original_indices = range(len(times))

        # joint names that the joint positions are given for
        self.joint_names = joint_names

        # When the last point of the trajectory is reached, done becomes True
        self.done = False
    
    def scale_global_speed(self, scalar: float):
        """
        Scales the speed of the trajectory globally, meaning that every
        keyframe's time gets scaled by the same scalar.
        When scalar > 1 the movement is slower, when 0 < scalar < 1 it
        gets faster.

        Parameters:
        - scalar: Every time stamp gets multiplied with this scalar.
        """
        for i in range(len(self.times)):
            self.times[i] = self.times[i] * scalar

    def enforce_joint_limits(self, robot: RobotCommander, move_group: Optional[str] = None):
        """
        Check all joint state values in positions and enforce joint limits.
        Values that exceed the upper bound are capped to the upper bound's exact value,
        values that fall under the lower bound are accordingly set to the lower bound's value.

        Parameters:
        - robot: RobotCommander with the robot description that holds the correct joint limits.
        - move_group: Name of the move group that this trajectory is using for it's joint mapping.
        """
        if move_group is None:
            move_group = robot.get_group_names()[0]

        for i in range(len(self.positions)):
            for j in range(len(self.positions[i])):
                joint = robot.get_joint(robot.get_group(move_group).get_active_joints()[j])

                if self.positions[i][j] > joint.max_bound():
                    self.positions[i][j] = joint.max_bound()
                elif self.positions[i][j] < joint.min_bound():
                    self.positions[i][j] = joint.min_bound()
            

    def get_position_at(self, timestamp: float, original=False) -> np.ndarray:
        """
        Interpolates and returns the point at the specified timestamp (in seconds).
        If original is True, use the original trajectory instead of the modified one.

        Parameters:
        - timestamp: Timestamp to get the joint values for. If the timestamp is lower than \
            the first timestamp in this trajectory, the first value is returned. If it is bigger \
            than the last timestamp (or than the trajectory length), the last value is returned.
        - original: If True, the unmodified original values that the trajectory planner was \
            initialized with are used.
        
        Returns:
        - Joint value array for the specified time stamp
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
        

    def fill_up(self, frequency: float):
        """
        Fills up the times and positions arrays with interpolated values so that the
        resulting times are filled with a frequency of >frequency per second.

        Parameters:
        - frequency: Minimum frequency that this trajectory should have (in Hz).
        """
        new_positions = copy.deepcopy(self.positions)
        new_times = copy.deepcopy(self.times)

        # go through every intervall and check if it satisfies the rate
        added = 0
        self.original_indices = []

        for i in range(len(self.times) - 1):

            self.original_indices.append(i + added)
            
            interval = self.times[i + 1] - self.times[i]
            
            # if the intervall between these two timestamps it too big, it doesn't satisfy the rate
            # that means new keyframes need to be inserted.
            if interval > 1/frequency:
                
                insert_times = np.linspace(self.times[i], self.times[i+1],
                                       int(frequency*interval)+1)[1:-1]
                
                for idx, time in enumerate(insert_times):
                    index = i + added + idx + 1
                    new_times = np.insert(new_times, index, time)
                    new_positions = np.insert(new_positions, index, self.get_position_at(time), axis=0)
                    
                added += len(insert_times)
        
        # add final index
        self.original_indices.append(added + len(self.times) - 1)
        
        # finally save new times and positions
        self.positions = new_positions
        self.times = new_times
        
        # return the indices of the original keyframes
        return self.original_indices

    

        
