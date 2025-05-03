import copy
import numpy as np
import tf.transformations
from geometry_msgs.msg import Pose

class TrajectoryPlanner:

    def __init__(self, times, positions):
        """
        Initialize trajectory planner.
        times -- Array with timestamps for the keyframes
        positions -- Array with joint positions for the keyframes
        """
        # set true, unmodified times
        self.true_times = times

        # set array with timestamps for the keyframes
        self.times = copy.deepcopy(times)

        # set true, unmodified positions
        self.true_positions = positions

        # set positions that will be modified through expression transforms
        self.positions = copy.deepcopy(positions)

        # When the last point of the trajectory is reached, done becomes True
        self.done = False
    
    def scale_global_speed(self, scalar):
        """
        Scales the speed of the trajectory globally, meaning that every
        keyframe's time gets scaled by the same scalar.
        When scalar > 1 the movement is slower, when 0 < scalar < 1 it
        gets faster.
        """
        for i in range(len(self.times)):
            self.times[i] = self.times[i] * scalar
    
    def add_jitter(self, amount=0.05):
        """
        Applies a certain amount of randomness to a motion to make it seem
        less confident. The randomness is scaled down at the beginning and
        the end to avoid conflicts with the functional objective.
        """

        # first, generate random summands for every position
        summands = np.random.normal(0.0, amount, self.positions.shape)

        # make the effect fade in and out at the beginning and end of the motion
        # to ensure that the start and end point of the motion stay the same
        parabola = -0.05 * self.times[-1] * self.times * (self.times - self.times[-1])
        cut = np.min([np.ones(parabola.shape), parabola], axis=0)
        
        # apply to positions
        self.positions += (summands.T * cut).T

        # finally, scale down global scale a little
        self.scale_global_speed(1.0 + max(amount / 2, 0.3))

    def add_gaze(self, point):
        pass

    def get_position_at(self, timestamp, original=False):
        """
        Interpolates and returns the point at the specified timestamp (in seconds).
        If original is True, use the original trajectory instead of the modified one.
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

    def apply_bezier_at(self, index0, index1, cp0, cp1):
        """
        Scales the velocity of the motion between two keyframes according to
        a Bézier curve. The curve assumes the position at index0 to be (0,0)
        and the position at index1 to be (1,1) to unify the choice of the
        control points cp0 and cp1 across different joint states.
        """

        # check if the two points are the same
        if (self.times[index0] == self.times[index1]):
            return

        # first, calculate the bezier curve with n points
        n = 20

        bezier_fn = lambda t: (1 - t)**3 * np.array([0,0]) + 3*(1 - t)**2 * t * cp0 + 3 * (1 - t) * t**2 * cp1 + t**3 * np.array([1,1])
        curve = bezier_fn(np.linspace([0,0], [1,1], n)).T
        #print("CURVE:\n", curve)

        # now interpolate exact position for every timestamp
        for idx in range(index1 - index0):
            
            # get x coordinate on the curve
            # for that normalize the times-intervall that this curve operates on
            x = (self.times[index0 + idx] - self.times[index0]) / (self.times[index1] - self.times[index0])
            time = 0.0
            if x == 0:
                continue
            for i in range(len(curve[0])):
                if x <= curve[0][i]:
                    diff = curve[0, i] - curve[0, i - 1]
                    prog = x - curve[0, i - 1]
                    scal = prog / diff
                    time = curve[1, i - 1] + (curve[1, i] - curve[1, i - 1]) * scal
                    break
            
            # finally apply calculated time back to times array
            self.times[index0 + idx] = time * (self.times[index1] - self.times[index0]) + self.times[index0]

    def fill_up(self, frequency):
        """
        Fills up the times and positions arrays with interpolated values so that the
        resulting times are filled with a frequency of >frequency per second.
        """
        new_positions = copy.deepcopy(self.positions)
        new_times = copy.deepcopy(self.times)

        # go through every intervall and check if it satisfies the rate
        added = 0
        original_indices = []

        for i in range(len(self.times) - 1):

            original_indices.append(i + added)

            intervall = self.times[i + 1] - self.times[i]

            # if the intervall between these two timestamps it too big, it doesn't satisfy the rate
            # that means new keyframes need to be inserted.
            if intervall > 1/frequency:
                
                j_idx = 0
                added_now = 0
                while (j_idx + 1) * 1/frequency < intervall:
                    new_times = np.insert(new_times, i + j_idx + added + 1, self.times[i] + 1/frequency * (j_idx + 1))
                    new_positions = np.insert(new_positions, i + j_idx + added + 1, self.get_position_at(new_times[i+j_idx + added + 1]), axis=0)
                    j_idx += 1
                    added_now += 1
                added += added_now
        
        # finally save new times and positions
        self.positions = new_positions
        self.times = new_times

        # return the indices of the original keyframes
        original_indices.append(added)
        return original_indices

    def _get_pointing_pose(frame, point, axis=[0, 0, 1], up_vector=[1, 0, 0]):
        """
        Get end effector pose for pointing a axis in the link to a specific point in space.

        Parameters:
        - frame: Position of the end effector link
        - point: Position of the point that the link should point towards
        - axis: The axis that should point at the point
        - up_vector: When pointing at the point, this axis will point as far upwards as possible.
        """

        # get direction vector
        direction = np.array(point) - np.array(frame)
        direction /= np.linalg.norm(direction)

        # normalize axis
        axis = np.array(axis) / np.linalg.norm(axis)

        # align axis with end effector:

        # get rotation axis: EEF can be rotated around axis orthogonal to both the direction vector and
        # the axis vector, so calculate cross product
        rotation_axis = np.cross(axis, direction)

        # get rotation angle
        angle = np.dot(axis, direction)

        # if the norm of the cross product is (close to) zero, then axis and direction are already parallel
        # in that case, check if they point in the same direction
        if np.linalg.norm(rotation_axis) < 0.0001:
            if angle > 0:
                # they point in the same direction; no further transformation is needed
                rotation_matrix = np.eye(3)
            else:
                # they point into opposite directions; rotate 180 degrees
                # calculate orthogonal axis to rotate around
                orthogonal = np.array([1, 0, 0])
                if np.allclose(orthogonal, axis, atol=0.01):
                    orthogonal = np.array([0, 1, 0])
                orthogonal = np.cross(axis, orthogonal)
                orthogonal /= np.linalg.norm(orthogonal)

                # build rotation matrix (see Rodrigues' rotation formula)
                skew_mat = np.array([[0, -orthogonal[2], orthogonal[1]],
                                    [orthogonal[2], 0, -orthogonal[0]],
                                    [-orthogonal[1], orthogonal[0], 0]])
                rotation_matrix = np.eye(3) + np.sin(np.pi) * skew_mat \
                                + (1 - np.cos(np.pi)) * (skew_mat @ skew_mat)
        else:
            # if axes are not aligned, rotate
            skew_mat = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                                [rotation_axis[2], 0, -rotation_axis[0]],
                                [-rotation_axis[1], rotation_axis[0], 0]])
            rotation_matrix = np.eye(3) + skew_mat + skew_mat@skew_mat * \
                            ((1 - angle) / np.linalg.norm(rotation_axis)**2)
        
        # rotate up vector
        up_vector = rotation_matrix @ up_vector

        # use Gram-Schmidt process to build frame, where the z axis is the direction axis
        # and the y axis points up (as far as possible)
        up_world_vector = np.array([0, 0, 1])
        
        # calculate angle between up_vector and up_world_vector
        angle = np.arccos(np.clip(np.dot(up_vector, up_world_vector), -1.0, 1.0))

        # if dot product of the rotation axis and the cross product of up_vector and up_world_vector
        # is negative, the rotation needs to be inverted
        if np.dot(np.cross(up_vector, up_world_vector), direction) < 0:
            angle *= -1
        
        # build rotation matrix to align the pointing frame better with the up vector
        skew_mat = np.array([[0, -direction[2], direction[1]],
                                [direction[2], 0, -direction[0]],
                                [-direction[1], direction[0], 0]])
        up_rotation_matrix = np.eye(3) + np.sin(angle) * skew_mat \
                            + (1 - np.cos(angle)) * (skew_mat @ skew_mat)

        # build final rotation matrix
        final_rotation = np.eye(4)
        final_rotation[:3, :3] = up_rotation_matrix @ rotation_matrix
        quaternion = tf.quaternion_from_matrix(final_rotation)

        pose = Pose()
        pose.position.x = frame[0]
        pose.position.y = frame[1]
        pose.position.z = frame[2]
        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]

        return pose

