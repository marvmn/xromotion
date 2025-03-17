import numpy as np
import copy

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
    
    def add_uncertainty(self, amount=0.3):
        """
        Applies a certain amount of randomness to a motion to make it seem
        less confidant. The randomness is scaled down at the beginning and
        the end to avoid conflicts with the functional objective.
        """

        # first, generate random summands for every position
        summands = np.random.normal(0.0, amount, self.positions.shape)

        # scale the summands over time according to a parabola
        # to ensure that the start and end point of the motion stay the same
        parabola = -0.01 * (self.times + 0.0) * (self.times - self.times[len(self.times) - 1])
        
        # apply to positions
        self.positions += (summands.T * parabola).T

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


