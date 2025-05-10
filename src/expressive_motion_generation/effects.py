""" Messages for effects """

import numpy as np
from typing import Optional
from expressive_motion_generation.trajectory_planner import TrajectoryPlanner
from expressive_motion_generation.animation_execution import Animation

class Effect:
    """ Base class for all effects """

    def __init__(self, start_index=0, stop_index=-1):
        """
        Parameters:
        - start_index: Keyframe index where the effect should begin.
        - stop_index: Keyframe index where the effect should end. If this is -1, the last keyframe will be used. 
        """
        self.start_index: int = start_index
        self.stop_index: int = stop_index
    
    def apply(self, trajectory_planner: TrajectoryPlanner, animation: Optional[Animation] = None):
        """ Apply this effect to a given trajectory planner.
         If the task type is animation, providing the animation can add
          metadata for application. """
        pass

class JitterEffect(Effect):
    """ Jitter effect: Adds shakiness to the motion. Amount indicates the maximum modification value
     that is added or subtracted from the joint values in radians. """

    def __init__(self, amount: float = 0.01, fill: int = 0,
                 start_index:int = 0, stop_index:int = -1):
        """
        Parameters:
        - amount: Maximum radian value that is added or subtracted.
        - fill: The rate that the trajectory planner is filled up to to make the effect applicable.
        - start_index: Keyframe index where the effect should begin.
        - stop_index: Keyframe index where the effect should end. If this is -1, the last keyframe will be used. 
        """
        super().__init__(start_index, stop_index)
        self.amount = amount
        self.fill = fill
    
    def apply(self, trajectory_planner: TrajectoryPlanner, animation: Optional[Animation] = None):
        if self.fill > 0:
            trajectory_planner.fill_up(self.fill)
        trajectory_planner.add_jitter(self.amount)


class GazeEffect(Effect):
    """ Gaze effect: Lets a link orient a given axis through the link at a specific point. This can be used
    to have the robot point at something or 'look' at something. """

    def __init__(self, point, link: str, move_group: str, 
                 axis=[0,0,1], movable=[], start_index:int = 0, stop_index:int = -1):
        """
        Parameters:
        - point: Coordinates of point to look at
        - link: Name of link to point towards point
        - move_group: MoveIt move group that uses the current joint group
        - axis: Direction vector of axis to point towards point
        - movable: List of joint indices to use for the action. If none are specified, use all joints.
        - start_index: Keyframe index where the effect should begin.
        - stop_index: Keyframe index where the effect should end. If this is -1, the last keyframe will be used. 
        """
        super().__init__(start_index, stop_index)
        self.point = np.array(point)
        self.link = link
        self.move_group = move_group
        self.axis = np.array(axis)
        self.movable = movable
    
    def apply(self, trajectory_planner: TrajectoryPlanner, animation: Optional[Animation] = None):
        print(f'Start: {self.start_index}, Stop: {self.stop_index}')
        # if animation, only apply on keyframes and then fill up again
        if not animation is None:

            new_trajectory_planner = TrajectoryPlanner(animation.times, animation.positions)
            new_trajectory_planner.add_gaze(self.point, self.link, self.move_group, 
                                            self.axis, self.movable, self.start_index, 
                                            self.stop_index)

            # apply to old animation
            animation.positions = new_trajectory_planner.positions
            animation._reload_trajectory()

            # apply to trajectory planner
            trajectory_planner.positions = animation.trajectory_planner.positions
            trajectory_planner.times = animation.trajectory_planner.times
                
        # if not, just apply
        else:
            trajectory_planner.add_gaze(self.point, self.link, self.move_group, 
                                        self.axis, self.movable, self.start_index, 
                                        self.stop_index)
