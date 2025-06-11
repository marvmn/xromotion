from expressive_motion_generation.trajectory_planner import TrajectoryPlanner
import json
import os
import numpy as np

# messages
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Duration

# yaml
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

class BezierCurve:

    def __init__(self, indices=(0.0, 0.0), control_point0=np.array([0.5, 0.5]), control_point1=np.array([0.5, 0.5])):
        self.indices = indices
        self.control_point0 = control_point0
        self.control_point1 = control_point1
    
    def __str__(self):
        return "" + str(self.indices[0]) + ":" + str(self.indices[1]) + "#" \
                  + str(self.control_point0.tolist()) + "#" + str(self.control_point1.tolist())

class Animation:

    def __init__(self, animation_path = None):
        """
        Initializes and loads animation from the specified path if not None
        """

        if animation_path is None:
            self.name = ''
            self.move_group = ''
            self.joint_names = ''
            self.frame_id = ''
            self.times = np.array([])
            self.positions = np.array([])
            self.beziers = []
            self.relative = False

            # apply to trajectory planner
            self._reload_trajectory()

        else:
            file = None
            try:
                file = open(os.path.join(os.getcwd(), animation_path), "r")
            except:
                raise FileNotFoundError
            
            # read file and compute times and joint goals
            self._load_yaml(file)
            
            # apply to trajectory planner
            self._reload_trajectory()
            
            # finally close file
            file.close()

    def _reload_trajectory(self, base_position=None):
        """
        Load trajectory planner and apply bezier curves. If this is a relative animation,
        a base position can be given. The relative animation will then be computed from the
        given position.

        Parameters:
        - base_position: If not None, relative animations will be applied from this position
        """

        # load trajectory planner
        self.trajectory_planner = TrajectoryPlanner(self.times, self.positions, self.joint_names)

        # if relative and base position is given, compute absolute positions
        if self.relative and base_position is not None:

            # first check that the base position has the correct length
            assert(len(base_position) == len(self.positions[0])), f"Base Positions has the wrong length: {len(base_position)} \
                (Should be {len(self.positions[0])})"

            # add the base position to every position in the trajectory
            matrix = np.tile(base_position, (len(self.times), 1))
            self.trajectory_planner.positions += matrix

        # fill up to make bezier curves possible
        self.original_indices = self.trajectory_planner.fill_up(20)

        # go through beziers and add them to trajectory
        for i in range(len(self.beziers)):
            curve = self.beziers[i]
            self.trajectory_planner.apply_bezier_at(self.original_indices[curve.indices[0]], 
                                                    self.original_indices[curve.indices[1]], 
                                                    curve.control_point0, curve.control_point1)

    def _load_yaml(self, file):
        """
        Load an animation file (YAML format)
        """
        data = load(file, Loader=Loader)

        # load data
        self.name = data["header"]["animation_name"]
        self.move_group = data["header"]["move_group"]
        self.joint_names = data["trajectory"]["joint_names"]
        self.frame_id = data["trajectory"]["header"]["frame_id"]
        self.relative = data["header"]["relative"]

        # load trajectory
        self.positions = []
        self.times = []
        data_points = data["trajectory"]["points"]

        for i in range(len(data_points)):
            self.positions.append(data_points[i]["positions"])
            self.times.append(data_points[i]["time_from_start"]["data"])
        
        # load curves
        self.beziers = []
        data_curves = data["curves"]

        for i in range(len(data_curves)):
            bezier = BezierCurve((data_curves[i]["indices"][0], data_curves[i]["indices"][1]), 
                                 data_curves[i]["control_point0"], data_curves[i]["control_point1"])
            self.beziers.append(bezier)
        
        # convert to numpy arrays
        self.positions = np.array(self.positions)
        self.times = np.array(self.times)
    
    def add_keyframe(self, time, positions):
        """
        Add a keyframe with the given position at the specified time
        """
        # search correct position
        index = np.searchsorted(self.times, time)

        # insert values
        self.times = np.insert(self.times, index, time)
        self.positions = np.insert(self.positions, index, positions, axis=0)

        # adjust bezier curve indices
        for bezier in self.beziers:
            if bezier.indices[0] >= index:
                bezier.indices = (bezier.indices[0] + 1, bezier.indices[1] + 1)
            elif bezier.indices[1] >= index:
                bezier.indices = (bezier.indices[0], bezier.indices[1] + 1)

        # reload trajectory
        self._reload_trajectory()
        

    def add_bezier(self, index0, index1, control_point0, control_point1):
        """
        Add a Bezier curve to this animation at the specified indices.

        Parameters:
        - index0: Start index of the interval that should be modified by the curve.
        - index1: End index of the interval that should be modified by the curve.
        - control_point0: XY-Coordinates of the first control point in the interval (0, 0) to (1,1)
        - control_point1: XY-Coordinates of the second control point in the interval (0, 0) to (1,1)
        
        """
        bezier = BezierCurve((index0, index1), np.array(control_point0),
                             np.array(control_point1))
        self.beziers.append(bezier)

    def save_yaml(self, file):
        """
        Save the animation data as in YAML format in the specified file
        """

        # save everything in a dictionary
        data = {}

        # save animation header
        data['header'] = {'animation_name': self.name,
                          'move_group': self.move_group,
                          'relative': self.relative}
        
        # save trajectory
        trajectory = JointTrajectory(joint_names=self.joint_names)
        trajectory.header.frame_id = self.frame_id

        for i in range(len(self.positions)):
            point = JointTrajectoryPoint()
            point.time_from_start = Duration(self.times[i])
            point.positions = self.positions[i].tolist()
            trajectory.points.append(point)
        
        data['trajectory'] = load(str(trajectory), Loader=Loader)

        # save curves
        data['curves'] = []

        for bezier in self.beziers:
            data['curves'].append({'control_point0': bezier.control_point0,
                                   'control_point1': bezier.control_point1,
                                   'indices': bezier.indices})

        # finished building dictionary, now save this in the yaml file
        print(dump(data), file=file)
