from expressive_motion_generation.trajectory_planner import TrajectoryPlanner
import json
import os
import numpy as np

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

    def __init__(self, animation_path):
        """
        Initializes and loads animation from the specified path
        """
        file = None
        try:
            file = open(os.path.join(os.getcwd(), animation_path), "r")
        except:
            print("ERROR: File " + os.path.join(os.getcwd(), animation_path) + " could not be opened.")
            return
        
        # read file and compute times and joint goals
        self._load_yaml(file)
        
        # apply to trajectory planner
        self.trajectory_planner = TrajectoryPlanner(np.array(self.times), np.array(self.positions))

        # fill up to make bezier curves possible
        original_indices = self.trajectory_planner.fill_up(20)

        # go through beziers and add them to trajectory
        for i in range(len(self.beziers)):
            curve = self.beziers[i]
            self.trajectory_planner.apply_bezier_at(original_indices[curve.indices[0]], 
                                                    original_indices[curve.indices[1]], 
                                                    curve.control_point0, curve.control_point1)
        
        # finally close file
        file.close()

    def _load_yaml(self, file):
        """
        Load an animation file (YAML format)
        """
        data = load(file, Loader=Loader)

        # load data
        self.name = data["header"]["animation_name"]
        self.joint_names = data["trajectory"]["joint_names"]
        self.frame_id = data["trajectory"]["header"]["frame_id"]

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
