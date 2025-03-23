from expressive_motion_generation.trajectory_planner import TrajectoryPlanner
import json
import os
import numpy as np

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
        times = []
        positions = []
        beziers = []

        # parse 
        positions_done = False
        for line in file:
            # skip empty lines
            if not line:
                continue

            # check if end of positions was reached
            if line == "CURVES\n":
                positions_done = True
                continue

            parts = line.split("#")
            
            if positions_done:
                # every curve line has the following structure:
                # <index1>:<index2> # <control point 1> # <control point 2>
                if not len(parts) == 3:
                    print("WARNING: Line was skipped.")
                    continue

                # read parts
                curve = BezierCurve(parts[0].split(':'), json.loads(parts[1]), json.loads(parts[2]))
                curve.indices[0] = int(curve.indices[0])
                curve.indices[1] = int(curve.indices[1])
                beziers.append(curve)

            else: 
                # every position line has the following structure:
                # <timestamp> # <joint positions> 
                if not len(parts) == 2:
                    print("WARNING: Line was skipped.")
                    continue

                # read parts
                positions.append(json.loads(parts[1]))
                times.append(float(parts[0]))
        
        # apply to trajectory planner
        self.trajectory_planner = TrajectoryPlanner(np.array(times), np.array(positions))

        # fill up to make bezier curves possible
        original_indices = self.trajectory_planner.fill_up(20)

        import matplotlib.pyplot as plt
        plt.plot(self.trajectory_planner.times, self.trajectory_planner.positions)
        plt.savefig("anim_shit_orig")

        # go through beziers and add them to trajectory
        for i in range(len(beziers)):
            curve = beziers[i]
            self.trajectory_planner.apply_bezier_at(original_indices[curve.indices[0]], 
                                                    original_indices[curve.indices[1]], 
                                                    curve.control_point0, curve.control_point1)
        
        plt.figure()
        plt.plot(self.trajectory_planner.times, self.trajectory_planner.positions)
        plt.savefig("anim_shit")

        self.trajectory_planner.scale_global_speed(1.5)

        # finally close file
        file.close()
