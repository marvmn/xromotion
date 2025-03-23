from expressive_motion_generation.trajectory_planner import TrajectoryPlanner
import json
import os
import numpy as np

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

        for line in file:
            # skip empty lines
            if not line:
                continue

            # every line has the following structure:
            #  <joint positions> # <timestamp> # <control point 1> # <control point 2>
            parts = line.split("#")
            
            if not len(parts) == 4:
                print("WARNING: Line was skipped. ({line})")
                continue

            # read parts
            positions.append(json.loads(parts[0]))
            times.append(float(parts[1]))
            beziers.append([json.loads(parts[2]), json.loads(parts[3])])

        # apply to trajectory planner
        self.trajectory_planner = TrajectoryPlanner(np.array(times), np.array(positions))

        # fill up to make bezier curves possible
        original_indices = self.trajectory_planner.fill_up(20)

        import matplotlib.pyplot as plt
        plt.plot(self.trajectory_planner.times, self.trajectory_planner.positions)
        plt.savefig("anim_shit_orig")

        # go through beziers and add them to trajectory
        for i in range(len(beziers) - 1):
            self.trajectory_planner.apply_bezier_at(original_indices[i], original_indices[i + 1], beziers[i][0], beziers[i][1])
        
        plt.figure()
        plt.plot(self.trajectory_planner.times, self.trajectory_planner.positions)
        plt.savefig("anim_shit")

        self.trajectory_planner.scale_global_speed(1.5)

        # finally close file
        file.close()
