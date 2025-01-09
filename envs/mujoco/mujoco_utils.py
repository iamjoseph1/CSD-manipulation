from collections import OrderedDict

import akro
import numpy as np
from gym import spaces


def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'), dtype=np.float32)
        high = np.full(observation.shape, float('inf'), dtype=np.float32)
        space = akro.Box(low=low, high=high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


class MujocoTrait:
    # def __init__(self, env):
    #     self.env = env
        
    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = akro.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def render(self,
               mode='human',
               width=100,
               height=100,
               camera_id=None,
               camera_name=None):
        if hasattr(self, 'render_hw') and self.render_hw is not None:
            width = self.render_hw
            height = self.render_hw

        base_env = self.env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        # print("base_env.render :", base_env.render)

        # ############################
        # mode = 'rgb_array'
        # ############################
        
        try:
            # print("try succeed : ", base_env.render(mode))
            return base_env.render(mode, width=width, height=height)
        except TypeError:  # If `mode` is not supported, fallback to calling without arguments
            print("try failed :", base_env.render())
            return base_env.render()
        # return self.env.render(mode=mode)

    def plot_trajectory(self, trajectory, color, ax, target):
        # ############################
        # print("Trajectory shape before squeeze:", trajectory.shape)
        trajectory = trajectory.squeeze()
        # print("Trajectory shape after squeeze:", trajectory.shape)
        # ############################
        if target == 'eef':
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color=color, linewidth=0.7)
        elif target == 'obj':
            ############################
            # print("Trajectory :", trajectory)
            ############################
            ax.plot(trajectory[:,0], trajectory[:, 1], color=color, linewidth=0.7)
        else:
            assert False

    def plot_trajectories(self, trajectories, colors, plot_axis, fm, target):
        ax = fm.ax
        square_axis_limit = 0.0
        for trajectory, color in zip(trajectories, colors):
            trajectory = np.array(trajectory)
            self.plot_trajectory(trajectory, color, ax, target)

            square_axis_limit = max(square_axis_limit, np.max(np.abs(trajectory[:, :2])))
        square_axis_limit = square_axis_limit * 1.2

        if plot_axis == 'free':
            return

        if plot_axis is None:
            plot_axis = [-square_axis_limit, square_axis_limit, -square_axis_limit, square_axis_limit]

        if plot_axis is not None:
            from matplotlib.ticker import MultipleLocator
            if target == 'eef':
                ax.set_xlim(plot_axis[:2])
                ax.set_ylim(plot_axis[2:4])
                ax.set_zlim(plot_axis[4:])
                x_major_locator = MultipleLocator(0.5)  # Set interval for x-axis ticks (e.g., 0.5)
                y_major_locator = MultipleLocator(0.5)  # Set interval for y-axis ticks (e.g., 0.5)
                z_major_locator = MultipleLocator(0.2)  # Set interval for z-axis ticks (e.g., 0.2)
                ax.xaxis.set_major_locator(x_major_locator)
                ax.yaxis.set_major_locator(y_major_locator)
                ax.zaxis.set_major_locator(z_major_locator)
            elif target == 'obj':
                ax.set_xlim(plot_axis[:2])
                ax.set_ylim(plot_axis[2:])
                x_major_locator = MultipleLocator(0.5)  # Set interval for x-axis ticks (e.g., 0.5)
                y_major_locator = MultipleLocator(0.5)  # Set interval for y-axis ticks (e.g., 0.5)
                ax.xaxis.set_major_locator(x_major_locator)
                ax.yaxis.set_major_locator(y_major_locator)
            else:
                assert False
            # ax.set_aspect('equal')
        else:
            ax.axis('scaled')

    def render_trajectories(self, trajectories, colors, plot_axis, fm, target):
        coordinates_trajectories = self._get_coordinates_trajectories(trajectories, target)
        self.plot_trajectories(coordinates_trajectories, colors, plot_axis, fm, target)

    def _get_coordinates_trajectories(self, trajectories, target):
        coordinates_trajectories = []
        for trajectory in trajectories:
            if target == 'eef':
                coordinates_trajectories.append([trajectory['observations'][:,:3]]) ### 3D
            elif target == 'obj':
                coordinates_trajectories.append([trajectory['observations'][:,3:5]]) ### 2D
                # coordinates_trajectories.append([trajectory['observations'][:,4:6]]) ### 2D
            else:
                assert False
            # ############################
            # print("coordinates_trajectories_", target, ":", coordinates_trajectories)
            # ############################
        return coordinates_trajectories

    def calc_eval_metrics(self, trajectories, is_option_trajectories, coord_dims=None):
        eval_metrics = {}

        if coord_dims is not None:
            coords = []
            for traj in trajectories:
                traj1 = traj['observations'][:, 3:5] ### coord_dims-> [4:6]
                traj2 = traj['next_observations'][:, 3:5] ### coord_dims-> [4:6]
                # ############################
                # print("traj1: ", traj1)
                # ############################
                coords.append(traj1)
                coords.append(traj2)
            coords = np.concatenate(coords, axis=0)
            uniq_coords = np.unique(np.round(coords, decimals=2), axis=0)
            eval_metrics.update({
                'MjNumTrajs': len(trajectories),
                'MjAvgTrajLen': len(coords) / len(trajectories) - 1,
                'MjNumCoords': len(coords),
                'MjNumUniqueCoords': len(uniq_coords),
            })

        return eval_metrics