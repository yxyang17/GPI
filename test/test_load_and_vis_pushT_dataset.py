import numpy as np
import torch
from pusht.datasets import load_episode_dataset
from gpi.database import StateDatabase
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt

dataset_path = "models/pusht_cchi_v7_replay.zarr.zip"
use_relative_action = True
use_object_centric_frame = True
dataset = load_episode_dataset(dataset_path, use_relative_action=use_relative_action, use_object_centric_frame=use_object_centric_frame)

# database = StateDatabase(
        #     dataset,
        #     device=None,
        #     subset_size=None,
        #     batch_size=500_000,
        # )

# print("Database initialized successfully.")
# print(f"Total states in database: {len(database)}")

# print(f"states shape: {database.states.size()}")

print(dataset[0]['action'].shape)
print(dataset[0]['obs'].shape)

# combine and plot multiple episodes
episode_indices = [11, 12]  # change to the episodes you want to show
episode_indices = [i for i in range(len(dataset.episodes))]  # all episodes
episode_indices = [i for i in range(20)]  # all episodes

def plot_line_segments(ax, points, color, linewidth=2, label=None):
    # build segments for LineCollection and plot with single color (prevents connecting different episodes)
    segments = np.stack([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, colors=[color], linewidths=linewidth, label=label)
    ax.add_collection(lc)
    ax.autoscale_view()
    # start / end markers
    ax.scatter(points[0, 0], points[0, 1], c=[color], s=40, marker='o', zorder=5)
    ax.scatter(points[-1, 0], points[-1, 1], c=[color], s=40, marker='x', zorder=5)
    

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(episode_indices))))

for i, ep in enumerate(episode_indices):
    action_norm = dataset[ep]['action']
    obs_norm = dataset[ep]['obs']

    action = dataset.unnormalize_action(action_norm)
    obs = dataset.unnormalize_obs(obs_norm)

    c = colors[i % len(colors)]
    label_action = f'Action ep {ep}'
    label_obj = f'Objects ep {ep}'

    # action trajectory (global)
    axes[0][0].plot(action[:, 0], action[:, 1], '-', color=c, linewidth=1.5, label=label_action)
    axes[0][0].scatter(action[0, 0], action[0, 1], c=[c], s=60, marker='o', zorder=5)
    axes[0][0].scatter(action[-1, 0], action[-1, 1], c=[c], s=60, marker='x', zorder=5)

    # object trajectories (global) - two objects assumed in obs: cols [0:2] and [2:4]
    plot_line_segments(axes[0][1], obs[:, :2], color=c, label=label_obj if i == 0 else None)
    plot_line_segments(axes[0][1], obs[:, 2:4], color=c, label=None)

    # convert to relative frame per-episode
    obs_t = torch.from_numpy(obs.astype(np.float32))
    action_t = torch.from_numpy(action.astype(np.float32))
    obs_relative = dataset.global_state_to_relative(obs_t).numpy()
    action_relative = dataset.global_action_to_relative(obs_t, action_t).numpy()
    pusher_pos_relative = dataset.global_action_to_relative(obs_t, obs_t[:,:2]).numpy()

    # action relative
    plot_line_segments(axes[1][0], pusher_pos_relative[:, :2], color=c, label=None)

    # objects relative
    plot_line_segments(axes[1][1], obs_relative[:, :2], color=c, label=None)


# formatting axes
axes[0][0].set_title('Action Trajectories (Global)')
axes[0][0].set_xlabel('X'); axes[0][0].set_ylabel('Y'); axes[0][0].grid(True); axes[0][0].axis('equal')
axes[0][0].legend(loc='best', fontsize='small')

axes[0][1].set_title('Object Trajectories (Global)')
axes[0][1].set_xlabel('X'); axes[0][1].set_ylabel('Y'); axes[0][1].grid(True); axes[0][1].axis('equal')

axes[1][0].set_title('Action Trajectories (Relative)')
axes[1][0].set_xlabel('X'); axes[1][0].set_ylabel('Y'); axes[1][0].grid(True); axes[1][0].axis('equal')

axes[1][1].set_title('Object Trajectories (Relative)')
axes[1][1].set_xlabel('X'); axes[1][1].set_ylabel('Y'); axes[1][1].grid(True); axes[1][1].axis('equal')


def set_limits(ax):
    ax.set_xlim(0, 512)
    ax.set_ylim(512, 0)  # invert y by ordering high->low (better than invert_yaxis)
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0)
    ax.set_autoscale_on(False)
    
set_limits(axes[0][0])
set_limits(axes[0][1])

axes[1][0].set_xlim(-256, 256)
axes[1][0].set_ylim(256, -256)  # invert y by ordering high->low (better than invert_yaxis)
axes[1][0].set_aspect("equal", adjustable="box")
axes[1][0].margins(0)
axes[1][0].set_autoscale_on(False)


axes[1][1].set_xlim(-256, 256)
axes[1][1].set_ylim(256, -256)  # invert y by ordering high->low (better than invert_yaxis)
axes[1][1].set_aspect("equal", adjustable="box")
axes[1][1].margins(0)
axes[1][1].set_autoscale_on(False)

plt.tight_layout()
plt.show()
