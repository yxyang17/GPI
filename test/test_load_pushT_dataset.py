import numpy as np
from pusht.datasets import load_episode_dataset
from gpi.database import StateDatabase

dataset_path = "models/pusht_cchi_v7_replay.zarr.zip"
use_relative_action = False
dataset = load_episode_dataset(
            dataset_path,
            use_relative_action=use_relative_action,
        )
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

episode_index = 0

action_normalized = dataset[episode_index]['action']
obs_normalized = dataset[episode_index]['obs']

action = dataset.unnormalize_action(action_normalized)
obs = dataset.unnormalize_obs(obs_normalized)

# Visualize action and observation
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot trajectories on 2D axes
axes[0].plot(action[:, 0], action[:, 1], 'b-', linewidth=2, label='Action Trajectory')
axes[0].scatter(action[0, 0], action[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
axes[0].scatter(action[-1, 0], action[-1, 1], c='red', s=100, marker='x', label='End', zorder=5)
axes[0].set_title('Action Trajectory (X-Y Coordinates)')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].legend(loc='best')
axes[0].grid(True)
axes[0].axis('equal')

# Plot object trajectories
axes[1].plot(obs[:, 0], obs[:, 1], 'r-', linewidth=2, label='Object 1')
axes[1].plot(obs[:, 2], obs[:, 3], 'g-', linewidth=2, label='Object 2')
axes[1].scatter(obs[0, 0], obs[0, 1], c='red', s=100, marker='o', zorder=5)
axes[1].scatter(obs[0, 2], obs[0, 3], c='green', s=100, marker='o', zorder=5)
axes[1].scatter(obs[-1, 0], obs[-1, 1], c='red', s=100, marker='x', zorder=5)
axes[1].scatter(obs[-1, 2], obs[-1, 3], c='green', s=100, marker='x', zorder=5)
axes[1].set_title('Object Trajectories')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
axes[1].legend(loc='best')
axes[1].grid(True)
axes[1].axis('equal')

plt.tight_layout()
plt.show()
