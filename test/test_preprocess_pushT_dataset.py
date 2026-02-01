import numpy as np
import torch
from pusht.datasets import load_episode_dataset
from gpi.database import StateDatabase
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import skimage.transform as st
from matplotlib.patches import Rectangle, Circle
from matplotlib.transforms import Affine2D


def set_plot_limits(ax):
    ax.set_xlim(0, 512)
    ax.set_ylim(512, 0)  # invert y by ordering high->low (better than invert_yaxis)
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0)
    ax.set_autoscale_on(False)
    
def vis_data(dataset, episode_indices: list=[], use_object_centric_frame: bool = False):

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

        if use_object_centric_frame:            
            # convert to global from relative frame per-episode
            obs_relative = obs
            action_relative = action
            pusher_pos_relative = obs[:, :2]
            
            obs_t = torch.from_numpy(obs.astype(np.float32))
            action_t = torch.from_numpy(action.astype(np.float32))
            obs_global = obs.copy()
            obs_global[:,:2] = dataset.relative_state_to_global(obs_t, obs_t[:,:2]).numpy()
            action_global = dataset.relative_action_to_global(obs_t, action_t).numpy()
            pusher_pos_global = dataset.relative_action_to_global(obs_t, obs_t[:,:2]).numpy()
        else:
            obs_global = obs
            action_global = action
            pusher_pos_global = obs[:, :2]

            obs_t = torch.from_numpy(obs.astype(np.float32))
            action_t = torch.from_numpy(action.astype(np.float32))
            obs_relative = obs.copy()
            obs_relative[:,:2] = dataset.global_state_to_relative(obs_t).numpy()
            action_relative = dataset.global_action_to_relative(obs_t, action_t).numpy()
            pusher_pos_relative = dataset.global_action_to_relative(obs_t, obs_t[:,:2]).numpy()

        # action trajectory (global)
        axes[0][0].plot(action_global[:, 0], action_global[:, 1], '-', color=c, linewidth=1.5, label=label_action)
        axes[0][0].scatter(action_global[0, 0], action_global[0, 1], c=[c], s=60, marker='o', zorder=5)
        axes[0][0].scatter(action_global[-1, 0], action_global[-1, 1], c=[c], s=60, marker='x', zorder=5)

        # object trajectories (global) - two objects assumed in obs: cols [0:2] and [2:4]
        plot_line_segments(axes[0][1], obs_global[:, :2], color=c, label=label_obj if i == 0 else None)
        plot_line_segments(axes[0][1], obs_global[:, 2:4], color=c, label=None)

        

        # action relative
        plot_line_segments(axes[1][0], pusher_pos_relative[:, :2], mask = obs_relative[:,-1], color=c, label=None)
        # objects relative
        plot_line_segments(axes[1][1], obs_relative[:, :2], mask = obs_relative[:,-1], color=c, label=None)


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


        
    set_plot_limits(axes[0][0])
    set_plot_limits(axes[0][1])

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


def plot_line_segments(ax, points, color, mask = None, linewidth=2, label=None):
    # build segments for LineCollection and plot with single color (prevents connecting different episodes)
    segments = np.stack([points[:-1], points[1:]], axis=1)
    # lc = LineCollection(segments, colors=[color], linewidths=linewidth, label=label)
    # ax.add_collection(lc)
    # ax.autoscale_view()
    # start / end markers
    ax.scatter(points[0, 0], points[0, 1], c=[color], s=40, marker='o', zorder=5)
    ax.scatter(points[-1, 0], points[-1, 1], c=[color], s=40, marker='x', zorder=5)

    if mask is not None:
        # print(mask)
        # highlight points when point is in contact
        contact_points = points[mask.astype(bool)]
        ax.scatter(contact_points[:, 0], contact_points[:, 1], c='red', s=10, marker='.', zorder=6)
    # highlight point when poin
    

def create_pusht_pts(pts_num: int = 1024, seed: int = 0) -> np.ndarray:
    """
    Sample interior points of a PushT T-shape in the object frame.
    Units match the repo: x in [-60,60], y in [0,120].
    Deterministic if seed is fixed.
    Returns: (N,2)
    """
    rng = np.random.default_rng(seed)

    size_b = 120 * 30   # bottom bar area
    size_t = 90 * 30    # top stem area
    num_b = int(pts_num * size_b / (size_b + size_t))
    num_t = pts_num - num_b

    pts_b = rng.random((num_b, 2))
    pts_b[:, 0] = pts_b[:, 0] * 120 - 60   # [-60, 60]
    pts_b[:, 1] = pts_b[:, 1] * 30         # [0, 30]

    pts_t = rng.random((num_t, 2))
    pts_t[:, 0] = pts_t[:, 0] * 30 - 15    # [-15, 15]
    pts_t[:, 1] = pts_t[:, 1] * 90 + 30    # [30, 120]

    return np.concatenate([pts_b, pts_t], axis=0)



# -----------------------------
# 2D transform helpers
# -----------------------------

def posrad_to_Tmat(pos_xy: np.ndarray, rad: float) -> np.ndarray:
    """World-from-object transform T_W_O (3x3) for 2D pose (x,y,theta)."""
    c, s = float(np.cos(rad)), float(np.sin(rad))
    x, y = float(pos_xy[0]), float(pos_xy[1])
    return np.array([[c, -s, x],
                     [s,  c, y],
                     [0,  0, 1]], dtype=np.float64)

def trans_pt_2d(pt_xy: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply 3x3 homogeneous transform to a 2D point."""
    p = np.array([float(pt_xy[0]), float(pt_xy[1]), 1.0], dtype=np.float64)
    q = T @ p
    return q[:2]



def detect_contact_pusht(
    finger_pos: np.ndarray,  # (2,)
    obj_pos_world: np.ndarray,     # (2,)
    obj_rad: float,
    fin_rad: float,
    margin: float = 3.0,
    pts_num: int = 1024 * 5,
    use_object_centric_frame: bool = False,
    seed: int = 0,
) -> tuple[bool, float]:
    """
    Repo-style PushT contact heuristic.
    Returns (is_contact, min_dist) where min_dist is in object-frame units.
    """
    object_pcd = create_pusht_pts(pts_num, seed)

    if not use_object_centric_frame:
        # finger -> object frame
        T_W_O = posrad_to_Tmat(obj_pos_world, obj_rad)
        T_O_W = np.linalg.inv(T_W_O)
        finger_pos_obj = trans_pt_2d(finger_pos, T_O_W)
    else:
        finger_pos_obj = finger_pos
    # min distance to sampled interior points
    d = object_pcd - finger_pos_obj[None, :]
    min_dist = float(np.min(np.sqrt(np.sum(d * d, axis=1))))

    # threshold
    is_contact = (min_dist < float(fin_rad + margin))
    return is_contact, min_dist


def append_is_contact_to_traj(
    traj: np.ndarray,      # (N,5) [ax,ay,ox,oy,oa]
    fin_rad: float,
    margin: float = 3.0,
    pts_num: int = 1024 * 5,
    use_object_centric_frame: bool = False,
    seed: int = 0,
    dtype=np.float32,
) -> np.ndarray:
    """
    Returns (N,6): [ax, ay, ox, oy, oa, is_contact]
    is_contact stored as 0.0/1.0
    """
    assert traj.ndim == 2 and traj.shape[1] == 5, "traj must be (N,5) [ax,ay,ox,oy,oa]"
    N = traj.shape[0]

    out = np.empty((N, 6), dtype=dtype)
    out[:, :5] = traj.astype(dtype, copy=False)

    for i in range(N):
        ax, ay, ox, oy, oa = traj[i]
        is_c, _ = detect_contact_pusht(
            finger_pos=np.array([ax, ay], dtype=np.float64),
            obj_pos_world=np.array([ox, oy], dtype=np.float64),
            obj_rad=float(oa),
            fin_rad=float(fin_rad),
            margin=float(margin),
            pts_num=int(pts_num),
            use_object_centric_frame=use_object_centric_frame,
            seed=int(seed),
        )
        out[i, 5] = 1.0 if is_c else 0.0

    return out

def visual_pushT_dataset(replay_buffer, start_id=0, interv=3):
    """可视化pusht数据集"""
        
    # 构建物体点云
    pts = create_pusht_pts(pts_num=1024)
    
    for i in range(replay_buffer['keypoint'].shape[0])[start_id::interv]:
        print('='*10, i, '='*10)
        # print('state =', replay_buffer['state'][i])
        # print('action =', replay_buffer['action'][i])
        print('subgoal =', replay_buffer['subgoal'][i])
        print('reward =', replay_buffer['reward'][i])

        # 物体点云
        pos = replay_buffer['state'][i, 2:4]
        rot = replay_buffer['state'][i, 4]
        tf_img_obj = st.AffineTransform(translation=pos, rotation=rot)
        pts_global = tf_img_obj(pts)
        # 手指位置
        fin = Circle(
            (replay_buffer['state'][i, 0], replay_buffer['state'][i, 1]), 15, 
            color='black')
        # 子目标
        if replay_buffer['subgoal'][i, -1] > 0:
            fin_goal = Circle(
                (replay_buffer['subgoal'][i, 0], replay_buffer['subgoal'][i, 1]), 15, 
                color='r')
        # 关键点位置
        # kps_x = replay_buffer['keypoint'][i, :, 0]
        # kps_y = replay_buffer['keypoint'][i, :, 1]

        # 可视化
        fig, ax = plt.subplots()
        ax.scatter(pts_global[:, 0], pts_global[:, 1], c='b')
        # ax.scatter(kps_x, kps_y, c='r')
        ax.add_patch(fin)
        if replay_buffer['subgoal'][i, -1] > 0:
            ax.add_patch(fin_goal)
        ax.set_xlim([0,500])
        ax.set_ylim([0,500])
        ax.set_aspect('equal')
        plt.show()




if __name__ == "__main__":

    dataset_path = "models/pusht_cchi_v7_replay.zarr.zip"
    use_relative_action = True
    use_object_centric_frame = True
    detect_contact = True
    # use_relative_action = False
    # use_object_centric_frame = False
    # detect_contact = False
    dataset = load_episode_dataset(dataset_path, use_relative_action=use_relative_action, use_object_centric_frame=use_object_centric_frame, detect_contact=detect_contact)
    # database = StateDatabase(
            #     dataset,
            #     device=None,
            #     subset_size=None,
            #     batch_size=500_000,
            # )

    # print("Database initialized successfully.")
    # print(f"Total states in database: {len(database)}")

    # print(f"states shape: {database.states.size()}")

    # print(dataset[0]['action'].shape)
    # print(dataset[0]['obs'].shape)

    ########################################
    #  combine and plot multiple episodes
    ########################################
    # episode_indices = [11, 12]  # change to the episodes you want to show
    episode_indices = [i for i in range(len(dataset.episodes))]  # all episodes
    # episode_indices = [i for i in range(10)]  # all episodes

    # print("dataset obs shape", dataset[0]['obs'].shape)
    print("is contact calculated:", dataset[0]['obs'][:,-1])
    vis_data(dataset, episode_indices, use_object_centric_frame=use_object_centric_frame)
    ########################################


    # ########################################
    # #  get contact labels and append to dataset
    # ########################################
    # # traj = dataset.get_episode_data(0)['action']  # (N,5) [ax,ay,ox,oy,oa]
    
    # episode_index = 10
    # traj = dataset[episode_index]['obs']  # (N,5)
    # # unnormalize
    # traj = dataset.unnormalize_obs(traj)
    # traj_with_contact = append_is_contact_to_traj(
    #     traj=traj,
    #     fin_rad=15.0,
    #     margin=3.0,
    #     pts_num=1024 * 5,
    #     use_object_centric_frame=use_object_centric_frame,
    #     seed=0,
    #     dtype=np.float32,
    # )
    # print('traj_with_contact shape =', traj_with_contact.shape)
    
    # # check how many contact points vs non-contact
    # num_contact = int(np.sum(traj_with_contact[:, 5] > 0.5))
    # num_total = traj_with_contact.shape[0]
    # print(f'Contact points: {num_contact} / {num_total} ({100.0 * num_contact / num_total:.2f} %)')
    

    # if not use_object_centric_frame:
    #     # visualize contact points in world frame
    #     print(f'Visualizing contacts for episode {episode_index}, shape = {traj.shape}')
    #     for t in range(0, traj.shape[0],5):
    #         print(traj[t], 'is_contact =', traj_with_contact[t, 5])
    #         visualize_pusht_obs(traj_with_contact[t], pusher_radius=15.0)
    # else:
    #     # visualize contact points in object centric frame
    #     traj_with_contact_objframe = traj_with_contact.copy()
    #     for t in range(0, traj.shape[0]):            
    #         print(traj[t], 'is_contact =', traj_with_contact[t, 5])
    #         target_pos = np.array([256, 256, np.pi / 4]) - traj[t, 2:5]  # ox, oy, oa
    #         visualize_pusht_obs(traj_with_contact[t], pusher_radius=15.0, use_object_centric_frame=use_object_centric_frame, xlim=(-256,256), ylim=(256,-256), target_pos=target_pos)
    # ########################################
    
