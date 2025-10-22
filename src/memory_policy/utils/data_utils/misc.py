import torch
import numpy as np
import os, cv2, fnmatch, h5py
import matplotlib.pyplot as plt

from aloha.constants import JOINT_NAMES
STATE_NAMES = JOINT_NAMES + ['gripper']


def flatten_list(list_to_flatten):
    return [item for sublist in list_to_flatten for item in sublist]


def get_norm_stats(dataset_path_list):
    all_qpos_data = []
    all_action_data = []
    all_episode_len = []

    for dataset_path in dataset_path_list:
        try:
            with h5py.File(dataset_path, "r") as root:
                qpos = root["/observations/qpos"][()]
                # qvel = root['/observations/qvel'][()]
                if "/base_action" in root:
                    base_action = root["/base_action"][()]
                    base_action = preprocess_base_action(base_action)
                    action = np.concatenate([root["/action"][()], base_action], axis=-1)
                else:
                    action = root["/action"][()]
                    dummy_base_action = np.zeros([action.shape[0], 2])
                    action = np.concatenate([action, dummy_base_action], axis=-1)
        except Exception as e:
            print(f"Error loading {dataset_path} in get_norm_stats")
            print(e)
            quit()
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
        all_episode_len.append(len(qpos))
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0]).float()
    action_std = all_action_data.std(dim=[0]).float()
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0]).float()
    qpos_std = all_qpos_data.std(dim=[0]).float()
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    action_min = all_action_data.min(dim=0).values.float()
    action_max = all_action_data.max(dim=0).values.float()

    eps = 0.0001
    stats = {
        "action_mean": action_mean.numpy(),
        "action_std": action_std.numpy(),
        "action_min": action_min.numpy() - eps,
        "action_max": action_max.numpy() + eps,
        "qpos_mean": qpos_mean.numpy(),
        "qpos_std": qpos_std.numpy(),
        "example_qpos": qpos,
    }

    return stats, all_episode_len


def find_all_hdf5(dataset_dir, skip_mirrored_data):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, "*.hdf5"):
            if "features" in filename:
                continue
            if skip_mirrored_data and "mirror" in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))
    print(f"Found {len(hdf5_files)} hdf5 files")
    return hdf5_files


def BatchSampler(batch_size, episode_len_l, sample_weights):
    sample_probs = (
        np.array(sample_weights) / np.sum(sample_weights)
        if sample_weights is not None
        else None
    )
    sum_dataset_len_l = np.cumsum(
        [0] + [np.sum(episode_len) for episode_len in episode_len_l]
    )
    while True:
        batch = []
        for _ in range(batch_size):
            episode_idx = np.random.choice(len(episode_len_l), p=sample_probs)
            step_idx = np.random.randint(
                sum_dataset_len_l[episode_idx], sum_dataset_len_l[episode_idx + 1]
            )
            batch.append(step_idx)
        yield batch


def smooth_base_action(base_action):
    return np.stack(
        [
            np.convolve(base_action[:, i], np.ones(5) / 5, mode="same")
            for i in range(base_action.shape[1])
        ],
        axis=-1,
    ).astype(np.float32)


def preprocess_base_action(base_action):
    base_action = smooth_base_action(base_action)
    return base_action


def postprocess_base_action(base_action):
    linear_vel, angular_vel = base_action
    linear_vel *= 1.0
    angular_vel *= 1.0
    return np.array([linear_vel, angular_vel])


def visualize_joints(
    qpos_list, command_list, plot_path=None, ylim=None, label_overwrite=None
):
    if label_overwrite:
        label1, label2 = label_overwrite
    else:
        label1, label2 = "State", "Command"

    qpos = np.array(qpos_list)  # ts, dim
    command = np.array(command_list)
    num_ts, num_dim = qpos.shape
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    # plot joint state
    all_names = [name + "_left" for name in STATE_NAMES] + [
        name + "_right" for name in STATE_NAMES
    ]
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(qpos[:, dim_idx], label=label1)
        ax.set_title(f"Joint {dim_idx}: {all_names[dim_idx]}")
        ax.legend()

    # plot arm command
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(command[:, dim_idx], label=label2)
        ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved qpos plot to: {plot_path}")
    plt.close()


def visualize_timestamp(t_list, dataset_path):
    plot_path = dataset_path.replace(".pkl", "_timestamp.png")
    h, w = 4, 10
    fig, axs = plt.subplots(2, 1, figsize=(w, h * 2))
    # process t_list
    t_float = []
    for secs, nsecs in t_list:
        t_float.append(secs + nsecs * 10e-10)
    t_float = np.array(t_float)

    ax = axs[0]
    ax.plot(np.arange(len(t_float)), t_float)
    ax.set_title("Camera frame timestamps")
    ax.set_xlabel("timestep")
    ax.set_ylabel("time (sec)")

    ax = axs[1]
    ax.plot(np.arange(len(t_float) - 1), t_float[:-1] - t_float[1:])
    ax.set_title("dt")
    ax.set_xlabel("timestep")
    ax.set_ylabel("time (sec)")

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved timestamp plot to: {plot_path}")
    plt.close()


def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + ".hdf5")
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        qpos = root["/observations/qpos"][()]
        qvel = root["/observations/qvel"][()]
        action = root["/action"][()]
        image_dict = dict()
        for cam_name in root["/observations/images/"].keys():
            image_dict[cam_name] = root[f"/observations/images/{cam_name}"][()]

    return qpos, qvel, action, image_dict
