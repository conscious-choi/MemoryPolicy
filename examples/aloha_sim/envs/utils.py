import numpy as np
from .constants import DT

def sample_box_pose():
    """
    use when simulation evaluation (transfer)
    """
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    """
    use when simulation evaluation (insertion)
    """
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

import matplotlib.pyplot as plt

def on_screen_init(env):
    onscreen_cam = 'angle'
    ax = plt.subplot()
    plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
    plt.ion()
    return plt, plt_img

def on_screen_render(env, plt_img, plt):
    onscreen_cam = 'angle'
    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
    plt_img.set_data(image)
    plt.pause(DT)

    return plt, plt_img