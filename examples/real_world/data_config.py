MAE_DATA_DIR = "/home/data/dataset/mae_act"

REAL_WORLD_CONFIG = {
    "camera_names": ['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
    "inference_timestep": 1250,
    "dataset_dir": MAE_DATA_DIR,
    "tasks": {
        "toasting": {
            "dataset_dir": MAE_DATA_DIR + '/toasting',
            'num_episodes': 72, # no use but just writed.
            'episode_len': 1250, # no use but just writed. (deleted since my collections' dataset timestep would vary)
        },
        "table": {
            "dataset_dir": MAE_DATA_DIR + '/table',
            'num_episodes': 70,
            'episode_len': 1250,
        },
        "dish": {
            "dataset_dir": MAE_DATA_DIR + '/dish',
            'num_episodes': 70,
            'episode_len': 1250,
        },
        "eggplant": {
            "dataset_dir": MAE_DATA_DIR + '/eggplant',
            'num_episodes': 70,
            'episode_len': 1250,
        },
        "boil": {
            "dataset_dir": MAE_DATA_DIR + '/boil',
            'num_episodes': 70,
            'episode_len': 1250,
        },
        "cereal": {
            "dataset_dir": MAE_DATA_DIR + '/cereal',
            'num_episodes': 70,
            'episode_len': 1250,
        },
        "donut": {
            "dataset_dir": MAE_DATA_DIR + '/donut',
            'num_episodes': 70,
            'episode_len': 1250,
        },
    }
}
