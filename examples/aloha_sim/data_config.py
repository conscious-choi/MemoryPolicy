SIM_DATA_DIR = "/data1/dataset/act"

ALOHA_SIM_CONFIGS = {
    'camera_names': ['top'],
    "inference_timestep": 400,
    "dataset_dir": SIM_DATA_DIR,
    'tasks': {
        "sim_transfer_cube_scripted": {
            "dataset_dir": SIM_DATA_DIR + '/sim_transfer_cube_scripted',
            'num_episodes': 50,
        },
        "sim_insertion_scripted": {
            "dataset_dir": SIM_DATA_DIR + '/sim_insertion_scripted',
            'num_episodes': 50,
        }
    },
}