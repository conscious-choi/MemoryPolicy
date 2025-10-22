from memory_policy.configs import initialize

from .data_config import REAL_WORLD_CONFIG
from .argparser import real_world_parser
from .infer import inference
from .train import training

def main():
    # get arguments of memory_policy and add the real world env's arguments
    args = initialize(real_world_parser)
    
    # add task configs to arguments
    args.task.camera_names = REAL_WORLD_CONFIG["camera_names"]
    args.task.inference_timestep = REAL_WORLD_CONFIG["inference_timestep"]
    args.task.dataset_dir_list = [task["dataset_dir"] for task in REAL_WORLD_CONFIG["tasks"]]
    
    if not args.inference:
        training(args)
    else:
        inference(args)

if __name__ == "__main__":
    main()