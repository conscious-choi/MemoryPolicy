from memory_policy.configs import initialize

from .data_config import ALOHA_SIM_CONFIG
from .argparser import aloha_sim_parser
from .infer import inference
from .train import training

def main():
    # get arguments of memory_policy and add the real world env's arguments
    args = initialize(aloha_sim_parser)
    
    # add task configs to arguments
    args.task.camera_names = ALOHA_SIM_CONFIG["camera_names"]
    args.task.inference_timestep = ALOHA_SIM_CONFIG["inference_timestep"]
    args.task.dataset_dir_list = [task["dataset_dir"] for task in ALOHA_SIM_CONFIG["tasks"]]
    
    if not args.inference:
        training(args)
    else:
        inference(args)

if __name__ == "__main__":
    main()