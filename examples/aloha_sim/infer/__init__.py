import os
import time
import torch
import wandb
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from aloha_sim.envs.utils import sample_box_pose, sample_insertion_pose
from aloha_sim.envs.visualize_episodes import save_videos
from aloha_sim.envs.sim_env import BOX_POSE

def inference(args):
    from memory_policy.utils import set_seed
    set_seed(args.seed)

    from memory_policy.utils import init_wandb
    init_wandb(args)

    from memory_policy.utils import make_ckpt_dir, save_arguments
    make_ckpt_dir(args)
    save_arguments(args)

    eval_bc(args)


def eval_bc(args):
    # parse arguments
    ckpt_dir = args.ckpt_dir
    state_dim = args.state_dim
    real_robot = args.real_robot
    eval_task = args.eval_task
    action_dim = args.action_dim

    camera_names = args.task.camera_names
    max_timesteps = args.task.inference_timestep

    num_queries = args.num_queries
    temporal_agg = args.infer.temporal_agg
    num_rollouts = args.infer.num_rollouts
    onscreen_render = args.infer.onscreen_render

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, args.ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.deserialize(torch.load(ckpt_path), map_location="cpu")
    print(loading_status)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    
    policy.eval()
    
    # get qpos and action normalizer
    from memory_policy.utils import get_normalizer
    pre_process, post_process = get_normalizer(args)

    # create infer directory
    from memory_policy.utils import make_infer_dir
    infer_dir = make_infer_dir(args)
    
    # create aloha simulation
    from aloha_sim.envs.sim_env import make_sim_env
    env = make_sim_env(eval_task)
    env_max_reward = env.task.max_reward

    query_frequency = 1 if temporal_agg else num_queries

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    episode_returns = []
    highest_rewards = []
    
    for rollout_id in range(num_rollouts):
        rollout_id += 0

        ### set tasks
        if "sim_transfer_cube" in eval_task:
            BOX_POSE[0] = sample_box_pose()
        elif "sim_insertion" in eval_task:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose())

        ts = env.reset()

        ### onscreen render
        if onscreen_render:
            from aloha_sim.envs.utils import on_screen_init
            plt, plt_img = on_screen_init(env)

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, action_dim], device=device)

        # make list and empty tensors to record
        qpos_history_raw = np.zeros((max_timesteps, state_dim))
        image_list = [] # for visualization
        rewards = []

        ### evaluation start
        with torch.inference_mode():
            for t in tqdm(range(max_timesteps), total=max_timesteps):

                ### update onscreen render and wait for DT
                if onscreen_render:
                    from aloha_sim.envs.utils import on_screen_render
                    plt, plt_img = on_screen_render(env, plt_img, plt)

                ### process current observation before stepping
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})

                qpos_numpy = np.array(obs['qpos'])
                qpos_history_raw[t] = qpos_numpy
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
                
                ### query policy
                if t % query_frequency == 0:
                    from memory_policy.utils import get_image
                    curr_image = get_image(ts, camera_names)
                    all_actions = policy(qpos, curr_image)

                from memory_policy.utils import action_temporal_ensemble, action_chunk

                if temporal_agg:
                    raw_action = action_temporal_ensemble(t, all_time_actions, num_queries, all_actions, base_deplay=0)
                else:
                    raw_action = action_chunk(t, all_actions, query_frequency)

                ### post-process actions
                raw_action = raw_action.squeeze(0).detach().cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action[:-2]

                # step the environment
                ts = env.step(target_qpos)

                rewards.append(float(ts.reward))

            if onscreen_render:
                plt.close()

        ### reward
        rewards = np.asarray(rewards, dtype=float)
        episode_return = float(np.nan_to_num(rewards).sum())
        episode_returns.append(episode_return)
        episode_highest_reward = float(np.nanmax(rewards))
        highest_rewards.append(episode_highest_reward)
        print((
            f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, '
            f'{env_max_reward=}, Success: {episode_highest_reward==env_max_reward}'
        ))

    from memory_policy.utils import record_summary
    success_rate, avg_return = record_summary(infer_dir, highest_rewards, env_max_reward, episode_returns, num_rollouts)

    


