import os
import time
import torch
import wandb
import pickle
import numpy as np
import matplotlib.pyplot as plt

from memory_policy.utils import set_seed
from aloha.constants import FPS, FOLLOWER_GRIPPER_JOINT_OPEN, TASK_CONFIGS

def inference(args):
    set_seed(1)

    from memory_policy.utils import init_wandb
    init_wandb(args)

    from memory_policy.utils import make_ckpt_dir, save_arguments
    make_ckpt_dir(args)
    save_arguments(args)

    eval_bc(args)


def eval_bc(args):
    ckpt_dir = args.ckpt_dir
    state_dim = args.state_dim
    task_name = args.task_name
    real_robot = args.real_robot

    camera_names = args.task.camera_names
    max_timesteps = args.task.inference_timestep

    temporal_agg = args.infer.temporal_agg
    num_rollouts = args.infer.num_rollouts
    onscreen_render = args.infer.onscreen_render
    
    onscreen_cam = 'angle'

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, args.ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.deserialize(torch.load(ckpt_path))
    print(loading_status)

    policy.cuda()
    policy.eval()
    
    from memory_policy.utils import get_normalizer
    pre_process, post_process = get_normalizer(args)

    from memory_policy.utils import init_aloha_node
    node = init_aloha_node(args)

    from memory_policy.utils import make_infer_dir
    infer_dir = make_infer_dir(args)
    
    if real_robot:
        env_max_reward = 0

    query_frequency = args.num_queries

    if temporal_agg:
        query_frequency = 1
        num_queries = args.num_queries

    if real_robot:
        BASE_DELAY = 13
        query_frequency -= BASE_DELAY

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    episode_returns = []
    highest_rewards = []
    
    for rollout_id in range(num_rollouts):
        rollout_id += 0

        ts = env.reset()

        ### onscreen render
        if onscreen_render:
            from memory_policy.utils import on_screen_init
            plt, plt_img = on_screen_init(env)

        # evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, 16]).cuda()

        qpos_history_raw = np.zeros((max_timesteps, state_dim))
        image_list = [] # for visualization
        rewards = []
        with torch.inference_mode():
            start_time = time.time()
            DT = 1 / FPS
            culmulated_delay = 0
            for t in range(max_timesteps):

                ### update onscreen render and wait for DT
                if onscreen_render:
                    from memory_policy.utils import on_screen_render
                    plt, plt_img = on_screen_render(env, plt_img, plt)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})

                qpos_numpy = np.array(obs['qpos'])
                qpos_history_raw[t] = qpos_numpy
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                
                if t % query_frequency == 0:
                    curr_image = get_image(ts, camera_names)

                if t == 0:
                    # warm up
                    for _ in range(10):
                        policy(qpos, curr_image)
                    print('network warm up done')
                    time1 = time.time()

                if t % query_frequency == 0:
                    else:
                        all_actions = policy(qpos, curr_image)
                    if real_robot:
                        all_actions = torch.cat([all_actions[:, :-BASE_DELAY, :-2], all_actions[:, BASE_DELAY:, -2:]], dim=2)

                from memory_policy.utils import action_temporal_ensemble, action_chunk

                if temporal_agg:
                    raw_action = action_temporal_ensemble(t, all_time_actions, num_queries, all_actions, BASE_DELAY)
                else:
                    raw_action = action_chunk(t, all_actions, query_frequency)
                    

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action[:-2]

                base_action = action[-2:]

                # step the environment
                if real_robot:
                    ts = env.step(target_qpos, base_action)
                else:
                    ts = env.step(target_qpos)

                rewards.append(ts.reward)
                
                duration = time.time() - time1
                sleep_time = max(0, DT - duration)
                time.sleep(sleep_time)
                if duration >= DT:
                    culmulated_delay += (duration - DT)
                    print((
                        f'Warning: step duration: {duration:.3f} s at step {t} longer than DT: '
                        f'{DT} s, culmulated delay: {culmulated_delay:.3f} s'
                    ))

            print(f'Avg fps {max_timesteps / (time.time() - start_time)}')
            plt.close()
        
        # if real robot, set gripper to open and save qpos history
        if real_robot:
            from memory_policy.utils import reset_grippers
            reset_grippers(env)

            from memory_policy.utils import save_qpos_history_when_real_robot
            save_qpos_history_when_real_robot(args, infer_dir, qpos_history_raw)

        # if not real robot, append the rewards
        if not real_robot:
            rewards = np.array(rewards)
            episode_return = np.sum(rewards[rewards!=None])
            episode_returns.append(episode_return)
            episode_highest_reward = np.max(rewards)
            highest_rewards.append(episode_highest_reward)
            print((
                f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, '
                f'{env_max_reward=}, Success: {episode_highest_reward==env_max_reward}'
            ))

    # if not real robot, record success rate.
    if not real_robot:
        from memory_policy.utils import record_summary
        success_rate, avg_return = record_summary(infer_dir, highest_rewards, env_max_reward, episode_returns, num_rollouts)

    


