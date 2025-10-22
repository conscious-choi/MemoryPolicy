import os
import torch
import pickle
import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt

def get_image(ts, camera_names):
    curr_images = []
    
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation["images"][cam_name], "h w c -> c h w")
        curr_images.append(curr_image)

    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

    return curr_image

def get_normalizer(args):
    stats_path = os.path.join(args.ckpt_dir, f'dataset_stats.pkl')

    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    return pre_process, post_process

def save_qpos_history_when_real_robot(args, infer_dir, qpos_history_raw):
    # save qpos_history_raw
    np.save(os.path.join(infer_dir, f'qpos.npy'), qpos_history_raw)
    plt.figure(figsize=(10, 20))
    # plot qpos_history_raw for each qpos dim using subplots
    for i in range(args.state_dim):
        plt.subplot(args.state_dim, 1, i+1)
        plt.plot(qpos_history_raw[:, i])
        # remove x axis
        if i != args.state_dim - 1:
            plt.xticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(infer_dir, f'qpos.png'))
    plt.close()


def make_infer_dir(args):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    base = Path(args.ckpt_dir)

    # timezone 명시 — 서울
    now = datetime.now(ZoneInfo("Asia/Seoul"))

    # outputs/2025/10/18/1427 같은 폴더 경로
    folder = base / "inference" / args.ckpt_name / now.strftime("%Y/%m/%d/%H%M")
    folder.mkdir(parents=True, exist_ok=True)

    return folder

def record_summary(infer_dir, highest_rewards, env_max_reward, episode_returns, num_rollouts):
    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result.txt'
    with open(os.path.join(infer_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return