import torch
import numpy as np

def action_temporal_ensemble(t, all_time_actions, num_queries, all_actions, base_delay):
    all_time_actions[[t], t:t+num_queries-base_delay] = all_actions
    actions_for_curr_step = all_time_actions[:, t]
    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
    actions_for_curr_step = actions_for_curr_step[actions_populated]
    k = 0.01
    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
    exp_weights = exp_weights / exp_weights.sum()
    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
    return raw_action

def action_chunk(t, all_actions, query_frequency):
    raw_action = all_actions[:, t % query_frequency]
    return raw_action