import os
import torch
import pickle
import wandb
import numpy as np

from tqdm import tqdm
from copy import deepcopy
from itertools import repeat

from memory_policy.utils import load_data, compute_dict_mean, set_seed

def training(args):
    train_dataloader, val_dataloader, stats = load_data(args)

    # save dataset stats
    stats_path = os.path.join(args.ckpt_dir, 'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(args, train_dataloader, val_dataloaders)
    
    if args.use_wandb:
        wandb.finish()


def forward_pass(data, policy):
    """Batch tuple -> device 이동 후 policy 호출"""
    image_data, qpos_data, action_data, is_pad = data

    device = next(policy.parameters()).device

    image_data = image_data.to(device, non_blocking=True)
    qpos_data  = qpos_data.to(device, non_blocking=True)
    action_data = action_data.to(device, non_blocking=True)
    is_pad = is_pad.to(device, non_blocking=True)
    
    return policy(qpos_data, image_data, action_data, is_pad)



def train_bc(args, train_dataloader, val_dataloaders):
    # arguments
    seed = args.seed
    ckpt_dir = args.ckpt_dir
    use_wandb = args.use_wandb

    # train arguments
    num_steps = args.train.num_steps
    save_every = args.train.save_every
    validate_every = args.train.validate_every
    resume_ckpt_path = args.train.resume_ckpt_path

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##### 1. Validation Function ###

    def validation(policy, val_dataloaders, min_val_loss, best_ckpt_info, step):
        print('validating')
        policy.eval()

        per_task_val_loader_summaries = []
        with torch.no_grad():
            for i, val_loader in enumerate(val_dataloaders):
                if val_loader is None:
                    continue
                val_dicts = []
                for batch_idx, data in enumerate(val_loader):
                    fdict = forward_pass(data, policy)
                    val_dicts.append(fdict)
                    if batch_idx >= 50:
                        break
                if not val_dicts:
                    continue
                summary = compute_dict_mean(val_dicts)  # -> {k: float}
                per_task_val_loader_summaries.append(summary)
                # 로깅 (폴더별)
                if use_wandb:
                    wandb.log({f"val_task_{i}_{k}": v for k, v in summary.items()}, step=step)

        # 폴더별 요약을 평균 내 전역 val loss 산출
        if per_task_val_loader_summaries:
            keys = per_task_val_loader_summaries[0].keys()
            merged = {k: sum(s[k] for s in per_task_val_loader_summaries) / len(per_task_val_loader_summaries) for k in keys}
        else:
            merged = {"loss": float("inf")}

        epoch_val_loss = merged["loss"]
        if epoch_val_loss < min_val_loss:
            min_val_loss = epoch_val_loss
            best_ckpt_info = (step, min_val_loss, deepcopy(policy.serialize()))

        print(f'Val loss:   {epoch_val_loss:.5f}')
        return min_val_loss, best_ckpt_info
        

    ### 2. Training Function ###

    def gradient_descent(step, policy, optimizer, scaler, train_iter, forward_pass):
        # training
        policy.train()
        optimizer.zero_grad(set_to_none=True)

        data = next(train_iter)
        use_amp = (device.type == "cuda")
        with torch.cuda.amp.autocast(enabled=use_amp):
            forward_dict = forward_pass(data, policy)
            loss = forward_dict["loss"]

        if scaler is not None and use_amp::
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if use_wandb:
            # avoid logging huge tensors
            to_log = {}
            for k, v in forward_dict.items():
                if torch.is_tensor(v) and v.numel() == 1:
                    to_log[k] = v.item()
            to_log["train_loss"] = loss.item()
            wandb.log(to_log, step=step)

        if step % save_every == 0:
            tmp_path = os.path.join(ckpt_dir, f".policy_step_{step}_seed_{seed}.ckpt.tmp")
            final_path = os.path.join(ckpt_dir, f"policy_step_{step}_seed_{seed}.ckpt")
            torch.save(policy.serialize(), tmp_path)
            os.replace(tmp_path, final_path)

    #### 3. Load Policy ###

    policy = make_policy(policy_class, policy_config)

    if resume_ckpt_path is not None:
        state = torch.load(resume_ckpt_path, map_location="cpu")
        loading_status = policy.deserialize(state)
        print(f'Resume policy from: {resume_ckpt_path}, Status: {loading_status}')
    
    policy.to(device)
    optimizer = make_optimizer(policy_class, policy)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    
    ### 4. Set Auxilary ###
    
    min_val_loss = np.inf
    best_ckpt_info = None

    ### 5. Train Start ###
    train_iter = repeater(train_dataloader)
    
    for step in tqdm(range(num_steps+1)):
        # validation
        if step % validate_every == 0:
            min_val_loss, best_ckpt_info = validation(policy, val_dataloaders, min_val_loss, best_ckpt_info, step)

        gradient_descent(step, policy, optimizer, scaler, train_iter, forward_pass)

    ### 6. Train End ###

    # save last checkpoint
    ckpt_path = os.path.join(ckpt_dir, 'policy_last.ckpt')
    torch.save(policy.serialize(), ckpt_path)

    # save best checkpoint
    if best_ckpt_info is not None:
        best_step, min_val_loss, best_state_dict = best_ckpt_info
        ckpt_path = os.path.join(ckpt_dir, f"policy_step_{best_step}_seed_{seed}.ckpt")
        torch.save(best_state_dict, ckpt_path)
        print(f"Training finished\nSeed {seed}, val loss {min_val_loss:.6f} at step {best_step}")
    else:
        print("Training finished without improvement on validation set")

    return best_ckpt_info


def repeater(data_loader):
    epoch = 0
    for loader in repeat(data_loader):
        for data in loader:
            yield data
        print(f'Epoch {epoch} done')
        epoch += 1


