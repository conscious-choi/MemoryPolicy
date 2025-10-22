import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from itertools import chain

def _to_scalar(x):
    # 스칼라로 강제 변환
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, np.generic):  # numpy 스칼라
        return float(x)
    if isinstance(x, np.ndarray):
        return float(x.mean())  # 배열이면 평균으로 스칼라화
    if torch.is_tensor(x):
        with torch.no_grad():
            if x.numel() == 1:
                return x.detach().cpu().item()
            return x.detach().float().mean().cpu().item()
    # 마지막 방어선
    try:
        return float(x)
    except Exception:
        raise TypeError(f"Unsupported value type for mean: {type(x)}")

def compute_dict_mean(epoch_dicts):
    """
    epoch_dicts는 [dict] 또는 [list[dict], dict, ...] 혼재 가능
    반환은 {key: float}이며 GPU 텐서는 모두 CPU float로 변환됨
    """
    # 1단계 평탄화
    def _iter_dicts(seq):
        for item in seq:
            if isinstance(item, (list, tuple)):
                for d in item:
                    yield d
            else:
                yield item

    flat = list(_iter_dicts(epoch_dicts))
    if not flat:
        raise ValueError("epoch_dicts is empty")

    # 2단계 키별 누적 합과 개수 집계
    sums = defaultdict(float)
    counts = defaultdict(int)

    for d in flat:
        for k, v in d.items():
            val = _to_scalar(v)
            sums[k] += val
            counts[k] += 1

    # 3단계 평균
    return {k: (sums[k] / counts[k]) for k in sums.keys()}

def detach_dict(d):
    """
    텐서는 .detach().cpu()로 옮기고 나머지는 그대로 둠
    값이 스칼라 텐서라면 .item()까지 하고 싶으면 아래 한 줄을 바꿔도 됨
    """
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu()
            # 스칼라로 강제하고 싶으면 다음 줄 사용
            # out[k] = v.detach().cpu().item() if v.numel() == 1 else v.detach().cpu()
        elif isinstance(v, np.ndarray):
            out[k] = np.array(v)  # 보존
        else:
            out[k] = v
    return out



def plot_history(
    train_history, validation_history, num_epochs, ckpt_dir, seed, task_name=None
):
    """
    only for train
    """
    if len(train_history) == 0 or len(validation_history) == 0:
        return
    for key in train_history[0]:
        # We only plot if we have consistent logs
        if task_name:
            plot_path = os.path.join(
                ckpt_dir, task_name, f"train_val_{key}_seed_{seed}.png"
            )
            os.makedirs(os.path.join(ckpt_dir, task_name), exist_ok=True)
        else:
            plot_path = os.path.join(ckpt_dir, f"train_val_{key}_seed_{seed}.png")
        
        plt.figure()

        train_values = [
            v.item() if torch.is_tensor(v) else float(v)
            for v in (summary[key] for summary in train_history)
        ]
        val_values = [
            v.item() if torch.is_tensor(v) else float(v)
            for v in (summary[key] for summary in validation_history)
        ]
        
        # train_values = [summary[key].item() for summary in train_history]
        # val_values = [summary[key].item() for summary in validation_history]
        plt.plot(
            np.linspace(0, num_epochs - 1, len(train_history)),
            train_values,
            label="train",
        )
        plt.plot(
            np.linspace(0, num_epochs - 1, len(validation_history)),
            val_values,
            label="validation",
        )
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
        plt.close()
    print(f"Saved plots to {ckpt_dir}")
    

def save_stats(args, stats):
    assert args.mode != "infer"

    stats_path = os.path.join(args.ckpt_dir, "dataset_stats.pkl")

    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)