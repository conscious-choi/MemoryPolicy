import torch
import wandb
import numpy as np
import random, json

def set_seed(args):
    """Set the seed for reproducibility across multiple libraries."""
    seed = args.seed

    # Python, NumPy
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 선택 사항  CUDA 결정론 강화
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 결정론 알고리즘 강제  일부 연산은 느려질 수 있음
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

def init_wandb(args):
    if not args.infer:
        cfg_dict = args.as_dict()
        wandb.init(
            project=args.wandb.project,
            reinit=True,
            entity=args.wandb.entity,
            name=args.exp_name,
            config=cfg_dict,  # 시작부터 설정 기록
        )

def make_ckpt_dir(args):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    base = Path(args.ckpt_dir)

    # timezone 명시 — 서울
    now = datetime.now(ZoneInfo("Asia/Seoul"))

    # outputs/2025/10/18/1427 같은 폴더 경로
    folder = base / now.strftime("%Y/%m/%d/%H%M")
    folder.mkdir(parents=True, exist_ok=True)

    args.ckpt_dir = folder

def save_arguments(args):
    with open(f"{args.ckpt_dir}/arguments.json", "w", encoding="utf-8") as f:
        json.dump(args.as_dict(), f, indent=2, ensure_ascii=False)


