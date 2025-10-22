import os

import cv2
import h5py
import numpy as np
import torch

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from .misc import flatten_list, get_norm_stats, find_all_hdf5, BatchSampler, smooth_base_action, preprocess_base_action, postprocess_base_action, load_hdf5


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path_list,
        camera_names,
        norm_stats,
        episode_ids,
        episode_len,
        chunk_size,
        policy_class,
    ):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_path_list = dataset_path_list
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.episode_len = episode_len
        self.chunk_size = chunk_size
        self.cumulative_len = np.cumsum(self.episode_len)
        self.max_episode_len = max(episode_len)
        self.policy_class = policy_class
        self.augment_images = False
        self.transformations = None

        self._action_mean = torch.as_tensor(self.norm_stats["action_mean"], dtype=torch.float32)
        self._action_std = torch.as_tensor(self.norm_stats["action_std"],  dtype=torch.float32)
        self._qpos_mean = torch.as_tensor(self.norm_stats["qpos_mean"],   dtype=torch.float32)
        self._qpos_std = torch.as_tensor(self.norm_stats["qpos_std"],    dtype=torch.float32)

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(
            self.cumulative_len > index
        )  # argmax returns first True index
        start_ts = index - (
            self.cumulative_len[episode_index] - self.episode_len[episode_index]
        )
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts

    def __getitem__(self, index):
        episode_id, start_ts = self._locate_transition(index)
        dataset_path = self.dataset_path_list[episode_id]
        try:
            # print(dataset_path)
            with h5py.File(dataset_path, "r") as root:
                is_sim = root.attrs.get("sim", False)
                compressed = root.attrs.get("compress", False)
                if "/base_action" in root:
                    base_action = root["/base_action"][()]
                    base_action = preprocess_base_action(base_action)
                    action = np.concatenate([root["/action"][()], base_action], axis=-1)
                else:
                    action = root["/action"][()]
                    dummy_base_action = np.zeros([action.shape[0], 2])
                    action = np.concatenate([action, dummy_base_action], axis=-1)
                original_action_shape = action.shape
                episode_len = original_action_shape[0]
                # get observation at start_ts only
                qpos = root["/observations/qpos"][start_ts]
                # qvel = root['/observations/qvel'][start_ts]
                image_dict = dict()
                for cam_name in self.camera_names:
                    image_dict[cam_name] = root[f"/observations/images/{cam_name}"][
                        start_ts
                    ]

                if compressed:
                    for cam_name in image_dict.keys():
                        decompressed_image = cv2.imdecode(image_dict[cam_name], 1)
                        image_dict[cam_name] = np.array(decompressed_image)

                # get all actions after and including start_ts
                if is_sim:
                    action = action[start_ts:]
                    action_len = episode_len - start_ts
                else:
                    # hack, to make timesteps more aligned
                    action = action[max(0, start_ts - 1) :]
                    action_len = episode_len - max(0, start_ts - 1)

            padded_action = np.zeros(
                (self.max_episode_len, original_action_shape[1]),
                dtype=np.float32,
            )
            padded_action[:action_len] = action
            is_pad = np.zeros(self.max_episode_len)
            is_pad[action_len:] = 1

            padded_action = padded_action[: self.chunk_size]
            is_pad = is_pad[: self.chunk_size]

            # new axis for different cameras
            all_cam_images = []
            for cam_name in self.camera_names:
                all_cam_images.append(image_dict[cam_name])
            all_cam_images = np.stack(all_cam_images, axis=0)

            # construct observations
            image_data = torch.from_numpy(all_cam_images)                # K H W C  uint8 예상
            qpos_data = torch.from_numpy(qpos).to(torch.float32)
            action_data = torch.from_numpy(padded_action).to(torch.float32)
            is_pad = torch.from_numpy(is_pad.astype(np.uint8)).to(torch.bool)

            # channel last
            image_data = image_data.permute(0, 3, 1, 2).contiguous()     # K C H W

            # augmentation
            if self.transformations is None:
                print("Initializing transformations")
                original_size = image_data.shape[2:] # H W
                ratio = 0.95
                self.transformations = [
                    transforms.RandomCrop(
                        size=[
                            int(original_size[0] * ratio),
                            int(original_size[1] * ratio),
                        ]
                    ),
                    transforms.Resize(original_size, antialias=True),
                    transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
                    transforms.ColorJitter(
                        brightness=0.3, contrast=0.4, saturation=0.5
                    ),
                ]

            if self.augment_images:
                for transform in self.transformations:
                    image_data = transform(image_data)

            # normalize image and change dtype to float
            image_data = image_data.to(torch.float32).div_(255.0)

            # normalize to mean 0 std 1
            action_data = (action_data - self._action_mean) / self._action_std
            qpos_data   = (qpos_data   - self._qpos_mean)   / self._qpos_std

        except Exception as e:
            # 학습 전체를 중단하지 않고 호출자에게 예외를 전달
            raise RuntimeError(f"__getitem__ failed on {dataset_path}  {e}") from e

        return image_data, qpos_data, action_data, is_pad

def load_data(args):
    """
    목적
      train은 모든 폴더의 train 파트를 합쳐 하나의 DataLoader를 만든다
      validation은 폴더마다 개별 DataLoader를 만든다

    반환
      train_dataloader
      val_dataloaders  폴더 순서와 동일한 리스트  폴더에 val이 없으면 None
      norm_stats
      is_sim  Dataset 플래그가 있으면 전달  없으면 False
    """

    chunk_size = args.decoder.chunk_size

    camera_names = args.task..camera_names
    sample_weight = args.task.sample_weights
    dataset_dir_l = args.task.dataset_dir_list

    batch_size = args.train.batch_size
    train_ratio = args.train.train_ratio
    
    if isinstance(dataset_dir_l, str):
        dataset_dir_l = [dataset_dir_l]

    dataset_path_list_list = []
    for dataset_dir in dataset_dir_l:
        paths = find_all_hdf5(dataset_dir, skip_mirrored_data=False)
        paths = [p for p in paths]
        dataset_path_list_list.append(paths)

    # 2 전역 인덱싱을 위한 길이와 누적합 오프셋 준비
    num_episodes_l = [len(paths) for paths in dataset_path_list_list]
    num_episodes_cumsum = np.cumsum([0] + num_episodes_l)  # 각 폴더의 전역 시작 인덱스

    # 3 전 폴더 파일 평탄화 리스트  Dataset은 전역 인덱스로 접근
    dataset_path_list = flatten_list(dataset_path_list_list)

    # 4 폴더별로 독립 분할을 수행한 뒤
    #   train은 합치고 val은 폴더별로 유지
    train_episode_ids_l = []
    val_episode_ids_l = []
    for folder_idx, num in enumerate(num_episodes_l):
        if num == 0:
            train_episode_ids_l.append(np.array([], dtype=int))
            val_episode_ids_l.append(np.array([], dtype=int))
            continue
        shuffled = np.random.permutation(num)
        split = int(train_ratio * num)
        local_train = shuffled[:split]
        local_val = shuffled[split:]
        offset = num_episodes_cumsum[folder_idx]
        train_episode_ids_l.append(local_train + offset)
        val_episode_ids_l.append(local_val + offset)

    train_episode_ids = np.concatenate(train_episode_ids_l) if len(train_episode_ids_l) else np.array([], dtype=int)

    print(f"\nData from folders  {dataset_dir_l}")
    print(f"train per folder  {[len(x) for x in train_episode_ids_l]}")
    print(f"val per folder    {[len(x) for x in val_episode_ids_l]}\n")

    # 5 통계 및 에피소드 길이 수집
    #   get_norm_stats는 파일 전체를 훑는다  대용량이면 사전 계산 파일을 쓰는 것을 권장
    _, all_episode_len = get_norm_stats(dataset_path_list)
    train_episode_len_l = [[all_episode_len[i] for i in ids] for ids in train_episode_ids_l]
    val_episode_len_l   = [[all_episode_len[i] for i in ids] for ids in val_episode_ids_l]
    train_episode_len   = flatten_list(train_episode_len_l)

    # 정규화 통계는 별도 경로에서 읽을 수도 있다
    if stats_dir_l is None:
        stats_dir_l = dataset_dir_l
    elif isinstance(stats_dir_l, str):
        stats_dir_l = [stats_dir_l]

    # 모든 데이터셋으로 norm stats 계산
    norm_stats, _ = get_norm_stats(
        flatten_list([find_all_hdf5(stats_dir, skip_mirrored_data=False) for stats_dir in stats_dir_l])
    )
    print(f"norm stats from  {stats_dir_l}")

    # 6 Sampler 구성
    #   train은 모든 폴더의 train을 하나로 합친 샘플러
    batch_sampler_train = BatchSampler(
        batch_size,
        train_episode_len_l,   # 폴더별 길이 리스트
        sample_weights,        # 폴더별 확률 가중치  None이면 균등
    )

    # 7 Dataset과 DataLoader 생성
    #   train 하나
    train_dataset = EpisodicDataset(
        dataset_path_list=dataset_path_list,
        camera_names=camera_names,
        norm_stats=norm_stats,
        episode_ids=train_episode_ids,
        episode_len=train_episode_len,
        chunk_size=chunk_size,
    )
    train_num_workers = 8
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler_train,
        pin_memory=True,
        num_workers=train_num_workers,
        prefetch_factor=2,
    )

    #   validation은 폴더별로 여러 개
    val_datasets = []
    val_dataloaders = []
    for folder_idx, ids in enumerate(val_episode_ids_l):
        if len(ids) == 0:
            val_datasets.append(None)
            val_dataloaders.append(None)
            continue
        folder_val_len = [all_episode_len[i] for i in ids]
        folder_len_l = [folder_val_len]     # BatchSampler 입력 형식에 맞춤
        sampler = BatchSampler(batch_size, folder_len_l, None)
        ds = EpisodicDataset(
            dataset_path_list=dataset_path_list,
            camera_names=camera_names,
            norm_stats=norm_stats,
            episode_ids=ids,
            episode_len=folder_val_len,
            chunk_size=chunk_size,
        )
        workers = 8 if ds.augment_images else 2
        dl = DataLoader(
            ds,
            batch_sampler=sampler,
            pin_memory=True,
            num_workers=workers,
            prefetch_factor=2,
        )
        val_datasets.append(ds)
        val_dataloaders.append(dl)

    return train_dataloader, val_dataloaders, norm_stats


def original_load_data_act_plus_plus(
    dataset_dir_l,
    name_filter,
    camera_names,
    batch_size_train,
    batch_size_val,
    chunk_size,
    skip_mirrored_data=False,
    policy_class=None,
    stats_dir_l=None,
    sample_weights=None,
    train_ratio=0.99,
):
    """
    original dataloader comes from act plus plus interbotix
    co-training

    treat first folder dataset as main dataset
    only first dataset consist validation set and all datasets from other folders become trainset.
    """
    if type(dataset_dir_l) == str:
        dataset_dir_l = [dataset_dir_l]
    
    dataset_path_list_list = [find_all_hdf5(dataset_dir, skip_mirrored_data) for dataset_dir in dataset_dir_l]
    
    dataset_path_list = flatten_list(dataset_path_list_list)
    dataset_path_list = [n for n in dataset_path_list if name_filter(n)]
    num_episodes_l = [len(dataset_path_list) for dataset_path_list in dataset_path_list_list]
    num_episodes_cumsum = np.cumsum(num_episodes_l)

    # obtain train test split on dataset_dir_l[0]
    num_episodes_0 = len(dataset_path_list_list[0])
    shuffled_episode_ids_0 = np.random.permutation(num_episodes_0)
    train_episode_ids_0 = shuffled_episode_ids_0[: int(train_ratio * num_episodes_0)]
    val_episode_ids_0 = shuffled_episode_ids_0[int(train_ratio * num_episodes_0) :]
    train_episode_ids_l = [train_episode_ids_0] + [
        np.arange(num_episodes) + num_episodes_cumsum[idx]
        for idx, num_episodes in enumerate(num_episodes_l[1:])
    ]
    val_episode_ids_l = [val_episode_ids_0]
    train_episode_ids = np.concatenate(train_episode_ids_l)
    val_episode_ids = np.concatenate(val_episode_ids_l)
    print(
        f"\n\nData from: {dataset_dir_l}\n- Train on {[len(x) for x in train_episode_ids_l]} episodes\n- Test on {[len(x) for x in val_episode_ids_l]} episodes\n\n"
    )
    _, all_episode_len = get_norm_stats(dataset_path_list)
    train_episode_len_l = [
        [all_episode_len[i] for i in train_episode_ids]
        for train_episode_ids in train_episode_ids_l
    ]
    val_episode_len_l = [
        [all_episode_len[i] for i in val_episode_ids]
        for val_episode_ids in val_episode_ids_l
    ]
    train_episode_len = flatten_list(train_episode_len_l)
    val_episode_len = flatten_list(val_episode_len_l)
    if stats_dir_l is None:
        stats_dir_l = dataset_dir_l
    elif type(stats_dir_l) == str:
        stats_dir_l = [stats_dir_l]
    norm_stats, _ = get_norm_stats(
        flatten_list(
            [find_all_hdf5(stats_dir, skip_mirrored_data) for stats_dir in stats_dir_l]
        )
    )
    print(f"Norm stats from: {stats_dir_l}")

    batch_sampler_train = BatchSampler(
        batch_size_train,
        train_episode_len_l,
        sample_weights,
    )
    batch_sampler_val = BatchSampler(
        batch_size_val,
        val_episode_len_l,
        None,
    )

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(
        dataset_path_list,
        camera_names,
        norm_stats,
        train_episode_ids,
        train_episode_len,
        chunk_size,
        policy_class,
    )
    val_dataset = EpisodicDataset(
        dataset_path_list,
        camera_names,
        norm_stats,
        val_episode_ids,
        val_episode_len,
        chunk_size,
        policy_class,
    )
    train_num_workers = 8
    val_num_workers = 8 if train_dataset.augment_images else 2
    print(
        f"Augment images: {train_dataset.augment_images}, train_num_workers: {train_num_workers}, val_num_workers: {val_num_workers}"
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler_train,
        pin_memory=True,
        num_workers=train_num_workers,
        prefetch_factor=2,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_sampler=batch_sampler_val,
        pin_memory=True,
        num_workers=val_num_workers,
        prefetch_factor=2,
    )

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim
