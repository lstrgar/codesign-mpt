from torch.utils.data import Dataset
import numpy as np


class EvoTestDataset(Dataset):
    def __init__(self, robots, targets, terrain):
        super().__init__()
        self.robots = robots
        self.targets = targets
        self.terrain = terrain
        assert len(self.targets) == len(self.terrain)
        self.n_envs = len(self.targets)
        self.n_robots = len(self.robots)

    def __len__(self):
        return self.n_envs * self.n_robots
    
    def __getitem__(self, idx):
        robot_idx = idx // self.n_envs
        env_idx = idx % self.n_envs
        robot = self.robots[robot_idx]
        masses = robot["masses"]
        springs = robot["springs"]
        return {
            "target": self.targets[env_idx],
            "terrain": self.terrain[env_idx],
            "n_masses": len(masses),
            "masses": masses,
            "springs": springs,
            "dataset_idx": idx,
            "env_idx": env_idx,
            "robot_idx": robot_idx
        }


class EvoTrainDataset(Dataset):
    def __init__(self, robots, targets, terrain):
        super().__init__()
        self.robots = robots
        self.targets = targets
        self.terrain = terrain
        assert len(self.targets) == len(self.terrain)
        self.n_envs = len(self.targets)
        self.n_robots = len(self.robots)
        assert self.n_envs % self.n_robots == 0
        self.n_samples_per_robot = self.n_envs // self.n_robots

    def __len__(self):
        return self.n_envs
    
    def __getitem__(self, idx):
        robot_idx = idx // self.n_samples_per_robot
        env_idx = idx
        robot = self.robots[robot_idx]
        masses = robot["masses"]
        springs = robot["springs"]
        return {
            "target": self.targets[env_idx],
            "terrain": self.terrain[env_idx],
            "n_masses": len(masses),
            "masses": masses,
            "springs": springs,
            "dataset_idx": idx,
            "env_idx": env_idx,
            "robot_idx": robot_idx
        }


class UniversalControllerDataset(Dataset):
    def __init__(self, robots, targets, terrain):
        super().__init__()
        self.targets = targets
        self.terrain = terrain
        self.robots = robots
        assert len(self.targets) == len(self.terrain)
        assert len(self.targets) == len(self.robots)

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        masses = self.robots[idx]["masses"]
        springs = self.robots[idx]["springs"]
        return {
            "target": self.targets[idx],
            "terrain": self.terrain[idx],
            "n_masses": len(masses),
            "masses": masses,
            "springs": springs,
            "dataset_idx": idx,
            "env_idx": idx,
            "robot_idx": idx
        }


def batch_items(items):
    batched_items = []
    for i, m in enumerate(items):
        b = [i] * len(m)
        batched_items.append(np.stack([b, m], axis=1))
    batched_items = np.vstack(batched_items)
    return batched_items


def collate(batch):
    terrain = np.stack([item["terrain"] for item in batch])
    targets = np.stack([item["target"] for item in batch])
    masses = batch_items([item["masses"] for item in batch])
    springs = batch_items([item["springs"] for item in batch])
    n_masses = np.array([item["n_masses"] for item in batch])
    dataset_idxs = np.array([item["dataset_idx"] for item in batch])
    robot_idxs = np.array([item["robot_idx"] for item in batch])
    env_idxs = np.array([item["env_idx"] for item in batch])
    return {
        "terrain": terrain.astype(np.float32),
        "targets": targets.astype(np.float32),
        "masses": masses.astype(np.int32),
        "springs": springs.astype(np.int32),
        "n_masses": n_masses.astype(np.int32),
        "dataset_idxs": dataset_idxs.astype(np.int32),
        "robot_idxs": robot_idxs.astype(np.int32),
        "env_idxs": env_idxs.astype(np.int32)
    }