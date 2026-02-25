import yaml, numpy as np, torch, os, shutil
from tqdm import tqdm
from datetime import datetime as dt
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from model import Brain
from simulator import Simulator
from robot import load_population
from environment import random_environments
from dataset import UniversalControllerDataset, collate


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.yml")
    parser.add_argument("--pop_file", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="./results/pretrain")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    outdir = os.path.join(args.outdir, dt.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(outdir, exist_ok=True)

    shutil.copy(args.config, os.path.join(outdir, "config.yml"))
    with open(os.path.join(outdir, "args.txt"), "w") as f:
        f.write(str(args))

    device = torch.device("cuda:0")

    robots, workspace, polycube_set = load_population(args.pop_file)

    targets, terrains = random_environments(len(robots), config, os.path.join(outdir, "pretrain_environments.npy"))

    dataset = UniversalControllerDataset(robots, targets, terrains)

    assert len(robots) % config["pretrain_batch_size"] == 0, f"Number of robots {len(robots)} must be a multiple of batch size {config['pretrain_batch_size']}"
    dataloader = DataLoader(dataset, batch_size=config["pretrain_batch_size"], shuffle=True, collate_fn=collate)

    ckpt_dir = os.path.join(outdir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    controller = Brain(config, workspace, ckpt_dir, device)

    config["batch_size"] = config["pretrain_batch_size"]

    simulator = Simulator(config, workspace, controller, args.seed)

    pbar = tqdm(total=len(dataloader))

    loss_history = []

    for i, batch in enumerate(dataloader):
        losses = simulator.train(batch)
        loss_history.append(losses)
        avg_loss = np.array(loss_history).mean()
        pbar.set_description(f"Loss: {avg_loss:.4f}")
        pbar.update(1)
        if (i+1) % config["pretrain_save_interval"] == 0:
            controller.save()
            np.save(os.path.join(outdir, "losses.npy"), np.array(loss_history))
    pbar.close()

    # save final results
    np.save(os.path.join(outdir, "losses.npy"), np.array(loss_history))
    controller.save()