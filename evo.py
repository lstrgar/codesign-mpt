import torch, os, numpy as np, yaml, json, shutil
from datetime import datetime
from argparse import ArgumentParser
from tqdm import tqdm
from time import time
from torch.utils.data import DataLoader

from model import Brain, DecayingCosineAnnealingWarmRestarts
from simulator import Simulator
from robot import load_population, variation
from dataset import EvoTestDataset, EvoTrainDataset, collate
from environment import eval_environments, random_environments


def eval_pop(population, targets, terrains, sim, config):
    dataset = EvoTestDataset(population, targets, terrains)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate)
    losses = np.zeros((len(population), len(targets)), dtype=np.float32)
    for batch in loader:
        batch_losses = sim.test(batch)
        for k in range(len(batch_losses)):
            losses[batch["robot_idxs"][k], batch["env_idxs"][k]] = batch_losses[k]
    return losses


def reset_controller(controller, config, model_path):
    controller.load(model_path, finetune=True)
    controller.optimizer = torch.optim.Adam(
        controller.network.parameters(),
        lr=config["ft_init_lr"],
        betas=(config["beta1"], config["beta2"]),
        weight_decay=config["weight_decay"]
    )
    controller.lr_scheduler = DecayingCosineAnnealingWarmRestarts(
        controller.optimizer,
        initial_cycle_length=config["ft_cycle"],
        cycle_growth_factor=1,
        lr_decay_factor=1,
        min_lr=config["ft_min_lr"]
    )
    controller.step_count = 0

def train_pop(population, offspring, sim, config, envs_per_robot):
    whole_pop = population + offspring
    targets, terrains = random_environments(len(whole_pop) * envs_per_robot, config)
    dataset = EvoTrainDataset(whole_pop, targets, terrains)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate)
    for batch in loader:
        sim.train(batch)


def select_population(population, offspring, pop_losses, off_losses):
    combined = population + offspring
    combined_losses = np.concatenate([pop_losses, off_losses], axis=0)
    top_idxs = np.argsort(combined_losses.mean(axis=1))[:len(population)]
    return [combined[i] for i in top_idxs], combined_losses[top_idxs]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["zeroshot", "fewshot", "baseline"], required=True)
    parser.add_argument("--config", type=str, default="./config.yml")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--pop_file", type=str, required=True)
    parser.add_argument("--pretrain_pop_file", type=str, default=None)
    parser.add_argument("--outdir", type=str, default="./results/evo")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if args.mode in ("zeroshot", "fewshot"):
        if args.model_path is None:
            parser.error(f"--model_path required for mode '{args.mode}'")
        if args.pretrain_pop_file is None:
            parser.error(f"--pretrain_pop_file required for mode '{args.mode}'")

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["batch_size"] = config["evo_pop_size"]
    config["needs_grad"] = args.mode in ("fewshot", "baseline")

    assert config["batch_size"] == config["evo_pop_size"], "batch_size must equal evo_pop_size for per-generation grad-step counts"

    max_iters = config["baseline_max_iters"] if args.mode == "baseline" else config["zeroshot_max_iters"] if args.mode == "zeroshot" else config["fewshot_max_iters"]

    outdir = os.path.join(args.outdir, args.mode, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(outdir, exist_ok=True)
    shutil.copy(args.config, os.path.join(outdir, "config.yml"))
    with open(os.path.join(outdir, "args.txt"), "w") as f:
        f.write(str(args))

    device = torch.device("cuda:0")

    population, workspace, polycube_set = load_population(args.pop_file)
    if args.pretrain_pop_file is not None:
        _, _, pretrain_polycube_set = load_population(args.pretrain_pop_file, polycubes_only=True)
        polycube_set.extend(pretrain_polycube_set)

    targets, terrains = eval_environments(config, os.path.join(outdir, "environments.npy"))

    controller = Brain(config, workspace, None, device)
    if args.model_path is not None:
        controller.load(args.model_path)

    sim = Simulator(config, workspace, controller, args.seed)

    log_path = os.path.join(outdir, "performance.jsonl")

    losses = eval_pop(population, targets, terrains, sim, config)
    with open(log_path, "a") as f:
        f.write(json.dumps({"gen": -1, "mean_loss": float(losses.mean())}) + "\n")

    loss_trajectory = [losses]
    polycubes = [np.stack([r["polycube"] for r in population])]

    pbar = tqdm(range(max_iters))
    for i in pbar:
        t0 = time()
        offspring, mut_parent_idxs, xover_parent_idxs = variation(
            population, polycube_set,
            workspace["mass_lookup"], workspace["spring_lookup"],
            config["mutate_frac"]
        )
        if args.mode == "fewshot":
            reset_controller(controller, config, args.model_path)
            train_pop(population, offspring, sim, config, config["train_per_gen"])
            losses = eval_pop(population, targets, terrains, sim, config)
        elif args.mode == "baseline":
            train_pop(population, offspring, sim, config, config["baseline_train_per_gen"])
            losses = eval_pop(population, targets, terrains, sim, config)
        off_losses = eval_pop(offspring, targets, terrains, sim, config)
        population, losses = select_population(population, offspring, losses, off_losses)
        loss_trajectory.append(losses)
        polycubes.append(np.stack([r["polycube"] for r in population]))
        with open(log_path, "a") as f:
            f.write(json.dumps({"gen": i, "mean_loss": float(losses.mean())}) + "\n")
        pbar.set_description(f"Gen {i} Loss {losses.mean():.4f} Took {time()-t0:.1f}s")

    np.save(os.path.join(outdir, "polycubes.npy"), np.array(polycubes))
    np.save(os.path.join(outdir, "losses.npy"), np.array(loss_trajectory))
    print("Done")