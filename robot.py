import numpy as np, h5py, torch, shutil, os, yaml
from datetime import datetime as dt
from argparse import ArgumentParser
from tqdm import tqdm

from polycube import (
    random_polycube_population,
    mutate_polycube_population,
    xover_polycube_population,
    PolycubeSet,
)


def get_cube_masses(x, y, z, side_length):
    s = side_length
    bot_front_left  = (x,     y,     z     )
    bot_front_right = (x + s, y,     z     )
    bot_back_left   = (x,     y + s, z     )
    bot_back_right  = (x + s, y + s, z     )
    top_front_left  = (x,     y,     z + s)
    top_front_right = (x + s, y,     z + s)
    top_back_left   = (x,     y + s, z + s)
    top_back_right  = (x + s, y + s, z + s)
    
    masses = [
        bot_front_left,
        bot_front_right,
        bot_back_left,
        bot_back_right,
        top_front_left,
        top_front_right,
        top_back_left,
        top_back_right
    ]
    
    return [
        (
            np.round(p[0], 1).item(), 
            np.round(p[1], 1).item(), 
            np.round(p[2], 1).item()
        ) for p in masses
    ]


def get_cube_springs(cube_masses, mass_index):
    springs = []
    for i, mi in enumerate(cube_masses):
        for j, mj in enumerate(cube_masses):
            if (i == j) or not np.isclose(np.array(mi), np.array(mj), atol=1e-6).any():
                continue
            spring = tuple(sorted([mass_index[mi], mass_index[mj]]))
            if spring not in springs:
                springs.append(spring)
    return springs


def build_workspace(dims, side_length):
    masses = []
    mass_index = {}     # (x,y,z) position -> index in masses list
    springs = []
    spring_index = {}   # (m1_idx, m2_idx) tuple -> index in springs list
    spring_lengths = []
    mass_lookup = {}
    spring_lookup = {}

    for x in range(dims[0]):
        for y in range(dims[1]):
            for z in range(dims[2]):
                cube_masses = get_cube_masses(
                    x * side_length, 
                    y * side_length, 
                    z * side_length, 
                    side_length
                )
                for m in cube_masses:
                    if m not in mass_index:
                        mass_index[m] = len(masses)
                        masses.append(m)
                mass_lookup[(x, y, z)] = [mass_index[m] for m in cube_masses]
                cube_springs = get_cube_springs(cube_masses, mass_index)
                for s in cube_springs:
                    if s not in spring_index:
                        spring_index[s] = len(springs)
                        springs.append(s)
                        m1, m2 = masses[s[0]], masses[s[1]]
                        spring_lengths.append(np.linalg.norm(np.array(m1) - np.array(m2)))
                spring_lookup[(x, y, z)] = [spring_index[s] for s in cube_springs]
                
    return masses, springs, spring_lengths, mass_lookup, spring_lookup


def build_robot(polycube, pc_hash, mass_lookup, spring_lookup):
    masses = []
    springs = []
    
    x, y, z = np.where(polycube == 1)
    for i in range(len(x)):
        masses += mass_lookup[(x[i], y[i], z[i])]
        springs += spring_lookup[(x[i], y[i], z[i])]

    masses = np.unique(masses)
    springs = np.unique(springs)

    return {
        "masses": masses,
        "springs": springs,
        "polycube": polycube,
        "pc_hash": pc_hash
    }


def variation(population, polycube_set, mass_lookup, spring_lookup, mutate_frac):
    n_mutate = int(mutate_frac * len(population))
    mutate_idxs = np.random.choice(len(population), size=n_mutate, replace=False)
    xover_idxs = np.setdiff1d(np.arange(len(population)), mutate_idxs)
    to_mutate = [population[i] for i in mutate_idxs]
    to_xover = [population[i] for i in xover_idxs]
    mutated = mutate(to_mutate, polycube_set, mass_lookup, spring_lookup)
    mutated_parent_idxs = [[mi] for mi in mutate_idxs]
    xovered, xover_parent_idxs = xover(to_xover, xover_idxs, polycube_set, mass_lookup, spring_lookup)
    new_population = mutated + xovered
    return new_population, mutated_parent_idxs, xover_parent_idxs


def mutate(population, polycube_set, mass_lookup, spring_lookup):
    offspring = []
    polycubes = [p["polycube"] for p in population]
    mutated_polycubes, mutated_pc_hashes = mutate_polycube_population(polycubes, polycube_set)
    for pc, pc_hash in zip(mutated_polycubes, mutated_pc_hashes):
        offspring.append(build_robot(pc, pc_hash, mass_lookup, spring_lookup))
    return offspring


def xover(population, population_idxs, polycube_set, mass_lookup, spring_lookup):
    offspring = []
    polycubes = [p["polycube"] for p in population]
    xovered_polycubes, xovered_pc_hashes, parent_idxs = xover_polycube_population(polycubes, polycube_set)
    parent_idxs = [[population_idxs[k] for k in pidx] for pidx in parent_idxs]
    for pc, pc_hash in zip(xovered_polycubes, xovered_pc_hashes):
        offspring.append(build_robot(pc, pc_hash, mass_lookup, spring_lookup))
    return offspring, parent_idxs


def random_population(n, config):
    workspace_dims = config["workspace_dims"]
    side_length = config["side_length"]
    
    masses, springs, spring_lengths, \
        mass_lookup, spring_lookup = build_workspace(workspace_dims, side_length)

    polycubes, hashes, polycube_set = random_polycube_population(n, config)
    
    robots = []
    for pc, h in tqdm(zip(polycubes, hashes), desc="Building robots"):
        robots.append(build_robot(pc, h, mass_lookup, spring_lookup))

    workspace = {
        "masses": masses,
        "springs": springs,
        "spring_lengths": spring_lengths,
        "mass_lookup": mass_lookup,
        "spring_lookup": spring_lookup
    }

    return robots, workspace, polycube_set


def save_population(filename, robots, workspace):
    with h5py.File(filename, "w") as f:
        ws = f.create_group("workspace")
        ws.create_dataset("masses", data=workspace["masses"])
        ws.create_dataset("springs", data=workspace["springs"])
        ws.create_dataset("spring_lengths", data=workspace["spring_lengths"])

        ml = ws.create_group("mass_lookup")
        ml.create_dataset("keys", data=list(workspace["mass_lookup"].keys()))
        ml.create_dataset("values", data=list(workspace["mass_lookup"].values()))
        sl = ws.create_group("spring_lookup")
        sl.create_dataset("keys", data=list(workspace["spring_lookup"].keys()))
        sl.create_dataset("values", data=list(workspace["spring_lookup"].values()))

        rs = f.create_group("robots")
        vlen_dt = h5py.vlen_dtype(np.int32)
        str_dt = h5py.string_dtype(encoding="utf-8")
        rs.create_dataset("masses", data=[r["masses"] for r in robots], dtype=vlen_dt)
        rs.create_dataset("springs", data=[r["springs"] for r in robots], dtype=vlen_dt)
        rs.create_dataset("polycubes", data=[r["polycube"] for r in robots], dtype=np.uint8)
        rs.create_dataset("pc_hashes", data=[r["pc_hash"] for r in robots], dtype=str_dt)


def load_population(filename, polycubes_only=False):
    with h5py.File(filename, "r") as f:
        pc_hashes = [h.decode("utf-8") for h in f["robots/pc_hashes"][:]]
        polycube_set = PolycubeSet(pc_hashes)

        if polycubes_only:
            return None, None, polycube_set

        robots = []
        masses = f["robots/masses"][:]
        springs = f["robots/springs"][:]
        polycubes = f["robots/polycubes"][:]

        for i in range(len(masses)):
            robots.append({
                "masses": masses[i],
                "springs": springs[i],
                "polycube": polycubes[i],
                "pc_hash": pc_hashes[i]
            })

        workspace = {
            "masses": [tuple(np.round(m, 1).tolist()) for m in f["workspace/masses"][:]],
            "springs": [tuple(s.tolist()) for s in f["workspace/springs"][:]],
            "spring_lengths": f["workspace/spring_lengths"][:],
        }
        ml_keys = f["workspace/mass_lookup/keys"][:]
        ml_values = f["workspace/mass_lookup/values"][:]
        workspace["mass_lookup"] = {tuple(k.tolist()): v.tolist() for k, v in zip(ml_keys, ml_values)}
        sl_keys = f["workspace/spring_lookup/keys"][:]
        sl_values = f["workspace/spring_lookup/values"][:]
        workspace["spring_lookup"] = {tuple(k.tolist()): v.tolist() for k, v in zip(sl_keys, sl_values)}

    return robots, workspace, polycube_set


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.yml")
    parser.add_argument("--outdir", type=str, default="./dataset")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    outdir = os.path.join(args.outdir, dt.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(outdir, exist_ok=True)

    shutil.copy(args.config, os.path.join(outdir, "config.yml"))

    with open(os.path.join(outdir, "args.txt"), "w") as f:
        f.write(str(args))

    n = config["n_pretrain_robots"] + config["evo_pop_size"]
    robots, workspace, polycube_set = random_population(n, config)
    np.random.shuffle(robots)
    pretrain_robots = robots[:config["n_pretrain_robots"]]
    init_evo_robots = robots[config["n_pretrain_robots"]:]
    pretrain_outfile = os.path.join(outdir, "pretrain_robots.h5")
    init_evo_outfile = os.path.join(outdir, "init_evo_pop.h5")
    save_population(pretrain_outfile, pretrain_robots, workspace)
    save_population(init_evo_outfile, init_evo_robots, workspace)