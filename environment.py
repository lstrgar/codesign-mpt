import numpy as np


def eval_targets(config):
    n_angles = config["eval_per_gen"] // 2
    theta = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    theta_offset = np.pi / n_angles
    x = np.zeros(config["eval_per_gen"], dtype=np.float32)
    y = np.zeros(config["eval_per_gen"], dtype=np.float32)
    rmin, rmax = config["eval_target_radii"]
    center_point = config["grid_max"] / 2
    x[:n_angles] = rmax * np.cos(theta) + center_point
    y[:n_angles] = rmax * np.sin(theta) + center_point
    x[n_angles:] = rmin * np.cos(theta + theta_offset) + center_point
    y[n_angles:] = rmin * np.sin(theta + theta_offset) + center_point
    targets = np.stack([x, y], axis=-1).astype(np.float32)
    return targets


def eval_terrains(config):
    size = (config["eval_per_gen"], config["grid_dim"], config["grid_dim"])
    sigma_max = config["terrain_sigma_range"][1]
    n_sigmas = config["eval_per_gen"] // 2
    sigmas = np.array([sigma_max * i / n_sigmas for i in range(1, n_sigmas + 1)])
    sigmas2 = sigmas.copy()
    np.random.shuffle(sigmas)
    np.random.shuffle(sigmas2)
    sigmas = np.concatenate([sigmas, sigmas2])
    sigmas = sigmas.reshape(config["eval_per_gen"], 1, 1)
    z = np.random.normal(
        loc=config["terrain_mu"],
        scale=sigmas,
        size=size
    ).astype(np.float32)
    for i in range(config["eval_per_gen"]):
        z[i] = z[i] - z[i].min() + config["terrain_zmin"]
    return z


def eval_environments(config, outfile=None):
    targets = eval_targets(config)
    terrain = eval_terrains(config)
    if outfile is not None:
        np.save(outfile, {
            "targets": targets,
            "terrain": terrain
        })
    return targets, terrain


def sample_targets(n, config):
    size = (n, 1)
    theta = np.random.uniform(low=0, high=2*np.pi, size=size)
    rmin, rmax = config["pretrain_target_radii"]
    r_squared = np.random.uniform(low=rmin**2, high=rmax**2, size=size)
    r = np.sqrt(r_squared)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    center_point = config["grid_max"] / 2
    x += center_point
    y += center_point
    return np.concatenate([x, y], axis=-1).astype(np.float32)


def sample_terrain(n, config):
    size = (n, config["grid_dim"], config["grid_dim"])
    sigma_min, sigma_max = config["terrain_sigma_range"]
    sigmas = np.random.uniform(low=sigma_min, high=sigma_max, size=n)
    sigmas = sigmas.reshape(n, 1, 1)
    z = np.random.normal(
        loc=config["terrain_mu"],
        scale=sigmas,
        size=size
    ).astype(np.float32)
    for i in range(n):
        z[i] = z[i] - z[i].min() + config["terrain_zmin"]
    return z


def random_environments(n, config, outfile=None):
    targets = sample_targets(n, config)
    terrain = sample_terrain(n, config)
    if outfile is not None:
        np.save(outfile, {
            "targets": targets,
            "terrain": terrain
        })
    return targets, terrain