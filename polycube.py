import numpy as np, hashlib, numba
from tqdm import tqdm

@numba.jit(nopython=True)
def rotate_array_xy_90(arr, k):
    k = k % 4
    if k == 0:
        return arr.copy()
    
    s0, s1, s2 = arr.shape
    
    if k == 1 or k == 3:
        out = np.zeros((s1, s0, s2), dtype=arr.dtype)
    else:
        out = np.zeros(arr.shape, dtype=arr.dtype)
    
    if k == 1:
        for i in range(s0):
            for j in range(s1):
                for n in range(s2):
                    out[j, s0-1-i, n] = arr[i, j, n]
    elif k == 2:
        for i in range(s0):
            for j in range(s1):
                for n in range(s2):
                    out[s0-1-i, s1-1-j, n] = arr[i, j, n]
    else:
        for i in range(s0):
            for j in range(s1):
                for n in range(s2):
                    out[s1-1-j, i, n] = arr[i, j, n]

    return out


@numba.jit(nopython=True)
def reflect_array(arr, axis):
    result = np.zeros_like(arr)
    x_dim, y_dim, z_dim = arr.shape
    
    if axis == 0:
        for x in range(x_dim):
            for y in range(y_dim):
                for z in range(z_dim):
                    result[x_dim-1-x, y, z] = arr[x, y, z]
    elif axis == 1:
        for x in range(x_dim):
            for y in range(y_dim):
                for z in range(z_dim):
                    result[x, y_dim-1-y, z] = arr[x, y, z]
    else:
        raise ValueError("Invalid axis parameter")

    return result


@numba.jit(nopython=True)
def lexicographic_less(arr1, arr2):
    min_len = min(len(arr1), len(arr2))
    for i in range(min_len):
        if arr1[i] < arr2[i]:
            return True
        elif arr1[i] > arr2[i]:
            return False
    return len(arr1) < len(arr2)


@numba.jit(nopython=True)
def get_canonical_form(binary_tensor):
    filled_coords = np.where(binary_tensor == 1)
    if len(filled_coords[0]) == 0:
        return binary_tensor
    
    min_x, max_x = filled_coords[0].min(), filled_coords[0].max()
    min_y, max_y = filled_coords[1].min(), filled_coords[1].max()
    min_z, max_z = filled_coords[2].min(), filled_coords[2].max()
    cuboid = binary_tensor[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]
    
    min_tensor = None
    min_flat = None
    
    for k in range(4):
        rotated = rotate_array_xy_90(cuboid, k)
        flat = rotated.ravel()
        if min_flat is None or lexicographic_less(flat, min_flat):
            min_flat = flat.copy()
            min_tensor = rotated.copy()
    
    cuboid_xref = reflect_array(cuboid, 0)
    cuboid_yref = reflect_array(cuboid, 1)
    
    for k in range(4):
        rotated = rotate_array_xy_90(cuboid_xref, k)
        flat = rotated.ravel()
        if min_flat is None or lexicographic_less(flat, min_flat):
            min_flat = flat.copy()
            min_tensor = rotated.copy()

    for k in range(4):
        rotated = rotate_array_xy_90(cuboid_yref, k)
        flat = rotated.ravel()
        if min_flat is None or lexicographic_less(flat, min_flat):
            min_flat = flat.copy()
            min_tensor = rotated.copy()
    
    return min_tensor


def hash_tensor(tensor):
    shape_bytes = np.array(tensor.shape, dtype=np.int64).tobytes()
    return hashlib.sha256(shape_bytes + tensor.tobytes()).hexdigest()


class PolycubeSet:
    def __init__(self, pc_hashes=None):
        if pc_hashes is not None:
            assert len(pc_hashes) == len(set(pc_hashes))
            self.unique_forms = set(pc_hashes)
        else:
            self.unique_forms = set()

    def extend(self, other_polycube_set):
        pc_hashes = other_polycube_set.unique_forms
        prev_len = len(self.unique_forms)
        for pc_hash in pc_hashes:
            assert pc_hash not in self.unique_forms
        self.unique_forms.update(pc_hashes)
        assert len(self.unique_forms) == prev_len + len(pc_hashes)

    def is_unique(self, binary_tensor):
        canonical = get_canonical_form(binary_tensor)
        tensor_hash = hash_tensor(canonical)
        return tensor_hash not in self.unique_forms
    
    def add_if_unique(self, binary_tensor):
        canonical = get_canonical_form(binary_tensor)
        tensor_hash = hash_tensor(canonical)
        
        if tensor_hash in self.unique_forms:
            return False, tensor_hash
        else:
            self.unique_forms.add(tensor_hash)
            return True, tensor_hash
    
    def __len__(self):
        return len(self.unique_forms)


@numba.jit(nopython=True)
def flood_fill_3d(array, x, y, z, label):
    if array[x, y, z] != 1:
        return 0
    
    stack = [(x, y, z)]
    array[x, y, z] = label
    size = 0
    
    while len(stack) > 0:
        sx, sy, sz = stack.pop()
        size += 1
        for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
            nx, ny, nz = sx + dx, sy + dy, sz + dz
            cond = (
                0 <= nx < array.shape[0] and 
                0 <= ny < array.shape[1] and 
                0 <= nz < array.shape[2] and 
                array[nx, ny, nz] == 1
            )
            if cond:
                array[nx, ny, nz] = label
                stack.append((nx, ny, nz))

    return size


@numba.jit(nopython=True)
def largest_component(workspace):
    labels = workspace.copy()
        
    current_label = 2
    sizes = []
    
    for x in range(workspace.shape[0]):
        for y in range(workspace.shape[1]):
            for z in range(workspace.shape[2]):
                if workspace[x, y, z] == 1 and labels[x, y, z] == 1:
                    size = flood_fill_3d(labels, x, y, z, current_label)
                    sizes.append((size, current_label))
                    current_label += 1
        
    max_size = 0
    max_label = 0
    for size, label in sizes:
        if size > max_size:
            max_size = size
            max_label = label
            
    result = np.zeros_like(workspace, dtype=np.int8)
    for x in range(workspace.shape[0]):
        for y in range(workspace.shape[1]):
            for z in range(workspace.shape[2]):
                if labels[x, y, z] == max_label:
                    result[x, y, z] = 1

    return result


@numba.jit(nopython=True)
def shift_to_bottom_center(polycube, dims):
    x_idx, y_idx, z_idx = np.where(polycube == 1)
    if len(x_idx) == 0:
        return polycube
        
    com_x, com_y = x_idx.mean(), y_idx.mean()
    min_x, max_x = x_idx.min(), x_idx.max()
    min_y, max_y = y_idx.min(), y_idx.max()
    min_z = z_idx.min()
    
    polycube_center_x = (dims[0] - 1) / 2
    polycube_center_y = (dims[1] - 1) / 2
    
    possible_x_shifts = range(int(-min_x), int(dims[0] - max_x))
    possible_y_shifts = range(int(-min_y), int(dims[1] - max_y))
    
    best_score = float('inf')
    best_shifts = (0, 0)
    
    for x_shift in possible_x_shifts:
        for y_shift in possible_y_shifts:
            new_com_x = com_x + x_shift
            new_com_y = com_y + y_shift
            
            score = (new_com_x - polycube_center_x) ** 2 + (new_com_y - polycube_center_y) ** 2
            
            if score < best_score:
                best_score = score
                best_shifts = (x_shift, y_shift)
    
    x_shift, y_shift = best_shifts
    z_shift = -min_z
    
    new_polycube = np.zeros((int(dims[0]), int(dims[1]), int(dims[2])), dtype=np.int8)
    new_x = x_idx + x_shift
    new_y = y_idx + y_shift
    new_z = z_idx + z_shift
    
    for i in range(len(new_x)):
        new_polycube[new_x[i], new_y[i], new_z[i]] = 1
    return new_polycube


def mutate_polycube(polycube, polycube_set):
    new_polycube = polycube.copy()
    p = 1 / new_polycube.size
    d0, d1, d2 = new_polycube.shape
    while True:
        xor_mask = np.random.rand(d0, d1, d2) < p
        new_polycube = new_polycube ^ xor_mask
        new_polycube = largest_component(new_polycube)
        p *= 1.025
        if new_polycube.sum() > 0:
            uniq, pc_hash = polycube_set.add_if_unique(new_polycube)
            if uniq:
                new_polycube = shift_to_bottom_center(new_polycube, new_polycube.shape)
                return new_polycube, pc_hash


def mutate_polycube_population(polycubes, polycube_set):
    offspring = []
    pc_hashes = []
    for polycube in polycubes:
        new_polycube, pc_hash = mutate_polycube(polycube, polycube_set)
        offspring.append(new_polycube)
        pc_hashes.append(pc_hash)
    return offspring, pc_hashes


@numba.jit(nopython=True)
def xover_xor(g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    result = np.zeros_like(g1, dtype=g1.dtype)
    for i in range(g1.shape[0]):
        for j in range(g1.shape[1]):
            for k in range(g1.shape[2]):
                result[i, j, k] = (g1[i, j, k] != g2[i, j, k])
    return result


def xover_polycube_population(polycubes, polycube_set, max_xover_attempts=100):
    offspring = []
    pc_hashes = []
    parent_idxs = []
    consecutive_failures = 0
    while len(offspring) < len(polycubes):
        i, j = np.random.choice(len(polycubes), 2, replace=False)
        child = xover_xor(polycubes[i], polycubes[j])
        child = largest_component(child)
        if child.sum() == 0:
            consecutive_failures += 1
        else:
            uniq, pc_hash = polycube_set.add_if_unique(child)
            if uniq:
                child = shift_to_bottom_center(child, child.shape)
                offspring.append(child)
                pc_hashes.append(pc_hash)
                parent_idxs.append([i, j])
                consecutive_failures = 0
            else:
                consecutive_failures += 1
        if consecutive_failures >= max_xover_attempts:
            parent_idx = np.random.randint(len(polycubes))
            child, pc_hash = mutate_polycube(polycubes[parent_idx], polycube_set)
            offspring.append(child)
            pc_hashes.append(pc_hash)
            parent_idxs.append([parent_idx])
            consecutive_failures = 0
    return offspring, pc_hashes, parent_idxs


def random_polycube(workspace_dims, p_min, p_max, p_mu, p_sigma, volume_domain, volume_map):
    volume = np.random.choice(volume_domain)
    dims = volume_map[volume][np.random.randint(len(volume_map[volume]))]
    polycube = None
    while True:
        p = min(max(p_min, np.random.normal(p_mu, p_sigma)), p_max)
        polycube = np.zeros(dims, dtype=np.int8)
        polycube[np.random.random(dims) < p] = 1
        if polycube.sum() > 0:
            break
    polycube = largest_component(polycube)
    polycube = shift_to_bottom_center(polycube, workspace_dims)
    return polycube


def random_polycube_population(n, config):
    workspace_dims = config["workspace_dims"]
    
    polycube_set = PolycubeSet()
    polycubes = []
    polycube_hashes = []

    pbar = tqdm(total=n, desc="Generating polycubes")
    current_size = 0

    volume_domain = []
    volume_map = {}
    for i in range(1, config["workspace_dims"][0] + 1):
        for j in range(1, config["workspace_dims"][1] + 1):
            for k in range(1, config["workspace_dims"][2] + 1):
                volume_domain.append(i * j * k)
                if i * j * k not in volume_map:
                    volume_map[i * j * k] = []
                volume_map[i * j * k].append((i, j, k))
    volume_domain = np.array(volume_domain)
    volume_domain = np.unique(volume_domain)
    
    while current_size < n:
        polycube = random_polycube(
            workspace_dims, 
            config["p_min"], 
            config["p_max"], 
            config["p_mu"], 
            config["p_sigma"], 
            volume_domain, 
            volume_map
        )

        valid, pc_hash = polycube_set.add_if_unique(polycube)
        
        if valid:
            polycubes.append(polycube)
            polycube_hashes.append(pc_hash)
            pbar.update(len(polycubes) - current_size)
            current_size = len(polycubes)
    
    pbar.close()
    
    return polycubes, polycube_hashes, polycube_set