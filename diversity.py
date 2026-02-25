import os, numpy as np
from scipy.spatial.distance import pdist
from argparse import ArgumentParser


def diversity(polycubes):
    """Mean pairwise Hamming distance normalized by genotype size.

    polycubes: (pop_size, L, W, H) binary array for one generation.
    """
    flat = polycubes.reshape(polycubes.shape[0], -1)
    return pdist(flat, metric="hamming").mean()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("polycubes_file", type=str)
    parser.add_argument("--outfile", type=str, default=None)
    args = parser.parse_args()

    pops = np.load(args.polycubes_file)
    divs = np.array([diversity(p) for p in pops])

    outfile = args.outfile or os.path.join(os.path.dirname(args.polycubes_file), "diversity.npy")
    np.save(outfile, divs)
    print(f"Saved {len(divs)} generations of diversity to {outfile}")
