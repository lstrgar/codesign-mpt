<div align="center">

# Accelerated co-design of robots </br> through morphological pretraining

## Luke Strgar & Sam Kriegman

### Proceedings of ***The International Conference on Learning Representations (ICLR)***, 2026

</div>

<img width="7388" height="4200" alt="RobotTeaser" src="https://github.com/user-attachments/assets/f70ec71f-3c9b-4d93-a499-d1fd32e6180c" />

<div align="center">
    
### :star: [Project Page](https://lukestrgar.com/codesign-mpt-project-page/)    :star: [OpenReview](https://openreview.net/forum?id=WVliGyFwZv)     :star: [Citation](https://github.com/lstrgar/codesign-mpt/blob/main/README.md#citation)

</div>

## Installation

### Step 1: Conda Env & Taichi

```
conda create --name codesign-mpt
conda activate codesign-mpt
conda install python=3.11.9
pip install taichi==1.7.2
```
### Step 2: PyTorch
- At time of writing, the latest PyTorch version is 2.10.0. 
- This repository assumes a CUDA compatible GPU. 
- With this information, please install the appropriate version of PyTorch for your system.

### Step 3: Pip Packages

```
pip install h5py==3.16.0 pyaml ipykernel numba==0.64.0 tqdm==4.67.3 scipy==1.17.1 matplotlib==3.10.9
```

## Notes

- This code was developed and tested on Ubuntu 24.04.3 using one NVIDIA H100 GPU.
- Experiments reported in the paper involved parallel training and evolution across multiple GPUs. For the purposes of simplicity and wider accessibility, we do not support multi-GPU experiments here. 

## Usage

### Generate Training Data

```
python robot.py --config ./config.yml --seed 17 --outdir dataset
```

### Pretrain Universal Controller

```
python pretrain.py --pop_file dataset/<run>/pretrain_robots.h5 --seed 17
```

### Run Zero-Shot Evolution

```
python evo.py --mode zeroshot \
    --pop_file dataset/<run>/init_evo_pop.h5 \
    --pretrain_pop_file dataset/<run>/pretrain_robots.h5 \
    --model_path pretrain/<run>/ckpts/<step>.pth \
    --seed 17
```

### Run Few-Shot Evolution

```
python evo.py --mode fewshot \
    --pop_file dataset/<run>/init_evo_pop.h5 \
    --pretrain_pop_file dataset/<run>/pretrain_robots.h5 \
    --model_path pretrain/<run>/ckpts/<step>.pth \
    --seed 17
```

### Run the Baseline (Simultaneous Co-Design)

```
python evo.py --mode baseline --pop_file dataset/<run>/init_evo_pop.h5 --seed 17
```

All three modes write to `./results/evo/<mode>/<timestamp>/` by default (override with `--outdir`). Each run produces:

- `losses.npy` — full per-generation × per-robot × per-eval-env loss tensor, shape `(generations+1, pop_size, n_eval_envs)`
- `polycubes.npy` — population morphologies per generation, shape `(generations+1, pop_size, L, W, H)`
- `performance.jsonl` — streaming progress log (mean loss per generation)
- `environments.npy`, `config.yml`, `args.txt`

### Compute Diversity

After an evo run, compute mean pairwise Hamming diversity across generations:

```
python diversity.py results/evo/<mode>/<run>/polycubes.npy
```

Saves `diversity.npy` next to the input by default; override with `--outfile`.

### View Results

`plot_results.ipynb` can be used through experiments to plot loss and diversity trajectories.


## Citation

```
@inproceedings{
  strgar2026accelerated,
  title={Accelerated co-design of robots through morphological pretraining},
  author={Strgar, Luke and Kriegman, Sam},
  booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2026},
  url={https://openreview.net/forum?id=WVliGyFwZv}
}
```
