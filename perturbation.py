# modified from https://github.com/a1k12/moleculetda/blob/main/src/moleculetda/read_file.py

import glob
import os
import pickle
import random
import string
from pathlib import Path
from typing import Tuple, Union

import cohortintgrad as csig
import numpy as np
import pandas as pd
import torch
import tqdm
from moleculetda.construct_pd import construct_pds
from moleculetda.read_file import make_supercell
from moleculetda.vectorize_pds import diagrams_to_arrays, pd_vectorization
from pymatgen.core import Structure

device = "cuda" if torch.cuda.is_available() else "cpu"


def read_and_perturb_cif(
    filename: Union[str, Path], dumpname: Union[str, Path], perturb: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    structure = Structure.from_file(filename)
    structure.perturb(perturb)
    relax = structure.relax()
    with open(dumpname, mode="wb") as f:
        pickle.dump(relax, f)
    return 0


def structure_to_pd(
    dumpname: Union[str, Path],
    supercell_size,
    periodic: bool = False,
):
    with open(dumpname, mode="rb") as f:
        perturbed_structure = pickle.load(f)
    lattice_matrix = perturbed_structure.lattice.matrix
    xyz = perturbed_structure.cart_coords

    coords = make_supercell(xyz, lattice=lattice_matrix, size=supercell_size)
    dgms = construct_pds(coords, periodic=periodic, weights=None)
    arr_dgms = diagrams_to_arrays(dgms)  # convert to array representations
    return arr_dgms


def randomname(n):
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


def dgm_dump(pkl_suffix: str = "perturb3565/*pkl", dgm_dir: str = "perturb3565_dgm"):
    fl = glob.glob(pkl_suffix)
    for fn in tqdm.tqdm(fl):
        if os.path.exists(os.path.join(dgm_dir, f"{fn.split('/')[-1]}_dgm.pkl")):
            continue
        arr_dgms = structure_to_pd(fn, supercell_size=100, periodic=False)
        if "dim1" not in arr_dgms.keys() or "dim2" not in arr_dgms.keys():
            continue
        with open(
            os.path.join(dgm_dir, f"{fn.split('/')[-1]}_dgm.pkl"), mode="wb"
        ) as f:
            pickle.dump(arr_dgms, f)
    return 0


def construct_hist_unit_transnegative(
    original_structure_pkl: str,
    pkl_suffix: str = "perturb3565/*pkl",  # perturbed file lists
    output_feat_df_fn: str = "bins_perturb_unit36_max72.csv",  # target
    unit: int = 36,
    max_size: int = 72,
    cutoff: bool = False,
):
    hhist = dict()
    filelist = [original_structure_pkl] + sorted(glob.glob(pkl_suffix))
    for counter, dumpname in enumerate(tqdm.tqdm(filelist)):
        with open(dumpname, mode="rb") as f:
            perturbed_structure = pickle.load(f)
        hhist[counter] = dict()
        coords = perturbed_structure.cart_coords

        for i in range(3):
            trans = np.min(coords, axis=0)[i]
            if trans < 0:
                # print(coords)
                coords[:, i] -= trans

        df = pd.DataFrame(coords)
        ddf = pd.concat(
            [df[i].apply(lambda x: int(x * unit / max_size)) for i in range(3)], axis=1
        )
        for i in ddf.index:
            ser = ddf.loc[i]
            if cutoff:
                if (ser[0] > unit) or (ser[1] > unit) or (ser[2] > unit):
                    continue
            key = (
                f"{ser[0]/unit*max_size}-{ser[1]/unit*max_size}-{ser[2]/unit*max_size}"
            )
            if key not in hhist[counter].keys():
                hhist[counter][key] = 0
            hhist[counter][key] += 1
    tot_df = pd.DataFrame(hhist).T.sort_index(axis=1)
    tot_df = tot_df.fillna(0)
    tot_df.to_csv(output_feat_df_fn)
    return tot_df


def dgm_to_landscape(
    original_dgm: str,
    dgm_prefix: str,
    output_npy_fn: str,
    px_size=54,
    onemaxB: float = 27.0,
    twomaxB: float = 27.0,
    onemaxP: float = 8.8,
    twomaxP: float = 3.5,
):
    xx = list()
    dgm_stack_dict = dict()
    spec_dict: dict = {
        1: {"maxB": onemaxB, "maxP": onemaxP, "minBD": 0.0},
        2: {"maxB": twomaxB, "maxP": twomaxP, "minBD": 0.0},
    }

    dgm_fl = [original_dgm] + sorted(glob.glob(dgm_prefix))
    for counter, dgm_fn in enumerate(tqdm.tqdm(dgm_fl)):
        with open(dgm_fn, mode="rb") as f:
            arr_dgms = pickle.load(f)
        dgm_stack_dict[counter] = dgm_fn.split("/")[-1]
        dgm_1d = arr_dgms["dim1"]
        dgm_2d = arr_dgms["dim2"]

        vec1 = pd_vectorization(
            dgm_1d,
            spread=0.15,
            weighting="identity",
            pixels=[px_size, px_size],
            specs=spec_dict[1],
        )
        vec2 = pd_vectorization(
            dgm_2d,
            spread=0.15,
            weighting="identity",
            pixels=[px_size, px_size],
            specs=spec_dict[2],
        )
        xx.append(
            np.vstack(
                [vec1.reshape(1, px_size, px_size), vec2.reshape(1, px_size, px_size)]
            ).reshape(1, 2, px_size, px_size)
        )
    x = np.vstack(xx)
    np.save(output_npy_fn, x)


def igcs_total_single(
    feat_df_fn="bins_perturb_unit36_max72.csv",
    target_fn="target_unit50.npy",  # made from x above
    output_igcs_npz_fn="igcs_direct_50_100",
    ratio: float = 0.1,
):
    tot_df = pd.read_csv(feat_df_fn).set_index("Unnamed: 0")
    target = np.load(target_fn)

    # desc = tot_df.describe()
    # tot_df = tot_df.drop(tot_df.columns[np.where(desc.loc["max"] == 0.0)[0]], axis=1)

    IG = csig.CohortIntGrad(
        torch.Tensor(tot_df.values).to(device),
        torch.Tensor(target).to(device),
        ratio=ratio,
        n_step=50,
    )

    t_id = 0
    print(t_id)

    ig = IG.igcs_single(t_id=t_id)
    np.save(
        f'{output_igcs_npz_fn}_{feat_df_fn.split("/")[-1].split(".")[0]}_{t_id}',
        ig.to("cpu").detach().numpy(),
    )
