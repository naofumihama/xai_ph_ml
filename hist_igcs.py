import argparse
import gc
import json
import os

import cohortintgrad as csig
import numpy as np
import pandas as pd
import torch
import tqdm
from moleculetda.read_file import read_data
from pymatgen.core import Structure

device = "cuda" if torch.cuda.is_available() else "cpu"


def construct_hist(
    hist_dir: str = "hist3d",
    feat_df_fn: str = "bins_test800.csv",
    adsorption_df_fn: str = "CO2_2.5bar.adsorption_norm_100A_4000.csv",
):
    with open("map_from_testID_to_cifID.json", mode="r") as f:
        map_js = json.load(f)
    hhist = dict()
    for cif_i in tqdm.tqdm(map_js.values()):
        cif_fn = f"hmof/hMOF-{cif_i}.cif"
        hhist[cif_i] = dict()
        coords, _ = read_data(
            cif_fn, size=100, supercell=True, periodic=False, weighted=False
        )
        df = pd.DataFrame(coords)
        ddf = pd.concat([df[i].apply(lambda x: int(x / 10)) for i in range(3)], axis=1)
        for i in ddf.index:
            ser = ddf.loc[i]
            key = f"{ser[0]}-{ser[1]}-{ser[2]}"
            if key not in hhist[cif_i].keys():
                hhist[cif_i][key] = 0
            hhist[cif_i][key] += 1
    tot_df = pd.DataFrame(hhist).T.sort_index(axis=1)
    tot_df = tot_df.fillna(0)
    tot_df.to_csv(os.path.join(hist_dir, feat_df_fn))

    ads_df = pd.read_csv(adsorption_df_fn).set_index("Unnamed: 0")
    target = np.hstack(
        [
            ads_df.loc[np.where(ads_df["fn"] == f"hmof/hMOF-{i}.cif")[0].item()][
                "adsorption"
            ]
            for i in tot_df.index
        ]
    )
    np.save(os.path.join(hist_dir, "target"), target)


def construct_hist_unit(
    hist_dir: str = "hist3d",
    feat_df_fn: str = "bins_test800_unit20.csv",
    adsorption_df_fn: str = "CO2_2.5bar.adsorption_norm_100A_4000.csv",
    unit: int = 20,
    max_size: int = 20,
    cif_prefix: str = "hmof/hMOF-",
    map_js_fn: str = "map_from_testID_to_cifID.json",
    cutoff: bool = False,
):
    os.makedirs(hist_dir, exist_ok=True)
    with open(map_js_fn, mode="r") as f:
        map_js = json.load(f)
    hhist = dict()
    for cif_i in tqdm.tqdm(map_js.values()):
        cif_fn = f"{cif_prefix}{cif_i}.cif"
        hhist[cif_i] = dict()
        coords = read_data(cif_fn, supercell=False, periodic=False, weighted=False)
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
            if key not in hhist[cif_i].keys():
                hhist[cif_i][key] = 0
            hhist[cif_i][key] += 1
    tot_df = pd.DataFrame(hhist).T.sort_index(axis=1)
    tot_df = tot_df.fillna(0)
    tot_df.to_csv(os.path.join(hist_dir, feat_df_fn))

    ads_df = pd.read_csv(adsorption_df_fn).set_index("Unnamed: 0")
    target = np.hstack(
        [
            ads_df.loc[np.where(ads_df["fn"] == f"{cif_prefix}{i}.cif")[0].item()][
                "adsorption"
            ]
            for i in tot_df.index
        ]
    )
    np.save(os.path.join(hist_dir, f"target_unit{unit}"), target)


def construct_hist_unit_transnegative(
    hist_dir: str = "hist3d",
    output_feat_df_fn: str = "bins_test800_unit20.csv",
    adsorption_df_fn: str = "CO2_2.5bar.adsorption_norm_100A_4000.csv",
    unit: int = 20,
    max_size: int = 20,
    cif_prefix: str = "hmof/hMOF-",
    map_js_fn: str = "map_from_testID_to_cifID.json",
    cutoff: bool = False,
):
    os.makedirs(hist_dir, exist_ok=True)
    with open(map_js_fn, mode="r") as f:
        map_js = json.load(f)
    hhist = dict()
    for cif_i in tqdm.tqdm(map_js.values()):
        cif_fn = f"{cif_prefix}{cif_i}.cif"
        hhist[cif_i] = dict()
        coords = read_data(cif_fn, supercell=False, periodic=False, weighted=False)

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
            if key not in hhist[cif_i].keys():
                hhist[cif_i][key] = 0
            hhist[cif_i][key] += 1
    tot_df = pd.DataFrame(hhist).T.sort_index(axis=1)
    tot_df = tot_df.fillna(0)
    tot_df.to_csv(os.path.join(hist_dir, output_feat_df_fn))

    ads_df = pd.read_csv(adsorption_df_fn).set_index("Unnamed: 0")
    target = np.hstack(
        [
            ads_df.loc[np.where(ads_df["fn"] == f"{cif_prefix}{i}.cif")[0].item()][
                "adsorption"
            ]
            for i in tot_df.index
        ]
    )
    np.save(os.path.join(hist_dir, f"target_unit{unit}_transnegative"), target)


def construct_hist_unit_translational(
    hist_dir: str = "hist3d_translational",
    feat_df_fn: str = "bins_test800_unit10.csv",
    adsorption_df_fn: str = "CO2_2.5bar.adsorption_norm_100A_4000.csv",
    unit: int = 10,
    max_size: int = 20,
):
    os.makedirs(hist_dir, exist_ok=True)
    with open("map_from_testID_to_cifID.json", mode="r") as f:
        map_js = json.load(f)
    hhist = dict()
    for cif_i in tqdm.tqdm(map_js.values()):
        cif_fn = f"hmof/hMOF-{cif_i}.cif"
        s = Structure.from_file(cif_fn)
        s = Structure.from_file(cif_fn)
        unit_cube = s.lattice.matrix
        hhist[cif_i] = dict()
        coords = read_data(cif_fn, supercell=False, periodic=False, weighted=False)
        df = pd.DataFrame(coords)
        df = df + [np.random.random_sample() * unit_cube[i, i] for i in range(3)]

        ddf = pd.concat(
            [
                df[i].apply(
                    lambda x: int((x % unit_cube[i, i]) / unit_cube[i, i] * unit)
                )
                for i in range(3)
            ],
            axis=1,
        )
        for i in ddf.index:
            ser = ddf.loc[i]
            key = f"{ser[0]}-{ser[1]}-{ser[2]}"
            if key not in hhist[cif_i].keys():
                hhist[cif_i][key] = 0
            hhist[cif_i][key] += 1
    tot_df = pd.DataFrame(hhist).T.sort_index(axis=1)
    tot_df = tot_df.fillna(0)
    tot_df.to_csv(os.path.join(hist_dir, feat_df_fn))

    ads_df = pd.read_csv(adsorption_df_fn).set_index("Unnamed: 0")
    target = np.hstack(
        [
            ads_df.loc[np.where(ads_df["fn"] == f"hmof/hMOF-{i}.cif")[0].item()][
                "adsorption"
            ]
            for i in tot_df.index
        ]
    )
    np.save(os.path.join(hist_dir, f"target_unit{unit}_translational"), target)


def igcs_total(
    hist_dir: str = "hist3d",
    feat_df_fn: str = "bins_test800.csv",
    igcs_npz_fn: str = "igcs_direct",
    target_fn: str = "target_unit10.npy",
):
    tot_df = pd.read_csv(os.path.join(hist_dir, feat_df_fn)).set_index("Unnamed: 0")

    target = np.load(os.path.join(hist_dir, target_fn))
    IG = csig.CohortIntGrad(
        torch.Tensor(tot_df.values).to(device),
        torch.Tensor(target).to(device),
        ratio=0.1,
        n_step=50,
    )

    ig, rd = IG.igcs_stack(list(range(tot_df.values.shape[0])))
    np.save(
        os.path.join(hist_dir, f'{igcs_npz_fn}_{feat_df_fn.split(".")[0]}'),
        ig.to("cpu").detach().numpy(),
    )


def igcs_total_single(
    start_id: int,
    repeat_iter: int = 30,
    hist_dir="tobacco_grid",
    feat_df_fn="bins_tobacco_unit50_max_100.csv",
    igcs_npz_fn="igcs_direct_50_100",
    target_fn="target_unit50.npy",
):
    tot_df = pd.read_csv(os.path.join(hist_dir, feat_df_fn)).set_index("Unnamed: 0")

    target = np.load(os.path.join(hist_dir, target_fn))
    IG = csig.CohortIntGrad(
        torch.Tensor(tot_df.values).to(device),
        torch.Tensor(target).to(device),
        ratio=0.01,
        n_step=50,
    )
    del tot_df
    gc.collect()
    for t_id in range(start_id, repeat_iter + start_id):
        print(t_id)
        if os.path.exists(
            os.path.join(
                hist_dir, f'{igcs_npz_fn}_{feat_df_fn.split(".")[0]}_{t_id}.npy'
            )
        ):
            continue
        ig = IG.igcs_single(t_id=t_id)
        np.save(
            os.path.join(hist_dir, f'{igcs_npz_fn}_{feat_df_fn.split(".")[0]}_{t_id}'),
            ig.to("cpu").detach().numpy(),
        )


def igcs_total_single_1000(
    num_data: int = 1000,
    hist_dir="tobacco_grid",
    feat_df_fn="bins_tobacco_unit50_max_100.csv",
    igcs_npz_fn="igcs_direct_50_100",
    target_fn="target_unit50.npy",
):
    tot_df = (
        pd.read_csv(os.path.join(hist_dir, feat_df_fn))
        .set_index("Unnamed: 0")
        .iloc[list(range(num_data))]
    )

    target = np.load(os.path.join(hist_dir, target_fn))[:num_data]

    desc = tot_df.describe()
    tot_df = tot_df.drop(tot_df.columns[np.where(desc.loc["max"] == 0.0)[0]], axis=1)

    IG = csig.CohortIntGrad(
        torch.Tensor(tot_df.values).to(device).half(),
        torch.Tensor(target).to(device).half(),
        ratio=0.1,
        n_step=20,
    )
    del tot_df, target
    gc.collect()
    for t_id in range(num_data):
        print(t_id)
        if os.path.exists(
            os.path.join(
                hist_dir, f'{igcs_npz_fn}_{feat_df_fn.split(".")[0]}_{t_id}.npy'
            )
        ):
            continue
        ig = IG.igcs_single(t_id=t_id)
        np.save(
            os.path.join(hist_dir, f'{igcs_npz_fn}_{feat_df_fn.split(".")[0]}_{t_id}'),
            ig.to("cpu").detach().numpy(),
        )
        del ig
        gc.collect()


def igcs_for_igcs(
    start: int = 0,
    end: int = 5832,
    hist_dir: str = "hist3d",
    feat_df_fn: str = "bins_test800.csv",
    igcs_npz_prefix: str = "igcs_forigcs_feat",
    original_igcs_torch: str = "igcs_25_100A_4000_cube",
):
    tot_df = pd.read_csv(os.path.join(hist_dir, feat_df_fn)).set_index("Unnamed: 0")
    igcs_cube = torch.load(original_igcs_torch).to("cpu").detach().numpy()

    for i in tqdm.tqdm(range(start, end)):
        # print(os.path.join(hist_dir, f"{igcs_npz_prefix}{i}"))
        if os.path.exists(os.path.join(hist_dir, f"{igcs_npz_prefix}{i}.npy")):

            continue
        target = igcs_cube[:, i]
        IG = csig.CohortIntGrad(
            torch.Tensor(tot_df.values).to(device),
            torch.Tensor(target).to(device),
            ratio=0.1,
            n_step=50,
        )
        ig, rd = IG.igcs_stack(list(range(tot_df.values.shape[0])))
        ig = ig.to("cpu").detach().numpy()
        np.save(os.path.join(hist_dir, f"{igcs_npz_prefix}{i}"), ig)
        del ig, IG, target, rd
        gc.collect()


def igcs_for_igcs_strictid(
    id: int = 0,
    hist_dir: str = "hist3d",
    feat_df_fn: str = "bins_test800.csv",
    igcs_npz_prefix: str = "igcs_forigcs_feat",
    original_igcs_torch: str = "igcs_25_100A_4000_cube",
):
    tot_df = pd.read_csv(os.path.join(hist_dir, feat_df_fn)).set_index("Unnamed: 0")
    igcs_cube = torch.load(original_igcs_torch).to("cpu").detach().numpy()

    # print(os.path.join(hist_dir, f"{igcs_npz_prefix}{i}"))
    if os.path.exists(os.path.join(hist_dir, f"{igcs_npz_prefix}{id}.npy")):

        return 0
    target = igcs_cube[:, id]
    IG = csig.CohortIntGrad(
        torch.Tensor(tot_df.values).to(device),
        torch.Tensor(target).to(device),
        ratio=0.1,
        n_step=50,
    )
    ig, rd = IG.igcs_stack(list(range(tot_df.values.shape[0])))
    ig = ig.to("cpu").detach().numpy()
    np.save(os.path.join(hist_dir, f"{igcs_npz_prefix}{id}"), ig)
    del ig, IG, target, rd
    gc.collect()


def divide_igcs_to_atoms(
    cif_id: int = 2289,
    cif_fn: str = "hmof/hMOF-2289.cif",
    igcs_fn: str = "hist3d_re/igcs_direct_bins_test800_unit10.npy",
    feat_df_fn: str = "hist3d_re/bins_test800_unit10.csv",
    unit=10,
    max_size=20,
):
    # s = Structure.from_file(cif_fn)

    coords = read_data(cif_fn, supercell=False, periodic=False, weighted=False)
    df = pd.read_csv(feat_df_fn).set_index("Unnamed: 0")
    igcs_direct = np.load(igcs_fn)
    igcs_df = pd.DataFrame(igcs_direct, columns=df.columns, index=df.index)

    divided_full = dict()
    # not_exist_counter = 0
    for i in range(coords.shape[0]):
        x, y, z = (coords[i] * unit / max_size).astype(int) / unit * max_size
        if f"{x}-{y}-{z}" not in igcs_df.columns:
            # print(f'Column {f"{x}-{y}-{z}"} does not exist!')
            # not_exist_counter += 1
            continue
        contrib = igcs_df.loc[cif_id][f"{x}-{y}-{z}"] / df.loc[cif_id][f"{x}-{y}-{z}"]
        divided_full[i] = contrib
    ratio_df_full = pd.Series(divided_full, index=divided_full.keys())
    # print(
    #    f"{not_exist_counter} atoms are ignored: total #atoms={coords.shape[0]} ({np.round(100* not_exist_counter / coords.shape[0],2)}%)"
    # )
    return ratio_df_full


def num_ignore_check(
    cif_fn: str = "hmof/hMOF-2289.cif",
    cut_df_fn: str = "tobacco_grid/bin_cut_df.csv",
    unit=50,
    max_size=100,
):
    cut_df = pd.read_csv(cut_df_fn).set_index("Unnamed: 0")
    coords = read_data(cif_fn, supercell=False, periodic=False, weighted=False)
    not_exist_counter = 0
    for i in range(coords.shape[0]):
        x, y, z = (coords[i] * unit / max_size).astype(int) / unit * max_size
        if f"{x}-{y}-{z}" not in cut_df.columns:
            not_exist_counter += 1
    print(
        f"{not_exist_counter} atoms are ignored: total #atoms={coords.shape[0]} ({np.round(100* not_exist_counter / coords.shape[0],2)}%)"
    )
    return not_exist_counter, coords.shape[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start", type=int, help="nsample")
    parser.add_argument("-e", "--end", type=int, default=0, help="nsample")
    args = parser.parse_args()
    """igcs_for_igcs(
        start=args.start,
        end=args.end,
        hist_dir="hist3d_re",
        feat_df_fn="bins_test800_unit10.csv",
    )"""
    igcs_total(
        hist_dir="hist3d_translational/",
        feat_df_fn="bins_test800_unit20.csv",
        target_fn="target_unit20_translational.npy",
    )
