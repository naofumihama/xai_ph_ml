import gc
import itertools
import json
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from moleculetda.vectorize_pds import pd_vectorization
from sklearn.model_selection import train_test_split

import data_construct


def dataload(
    npz_fn="CO2_2.5bar.adsorption_norm_100A_4000_cube.npy",
    df_fn="CO2_2.5bar.adsorption_norm_100A_4000.csv",
):
    xx = np.load(npz_fn)
    df = pd.read_csv(df_fn).set_index("Unnamed: 0")
    y = df["adsorption"].values
    del df
    gc.collect()
    train_x, test_x, train_y, test_y = train_test_split(
        xx, y, test_size=0.2, random_state=1018
    )
    del xx
    gc.collect()
    train_x = train_x.reshape(train_x.shape[0], -1)
    test_x = test_x.reshape(test_x.shape[0], -1)
    return train_x, test_x, train_y, test_y


def modelload(model_pkl_fn: str = "model2.5_rf_norm_100A_4000_cube.pkl"):
    with open(model_pkl_fn, mode="rb") as f:
        rg = pickle.load(f)
    return rg


def plot_original_csig(
    data_id: int,
    cif_id: int,
    x,
    igcs_fn: str = "igcs_25_100A_4000_cube",
    dgm_dir: str = "hmof_co2_100A_25bar_cube",
    px_size: int = 54,
    onemaxB: float = 27.0,
    twomaxB: float = 27.0,
    onemaxP: float = 8.8,
    twomaxP: float = 3.5,
    prefix: str = "test_igcs",
    db_prefix: str = "hMOF-",
):

    igcs = torch.load(igcs_fn).to("cpu").detach().numpy()
    # _, x, _, y = dataload()
    with open(
        os.path.join(dgm_dir, f"{db_prefix}{cif_id}.json_dgm.pkl"), mode="rb"
    ) as f:
        dgm = pickle.load(f)

    fig, ax = plt.subplots(3, 2, figsize=(13, 21))
    xt1, yt1 = data_construct.ticks(
        px_size=px_size, max_birth=onemaxB, max_persistence=onemaxP
    )
    xt2, yt2 = data_construct.ticks(
        px_size=px_size, max_birth=twomaxB, max_persistence=twomaxP
    )
    tick_num = len(xt1) - 1
    vmax = np.max(abs(igcs[data_id]))

    for i in range(2):
        ax[0, i].scatter(dgm[f"dim{i+1}"]["birth"], dgm[f"dim{i+1}"]["death"], s=3)
        ax[0, i].plot(
            [0, np.max(dgm[f"dim{i+1}"]["death"])],
            [0, np.max(dgm[f"dim{i+1}"]["death"])],
            c="0.6",
        )
        ax[0, i].set_xlabel("Birth (angstrom)", fontsize=18)
        ax[0, i].set_ylabel("Death (angstrom)", fontsize=18)
        ax[0, i].tick_params(axis="x", labelsize=18)
        ax[0, i].tick_params(axis="y", labelsize=18)

        im0_content = ax[1, i].imshow(
            x[data_id].reshape(2, px_size, px_size)[i, ::-1],
            cmap="Purples",
            norm=matplotlib.colors.SymLogNorm(
                linthresh=1, vmin=0, vmax=np.max(x[data_id])
            ),
        )
        cb = fig.colorbar(im0_content, ax=ax[1, i])
        cb.ax.tick_params(labelsize=18)

        im1_content = ax[2, i].imshow(
            igcs[data_id].reshape(2, px_size, px_size)[i, ::-1],
            cmap="seismic",
            vmin=-vmax,
            vmax=vmax,
        )
        ccb = fig.colorbar(im1_content, ax=ax[2, i])
        ccb.ax.tick_params(labelsize=18)

    for i, j in itertools.product(range(1, 3), range(2)):
        ax[i, j].invert_yaxis()
        ax[i, j].set_xticks(
            range(0, px_size + int(px_size / tick_num), int(px_size / tick_num))
        )
        ax[i, j].set_yticks(
            range(0, px_size + int(px_size / tick_num), int(px_size / tick_num))
        )
        ax[i, j].set_xlabel("Birth (angstrom)", fontsize=18)
        ax[i, j].set_ylabel("Persistence (angstrom)", fontsize=18)

    for i in range(1, 3):
        ax[i, 0].set_xticklabels(xt1, fontsize=18, rotation=270)
        ax[i, 0].set_yticklabels(yt1, fontsize=18)
        ax[i, 1].set_xticklabels(xt2, fontsize=18, rotation=270)
        ax[i, 1].set_yticklabels(yt2, fontsize=18)

    for j in range(2):
        ax[0, j].set_title(f"Original Persistent Diagram of H{j+1}", fontsize=18)
        ax[1, j].set_title(f"Original H{j+1}", fontsize=18)
        ax[2, j].set_title(f"IGCS for H{j+1}", fontsize=18)
    # fig.suptitle(f"{db_prefix}{cif_id}.cif, {mode} value = {y[data_id]}", fontsize=18)
    # fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.tight_layout()
    fig.savefig(f"{prefix}_{cif_id}.png")
    plt.close()
    plt.clf()


def plot_original_csig_fromsingle(
    data_id: int,
    cif_id: int,
    x,
    igcs_fn: str = "igcs_25_100A_4000_cube",
    dgm_dir: str = "tobacco_1.0_CH4_100bar/",
    px_size: int = 54,
    onemaxB: float = 27.0,
    twomaxB: float = 27.0,
    onemaxP: float = 8.8,
    twomaxP: float = 3.5,
    db_prefix: str = "tobmof-",
    prefix: str = "igcs_tobacco_plot",
):

    igcs = torch.load(igcs_fn).to("cpu").detach().numpy()
    # _, x, _, y = dataload()
    with open(
        os.path.join(dgm_dir, f"{db_prefix}{cif_id}.json_dgm.pkl"), mode="rb"
    ) as f:
        dgm = pickle.load(f)

    fig, ax = plt.subplots(3, 2, figsize=(13, 21))
    xt1, yt1 = data_construct.ticks(
        px_size=px_size, max_birth=onemaxB, max_persistence=onemaxP
    )
    xt2, yt2 = data_construct.ticks(
        px_size=px_size, max_birth=twomaxB, max_persistence=twomaxP
    )
    tick_num = len(xt1) - 1
    vmax = np.max(abs(igcs))

    for i in range(2):
        ax[0, i].scatter(dgm[f"dim{i+1}"]["birth"], dgm[f"dim{i+1}"]["death"], s=3)
        ax[0, i].plot(
            [0, np.max(dgm[f"dim{i+1}"]["death"])],
            [0, np.max(dgm[f"dim{i+1}"]["death"])],
            c="0.6",
        )
        ax[0, i].set_xlabel("Birth (angstrom)", fontsize=18)
        ax[0, i].set_ylabel("Death (angstrom)", fontsize=18)
        ax[0, i].tick_params(axis="x", labelsize=18)
        ax[0, i].tick_params(axis="y", labelsize=18)

        im0_content = ax[1, i].imshow(
            x[data_id].reshape(2, px_size, px_size)[i, ::-1],
            cmap="Purples",
            norm=matplotlib.colors.SymLogNorm(
                linthresh=1, vmin=0, vmax=np.max(x[data_id])
            ),
        )
        cb = fig.colorbar(im0_content, ax=ax[1, i])
        cb.ax.tick_params(labelsize=18)

        im1_content = ax[2, i].imshow(
            igcs.reshape(2, px_size, px_size)[i, ::-1],
            cmap="seismic",
            vmin=-vmax,
            vmax=vmax,
        )
        ccb = fig.colorbar(im1_content, ax=ax[2, i])
        ccb.ax.tick_params(labelsize=18)

    for i, j in itertools.product(range(1, 3), range(2)):
        ax[i, j].invert_yaxis()
        ax[i, j].set_xticks(
            range(0, px_size + int(px_size / tick_num), int(px_size / tick_num))
        )
        ax[i, j].set_yticks(
            range(0, px_size + int(px_size / tick_num), int(px_size / tick_num))
        )
        ax[i, j].set_xlabel("Birth (angstrom)", fontsize=18)
        ax[i, j].set_ylabel("Persistence (angstrom)", fontsize=18)

    for i in range(1, 3):
        ax[i, 0].set_xticklabels(xt1, fontsize=18, rotation=270)
        ax[i, 0].set_yticklabels(yt1, fontsize=18)
        ax[i, 1].set_xticklabels(xt2, fontsize=18, rotation=270)
        ax[i, 1].set_yticklabels(yt2, fontsize=18)

    for j in range(2):
        ax[0, j].set_title(f"Original Persistent Diagram of H{j+1}", fontsize=18)
        ax[1, j].set_title(f"Original H{j+1}", fontsize=18)
        ax[2, j].set_title(f"IGCS for H{j+1}", fontsize=18)
    # fig.suptitle(f"{db_prefix}{cif_id}.cif, {mode} value = {y[data_id]}", fontsize=18)
    # fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.tight_layout()
    fig.savefig(f"{prefix}_{cif_id}.png")
    plt.close()
    plt.clf()


def plot_original_csig_short(
    data_id: int,
    cif_id: int,
    x,
    igcs_fn: str = "igcs_25_100A_4000_cube",
    px_size: int = 54,
    onemaxB: float = 27.0,
    twomaxB: float = 27.0,
    onemaxP: float = 8.8,
    twomaxP: float = 3.5,
    prefix: str = "test_igcs",
):

    igcs = torch.load(igcs_fn).to("cpu").detach().numpy()
    # _, x, _, y = dataload()
    # with open(
    #    os.path.join(dgm_dir, f"{db_prefix}{cif_id}.json_dgm.pkl"), mode="rb"
    # ) as f:
    #    dgm = pickle.load(f)

    fig, ax = plt.subplots(2, 2, figsize=(13, 13))
    xt1, yt1 = data_construct.ticks(
        px_size=px_size, max_birth=onemaxB, max_persistence=onemaxP
    )
    xt2, yt2 = data_construct.ticks(
        px_size=px_size, max_birth=twomaxB, max_persistence=twomaxP
    )
    tick_num = len(xt1) - 1
    vmax = np.max(abs(igcs[data_id]))

    for i in range(2):
        # ax[0, i].scatter(dgm[f"dim{i+1}"]["birth"], dgm[f"dim{i+1}"]["death"], s=3)
        # ax[0, i].plot(
        #    [0, np.max(dgm[f"dim{i+1}"]["death"])],
        #    [0, np.max(dgm[f"dim{i+1}"]["death"])],
        #    c="0.6",
        # )
        # ax[0, i].set_xlabel("Birth (angstrom)", fontsize=18)
        # ax[0, i].set_ylabel("Death (angstrom)", fontsize=18)
        # ax[0, i].tick_params(axis="x", labelsize=18)
        # ax[0, i].tick_params(axis="y", labelsize=18)

        im0_content = ax[0, i].imshow(
            x[data_id].reshape(2, px_size, px_size)[i, ::-1],
            cmap="Purples",
            norm=matplotlib.colors.SymLogNorm(
                linthresh=1, vmin=0, vmax=np.max(x[data_id])
            ),
        )
        cb = fig.colorbar(im0_content, ax=ax[0, i])
        cb.ax.tick_params(labelsize=18)

        im1_content = ax[1, i].imshow(
            igcs[data_id].reshape(2, px_size, px_size)[i, ::-1],
            cmap="seismic",
            vmin=-vmax,
            vmax=vmax,
        )
        ccb = fig.colorbar(im1_content, ax=ax[1, i])
        ccb.ax.tick_params(labelsize=18)

    for i, j in itertools.product(range(2), range(2)):
        ax[i, j].invert_yaxis()
        ax[i, j].set_xticks(
            range(0, px_size + int(px_size / tick_num), int(px_size / tick_num))
        )
        ax[i, j].set_yticks(
            range(0, px_size + int(px_size / tick_num), int(px_size / tick_num))
        )
        ax[i, j].set_xlabel("Birth (angstrom)", fontsize=18)
        ax[i, j].set_ylabel("Persistence (angstrom)", fontsize=18)

    for i in range(2):
        ax[i, 0].set_xticklabels(xt1, fontsize=18, rotation=270)
        ax[i, 0].set_yticklabels(yt1, fontsize=18)
        ax[i, 1].set_xticklabels(xt2, fontsize=18, rotation=270)
        ax[i, 1].set_yticklabels(yt2, fontsize=18)

    for j in range(2):
        # ax[0, j].set_title(f"Original Persistent Diagram of H{j+1}", fontsize=18)
        ax[0, j].set_title(f"Original H{j+1}", fontsize=18)
        ax[1, j].set_title(f"IGCS for H{j+1}", fontsize=18)
    # fig.suptitle(f"{db_prefix}{cif_id}.cif, {mode} value = {y[data_id]}", fontsize=18)
    # fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.tight_layout()
    fig.savefig(f"{prefix}_{cif_id}.png")
    plt.close()
    plt.clf()


def plot_original_csig_short_fromsingle(
    data_id: int,
    cif_id: int,
    x,
    igcs_fn: str = "igcs_25_100A_4000_cube",
    px_size: int = 54,
    onemaxB: float = 27.0,
    twomaxB: float = 27.0,
    onemaxP: float = 8.8,
    twomaxP: float = 3.5,
    prefix: str = "igcs_tobacco_plot",
):

    igcs = torch.load(igcs_fn).to("cpu").detach().numpy()
    # _, x, _, y = dataload()
    # with open(
    #    os.path.join(dgm_dir, f"{db_prefix}{cif_id}.json_dgm.pkl"), mode="rb"
    # ) as f:
    #    dgm = pickle.load(f)

    fig, ax = plt.subplots(2, 2, figsize=(13, 13))
    xt1, yt1 = data_construct.ticks(
        px_size=px_size, max_birth=onemaxB, max_persistence=onemaxP
    )
    xt2, yt2 = data_construct.ticks(
        px_size=px_size, max_birth=twomaxB, max_persistence=twomaxP
    )
    tick_num = len(xt1) - 1
    vmax = np.max(abs(igcs))

    for i in range(2):

        im0_content = ax[0, i].imshow(
            x[data_id].reshape(2, px_size, px_size)[i, ::-1],
            cmap="Purples",
            norm=matplotlib.colors.SymLogNorm(
                linthresh=1, vmin=0, vmax=np.max(x[data_id])
            ),
        )
        cb = fig.colorbar(im0_content, ax=ax[0, i])
        cb.ax.tick_params(labelsize=18)

        im1_content = ax[1, i].imshow(
            igcs.reshape(2, px_size, px_size)[i, ::-1],
            cmap="seismic",
            vmin=-vmax,
            vmax=vmax,
        )
        ccb = fig.colorbar(im1_content, ax=ax[1, i])
        ccb.ax.tick_params(labelsize=18)

    for i, j in itertools.product(range(2), range(2)):
        ax[i, j].invert_yaxis()
        ax[i, j].set_xticks(
            range(0, px_size + int(px_size / tick_num), int(px_size / tick_num))
        )
        ax[i, j].set_yticks(
            range(0, px_size + int(px_size / tick_num), int(px_size / tick_num))
        )
        ax[i, j].set_xlabel("Birth (angstrom)", fontsize=18)
        ax[i, j].set_ylabel("Persistence (angstrom)", fontsize=18)

    for i in range(2):
        ax[i, 0].set_xticklabels(xt1, fontsize=18, rotation=270)
        ax[i, 0].set_yticklabels(yt1, fontsize=18)
        ax[i, 1].set_xticklabels(xt2, fontsize=18, rotation=270)
        ax[i, 1].set_yticklabels(yt2, fontsize=18)

    for j in range(2):
        # ax[0, j].set_title(f"Original Persistent Diagram of H{j+1}", fontsize=18)
        ax[0, j].set_title(f"Original H{j+1}", fontsize=18)
        ax[1, j].set_title(f"IGCS for H{j+1}", fontsize=18)
    # fig.suptitle(f"{db_prefix}{cif_id}.cif, {mode} value = {y[data_id]}", fontsize=18)
    # fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.tight_layout()
    fig.savefig(f"{prefix}_{cif_id}.png")
    plt.close()
    plt.clf()


def plot_only_pd(
    cif_id: int,
    # x: np.ndarray,
    # target: float,
    dgm_dir: str = "tobacco_1.0_CH4_100bar",
    adsorption_df_fn: str = "tobacco_1.0_CH4_100bar_sc.csv",
    px_size: float = 54,
    onemaxB: float = 27.0,
    twomaxB: float = 27.0,
    onemaxP: float = 8.8,
    twomaxP: float = 3.5,
    db_prefix: str = "tobmof-",
    prefix: str = "landscape",
):

    with open(
        os.path.join(dgm_dir, f"{db_prefix}{cif_id}.json_dgm.pkl"), mode="rb"
    ) as f:
        dgm = pickle.load(f)

    spec_dict = {
        1: {"maxB": onemaxB, "maxP": onemaxP, "minBD": 0.0},
        2: {"maxB": twomaxB, "maxP": twomaxP, "minBD": 0.0},
    }
    dgm_1d = dgm["dim1"]
    dgm_2d = dgm["dim2"]

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
    x = np.vstack(
        [vec1.reshape(1, px_size, px_size), vec2.reshape(1, px_size, px_size)]
    )

    ads_df = pd.read_csv(adsorption_df_fn).set_index("Unnamed: 0")
    target = ads_df[ads_df["fn"].apply(lambda x: x.split("-")[1]) == f"{cif_id}.cif"][
        "adsorption"
    ].item()

    fig, ax = plt.subplots(2, 2, figsize=(13, 13))
    xt1, yt1 = data_construct.ticks(
        px_size=px_size, max_birth=onemaxB, max_persistence=onemaxP
    )
    xt2, yt2 = data_construct.ticks(
        px_size=px_size, max_birth=twomaxB, max_persistence=twomaxP
    )
    tick_num = len(xt1) - 1

    for i in range(2):
        ax[0, i].scatter(dgm[f"dim{i+1}"]["birth"], dgm[f"dim{i+1}"]["death"], s=3)
        ax[0, i].plot(
            [0, np.max(dgm[f"dim{i+1}"]["death"])],
            [0, np.max(dgm[f"dim{i+1}"]["death"])],
            c="0.6",
        )
        ax[0, i].set_xlabel("Birth (angstrom)")
        ax[0, i].set_ylabel("Death (angstrom)")

        im0_content = ax[1, i].imshow(
            x.reshape(2, px_size, px_size)[i, ::-1],
            cmap="Purples",
            norm=matplotlib.colors.SymLogNorm(linthresh=1, vmin=0, vmax=np.max(x)),
        )
        fig.colorbar(im0_content, ax=ax[1, i])

        ax[1, i].invert_yaxis()
        ax[1, i].set_xticks(
            range(0, px_size + int(px_size / tick_num), int(px_size / tick_num))
        )
        ax[1, i].set_yticks(
            range(0, px_size + int(px_size / tick_num), int(px_size / tick_num))
        )
        ax[1, i].set_xlabel("Birth (angstrom)")
        ax[1, i].set_ylabel("Persistence (angstrom)")

        ax[0, i].set_title(f"Original Persistent Diagram of H{i+1}")
        ax[1, i].set_title(f"Original H{i+1}")

    ax[1, 0].set_xticklabels(xt1)
    ax[1, 0].set_yticklabels(yt1)
    ax[1, 1].set_xticklabels(xt2)
    ax[1, 1].set_yticklabels(yt2)

    fig.suptitle(f"{db_prefix}{cif_id}.cif, Annotated value = {target}", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{prefix}_{cif_id}.png")
    plt.close()
    plt.clf()


def cif_from_x_id(
    data_id: int,
    df_fn: str = "CO2_2.5bar.adsorption_norm_100A_4000.csv",
    npz_fn: str = "CO2_2.5bar.adsorption_norm_100A_4000_cube.npy",
):
    _, x, _, y = dataload(npz_fn=npz_fn, df_fn=df_fn)
    df = pd.read_csv(df_fn).set_index("Unnamed: 0")
    npz = np.load(npz_fn)
    id = np.where((npz.reshape(-1, x.shape[1]) == x[data_id]).all(axis=1))[0].item()
    ser = df.loc[id]
    cif_id = int(ser["fn"].split("-")[-1].split(".")[0])
    return cif_id, ser


def map_js_dump(
    df_fn: str = "CO2_2.5bar.adsorption_norm_100A_4000.csv",
    output_fn: str = "map_from_testID_to_cifID.json",
    check_npz_fn: str = "CO2_2.5bar.adsorption_norm_100A_4000_cube.npy",
):
    df = pd.read_csv(df_fn).set_index("Unnamed: 0")
    _, xx, _, yy = train_test_split(
        df["id"], df["adsorption"], test_size=0.2, random_state=1018
    )
    _, castx, _, casty = dataload(npz_fn=check_npz_fn, df_fn=df_fn)
    map_js = dict()
    for i in range(xx.shape[0]):
        target = df.loc[xx].iloc[i]
        target_cif_id = target["fn"].split("-")[1].split(".")[0]
        if casty[i] != target["adsorption"]:
            print(i, target, yy.iloc[i], casty[i])
        map_js[str(i)] = int(target_cif_id)
    with open(output_fn, mode="w") as f:
        json.dump(map_js, f)


def calc_where_feat(
    feat_id: int,
    px_size: float = 54,
    onemaxB: float = 27.0,
    twomaxB: float = 27.0,
    onemaxP: float = 8.8,
    twomaxP: float = 3.5,
):
    dim = int(feat_id / (px_size**2))
    res = feat_id % (px_size**2)
    raw = int(res / px_size)
    col = res % px_size

    real_dim = dim + 1
    if dim == 0:
        real_raw = onemaxP / px_size * (px_size - raw)
        real_res = onemaxB / px_size * col
    if dim == 1:
        real_raw = twomaxP / px_size * (px_size - raw)
        real_res = twomaxB / px_size * col

    return dim, raw, col, real_dim, np.round(real_raw, 2), real_res
