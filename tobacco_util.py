import gc
import glob
import itertools
import json
import os
import shutil

import cohortintgrad as csig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.model_selection import train_test_split

import data_construct
import util_xtda_chem


def cat_feat_ver1(
    item="Methane",
    pressure_item=100,
    tobacco_dir: str = "3rdparty/tobacco_1.0",
):  # dirname
    dat = pd.read_csv(os.path.join(tobacco_dir, "mofs_map.dat"), sep="  ", header=None)
    dat[2] = dat[1].apply(lambda x: x.split("_")[0])
    dat[3] = dat[1].apply(lambda x: "_".join(x.split("_")[1:5]))
    if os.path.exists(os.path.join(tobacco_dir, "mof_13269.cif")):
        os.rename(
            os.path.join(tobacco_dir, "mof_13269.cif"),
            os.path.join(tobacco_dir, "tobacco_cleaner.py"),
        )  # if os.path.exisits
    if os.path.exists(os.path.join(tobacco_dir, "mof_5151.cif")):
        os.rename(
            os.path.join(tobacco_dir, "mof_5151.cif"),
            os.path.join(tobacco_dir, "mof_list.dat"),
        )
    if os.path.exists(os.path.join(tobacco_dir, "mof_7874.cif")):
        os.rename(
            os.path.join(tobacco_dir, "mof_7874.cif"),
            os.path.join(tobacco_dir, "rename.sh"),
        )
    if os.path.exists(tobacco_dir):
        shutil.copy(
            "tobacco_cif/tobmof-5933.cif",
            os.path.join(tobacco_dir, "mof_5934.cif"),  # wget
        )
    datt = dat.drop([8186, 5150, 7873, 13268])  # git issue  8187 is empty
    datt.index = np.arange(1, len(datt) + 1)
    datt[4] = datt[1].apply(
        lambda x: "_".join(x.split("_")[5:9] if not x.split("_")[5] == "" else "")
    )
    datt[5] = datt[1].apply(
        lambda x: "_".join(x.split("_")[9:])
        if not x.split("_")[5] == ""
        else "_".join(x.split("_")[6:])
    )
    datt["template"] = datt[2].astype("category").cat.codes
    datt["node1"] = datt[3].astype("category").cat.codes
    datt["node2"] = datt[4].astype("category").cat.codes
    datt["edge"] = datt[5].astype("category").cat.codes

    for cif_id in tqdm.tqdm(datt.index):
        # print(cif_id)
        js_fn = f"tobacco_cif/tobmof-{cif_id}.json"  # wget
        with open(js_fn, mode="r") as f:
            js = json.load(f)
        for i in range(len(js["isotherms"])):
            for j in range(len(js["isotherms"][i]["adsorbates"])):
                name = js["isotherms"][i]["adsorbates"][j]["name"]
                # if name not in ['Nitrogen', 'Argon']:
                # print(name)
                if name == item:
                    # print(name)
                    for k in range(len(js["isotherms"][i]["isotherm_data"])):
                        pressure = js["isotherms"][i]["isotherm_data"][k]["pressure"]
                        adsorption = js["isotherms"][i]["isotherm_data"][k][
                            "total_adsorption"
                        ]
                        if pressure == pressure_item:
                            datt.loc[cif_id, "adsorption"] = adsorption

    datt.to_csv("cat_feat_tobacco_1.0.csv")
    return 0


def cut_hist_bin_df(
    bin_fn: str = "tobacco_grid/bins_tobacco_unit50_max_100.csv",
    num_data: int = 1000,
    drop_ratio: float = 0.03,
    npz_fn: str = "tobacco_1.0_CH4_100bar_x_sc.npy",
    df_fn: str = "tobacco_1.0_CH4_100bar_sc.csv",
):
    bin_df = pd.read_csv(bin_fn).set_index("Unnamed: 0")
    _, x, _, y = util_xtda_chem.dataload(npz_fn=npz_fn, df_fn=df_fn)
    x = x[:num_data]
    y = y[:num_data]
    cut_bin_df = bin_df.iloc[list(range(num_data))]  # index are cif_id
    tmp = cut_bin_df.describe().loc["mean"].sort_values(ascending=False)
    drop_df = cut_bin_df.drop(tmp[tmp <= drop_ratio].index, axis=1)
    return x, y, drop_df


def cs_for_csig_dump_px_by_px(
    dir_name: str = "tobacco_cat_predict",
    igcs_cube_fn="igcs_tobacco_100bar_cube_predict_0.01",
    map_js_fn: str = "map_from_testID_to_cifID_tobacco.json",
    cat_df_fn: str = "cat_feat_tobacco_1.0.csv",
    output_stack_fn: str = "cs_for_igcs_allstack_predict_001.npy",
    px_size: int = 54,
    cohort_size: int = 1000,
):
    igcs_cube = torch.load(igcs_cube_fn)
    with open(map_js_fn, mode="r") as f:
        map_js = json.load(f)
    cat_df = pd.read_csv(cat_df_fn).set_index("Unnamed: 0")
    cat_x = (
        cat_df.loc[map_js.values()]
        .iloc[:cohort_size][["template", "node1", "node2", "edge"]]
        .values
    )
    for px in tqdm.tqdm(range(2 * px_size * px_size)):
        IG = csig.CohortIntGrad(x=cat_x, y=igcs_cube[:, px], n_step=50, ratio=0.01)
        ig = np.vstack([IG.cohort_kernel_shap(t_id=i) for i in range(cohort_size)])
        np.save(os.path.join(dir_name, f"cs_for_igcs_predict_001_{px}.npy"), ig)
        del IG, ig
        gc.collect()
    ig_all = np.vstack(
        [
            np.load(
                os.path.join(dir_name, f"cs_for_igcs_predict_001_{px}.npy")
            ).reshape(1, cohort_size, cat_x.shape[1])
            for px in range(2 * px_size * px_size)
        ]
    )
    np.save(os.path.join(dir_name, output_stack_fn), ig_all)
    return 0


def cs_for_csig_dump_px_by_px_forsingle(
    cif_id: int,
    dir_name: str = "tobacco_cat_predict",
    igcs_cube_fn="igcs_tobacco_100bar_cube_predict_0.01",
    map_js_fn: str = "map_from_testID_to_cifID_tobacco.json",
    cat_df_fn: str = "cat_feat_tobacco_1.0.csv",
    output_stack_fn: str = "cs_for_igcs_allstack_predict_001_4091.npy",
    px_size: int = 54,
    cohort_size: int = 1000,
):
    igcs_cube = torch.load(igcs_cube_fn)
    with open(map_js_fn, mode="r") as f:
        map_js = json.load(f)
    data_id = np.where(np.array(list(map_js.values())) == cif_id)[0].item()
    cat_df = pd.read_csv(cat_df_fn).set_index("Unnamed: 0")
    cat_x = (
        cat_df.loc[map_js.values()]
        .iloc[:cohort_size][["template", "node1", "node2", "edge"]]
        .values
    )
    for px in tqdm.tqdm(range(2 * px_size * px_size)):
        IG = csig.CohortIntGrad(x=cat_x, y=igcs_cube[:, px], n_step=50, ratio=0.01)
        ig = IG.cohort_kernel_shap(t_id=data_id)
        np.save(
            os.path.join(dir_name, f"cs_for_igcs_predict_001_{px}_{cif_id}.npy"), ig
        )
        del IG, ig
        gc.collect()
    ig_all = np.vstack(
        [
            np.load(
                os.path.join(dir_name, f"cs_for_igcs_predict_001_{px}_{cif_id}.npy")
            ).reshape(1, cat_x.shape[1])
            for px in range(2 * px_size * px_size)
        ]
    )
    np.save(os.path.join(dir_name, output_stack_fn), ig_all)
    return 0


def csig_for_csig(
    data_id: int,
    x: np.ndarray,
    y: np.ndarray,
    onemaxB: float = 27.0,
    twomaxB: float = 27.0,
    onemaxP: float = 8.8,
    twomaxP: float = 3.5,
    px_size: int = 54,
    # npz_fn: str = "tobacco_1.0_CH4_100bar_x_sc.npy",
    df_fn: str = "tobacco_1.0_CH4_100bar_sc.csv",
    mode: str = "",
):
    # _, x, _, y = util_xtda_chem.dataload(npz_fn=npz_fn, df_fn=df_fn)
    df = pd.read_csv(df_fn).set_index("Unnamed: 0")
    # cut_y = y[:1000]
    _, test_x, _, _ = train_test_split(
        df["fn"].values, df["adsorption"].values, test_size=0.2, random_state=1018
    )
    test_x_cif = [int(i.split("-")[1].split(".")[0]) for i in test_x]
    cat_df = pd.read_csv("cat_feat_tobacco_1.0.csv").set_index("Unnamed: 0")
    xt1, yt1 = data_construct.ticks(
        px_size=px_size, max_birth=onemaxB, max_persistence=onemaxP
    )
    xt2, yt2 = data_construct.ticks(
        px_size=px_size, max_birth=twomaxB, max_persistence=twomaxP
    )
    tick_num = len(xt1) - 1
    if mode == "001":
        all_np_fn = "tobacco_cat/cs_for_igcs_allstack_001.npy"
        igcs_cube_fn = "igcs_tobacco_100bar_cube_0.01"
        prefix = "cs_for_igcs/tobacco_igcs_cube_001_"
        fig_fn = (
            f"cs_for_igcs/cs_for_igcs_threshold_001_tobmof-{test_x_cif[data_id]}.png"
        )
        annotated = "Annotated"
    elif mode == "predict_001":
        all_np_fn = "tobacco_cat_predict/cs_for_igcs_allstack_predict_001.npy"
        igcs_cube_fn = "igcs_tobacco_100bar_cube_predict_0.01"
        prefix = "cs_for_igcs_predict/tobacco_igcs_cube_001_"
        fig_fn = f"cs_for_igcs_predict/cs_for_igcs_threshold_001_tobmof-{test_x_cif[data_id]}.png"
        annotated = "Predict"

    else:
        all_np_fn = "tobacco_cat/cs_for_igcs_allstack.npy"
        igcs_cube_fn = "igcs_tobacco_100bar_cube"
        prefix = "cs_for_igcs/tobacco_igcs_cube_"
        fig_fn = f"cs_for_igcs/cs_for_igcs_tobmof-{test_x_cif[data_id]}.png"
        annotated = "Annotated"
    all_np = np.load(all_np_fn)
    igcs_target = all_np[:, data_id]
    vmax = np.max(abs(igcs_target))
    fig, ax = plt.subplots(4, 2, figsize=(13, 26))
    for i, j in itertools.product(range(4), range(2)):
        ax[i, j].set_title(
            f"CS of {cat_df.loc[test_x_cif[data_id]][i+2]} for cycles in H{j+1}\nsum of attr={np.round(np.sum(igcs_target[:,i].reshape(2, px_size, px_size)[j, ::-1]),2)}",
            fontsize=18,
        )
        im_content = ax[i, j].imshow(
            igcs_target[:, i].reshape(2, px_size, px_size)[j, ::-1],
            cmap="seismic",
            vmin=-vmax,
            vmax=vmax,
        )
        cb = fig.colorbar(im_content, ax=ax[i, j])
        cb.ax.tick_params(labelsize=18)

        ax[i, j].invert_yaxis()
        ax[i, j].set_xticks(
            range(0, px_size + int(px_size / tick_num), int(px_size / tick_num))
        )
        ax[i, j].set_yticks(
            range(0, px_size + int(px_size / tick_num), int(px_size / tick_num))
        )
        ax[i, j].set_xlabel("Birth (angstrom)", fontsize=18)
        ax[i, j].set_ylabel("Persistence (angstrom)", fontsize=18)
    for i in range(4):
        ax[i, 0].set_xticklabels(xt1, fontsize=18, rotation=270)
        ax[i, 0].set_yticklabels(yt1, fontsize=18)
        ax[i, 1].set_xticklabels(xt2, fontsize=18, rotation=270)
        ax[i, 1].set_yticklabels(yt2, fontsize=18)
    # fig.suptitle(
    #    f"{cat_df.loc[ test_x_cif[data_id]][1]}, {annotated} value = {y[data_id]}",
    #    fontsize=18,
    # )
    # fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.tight_layout()

    fig.savefig(fig_fn)
    plt.close()
    plt.clf()
    """
    with open("map_from_testID_to_cifID_tobacco.json", mode="r") as f:
        map_js = json.load(f)

    cif_id = test_x_cif[data_id]

    util_xtda_chem.plot_original_csig(
        data_id=np.where(np.array(list(map_js.values())) == cif_id)[0].item(),
        cif_id=cif_id,
        x=x,
        y=y,
        igcs_fn=igcs_cube_fn,
        dgm_dir="tobacco_1.0_CH4_100bar/",
        prefix=prefix,
        db_prefix="tobmof-",
        mode=annotated,
    )
"""
