import gc
import glob
import json
import os
import pickle
import random
random.seed(42)
# import matplotlib.pyplot as plt
import warnings

import hydra
import matplotlib.pyplot as plt
import mlflow
import numpy as np

import pandas as pd
import tqdm
from moleculetda.structure_to_vectorization import structure_to_pd
from moleculetda.vectorize_pds import pd_vectorization
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def adsorption_json(
    file_prefix: str = "hmof/hMOF-",
    item: str = "CarbonDioxide",
    pressure_item: float = 2.5,
    output_fn: str = "CO2_2.5bar.adsorption.json",
):
    fl = glob.glob(f"{file_prefix}*.json")
    dict25 = dict()
    for js_fn in tqdm.tqdm(fl):
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
                            if js_fn in dict25.keys():
                                print(js_fn, i, j, k)
                            dict25[js_fn] = adsorption
                            # print(js_fn, i,j,k, pressure, adsorption)
    with open(output_fn, mode="w") as f:
        json.dump(dict25, f)
    return dict25


def pd_array(
    dict_fn: str = "CO2_2.5bar.adsorption.json",
    dgm_dir: str = "hmof_co2_25bar",
    px_size: int = 50,
    output_df_fn: str = "CO2_2.5bar.adsorption_norm.csv",
    output_npz_fn: str = "CO2_2.5bar.adsorption_x_norm",
    spec_dict: dict = {
        1: {"maxB": 12.0, "maxP": 6.5, "minBD": 0.0},
        2: {"maxB": 12.0, "maxP": 6.5, "minBD": 0.0},
    },
    part: int = 0,
):
    with open(dict_fn, mode="r") as f:
        dict25 = json.load(f)
    os.makedirs(dgm_dir, exist_ok=True)
    df = pd.DataFrame(columns=["id", "fn", "adsorption"])
    xx = list()
    # yy = list()
    counter = 0
    k_l = sorted(list(dict25.keys()))
    if not part == 0:
        k_l = random.sample(k_l, part)
    for i, k in enumerate(tqdm.tqdm(k_l)):

        cif_fn = f"{k.split('.')[0]}.cif"

        if not os.path.exists(os.path.join(dgm_dir, f"{k.split('/')[-1]}_dgm.pkl")):
            arr_dgms = structure_to_pd(cif_fn, supercell_size=100, periodic=False)
            if "dim1" not in arr_dgms.keys() or "dim2" not in arr_dgms.keys():
                continue

            with open(
                os.path.join(dgm_dir, f"{k.split('/')[-1]}_dgm.pkl"), mode="wb"
            ) as f:
                pickle.dump(arr_dgms, f)

        else:
            with open(
                os.path.join(dgm_dir, f"{k.split('/')[-1]}_dgm.pkl"), mode="rb"
            ) as f:
                arr_dgms = pickle.load(f)
            print(os.path.join(dgm_dir, f"{k.split('/')[-1]}_dgm.pkl"))

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

        # df.append({"id": i, "fn": cif_fn, "adsorption": dict25[k]}, ignore_index=True)
        df.loc[counter] = {"id": i, "fn": cif_fn, "adsorption": dict25[k]}
        counter += 1

    df.to_csv(output_df_fn)
    # print(output_df_fn)
    x = np.vstack(xx)
    np.save(output_npz_fn, x)
    return df


def only_df(
    dict_fn: str = "CO2_2.5bar.adsorption.json",
    dgm_dir: str = "hmof_co2_25bar",
    output_df_fn: str = "CO2_2.5bar.adsorption_norm.csv",
):
    with open(dict_fn, mode="r") as f:
        dict25 = json.load(f)
    df = pd.DataFrame(columns=["id", "fn", "adsorption"])
    k_l = sorted(list(dict25.keys()))
    counter = 0
    for i, k in enumerate(tqdm.tqdm(k_l)):
        if not os.path.exists(os.path.join(dgm_dir, f"{k.split('/')[-1]}_dgm.pkl")):
            continue
        cif_fn = f"{k.split('.')[0]}.cif"
        df.loc[counter] = {"id": i, "fn": cif_fn, "adsorption": dict25[k]}
        counter += 1
    df.to_csv(output_df_fn)
    return df


def from_single_np(
    single_np_prefix: str = "CO2_2.5bar.adsorption_x_norm_",
    single_np_suffix: str = "_13.3_6.5_13.3_6.5.npy",
    df_fn: str = "CO2_2.5bar.adsorption_norm.csv",
    output_npz_fn: str = "CO2_2.5bar.adsorption_x_norm",
):
    df = pd.read_csv(df_fn).set_index("Unnamed: 0")
    xx = list()
    for fn in df["fn"]:
        i = fn.split("-")[-1].split(".")[0]
        single_np_fn = f"{single_np_prefix}{i}{single_np_suffix}"
        if not os.path.exists(single_np_fn):
            continue
        xx.append(np.load(single_np_fn))
    x = np.vstack(xx)
    np.save(output_npz_fn, x)
    return df


def ticks(px_size: int = 50, max_birth: float = 12.0, max_persistence: float = 6.5):
    ticks = list(range(0, px_size + int(px_size / 9), int(px_size / 9)))
    # print(ticks)
    ticklabels_x = [(max_birth / px_size) * i for i in ticks]
    ticklabels_y = [(max_persistence / px_size) * i for i in ticks]

    ticklabels_x = [round(elem, 2) for elem in ticklabels_x]
    ticklabels_y = [round(elem, 2) for elem in ticklabels_y]

    # start from (0, 0)
    # print(ticklabels_x)
    # ticklabels_x.insert(0, 0)
    # ticklabels_y.insert(0, 0)

    return ticklabels_x, ticklabels_y


def rf_fit(
    npz_fn: str = "CO2_2.5bar.adsorption_x_norm.npy",
    df_fn: str = "CO2_2.5bar.adsorption_norm.csv",
    model_pkl_fn: str = "model2.5_rf_norm.pkl",
    save: bool = True,
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
    # print(train_x.shape)
    rg = RFR(n_jobs=-1, random_state=42, n_estimators=500)

    rg.fit(train_x, train_y)

    if save:
        with open(model_pkl_fn, mode="wb") as f:
            pickle.dump(rg, f)
    R2 = rg.score(test_x.reshape(test_x.shape[0], -1), test_y)
    print(f"R2 Score: {R2}")
    return rg, R2


def feat_importance_plot(
    model_pkl_fn: str = "model2.5_rf_norm.pkl",
    px_size: int = 50,
    img_fn: str = "test_rf_fa_25.png",
    one_maxB: float = 12.0,
    two_maxB: float = 12.0,
    one_maxP: float = 6.5,
    two_maxP: float = 2.5,
):
    with open(model_pkl_fn, mode="rb") as f:
        rg = pickle.load(f)
    xt1, yt1 = ticks(px_size=px_size, max_birth=one_maxB, max_persistence=one_maxP)
    xt2, yt2 = ticks(px_size=px_size, max_birth=two_maxB, max_persistence=two_maxP)

    fi = rg.feature_importances_
    ffi = fi.reshape(2, px_size, px_size)

    fig, ax = plt.subplots(1, 2, dpi=100, figsize=(13, 6))
    for i in [0, 1]:
        oo = ffi[i, ::-1]
        img_content = ax[i].imshow(oo, cmap="Purples")
        cb = fig.colorbar(img_content, ax=ax[i])
        cb.ax.tick_params(labelsize=18)
        ax[i].invert_yaxis()
        ax[i].set_xticks(range(0, px_size + int(px_size / 9), int(px_size / 9)))
        # print(list(range(0, px_size + int(px_size / 9), int(px_size / 9))))
        ax[i].set_yticks(range(0, px_size + int(px_size / 9), int(px_size / 9)))
        if i == 0:
            ax[i].set_xticklabels(xt1, fontsize=18, rotation=270)
            ax[i].set_yticklabels(yt1, fontsize=18)
            ax[i].set_title("Feature Importance of H1", fontsize=18)
        elif i == 1:
            ax[i].set_xticklabels(xt2, fontsize=18, rotation=270)
            ax[i].set_yticklabels(yt2, fontsize=18)
            ax[i].set_title("Feature Importance of H2", fontsize=18)
        ax[i].set_xlabel("Birth (angstrom)", fontsize=18)
        ax[i].set_ylabel("Persistence (angstrom)", fontsize=18)
    fig.tight_layout()
    fig.savefig(img_fn)
    plt.close()
    plt.clf()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def kick(cfg: DictConfig):
    out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    config = OmegaConf.to_yaml(cfg)
    print(config)

    with mlflow.start_run():
        df_fn = f"CO2_{cfg.bar}bar.adsorption_norm_{cfg.one_maxB}_{cfg.one_maxP}_{cfg.two_maxB}_{cfg.two_maxP}.csv"
        npz_fn = f"CO2_{cfg.bar}bar.adsorption_norm_{cfg.one_maxB}_{cfg.one_maxP}_{cfg.two_maxB}_{cfg.two_maxP}.npy"
        model_fn = f"CO2_{cfg.bar}bar.adsorption_norm_{cfg.one_maxB}_{cfg.one_maxP}_{cfg.two_maxB}_{cfg.two_maxP}.pkl"
        _ = pd_array(
            dict_fn=f"CO2_{cfg.bar}bar.adsorption.json",
            output_df_fn=df_fn,
            output_npz_fn=npz_fn,
            spec_dict={
                1: {"maxB": cfg.one_maxB, "maxP": cfg.one_maxP, "minBD": 0.0},
                2: {"maxB": cfg.two_maxB, "maxP": cfg.two_maxB, "minBD": 0.0},
            },
            part=2000,
        )
        _, R2 = rf_fit(npz_fn=npz_fn, df_fn=df_fn, model_pkl_fn=model_fn, save=False)

        mlflow.log_artifact(os.path.join(out_dir, ".hydra/config.yaml"))
        mlflow.log_artifact(os.path.join(out_dir, ".hydra/hydra.yaml"))
        mlflow.log_artifact(os.path.join(out_dir, ".hydra/overrides.yaml"))
        mlflow.log_metric("R2_coeff", R2)

        mlflow.log_params(cfg)
        for fn in [df_fn, npz_fn, model_fn]:
            if os.path.exists(fn):
                os.remove(fn)
                # print(fn)

    return 1 - R2


if __name__ == "__main__":
    kick()
