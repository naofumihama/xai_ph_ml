import os

import dionysus as d
import homcloud.interface as hc
import numpy as np
import pandas as pd
from moleculetda.construct_pd import construct_pds, get_alpha_shapes
from moleculetda.read_file import read_data
from pymatgen.core import Structure


class Periodic_Structure:
    def __init__(
        self,
        cif_fn: str,
        pdgm_fn: str,
        px_size: float = 54,
        onemaxB: float = 27.0,
        twomaxB: float = 27.0,
        onemaxP: float = 8.8,
        twomaxP: float = 3.5,
    ):
        self.cif_fn = cif_fn
        self.coords = read_data(cif_fn, supercell=False, periodic=False, weighted=False)
        self.coords_s, _ = read_data(
            cif_fn, size=100, supercell=True, periodic=False, weighted=False
        )
        self.fs = get_alpha_shapes(self.coords_s)
        self.ffs = d.Filtration(self.fs)
        self.ms = d.homology_persistence(self.ffs)
        self.dgmss = d.init_diagrams(self.ms, self.ffs)
        self.dgmss_s = construct_pds(self.coords_s, False, False)
        self.s = Structure.from_file(self.cif_fn)
        self.unit_cube = self.s.lattice.matrix

        self.px_size = px_size
        self.onemaxB = onemaxB
        self.onemaxP = onemaxP
        self.twomaxB = twomaxB
        self.twomaxP = twomaxP
        self.construct_pdgm(fn=pdgm_fn)

    def atom_compo(self, atom_id, search_len):
        for i in range(search_len):
            if atom_id in self.ffs[i]:
                print(i, self.ffs[i])

    def ms_view(self, id):
        piece = set()
        let = str(self.ms[id]).replace("1*", "").split(" + ")
        print(let)
        if len(let) == 1 and let[0] == "":
            print(self.ffs[id])
            for _l in self.ffs[id]:
                piece.add(_l)
        for ll in let:
            if ll == "":
                continue
            l_piece = int(ll)
            if len(self.ms[l_piece]) == 0:
                print(l_piece, self.ffs[l_piece])
                for _l in self.ffs[l_piece]:
                    piece.add(_l)
                print(d.closure([self.ffs[l_piece]], k=1))
            else:
                print(l_piece, self.ms[l_piece])
                tmp = self.ms_view(l_piece)
                piece = piece | tmp
        return piece

    def similar_distance(self, id, place: int = 0):
        target = np.where(
            np.isclose(
                self.coords[:, place].astype(float),
                (
                    self.coords_s[id]
                    - self.unit_cube[0]
                    * int(self.coords_s[id][0] / self.unit_cube[0, 0])
                    - self.unit_cube[1]
                    * int(self.coords_s[id][1] / self.unit_cube[1, 1])
                    - self.unit_cube[2]
                    * int(self.coords_s[id][2] / self.unit_cube[2, 2])
                )[place],
                atol=1.0e-6,
            )
        )
        if not len(target) == 0:
            if not len(target[0]) == 1:
                # print(id, target)
                if not place == 2:
                    return self.similar_distance(id, place=place + 1)
                else:
                    raise AssertionError
            else:
                return target[0].item()
        else:
            raise AssertionError

    def print_tonikaku(self):
        component = set()
        reduced_comp = set()
        for j, i in enumerate(self.dgmss_s[1]):
            if np.sqrt(i.death) - np.sqrt(i.birth) < 2.0:
                continue
            print(
                i,
                i.data,
                j,
                np.sqrt(i.death),
                np.sqrt(i.birth),
                np.sqrt(i.death) - np.sqrt(i.birth),
            )
            component = component | self.ms_view(i.data)
            reduced_comp = reduced_comp | {
                self.similar_distance(c) for c in self.ms_view(i.data)
            }
            print({self.similar_distance(c) for c in self.ms_view(i.data)})
            print(f"pair: {self.ms.pair(i.data)}")
            print(
                {self.similar_distance(c) for c in self.ms_view(self.ms.pair(i.data))}
            )
        return component, reduced_comp

    def construct_pdgm(self, fn: str):
        if not os.path.exists(fn):
            hc.PDList.from_alpha_filtration(
                self.coords_s, save_to=fn, save_boundary_map=True
            )
        self.pdlist = hc.PDList(fn)
        self.pd1 = self.pdlist.dth_diagram(1)
        self.pd2 = self.pdlist.dth_diagram(2)
        self.pd1_df = self.construct_birth_df(dim=1)
        self.pd2_df = self.construct_birth_df(dim=2)

    def construct_birth_df(self, dim: int = 1):
        df = pd.DataFrame(columns=["birth", "death"])
        tmp_pd = self.pdlist.dth_diagram(dim)
        df["birth"] = tmp_pd.births
        df["death"] = tmp_pd.deaths
        df["birth_sqrt"] = df["birth"].apply(np.sqrt)
        df["death_sqrt"] = df["death"].apply(np.sqrt)
        df = df[df["death_sqrt"] <= self.onemaxB + self.onemaxP]
        df = df[df["birth_sqrt"] <= self.onemaxB]
        df["persistence"] = df["death_sqrt"] - df["birth_sqrt"]
        return df

    def bdy_symbols(self, dim: int, pair_id: int, reduce: bool):
        tmp_pd = self.pdlist.dth_diagram(dim)
        pair = tmp_pd.pair(pair_id).optimal_volume().boundary_points_symbols()
        if reduce:
            bdys = [self.similar_distance(int(i)) for i in pair]
            return bdys
        else:
            return [int(i) for i in pair]
