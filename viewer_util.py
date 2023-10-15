import nglview as nv
import numpy as np
from pymatgen.transformations.advanced_transformations import (
    CubicSupercellTransformation,
)

import periodic_str


class ViewerUtil(periodic_str.Periodic_Structure):
    def __init__(
        self,
        cif_fn: str,
        pdgm_fn: str,
        px_size: float = 54,
        onemaxB: float = 27.0,
        twomaxB: float = 27.0,
        onemaxP: float = 8.8,
        twomaxP: float = 3.5,
        size: int = 100,
        unit: int = 10,
        max_size: int = 20,
    ):
        super().__init__(
            cif_fn=cif_fn,
            pdgm_fn=pdgm_fn,
            px_size=px_size,
            onemaxB=onemaxB,
            onemaxP=onemaxP,
            twomaxB=twomaxB,
            twomaxP=twomaxP,
        )

        self.size = size
        self.unit = unit
        self.max_size = max_size
        self.supercell_structure = CubicSupercellTransformation(
            min_length=size
        ).apply_transformation(self.s)

    def lattice_str(self, dim: int, pair_id: int):
        bdy = self.bdy_symbols(dim=dim, pair_id=pair_id, reduce=False)
        stk_a = self.coords_s[[int(i) for i in bdy]]
        ssdf = self.supercell_structure.as_dataframe()
        remove_index = ssdf[
            (
                ssdf["x"]
                < self.unit_cube[0, 0] * int(np.min(stk_a[:, 0]) / self.unit_cube[0, 0])
            )
            | (
                ssdf["x"]
                > self.unit_cube[0, 0]
                * (int(np.max(stk_a[:, 0]) / self.unit_cube[0, 0]) + 1)
            )
            | (
                ssdf["y"]
                < self.unit_cube[1, 1] * int(np.min(stk_a[:, 1]) / self.unit_cube[1, 1])
            )
            | (
                ssdf["y"]
                > self.unit_cube[1, 1]
                * (int(np.max(stk_a[:, 1]) / self.unit_cube[1, 1]) + 1)
            )
            | (
                ssdf["z"]
                < self.unit_cube[2, 2] * int(np.min(stk_a[:, 2]) / self.unit_cube[2, 2])
            )
            | (
                ssdf["z"]
                > self.unit_cube[2, 2]
                * (int(np.max(stk_a[:, 2]) / self.unit_cube[2, 2]) + 1)
            )
        ].index
        lattice_structure = self.supercell_structure.copy()
        lattice_structure.remove_sites(remove_index)
        return bdy, lattice_structure

    def viewer(self, dim: int, pair_id: int, mode: str = "line"):
        bdy, lattice_structure = self.lattice_str(dim=dim, pair_id=pair_id)
        view = nv.show_pymatgen(lattice_structure)
        view.camera = "perspective"
        if mode == "line":
            for i, k in enumerate(bdy):
                if i + 1 == len(bdy):
                    view.shape.add_arrow(
                        self.coords_s[int(k)],
                        self.coords_s[int(bdy[0])],
                        [0.5, 0.5, 0],
                        0.3,
                    )
                else:
                    view.shape.add_arrow(
                        self.coords_s[int(k)],
                        self.coords_s[int(bdy[i + 1])],
                        [0.5, 0.5, 0],
                        0.3,
                    )
        elif mode == "mesh":
            for i, k in enumerate(bdy):
                if i + 2 < len(bdy):
                    view.shape.add_mesh(
                        list(
                            np.hstack(
                                [
                                    self.coords_s[int(bdy[i])],
                                    self.coords_s[int(bdy[i + 1])],
                                    self.coords_s[int(bdy[i + 2])],
                                ]
                            )
                        ),
                        [0.8 for i in range(3 * 3)],
                    )
        else:
            raise NotImplementedError
        return view
