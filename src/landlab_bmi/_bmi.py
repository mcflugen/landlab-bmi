from __future__ import annotations

import inspect
from typing import Any

import numpy as np
from bmipy import Bmi
from landlab import HexModelGrid
from landlab import RasterModelGrid
from landlab.core import load_params
from landlab.core.model_component import Component
from landlab.grid.create import create_grid
from numpy.typing import NDArray

from landlab_bmi._constants import BMI_GRID
from landlab_bmi._constants import BMI_LOCATION
from landlab_bmi._time_stepper import TimeStepper


class LandlabBmi(Bmi):
    _cls: Component | None = None

    def finalize(self) -> None:
        pass

    def get_bmi_version(self) -> str:
        return "2.0.0"

    def get_component_name(self) -> str:
        if self._cls is None:
            raise ValueError("subclass must define `_cls`")
        return self._cls.name

    def get_current_time(self) -> float:
        return self._clock.time

    def get_end_time(self) -> float | None:
        return self._clock.stop

    def get_grid_edge_count(self, grid: int) -> int:
        if grid == 0:
            return self._base.grid.number_of_links
        elif grid == 1:
            return self._base.grid.number_of_faces
        else:
            raise KeyError(grid)

    def get_grid_edge_nodes(self, grid: int, edge_nodes: NDArray[np.integer]) -> None:
        if grid == 0:
            edge_nodes[:] = self._base.grid.nodes_at_link.flat
        elif grid == 1:
            edge_nodes[:] = self._base.grid.corners_at_face.flat

    def get_grid_face_count(self, grid: int) -> int:
        if grid == 0:
            return self._base.grid.number_of_patches
        elif grid == 1:
            return self._base.grid.number_of_cells
        else:
            raise KeyError(grid)

    def get_grid_face_edges(self, grid: int, face_edges: NDArray[np.integer]) -> None:
        if grid == 0:
            face_edges[:] = self._base.grid.links_at_patch.flat
        elif grid == 1:
            face_edges[:] = self._base.grid.faces_at_cell.flat

    def get_grid_face_nodes(self, grid: int, face_nodes: NDArray[np.integer]) -> None:
        if grid == 0:
            face_nodes[:] = self._base.grid.nodes_at_patch
        elif grid == 1:
            face_nodes[:] = self._base.grid.corners_at_cell

    def get_grid_node_count(self, grid: int) -> int:
        if grid == 0:
            return self._base.grid.number_of_nodes
        elif grid == 1:
            return self._base.grid.number_of_corners
        else:
            raise KeyError(grid)

    def get_grid_nodes_per_face(
        self, grid: int, nodes_per_face: NDArray[np.integer]
    ) -> None:
        if grid == 0:
            nodes_per_face[:] = np.full(self._base.grid.number_of_nodes, 3, dtype=int)
        elif grid == 1 and isinstance(self._base.grid, HexModelGrid):
            nodes_per_face[:] = np.full(self._base.grid.number_of_faces, 6, dtype=int)

    def get_grid_origin(self, grid: int, origin: NDArray[np.floating]) -> None:
        if grid == 0:
            origin[:] = (self._base.grid.node_y[0], self._base.grid.node_x[0])
        elif grid == 1:
            origin[:] = (
                self._base.grid.node_y[0] + self._base.grid.dy * 0.5,
                self._base.grid.node_x[0] + self._base.grid.dx * 0.5,
            )

    def get_grid_rank(self, grid: int) -> int:
        return 2 if grid in (0, 1) else 0

    def get_grid_shape(self, grid: int, shape: NDArray[np.integer]) -> None:
        if grid == 0:
            shape[:] = (
                self._base.grid.number_of_node_rows,
                self._base.grid.number_of_node_columns,
            )
        elif grid == 1:
            shape[:] = (
                self._base.grid.number_of_node_rows - 1,
                self._base.grid.number_of_node_columns - 1,
            )

    def get_grid_size(self, grid: int) -> int:
        if grid == 0:
            return self._base.grid.number_of_nodes
        elif grid == 1:
            return self._base.grid.number_of_corners
        else:
            raise KeyError(grid)

    def get_grid_spacing(self, grid: int, spacing: NDArray[np.floating]) -> None:
        spacing[:] = (self._base.grid.dy, self._base.grid.dx)

    def get_grid_type(self, grid: int) -> str:
        if grid == 2:
            return "scalar"
        elif isinstance(self._base.grid, RasterModelGrid):
            return "uniform_rectilinear"
        else:
            return "unstructured"

    def get_grid_x(self, grid: int, x: NDArray[np.floating]) -> None:
        if grid == 0:
            x[:] = self._base.grid.x_of_node
        elif grid == 1:
            x[:] = self._base.grid.x_of_corner

    def get_grid_y(self, grid: int, y: NDArray[np.floating]) -> None:
        if grid == 0:
            return self._base.grid.y_of_node
        elif grid == 1:
            return self._base.grid.y_of_corner

    def get_grid_z(self, grid: int, z: NDArray[np.floating]) -> None:
        raise NotImplementedError("get_grid_z")

    def get_input_item_count(self) -> int:
        return len(self._input_var_names)

    def get_input_var_names(self) -> tuple[str, ...]:
        return self._input_var_names

    def get_output_item_count(self) -> int:
        return len(self._output_var_names)

    def get_output_var_names(self) -> tuple[str, ...]:
        return self._output_var_names

    def get_start_time(self) -> float:
        return self._clock.start

    def get_time_step(self) -> float:
        return self._clock.step

    def get_time_units(self) -> str:
        return self._clock.units

    def get_value(self, name: str, dest: NDArray[Any]) -> None:
        at = self._info[name]["mapping"]
        dest[:] = self._base.grid[at][name]

    def get_value_at_indices(
        self, name: str, dest: NDArray[Any], inds: NDArray[np.integer]
    ) -> None:
        at = self._info[name]["mapping"]
        dest[:] = self._base.grid[at][name][inds]

    def get_value_ptr(self, name: str) -> NDArray[Any]:
        at = self._info[name]["mapping"]
        return self._base.grid[at][name]

    def get_var_grid(self, name: str) -> int:
        at = self._info[name]["mapping"]
        return BMI_GRID[at]

    def get_var_itemsize(self, name: str) -> int:
        at = self._info[name]["mapping"]
        return self._base.grid[at][name].itemsize

    def get_var_location(self, name: str) -> str:
        return BMI_LOCATION[self._info[name]["mapping"]]

    def get_var_nbytes(self, name: str) -> int:
        at = self._info[name]["mapping"]
        return self._base.grid[at][name].nbytes

    def get_var_type(self, name: str) -> str:
        at = self._info[name]["mapping"]
        return str(self._base.grid[at][name].dtype)

    def get_var_units(self, name: str) -> str:
        return self._info[name]["units"]

    def initialize(self, config_file: str) -> None:
        if self._cls is None:
            raise ValueError("subclass must define `_cls`")

        grid = create_grid(config_file, section="grid")

        if not grid:
            raise ValueError(f"no grid in config file ({config_file})")
        elif isinstance(grid, list):
            raise ValueError(f"multiple grids in config file ({config_file})")

        params = load_params(config_file)
        params.pop("grid")
        clock_params = params.pop("clock")
        self._clock = TimeStepper(**clock_params)

        self._input_var_names = tuple(
            name
            for name, info in self._cls._info.items()
            if info["intent"].startswith("in")
        )
        self._output_var_names = tuple(
            name
            for name, info in self._cls._info.items()
            if info["intent"].endswith("out")
        )
        for var, info in self._cls._info.items():
            if info["intent"] and info["intent"].startswith("in"):
                grid.add_empty(var, at=info["mapping"], dtype=info["dtype"])

        self._info = {
            name: {
                "intent": info["intent"],
                "mapping": info["mapping"],
                "units": info["units"],
            }
            for name, info in self._cls._info.items()
        }

        self._base = self._cls(grid, **params.pop(self._cls.__name__, {}))
        self._base.grid.at_node["boundary_condition_flag"] = (
            self._base.grid.status_at_node
        )

    def set_value(self, name: str, src: NDArray[Any]) -> None:
        if name == "boundary_condition_flag":
            self._base.grid.status_at_node = src.flat
        else:
            at = self._info[name]["mapping"]
            self._base.grid[at][name][:] = src.flat

    def set_value_at_indices(
        self, name: str, inds: NDArray[np.integer], src: NDArray[Any]
    ) -> None:
        at = self._info[name]["mapping"]
        self._base.grid[at][name][inds] = src

    def update(self) -> None:
        if hasattr(self._base, "update"):
            self._base.update()
        elif hasattr(self._base, "run_one_step"):
            args = []
            for name, arg in inspect.signature(
                self._base.run_one_step
            ).parameters.items():
                if arg.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                    args.append(name)

            if len(args) == 0 or "dt" not in args:
                self._base.run_one_step()
            else:
                self._base.run_one_step(self._clock.step)

        self._clock.advance()

    def update_until(self, time: float) -> None:
        time_step = self._clock.step
        remaining = time - self.get_current_time()
        while remaining > 0.0:
            dt = min(time_step, remaining)
            self._clock.step = dt
            self.update()
            remaining -= dt
            self._clock.step = time_step
