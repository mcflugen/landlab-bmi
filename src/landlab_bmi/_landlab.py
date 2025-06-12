from __future__ import annotations

import contextlib
import fnmatch
import os
import tempfile
from collections.abc import Hashable
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from typing import TextIO

import numpy as np
from landlab import ModelGrid
from landlab import RasterModelGrid
from landlab.field.graph_field import GraphFields
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from sensible_bmi._grid import SensibleGrid
from sensible_bmi._var import SensibleVar
from sensible_bmi.sensible_bmi import SensibleBmi
from sensible_bmi.sensible_bmi import make_sensible

LANDLAB_LOCATION = {
    "node": "node",
    "edge": "link",
    "face": "patch",
    None: "grid",
}


class MissingFieldError(Exception):
    def __init__(self, name: str):
        self._name = name

    def __str__(self) -> str:
        return f"missing field: {self._name!r}"


def _find_var_in_grid(name: str, grid: GraphFields) -> str:
    for at in grid.groups:
        if grid.has_field(name, at=at):
            return at
    else:
        raise MissingFieldError(name)


def _find_vars(
    fields: GraphFields,
    include: Iterable[str] | str = "*",
    exclude: Iterable[str] | str | None = None,
) -> set[str]:
    if isinstance(include, str):
        include = [include]
    if isinstance(exclude, str):
        exclude = [exclude]

    layer_groups = {"_".join(["layer", at]) for at in fields.groups}
    layer_groups.add("layer")

    canonical_names = set()
    for at in fields.groups | layer_groups:
        with contextlib.suppress(KeyError):
            canonical_names.update([f"at_{at}:{name}" for name in fields[at]])

    names = set()
    for pattern in include:
        names.update(fnmatch.filter(canonical_names, pattern))
    for pattern in exclude or []:
        names.difference_update(fnmatch.filter(canonical_names, pattern))

    return names


class GridManager(Mapping[Hashable, ModelGrid]):
    def __init__(
        self, grids: dict[Hashable, ModelGrid] | Iterable[tuple[Hashable, ModelGrid]]
    ) -> None:
        self._grids = dict(grids)

    def __getitem__(self, key: Hashable) -> ModelGrid:
        return self._grids[key]

    def __iter__(self) -> Iterator[Hashable]:
        yield from self._grids.keys()

    def __len__(self) -> int:
        return len(self._grids)

    def update(self) -> None:
        raise NotImplementedError("update")

    def find(self, name: str) -> tuple[ModelGrid, str]:
        # names = {}
        # for id_, grid in self._grids.items():
        #     vars_ = _find_vars(grid, include=f"*:{name}")
        #     if vars_:
        #         names[id_] = vars_
        # return names

        for grid in self._grids.values():
            try:
                at = _find_var_in_grid(name, grid)
            except MissingFieldError:
                continue
            else:
                return grid, at
        else:
            raise MissingFieldError(name)

    def getvalue(self, name: str) -> NDArray[np.number]:
        grid, at = self.find(name)
        return grid.field_values(name, at=at).copy()

    def setvalue(self, name: str, values: ArrayLike) -> None:
        grid, at = self.find(name)
        grid.field_values(name, at=at)[:] = values


class BmiGridManager(GridManager):
    """
    >>> from bmi_wavewatch3 import BmiWaveWatch3
    >>> from landlab.components import LinearDiffuser
    >>> from landlab_bmi.adapter_factory import create_landlab_adapter

    >>> class Ww3Grids(BmiGridManager):
    ...     _cls = BmiWaveWatch3

    >>> ww3 = BmiWaveWatch3()

    >>> Ww3Grids = create_landlab_adapter("Ww3Grids", BmiWaveWatch3)

    >>> config_file = '''
    ... [wavewatch3]
    ... source = "multigrid"
    ... grid = "glo_30m"
    ... date = "2010-05-22"
    ... '''
    >>> ww3 = Ww3Grids(config_file)
    >>> _ = ww3[0].add_empty("topographic__elevation", at="node", dtype=float)

    >>> diffuse = LinearDiffuser(ww3[0])
    >>> for _ in range(10):
    ...     ww3.update()
    ...     ww3[0].at_node["topographic__elevation"][:] = ww3[0].at_node["wave_height"]
    ...     diffuse.run_one_step(1.0)
    """

    _cls = None

    def __init__(self, config_file: str | TextIO | None) -> None:
        if self._cls is None:
            raise ValueError("subclasses of `GridManager` must define `_cls`")
        bmi: SensibleBmi = make_sensible(self._cls.__name__, self._cls)()

        if config_file:
            with as_text_file(config_file) as path:
                bmi.initialize(path)
        else:
            bmi.initialize(config_file)

        grids = {
            grid.id: create_model_grid_from_bmi(grid) for grid in bmi.grid.values()
        }

        if any(var.grid is None for var in bmi.var.values()):
            if len(grids) == 1:
                grids[None] = list(grids.values())[0]
            else:
                grids[None] = GraphFields({"grid": None})

        for var in (bmi.var[name] for name in bmi.output_var_names):
            grids[var.grid].add_field(
                var.name,
                np.atleast_1d(var.get()),
                at=LANDLAB_LOCATION[var.location],
                units=var.units,
            )
        self._bmi: SensibleBmi = bmi

        self._info = {}
        for var in (bmi.var[name] for name in set(self.outputs) | set(self.inputs)):
            intent = ""
            if var.name in bmi.input_var_names:
                intent += "in"
            if var.name in bmi.output_var_names:
                intent += "out"
            self._info[var.name] = create_var_info_from_bmi(
                var, intent=intent, optional=False
            )

        self._grids = grids

        super().__init__(grids)

    @property
    def grid(self) -> ModelGrid | MappingProxyType[int | None, ModelGrid]:
        """Get the model's grid or grids.

        If only one grid exists, or multiple grids can be combined into one grid,
        return it directly; otherwise, return an immutable mapping of grid ids to
        grids.
        """
        grids = self._grids.copy()

        scalar_grid = grids.pop(None, None)

        if len(grids) == 1:
            only_grid = list(grids.values())[0]
            if scalar_grid is only_grid or scalar_grid is None:
                return only_grid

        if scalar_grid is None:
            return MappingProxyType(grids)
        else:
            return MappingProxyType({None: scalar_grid, **grids})

    @property
    def inputs(self) -> tuple[str, ...]:
        return self._bmi.input_var_names

    @property
    def outputs(self) -> tuple[str, ...]:
        return self._bmi.output_var_names

    def set(self, name: str, values: NDArray[np.number]) -> None:
        try:
            grid, at = self.find(name)
        except MissingFieldError:
            var = self._bmi.var[name]
            grid = self._grids[var.grid]
            grid.add_field(
                var.name,
                np.atleast_1d(values),
                at=LANDLAB_LOCATION[var.location],
                units=var.units,
            )
        else:
            grid.field_values(name, at=at).fill(values)

    def update(self, names: Iterable[str] = ()) -> None:
        self._copy_fields_to_bmi(self._bmi.input_var_names)
        self._bmi.update()
        self._copy_bmi_to_fields(self._bmi.output_var_names)

    def _copy_fields_to_bmi(self, names: Iterable[str] = ()) -> None:
        """Update BMI variables with grid fields."""
        missing = {
            var.name: var
            for var in (self._bmi.var[name] for name in names)
            if not self._grids[var.grid].has_field(
                var.name, at=LANDLAB_LOCATION[var.location]
            )
        }
        if missing:
            raise ValueError(
                "\n".join(
                    ["missing input variables:"]
                    + sorted(
                        f"    grid {var.grid}: {var.name}" for var in missing.values()
                    )
                )
            )

        for var in (self._bmi.var[name] for name in names):
            values = self._grids[var.grid].field_values(
                var.name, at=LANDLAB_LOCATION[var.location]
            )
            var.set(values)

    def _copy_bmi_to_fields(self, names: Iterable[str] = ()) -> None:
        """Update grid fields with BMI variables."""
        for var in (self._bmi.var[name] for name in names):
            values = self._grids[var.grid].field_values(
                var.name, at=LANDLAB_LOCATION[var.location]
            )
            var.get(out=values)


def create_model_grid_from_bmi(grid: SensibleGrid) -> ModelGrid:
    if grid.type == "uniform_rectilinear":
        return RasterModelGrid(grid.shape, xy_spacing=grid.spacing)
    else:
        raise ValueError(
            f"{grid.type!r}: BMI grid type not supported for grid with id {grid.id}"
        )


def create_var_info_from_bmi(
    var: SensibleVar, optional: bool = True, intent: str = "inout"
) -> dict[str, Any]:
    return {
        "dtype": np.dtype(var.type),
        "intent": intent,
        "optional": optional,
        "units": var.units,
        "mapping": LANDLAB_LOCATION[var.location],
        "doc": var.name,
    }


@contextmanager
def as_text_file(source: str | TextIO) -> Iterator[str]:
    resolved_file = None
    try:
        resolved_file = get_or_create_file_path(source)
        yield resolved_file.path
    finally:
        if (
            resolved_file
            and resolved_file.is_temp
            and os.path.exists(resolved_file.path)
        ):
            os.unlink(resolved_file.path)


@dataclass
class ResolvedFile:
    path: str
    is_temp: bool = False


def get_or_create_file_path(source: str | TextIO) -> ResolvedFile:
    def write_to_temp(content: str) -> str:
        if not isinstance(content, str):
            raise TypeError("content is not str")
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(content)
        return tmp.name

    if hasattr(source, "read"):
        return ResolvedFile(path=write_to_temp(source.read()), is_temp=True)
    elif isinstance(source, str):
        if os.path.exists(source):
            return ResolvedFile(path=source)
        return ResolvedFile(path=write_to_temp(source), is_temp=True)
    else:
        raise TypeError(f"unsupported input type: {type(source)}")
