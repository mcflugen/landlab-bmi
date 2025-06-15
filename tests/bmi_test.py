from __future__ import annotations

import numpy as np
from landlab.core.model_component import Component

from landlab_bmi._bmi import LandlabBmi


class DummyComponent(Component):
    _name = "DummyComponent"
    _info = {}

    def __init__(self, grid, **kwds):
        super().__init__(grid)


class DummyBmi(LandlabBmi):
    _cls = DummyComponent


def test_set_boundary_condition_flag(tmp_path):
    config = """\
grid:
  RasterModelGrid:
    - [2, 3]
clock:
  start: 0
  stop: 1
  step: 1
DummyComponent: {}
"""

    cfg_file = tmp_path / "config.yml"
    cfg_file.write_text(config)

    bmi = DummyBmi()
    bmi.initialize(str(cfg_file))
    grid = bmi._base.grid

    old_status = grid.status_at_node
    new_status = grid.zeros(at="node")
    assert not np.array_equal(old_status, new_status)

    bmi.set_value("boundary_condition_flag", new_status)

    assert np.may_share_memory(old_status, grid.status_at_node)
    assert not isinstance(grid.status_at_node, np.flatiter)
    assert np.array_equal(grid.status_at_node, new_status)
    assert np.array_equal(grid.at_node["boundary_condition_flag"], new_status)
