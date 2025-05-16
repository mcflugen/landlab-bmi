from __future__ import annotations

from bmipy import Bmi
from landlab.core.model_component import Component

from landlab_bmi._bmi import LandlabBmi
from landlab_bmi._landlab import BmiGridManager


def create_bmi_adapter(class_name: str, component: type[Component]) -> type[Bmi]:
    return type(class_name, (LandlabBmi,), {"_cls": component})


def make_landlab(class_name: str, component: type[Bmi]) -> type[BmiGridManager]:
    return type(class_name, (BmiGridManager,), {"_cls": component})
