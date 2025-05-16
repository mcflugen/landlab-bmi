from __future__ import annotations

from bmipy import Bmi
from landlab.core.model_component import Component

from landlab_bmi._bmi import LandlabBmi
from landlab_bmi._landlab import BmiGridManager


def create_bmi_adapter(class_name: str, component: type[Component]) -> type[Bmi]:
    """Dynamically create a new BMI-compatible class for a given Landlab Component.

    Parameters
    ----------
    class_name : str
        The name of the generated BMI class.
    component : type[Component]
        The Landlab Component class to be wrapped.

    Returns
    -------
    type[Bmi]
        A new class inheriting from LandlabBmi, wrapping the provided component.
    """
    return type(class_name, (LandlabBmi,), {"_cls": component})


def create_landlab_adapter(
    class_name: str, component: type[Bmi]
) -> type[BmiGridManager]:
    return type(class_name, (BmiGridManager,), {"_cls": component})
