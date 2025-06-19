from __future__ import annotations

import pytest
from numpy.testing import assert_array_almost_equal

from landlab_bmi._time_stepper import EndOfTimeError
from landlab_bmi._time_stepper import TimeStepper


@pytest.mark.parametrize("attr", ("start", "stop", "step"))
@pytest.mark.parametrize("value", (0.0, 1.0, -1.0))
def test_properties(attr, value):
    time_stepper = TimeStepper(**{attr: value})
    assert getattr(time_stepper, attr) == value


@pytest.mark.parametrize("start", (0.0, 1.0, -1.0))
def test_time_at_start(start):
    time_stepper = TimeStepper(start=start)
    assert time_stepper.time == start
    assert time_stepper.start == start


def test_units_default_is_seconds():
    assert TimeStepper().units == "s"


@pytest.mark.parametrize("stop", (0.0, 1e-12, -1e-12, -0.1))
@pytest.mark.parametrize("step", (0.0, 0.01, 1.0, 10.0))
def test_time_stepper_advance_raises_end_of_time(stop, step):
    time_stepper = TimeStepper(start=0.0, stop=stop, step=step)
    with pytest.raises(EndOfTimeError):
        time_stepper.advance()
    assert time_stepper.time == 0.0


@pytest.mark.parametrize("n_steps", (1, 2, 4))
@pytest.mark.parametrize("step", (0.01, 1.0, 10.0))
def test_iterator_with_whole_steps(n_steps, step):
    stop = n_steps * step
    time_stepper = TimeStepper(start=0, stop=stop, step=step)

    times = list(time_stepper)
    assert time_stepper.time == pytest.approx(stop)
    assert times[-1] == pytest.approx(stop)


@pytest.mark.parametrize("step", (0.01, 1.0, 10.0))
def test_iterator_without_stop(step):
    n_steps = 10

    expected = list(TimeStepper(start=0, stop=step * n_steps, step=step))

    time_stepper = TimeStepper(start=0, stop=None, step=step)
    actual = [time_stepper.advance() for _ in range(n_steps)]
    assert_array_almost_equal(actual, expected)
