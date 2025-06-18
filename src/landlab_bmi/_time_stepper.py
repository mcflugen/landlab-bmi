from __future__ import annotations

import functools
from collections.abc import Iterator

import numpy as np


class EndOfTimeError(Exception):
    pass


class TimeStepper:
    """Step through time.

    Parameters
    ----------
    start : float, optional
        Clock start time.
    stop : float, optional
        Stop time.
    step : float, optional
        Time step.

    Examples
    --------
    >>> from landlab_bmi._time_stepper import TimeStepper
    >>> time_stepper = TimeStepper()
    >>> time_stepper.start
    0.0
    >>> time_stepper.stop is None
    True
    >>> time_stepper.step
    1.0
    >>> time_stepper.time
    0.0
    >>> for _ in range(10):
    ...     time_stepper.advance()
    ...
    >>> time_stepper.time
    10.0
    >>> time_stepper = TimeStepper(1.0, 13.0, 2.0)
    >>> [time for time in time_stepper]
    [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]
    """

    def __init__(
        self,
        start: float = 0.0,
        stop: float | None = None,
        step: float = 1.0,
        units: str = "s",
    ) -> None:
        self._start = start
        self._stop = stop
        self._step = step
        self._units = units

        self._time = start

        if stop is None:
            self._is_time_to_stop = _never_stop
        else:
            self._is_time_to_stop = functools.partial(_at_or_past_stop, stop=stop)

    def __iter__(self) -> Iterator[float]:
        return self

    def __next__(self) -> float:
        try:
            self.advance()
        except EndOfTimeError:
            raise StopIteration
        return self._time

    @property
    def time(self) -> float:
        """Current time."""
        return self._time

    @property
    def start(self) -> float:
        """Start time."""
        return self._start

    @property
    def stop(self) -> float | None:
        """Stop time."""
        return self._stop

    @property
    def step(self) -> float:
        """Time Step."""
        return self._step

    @step.setter
    def step(self, new_val: float) -> None:
        """Change the time step."""
        self._step = new_val

    @property
    def units(self) -> str:
        """Time units."""
        return self._units

    def advance(self) -> None:
        """Advance the time by one step.

        Increments the internal time by `step`. If the current time is already
        at or beyond the `stop` value, advancing is not allowed and an
        `EndOfTime` exception is raised. The final value of `time` may equal or
        exceed `stop`, but no further advances are permitted once this condition
        is reached.

        Raises
        ------
        EndOfTime
            If the time is already at or beyond the stop value.
        """
        if self._is_time_to_stop(self._time):
            raise EndOfTimeError(
                f"unable to advance from {self._time} to {self._time + self._step}"
                f"(stop time is {self._stop})"
            )
        self._time += self._step


def _never_stop(now: float) -> bool:
    return False


def _at_or_past_stop(now: float, stop: float) -> bool:
    return now > stop or bool(np.isclose(now, stop))
