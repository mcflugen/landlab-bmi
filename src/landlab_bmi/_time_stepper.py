from __future__ import annotations

from collections.abc import Iterator


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

    def __iter__(self) -> Iterator[float]:
        if self._stop is None:
            while 1:
                yield self._time
                self._time += self._step
        else:
            while self._time < self._stop:
                yield self._time
                self._time += self._step
        return

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
        """Advance the time stepper by one time step."""
        self._time += self.step
        if self._stop is not None and self._time > self._stop:
            raise EndOfTimeError(
                f"current time is greater than stop time ({self._time} > {self._stop})"
            )
