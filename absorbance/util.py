from dataclasses import dataclass


@dataclass
class AbsorbanceOptions:
    target: float
    standard_deviation: float
    oscillator_threshold: float
    energy_threshold: float
