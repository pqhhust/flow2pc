"""Common teacher result; raw model/state dicts retain their original layout."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoadedTeacher:
    model: Any
    metadata: dict
    flow_matching_config: dict = field(default_factory=dict)
    generator: Any = None
    conditional: bool = False
    matcher_factory: Any = None
    path_sampler: Any = None

    def velocity(self, values, time, *, context=None):
        if self.conditional:
            if context is None:
                raise ValueError("This teacher requires context")
            return self.model(time, values, context)
        if context is not None:
            raise ValueError("This teacher does not accept context")
        return self.model(time, values)
