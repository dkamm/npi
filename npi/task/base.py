"""Tasks should subclass these classes and define these methods."""

import enum

class Environment:
    """Environment base class."""

    def observe(self):
        raise NotImplementedError

    def step(self, action, *args):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

class Subroutine(enum.Enum):
    """Tasks are expected to provide a subroutine enum."""

    ACT = 0

def run_reference_program(*args):
    """Returns a dict[list] with observations and trace as keys."""
    raise NotImplementedError

def get_encoder(state_dim=32, num_arguments=3, argument_dim=10):
    """Returns a keras model which has two input layers named 'observation_input_layer' and 'arguments_input_layer'."""
    raise NotImplementedError

