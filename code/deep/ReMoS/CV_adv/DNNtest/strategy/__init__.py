# Import strategies.
# from .adapt import AdaptiveParameterizedStrategy
# from .adapt import ParameterizedStrategy
from .deepxplore import UncoveredRandomStrategy
from .dlfuzz import DLFuzzRoundRobin
from .dlfuzz import MostCoveredStrategy
from .random import RandomStrategy

# Aliases for some strategies.
# Adapt = AdaptiveParameterizedStrategy
DeepXplore = UncoveredRandomStrategy
DLFuzzFirst = MostCoveredStrategy
