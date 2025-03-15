__version__ = "0.7.1"
__author__ = "Gavin Bosman"

from . import core
from . import utils
from .utils.setup_logging import setup_logging
# make all main functions available via top level import

__all__ = [
    "core", "utils"
]