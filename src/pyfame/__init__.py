__version__ = "1.0.0"
__author__ = "Gavin Bosman"

# Look into changing these to explicit names
from pyfame.analyse import *
from pyfame.file_access import *
from pyfame.mesh import *
from pyfame.utilities import *
from pyfame.layer import *
from pyfame.logging import *

from pyfame.analyse import __all__ as __analyse__all__
from pyfame.file_access import __all__ as __io__all__
from pyfame.mesh import __all__ as __mesh__all__
from pyfame.utilities import __all__ as __util__all__
from pyfame.layer import __all__ as __layer__all__
from pyfame.logging import __all__ as __log__all__

__all__ = __analyse__all__ + __io__all__ + __mesh__all__ + __util__all__ + __layer__all__ + __log__all__