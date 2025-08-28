"""
DTP Linear Algebra
Exposes core linear algebra operations and decompositions:
basic operations, LU, QR, Eigen, SVD.
"""

from .core import *
from .lu import *
from .qr import *
from .eigen import *
from .svd import *
# Any built-in script can do: from dtp_linear_algebra import matmul, lu_decomposition, eigen_decomposition
