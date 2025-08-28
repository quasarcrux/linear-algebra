from typing import List, Tuple
from build.core import MatrixType, zeros, identity, copy_mat, shape, matvec, vector_norm, sub_vector, scale_vector

# LU decomposition (Doolittle) with partial pivoting
# Returns: P, L, U  such that, P*A = L*U
# P is represented as a permutation matrix
def lu_decomposition(A: MatrixType, pivot: bool = True) -> Tuple[MatrixType, MatrixType, MatrixType]:
    n, m = shape(A)
    if n != m:
        raise ValueError("lu_decomposition: matrix must be square")
    A = copy_mat(A)
    P = identity(n)
    L = zeros(n, n)
    U = zeros(n, n)
    for k in range(n):
        # pivot
        if pivot:
            max_row = max(range(k, n), key=lambda i: abs(A[i][k]))
            if k != max_row:
                A[k], A[max_row] = A[max_row], A[k]
                P[k], P[max_row] = P[max_row], P[k]
                L[k], L[max_row] = L[max_row], L[k]  # swap lower part
        pivot_val = A[k][k]
        if abs(pivot_val) < 1e-15:
            pivot_val = 0.0
        # U row
        U[k][k] = pivot_val
        for j in range(k + 1, n):
            U[k][j] = A[k][j]
        # L column below diagonal
        for i in range(k + 1, n):
            if pivot_val == 0.0:
                L[i][k] = 0.0
            else:
                L[i][k] = A[i][k] / pivot_val
            # eliminate
            for j in range(k + 1, n):
                A[i][j] -= L[i][k] * U[k][j]
    # fill diagonal of L with ones
    for i in range(n):
        L[i][i] = 1.0
    return P, L, U

# Solve linear system given LU decomposition
# L y = P b; U x = y
def lu_solve(L: MatrixType, U: MatrixType, P: MatrixType, b: List[float]) -> List[float]:
    n, _ = shape(L)
    if len(b) != n:
        raise ValueError("lu_solve: incompatible right-hand side length")

    # Apply permutation P to b: Pb = P * b
    Pb = [0.0] * n
    for i in range(n):
        s = 0.0
        Pi = P[i]
        for j in range(n):
            s += Pi[j] * b[j]
        Pb[i] = s

    # forward substitution Ly = Pb
    y = [0.0] * n
    for i in range(n):
        s = Pb[i]
        for j in range(i):
            s -= L[i][j] * y[j]
        if abs(L[i][i]) > 1e-15:
            y[i] = s / L[i][i]
        else:
            y[i] = s

    # back substitution Ux = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= U[i][j] * x[j]
        if abs(U[i][i]) > 1e-15:
            x[i] = s / U[i][i]
        else:
            x[i] = s
    return x

# _______________________________
# Determinant via LU (with pivots)
def determinant(A: MatrixType) -> float:
    if not shape(A)[0] == shape(A)[1]:
        raise ValueError("determinant: must be square")
    P, L, U = lu_decomposition(A, pivot=True)
    n, _ = shape(U)
    detU = 1.0
    for i in range(n):
        detU *= U[i][i]
    # determinant of P: +1 or -1 depending on number of row swaps
    perm = [row.index(1) for row in P]
    swaps = 0
    perm_copy = perm[:]
    for i in range(len(perm_copy)):
        while perm_copy[i] != i:
            j = perm_copy[i]
            perm_copy[i], perm_copy[j] = perm_copy[j], perm_copy[i]
            swaps += 1
    sign = -1 if swaps % 2 != 0 else 1
    return sign * detU

def inverse(A: MatrixType) -> MatrixType:
    if not shape(A)[0] == shape(A)[1]:
        raise ValueError("inverse: must be square")
    n, _ = shape(A)
    P, L, U = lu_decomposition(A, pivot=True)
    invA = zeros(n, n)
    I = identity(n)
    for j in range(n):
        e = [I[i][j] for i in range(n)]
        x = lu_solve(L, U, P, e)
        for i in range(n):
            invA[i][j] = x[i]
    return invA
