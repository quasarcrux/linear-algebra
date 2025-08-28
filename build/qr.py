from typing import List, Tuple
import random
import math
from build.core import (
    MatrixType, zeros, shape, copy_mat, matmul, matvec,
    vector_dot, vector_norm, scale_vector, sub_vector
)

# Modified Gram-Schmidt QR decomposition
# Returns Q (m x n with orthonormal columns) and R (n x n)
def qr_decomposition_mgs(A: MatrixType) -> Tuple[MatrixType, MatrixType]:
    m, n = shape(A)
    V = [[A[i][j] for i in range(m)] for j in range(n)]  # columns
    Q_cols = []
    R = zeros(n, n)
    for i in range(n):
        vi = V[i]
        for j in range(i):
            qj = Q_cols[j]
            r = vector_dot(qj, vi)
            R[j][i] = r
            vi = [vi[k] - r * qj[k] for k in range(m)]
        norm_vi = vector_norm(vi)
        if norm_vi < 1e-15:
            qi = [0.0] * m
            R[i][i] = 0.0
        else:
            qi = [x / norm_vi for x in vi]
            R[i][i] = norm_vi
        Q_cols.append(qi)
    Q = zeros(m, n)
    for j in range(n):
        col = Q_cols[j]
        for i in range(m):
            Q[i][j] = col[i]
    return Q, R

# Householder QR decomposition (full)
def qr_decomposition_householder(A: MatrixType) -> Tuple[MatrixType, MatrixType]:
    A = copy_mat(A)
    m, n = shape(A)
    Q = zeros(m, m)
    for i in range(m):
        Q[i][i] = 1.0
    for k in range(min(m, n)):
        x = [A[i][k] for i in range(k, m)]
        normx = vector_norm(x)
        if normx == 0.0:
            continue
        sign = 1.0 if x[0] >= 0 else -1.0
        u1 = x[0] + sign * normx
        w = [u1] + x[1:]
        wnorm = vector_norm(w)
        if wnorm == 0.0:
            continue
        v = [wi / wnorm for wi in w]
        # Apply Householder to A
        for j in range(k, n):
            dot_v = sum(v[i - k] * A[i][j] for i in range(k, m))
            for i in range(k, m):
                A[i][j] -= 2.0 * v[i - k] * dot_v
        # Update Q
        for i in range(m):
            dot_v = sum(Q[i][t] * v[t - k] for t in range(k, m))
            for t in range(k, m):
                Q[i][t] -= 2.0 * dot_v * v[t - k]
    R = zeros(m, n)
    for i in range(m):
        for j in range(n):
            R[i][j] = A[i][j]
    return Q, R

# QR iteration for symmetric matrices
def qr_algorithm_eig(A: MatrixType, max_iter: int = 200, tol: float = 1e-10) -> Tuple[List[float], MatrixType]:
    n, m = shape(A)
    if n != m:
        raise ValueError("qr_algorithm_eig: A must be square")
    B = copy_mat(A)
    Q_total = zeros(n, n)
    for i in range(n):
        Q_total[i][i] = 1.0
    for _ in range(max_iter):
        Q, R = qr_decomposition_householder(B)
        B = matmul(R, Q)
        Q_total = matmul(Q_total, Q)
        off = sum(abs(B[i][j]) for i in range(n) for j in range(n) if i != j)
        if off < tol:
            break
    eigenvalues = [B[i][i] for i in range(n)]
    return eigenvalues, Q_total

# Power iteration for dominant eigenpair
def power_iteration(A: MatrixType, max_iter: int = 2000, tol: float = 1e-10) -> Tuple[float, List[float]]:
    n, m = shape(A)
    if n != m:
        raise ValueError("power_iteration: matrix must be square")
    b = [random.random() for _ in range(n)]
    norm_b = vector_norm(b)
    b = [x / norm_b for x in b]
    lambda_old = 0.0
    for _ in range(max_iter):
        b_next = matvec(A, b)
        norm_next = vector_norm(b_next)
        if norm_next == 0.0:
            break
        b_next = [x / norm_next for x in b_next]
        Ab = matvec(A, b_next)
        lam = vector_dot(b_next, Ab)
        if abs(lam - lambda_old) < tol:
            return lam, b_next
        lambda_old = lam
        b = b_next
    Ab = matvec(A, b)
    lam = vector_dot(b, Ab)
    return lam, b

def eigen_decomposition(A: MatrixType, symmetric: bool = False) -> Tuple[List[float], MatrixType]:
    if symmetric:
        vals, vecs = qr_algorithm_eig(A)
    else:
        vals, vecs = qr_algorithm_eig(A)
    # sort descending
    pairs = list(zip(vals, [[vecs[i][j] for i in range(len(vecs))] for j in range(len(vecs))]))
    pairs.sort(key=lambda p: abs(p[0]), reverse=True)
    vals_sorted = [p[0] for p in pairs]
    n = len(vals_sorted)
    V = zeros(n, n)
    for j in range(n):
        col = pairs[j][1]
        norm_col = vector_norm(col)
        if norm_col > 1e-15:
            for i in range(n):
                V[i][j] = col[i] / norm_col
    return vals_sorted, V

# SVD via eigen-decomposition of A^T A
def svd_via_ata(A: MatrixType, tol: float = 1e-12) -> Tuple[MatrixType, List[float], MatrixType]:
    m, n = shape(A)
    At = [[A[j][i] for j in range(m)] for i in range(n)]
    AtA = matmul(At, A)
    eigvals, V = eigen_decomposition(AtA, symmetric=True)
    singulars_all = [math.sqrt(ev) if ev > 0 else 0.0 for ev in eigvals]
    U = zeros(m, m)
    Vt = zeros(n, n)
    for j in range(n):
        vj = [V[i][j] for i in range(n)]
        for i in range(n):
            Vt[j][i] = vj[i]
    k = min(m, n)
    singulars = singulars_all[:k]
    for j in range(k):
        sigma = singulars[j]
        vj = [V[i][j] for i in range(n)]
        Av = matvec(A, vj)
        if sigma > tol:
            uj = [x / sigma for x in Av]
        else:
            uj = [0.0] * m
        for i in range(m):
            U[i][j] = uj[i] if i < len(uj) else 0.0
    return U, singulars, Vt
