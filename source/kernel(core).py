import math
import copy
from typing import List, Tuple

Number = float
MatrixType = List[List[Number]]

#
# Matrix creation and inspection
#

def zeros(rows: int, cols: int) -> MatrixType:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]

def identity(n: int) -> MatrixType:
    I = zeros(n, n)
    for i in range(n):
        I[i][i] = 1.0
    return I

def shape(A: MatrixType) -> Tuple[int, int]:
    return len(A), (len(A[0]) if A else 0)

def copy_mat(A: MatrixType) -> MatrixType:
    return [row[:] for row in A]

def is_square(A: MatrixType) -> bool:
    r, c = shape(A)
    return r == c

# Basic matrix operations
def transpose(A: MatrixType) -> MatrixType:
    r, c = shape(A)
    T = zeros(c, r)
    for i in range(r):
        for j in range(c):
            T[j][i] = A[i][j]
    return T

def matmul(A: MatrixType, B: MatrixType) -> MatrixType:
    ar, ac = shape(A)
    br, bc = shape(B)
    if ac != br:
        raise ValueError("matmul: incompatible shapes")
    C = zeros(ar, bc)
    for i in range(ar):
        Ai = A[i]
        Ci = C[i]
        for k in range(ac):
            Aik = Ai[k]
            Bk = B[k]
            for j in range(bc):
                Ci[j] += Aik * Bk[j]
    return C

def matvec(A: MatrixType, v: List[Number]) -> List[Number]:
    r, c = shape(A)
    if c != len(v):
        raise ValueError("matvec: incompatible shapes")
    out = [0.0] * r
    for i in range(r):
        s = 0.0
        row = A[i]
        for j in range(c):
            s += row[j] * v[j]
        out[i] = s
    return out

def scalar_mul(A: MatrixType, s: Number) -> MatrixType:
    r, c = shape(A)
    R = zeros(r, c)
    for i in range(r):
        for j in range(c):
            R[i][j] = A[i][j] * s
    return R

def add(A: MatrixType, B: MatrixType) -> MatrixType:
    r, c = shape(A)
    if shape(B) != (r, c):
        raise ValueError("add: shapes differ")
    R = zeros(r, c)
    for i in range(r):
        for j in range(c):
            R[i][j] = A[i][j] + B[i][j]
    return R

def sub(A: MatrixType, B: MatrixType) -> MatrixType:
    r, c = shape(A)
    if shape(B) != (r, c):
        raise ValueError("sub: shapes differ")
    R = zeros(r, c)
    for i in range(r):
        for j in range(c):
            R[i][j] = A[i][j] - B[i][j]
    return R
  
def vector_dot(u: List[Number], v: List[Number]) -> Number:
    if len(u) != len(v):
        raise ValueError("vector_dot: length mismatch")
    s = 0.0
    for i in range(len(u)):
        s += u[i] * v[i]
    return s

def vector_norm(u: List[Number]) -> Number:
    return math.sqrt(vector_dot(u, u))

def scale_vector(u: List[Number], s: Number) -> List[Number]:
    return [x * s for x in u]

def add_vector(u: List[Number], v: List[Number]) -> List[Number]:
    if len(u) != len(v):
        raise ValueError("add_vector: mismatch")
    return [u[i] + v[i] for i in range(len(u))]

def sub_vector(u: List[Number], v: List[Number]) -> List[Number]:
    if len(u) != len(v):
        raise ValueError("sub_vector: mismatch")
    return [u[i] - v[i] for i in range(len(u))]

# norms
def frobenius_norm(A: MatrixType) -> Number:
    r, c = shape(A)
    s = 0.0
    for i in range(r):
        for j in range(c):
            v = A[i][j]
            s += v * v
    return math.sqrt(s)

#
# Pretty printing
#


def _fmt(x: Number, width=10, prec=6) -> str:
    fmt = f"{{: {width}.{prec}g}}"
    return fmt.format(x)

def print_matrix(A: MatrixType, name: str = "A"):
    r, c = shape(A)
    print(f"{name}: {r}x{c}")
    for i in range(r):
        row = " ".join(_fmt(v) for v in A[i])
        print("  ", row)

#
# Any matrix utility
# 

import random

def random_matrix(m: int, n: int, low: float = -1.0, high: float = 1.0) -> MatrixType:
    return [[random.uniform(low, high) for _ in range(n)] for __ in range(m)]

def approx_equal(a: float, b: float, tol: float = 1e-8) -> bool:
    return abs(a - b) <= tol
