## Project objective
The mission is to build a linear algebra framework in pure Python (without external dependencies), with matrix and vector operations LU/ QR decomposition with pivoting and numerical stability, step by step the classical algorithms, from Doolittle to Householder and SVD.

# Exposure Stability
- 100% pure Python: ideal for studying internal mechanics without hiding
- **Full support for rectangular and square matrices**
- Numerical stability: pivoted in LU, Householder in QR

(Robust implementations, with fallback in degenerate cases (zero columns, etc.).)

*Modularity*: each part (core, LU, QR) can be used separately or integrated into demos.

### Purpose: (no educational)
Unfortunately, it allows you to understand **how LU, QR, eigenvalues ​​and SVD work internally**, but that is nothing compared to how big your mother's fat ass is, although it surpasses it when you realize that not only do you learn that, **seeing the step-by-step construction of orthonormal vectors and permutation matrices** improves your coefficient, but a solid foundation is what it contains: **to implement more advanced methods (PCA, regression, optimization, etc)**
