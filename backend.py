import numpy as np

# Task a)
# Implement a method performing least squares approximation of a linear courve.
# Input: Vectors x,y. Both 1D np.array of same size.
# Output: list of factors [m, b] representing the linear courve f(x) = mx + b.
def linearLSQ(x: np.array, y: np.array) -> list:
  x_ones = np.ones(x.shape[0])
  X = np.column_stack((x_ones, x))
  X_transpose_X = np.dot(X.T, X)
  X_transpose_y = np.dot(X.T, y.T)
  lsq = np.dot(np.linalg.inv(X_transpose_X),X_transpose_y)
  lst = []
  lst.append(lsq[1])
  lst.append(lsq[0])
  return lst

# Task b)
# Implement a method, orthogornalizing the given basis.
# Input: sourceBase - list of vectors, as in a)
# Output: orthoronalizedBase - list of vectors, with same size and shape as sourceBase
def orthonormalize(sourceBase: list) -> list:
    A = np.array(sourceBase)
    n = A.shape[1]

    def norm(x):
        return (sum([z**2 for z in x]))**0.5

    for j in range(n):
        # To orthogonalize the vector in column j with respect to the
        # previous vectors, subtract from it its projection onto
        # each of the previous vectors.
        for k in range(j):
            A[:, j] -= np.dot(A[:, k], A[:, j]) * A[:, k]
        np.seterr(divide='ignore', invalid='ignore')
        A[:, j] = 0 if norm(A[:, j]) == 0 else A[:, j] / norm(A[:, j])

    orthogonal_basis = A.tolist()

    for i, vector in enumerate(orthogonal_basis):
        length = norm(vector)
        for element in vector:
            element /= length


    return list(np.array(orthogonal_basis))

