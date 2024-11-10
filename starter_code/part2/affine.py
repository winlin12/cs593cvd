import torch
from torch import Tensor

def normalize_measurements(X: Tensor) -> Tensor:
    r'''
    x1(1)  x2(1)  x3(1) ... xN(1)
    y1(1)  y2(1)  y3(1) ... yN(1)
    ...
    x1(M)  x2(M)  x3(M) ... xN(M)
    y1(M)  y2(M)  y3(M) ... yN(M)
    '''
    # YOUR CODE HERE
    X_centroid = torch.zeros(X.shape[0])
    for i in range(X.shape[0]):
        X_centroid[i] = torch.mean(X[i, :])
        X[i, :] = X[i, :] - X_centroid[i]
    print(X, X.shape)
    return X, X_centroid

def get_structure_and_motion(
    D: Tensor, 
    k: int = 3
) -> tuple[Tensor, Tensor]:
    # YOUR CODE HERE
    U, epsilon, V = torch.svd(D)
    Uk = U[:, :k]
    epsilonk = epsilon[:k]

    # Convert epsilonk to a diagonal matrix
    epsilonk_diag = torch.diag(torch.sqrt(epsilonk))

    # Use epsilonk_diag for matrix multiplication
    Vk = V[:, :k].T  # Transpose V's selected components for compatibility

    # Compute M and S
    M = Uk @ epsilonk_diag
    S = epsilonk_diag @ Vk

    print("Shapes:")
    print("Uk:", Uk.shape)
    print("epsilonk_diag:", epsilonk_diag.shape)
    print("Vk:", Vk.shape)
    print("M:", M.shape)
    print("S:", S.shape)

    return M, S

def get_Q(M) -> Tensor:
    # YOUR CODE HERE
    num_views = M.shape[0] // 2  # Assuming each camera has 2 rows in M
    k = M.shape[1]
    
    # Solve the linear system for the entries of G
    # torch.linalg.lstsq returns a result object; we extract the solution as G_entries
    L_entries = torch.linalg.lstsq(M, M)  # The first output is the solution
    G_solution = L_entries.solution
    # Reshape G_entries into the matrix form
    G = G_solution.view(k, k)

    # Use Cholesky decomposition or SVD to find Q
    Q = torch.linalg.cholesky(G)

    return Q
