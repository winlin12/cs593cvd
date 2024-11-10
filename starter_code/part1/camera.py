import torch
from torch import Tensor
def fit_fundamental_unnormalized(matches: Tensor) -> Tensor:
    r'''
    Fundamental matrix using unnormalized algorithm
    
    Arguments
    - `matches`: [`num_matches`, `4`]
    e.g. (x1, y1, x2, y2)
    the first two numbers is a point in the first image
    the last two numbers is a point in the second image
    
    Return
    - `F`: [`3`, `3`]
    '''
    # YOUR CODE HERE
    N = matches.shape[0]
    A = torch.zeros((N, 9), dtype=matches.dtype, device=matches.device)

    # Populate A matrix
    A[:, 0] = matches[:, 0] * matches[:, 2]  # x1 * x2
    A[:, 1] = matches[:, 0] * matches[:, 3]  # x1 * y2
    A[:, 2] = matches[:, 0]                  # x1
    A[:, 3] = matches[:, 1] * matches[:, 2]  # y1 * x2
    A[:, 4] = matches[:, 1] * matches[:, 3]  # y1 * y2
    A[:, 5] = matches[:, 1]                  # y1
    A[:, 6] = matches[:, 2]                  # x2
    A[:, 7] = matches[:, 3]                  # y2
    A[:, 8] = 1

    # SVD of A
    _, _, V = torch.svd(A)
    F = V[:, -1].view(3, 3)  # The last column of V reshaped to 3x3

    # Enforce rank-2 constraint on F
    U, S, Vt = torch.svd(F)
    S[-1] = 0  # Set the smallest singular value to 0
    F = U @ torch.diag(S) @ Vt

    return F

def fit_fundamental_normalized(matches: Tensor) -> Tensor:
    r'''
    Fundamental matrix using normalized algorithm
    
    Arguments
    - `matches`: [`num_matches`, 4]
    e.g. (x1, y1, x2, y2)
    the first two numbers is a point in the first image
    the last two numbers is a point in the second image
    
    Return
    - `F`: [3, 3]
    '''
    # YOUR CODE HERE
    points1 = matches[:, :2]
    points2 = matches[:, 2:]

    # Normalize points
    norm_points1, T1 = normalize_points(points1)
    norm_points2, T2 = normalize_points(points2)

    # Stack normalized points into a new matches tensor
    norm_matches = torch.cat((norm_points1, norm_points2), dim=1)

    # Compute unnormalized F with normalized points
    F_normalized = fit_fundamental_unnormalized(norm_matches)

    # Denormalize
    F = T2.T @ F_normalized @ T1

    return F
def normalize_points(points):
    # points: Nx2 tensor of 2D points

    # Compute the mean of the points
    mean = torch.mean(points, dim=0)

    # Translate points by subtracting the mean
    centered_points = points - mean

    # Compute the average distance to the mean
    dists = torch.norm(centered_points, dim=1)
    avg_dist = torch.mean(dists)

    # Compute the scale factor
    scale = torch.sqrt(torch.tensor(2.0) / avg_dist)

    # Construct the normalization matrix
    T = torch.tensor([[scale, 0, -scale * mean[0]],
                      [0, scale, -scale * mean[1]],
                      [0, 0, 1]], device=points.device)

    # Normalize the points
    # Convert points to homogeneous coordinates (Nx3) and apply T
    points_homog = torch.cat([points, torch.ones(points.shape[0], 1, device=points.device)], dim=1)
    normalized_points_homog = (T @ points_homog.T).T

    # Return normalized points (drop homogeneous coordinate) and the transformation matrix
    return normalized_points_homog[:, :2], T

def camera_calibration(pts_3d: Tensor, pts_2d: Tensor) -> Tensor:
    r"""
    Camera Calibration

    Arguments
    - `pts_3d`: [`num_points`, `3`]
    - `pts_2d`: [`num_points`, `2`]

    Return
    - `projection`: [`3`, `4`]
    """
    # YOUR CODE HERE
    points_3d_homog = torch.cat((pts_3d, torch.ones((pts_3d.shape[0], 1), device=pts_3d.device)), dim=1)
    
    # Convert 2D points to homogeneous coordinates (N x 3)
    points_2d_homog = torch.cat((pts_2d, torch.ones((pts_2d.shape[0], 1), device=pts_2d.device)), dim=1)

    # Initialize matrix A for solving AP = 0
    N = pts_2d.shape[0]
    A = torch.zeros((2 * N, 12), dtype=pts_3d.dtype, device=pts_3d.device)

    # Build the A matrix for each correspondence
    for i in range(N):
        X, Y, Z, W = points_3d_homog[i]
        x, y, w = points_2d_homog[i]

        # First row for the correspondence
        A[2 * i] = torch.tensor([X, Y, Z, W, 0, 0, 0, 0, -x * X, -x * Y, -x * Z, -x * W], device=pts_3d.device)
        
        # Second row for the correspondence
        A[2 * i + 1] = torch.tensor([0, 0, 0, 0, X, Y, Z, W, -y * X, -y * Y, -y * Z, -y * W], device=pts_3d.device)

    # Solve for P by finding the singular vector corresponding to the smallest singular value
    _, _, V = torch.svd(A)
    P = V[:, -1].reshape(3, 4)  # Last column of V reshaped to 3x4

    return P

def triangulation(matches, proj1, proj2) -> Tensor:
    """
    Triangulation

    Arguments
    - `matches`: [`num_points`, `4`]
    - `proj1`: [`3`, `4`]
    - `proj2`: [`3`, `4`]

    Return
    - `X`: [`N`, `3`]
    """
    # YOUR CODE HERE
    X = torch.zeros((matches.shape[0], 3), dtype=matches.dtype, device=matches.device)
    for i in range(matches.shape[0]):
        x1, y1, x2, y2 = matches[i]
        X[i] = triangulation_single(x1, y1, x2, y2, proj1, proj2)
    return X
def triangulation_single(x1, y1, x2, y2, P1, P2) -> Tensor:
    """
    Return
    - `X`: [`3`]
    """
    # YOUR CODE HERE
    A = torch.zeros((4, 4), dtype=P1.dtype, device=P1.device)

    # First image equations
    A[0] = x1 * P1[2] - P1[0]
    A[1] = y1 * P1[2] - P1[1]

    # Second image equations
    A[2] = x2 * P2[2] - P2[0]
    A[3] = y2 * P2[2] - P2[1]

    # Solve for X using SVD
    _, _, V = torch.svd(A)
    X = V[:, -1]  # The solution is the last column of V

    # Convert to 3D by normalizing (homogeneous coordinates)
    X /= X[-1]
    return X[:3]
