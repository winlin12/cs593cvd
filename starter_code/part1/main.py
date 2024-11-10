import os
import argparse

from camera import (
    fit_fundamental_unnormalized, 
    fit_fundamental_normalized, 
    camera_calibration,
    triangulation,
)
from utils.io_helper import torch_read_image, torch_loadtxt
from utils.figure_helper import visualize_matches, visualize_fundamental
from utils.math_helper import get_residual, evaluate_points

def main(args) -> None:
    os.makedirs('results', exist_ok=True)
    image1_fname = f'data/{args.image_name}1.jpg' 
    image2_fname = f'data/{args.image_name}2.jpg'
    matches_fname = f'data/{args.image_name}_matches.txt'

    image1 = torch_read_image(image1_fname, gray=False)
    image2 = torch_read_image(image2_fname, gray=False)
    matches = torch_loadtxt(matches_fname, dtype=image1.dtype)

    visualize_matches(
        image1, image2, matches, 
        f'results/{args.image_name}-matches.png')

    fundamental_unnorm = fit_fundamental_unnormalized(matches)
    print('Fundamental non-normalized = ')
    print(fundamental_unnorm)
    pt_line_dist, direction = get_residual(fundamental_unnorm, matches)
    print(f'Unnorm MSE = {pt_line_dist.square().mean()}')
    visualize_fundamental(
        image2, matches, 
        pt_line_dist, direction,
        f'results/{args.image_name}-F_unnorm.png')
   
    fundamental_norm = fit_fundamental_normalized(matches)
    print('Fundamental normalized = ')
    print(fundamental_norm)
    pt_line_dist, direction = get_residual(fundamental_norm, matches)
    print(f'Norm MSE = {pt_line_dist.square().mean()}')
    visualize_fundamental(
        image2, matches, 
        pt_line_dist, direction,
        f'results/{args.image_name}-F_norm.png')
    
    match args.image_name:
        case 'lab':
            point3d_fname = f'data/{args.image_name}_3d.txt'
            points3d = torch_loadtxt(point3d_fname, image1.dtype)

            image1_proj = camera_calibration(points3d, matches[:, :2])
            image2_proj = camera_calibration(points3d, matches[:, 2:])

            _, residual1 = evaluate_points(image1_proj, matches[:, :2], points3d)
            _, residual2 = evaluate_points(image2_proj, matches[:, 2:], points3d)

            print(f'Image 1 MSE = {residual1}')
            print(f'Image 2 MSE = {residual2}')
        case 'library':
            proj1_fname = f'data/{args.image_name}1_camera.txt'
            image1_proj = torch_loadtxt(proj1_fname, image1.dtype)
            proj2_fname = f'data/{args.image_name}2_camera.txt'
            image2_proj = torch_loadtxt(proj2_fname, image1.dtype)
        case _:
            raise ValueError(args.image_name)

    print('Image 1 camera projection matrix = ')
    print(image1_proj)

    print('Image 2 camera projection matrix = ')
    print(image2_proj)

    points3d_pred = triangulation(matches, image1_proj, image2_proj)
    
    match args.image_name:
        case 'lab':
            residual = (points3d_pred - points3d).square().sum(1).mean()
            print(f'Triangulation 3D MSE = {residual}')
        case _:
            raise ValueError(args.image_name)

    _, proj2d_mse_1 = evaluate_points(image1_proj, matches[:, :2], points3d_pred)
    _, proj2d_mse_2 = evaluate_points(image2_proj, matches[:, 2:], points3d_pred)
    print('2D reprojection error for Image 1 = ', proj2d_mse_1.item())
    print('2D reprojection error for Image 2 = ', proj2d_mse_2.item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CS59300CVD Assignment 4 Part 1')
    parser.add_argument(
        '-i', '--image_name', default='library',
        type=str, help='Input image name')
    args = parser.parse_args()
    main(args)

