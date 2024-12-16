# main.py
from utils import read_images
from visualization import (
    compute_3d_points,  # 3D 점 계산
    plot_3d_points
    
)

from processing import (
    extract_sift_features,  # SIFT 특징 추출 및 매칭
    estimate_essential_matrix,  # Essential Matrix 추정
    decompose_essential_matrix,  # Essential Matrix 분해
    setup_camera_matrices,
    convert_to_homogeneous,
    
)
from matplotlib import pyplot as plt

def main():
    image_directory = './data/nutellar2/'

    image1_filename = 'nutella13.jpg'
    image2_filename = 'nutella14.jpg'
    img1, img2 = read_images(image_directory, image1_filename, image2_filename)
    matches_good, keypoints1, keypoints2 = extract_sift_features(img1, img2)
    essential_matrix, inliers1, inliers2 = estimate_essential_matrix(matches_good, keypoints1, keypoints2)
    camera_matrix = decompose_essential_matrix(essential_matrix, inliers1, inliers2)
    Rt0, Rt1 = setup_camera_matrices(camera_matrix)
    homo_points1, homo_points2 = convert_to_homogeneous(inliers1, inliers2, len(inliers1))
    points_3d = compute_3d_points(homo_points1, homo_points2, Rt0, Rt1)

    plot_3d_points(points_3d)



if __name__ == "__main__":
    main()

    