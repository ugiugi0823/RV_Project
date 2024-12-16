# visualization.py
import matplotlib.pyplot as plt
import cv2
import numpy as np
from processing import (
    triangulate_3d_points
    
    
)


# Generate all 3D points
def compute_3d_points(points1, points2, rt0, rt1):
    points_3d = []
    for pt1, pt2 in zip(points1, points2):
        point_3d = triangulate_3d_points(rt0, rt1, pt1, pt2)
        points_3d.append(point_3d)
    points_3d = np.array(points_3d).T
    return points_3d

# Visualize 3D points
def plot_3d_points(points_3d):
    x_vals, y_vals, z_vals = points_3d[0], points_3d[1], points_3d[2]

    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes(projection='3d')
    ax.scatter3D(x_vals, y_vals, z_vals, c='b', marker='o')
    plt.savefig("./result/3d_points_visualization.png", bbox_inches='tight')  # Save as PNG
    plt.show()