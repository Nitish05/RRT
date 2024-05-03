import numpy as np
import matplotlib.pyplot as plt
import math

def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))
    # return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5
def sample_points_in_ellipse(start, goal, num_points, C):
    a = distance(start, goal) / 2  # Major axis half-length
    print(distance(start, goal))
    b = math.sqrt(C**2 - distance(start, goal)**2)/2   # Minor axis half-length
    center = ((start[0] + goal[0]) / 2, (start[1] + goal[1]) / 2)
    # print(center)
    theta = math.atan2(goal[1] - start[1], goal[0] - start[0])  # C
    # Generate random angles
    angles = np.random.uniform(0, 2 * np.pi, num_points)
    # Generate random radii
    radii = np.sqrt(np.random.uniform(0, 1, num_points))
    # Calculate x, y coordinates in the unit circle
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    # Scale by ellipse axes
    x *= a
    y *= b
    # Rotate points by theta
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    x_rot = cos_theta * x - sin_theta * y
    y_rot = sin_theta * x + cos_theta * y
    # Translate to center
    x_final = x_rot + center[0]
    y_final = y_rot + center[1]
    return x_final, y_final

# Ellipse parameters
start = (50, 100)  # Input start as a tuple (X, Y)
goal = (100, 100)  # Input goal as a tuple (X, Y)
theta = np.pi/4  # Rotation angle in radians

# Sample points
num_points = 1000
x_points, y_points = sample_points_in_ellipse(start, goal, num_points, 60)

# Plotting
plt.figure(figsize=(8, 8))
plt.scatter(x_points, y_points, s=1)
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Random Points Inside an Ellipse')
plt.show()
