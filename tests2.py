import cv2
import numpy as np
import random
import time

# Canvas dimensions
canvas_height = 200
canvas_width = 600

# Define the colors
obstacle_color = (0, 0, 0)
free_space_color = (255, 255, 255)
path_color = (0, 255, 0)
robo_radius = 22
clearance_distance = 5

# Initialize a white canvas
canvas = np.ones((canvas_height, canvas_width, 3), dtype="uint8") * 255

# Define obstacles
def obstacles(node):
    x, y = node
    circle_center = (420, 120)
    radius = 60
    # Define obstacle conditions
    return ((x >= 150 and x <= 175 and y >= 100 and y <= 200) or
            (x >= 250 and x <= 275 and y >= 0 and y <= 100) or
            ((x - circle_center[0])**2 + (y - circle_center[1])**2 <= radius**2))

def clearance(x, y, clearance):
    clearance += robo_radius
    circle_center = (420, 120)
    radius = 60 + clearance
    # Define clearance conditions
    return ((x >= 150 - clearance and x <= 175 + clearance and y >= 100 - clearance and y <= 200 + clearance) or
            (x >= 250 - clearance and x <= 275 + clearance and y >= 0 - clearance and y <= 100 + clearance) or
            ((x - circle_center[0])**2 + (y - circle_center[1])**2 <= radius**2) or
            (x <= clearance or x >= canvas_width - clearance or y <= clearance or y >= canvas_height - clearance))

def is_free(x, y):
    return not (obstacles((x, y)) or clearance(x, y, clearance_distance))

# Draw obstacles
for i in range(canvas_width):
    for j in range(canvas_height):
        if not is_free(i, j):
            canvas[j, i] = obstacle_color

def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def nearest_node(tree, point):
    return min(tree, key=lambda x: distance(x, point))

def extend(tree, nearest, new_point, step_size=10):
    direction = np.array(new_point) - np.array(nearest)
    length = np.linalg.norm(direction)
    if length == 0:
        return None  
    direction = direction / length
    current_length = min(step_size, length)
    new_node = tuple(np.array(nearest) + direction * current_length)
    new_node = tuple(map(int, new_node))

    if is_free(*new_node):
        tree[new_node] = nearest
        return new_node
    return None

def connect_trees(tree1, tree2):
    for new_point in tree1.keys():
        nearest = nearest_node(tree2, new_point)
        if distance(nearest, new_point) < 10:
            return nearest, new_point
    return None, None

def bidirectional_rrt(start, goal, iterations=5000):
    tree_a = {start: None}
    tree_b = {goal: None}
    for i in range(iterations):
        rand_point = (random.randint(0, canvas_width-1), random.randint(0, canvas_height-1))
        if not is_free(*rand_point):
            continue

        if i % 2 == 0:
            nearest = nearest_node(tree_a, rand_point)
            new_node = extend(tree_a, nearest, rand_point)
            if new_node:
                connection = connect_trees(tree_a, tree_b)
                if connection[0]:
                    return tree_a, tree_b, connection
        else:
            nearest = nearest_node(tree_b, rand_point)
            new_node = extend(tree_b, nearest, rand_point)
            if new_node:
                connection = connect_trees(tree_b, tree_a)
                if connection[0]:
                    return tree_a, tree_b, connection
    return tree_a, tree_b, None

def reconstruct_bidirectional_path(tree_a, tree_b, connect_a, connect_b):
    path = []
    step = connect_a
    while step:
        path.append(step)
        step = tree_a[step]
    path = path[::-1]  # reverse path to start from the starting node

    step = connect_b
    while step:
        path.append(step)
        step = tree_b[step]
    return path

# Define start and goal
start = (50, 100)
goal = (550, 100)

start_time = time.time()
tree_a, tree_b, connection = bidirectional_rrt(start, goal)
if connection:
    path = reconstruct_bidirectional_path(tree_a, tree_b, connection[0], connection[1])
    print("Path found:", path)

    # Drawing path
    for i in range(len(path) - 1):
        cv2.line(canvas, path[i], path[i + 1], (255, 0, 0), 2)

end_time = time.time()
print("Time taken: ", end_time - start_time)

cv2.imshow("Bidirectional RRT", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
