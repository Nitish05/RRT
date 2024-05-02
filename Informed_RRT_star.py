import cv2
import numpy as np
import random
import time

# Canvas dimensions
canvas_height = 400
canvas_width = 1200

# Define the colors
clearance_color = (127, 127, 127)
obstacle_color = (0, 0, 0)
free_space_color = (255, 255, 255)
threshold = 2
path_color = (0, 255, 0)
clearance_distance = 5
robo_radius = 5

# Initialize a white canvas
canvas = np.ones((canvas_height, canvas_width, 3), dtype="uint8") * 255

# Define obstacles using half-plane model
def obstacles(node):
    x, y = node
    y = abs(y - canvas_height)
    Circ_center = (420, 120)
    R = 60
    Xc, Yc = Circ_center
    obstacles = [
        (x >= 150 and x <= 175 and y <= 200 and y >= 100), 
        (x >= 250 and x <= 275 and y <= 100 and y >= 0),
        (((x - Xc)**2 + (y - Yc)**2) <= R**2),        
    ]
    return any(obstacles)

def clearance(x, y, clearance):
    clearance += robo_radius
    y = abs(y - canvas_height)
    Circ_center = (420, 120)
    R = 60 + clearance
    Xc, Yc = Circ_center
    clearance_zones = [
        (x >= 150 - clearance and x <= 175 + clearance and y <= 200 + clearance and y >= 100 - clearance),
        (x >= 250 - clearance and x <= 275 + clearance and y <= 100 + clearance and y >= 0 - clearance),
        (((x - Xc)**2 + (y - Yc)**2) <= R**2),
        (x <= clearance or x >= canvas_width - clearance or y <= clearance or y >= canvas_height - clearance),
    ]
    return any(clearance_zones)

for x in range(canvas_width):
    for y in range(canvas_height):
        if clearance(x, y, clearance_distance):
            canvas[y, x] = clearance_color
        if obstacles((x, y)):
            canvas[y, x] = obstacle_color

def is_free(x, y):
    return not (obstacles((x, y)) or clearance(x, y, clearance_distance))

def sample_ellipse(center, c_best, c_min, x_center, y_center):
    if c_best >= c_min:
        r = [c_best / c_min, np.sqrt((c_best**2 - c_min**2) / c_best**2) if c_best**2 > c_min**2 else 0]
        L = np.linalg.cholesky(np.diag(r))
        while True:
            sample = np.dot(L, np.random.normal(size=(2,))) + center
            if is_free(int(sample[0]), int(sample[1])):
                return tuple(map(int, sample))
    # else:
    #     # Fall back to random sampling if ellipse parameters are not valid
    #     return (random.randint(0, canvas_width), random.randint(0, canvas_height))

def RRT_star(start, goal, iterations=5000):
    tree = {start: None}
    c_min = np.linalg.norm(np.array(start) - np.array(goal, dtype=np.float64))
    c_best = float('inf')
    center = np.array([start, goal], dtype=np.float64).mean(axis=0)
    x_center, y_center = center
    goal_node = None

    for _ in range(iterations):
        if c_best == float('inf') or c_best < c_min:
            rand_point = (random.randint(0, canvas_width), random.randint(0, canvas_height))
        else:
            rand_point = sample_ellipse(center, c_best, c_min, x_center, y_center)
        nearest = min(tree, key=lambda x: np.linalg.norm(np.array(x, dtype=np.float64) - np.array(rand_point, dtype=np.float64)))
        direction = np.array(rand_point, dtype=np.float64) - np.array(nearest, dtype=np.float64)
        length = np.linalg.norm(direction)
        if length == 0:  # Ensure we do not divide by zero
            continue
        direction = direction / length
        step_size = 15
        new_node = tuple(np.array(nearest, dtype=np.float64) + direction * min(step_size, length))
        new_node = tuple(map(int, new_node))
        if is_free(*new_node):
            tree[new_node] = nearest
            cv2.line(canvas, nearest, new_node, path_color, 1)
            dist_to_goal = np.linalg.norm(np.array(new_node, dtype=np.float64) - np.array(goal, dtype=np.float64))
            if dist_to_goal < 10 and dist_to_goal < c_best:
                c_best = dist_to_goal
                goal_node = new_node

    return tree, goal_node




def reconstruct_path(tree, start, goal_node):
    path = []
    step = goal_node
    while step != start:
        path.append(step)
        step = tree[step]
    path.append(start)
    path.reverse()
    return path

def draw_path(path):
    for i in range(len(path) - 1):
        cv2.line(canvas, path[i], path[i + 1], (255, 0, 0), 2)

start = (100, 100)
goal = (300, 100)

start_time = time.time()
tree, last_node = RRT_star(start, goal)
if last_node:
    path = reconstruct_path(tree, start, last_node)
    draw_path(path)
end_time = time.time()
print("Time taken: ", end_time - start_time)

cv2.imshow("Path Planning with Informed RRT*", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
