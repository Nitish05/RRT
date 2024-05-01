import cv2
import numpy as np
import random
import time

# Canvas dimensions
canvas_height = 200
canvas_width = 600

# Define the colors
clearance_color = (127, 127, 127)
obstacle_color = (0, 0, 0)
free_space_color = (255, 255, 255)
path_color = (0, 255, 0)
clearance_distance = 5
robo_radius = 5

# Initialize a white canvas
canvas = np.ones((canvas_height, canvas_width, 3), dtype="uint8") * 255

# Define obstacles using half plane model
def obstacles(node):
    x, y = node
    Circ_center = (420, 120)
    R = 60
    Xc, Yc = Circ_center
    y = abs(y - canvas_height)
    obstacle_list = [
        (x >= 150 and x <= 175 and y <= 200 and y >= 100),
        (x >= 250 and x <= 275 and y <= 100 and y >= 0),
        (((x - Xc) ** 2 + (y - Yc) ** 2) <= R ** 2),
    ]
    return any(obstacle_list)

def clearance(x, y, clearance_value):
    clearance_value += robo_radius
    Circ_center = (420, 120)
    R = 60 + clearance_value
    Xc, Yc = Circ_center
    y = abs(y - canvas_height)
    clearance_zones = [
        (x >= 150 - clearance_value and x <= 175 + clearance_value and y <= 200 + clearance_value and y >= 100 - clearance_value),
        (x >= 250 - clearance_value and x <= 275 + clearance_value and y <= 100 + clearance_value and y >= 0 - clearance_value),
        (((x - Xc) ** 2 + (y - Yc) ** 2) <= R ** 2),
        (x <= clearance_value or x >= canvas_width - clearance_value or y <= clearance_value or y >= canvas_height - clearance_value),
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

def bresenham_line(x0, y0, x1, y1):
    """Generate points between (x0, y0) and (x1, y1) using Bresenham's line algorithm."""
    points = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return points

def is_free_path(x0, y0, x1, y1):
    """Check if a direct path between two points intersects any obstacles."""
    for x, y in bresenham_line(x0, y0, x1, y1):
        if obstacles((x, y)):  # Use the correct function to check for obstacles
            return False
    return True

def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def nearest_nodes(tree, point, radius=20):
    return [node for node in tree if distance(node, point) < radius]

def cost(tree, node):
    if node not in tree:
        return float('inf')
    total_cost = 0
    step = node
    while tree[step] is not None:
        total_cost += distance(step, tree[step])
        step = tree[step]
    return total_cost

def extend(tree, nearest, new_point, step_size=10):
    direction = np.array(new_point) - np.array(nearest)
    length = np.linalg.norm(direction)
    if length == 0:
        return None
    direction = direction / length
    current_length = min(step_size, length)
    new_node = tuple(np.array(nearest) + direction * current_length)
    new_node = tuple(map(int, new_node))
    if is_free_path(nearest[0], nearest[1], new_node[0], new_node[1]):
        return new_node
    return None

def choose_parent(tree, new_node, near_nodes):
    best_parent = None
    best_cost = float('inf')
    for node in near_nodes:
        if is_free(*new_node) and is_free(*node) and cost(tree, node) + distance(node, new_node) < best_cost:
            best_parent = node
            best_cost = cost(tree, node) + distance(node, new_node)
    if best_parent:
        tree[new_node] = best_parent
        cv2.line(canvas, best_parent, new_node, path_color, 1)
    return tree

def rewire(tree, new_node, near_nodes):
    for node in near_nodes:
        if is_free(*new_node) and is_free(*node) and cost(tree, new_node) + distance(new_node, node) < cost(tree, node):
            tree[node] = new_node
            cv2.line(canvas, new_node, node, path_color, 1)
    return tree

def rewire_goal(tree, goal_node, near_nodes):
    for node in near_nodes:
        if is_free(*goal_node) and is_free(*node) and cost(tree, goal_node) + distance(goal_node, node) < cost(tree, node):
            tree[node] = goal_node
            cv2.line(canvas, goal_node, node, path_color, 1)
    return tree

def RRT_star(start, goal, iterations=5000, search_radius=15):
    tree = {start: None}
    goal_node = None
    for _ in range(iterations):
        rand_point = (random.randint(0, canvas_width), random.randint(0, canvas_height)) if random.randint(0, 100) > 5 else goal
        nearest = min(tree, key=lambda x: distance(x, rand_point))
        new_node = extend(tree, nearest, rand_point)
        if new_node:
            near_nodes = nearest_nodes(tree, new_node, search_radius)
            tree = choose_parent(tree, new_node, near_nodes)
            tree = rewire(tree, new_node, near_nodes)
            if distance(new_node, goal) < 10:
                if goal_node is None or cost(tree, new_node) < cost(tree, goal_node):
                    goal_node = new_node
                tree = rewire_goal(tree, goal_node, near_nodes)
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

start = (50, 100)  # Input start as a tuple (X, Y)
goal = (550, 100)  # Input goal as a tuple (X, Y)

start_time = time.time()
tree, last_node = RRT_star(start, goal)
if last_node:
    path = reconstruct_path(tree, start, last_node)
    draw_path(path)
end_time = time.time()
print("Time taken: ", end_time - start_time)

cv2.imshow("Path Planning with RRT*", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
