import cv2
import numpy as np
from queue import PriorityQueue
import time
import random
import math

# Canvas dimensions
canvas_height = 600
canvas_width = 600

# Define the colors
clearance_color = (127, 127, 127)
obstacle_color = (0, 0, 0)
free_space_color = (255, 255, 255)
threshold = 2
path_color = (0, 255, 0)
clearance_distance = 5
robo_radius = 22
nodes = []



# Initialize a white canvas
canvas = np.ones((canvas_height, canvas_width, 3), dtype="uint8") * 255

# Define obstacles using half plane model
def obstacles(node):
    x, y = node
    Circ_center = (420, 120)
    R = 60
    Xc, Yc = Circ_center
    # y = abs(y - canvas_height)
    obstacles = [
        (x >= 150 and x <= 175 and y <= 200 and y >= 100), 
        (x >= 250 and x <= 275 and y <= 100 and y >= 0),
        (((x - Xc)**2 + (y - Yc)**2) <= R**2),        
    ]
    return any(obstacles)

def clearance(x, y, clearance):
    clearance += robo_radius
    Circ_center = (420, 120)
    R = 60 + clearance
    Xc, Yc = Circ_center
    # y = abs(y - canvas_height)
    clearance_zones = [
        (x >= 150 - clearance and x <= 175 + clearance and y <= 200 + clearance  and y >= 100 - clearance),
        (x >= 250 - clearance and x <= 275 + clearance and y <= 100 + clearance and y >= 0 - clearance),
        (((x - Xc)**2 + (y - Yc)**2) <= R**2),
        (x <= clearance or x >= canvas_width - clearance or y <= clearance or y >= canvas_height - clearance),
    ]
    return any(clearance_zones)

def is_free(x, y):
    return not (obstacles((x, y)) or clearance(x, y, clearance_distance))
    # return True

for x in range(canvas_width):
    for y in range(canvas_height):
        if is_free(x, y):
            nodes.append((x, y))
        else:
            canvas[y, x] = obstacle_color


def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))
    # return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

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
    return new_node

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


# def sample_point_in_ellipse(start, goal, C):
#     # print(C)
#     # print(distance(start, goal))
#     a = distance(start, goal) / 2  # Major axis half-length
    
#     b = math.sqrt(C**2 - distance(start, goal)**2)/2   # Minor axis half-length
#     center = ((start[0] + goal[0]) / 2, (start[1] + goal[1]) / 2)
#     # print(center)
#     theta = math.atan2(goal[1] - start[1], goal[0] - start[0])  # Correct orientation
#     # theta = 0
#     # print(theta)

#     angle = random.uniform(0, 2 * math.pi)
#     r = a * b / math.sqrt((b * math.cos(angle))**2 + (a * math.sin(angle))**2)
#     x = center[0] + r * math.cos(angle + theta)
#     y = center[1] + r * math.sin(angle + theta)

    # return (int(x), int(y))

def sample_points_in_ellipse(start, goal, num_points, C):
    a = distance(start, goal) / 2  # Major axis half-length
    # print(distance(start, goal))
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



def RRT_star(start, goal, iterations=int(len(nodes)*0.02), search_radius=20):
    GOAL_REACHED = False
    tree = {start: None}
    goal_node = None
    available_nodes = nodes.copy()
    for _ in range(iterations):
        # rand_point = random.choice(nodes) if random.randint(0, 100) > 5 else goal
        if GOAL_REACHED:
            # print("Goal Reached")
            # rand_point = sample_points_in_ellipse(start, goal_node, num_points=1000, C= cost_n)
            x_final, y_final = sample_points_in_ellipse(start, goal_node, num_points=1000, C= cost_n)
            valid_points = [(x, y) for x, y in zip(x_final, y_final) if is_free(x, y)]
            if valid_points:
                rand_point = random.choice(valid_points)
            else:
                continue
            # print(rand_point)

        if random.randint(0, 100) > 5:
            if available_nodes:
                rand_point = random.randint(0, len(available_nodes) - 1)
                rand_point = available_nodes.pop(rand_point)
            else:
                break
            
        else:
            rand_point = goal
        nearest = min(tree, key=lambda x: distance(x, rand_point))
        new_node = extend(tree, nearest, rand_point)
        if new_node:
            near_nodes = nearest_nodes(tree, new_node, search_radius)
            tree = choose_parent(tree, new_node, near_nodes)
            tree = rewire(tree, new_node, near_nodes)
            if distance(new_node, goal) < 10:
                GOAL_REACHED = True
                if goal_node is None or cost(tree, new_node) < cost(tree, goal_node):
                    goal_node = new_node
                tree = rewire_goal(tree, goal_node, near_nodes)
                cost_n = cost(tree, new_node)
                # print(cost_n)


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

start = (250, 300)  # Input start as a tuple (X, Y)
goal = (400, 300)  # Input goal as a tuple (X, Y)

start_time = time.time()
tree, last_node = RRT_star(start, goal)
if last_node:
    path = reconstruct_path(tree, start, last_node)
    print(path)
    draw_path(path)
end_time = time.time()
print("Time taken: ", end_time - start_time)

cv2.imshow("Path Planning with RRT*", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
