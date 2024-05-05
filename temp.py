import cv2
import numpy as np
from queue import PriorityQueue
import time
import random

# Canvas dimensions
canvas_height = 200
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
obs =set()
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

for x in range(canvas_width):
    for y in range(canvas_height):
        if is_free(x, y):
            nodes.append((x, y))
        else:
            canvas[y, x] = obstacle_color
            obs.add((x, y))


def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def nearest_nodes(tree, point, radius=20):
    return [node for node in tree if distance(node, point) < radius]

def get_parent_nodes(tree, node, depth):
    """
    Get parent nodes of a given node up to a certain depth.
    
    Args:
    tree (dict): The tree structure with nodes as keys and their parent as values.
    node (tuple): The node for which parent nodes are being retrieved.
    depth (int): The maximum depth to retrieve parents for, with 0 being the node itself.
    
    Returns:
    list: A list of parent nodes up to the specified depth.
    """
    parents = []
    current = node
    while depth > 0 and tree.get(current) is not None:
        parent = tree[current]
        parents.append(parent)
        current = parent
        depth -= 1
    return parents

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


def is_free_path(fr, to, obstacle_set):
    """
    Check if the path between two points is free of obstacles using Bresenham's Line Algorithm.
    
    :param x1, y1: int - Starting point coordinates
    :param x2, y2: int - Ending point coordinates
    :param obstacle_set: set - Set of obstacle points (x, y)
    :return: bool - True if path is clear, False if obstructed
    """
    x1, y1 = fr
    x2, y2 = to
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx + dy

    while True:
        if (x1, y1) in obstacle_set:
            return False
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x1 += sx
        if e2 <= dx:
            err += dx
            y1 += sy

    return True


def Q_rewire(tree, new_node, near_nodes):
    
    for node in near_nodes:
        for x_from in [new_node]+ get_parent_nodes(tree, new_node, 1):
            sigma = extend(tree, x_from, node)
            if cost(tree, x_from) + cost(tree, sigma) < cost(tree, node) and is_free_path(node, x_from, obs):
                if is_free(*sigma) and is_free(*node) and is_free(*x_from):
                    print(is_free_path(node, x_from, obs))
                    tree[node] = x_from
                    cv2.line(canvas, x_from, node, path_color, 1)
    return tree


def rewire_goal(tree, goal_node, near_nodes):
    for node in near_nodes:
        if is_free(*goal_node) and is_free(*node) and cost(tree, goal_node) + distance(goal_node, node) < cost(tree, node):
            tree[node] = goal_node
            cv2.line(canvas, goal_node, node, path_color, 1)
    return tree



def RRT_star(start, goal, iterations=int(len(nodes)*0.05), search_radius=20):
    tree = {start: None}
    goal_node = None
    available_nodes = nodes.copy()
    for _ in range(iterations):
        # rand_point = random.choice(nodes) if random.randint(0, 100) > 5 else goal
        if random.randint(0, 100) > 5:
            if available_nodes:
                rand_index = random.randint(0, len(available_nodes) - 1)
                rand_point = available_nodes.pop(rand_index)
                # rand_point = available_nodes[rand_index]
            else:
                break
            
        else:
            rand_point = goal
        nearest = min(tree, key=lambda x: distance(x, rand_point))
        new_node = extend(tree, nearest, rand_point)
        if new_node:
            near_nodes = nearest_nodes(tree, new_node, search_radius)
            # for node in near_nodes:
            #     parent_nodes = get_parent_nodes(tree, node, 1)
            #     near_nodes.extend(parent_nodes)
            tree = choose_parent(tree, new_node, near_nodes)
            tree = rewire(tree, new_node, near_nodes)
            if distance(new_node, goal) < 10:
                if goal_node is None or cost(tree, new_node) < cost(tree, goal_node):
                    goal_node = new_node
                tree = rewire(tree, goal_node, near_nodes)
    return tree, goal_node

def reconstruct_path(tree, start, goal_node):
    path = []
    step = goal_node
    while step != start:
        path.append(step)
        step = tree[step]
    path.append(start)
    path.reverse()
    path_in_meters = [((x -50) / 100, (y - 100) / 100) for x, y in path]
    return path_in_meters





def draw_path(path):
    for i in range(len(path) - 1):
        cv2.line(canvas, path[i], path[i + 1], (255, 0, 0), 2)  

start = (50, 100)  # Input start as a tuple (X, Y)
goal = (550, 100)  # Input goal as a tuple (X, Y)

start_time = time.time()
tree, last_node = RRT_star(start, goal)
if last_node:
    path = reconstruct_path(tree, start, last_node)
    print(path)
end_time = time.time()
print("Time taken: ", end_time - start_time)

cv2.imshow("Path Planning with RRT*", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
