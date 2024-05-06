import cv2
import numpy as np
from queue import PriorityQueue
import time
import random

# Canvas dimensions
canvas_height = 500
canvas_width = 500

# Define the colors
clearance_color = (127, 127, 127)
obstacle_color = (0, 0, 0)
free_space_color = (255, 255, 255)
threshold = 2
path_color = (0, 255, 0)
clearance_distance = 5
robo_radius = 22
nodes = []
Ancestory_Depth = 4
search_radius = 20
step_size = 15


# Initialize a white canvas
canvas = np.ones((canvas_height, canvas_width, 3), dtype="uint8") * 255


def obstacles(node):
    x, y = node
    Circ_center = (420, 120)
    R = 60
    # Xc, Yc = Circ_center
    y_transform = abs(y - canvas_height)
    obstacles = [
        (x >= 115 and x <= 135  and y_transform >= 125 and y_transform <= 375), 
        (x >= 135 and x <= 260 and y_transform >= 240 and y_transform <= 260 ),
        (x >= 240 and x <= 260 and y_transform >= 0 and y_transform <= 240),
        (x >= 240 and x <= 365  and y_transform >= 355 and y_transform <= 375),
        (x >= 365 and x <= 385 and y_transform >= 125 and y_transform <= 500 ),

    ]
    return any(obstacles)

def is_free(x, y):

    return not obstacles((x, y))

for x in range(canvas_width):
    for y in range(canvas_height):
        if is_free(x, y):
            nodes.append((x, y))
        else:
            canvas[y, x] = obstacle_color
            # obst.add((x, y))


def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def nearest_nodes(tree, point, radius=10):
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


def ReConstruct(path):
    if not path:
        return path
    simplified_path = [path[0]] 
    max_index = len(path) - 1
    i = 0
    
    while i < max_index:
        j = max_index
        while j > i + 1:
            if is_free_path(path[i], path[j]):
                break
            j -= 1
        simplified_path.append(path[j])
        i = j  
    return simplified_path




def extend(tree, nearest, new_point, step_size=15):
    direction = np.array(new_point) - np.array(nearest)
    length = np.linalg.norm(direction)
    if length < 1:
        return None
    direction = direction / length
    current_length = min(step_size, length)
    new_node = tuple(np.array(nearest) + direction * current_length)
    new_node = tuple(map(int, new_node))
    if is_free(*new_node) and is_free_path(nearest, new_node):
        return new_node
    return None



def is_free_path(fr, to):
    x1, y1 = fr
    x2, y2 = to
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        if not is_free(x1, y1):
            return False
        if x1 == x2 and y1 == y2:
            break
        
        e2 =  2*err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx: 
            err += dx
            y1 += sy

    return True



def get_parent_nodes(tree, node, depth):
    parents = []
    current = node
    while depth > 0 and tree.get(current) is not None:
        parent = tree[current]
        parents.append(parent)
        current = parent
        depth -= 1
    return parents[::-1]  


def choose_parent(tree, new_node, near_nodes):
    best_parent = None
    best_cost = float('inf')
    for node in near_nodes:
        if is_free_path(node, new_node) and is_free(*node) and cost(tree, node) + distance(node, new_node) < best_cost:
            best_parent = node
            best_cost = cost(tree, node) + distance(node, new_node)
    if best_parent:
        tree[new_node] = best_parent
        cv2.line(canvas, best_parent, new_node, path_color, 1)
    return tree

def q_rewire(tree, new_node, near_nodes_with_ancestry):
    for node in near_nodes_with_ancestry:
        for  x_from in [new_node] + get_parent_nodes(tree, new_node, Ancestory_Depth):
            sigma = extend(tree, x_from, node)
            if sigma and is_free(*sigma) and is_free_path(x_from, node) and cost(tree, x_from) + distance(x_from, sigma) < cost(tree, node) :
                tree[node] = x_from
                cv2.line(canvas, x_from, node, path_color, 1)
    return tree


def Quick_RRT_star(start, goal, iterations=3000, search_radius=20):
    tree = {start: None}
    goal_node = None
    available_nodes = nodes.copy()
    for u in range(iterations):
        # print("inside RRT_star")
        print(u)

        # rand_point = random.choice(nodes) if random.randint(0, 100) > 5 else goal
        if random.randint(0, 100) > 5:
                # rand_point = (random.randint(0, canvas_width), random.randint(0, canvas_height))
                rand_point = random.choice(available_nodes)
            
        else:
            rand_point = goal
        nearest = min(tree, key=lambda x: distance(x, rand_point))
        new_node = extend(tree, nearest, rand_point)
        if new_node:
            near_nodes = nearest_nodes(tree, new_node, search_radius)
            for n in near_nodes:
                ancestry = get_parent_nodes(tree, n, Ancestory_Depth)
                # print(f"Ancestry for {n}: {ancestry}")
                near_nodes_with_ancestry = near_nodes + ancestry
                # near_nodes.extend(ancestry)
            tree = choose_parent(tree, new_node, near_nodes)
            tree = q_rewire(tree, new_node, near_nodes_with_ancestry)
            if distance(new_node, goal) < 10:
                if goal_node is None or cost(tree, new_node) < cost(tree, goal_node):
                    goal_node = new_node
                tree = q_rewire(tree, goal_node, near_nodes)
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

def draw_path(path, color=(255, 0, 0)):
    for i in range(len(path) - 1):
        cv2.line(canvas, path[i], path[i + 1], color, 2)  

start = (50, 400)  # Input start as a tuple (X, Y)
goal = (450, 100)  # Input goal as a tuple (X, Y)

start_time = time.time()
tree, last_node = Quick_RRT_star(start, goal, search_radius)
if last_node:
    path = reconstruct_path(tree, start, last_node)
    path = ReConstruct(path)
    draw_path(path)

end_time = time.time()
print("Time taken: ", end_time - start_time)

cv2.imshow("Path Planning with Quick-RRT*", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
