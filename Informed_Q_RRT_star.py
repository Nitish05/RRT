import cv2
import numpy as np
import time
import random
import math
import matplotlib.pyplot as plt

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
AD = 1
IT = 2000




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

def get_parent_nodes(tree, node, depth):
    parents = []
    current = node
    while depth > 0 and tree.get(current) is not None:
        parent = tree[current]
        parents.append(parent)
        current = parent
        depth -= 1
    return parents[::-1] 

def q_rewire(tree, new_node, near_nodes_with_ancestry, ad):
    for node in near_nodes_with_ancestry:
        for  x_from in [new_node] + get_parent_nodes(tree, new_node, ad):
            sigma = extend(tree, x_from, node)
            if sigma and is_free(*sigma) and is_free_path(x_from, node) and cost(tree, x_from) + distance(x_from, sigma) < cost(tree, node) :
                tree[node] = x_from
                cv2.line(canvas, x_from, node, path_color, 1)
    return tree

def sample_points_in_ellipse(start, goal, num_points, C):
    a = distance(start, goal) / 2
    b = math.sqrt(C**2 - distance(start, goal)**2)/2   # Minor axis half-length
    center = ((start[0] + goal[0]) / 2, (start[1] + goal[1]) / 2)
    # print(center)
    theta = math.atan2(goal[1] - start[1], goal[0] - start[0])
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



def I_Q_RRT_star(start, goal, iterations=2000, search_radius=20, ad=1):
    GOAL_REACHED = False
    tree = {start: None}
    goal_node = None
    available_nodes = nodes.copy()
    time_costs =[]
    count = 0

    for _ in range(iterations):
        # print(iterations, ad)

        if GOAL_REACHED:
            x_final, y_final = sample_points_in_ellipse(start, goal_node, num_points=1000, C= cost_n)
            valid_points = [(x, y) for x, y in zip(x_final, y_final) if is_free(x, y)]
            if valid_points:
                rand_point = random.choice(valid_points)
            else:
                continue

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
            for n in near_nodes:
                ancestor_nodes = get_parent_nodes(tree, n, ad)
                near_nodes_with_ancestry = near_nodes + ancestor_nodes
            tree = choose_parent(tree, new_node, near_nodes_with_ancestry)
            tree = q_rewire(tree, new_node, near_nodes_with_ancestry, ad)
            if distance(new_node, goal) < 10:
                GOAL_REACHED = True
                if count == 0:
                    tg = time.time()
                    time_to_goal = tg - start_time
                    current_cost = cost(tree, new_node)
                    print(f"Time to goal: {time_to_goal}, Cost to goal: {current_cost}")
                    count += 1
                if goal_node is None or cost(tree, new_node) < cost(tree, goal_node):
                    goal_node = new_node
                tree = q_rewire(tree, goal_node, near_nodes_with_ancestry, ad)
                cost_n = cost(tree, new_node)
        current_time = time.time() - start_time
        current_cost = cost(tree, goal_node) if goal_node else float('inf')
        time_costs.append((current_time, current_cost))
    return tree, goal_node, time_costs

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

# start = (50, 400)  # Input start as a tuple (X, Y)
# goal = (450, 100)  # Input goal as a tuple (X, Y)

# cv2.circle(canvas, start, 5, (0, 0, 255), -1)
# cv2.circle(canvas, goal, 5, (0, 255, 0), -1)

# start_time = time.time()
# tree, last_node = Informed_RRT_star(start, goal)
# if last_node:
#     path = reconstruct_path(tree, start, last_node)
#     # print(path)
#     draw_path(path)
#     path_cost = cost(tree, last_node)
#     print("Path Cost: ", path_cost)
# end_time = time.time()
# print("Time taken: ", end_time - start_time)

# cv2.imshow("Path Planning with Informed RRT* + Q RRT*", canvas)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Iter = [1000, 2000, 3000, 4000, 5000, 10000]
times = []
costs = []
depths = []

DP = [1, 2, 3, 4, 5]

time_costs_by_depth = {ad: [] for ad in DP}


for AD in DP:
    canvas = np.ones((canvas_height, canvas_width, 3), dtype="uint8") * 255
    nodes = []
    for x in range(canvas_width):
        for y in range(canvas_height):
            if is_free(x, y):
                nodes.append((x, y))
            else:
                canvas[y, x] = obstacle_color 
    
    # canvascopy = canvas.copy()
    start = (50, 400)  # Input start as a tuple (X, Y)
    goal = (450, 100)  # Input goal as a tuple (X, Y)
    cv2.circle(canvas, start, 5, (0, 0, 255), -1)
    cv2.circle(canvas, goal, 5, (0, 255, 0), -1)
    path_cost = 0

    start_time = time.time()
    tree, last_node, time_costs = I_Q_RRT_star(start, goal, IT, 20, AD)
    end_time = time.time()
    if last_node:
        path = reconstruct_path(tree, start, last_node)
        # path = ReConstruct(path)
        draw_path(path)
        path_cost = cost(tree, last_node)
        # print(f"{IT} {AD} Path cost: ", path_cost)
    else:
        path_cost = float('inf')

    
    time_taken = end_time - start_time
    time_costs_by_depth[AD] = time_costs

    print(f"{IT} {AD} Path cost: {path_cost}")
    print(f"{IT} {AD} Time taken: {end_time - start_time}")

    cv2.imshow("Path Planning with Quick-RRT*", canvas)
    cv2.waitKey(0)
    # cv2.imwrite(f"I Q RRT {IT} {AD} path cost {path_cost} Time taken {time_taken}.png", canvas)
plt.figure(figsize=(10, 5))
for AD in DP:
    times, costs = zip(*time_costs_by_depth[AD])
    plt.plot(times, costs, marker='o', markersize=1, label=f'Q-depth={AD}')
    # plt.plot(times, costs, marker='^', markersize=8, linestyle='--', linewidth=1, label=f'Q-Depth={AD}')

plt.xlabel('Time (s)')
plt.ylabel('Cost to Goal')
plt.title('Cost to Goal over Time for Various Depths')
plt.legend()
plt.grid(True)
plt.show()

