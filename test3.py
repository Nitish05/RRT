# Import necessary libraries


#DO NOT DELETE



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
obs =set()

# Initialize a white canvas
canvas = np.ones((canvas_height, canvas_width, 3), dtype="uint8") * 255

# Define obstacles using half plane model
# def obstacles(node):
#     x, y = node
#     Circ_center = (420, 120)
#     R = 60
#     Xc, Yc = Circ_center
#     # y = abs(y - canvas_height)
#     obstacles = [
#         (x >= 150 and x <= 175 and y <= 200 and y >= 100), 
#         (x >= 250 and x <= 275 and y <= 100 and y >= 0),
#         (((x - Xc)**2 + (y - Yc)**2) <= R**2),        
#     ]
#     return any(obstacles)

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


# Function to check if the node is within the clearance zone
def clearance(x, y, clearance):
    clearance = clearance + robo_radius
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
    # return not (obstacles((x, y)) or clearance(x, y, clearance_distance))
    return not obstacles((x, y))
 


for i in range(canvas_width):
    for j in range(canvas_height):
        if is_free(i, j):
            nodes.append((i, j))
        else:
            canvas[j, i] = obstacle_color
            obs.add((i, j))


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
        cv2.line(canvas, tuple(map(int, nearest)), new_node, path_color, 1)  
        return new_node
    return None

def RRT(start, goal, iterations=1000):
    tree = {start: None}
    available_nodes = nodes.copy()
    for _ in range(iterations):
        if random.randint(0, 100) > 5:
            if available_nodes:
                rand_point = random.randint(0, len(available_nodes) - 1)
                rand_point = available_nodes.pop(rand_point)
            else:
                break
            
        else:
            rand_point = goal
        nearest = nearest_node(tree, rand_point)
        new_node = extend(tree, nearest, rand_point)
        if new_node and distance(new_node, goal) < 10: 
            return tree, new_node
    return tree, None


def reconstruct_path(tree, start, goal_node):
    path = []
    step = goal_node
    while step != start:
        path.append(step)
        step = tree[step]
    path.append(start)
    path.reverse()
    return path

def shortcut_path(path, obstacle_set):
    """
    Simplify the path by removing unnecessary intermediate points.
    
    :param path: list of tuples - The original path points [(x1, y1), (x2, y2), ..., (xn, yn)]
    :param obstacle_set: set of tuples - Set containing obstacle coordinates (x, y)
    :return: list of tuples - The simplified path
    """
    if not path:
        return path
    
    simplified_path = [path[0]]  # Always start with the first node
    max_index = len(path) - 1
    i = 0
    
    while i < max_index:
        j = max_index
        # Try to connect from current node i to the farthest node j
        while j > i + 1:
            if is_free_path(path[i][0], path[i][1], path[j][0], path[j][1], obstacle_set):
                # If path i to j is free of obstacles, move to j
                break
            j -= 1
        # Add node j to the simplified path and set i to j
        simplified_path.append(path[j])
        i = j  # Move to the last connected node
    
    return simplified_path

# Use Bresenham's Line Algorithm to check if the path is free of obstacles
def is_free_path(x1, y1, x2, y2, obstacle_set):
    """
    Check if the path between two points is free of obstacles using Bresenham's Line Algorithm.
    
    :param x1, y1: int - Starting point coordinates
    :param x2, y2: int - Ending point coordinates
    :param obstacle_set: set - Set of obstacle points (x, y)
    :return: bool - True if path is clear, False if obstructed
    """
    dx = abs(x2 - x1)
    dy = -abs(y2 - y1)
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


def draw_path(path, color=(255, 0, 0)):
    for i in range(len(path) - 1):
        cv2.line(canvas, path[i], path[i + 1], color, 2)  


start = (50, 250)
goal = (450, 250)
# Xi = input("Enter start point(X): ")
# Yi = input("Enter start point(Y): ")
# Xf = input("Enter goal point(X): ")
# Yf = input("Enter goal point(Y): ")
# start = (int(Xi), abs(canvas_height - int(Yi)))
# goal = (int(Xf), abs(canvas_height - int(Yf)))

start_time = time.time()
# cv2.circle(canvas, start, 5, (0, 0, 255), -1)
# cv2.circle(canvas, goal, 5, (0, 255, 0), -1)

for x in range(canvas_width):
    for y in range(canvas_height):
        if is_free(x, y):
            nodes.append((x, y))
        # else:
        #     canvas[y, x] = obstacle_color

tree, last_node = RRT(start, goal)
if last_node:
    # path = reconstruct_path(tree, start, last_node)
    path = reconstruct_path(tree, start, last_node)
    print(path)
    print("Path found")
    simplified_path = shortcut_path(path, obs)
    print(simplified_path)
    draw_path(path)
    draw_path(simplified_path, (0, 0, 255))

end_time = time.time()
print("Time taken: ", end_time - start_time)


cv2.imshow("Path Planning with RRT", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
