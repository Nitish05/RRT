# Import necessary libraries
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
    obstacles = [
        (x >= 150 and x <= 175 and y <= 200 and y >= 100), 
        (x >= 250 and x <= 275 and y <= 100 and y >= 0),
        (((x - Xc)**2 + (y - Yc)**2) <= R**2),        
    ]
    return any(obstacles)

# Function to check if the node is within the clearance zone
def clearance(x, y, clearance):
    clearance = clearance + robo_radius
    Circ_center = (420, 120)
    R = 60 + clearance
    Xc, Yc = Circ_center
    y = abs(y - canvas_height)
    clearance_zones = [
        (x >= 150 - clearance and x <= 175 + clearance and y <= 200 + clearance  and y >= 100 - clearance),
        (x >= 250 - clearance and x <= 275 + clearance and y <= 100 + clearance and y >= 0 - clearance),
        (((x - Xc)**2 + (y - Yc)**2) <= R**2),
        (x <= clearance or x >= canvas_width - clearance or y <= clearance or y >= canvas_height - clearance),
    ]
    return any(clearance_zones)

def is_free(x, y):
    return not (obstacles((x, y)) or clearance(x, y, clearance_distance))


def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def nearest_node(tree, point):
    return min(tree, key=lambda x: distance(x, point))


def extend(tree, nearest, new_point, step_size=2):
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

def RRT(start, goal, iterations=5000):
    tree = {start: None}
    for _ in range(iterations):
        if random.randint(0, 100) > 5: 
            rand_point = (random.randint(0, canvas_width), random.randint(0, canvas_height))
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


def draw_path(path):
    for i in range(len(path) - 1):
        cv2.line(canvas, path[i], path[i + 1], (255, 0, 0), 2)  


start = (50, 100)
goal = (550, 100)
# Xi = input("Enter start point(X): ")
# Yi = input("Enter start point(Y): ")
# Xf = input("Enter goal point(X): ")
# Yf = input("Enter goal point(Y): ")
# start = (int(Xi), abs(canvas_height - int(Yi)))
# goal = (int(Xf), abs(canvas_height - int(Yf)))

start_time = time.time()
cv2.circle(canvas, start, 5, (0, 0, 255), -1)
cv2.circle(canvas, goal, 5, (0, 255, 0), -1)

for x in range(canvas_width):
    for y in range(canvas_height):
        if clearance(x, y, clearance_distance):
            canvas[y, x] = clearance_color
        if obstacles((x, y)):
            canvas[y, x] = obstacle_color

tree, last_node = RRT(start, goal)
if last_node:
    path = reconstruct_path(tree, start, last_node)
    print(path)
    draw_path(path)

end_time = time.time()
print("Time taken: ", end_time - start_time)


cv2.imshow("Path Planning with RRT", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
