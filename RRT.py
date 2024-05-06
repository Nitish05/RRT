# Import necessary libraries
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

def RRT(start, goal, iterations=5000):
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

def calculate_path_cost(path):
    total_cost = 0
    for i in range(len(path) - 1):
        total_cost += distance(path[i], path[i + 1])
    return total_cost


def draw_path(path, color=(255, 0, 0)):
    for i in range(len(path) - 1):
        cv2.line(canvas, path[i], path[i + 1], color, 2)  


start = (50, 400)
goal = (450, 100)


start_time = time.time()
cv2.circle(canvas, start, 5, (0, 0, 255), -1)
cv2.circle(canvas, goal, 5, (0, 255, 0), -1)

for x in range(canvas_width):
    for y in range(canvas_height):
        if is_free(x, y):
            nodes.append((x, y))
        else:
            canvas[y, x] = obstacle_color

tree, last_node = RRT(start, goal)
if last_node:
    # path = reconstruct_path(tree, start, last_node)
    path = reconstruct_path(tree, start, last_node)
    draw_path(path)
    path_cost = calculate_path_cost(path)
    print("Path cost: ", path_cost)
   


end_time = time.time()
print("Time taken: ", end_time - start_time)


cv2.imshow("Path Planning with RRT", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
