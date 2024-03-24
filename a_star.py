'''
Link to the GitHub Repository:
https://github.com/vikrams169/A-Star-Path-Planner
'''

# Importing the required libraries
import os
import shutil
from heapq import *
import numpy as np
import pygame
import cv2
import imageio
import time
import copy

#New
#Initialising variables for map and frame
CLEARANCE = 5
RADIUS=5
map_size = [1200,500]
frame = np.full([map_size[1],map_size[0],3], (0,255,0)).astype(np.uint8)
obstacle_frame = np.full([map_size[1],map_size[0],3], (0,255,0)).astype(np.uint8)
#Size for output video frames
size = (map_size[0], map_size[1])
result = cv2.VideoWriter('a_star.mp4',
                         cv2.VideoWriter_fourcc(*'MP4V'),
                         20, size)


# Defining colour values across the RGB Scale
WHITE = (255,255,255)
BLACK = (0,0,0)
GRAY = (128,128,128)
GREEN = (0,255,0)
RED = (255,0,0)
BLUE = (0,0,255)
ORANGE = (255,164.5,0)

# Size of the grid and dimensions of the grid block
GRID_SIZE = (1200,500)
CLEARANCE = 5
STEP_SIZE = 10
WIDTH_X = 1
WIDTH_Y = 1

# Names of the directory where the animation frames and the video generated would be saved to
FRAME_DIRECTORY = "animation_frames"
VIDEO_NAME = "sample_video.mp4"

# Node Info Explanation
'''
for each cell in the grid (indexed by node_states[row_index][column_index][orientation_index]), there are six dimensions
Index 0: Node state i.e. -1 for not explored yet, 0 for open, 1 for closed
Index 1: Parent node row index
Index 2: Parent node column index
Index 3: Parent node orientation (one of {0,1,2,3,4,5,6,7,8,9,10,11})
Index 4: Total cost (C2C + C2G)
Index 5: CTC
'''
node_info = -1*np.ones((GRID_SIZE[0],GRID_SIZE[1],12,6)) # Information about node distance, parents, and status
min_heap = [] # Min-heap to store nodes initialized as a list
heapify(min_heap) # Making the min-heap list into a heap using the heapq library
start_pos = None # Start position of the planner
goal_pos = None # Goal position of the planner
solved = False # Boolean to check whether the path has been fully solved or not
iteration = 0 # Number of iterations in A* so far
frame_number = 0 # Current number of frames that have been saved


# Return a boolean value of whether a grid cell lies within an obstacle (or in its clearance)
def in_obstacle(loc):
    global obstacle_frame
    # print(loc)
    # if np.array_equal(viz_window.get_at(loc[:-1])[:3],GRAY) or np.array_equal(viz_window.get_at(loc[:-1])[:3],BLACK):
    if np.all(obstacle_frame[int(loc[1]),int(loc[0])]):
        return False
    else:
        return True

# Returning whether a current grid cell is the goal or not
def reached_goal(loc):
    if ((goal_pos[0]-loc[0])**2 + (goal_pos[1]-loc[1])**2)**0.5 <= STEP_SIZE//2 and goal_pos[2]==loc[2]:
        return True
    else:
        return False
    
# Backtracking to find the path between the start and goal locations
def compute_final_path(final_reached_loc):
    global frame
    path_nodes = [final_reached_loc]
    while not np.array_equal(path_nodes[-1],start_pos):
        parent_loc = path_nodes[-1]
        path_nodes.append((int(node_info[parent_loc[0],parent_loc[1],parent_loc[2],1]),int(node_info[parent_loc[0],parent_loc[1],parent_loc[2],2]),int(node_info[parent_loc[0],parent_loc[1],parent_loc[2],3])))
    for loc in path_nodes:
        cv2.circle(frame, [loc[0],loc[1]], 2, (0,0,255), -1) 
        result.write(frame)

# Adding a node to the min-heap
def add_node_to_heap(parent_loc,new_loc):
    global min_heap
    if new_loc[0] < 0 or new_loc[0] >= GRID_SIZE[0] or new_loc[1] < 0 or new_loc[1] >= GRID_SIZE[1] or in_obstacle(new_loc) or node_info[new_loc[0],new_loc[1],new_loc[2],0] == 1:
        return
    dist_c2c = node_info[parent_loc[0],parent_loc[1],parent_loc[2],5] + STEP_SIZE
    dist = dist_c2c + ((goal_pos[0]-new_loc[0])**2 + (goal_pos[1]-new_loc[1])**2)**0.5
    heappush(min_heap,(dist,new_loc,parent_loc))

# Adding all the neighbors of the current node to the min-heap
def add_neighbors(loc):
    for i in range(-2,3):
        theta_index = (loc[2]+i)%12
        theta = theta_index*np.pi/6
        x = int(loc[0] + STEP_SIZE*np.cos(theta))
        y = int(loc[1] + STEP_SIZE*np.sin(theta))
        add_node_to_heap(loc,(x,y,theta_index))

# Processing the current node that was returned by popping the min-heap
def process_node(node):
    global node_info, solved, iteration
    dist, loc, parent_loc = node
    if node_info[loc[0],loc[1],loc[2],0] == 1:
        return
    node_info[loc[0],loc[1],loc[2],0] = 1
    node_info[loc[0],loc[1],loc[2],1] = parent_loc[0]
    node_info[loc[0],loc[1],loc[2],2] = parent_loc[1]
    node_info[loc[0],loc[1],loc[2],3] = parent_loc[2]
    node_info[loc[0],loc[1],loc[2],4] = dist
    if np.array_equal(loc,start_pos):
        node_info[loc[0],loc[1],loc[2],5] = 0
    else:
        node_info[loc[0],loc[1],loc[2],5] = node_info[parent_loc[0],parent_loc[1],parent_loc[2],5] + STEP_SIZE
    frame[int(loc[1]),int(loc[0])] = [255,0,0]
    # pygame.draw.rect(viz_window,BLUE,(loc[0],loc[1],WIDTH_X,WIDTH_Y))
    if reached_goal(loc):
        solved = True
        compute_final_path(loc)
    add_neighbors(loc)
    iteration += 1
    if iteration%500 == 0:
        result.write(frame)
        # save_frame(viz_window)

# Wrapper function for the A* Algorithm
def a_star():
    global min_heap, solved
    starting_dist = ((goal_pos[0]-start_pos[0])**2 + (goal_pos[1]-start_pos[1])**2)**0.5
    heappush(min_heap,(starting_dist,start_pos,start_pos))
    while not solved:
        node = heappop(min_heap)
        process_node(node)

# Initializing start and goal positions using user input
def initialize_start_and_goal_pos():
    global start_pos, goal_pos
    print("Please enter the start and goal positions (in the coordinate system with the origin at the bottom left corner) starting from index 0")
    print("Make sure to to add locations within the 1200mm*500mm grid map avoiding obstacles accounting for a 5mm clearance")
    while(start_pos is None or goal_pos is None):
        start_x = int(input("Enter the X coordinate of the starting position: "))
        start_y = GRID_SIZE[1] - int(input("Enter the Y coordinate of the starting position: "))
        start_theta = (int(input("Enter the starting position orientation (in degress): "))//30)%12
        goal_x = int(input("Enter the X coordinate of the goal position: "))
        goal_y = GRID_SIZE[1] - int(input("Enter the Y coordinate of the goal position: "))
        goal_theta =  (int(input("Enter the goal position orientation (in degress): "))//30)%12
        if start_x < 0 or start_x >= GRID_SIZE[0] or start_y < 0 or start_y >= GRID_SIZE[1] or in_obstacle((start_x,start_y,start_theta)):
            print("Try again with a valid set of values")
            continue
        if goal_x < 0 or goal_x >= GRID_SIZE[0] or goal_y < 0 or goal_y >= GRID_SIZE[1] or in_obstacle((goal_x,goal_y,goal_theta)):
            print("Try again with a valid set of values")
            continue
        start_pos = (start_x,start_y,start_theta)
        goal_pos = (goal_x,goal_y,goal_theta)

# Initializing the map with obstacles in PyGame
def initialize_map():
    global obstacle_frame, frame
    #Walls
    walls_inflated = np.array([[[CLEARANCE,CLEARANCE],
            [map_size[0]-CLEARANCE,CLEARANCE], 
            [map_size[0]-CLEARANCE,map_size[1]-CLEARANCE],
            [0+CLEARANCE, map_size[1]-CLEARANCE]]
    ])
    #Obstacles
    rectangles_inflated = np.array([
        [(100-CLEARANCE,0),(175+CLEARANCE,0),(175+CLEARANCE,400+CLEARANCE),(100-CLEARANCE,400 + CLEARANCE)],
        [(275-CLEARANCE,100-CLEARANCE),(350+CLEARANCE,100-CLEARANCE),(350+CLEARANCE,500),(275-CLEARANCE,500)]
    ]) 
    rectangles = np.array([
        [(100,0),(175,0),(175,400),(100,400)],
        [(275,100),(350,100),(350,500),(275,500)]
    ])
    hexagon_inflated = np.array([
        [(650,100-CLEARANCE),(int(650+75*(3**0.5)+(CLEARANCE/2)*(3**0.5)),int(175-(CLEARANCE/2)*(3**0.5))),(int(650+75*(3**0.5)+(CLEARANCE/2)*(3**0.5)),int(325+(CLEARANCE/2)*(3**0.5))),(650,400+CLEARANCE),(int(650-75*(3**0.5)-(CLEARANCE/2)*(3**0.5)),int(325+(CLEARANCE/2)*(3**0.5))),(int(650-75*(3**0.5)-(CLEARANCE/2)*(3**0.5)),int(175-(CLEARANCE/2)*(3**0.5)))]
    ]) 
    hexagon = np.array([
        [(650,100),(780,175),(780,325),(650,400),(520,325),(520,175)]
    ])
    random_shape_inflated = np.array([
        [(900-CLEARANCE,50-CLEARANCE),(1100+CLEARANCE,50-CLEARANCE),(1100+CLEARANCE,450+CLEARANCE),(900-CLEARANCE,450+CLEARANCE),(900-CLEARANCE,375-CLEARANCE),(1020-CLEARANCE,375-CLEARANCE),(1020-CLEARANCE,125+CLEARANCE),(900-CLEARANCE,125+CLEARANCE)]
    ]) 
    random_shape = np.array([
        [(900,50),(1100,50),(1100,450),(900,450),(900,375),(1020,375),(1020,125),(900,125)]
    ]) 
    cv2.fillPoly(frame, pts=walls_inflated, color=(255, 255, 255))
    cv2.fillPoly(frame, pts=rectangles_inflated, color=(0, 255, 0))
    cv2.fillPoly(frame, pts=rectangles, color=(0, 0, 0))
    cv2.fillPoly(frame, pts=hexagon_inflated, color=(0, 255, 0))
    cv2.fillPoly(frame, pts=hexagon, color=(0, 0, 0))
    cv2.fillPoly(frame, pts=random_shape_inflated, color=(0, 255, 0))
    cv2.fillPoly(frame, pts=random_shape, color=(0, 0, 0))
    obstacle_frame = copy.deepcopy(frame)

# Visualise the output video generated
def visualise():
    cap = cv2.VideoCapture('a_star.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Exiting, end of video?")
            break
        cv2.imshow('A star Visualisation', frame)
        time.sleep(0.05)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    global node_states, start_pos, goal_pos, CLEARANCE, STEP_SIZE

    # Taking the clearnace as a user-input
    CLEARANCE = int(input("Enter the clearance/bloating value for the obstales and walls: "))

    # Taking the step-size as a user-input
    STEP_SIZE = int(input("Enter the step-size in the A* Algorithm: "))

    # Initializing the map with obstacles
    initialize_map()

    # Initializing the start and goal positions of the path from user input
    initialize_start_and_goal_pos()

    print("\nStarting the Pathfinding Process! This may take upto a minute.\n")
    # Running the A* Algorithm
    start_time = time.time()
    a_star()
    end_time = time.time()
    print("\nSuccess! Found the Optimal Path\n")
    print("\nTime taken for the pathfinding process using the A* Algorithm: ",end_time-start_time," seconds\n")
    print("\nPlaying the video animation of the path computation\n")
    result.release()
    visualise()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()