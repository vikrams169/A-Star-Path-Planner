# Importing the required libraries
import os
import shutil
from heapq import *
import numpy as np
import pygame
import cv2
import imageio
import time

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

# Playing the generated video
def play_video(video_file):
    vid = cv2.VideoCapture(video_file)
    while vid.isOpened():
        frame_present, frame = vid.read()
        if frame_present:
            cv2.imshow('Animation Video',frame)
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break
        else:
            break
    vid.release()
    cv2.destroyAllWindows()

# Function to make a video from a set of frames in a directory
def make_video(frames_directory,video_name):
    num_frames = len(os.listdir(frames_directory))
    with imageio.get_writer(video_name,mode='I',fps=50) as writer:
        for i in range(num_frames):
            image = imageio.imread(frames_directory+"/frame"+str(i)+".jpg")
            writer.append_data(image)

# Saving a frame to a directory
def save_frame(viz_window):
    global frame_number
    pygame.display.update()
    pygame.image.save(viz_window,FRAME_DIRECTORY+"/frame"+str(frame_number)+".jpg")
    frame_number += 1

# Return a boolean value of whether a grid cell lies within an obstacle (or in its clearance)
def in_obstacle(viz_window,loc):
    if np.array_equal(viz_window.get_at(loc[:-1])[:3],GRAY) or np.array_equal(viz_window.get_at(loc[:-1])[:3],BLACK):
        return True
    else:
        return False

# Returning whether a current grid cell is the goal or not
def reached_goal(loc):
    if ((goal_pos[0]-loc[0])**2 + (goal_pos[1]-loc[1])**2)**0.5 <= STEP_SIZE//2 and goal_pos[2]==loc[2]:
        return True
    else:
        return False
    
# Backtracking to find the path between the start and goal locations
def compute_final_path(viz_window,final_reached_loc):
    path_nodes = [final_reached_loc]
    while not np.array_equal(path_nodes[-1],start_pos):
        parent_loc = path_nodes[-1]
        path_nodes.append((int(node_info[parent_loc[0],parent_loc[1],parent_loc[2],1]),int(node_info[parent_loc[0],parent_loc[1],parent_loc[2],2]),int(node_info[parent_loc[0],parent_loc[1],parent_loc[2],3])))
    for loc in path_nodes:
        pygame.draw.rect(viz_window,GREEN,(loc[0],loc[1],5*WIDTH_X,5*WIDTH_Y))
    save_frame(viz_window)
    for i in range(50):
        save_frame(viz_window)

# Adding a node to the min-heap
def add_node_to_heap(viz_window,parent_loc,new_loc):
    global min_heap
    if new_loc[0] < 0 or new_loc[0] >= GRID_SIZE[0] or new_loc[1] < 0 or new_loc[1] >= GRID_SIZE[1] or in_obstacle(viz_window,new_loc) or node_info[new_loc[0],new_loc[1],new_loc[2],0] == 1:
        return
    dist_c2c = node_info[parent_loc[0],parent_loc[1],parent_loc[2],5] + STEP_SIZE
    dist = dist_c2c + ((goal_pos[0]-new_loc[0])**2 + (goal_pos[1]-new_loc[1])**2)**0.5
    heappush(min_heap,(dist,new_loc,parent_loc))

# Adding all the neighbors of the current node to the min-heap
def add_neighbors(viz_window,loc):
    for i in range(-2,3):
        theta_index = (loc[2]+i)%12
        theta = theta_index*np.pi/6
        x = int(loc[0] + STEP_SIZE*np.cos(theta))
        y = int(loc[1] + STEP_SIZE*np.sin(theta))
        add_node_to_heap(viz_window,loc,(x,y,theta_index))

# Processing the current node that was returned by popping the min-heap
def process_node(viz_window,node):
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
    pygame.draw.rect(viz_window,BLUE,(loc[0],loc[1],WIDTH_X,WIDTH_Y))
    if reached_goal(loc):
        solved = True
        compute_final_path(viz_window,loc)
    add_neighbors(viz_window,loc)
    iteration += 1
    if iteration%500 == 0:
        save_frame(viz_window)

# Wrapper function for the A* Algorithm
def a_star(viz_window):
    global min_heap, solved
    starting_dist = ((goal_pos[0]-start_pos[0])**2 + (goal_pos[1]-start_pos[1])**2)**0.5
    heappush(min_heap,(starting_dist,start_pos,start_pos))
    while not solved:
        node = heappop(min_heap)
        process_node(viz_window,node)

# Initializing start and goal positions using user input
def initialize_start_and_goal_pos(viz_window):
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
        if start_x < 0 or start_x >= GRID_SIZE[0] or start_y < 0 or start_y >= GRID_SIZE[1] or in_obstacle(viz_window,(start_x,start_y,start_theta)):
            print("Try again with a valid set of values")
            continue
        if goal_x < 0 or goal_x >= GRID_SIZE[0] or goal_y < 0 or goal_y >= GRID_SIZE[1] or in_obstacle(viz_window,(goal_x,goal_y,goal_theta)):
            print("Try again with a valid set of values")
            continue
        start_pos = (start_x,start_y,start_theta)
        goal_pos = (goal_x,goal_y,goal_theta)

# Initializing the map with obstacles in PyGame
def initialize_map(viz_window):
    wall1_bloated = [(0,0),(CLEARANCE,0),(CLEARANCE,500),(0,500)]
    wall2_bloated = [(0,0),(1200,0),(1200,CLEARANCE),(0,CLEARANCE)]
    wall3_bloated = [(1200-CLEARANCE,0),(1200,0),(1200,500),(1200-CLEARANCE,500)]
    wall4_bloated = [(0,500-CLEARANCE),(1200,500-CLEARANCE),(1200,500),(0,500)]
    obstacle1 = [(100,0),(175,0),(175,400),(100,400)]
    obstacle1_bloated = [(100-CLEARANCE,0),(175+CLEARANCE,0),(175+CLEARANCE,400+CLEARANCE),(100-CLEARANCE,400 + CLEARANCE)]
    obstacle2 = [(275,100),(350,100),(350,500),(275,500)]
    obstacle2_bloated = [(275-CLEARANCE,100-CLEARANCE),(350+CLEARANCE,100-CLEARANCE),(350+CLEARANCE,500),(275-CLEARANCE,500)]
    obstacle3 = [(650,100),(int(650+75*(3**0.5)),175),(int(650+75*(3**0.5)),325),(650,400),(int(650-75*(3**0.5)),325),(int(650-75*(3**0.5)),175)]
    obstacle3_bloated = [(650,100-CLEARANCE),(int(650+75*(3**0.5)+(CLEARANCE/2)*(3**0.5)),int(175-(CLEARANCE/2)*(3**0.5))),(int(650+75*(3**0.5)+(CLEARANCE/2)*(3**0.5)),int(325+(CLEARANCE/2)*(3**0.5))),(650,400+CLEARANCE),(int(650-75*(3**0.5)-(CLEARANCE/2)*(3**0.5)),int(325+(CLEARANCE/2)*(3**0.5))),(int(650-75*(3**0.5)-(CLEARANCE/2)*(3**0.5)),int(175-(CLEARANCE/2)*(3**0.5)))]
    obstacle4 = [(900,50),(1100,50),(1100,450),(900,450),(900,375),(1020,375),(1020,125),(900,125)]
    obstacle4_bloated = [(900-CLEARANCE,50-CLEARANCE),(1100+CLEARANCE,50-CLEARANCE),(1100+CLEARANCE,450+CLEARANCE),(900-CLEARANCE,450+CLEARANCE),(900-CLEARANCE,375-CLEARANCE),(1020-CLEARANCE,375-CLEARANCE),(1020-CLEARANCE,125+CLEARANCE),(900-CLEARANCE,125+CLEARANCE)]
    obstacles = [obstacle1,obstacle2,obstacle3,obstacle4]
    obstacles_bloated = [wall1_bloated,wall2_bloated,wall3_bloated,wall4_bloated,obstacle1_bloated,obstacle2_bloated,obstacle3_bloated,obstacle4_bloated]
    for obstacle in obstacles_bloated:
        pygame.draw.polygon(viz_window,GRAY,obstacle)
    for obstacle in obstacles:
        pygame.draw.polygon(viz_window,BLACK,obstacle)
    save_frame(viz_window)

def main():

    global node_states, start_pos, goal_pos, CLEARANCE, STEP_SIZE

    # Creating a directory to save the animation frames
    os.mkdir(FRAME_DIRECTORY)

    # Initializing the grid world as a pygame display window,
    pygame.display.set_caption('A* Path Finding Algorithm Visualization')
    viz_window = pygame.display.set_mode(GRID_SIZE,flags=pygame.HIDDEN)
    viz_window.fill(WHITE)
    save_frame(viz_window)

    # Taking the clearnace as a user-input
    CLEARANCE = int(input("Enter the clerance/bloating value for the obstales and walls: "))

    # Taking the step-size as a user-input
    STEP_SIZE = int(input("Enter the step-size in the A* Algorithm: "))

    # Initializing the map with obstacles
    initialize_map(viz_window)
    save_frame(viz_window)
    # Initializing the start and goal positions of the path from user input
    initialize_start_and_goal_pos(viz_window)

    print("\nStarting the Pathfinding Process! This may take upto a minute.\n")

    # Running the A* Algorithm
    start_time = time.time()
    a_star(viz_window)
    end_time = time.time()
    print("\nSuccess! Found the Optimal Path\n")
    print("\nTime taken for the pathfinding process using the A* Algorithm: ",end_time-start_time," seconds\n")

    #Making the video from the animation frames
    print("\nNow Generating the video for animation\n")
    make_video(FRAME_DIRECTORY,VIDEO_NAME)
    print("\nSaved the video as sample_video.mp4 in the same directory as this code file!\n")
    
    # Removing all the video frames
    shutil.rmtree(FRAME_DIRECTORY)

    # Playing the video
    print("\nPlaying the video animation of the path computation\n")
    play_video(VIDEO_NAME)

if __name__ == "__main__":
    main()