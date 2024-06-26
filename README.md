# A-Star-Path-Planner
An implementation of the A* Algorithm to plan optimal paths between a known starting and goal position in an obstacle-ridden 1200*500 grid world.

Link to the GitHub Repository can be found [here](https://github.com/vikrams169/A-Star-Path-Planner).

## Authors
<b>Vikram Setty</b>
<ul>
<li> UID: 119696897
<li> Directory ID: vikrams
</ul>
<b>Vinay Lanka</b>
<ul>
<li> UID: 12041665
<li> Directory ID: vlanka
</ul>

## Overview
On giving start and goal location coordinates, the path planner computes the shortest path using the A* Algorithm while avoiding obstacles (with a clearance of a used-specified input).

On finding the final path, the planner makes a video with intermediate frames and displays it as a pop-up animation.

A video simulating the planner computing the optimal trajectory can be found as `a_star.mp4`. Further, the final path visualization in the map looks as shown below. The black represents obstacles, gray represents obstacle and wall clearance, and the white areas are open space. The blue filled gridcells have already been explored by the planner and the final optimal path is shown in green.

<p align="center">
  <img src="sample_a_star_path.png"/>
</p>

This project majorly uses OpenCV for generating the visualiziation and OpenCV and ImageIO for displaying the animation.

## Dependencies
The dependencies for this Python 3 project include the following:
<ul>
<li> NumPy
<li> OpenCV
</ul>
They can be installed using the following commands.

```sh
    pip3 install numpy
    pip3 install opencv-python
```

## Running the Code
To run the code, execute the following command
```sh
    python3 a_star.py
```
On doing so, the terminal should prompt for the coordinate positions of start and goal locations which the user has to enter. Note a couple points:
<ul>
<li> Enter integer values
<li> Use the coordinate system considering the bottom-left of the window/map as the origin
<li> If any of the coordinate locations you have enetered is not valid i.e. out of map bounds, or within an obstacle/its clearance considering the radius of the robot (user-specified), you will be prompted to enter all the coordinate locations again till they are valid. Note that even the walls/boundaries of the grid world have a clearance based on a user-specified input.
</ul>

A sample set of start and goal positions (in the format [x,y,theta] with the orientation angle theta in degrees) to enter (that goes from one corner of the grid to the other) include the one below. This particular case can execute in about 2 minutes depending upon system specifications.
<ul>
<li> Clearance: 5
<li> Step Size: 10
<li> Robot Radius: 5
<li> Start Position: (10,10,0)
<li> Goal Position: (1190,490,30)
</ul>

After the program accepts your start and goal locations, it will start computing the path. Ater computing the final path, it will generate and display a video `a_star.mp4`from the saved frames and delete all the individual frames themselves. The total time taken to run the A* Algorithm will be displayed to the terminal as well.