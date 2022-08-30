# ROS-prioritized-exploration-turtlebot3

## Contributors
1. [Sayantan Datta](mailto:sdatta3@uncc.edu?subject=[GitHub]%20Source%20Prioritized%20Indoor%20Exploration)

Please email me for any bugs.

## Citation
If you are using this code, [please cite the paper.](https://ieeexplore.ieee.org/abstract/document/9636199) <br>

# Turtlebot3 Exploration

This package runs a turtlebot3 simulation over several simulation environments. The simulation environments were built in [Blender](https://www.blender.org/). 

## Installation

1. Install [ROS-noetic-full](http://wiki.ros.org/noetic/Installation)
2. Create a [catkin workspace](https://subscription.packtpub.com/book/hardware-and-creative/9781782175193/1/ch01lvl1sec11/creating-a-catkin-workspace) and pull the repository in the `src` directory.
3. Turtlebot package requires additional libraries, they are listed below.
    1. `sudo apt install ros-noetic-move-base ros-noetic-gmapping python3-networkx python3-scipy python3-skimage`<br>
4. Run `catkin_make` in the `catkin_workspace` directory.

Common problems are discussed [here](#Package-Dependencies-and-Common-Errors-without-them).


## Run sequence with launch files

If you want `Alice` behaviour change the launch files of `map_to_graph.launch` and 
`explore_environment.launch` before running the launch files. 

This runs the greedy version of the algorithms.

| Terminal | Command                                |
|----------|----------------------------------------|
|1         | `roscore` |
|2         | `roslaunch turtlebot3_exploration env_branched_corridor.launch` |
|3         | `roslaunch turtlebot3_slam turtlebot3_slam.launch slam_method:=gmapping`  |
|4         | `roslaunch turtlebot3_exploration map_to_graph.launch` |
|5         | `roslaunch turtlebot3_exploration explore_environment.launch`|

The exploration percentage shows up in the terminal executing the 'explore_environment.launch'.
**Note:** Sometimes, the exploration percentage may show up as greater than 100 percent. As the ground truth exploration might have traces of differently mapped regions (e.g., a wall mapped slightly thicker).

## How the exploration algorithm works

### Dependencies

1. Turtlebot slam package running a slam algorithm. Karto/Hector works better than gmapping. However, they require changed functions for `map_to_graph_coordinates` and `graph_to_map_coordinates`.
2. Turtlebot world with turtlebot3 in it.
3. The launch file for slam algorithm has a parameter to change the sensor scan distance. Does it work?


### ROS Nodes required to make

1. Make a node to identify the priority of the nodes.
2. Make a launch file that runs the required services.
	1. Service 1: Makes the map into a graph
	2. Service 2: Makes the graph into a weighted graph. (Requires the next service)
	3. Service 3: Makes a small region to a weight value.


## Problems and challenges

1. Could not get turtlebot navigation to work without a static map of the entire world.


### Package Dependencies and Common Errors without them

1. CMake Error at `/opt/ros/noetic/share/catkin/cmake/test/gtest.cmake:180 (add_executable)`: <br>
  add_executable cannot create target "test_transform_datatypes" because another target with the same name already exists.  The existing target is an executable created in source directory "/home/sayantan/catkin_ws/src/geometry2/tf2".  See documentation for policy CMP0002 for more details. <br>
**Solution**: [github memory](https://githubmemory.com/repo/ros/geometry/issues/213) resolved compile error by changing `geometry2/tf2/CMakelist.txt` to: <br>
`catkin_add_gtest(test_transform2_datatypes test/test_transform_datatypes.cpp)` <br>
`target_link_libraries(test_transform2_datatypes tf2  ${console_bridge_LIBRARIES})` <br>
`add_dependencies(test_transform2_datatypes ${catkin_EXPORTED_TARGETS})` <br>

2. If the error is similar to this: <br>
`...../devel/include/tf/FrameGraph.h:21:9: error: ‘FrameGraphRequest’ does not name a type; did you mean ‘FrameGraphResponse’?` <br>
   `21 | typedef FrameGraphRequest Request;` <br>
   `   |         ^~~~~~~~~~~~~~~~~` <br>
   `   |         FrameGraphResponse` <br>
**Solution: **Delete the build and devel directory in the catkin workspace and recompile using `catkin_make`. <br>
  
3. `RLException: Invalid <arg> tag: environment variable 'TURTLEBOT3_MODEL' is not set. ` <br>
`Arg xml is <arg name="model" default="$(env TURTLEBOT3_MODEL)" doc="model type [burger, waffle, waffle_pi]"/>`<br>
`The traceback for the exception was written to the log file`<br>
**Solution** export TURTLEBOT3\_MODEL=waffle\_pi<br>

4. Tire mesh missing: <br>
`process[rviz-3]: started with pid [469116]` <br>
`[ERROR] [1644502979.824366935, 31.340000000]: Could not load resource [package://turtlebot3_description/meshes/wheels/left_tire.stl]: Unable to open file "package://turtlebot3_description/meshes/wheels/left_tire.stl".` <br>
`[ERROR] [1644502979.824596531, 31.340000000]: Could not load resource [package://turtlebot3_description/meshes/wheels/right_tire.stl]: Unable to open file "package://turtlebot3_description/meshes/wheels/right_tire.stl".` <br>
**Solution**: Wheel mesh added. However file is not being tracked in github. So this needs to be done everytime. The files are available in the [turtlebot3 github repo](https://github.com/ROBOTIS-GIT/turtlebot3/tree/master/turtlebot3_description/meshes). Download and add the `wheels` directory  in the correct path where it says file is missing.<br>

5. `view_frames` does not work <br>
    (ros geometry bug)[https://github.com/ros/geometry/pull/193]
