# ROB301 Final Project – TurtleBot3 Localization & Control Node

This repository contains the Python ROS node developed for the ROB301 Final Design Project at the University of Toronto. The node enables a TurtleBot3 Waffle Pi to follow the hallway tape, detect office colors, localize itself on an 11-office loop, and perform the required mail-delivery routine.

## Overview
- **PID Line Following:** Uses `/line_idx` to track the white tape with proportional, integral, and derivative control, including deadband, hysteresis, anti-windup, and derivative filtering.
- **Bayesian Localization:** Maintains a probability distribution over hallway states and updates it using the project’s transition and measurement models to estimate the robot’s location.
- **Delivery Routine:** When the estimated state matches a target office with sufficient confidence, the robot performs a scripted delivery motion before resuming line following.

## File Structure
final_project.py # Main ROS node (PID control, localization, delivery)


## Usage
This node is intended to run in the course-provided TurtleBot3 ROS environment.  
Run using:
```bash
rosrun final_project final_project.py
