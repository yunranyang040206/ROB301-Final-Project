#!/usr/bin/env python
import rospy
import math
from geometry_msgs.msg import Twist
from std_msgs.msg import UInt32, Float64MultiArray
import numpy as np
import colorsys

import time
from std_msgs.msg import Float64

from typing import List, Optional, Dict

def fmt_vec(v: np.ndarray, precision: int = 4) -> str:
    return "[" + ", ".join(f"{x:.{precision}f}" for x in v) + "]"

def print_matrix(M: np.ndarray, name: str):
    print(f"{name} shape={M.shape}")
    for i, row in enumerate(M):
        print("  " + " ".join(f"{x: .3f}" for x in row))
    print()

class BayesLoc:
    def __init__(self, p0, colour_codes, colour_map, measurement_prob, IDX):
        """Bayes localization parameters 
        """
        self.num_states = len(p0)
        self.visited_deliveries = set()
        self.colour_codes = colour_codes
        self.colour_map = colour_map
        self.probability = p0
        self.state_prediction = np.ones(self.num_states, dtype=float) / self.num_states

        self.cur_colour = None  # most recent measured colour
        self.obs = None
        self.u = 1
        self.MEAS = measurement_prob
        self.IDX = IDX
        self.first_encount = True
        self.delivery = 0

        """Controller parameters
        """
        # publish motor commands
        self.cmd_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)

        # --------------------
        # parameters
        # --------------------
        self.image_width   = rospy.get_param("~image_width", 640)
        self.deadband_px   = rospy.get_param("~deadband_px", 8)
        self.forward_speed = rospy.get_param("~forward_speed", 0.04)
        self.turn_speed    = rospy.get_param("~turn_speed", 4)         # kept but not used as a limit
        self.hysteresis_px = rospy.get_param("~hysteresis_px", 10)

        # PID gains (keep your names)
        self.gain           = rospy.get_param("~kp", 0.004)            # Kp 0.006
        self.integral_gain  = rospy.get_param("~ki", 0.00001)          # Ki 0.00001
        self.derivative_gain= rospy.get_param("~kd", 0.0001)           # Kd 0.0004

        # Anti-windup and D filter
        self.i_limit = rospy.get_param("~i_limit", 2000.0)
        self.d_tau   = rospy.get_param("~d_tau", 0.05)


        # --- Gap crossing parameters ---
        self.lost_frames_to_hold = rospy.get_param("~lost_frames_to_hold", 3)   # frames until we declare "lost"
        self.gap_hold_time       = rospy.get_param("~gap_hold_time", 0.40)      # seconds to hold heading
        self.sweep_time          = rospy.get_param("~sweep_time", 2.0)          # seconds to sweep search
        self.hold_speed          = rospy.get_param("~hold_speed", 0.10)         # slower forward speed in GAP_HOLD
        self.sweep_speed         = rospy.get_param("~sweep_speed", 0.06)        # slow crawl in SWEEP
        self.hold_decay_tau      = rospy.get_param("~hold_decay_tau", 0.35)     # exp decay to straight
        self.sweep_omega         = rospy.get_param("~sweep_omega", 0.8)         # rad/s sweep magnitude
        self.omega_tau           = rospy.get_param("~omega_tau", 0.06)

        # loss handling
        self.state = "TRACK"  # TRACK, GAP_HOLD, SWEEP
        self.lost_count = 0
        self.loss_start_time = None
        self.hold_omega0 = 0.0  # omega at the moment we entered GAP_HOLD

        # command smoothing
        self.omega_cmd_filt = 0.0     

        # --------------------
        # state
        # --------------------
        self.center_px  = self.image_width // 2
        self.last_px    = None
        self.last_error = 0.0
        self.integral   = 0.0
        self.d_filt     = 0.0
        self.last_time  = rospy.get_time()
        self.avg_rgb = None

        # --- Go-straight-on-loss (instant) ---
        self.straight_speed    = rospy.get_param("~straight_speed", 0.10)
        self.max_straight_time = rospy.get_param("~max_straight_time", 1.0)  # safety cap (s)
        self._straight_start   = None
    
        # topic to detected line index
        topic_name_line = rospy.get_param("~line_topic", "line_idx")
        self.line_sub  = rospy.Subscriber(topic_name_line, UInt32, self.camera_callback, queue_size=10)
        topic_name_rgb = rospy.get_param("~line_topic", "mean_img_rgb")
        self.image_sub = rospy.Subscriber(topic_name_rgb, Float64MultiArray, self.camera_callback_image, queue_size=10)

    def camera_callback_image(self, msg):
        try:
            self.cur_colour = np.array(msg.data, dtype = float)  # [r, g, b]
        except Exception:
            self.cur_colour = None
        
        # print(self.cur_colour)
    # def colour_callback(self, msg):
    #     """
    #     callback function that receives the most recent colour measurement from the camera.
    #     """
    #     self.cur_colour = np.array(msg.data, dtype = float)  # [r, g, b]
    #     print(self.cur_colour)
    #     # update self.obs with colour string with observed rgb

    def camera_callback(self, msg):
        """callback for line index"""
        try:
            self.last_px = int(msg.data)
        except Exception:
            self.last_px = None

    def wait_for_colour(self):
        """Loop until a colour is received."""
        rate = rospy.Rate(100)
        while not rospy.is_shutdown() and self.cur_colour is None:
            rate.sleep()

    def transition_matrix(self) -> np.ndarray:
        """
        Build the N x N motion matrix T for a ring.
        T[i, j] = P(x_t = i | x_{t-1} = j, u_t = u).
        """
        u = self.u
        n = self.num_states
        if u == +1:    p_left, p_stay, p_right = 0.05, 0.1, 0.85 #0.05, 0.10, 0.85
        elif u == 0:   p_left, p_stay, p_right = 0.05, 0.90, 0.05 #0.05, 0.90, 0.05
        elif u == -1:  p_left, p_stay, p_right = 0.85, 0.10, 0.05 #0.85, 0.10, 0.05
        else:
            raise ValueError("u must be -1, 0, or +1")

        T = np.zeros((n, n), dtype=float)
        for j in range(n):
            i_left  = (j - 1) % n
            i_stay  = j
            i_right = (j + 1) % n
            T[i_left,  j] = p_left
            T[i_stay,  j] = p_stay
            T[i_right, j] = p_right
        return T

    def measurement_likelihood(self) -> np.ndarray:
        """
        Return L vector of length N, where L[i] = P(z | x=i).
        If z is None or not recognized, treat as 'nothing' (uninformative).
        """
        def classify_colour(rgb):
            # Standard colour centroids
            colours = {
                "yellow": np.array([161.09, 145.95, 133.85]),
                "green":  np.array([158.34, 148.19, 140.23]),
                "purple": np.array([159.91, 138.57, 164.17]),
                "orange": np.array([155.62, 135.64, 136.98]),
                "nothing": np.array([107.38, 102.16, 116.88])
            }

            rgb = np.array(rgb)
            # compute distances
            distances = {name: np.linalg.norm(rgb - centroid)
                        for name, centroid in colours.items()}

            # return the closest colour
            self.obs = min(distances, key=distances.get)
            return min(distances, key=distances.get)
            # if rgb[0] <= 170.0:
            #     return 'green'
            # elif 170.0 < rgb[0] <= 190.0:
            #     return 'yellow'
            # elif 190.0 < rgb[0] <= 215.0:
            #     return 'purple'
            # elif 215.0 < rgb[0]:
            #     return 'orange'
        
        z = classify_colour(self.cur_colour)
        # self.obs = z
        print(self.cur_colour, self.obs)
        world = self.colour_map
        if z == 'nothing':
            return None
        #     z = None
        # if z is None:
        #     return np.ones(len(world), dtype=float)
        if z not in self.MEAS:
            z = 'nothing'
        L = np.empty(len(world), dtype=float)
        for i, true_color in enumerate(world):
            L[i] = self.MEAS[z][ self.IDX[true_color] ]
        return L
    
    def bayes_step(self):
        """
        bel_prev: posterior from previous step (or prior at k=0)
        u: control at current step
        z: measurement at current step
        Returns bel_post: posterior after predict+update
        """
        bel_prev = self.state_prediction #np.ndarray
        u = self.u # speed int
        z = self.obs # str of colour
        world = self.colour_map # list[str]

        # (1) Predict: bel^- = T(u) @ bel_prev
        T = self.transition_matrix()

        bel_pred = T @ bel_prev
        
        # (2) Likelihood: L[i] = P(z | x=i)
        L = self.measurement_likelihood()

        if L is None:
            return [], 0

        # (3) Update (unnormalized): bel_unnorm = bel^- * L
        bel_unnorm = bel_pred * L

        # (4) Normalize
        s = bel_unnorm.sum()
        if s <= 0:
            # Guard fallback if everything was zero
            bel_post = np.ones_like(bel_unnorm) / len(bel_unnorm)
        else:
            bel_post = bel_unnorm / s

        # (5) Report MAP estimate
        map_idx = int(np.argmax(bel_post)) # can be used to make final prediction
        self.probability = bel_post
        return bel_post, map_idx

    # def state_model(self, u):
    #     """
    #     State model: p(x_{k+1} | x_k, u)

    #     TODO: complete this function
    #     """

    # def measurement_model(self, x):
    #     """
    #     Measurement model p(z_k | x_k = colour) - given the pixel intensity,
    #     what's the probability that of each possible colour z_k being observed?
    #     """
    #     if self.cur_colour is None:
    #         self.wait_for_colour()

    #     prob = np.zeros(len(self.colour_codes))

    #     """
    #     TODO: You need to compute the probability of states. You should return a 1x5 np.array
    #     Hint: find the euclidean distance between the measured RGB values (self.cur_colour)
    #         and the reference RGB values of each colour (self.ColourCodes).
    #     """

    #     return prob

    def do_delivery_turn(self):
        """
        Turn right 90 degrees, pause, then turn back left 90 degrees.
        """
        turn_speed = 0.5              # rad/s
        angle = math.pi / 2           # 90 degrees
        turn_time = angle / turn_speed

        twist = Twist()
        twist.linear.x = self.forward_speed
        twist.angular.z = 0.012
        start_time = rospy.Time.now().to_sec()
        while rospy.Time.now().to_sec() - start_time  < 4:
            self.cmd_pub.publish(twist)
            rate.sleep()

        # 1) Turn right 90°
        twist.linear.x = 0.0
        twist.angular.z = -turn_speed   # right is negative
        self.cmd_pub.publish(twist)
        start_time = rospy.Time.now().to_sec()
        while rospy.Time.now().to_sec() - start_time  < turn_time:
            self.cmd_pub.publish(twist)
            rate.sleep()


        # 3) Turn back left 90°
        twist.angular.z = +turn_speed   # left
        start_time = rospy.Time.now().to_sec()
        while rospy.Time.now().to_sec() - start_time  < turn_time:
            self.cmd_pub.publish(twist)
            rate.sleep()

        # 4) Stop rotation
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

    def run_filter(self, deliver = [3, 8, 11]) -> np.ndarray:
        bel, map_idx = self.bayes_step()
        if bel != []:
            self.state_prediction = bel
            world = self.colour_map
            self.delivery = map_idx+2
            print(self.state_prediction)
            print(f"\nMAP index = {map_idx+2},  MAP prob = {bel[map_idx]:.4f}")
            print(f"MAP position color = {world[map_idx]}")
            if (self.delivery in deliver
                and bel[map_idx] >= 0.45
                and self.delivery not in self.visited_deliveries):
                # remember we handled this location
                self.visited_deliveries.add(self.delivery)

                # perform the 90° right + 90° back turn
                self.do_delivery_turn()


    def follow_the_line(self):
        rate = rospy.Rate(30)
        stop_twist = Twist()
        duration = rospy.Duration(2)
        spot = 0
        line = True
        
        # line: 144.22, 131.12, 134.12
        # yellow: 175.41, 153.18, 131.60
        # green: 154.12, 163.95, 150.97
        # purple: 202.00, 129.62, 176.14
        # orange: 249.71, 138.42, 80.52
        # while not rospy.is_shutdown():
        now = rospy.get_time()
        dt = max(1e-3, now - self.last_time)

        # print(self.last_px)

        def is_line(self, thres = 50):
            if self.cur_colour is None:
                return True
            white_ref = np.array([148.5, 135.7, 140.9])

            diff = self.cur_colour - white_ref
            distance = np.linalg.norm(diff)
            if distance < thres:
                return True
            return False

        if self.last_px is None or self.cur_colour is None:
            stop_twist.linear.x = self.forward_speed
            stop_twist.angular.z = 0.0
            self.cmd_pub.publish(stop_twist)
            rate.sleep()
        elif is_line(self) and self.cur_colour[1] < 145 and self.cur_colour[0] < 155: #133< self.avg_rgb[0] < 155 and 120 < self.avg_rgb[1] < 140 and 125 < self.avg_rgb[2] < 145:
            self.first_encount = True
            # time step
            now = rospy.get_time()
            dt = max(1e-3, now - self.last_time)   # protect against tiny/zero dt

            # position error (center - measurement)
            err = self.center_px - self.last_px

            # deadband with hysteresis: zero small errors, keep sign around boundary
            if abs(err) <= self.deadband_px:
                err_eff = 0.0
            elif abs(err) <= self.deadband_px + self.hysteresis_px:
                # small buffer to avoid chattering
                err_eff = self.last_error
            else:
                err_eff = err

            # I term (with dt and anti-windup). Freeze inside deadband to avoid bias accumulation.
            if err_eff != 0.0:
                self.integral += err_eff * dt
                # clamp integral
                if self.integral > self.i_limit:
                    self.integral = self.i_limit
                elif self.integral < -self.i_limit:
                    self.integral = -self.i_limit

            # D term on error with first-order low-pass (reduces noise & derivative kick)
            de_dt  = (err_eff - self.last_error) / dt
            alpha  = dt / (self.d_tau + dt)        # 0 < alpha <= 1
            self.d_filt = (1.0 - alpha) * self.d_filt + alpha * de_dt

            omega = (
                self.gain * err_eff
                + self.integral_gain * self.integral
                + self.derivative_gain * self.d_filt
            )

            cmd = Twist()
            cmd.linear.x  = self.forward_speed
            cmd.angular.z = omega
            self.cmd_pub.publish(cmd)

            # update state
            self.last_error = err_eff
            self.last_time  = now
                        
        else:
            stop_twist.linear.x = self.forward_speed
            stop_twist.angular.z = 0.012
            self.cmd_pub.publish(stop_twist)
            if self.first_encount:
                self.first_encount = False
                return True
        
        return False

    # def state_predict(self):
    #     rospy.loginfo("predicting state")
    #     """
    #     TODO: Complete the state prediction function: update
    #     self.state_prediction with the predicted probability of being at each
    #     state (office)
    #     """

    # def state_update(self):
    #     rospy.loginfo("updating state")
    #     """
    #     TODO: Complete the state update function: update self.probabilities
    #     with the probability of being at each state
        # """


    

    


if __name__ == "__main__":
    # This is the known map of offices by colour
    # 0: red, 1: green, 2: blue, 3: yellow, 4: line
    # current map starting at cell #2 and ending at cell #12
    rospy.init_node("final_project")
    # colour_map = [3, 0, 1, 2, 2, 0, 1, 2, 3, 0, 1]

    colour_map = [
    'yellow', 'green', 'purple', 'orange', 'orange',
    'green', 'purple', 'orange', 'yellow', 'green', 'purple'
    ] # from 2 to 12

    COLORS = ['purple', 'green', 'yellow', 'orange'] # colour_codes
    IDX: Dict[str, int] = {c: i for i, c in enumerate(COLORS)}

    # Table 2: P(z | true_color)  (rows keyed by measured z)
    MEAS: Dict[str, List[float]] = {
        'purple':[0.60, 0.20, 0.05, 0.05],
        'green':[0.20, 0.60, 0.05, 0.05],
        'yellow':[0.05, 0.05, 0.65, 0.20],
        'orange':[0.05, 0.05, 0.15, 0.60],
        'nothing':[0.10, 0.10, 0.10, 0.10],
    }

    # calibrate these RGB values to recognize when you see a colour
    # you may find it easier to compare colour readings using a different
    # colour system, such as HSV (hue, saturation, value). To convert RGB to
    # HSV, use:
    # h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    colour_codes = [
        [167, 146, 158],  # red
        [163, 184, 100],  # green
        [173, 166, 171],  # blue
        [167, 170, 117],  # yellow
        [150, 150, 150],  # line
    ]

    # initial probability of being at a given office is uniform
    p0 = np.ones(len(colour_map), dtype = float) / len(colour_map)

    localizer = BayesLoc(p0, colour_codes, colour_map, measurement_prob = MEAS, IDX = IDX)

    rospy.sleep(0.5)
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        """
        TODO: complete this main loop by calling functions from BayesLoc, and
        adding your own high level and low level planning + control logic
        """

        pre = localizer.follow_the_line()

        if pre:
            localizer.run_filter()
        rate.sleep()

    rospy.loginfo("finished!")
    rospy.loginfo(localizer.probability)
