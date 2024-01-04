import numpy as np
import pkg_resources
import xml.etree.ElementTree as etxml
from gym_pybullet_drones.envs.single_agent_rl.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
import pybullet as p

class HoverAviary(BaseRLAviary):
    """Single agent RL problem: hover at position."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obstacles = True,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional"""  """"""  """
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps."""  """
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.DRONE_MODEL = drone_model
        self.URDF = self.DRONE_MODEL.value + ".urdf"
        self.M, \
        self.L, \
        self.THRUST2WEIGHT_RATIO, \
        self.J, \
        self.J_INV, \
        self.KF, \
        self.KM, \
        self.COLLISION_H,\
        self.COLLISION_R, \
        self.COLLISION_Z_OFFSET, \
        self.MAX_SPEED_KMH, \
        self.GND_EFF_COEFF, \
        self.PROP_RADIUS, \
        self.DRAG_COEFF, \
        self.DW_COEFF_1, \
        self.DW_COEFF_2, \
        self.DW_COEFF_3 = self._parseURDFParameters()
        self.NUM_DRONES = 1
        self.INIT_XYZS = np.vstack([np.array([x*4*self.L for x in range(self.NUM_DRONES)]), \
                                         np.array([y*4*self.L for y in range(self.NUM_DRONES)]), \
                                         np.ones(self.NUM_DRONES) * (self.COLLISION_H/2-self.COLLISION_Z_OFFSET+.1)]).transpose().reshape(self.NUM_DRONES, 3)
        self.INIT_RPYS = np.zeros((self.NUM_DRONES, 3))
        self.CLIENT = p.connect(p.DIRECT) 
        self.TARGET_POS = np.array([0,1,1])
        self.EPISODE_LEN_SEC = 8
        self.DRONE_IDS = np.array([p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/'+self.URDF),
                                              self.INIT_XYZS[i,:],
                                              p.getQuaternionFromEuler(self.INIT_RPYS[i,:]),
                                              flags = p.URDF_USE_INERTIA_FROM_FILE,
                                              physicsClientId=self.CLIENT
                                              ) for i in range(self.NUM_DRONES)])
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obstacles = obstacles,
                         obs=obs,
                         act=act
                         )

    ################################################################################
    def _parseURDFParameters(self):
        """Loads parameters from an URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.

        """
        URDF_TREE = etxml.parse(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/'+self.URDF)).getroot()
        M = float(URDF_TREE[1][0][1].attrib['value'])
        L = float(URDF_TREE[0].attrib['arm'])
        THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib['thrust2weight'])
        IXX = float(URDF_TREE[1][0][2].attrib['ixx'])
        IYY = float(URDF_TREE[1][0][2].attrib['iyy'])
        IZZ = float(URDF_TREE[1][0][2].attrib['izz'])
        J = np.diag([IXX, IYY, IZZ])
        J_INV = np.linalg.inv(J)
        KF = float(URDF_TREE[0].attrib['kf'])
        KM = float(URDF_TREE[0].attrib['km'])
        COLLISION_H = float(URDF_TREE[1][2][1][0].attrib['length'])
        COLLISION_R = float(URDF_TREE[1][2][1][0].attrib['radius'])
        COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
        COLLISION_Z_OFFSET = COLLISION_SHAPE_OFFSETS[2]
        MAX_SPEED_KMH = float(URDF_TREE[0].attrib['max_speed_kmh'])
        GND_EFF_COEFF = float(URDF_TREE[0].attrib['gnd_eff_coeff'])
        PROP_RADIUS = float(URDF_TREE[0].attrib['prop_radius'])
        DRAG_COEFF_XY = float(URDF_TREE[0].attrib['drag_coeff_xy'])
        DRAG_COEFF_Z = float(URDF_TREE[0].attrib['drag_coeff_z'])
        DRAG_COEFF = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])
        DW_COEFF_1 = float(URDF_TREE[0].attrib['dw_coeff_1'])
        DW_COEFF_2 = float(URDF_TREE[0].attrib['dw_coeff_2'])
        DW_COEFF_3 = float(URDF_TREE[0].attrib['dw_coeff_3'])
        return M, L, THRUST2WEIGHT_RATIO, J, J_INV, KF, KM, COLLISION_H, COLLISION_R, COLLISION_Z_OFFSET, MAX_SPEED_KMH, \
               GND_EFF_COEFF, PROP_RADIUS, DRAG_COEFF, DW_COEFF_1, DW_COEFF_2, DW_COEFF_3
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        #ret = max(0, 2 - np.linalg.norm(self.TARGET_POS-state[0:3])**4)
        ret = max(0, 1 - np.linalg.norm(self.TARGET_POS-state[0:3]))
        #ret = max(0, np.sqrt(2) - np.linalg.norm(self.TARGET_POS-state[0:3]))
        #
        return ret

    ################################################################################

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        #print(np.linalg.norm(self.TARGET_POS-state[0:3]))
        if np.linalg.norm(self.TARGET_POS-state[0:3]) < .001:
            return True
        else:
            return False

    ################################################################################

    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 2.0 # Truncate when the drone is too far away
             or abs(state[7]) > .4 or abs(state[8]) > .4 # Truncate when the drone is too tilted
        ):
            return True
        if self._check_collision_with_obstacle():
        # If collision detected, perform avoidance maneuver or return to target position
            self._avoid_obstacle()
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False
    def _get_drone_position(self):
        # Triển khai logic để lấy thông tin vị trí hiện tại của drone ở đây
        # Ví dụ: lấy thông tin vị trí từ trạng thái của drone hoặc hệ thống vận động
        drone_state = self._getDroneStateVector(0)  # Lấy trạng thái của drone
        drone_position = drone_state[0:3]  # Lấy thông tin vị trí từ trạng thái
        return drone_position
    
    def calculate_avoidance_action(self, drone_position, obstacle_position):
        avoidance_direction = drone_position - obstacle_position
        normalized_avoidance_direction = avoidance_direction / np.linalg.norm(avoidance_direction)
        avoidance_action = normalized_avoidance_direction * 10  # Điều chỉnh AVOIDANCE_FORCE theo cần thiết
        return avoidance_action

    def _apply_action(self, action):
        # Giả sử action đưa vào là một vectơ 4 chiều, có thể đại diện cho các giá trị điều khiển khác nhau
        # Ví dụ: action[0] có thể đại diện cho thông số điều khiển RPM

        # Lấy thông tin vị trí hiện tại của drone
        drone_position = self._get_drone_position()

        # Xác định vị trí của vật cản
        obstacle_position = np.array([0, 1, 1])  # Vị trí của vật cản

        # Tính toán khoảng cách giữa drone và vật cản
        distance_to_obstacle = np.linalg.norm(drone_position - obstacle_position)

        # Xác định khoảng cách an toàn để né vật cản
        safe_distance = 0.5  # Đây là một giá trị tùy ý, bạn có thể điều chỉnh nó

        # Nếu drone gần vật cản, thực hiện hành động né vật cản
        if distance_to_obstacle < safe_distance:
            # Thực hiện hành động né vật cản tại đây
            # Ví dụ: điều khiển drone di chuyển sang hướng khác để né vật cản
            avoidance_action = self.calculate_avoidance_action(drone_position, obstacle_position)
            self.set_drone_control_input(avoidance_action)
        else:
            # Nếu không gần vật cản, thực hiện hành động bình thường từ action được cung cấp
            self.set_drone_control_input(action)

    def send_control_to_motor(self, motor_rpm, motor_index):
        # Giả sử bạn sử dụng một API để điều khiển drone
        # Dưới đây là một mô phỏng về cách gửi giá trị RPM vào motor thích hợp của drone
        # Bạn cần thay thế phần mô phỏng này bằng API hoặc giao thức điều khiển thực tế
        mode = p.VELOCITY_CONTROL
        # motor_index = 0 # Example motor indices in PyBullet
        p.setJointMotorControl2(self.DRONE_IDS  ,  motor_index, mode, targetVelocity=500)
        # Ví dụ: Đoạn mã giả dưới đây mô phỏng việc gửi giá trị RPM vào motor
        # Thực tế, bạn cần sử dụng API của hệ thống điều khiển drone để điều khiển motor
        # Lệnh điều khiển thực tế có thể khác hoàn toàn tùy thuộc vào thiết bị và giao thức
        # api.send_rpm_to_motor(motor_index, rpm)

    def set_drone_control_input(self, action):
        num_motors = len(action)
        for i in range(num_motors):
        # Gửi giá trị điều khiển tới motor thứ i
            motor_rpm = action[i]  # Đây là giá trị điều khiển tương ứng với motor thứ i
        # Triển khai gửi giá trị điều khiển đến motor thứ i ở đây
        # Ví dụ: Gửi giá trị RPM vào motor thứ i
            self.send_control_to_motor(motor_rpm,  motor_index=i)
    
    
    ################################################################################
    def _check_collision_with_obstacle(self):
        drone_state = self._getDroneStateVector(0)  # Get the state of the drone
    # Assume obstacles are stored in a list or array
        obstacles = self._get_obstacle_position()  # Get positions of obstacles

    # Loop through all obstacles to check collision
        for obstacle in obstacles:
            distance_to_obstacle = np.linalg.norm(drone_state[0:3] - obstacle)
            if distance_to_obstacle < 10:
                return True  # Collision detected

        return False
    def _avoid_obstacle(self):
        drone_state = self._getDroneStateVector(0)  # Get the state of the drone
        obstacles = self._get_obstacle_position()  # Get positions of obstacles


    # Find the closest obstacle
        closest_obstacle = None
        closest_distance = float('inf')
        for obstacle in obstacles:
            distance_to_obstacle = np.linalg.norm(drone_state[0:3] - obstacle)
            if distance_to_obstacle < closest_distance:
                closest_obstacle = obstacle
                closest_distance = distance_to_obstacle

    # Perform maneuver to avoid the closest obstacle
        avoidance_direction = drone_state[0:3] - closest_obstacle
        normalized_avoidance_direction = avoidance_direction / np.linalg.norm(avoidance_direction)
        avoidance_action = normalized_avoidance_direction * 10 # Adjust AVOIDANCE_FORCE as needed

    # Apply the avoidance action to the drone
        self._apply_action(avoidance_action)
    def _return_to_target_position(self):
        drone_state = self._getDroneStateVector(0)  # Get the state of the drone
        target_position = self.TARGET_POS  # Get the target position

    # Calculate direction to return to target position
        return_direction = target_position - drone_state[0:3]
        normalized_return_direction = return_direction / np.linalg.norm(return_direction)
        return_action = normalized_return_direction * 10  # Adjust RETURN_FORCE as needed

    # Apply the action to the drone to return to the target position
        self._apply_action(return_action)
    
    def _get_obstacle_position(self):
    # Logic để lấy thông tin vị trí các vật cản
        obstacle_positions = [
            np.array([0,1,1])
        # ...
        ]
        return obstacle_positions

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years