#!/usr/bin/env python3
"""
Bimanual robot control with inference mode support.

"""

import sys
import time
import subprocess
import argparse
import yaml
import os
import numpy as np
import signal
from pathlib import Path

# Optional imports for simulation
try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("Warning: mujoco not available. Simulation will be disabled.")

try:
    import rerun as rr
    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False
    print("Warning: rerun not available. Visualization will be limited.")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plotting will be disabled.")

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Compute paths relative to this file's location
current_file_path = os.path.dirname(os.path.abspath(__file__))
# Go up 3 levels: inference -> bimanual -> ml -> robot-os
robot_os_root = os.path.abspath(os.path.join(current_file_path, "..", "..", ".."))
thirdparty_i2rt_path = os.path.join(robot_os_root, "robot-os", "thirdparty", "i2rt")
bimanual_path = os.path.join(robot_os_root, "ml", "bimanual")

# pi dependencies - using the new inference API
from openpi.training import config as _config
from openpi.policies import i2rt_policy
from openpi.policies import policy_config as _policy_config
# i2rt dependencies
sys.path.insert(0, thirdparty_i2rt_path)
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import GripperType

# Add parent directory to path for absolute imports
sys.path.insert(0, bimanual_path)
from i2rt_robot import I2RTRobot, PortalFollowerClient

from config import I2RTFollowerConfig

# ==================== MODE SELECTION ====================
# Set to "observation_only" for observation-only mode (no actuation)
# Set to "full_control" for full control mode with actuation
# Can also be set via command-line argument --mode
DEFAULT_MODE = "observation_only"  # Options: "observation_only" or "full_control"
# =======================================================


def check_can_interface(interface: str) -> bool:
    """Check if a CAN interface exists and is available"""
    try:
        result = subprocess.run(['ip', 'link', 'show', interface],
                              capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return False
        if 'state UP' in result.stdout or 'state UNKNOWN' in result.stdout:
            return True
        else:
            print(f"Warning: CAN interface {interface} exists but is not UP")
            return False
    except Exception as e:
        print(f"Error checking CAN interface {interface}: {e}")
        return False

def check_all_can_interfaces() -> bool:
    """Check if all required CAN interfaces exist"""
    required_interfaces = [
        'can_follower_r',
        'can_leader_r',
        'can_follower_l',
        'can_leader_l'
    ]
    
    missing_interfaces = []
    for interface in required_interfaces:
        if not check_can_interface(interface):
            missing_interfaces.append(interface)
    
    if missing_interfaces:
        raise RuntimeError(f"Missing or unavailable CAN interfaces: {', '.join(missing_interfaces)}")
    
    print("✓ All CAN interfaces are available")
    return True

def launch_gello_process(can_channel, gripper, server_port):
    """Launch a single follower gello process"""
    python_path = "python"
    script_path = os.path.join(thirdparty_i2rt_path, "scripts", "minimum_gello.py")
    
    cmd = [python_path, script_path,
           "--can_channel", can_channel,
           "--gripper", gripper,
           "--mode", "follower",
           "--server_port", str(server_port)]
    
    print(f"Starting: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(cmd)
        return process
    except Exception as e:
        print(f"Error starting process for {can_channel}: {e}")
        return None

def robot_obs_to_numpy(image):
    """Convert robot observation image to numpy format expected by policy"""
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    # If image is already in HWC format with uint8, return as is
    if len(image.shape) == 3 and image.shape[2] == 3:
        if image.dtype == np.uint8:
            return image
        else:
            # If float, scale to uint8
            return (image * 255).astype(np.uint8)
    
    # Otherwise assume it's in CHW format and needs conversion
    if len(image.shape) == 3 and image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        return image
    
    return image

class YAMLeaderRobot:
    """Wrapper for YAM leader robot with button and gripper support"""
    def __init__(self, robot):
        self._robot = robot
        self._motor_chain = robot.motor_chain

    def get_info(self):
        """Get joint positions and button states"""
        qpos = self._robot.get_observations()["joint_pos"]
        encoder_obs = self._motor_chain.get_same_bus_device_states()
        time.sleep(0.01)
        gripper_cmd = 1 - encoder_obs[0].position
        qpos_with_gripper = np.concatenate([qpos, [gripper_cmd]])
        return qpos_with_gripper, encoder_obs[0].io_inputs

    def command_joint_pos(self, joint_pos: np.ndarray):
        """Command joint positions (without gripper)"""
        assert joint_pos.shape[0] == 6
        self._robot.command_joint_pos(joint_pos)

    def update_kp_kd(self, kp: np.ndarray, kd: np.ndarray):
        """Update PD gains"""
        self._robot.update_kp_kd(kp, kd)

class BimanualTeleopWithInference:
    """
    Manages bimanual teleoperation with inference mode using policy_api.
    
    Button 0: Toggle sync/unsync (teleoperation) - only in full_control mode
    Button 1: Toggle inference mode (only when unsynced) - only in full_control mode
    """
    def __init__(self, config_name: str = "pi05_i2rt_lora", checkpoint_dir: str = "/home/i2rt/openpi/checkpoints/pi05_i2rt_lora/pi_lora_train/20000", bilateral_kp: float = 0.0, mode: str = "observation_only", sim: bool = False, mjcf_path: str = None, dataset_dir: str = None, episode_idx: int = 0, use_rerun: bool = False, max_actions_per_inference: int = None, execution_hz: float = 30.0, action_diff_threshold: float = 0.2):
        self.bilateral_kp = bilateral_kp
        self.mode = mode  # "observation_only" or "full_control"
        self.sim = sim
        self.mjcf_path = mjcf_path
        self.dataset_dir = dataset_dir
        self.episode_idx = episode_idx
        self.use_rerun = use_rerun and RERUN_AVAILABLE
        # Automatically enable plotting when in simulation mode
        self.plot_actions = sim and MATPLOTLIB_AVAILABLE
        # Inference execution parameters
        self.max_actions_per_inference = max_actions_per_inference
        self.execution_hz = execution_hz
        self.action_diff_threshold = action_diff_threshold
        
        # Store actions for plotting (always when in sim mode)
        if self.plot_actions:
            self.predicted_actions = []
            self.ground_truth_actions = []
            self.action_timestamps = []
            # Create plots directory
            self.plots_dir = Path("action_plots")
            self.plots_dir.mkdir(exist_ok=True)
            print(f"Action plots will be saved to: {self.plots_dir.absolute()}")
        
        if self.mode not in ["observation_only", "full_control"]:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'observation_only' or 'full_control'")
        
        print(f"Running in mode: {self.mode}")
        if self.sim:
            print("  - Simulation mode: Will replay episode in MuJoCo")
            if not MUJOCO_AVAILABLE:
                raise ImportError("MuJoCo is not available. Install with: pip install mujoco")
            if self.mjcf_path is None or not Path(self.mjcf_path).exists():
                raise FileNotFoundError(f"MuJoCo XML file not found at {self.mjcf_path}")
            if self.dataset_dir is None:
                raise ValueError("--dataset_dir is required when --sim is enabled")
        elif self.mode == "observation_only":
            print("  - Observation-only mode: Will receive observations, run inference, and print actions (NO ACTUATION)")
        else:
            print("  - Full control mode: Will actuate robots based on button controls and inference")
        
        # State flags
        self.synchronized = False
        self.inference_mode = False
        self.button0_prev = False
        self.button1_prev = False
        
        # Load inference model using the new API
        print("Loading inference model...")
        print(f"  Config name: {config_name}")
        print(f"  Checkpoint dir: {checkpoint_dir}")
        
        config = _config.get_config(config_name)
        self.policy = _policy_config.create_trained_policy(config, checkpoint_dir)
        print("✓ Inference model loaded successfully")

        # REAL ROBOT SETUP (skip if in simulation mode)
        if not self.sim:
            robot_cfg = I2RTFollowerConfig()
            self.inference_robot = I2RTRobot(robot_cfg)
            self.inference_robot.connect()
            print("Inference robot connected!")
        else:
            self.inference_robot = None
            # Load MuJoCo model for simulation
            self.mj_model = mujoco.MjModel.from_xml_path(str(self.mjcf_path))
            self.mj_data = mujoco.MjData(self.mj_model)
            # Don't initialize renderer - we don't need it for simulation replay
            # (Renderer requires display/OpenGL which may not be available in headless environments)
            self.mj_renderer = None
            self.use_renderer = False
            print(f"MuJoCo model loaded from: {self.mjcf_path}")
            
            # Load dataset for simulation
            print(f"Loading dataset from: {self.dataset_dir}")
            # Try to resolve repo_id and root
            if os.path.isdir(self.dataset_dir):
                # Local directory
                repo_id = os.path.basename(self.dataset_dir)
                root = self.dataset_dir
            else:
                # Assume it's a repo_id
                repo_id = self.dataset_dir
                root = None
            self.sim_dataset = LeRobotDataset(repo_id=repo_id, root=root, episodes=[self.episode_idx])
            print(f"Dataset loaded: episode {self.episode_idx}")
            
            # Get episode metadata
            episode_meta = self.sim_dataset.meta.episodes[self.episode_idx]
            self.sim_from_idx = int(float(episode_meta["dataset_from_index"]))
            self.sim_to_idx = int(float(episode_meta["dataset_to_index"]))
            self.sim_fps = float(self.sim_dataset.meta.fps)
            self.sim_dt = self.mj_model.opt.timestep
            print(f"Episode range: {self.sim_from_idx} to {self.sim_to_idx}")
            print(f"Dataset FPS: {self.sim_fps}")
            
            # Simulation state
            self.sim_frame_idx = 0
            self.sim_action_buffer = []
            self.sim_action_index = 0
            
            # Initialize Rerun if requested
            if self.use_rerun:
                rr.init(f"bimanual_inference/episode_{self.episode_idx}", spawn=True)
                print("Rerun visualization initialized")
        
        if self.mode == "full_control":
            # Setup leader and follower robots for full control mode
            gripper_type = GripperType.from_string_name("yam_teaching_handle")
            self.leader_right = YAMLeaderRobot(get_yam_robot(channel="can_leader_r", gripper_type=gripper_type))
            self.leader_left = YAMLeaderRobot(get_yam_robot(channel="can_leader_l", gripper_type=gripper_type))
            
            # Setup follower clients (for direct teleoperation)
            self.follower_right = PortalFollowerClient("127.0.0.1", 1234)
            self.follower_left = PortalFollowerClient("127.0.0.1", 1235)
            
            # Store initial kp values for bilateral control
            self.robot_kp_right = self.leader_right._robot._kp
            self.robot_kp_left = self.leader_left._robot._kp
        else:
            # Observation-only mode: leader and follower robots not needed
            self.leader_right = None
            self.leader_left = None
            self.follower_right = None
            self.follower_left = None
            self.robot_kp_right = None
            self.robot_kp_left = None
            print("  Leader and follower robots not initialized (observation-only mode)")

        # END REAL ROBOT SETUP
        
        # For real robot inference-execution cycle
        self.action_buffer = []  # Stores actions from inference
        self.action_index = 0    # Tracks which action to execute next
        self.last_execution_time = 0  # For execution timing
        
        # Set default max_actions_per_inference if not provided
        # Default to action_horizon from policy config, or 12 if not available
        if self.max_actions_per_inference is None:
            try:
                # Try to get action_horizon from policy config
                action_horizon = getattr(self.policy, 'action_horizon', None)
                if action_horizon is None:
                    # Try to get from model config
                    if hasattr(self.policy, 'config'):
                        action_horizon = getattr(self.policy.config, 'action_horizon', 12)
                    else:
                        action_horizon = 12
                self.max_actions_per_inference = action_horizon
            except Exception:
                self.max_actions_per_inference = 12
        
        print(f"Inference parameters: max_actions_per_inference={self.max_actions_per_inference}, execution_hz={self.execution_hz}, action_diff_threshold={self.action_diff_threshold}")

    def connect_inference_robot(self):
        """Connect the inference robot to read observations"""
        if self.inference_robot is not None:
            self.inference_robot.connect()
            print("Inference robot connected!")

    def slow_move_to_leader(self, leader_pos_right, leader_pos_left, duration: float = 1.0):
        """Slowly move followers to match leader positions"""
        current_follower_right = self.follower_right.get_joint_pos()
        current_follower_left = self.follower_left.get_joint_pos()
        
        steps = 100
        for i in range(steps):
            alpha = i / steps
            
            # Interpolate positions
            target_right = leader_pos_right * alpha + current_follower_right * (1 - alpha)
            target_left = leader_pos_left * alpha + current_follower_left * (1 - alpha)
            
            self.follower_right.command_joint_pos(target_right)
            self.follower_left.command_joint_pos(target_left)
            
            time.sleep(0.03)

    def handle_button_events(self, current_button):
        """Handle button press events with debouncing"""
        if self.mode == "observation_only":
            # Button handling disabled in observation-only mode
            return
            
        button0_pressed = current_button[0] > 0.5
        button1_pressed = current_button[1] > 0.5

        # Button 0: Toggle sync/unsync
        if button0_pressed and not self.button0_prev:
            if self.inference_mode:
                # If inference is active, just disable it and go to unsync
                print("🤖 Inference mode DISABLED (Button 0 pressed during inference)")
                self.inference_mode = False
                self.synchronized = False
                
                # Clear bilateral PD
                if self.leader_right is not None and self.leader_left is not None:
                    self.leader_right.update_kp_kd(kp=np.zeros(6), kd=np.zeros(6))
                    self.leader_left.update_kp_kd(kp=np.zeros(6), kd=np.zeros(6))
            else:
                # Normal sync toggle
                self.synchronized = not self.synchronized
                
                if self.synchronized:
                    print("🔗 SYNC mode activated")
                    # Get current positions
                    leader_pos_right, _ = self.leader_right.get_info()
                    leader_pos_left, _ = self.leader_left.get_info()
                    
                    # Set bilateral PD
                    self.leader_right.update_kp_kd(
                        kp=self.robot_kp_right * self.bilateral_kp, 
                        kd=np.zeros(6)
                    )
                    self.leader_left.update_kp_kd(
                        kp=self.robot_kp_left * self.bilateral_kp, 
                        kd=np.zeros(6)
                    )
                    
                    # Command leaders to current position
                    self.leader_right.command_joint_pos(leader_pos_right[:6])
                    self.leader_left.command_joint_pos(leader_pos_left[:6])
                    
                    # Slowly move followers to leader
                    self.slow_move_to_leader(leader_pos_right, leader_pos_left)
                else:
                    print("🔓 UNSYNC mode - robot stopped")
                    # Clear bilateral PD
                    if self.leader_right is not None and self.leader_left is not None:
                        self.leader_right.update_kp_kd(kp=np.zeros(6), kd=np.zeros(6))
                        self.leader_left.update_kp_kd(kp=np.zeros(6), kd=np.zeros(6))
        
        # Button 1: Toggle inference (only when unsynced)
        if button1_pressed and not self.button1_prev:
            if not self.synchronized:
                self.inference_mode = not self.inference_mode
                
                if self.inference_mode:
                    print("🤖 INFERENCE mode activated")
                else:
                    print("🛑 INFERENCE mode deactivated - robot stopped")
            else:
                print("⚠️  Cannot activate inference while in sync mode")
        
        # Update previous button states
        self.button0_prev = button0_pressed
        self.button1_prev = button1_pressed
    
    def run_teleoperation(self):
        """Execute one step of teleoperation"""
        if self.mode == "observation_only":
            # Teleoperation disabled in observation-only mode
            return
            
        # Get leader positions
        leader_pos_right, _ = self.leader_right.get_info()
        leader_pos_left, _ = self.leader_left.get_info()
        
        # Get follower positions
        follower_pos_right = self.follower_right.get_joint_pos()
        follower_pos_left = self.follower_left.get_joint_pos()

        # Limit follower and leader positions difference to 0.08 radians
        diff_right = leader_pos_right - follower_pos_right
        diff_left = leader_pos_left - follower_pos_left
        diff_threshold = 0.1 # 0.08
        diff_right = np.clip(diff_right, -diff_threshold, diff_threshold)
        diff_left = np.clip(diff_left, -diff_threshold, diff_threshold)
        follower_pos_right = leader_pos_right - diff_right
        follower_pos_left = leader_pos_left - diff_left
        
        # Command followers to match leaders
        self.follower_right.command_joint_pos(leader_pos_right)
        self.follower_left.command_joint_pos(leader_pos_left)
        
        # Set bilateral force (followers push back on leaders)
        self.leader_right.command_joint_pos(follower_pos_right[:6])
        self.leader_left.command_joint_pos(follower_pos_left[:6]) 

    def execute_inference_action(self, actions):
        """
        Extract joint positions from actions array and command follower arms (or print in observation-only mode).
        
        Args:
            actions: numpy array of shape [action_dim] where action_dim = 14 (7 per arm)
                     Format: [left.j0, left.j1, ..., left.j6, right.j0, right.j1, ..., right.j6]
        """
        # Extract left and right arm target positions (7 DOFs each)
        target_right = actions[:7]
        target_left = actions[7:14]
        
        if self.mode == "observation_only":
            # Print the actions instead of executing them
            print(f"  Action - Right arm: {target_right}")
            print(f"  Action - Left arm:  {target_left}")
            print(f"  Full action array: {actions}")
            return
        
        # Full control mode: actually command the robot
        # Get current follower positions
        follower_pos_right = self.follower_right.get_joint_pos()
        follower_pos_left = self.follower_left.get_joint_pos()
        
        # Limit target and current positions difference using action_diff_threshold
        diff_right = target_right - follower_pos_right
        diff_left = target_left - follower_pos_left
        diff_right = np.clip(diff_right, -self.action_diff_threshold, self.action_diff_threshold)
        diff_left = np.clip(diff_left, -self.action_diff_threshold, self.action_diff_threshold)
        target_right = follower_pos_right + diff_right
        target_left = follower_pos_left + diff_left
        
        # Command the follower arms
        self.follower_right.command_joint_pos(target_right)
        self.follower_left.command_joint_pos(target_left)
    
    def run_inference(self):
        """Execute one step of inference"""
        if not self.inference_mode:
            return

        # Simulation mode: replay episode with MuJoCo
        if self.sim:
            self.run_simulation_inference()
            return

        current_time = time.time()
        
        # Check if we need to run inference (buffer empty or executed required actions)
        actions_available = len(self.action_buffer)
        actions_limit = min(self.max_actions_per_inference, actions_available) if actions_available else 0
        need_inference = actions_available == 0 or self.action_index >= actions_limit
        
        if need_inference:
            print(f"\n{'='*60}")
            print(f"Running inference (action_index={self.action_index})...")
            
            # Get current robot observation
            raw_obs = self.inference_robot.get_observation()
            print(f"Received observation with keys: {list(raw_obs.keys())}")
            
            # Process images - map robot observation keys to policy input keys
            image_key_mapping = {
                'torso': 'observation/images/torso',
                'teleop_left': 'observation/images/teleop_left',
                'teleop_right': 'observation/images/teleop_right',
            }
            
            # Create example dict in the format expected by policy.infer()
            example = {}
            
            # Add images
            for robot_key, policy_key in image_key_mapping.items():
                if robot_key in raw_obs:
                    img = robot_obs_to_numpy(raw_obs[robot_key])
                    example[policy_key] = img
                    print(f"  Processed image: {policy_key}, shape: {img.shape}")
            
            # Crop head cam images if needed
            for k in list(example.keys()):
                if k in ["observation/images/head_cam_left", "observation/images/head_cam_right"]:
                    h = example[k].shape[0]
                    example[k] = example[k][h // 3 :, :, :]
            
            # Collect joint positions into a single state array
            joint_positions = []
            for j in range(7):
                key = f'right.j{j}.pos'
                if key in raw_obs:
                    joint_positions.append(float(raw_obs[key]))
            for j in range(7):
                key = f'left.j{j}.pos'
                if key in raw_obs:
                    joint_positions.append(float(raw_obs[key]))
            
            example['observation/state'] = np.array(joint_positions, dtype=np.float32)
            print(f"  Joint positions (state): {example['observation/state']}")
            
            # Add prompt (task description) - get from policy metadata if available, or use default
            robot_task_string = self.policy._metadata.get("robot_task_string", "Fold the shirt")
            example['prompt'] = robot_task_string
            
            # Run inference
            result = self.policy.infer(example)
            actions_batch = result["actions"]
            
            # Store actions in buffer (shape should be [25, action_dim])
            self.action_buffer = actions_batch
            self.action_index = 0
            
            print(f"Inference complete: got {actions_batch.shape[0]} actions")
            print(f"Action batch shape: {actions_batch.shape}")
            print(f"{'='*60}\n")
        
        # Execute or print actions at specified frequency (depending on mode)
        time_since_last_execution = current_time - self.last_execution_time
        execution_period = 1.0 / self.execution_hz
        
        if time_since_last_execution >= execution_period:
            # Get current action from buffer
            current_action = self.action_buffer[self.action_index]
            
            actions_limit = min(self.max_actions_per_inference, len(self.action_buffer))
            if actions_limit == 0 or self.action_index >= actions_limit:
                return
            
            # Get current action from buffer
            current_action = self.action_buffer[self.action_index]
            
            # Execute or print the action (depending on mode)
            self.execute_inference_action(current_action)
            
            # Update state
            self.action_index += 1
            fps = 1.0 / time_since_last_execution if time_since_last_execution > 0 else float("inf")
            print(f"Executed action {self.action_index}/{actions_limit} | fps: {fps:.2f}")
            self.last_execution_time = current_time

    def run_simulation_inference(self):
        """Run inference with MuJoCo simulation replay"""
        if self.sim_frame_idx >= (self.sim_to_idx - self.sim_from_idx):
            print("🛑 Simulation complete - episode finished")
            return
        
        # Get current frame from dataset
        dataset_idx = self.sim_from_idx + self.sim_frame_idx
        sample = self.sim_dataset[int(dataset_idx)]
        
        # Set Rerun time if using rerun
        if self.use_rerun:
            ts = float(sample.get("timestamp", self.sim_frame_idx / self.sim_fps))
            rr.set_time_seconds("timestamp", ts)
        
        # Get observation from dataset
        example = {}
        
        # Process images
        image_key_mapping = {
            'torso': 'observation/images/torso',
            'teleop_left': 'observation/images/teleop_left',
            'teleop_right': 'observation/images/teleop_right',
        }
        
        for robot_key, policy_key in image_key_mapping.items():
            # Try different possible keys
            for key in [robot_key, f"observation.images.{robot_key}", f"observation/images/{robot_key}"]:
                if key in sample:
                    img = sample[key]
                    if hasattr(img, 'numpy'):
                        img = img.numpy()
                    elif hasattr(img, 'cpu'):
                        img = img.cpu().numpy()
                    else:
                        img = np.array(img)
                    
                    # Convert to HWC uint8 format
                    if img.ndim == 3 and img.shape[0] == 3:  # C, H, W
                        img = np.transpose(img, (1, 2, 0))  # H, W, C
                    if img.dtype != np.uint8:
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                    
                    example[policy_key] = img
                    
                    # Log image to Rerun
                    if self.use_rerun:
                        rr.log(f"dataset/{policy_key}", rr.Image(img))
                    break
        
        # Get state
        state_key = None
        for key in ['observation/state', 'observation.state', 'qpos']:
            if key in sample:
                state_key = key
                break
        
        if state_key:
            qpos = sample[state_key]
            if hasattr(qpos, 'numpy'):
                qpos = qpos.numpy()
            elif hasattr(qpos, 'cpu'):
                qpos = qpos.cpu().numpy()
            else:
                qpos = np.array(qpos)
            example['observation/state'] = qpos.astype(np.float32)
        
        # Get prompt/task
        task_key = None
        for key in ['task', 'prompt', 'instruction']:
            if key in sample:
                task_key = key
                break
        
        if task_key:
            example['prompt'] = str(sample[task_key])
        else:
            example['prompt'] = "Fold the shirt"
        
        # Run inference if needed
        actions_limit = min(self.max_actions_per_inference, len(self.sim_action_buffer)) if len(self.sim_action_buffer) > 0 else 0
        if len(self.sim_action_buffer) == 0 or self.sim_action_index >= actions_limit:
            print(f"\n{'='*60}")
            print(f"Running inference (sim_frame_idx={self.sim_frame_idx})...")
            
            result = self.policy.infer(example)
            actions_batch = result["actions"]
            
            self.sim_action_buffer = actions_batch
            self.sim_action_index = 0
            print(f"Inference complete: got {actions_batch.shape[0]} actions")
            print(f"{'='*60}\n")
        
        # Get current action from buffer
        current_action = self.sim_action_buffer[self.sim_action_index]
        self.sim_action_index += 1
        
        # Get ground truth action for comparison
        u_gt = None
        if self.use_rerun or self.plot_actions:
            from lerobot.utils.constants import ACTION
            if ACTION in sample:
                u_gt_val = sample[ACTION]
                if hasattr(u_gt_val, 'numpy'):
                    u_gt = u_gt_val.numpy()
                elif hasattr(u_gt_val, 'cpu'):
                    u_gt = u_gt_val.cpu().numpy()
                else:
                    u_gt = np.array(u_gt_val)
        
        # Store actions for plotting (always when in sim mode)
        if self.plot_actions:
            u_pred = np.asarray(current_action, dtype=np.float32).reshape(-1)
            if u_gt is not None:
                u_gt_flat = np.asarray(u_gt, dtype=np.float32).reshape(-1)
                self.ground_truth_actions.append(u_gt_flat.copy())
            else:
                # If no GT available, use zeros as placeholder
                u_gt_flat = np.zeros_like(u_pred)
                self.ground_truth_actions.append(u_gt_flat.copy())
            self.predicted_actions.append(u_pred.copy())
            ts = float(sample.get("timestamp", self.sim_frame_idx / self.sim_fps))
            self.action_timestamps.append(ts)
        
        # Log actions to Rerun
        if self.use_rerun and u_gt is not None:
            u_pred = np.asarray(current_action, dtype=np.float32).reshape(-1)
            u_gt_flat = np.asarray(u_gt, dtype=np.float32).reshape(-1)
            n = min(len(u_pred), len(u_gt_flat))
            for i in range(n):
                rr.log(f"actions/gt/joint_{i}", rr.Scalars([float(u_gt_flat[i])]))
                rr.log(f"actions/pred/joint_{i}", rr.Scalars([float(u_pred[i])]))
        
        # Apply action to MuJoCo simulation
        # Reorder: [left(7), right(7)] -> [right(7), left(7)] for MuJoCo
        if len(current_action) == 14:
            u_apply = np.concatenate([current_action[7:], current_action[:7]])
        else:
            u_apply = current_action
        
        self.mj_data.ctrl[:] = u_apply
        
        # Calculate number of simulation steps until next frame
        if dataset_idx + 1 < self.sim_to_idx:
            sample_next = self.sim_dataset[int(dataset_idx + 1)]
            ts = float(sample.get("timestamp", 0))
            ts_next = float(sample_next.get("timestamp", ts + 1.0 / self.sim_fps))
        else:
            ts = float(sample.get("timestamp", 0))
            ts_next = ts + 1.0 / max(1.0, self.sim_fps)
        
        n_steps = max(1, int(round((ts_next - ts) / self.sim_dt)))
        
        # Step simulation
        for _ in range(n_steps):
            mujoco.mj_step(self.mj_model, self.mj_data)
        
        # Render and log to Rerun if enabled
        if self.use_rerun:
            # Try to create renderer for this frame if not already created
            if self.mj_renderer is None:
                try:
                    self.mj_renderer = mujoco.Renderer(self.mj_model, height=480, width=640)
                    self.use_renderer = True
                except Exception:
                    self.use_renderer = False
            
            if self.use_renderer and self.mj_renderer is not None:
                try:
                    self.mj_renderer.update_scene(self.mj_data, camera=0)
                    sim_image = self.mj_renderer.render()
                    rr.log("sim", rr.Image(sim_image))
                except Exception:
                    # Silently fail if rendering has issues
                    pass
        
        # Print action
        print(f"Sim step {self.sim_frame_idx}: Action applied, {n_steps} sim steps")
        
        # Advance frame
        self.sim_frame_idx += 1
    
    def plot_action_comparison(self):
        """Plot predicted vs ground truth actions and save to plots directory"""
        if not self.plot_actions or len(self.predicted_actions) == 0:
            return
        
        predicted = np.array(self.predicted_actions)
        ground_truth = np.array(self.ground_truth_actions)
        timestamps = np.array(self.action_timestamps)
        
        # Normalize timestamps to start from 0
        if len(timestamps) > 0:
            timestamps = timestamps - timestamps[0]
        
        n_joints = min(predicted.shape[1], ground_truth.shape[1])
        
        # Joint names for bimanual robot (7 joints per arm)
        joint_names = [
            'j0 (shoulder_pan)', 'j1 (shoulder_lift)', 'j2 (elbow)', 
            'j3 (wrist_1)', 'j4 (wrist_2)', 'j5 (wrist_3)', 'j6 (gripper)'
        ]
        
        # Split into left and right arms (assuming 14 joints total: 7 left + 7 right)
        n_joints_per_arm = 7
        left_arm_joints = min(n_joints_per_arm, n_joints // 2)
        right_arm_joints = min(n_joints_per_arm, n_joints - left_arm_joints)
        
        # Create figure with 2 columns: left arm and right arm
        fig = plt.figure(figsize=(16, 2 * max(left_arm_joints, right_arm_joints)))
        fig.suptitle(f'Predicted vs Ground Truth Actions - Episode {self.episode_idx}', fontsize=16, y=0.995)
        
        # Left arm plots (left column)
        for i in range(left_arm_joints):
            ax = plt.subplot(max(left_arm_joints, right_arm_joints), 2, 2 * i + 1)
            ax.plot(timestamps, predicted[:, i], label='Predicted', alpha=0.7, linewidth=1.5, color='blue')
            ax.plot(timestamps, ground_truth[:, i], label='Ground Truth', alpha=0.7, linewidth=1.5, linestyle='--', color='orange')
            ax.set_ylabel(f'Left {joint_names[i]}\n(rad)', fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            if i < left_arm_joints - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Time (s)', fontsize=10)
        
        # Right arm plots (right column)
        for i in range(right_arm_joints):
            ax = plt.subplot(max(left_arm_joints, right_arm_joints), 2, 2 * i + 2)
            right_idx = left_arm_joints + i
            ax.plot(timestamps, predicted[:, right_idx], label='Predicted', alpha=0.7, linewidth=1.5, color='blue')
            ax.plot(timestamps, ground_truth[:, right_idx], label='Ground Truth', alpha=0.7, linewidth=1.5, linestyle='--', color='orange')
            ax.set_ylabel(f'Right {joint_names[i]}\n(rad)', fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            if i < right_arm_joints - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Time (s)', fontsize=10)
        
        # Add column titles
        fig.text(0.25, 0.98, 'Right Arm', ha='center', fontsize=12, weight='bold')
        fig.text(0.75, 0.98, 'Left Arm', ha='center', fontsize=12, weight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save plot to plots directory
        plot_filename = self.plots_dir / f'action_comparison_episode_{self.episode_idx}.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"✓ Action comparison plot saved to: {plot_filename}")
        
        # Close figure to free memory
        plt.close(fig)
    
    def run(self):
        """Run the main control loop"""
        try:
            if self.sim:
                # Simulation mode: automatically enable inference and run until episode ends
                print("🤖 Starting simulation replay...")
                self.inference_mode = True
                while self.sim_frame_idx < (self.sim_to_idx - self.sim_from_idx):
                    self.run_inference()
                    time.sleep(0.01)  # Small delay for visualization
                print("✓ Simulation replay complete")
                
                # Automatically plot actions when in simulation mode
                if self.plot_actions:
                    print("Generating action comparison plot...")
                    self.plot_action_comparison()
                
                return
            elif self.mode == "observation_only":
                # Automatically enable inference mode (no button required)
                if not self.inference_mode:
                    print("🤖 Auto-enabling INFERENCE mode (observation-only mode)")
                    self.inference_mode = True
                
                while True:
                    # Always run inference mode (observation gathering and action printing)
                    if self.inference_mode:
                        self.run_inference()
                    else:
                        # If inference mode is disabled, just wait
                        time.sleep(0.1)
                    time.sleep(0.01)
            else:
                # Full control mode: use button controls
                while True:
                    # Get button state from right leader (buttons are shared)
                    _, current_button = self.leader_right.get_info()
                    
                    # Handle button events
                    self.handle_button_events(current_button)

                    # Execute appropriate control mode
                    if self.inference_mode:
                        self.run_inference()
                    elif self.synchronized:
                        self.run_teleoperation()
                    # else: do nothing (unsync mode)

                    time.sleep(0.01)
        except KeyboardInterrupt:
            print("\n\nStopping control loop...")
            



def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Bimanual robot control with inference mode support')
    parser.add_argument('--mode', type=str, default=DEFAULT_MODE,
                        choices=['observation_only', 'full_control'],
                        help='Mode to run: "observation_only" (no actuation, just print actions) or "full_control" (full robot control with buttons)')
    parser.add_argument('--sim', action='store_true', 
                        help='Enable MuJoCo simulation replay mode')
    parser.add_argument('--mjcf_path', type=str, default=None,
                        help='Path to MuJoCo XML file (required if --sim)')
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='Path to dataset directory or repo_id (required if --sim)')
    parser.add_argument('--episode_idx', type=int, default=0,
                        help='Episode index to replay (default: 0)')
    parser.add_argument('--use_rerun', action='store_true',
                        help='Use Rerun for visualization (requires rerun package)')
    parser.add_argument('--max-actions-per-inference', type=int, default=None,
                        help='Limit how many actions from each inference chunk are executed before next inference (default: None = use action_horizon from policy config). Use smaller values (e.g., 10-20) for more frequent inference if robot gets stuck.')
    parser.add_argument('--execution-hz', type=float, default=30.0,
                        help='Follower command frequency during inference (should match training frequency, default: 30 Hz). Using different frequency can cause robot to get stuck.')
    parser.add_argument('--action-diff-threshold', type=float, default=0.2,
                        help='Maximum change per action step in radians (limits how fast robot can move, default: 0.2). Increase if robot moves too slowly, decrease if unstable.')
    args = parser.parse_args()
    
    mode = args.mode
    sim = args.sim
    processes = []
    controller = None
    
    try:
        # Skip follower processes if in simulation mode
        if not sim:
            # Always start follower processes - needed for observations in both modes
            print("Launching follower processes...")
            follower_configs = [
                {'can_channel': 'can_follower_r', 'gripper': 'linear_4310', 'server_port': 1234},
                {'can_channel': 'can_follower_l', 'gripper': 'linear_4310', 'server_port': 1235}
            ]
            
            for config in follower_configs:
                process = launch_gello_process(**config)
                if process:
                    processes.append(process)
                    print(f"✓ Started follower process {process.pid} for {config['can_channel']}")
                else:
                    raise RuntimeError(f"Failed to start follower process for {config['can_channel']}")
            
            print(f"✓ Successfully launched {len(processes)} follower processes")
            print("Waiting for processes to initialize...")
            time.sleep(3)
        
        if sim:
            print("="*60)
            print("SIMULATION MODE - MuJoCo REPLAY")
            print("="*60)
            print("This script will:")
            print("  1. Load episode from dataset")
            print("  2. Run inference on each frame")
            print("  3. Apply actions to MuJoCo simulation")
            print("="*60)
        elif mode == "observation_only":
            print("="*60)
            print("OBSERVATION-ONLY MODE - NO ROBOT ACTUATION")
            print("="*60)
            print("This script will:")
            print("  1. Connect to robot to receive observations")
            print("  2. Pass observations to the model")
            print("  3. Print the output actions")
            print("="*60)
        else:
            print("="*60)
            print("FULL CONTROL MODE - ROBOT ACTUATION ENABLED")
            print("="*60)
            print("Checking CAN interfaces...")
            check_all_can_interfaces()
        
        # Setup controller with inference using the new API
        config_name = "pi0_i2rt_lora"
        checkpoint_dir = "/home/i2rt/openpi/checkpoints/pi05_i2rt_lora/pi_lora_train/20000"
        
        controller = BimanualTeleopWithInference(
            config_name=config_name,
            checkpoint_dir=checkpoint_dir,
            bilateral_kp=0.0,  # Set to > 0 for bilateral force feedback
            mode=mode,
            sim=sim,
            mjcf_path=args.mjcf_path,
            dataset_dir=args.dataset_dir,
            episode_idx=args.episode_idx,
            use_rerun=args.use_rerun,
            max_actions_per_inference=args.max_actions_per_inference,
            execution_hz=args.execution_hz,
            action_diff_threshold=args.action_diff_threshold,
        )

        # Run main control loop
        controller.run()


    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        # Clean up: terminate all follower processes (if any were started)
        if processes:
            print("\nCleaning up processes...")
            for process in processes:
                try:
                    print(f"Terminating process {process.pid}...")
                    process.terminate()
                    
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        print(f"Force killing process {process.pid}...")
                        process.kill()
                        process.wait()
                        
                except Exception as e:
                    print(f"Error terminating process {process.pid}: {e}")
            
            print("All processes terminated")

if __name__ == "__main__":
    main()
