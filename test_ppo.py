import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from ppo import PPO

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.envs.single_agent_rl.FlyThruGateAviary import FlyThruGateAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

#################################### Testing ###################################
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 3
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_RECORD_VISION = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48   
DEFAULT_DURATION_SEC =99999
DEFAULT_OBSTACLES = True
DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
def test(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):
    print("============================================================================================")

    ################## hyperparameters ##################

    # env_name = "CartPole-v1"
    # has_continuous_action_space = False
    # max_ep_len = 400
    # action_std = None

    # env_name = "LunarLander-v2"
    # has_continuous_action_space = False
    # max_ep_len = 300
    # action_std = None

    # env_name = "BipedalWalker-v2"
    # has_continuous_action_space = True
    # max_ep_len = 1500           # max timesteps in one episode
    # action_std = 0.1            # set same std for action distribution which was used while saving

    max_ep_len = 1000           # max timesteps in one episode
    action_std = 0.1            # set same std for action distribution which was used while saving

    render = True              # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames

    total_test_episodes = 10    # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    #####################################################

    filename = os.path.join(output_folder, 'recording_'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        print(filename)
        os.makedirs(filename+'/')

    env = HoverAviary(gui=gui,
                           obs=DEFAULT_OBS,
                           act=DEFAULT_ACT, 
                           obstacles=obstacles,
                           record=DEFAULT_RECORD_VIDEO)
    # env = FlyThruGateAviary(gui=DEFAULT_GUI,
    #                        obs=DEFAULT_OBS,
    #                        act=DEFAULT_ACT,
    #                        record=DEFAULT_RECORD_VIDEO)
    # state space dimension
    print(obstacles)
    state_dim = 12

    # action space dimension

    action_dim = 4

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)

    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num


    #checkpoint_path = "log_dir/5/3577_ppo_drone.pth"
    #checkpoint_path = "log_dir/6/4436_ppo_drone.pth"
    checkpoint_path = "log_dir/7/16351_ppo_drone.pth"
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    # for ep in range(1, total_test_episodes+1):
    #     ep_reward = 0
    #     state = env.reset()

    obs, info = env.reset(seed=42, options={})
    ep_reward = 0
    start_time = datetime.now().replace(microsecond=0)
    start = time.time()
    for i in range(0, int(duration_sec**env.CTRL_FREQ)):
        action = ppo_agent.select_action(obs)
        action = np.expand_dims(action, axis=0)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        env.render()
        sync(i, start, env.CTRL_TIMESTEP)
        if terminated:
            break

    # clear buffer
    ppo_agent.buffer.clear()

    test_running_reward +=  ep_reward
    print('Episode: {} \t\t Reward: {}'.format(0, round(ep_reward, 2)))
    ep_reward = 0

    env.close()

# print("============================================================================================")
#
# avg_test_reward = test_running_reward / total_test_episodes
# avg_test_reward = round(avg_test_reward, 2)
# print("average test reward : " + str(avg_test_reward))
#
# print("============================================================================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()
    test(**vars(ARGS))
