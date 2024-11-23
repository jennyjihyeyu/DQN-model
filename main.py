
import argparse
import numpy as np
import torch.optim as optim
from dqn_model import DQN
from dqn_learn import OptimizerSpec, dqn_learing
from utils.schedule import LinearSchedule
from lib.rl_env import RLNetworkEnv

BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1000000
LEARNING_STARTS = 50000
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 4
TARGER_UPDATE_FREQ = 10000
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01


def getargs():
    """Parse command line arguments."""


    args = argparse.ArgumentParser()
    args.add_argument('flow_path', help="Path to the input npy files describing flow profiles.")
    args.add_argument('route_path', help="Path to the input npy files describing flow routes.")
    args.add_argument('--simulation-time', type=float, default=100.0,
                      help="Total simulation time (in seconds).")
    args.add_argument('--scheduling-policy', type=str, default="fifo",
                      help="Type of scheduler applied at each hop of the network. Choose between 'fifo' and 'sced'.")
    args.add_argument('--shaping-mode', type=str, default="per_flow",
                      help="Type of traffic shapers applied. Choose among 'per_flow', 'interleaved', 'ingress',"
                           "and 'none'. Only active when scheduling policy is 'fifo'")
    args.add_argument('--buffer-bound', type=str, default='infinite',
                      help="Link buffer bound. Choose between 'infinite', and 'with_shaping'.")
    args.add_argument('--arrival-pattern-type', type=str, default="sync_burst",
                      help="Type of traffic arrival pattern. Choose among 'sync_burst', 'sync_smooth', and 'async'.")
    args.add_argument('--awake-dur', type=float, default=None, help="Flow awake time.")
    args.add_argument('--awake-dist', type=str, default="constant",
                      help="Flow awake time distribution. Choose between 'exponential' and 'constant'.")
    args.add_argument('--sync-jitter', type=float, default=0,
                      help="Jitter for synchronized flow burst. Only active when the arrival pattern is 'sync_burst'.")
    args.add_argument('--pause-interval', type=float, default=1,
                      help="The length of a time step (in second) for the reinforcement learning environment.")
    args.add_argument('--high-reward', type=float, default=1,
                      help="The highest possible reward received by a flow when its end-to-end delay is 0.")
    args.add_argument('--low-reward', type=float, default=0.1,
                      help="The reward received by a flow when its end-to-end delay is equal to the worst case bound.")
    args.add_argument('--penalty', type=float, default=-10,
                      help="The negative penalty received by a flow when its end-to-end delay exceeds the bound.")
    return args.parse_args()


def main(env, num_timesteps):

    def stopping_criterion(env):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return env.time / env.pause_interval >= num_timesteps

    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
    )

    exploration_schedule = LinearSchedule(1000000, 0.1)

    dqn_learing(
        env=env,
        q_func=DQN,
        optimizer_spec=optimizer_spec,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        frame_history_len=FRAME_HISTORY_LEN,
        target_update_freq=TARGER_UPDATE_FREQ,
    )
    

if __name__ == '__main__':
    args = getargs()
    flow_profile = np.load(args.flow_path)
    flow_route = np.load(args.route_path)
    shaping_delay = np.minimum(flow_profile[:, 2], flow_profile[:, 1] / flow_profile[:, 0])
    
    

    environment = RLNetworkEnv(flow_profile, flow_route, shaping_delay, simulation_time=args.simulation_time,
                               scheduling_policy=args.scheduling_policy, shaping_mode=args.shaping_mode,
                               buffer_bound=args.buffer_bound, arrival_pattern_type=args.arrival_pattern_type,
                               awake_dur=args.awake_dur, awake_dist=args.awake_dist, sync_jitter=args.sync_jitter,
                               arrival_pattern=None, keep_per_hop_departure=False, scaling_factor=1.0,
                               packet_size=1, pause_interval=args.pause_interval, high_reward=args.high_reward,
                               low_reward=args.low_reward, penalty=args.penalty)
    # Initialize the environment and get the initial state.

    # Change the index to select a different game.
  

    # Run training
    main(environment, num_timesteps=1e6)
