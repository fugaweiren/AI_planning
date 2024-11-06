import numpy as np
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
# from make_env import make_env
from gym.envs.registration import register
import gym
import tqdm
from plot import live_plot

import time
import threading
scenario = "multigrid-collect-v0"

register(
    id=scenario,
    entry_point='gym_multigrid.envs:CollectGame4HEnv10x10N2',
    # entry_point='gym_multigrid.envs:CollectGame1HEnv10x10',
)
env = gym.make('multigrid-collect-v0')

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

if __name__ == '__main__':
    env = env
    n_agents = env.agents.__len__()
    actor_dims = []
    for i in range(n_agents):
        x, y, properties  = env.observation_space.shape
        actor_dims.append(x*y*properties) # Check the observation space
    critic_dims = sum(actor_dims)

    # action space is a list of arrays, assume each agent has same action space
    n_actions = env.action_space.n
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                           fc1=64, fc2=64,  
                           alpha=0.01, beta=0.01, scenario=scenario,
                           chkpt_dir=r'D:/Projects/Courses/CS5446 AI Planning & Decision Making/Project/gym-multigrid-master/tmp/maddpg/')

    memory = MultiAgentReplayBuffer(10000, critic_dims, actor_dims, #1000000
                        n_actions, n_agents, batch_size=32) #1024

    PRINT_INTERVAL = 500
    N_GAMES = 50000
    MAX_STEPS = 60
    total_steps = 0
    score_history = []
    evaluate = False
    best_score = 0
    
    # Track the losses , max_reward & the critic loss etc
    # Integrate the KG Loss etc

    # Set up progress tracking
    progress_bar = tqdm.tqdm(range(1, N_GAMES + 1), postfix={'Total Rewards': 0})
    data_to_plot = {}
    for agent_idx in range(n_agents):
        data_to_plot[f'Agent {agent_idx} Actor Loss'] = []
        data_to_plot[f'Agent {agent_idx} Critic Loss'] = []
        data_to_plot[f'Agent {agent_idx} Reward'] = []
    data_to_plot["Total Reward"] =[]
    plot_update_counter = [0]


    # Start live_plot in a separate thread
    plot_thread = threading.Thread(target=live_plot, args=(data_to_plot,plot_update_counter))
    plot_thread.daemon = True  # Ensures the thread will exit when the main program ends
    plot_thread.start()


    # entropy_objective_history = []

    if evaluate:
        maddpg_agents.load_checkpoint()

    for i, iteration in enumerate(progress_bar):
        obs, _ = env.reset()
        score = 0
        done = False#[False]*n_agents
        episode_step = 0
        while not done: #any(done):
            # if evaluate:
            # env.render(mode='human', highlight=True)
            time.sleep(0.1) # to slow down the action for the video
            actions = maddpg_agents.choose_action(obs)
            obs_, reward, done, info = env.step(actions)

            state = obs   #obs_list_to_state_vector(obs)A
            state_ = obs_ #obs_list_to_state_vector(obs_)

            if episode_step >= MAX_STEPS:
                done = True #[True]*n_agents

            memory.store_transition(obs, state, actions, reward, obs_, state_, done)

            if total_steps % 100 == 0 and not evaluate:
                info = maddpg_agents.learn(memory)
                if info is not None:
                    for agent_idx in range(n_agents):
                        data_to_plot[f'Agent {agent_idx} Actor Loss'].append(info[agent_idx]["agent_loss"])
                        data_to_plot[f'Agent {agent_idx} Critic Loss'].append(info[agent_idx]["critic_loss"])
                        data_to_plot[f'Agent {agent_idx} Reward'].append(reward[agent_idx])
                
            
            obs = obs_
            score += sum(reward)
            total_steps += 1
            episode_step += 1
            plot_update_counter[0]+=1
        
        progress_bar.set_postfix({'Total Rewards': score})
        score_history.append(score)


        avg_score = np.mean(score_history[-100:])
        if not evaluate:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score))

        
        data_to_plot["Total Reward"] = score_history
        # if total_steps % 100 == 0 and not evaluate:
        #     live_plot(data_to_plot, plot_update_counter)
