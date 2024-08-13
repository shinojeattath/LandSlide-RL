import numpy as np
from landslide_env import LandslideEnv
from dqn_agent import DQNAgent

# Parameters
num_episodes = 10
batch_size = 32

# Initialize environment and agent
env = LandslideEnv(data_path="landslide_data.csv")
state_size = env.state_size
action_size = env.action_size
agent = DQNAgent(state_size, action_size)

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    
    print(f"Episode {episode+1}/{num_episodes}, Epsilon: {agent.epsilon}")

# Save the trained model
agent.model.save("dqn_landslide_model.h5")
