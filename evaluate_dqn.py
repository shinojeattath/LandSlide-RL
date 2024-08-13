import numpy as np
from landslide_env import LandslideEnv
from dqn_agent import DQNAgent
import tensorflow as tf
#from tensorflow.keras.models import load_model

def load_trained_model(model_path, state_size, action_size):
    agent = DQNAgent(state_size, action_size)
    agent.model = tf.keras.models.load_model(model_path)
    return agent


def predict_landslide(agent, state):
    state = np.reshape(state, [1, state_size])
    action = agent.act(state)
    return action

if __name__ == "__main__":
    model_path = "dqn_landslide_model.h5"
    env = LandslideEnv(data_path="landslide_data.csv")
    state_size = env.state_size
    action_size = env.action_size
    agent = load_trained_model(model_path, state_size, action_size)

    # Test prediction
    new_state = np.random.rand(state_size)  # Replace with actual state from dataset
    action = predict_landslide(agent, new_state)

    if action == 1:
        print("Landslide predicted.")
    else:
        print("No landslide predicted.")
