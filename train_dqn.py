from Test_Scenarios.TestScenario_CarEnv_05_Round import CarEnv_05_Round
from Agent.drl_library.dqn.dqn import DQN

EPISODES=2642

if __name__ == '__main__':

    # Create environment
    
    env = CarEnv_05_Round()

    model = DQN(env, batch_size=20)
    model.train(load_step=0, num_frames=300000,  gamma=0.99)
    # model.save("dqn_cartpole")