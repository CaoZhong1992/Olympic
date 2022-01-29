from Test_Scenarios.TestScenario_CarEnv_05_Round import CarEnv_05_Round
from Test_Scenarios.TestScenario_Town03_two_lane import CarEnv_03_Two_Lane
# from Agent.drl_library.dqn.dqn import DQN
from Agent.drl_library.dqn.dqn_test import DQNTest

EPISODES=2642

if __name__ == '__main__':

    # Create environment
    
    env = CarEnv_03_Two_Lane()

    model = DQNTest(env, batch_size=20)
    model.test(load_step=0, num_frames=300000,  gamma=0.99)