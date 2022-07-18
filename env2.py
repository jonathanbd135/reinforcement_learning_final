from train_agent import Agent

if __name__ == '__main__':
    ddqn = Agent(model_type='ddqn', betches_size=100, sync_models=100, per=True, env_number=2, loss='mse')
    ddqn.train_agent(episodes=1000, epsilon=0.01, discount_factor=0.9)
    print('Finished training!')
    # ddqn.q_model.save('/content/gdrive/MyDrive/model1')