from train_agent import Agent
from keras.callbacks import CSVLogger
import numpy as np
import pandas as pd

if __name__ == '__main__':
    episodes_to_learn = 1


    ddqn_1=Agent(model_type='ddqn', betches_size=100, sync_models=50, per=True, env_number=1, is_stochastic=True)
    ddqn_1.set_loger('huber_log_ddqn_1.csv')
    ddqn_1.train_agent(episodes=1, epsilon=0.6, discount_factor=0.9)
    while (np.average(ddqn_1.steps_taken[-5:]) < 50 and len(ddqn_1.steps_taken) < episodes_to_learn):
        ddqn_1.train_agent(episodes=5, epsilon=ddqn_1.epsilon, discount_factor=ddqn_1.discount_factor)
    ddqn_1.q_model.save(r'C:\Users\HP\Desktop\Finale\reinforcement_learning_final\weights\ex1_w\ddqn_1')
    print(f'Total episodes: {len(ddqn_1.steps_taken)}')
    loss_ddqn_1 = np.flip(pd.read_csv('huber_log_ddqn_1.csv', delimiter=';').loss.to_numpy())
    quantile_95 = np.quantile(loss_ddqn_1, 0.95)
    loss_ddqn_1 = np.arange(len(loss_ddqn_1[np.where(loss_ddqn_1 < quantile_95)])), loss_ddqn_1[np.where(loss_ddqn_1 < quantile_95)]
    np.savetxt("loss_ddqn_1.csv", loss_ddqn_1, delimiter=",")
    ddqn_1_actions_taken = np.array(ddqn_1.steps_taken)
    np.savetxt("ddqn_1_actions_taken.csv", ddqn_1_actions_taken, delimiter=",")
    ddqn_1_reward_eval = np.array(ddqn_1.reward_eval)
    np.savetxt("ddqn_1_reward_eval.csv", ddqn_1_reward_eval, delimiter=",")

    dqn_1 = Agent(model_type='dqn', betches_size=100, sync_models=50, per=True, env_number=1, is_stochastic=True)
    dqn_1.set_loger('huber_log_dqn_1.csv')
    dqn_1.train_agent(episodes=1, epsilon=0.6, discount_factor=0.9)
    ddqn_1.q_model.save(r'C:\Users\HP\Desktop\Finale\reinforcement_learning_final\weights\ex1_w\dqn_1')
    while (np.average(dqn_1.steps_taken[-5:]) < 50 and len(dqn_1.steps_taken) < episodes_to_learn):
        dqn_1.train_agent(episodes=5, epsilon=dqn_1.epsilon, discount_factor=dqn_1.discount_factor)
    dqn_1.q_model.save(r'C:\Users\HP\Desktop\Finale\reinforcement_learning_final\weights\ex1_w\dqn_1')
    print(f'Total episodes: {len(dqn_1.steps_taken)}')
    loss_dqn_1 = np.flip(pd.read_csv('huber_log_dqn_1.csv', delimiter=';').loss.to_numpy())
    quantile_95 = np.quantile(loss_dqn_1, 0.95)
    loss_dqn_1 = np.arange(len(loss_dqn_1[np.where(loss_dqn_1 < quantile_95)])), loss_dqn_1[
        np.where(loss_dqn_1 < quantile_95)]
    np.savetxt("loss_dqn_1.csv", loss_dqn_1, delimiter=",")
    dqn_1_actions_taken = np.array(dqn_1.steps_taken)
    np.savetxt("dqn_1_actions_taken.csv", dqn_1_actions_taken, delimiter=",")
    dqn_1_reward_eval = np.array(dqn_1.reward_eval)
    np.savetxt("dqn_1_reward_eval.csv", dqn_1_reward_eval, delimiter=",")

    ddqn_1=Agent(model_type='ddqn', betches_size=100, sync_models=50, per=False, env_number=1, is_stochastic=True)
    ddqn_1.set_loger('huber_log_ddqn_1_no_per.csv')
    ddqn_1.train_agent(episodes=1, epsilon=0.6, discount_factor=0.9)
    dqn_1.q_model.save(r'C:\Users\HP\Desktop\Finale\reinforcement_learning_final\weights\ex1_w\ddqn_1_no_per')
    while (np.average(ddqn_1.steps_taken[-5:]) < 50 and len(ddqn_1.steps_taken) < episodes_to_learn):
        ddqn_1.train_agent(episodes=5, epsilon=ddqn_1.epsilon, discount_factor=ddqn_1.discount_factor)
    print(f'Total episodes: {len(ddqn_1.steps_taken)}')
    loss_ddqn_1 = np.flip(pd.read_csv('huber_log_ddqn_1_no_per.csv', delimiter=';').loss.to_numpy())
    quantile_95 = np.quantile(loss_ddqn_1, 0.95)
    loss_ddqn_1 = np.arange(len(loss_ddqn_1[np.where(loss_ddqn_1 < quantile_95)])), loss_ddqn_1[np.where(loss_ddqn_1 < quantile_95)]
    np.savetxt("loss_ddqn_1_no_per.csv", loss_ddqn_1, delimiter=",")
    ddqn_1_actions_taken = np.array(ddqn_1.steps_taken)
    np.savetxt("ddqn_1_actions_taken_no_per.csv", ddqn_1_actions_taken, delimiter=",")
    ddqn_1_reward_eval = np.array(ddqn_1.reward_eval)
    np.savetxt("ddqn_1_reward_eval_no_per.csv", ddqn_1_reward_eval, delimiter=",")

    dqn_1 = Agent(model_type='dqn', betches_size=100, sync_models=50, per=False, env_number=1, is_stochastic=True)
    dqn_1.set_loger('huber_log_dqn_1_no_per.csv')
    dqn_1.train_agent(episodes=1, epsilon=0.6, discount_factor=0.9)
    dqn_1.q_model.save(r'C:\Users\HP\Desktop\Finale\reinforcement_learning_final\weights\ex1_w\dqn_1_no_per')
    while (np.average(dqn_1.steps_taken[-5:]) < 50 and len(dqn_1.steps_taken) < episodes_to_learn):
        dqn_1.train_agent(episodes=5, epsilon=dqn_1.epsilon, discount_factor=dqn_1.discount_factor)
    print(f'Total episodes: {len(dqn_1.steps_taken)}')
    loss_dqn_1 = np.flip(pd.read_csv('huber_log_dqn_1_no_per.csv', delimiter=';').loss.to_numpy())
    quantile_95 = np.quantile(loss_dqn_1, 0.95)
    loss_dqn_1 = np.arange(len(loss_dqn_1[np.where(loss_dqn_1 < quantile_95)])), loss_dqn_1[
        np.where(loss_dqn_1 < quantile_95)]
    np.savetxt("loss_dqn_1_no_per.csv", loss_dqn_1, delimiter=",")
    dqn_1_actions_taken = np.array(dqn_1.steps_taken)
    np.savetxt("dqn_1_actions_taken_no_per.csv", dqn_1_actions_taken, delimiter=",")
    dqn_1_reward_eval = np.array(dqn_1.reward_eval)
    np.savetxt("dqn_1_reward_eval_no_per.csv", dqn_1_reward_eval, delimiter=",")


    ddqn_2=Agent(model_type='ddqn', betches_size=100, sync_models=50, per=True, env_number=2, is_stochastic=False)
    ddqn_2.set_loger('huber_log_ddqn_2.csv')
    ddqn_2.train_agent(episodes=1, epsilon=0.6, discount_factor=0.9)
    while (np.average(ddqn_2.steps_taken[-5:]) < 50 and len(ddqn_2.steps_taken) < episodes_to_learn):
        ddqn_2.train_agent(episodes=5, epsilon=ddqn_2.epsilon, discount_factor=ddqn_2.discount_factor)
    ddqn_2.q_model.save(r'C:\Users\HP\Desktop\Finale\reinforcement_learning_final\weights\ex2_w\ddqn_2')
    print(f'Total episodes: {len(ddqn_2.steps_taken)}')
    loss_ddqn_2 = np.flip(pd.read_csv('huber_log_ddqn_2.csv', delimiter=';').loss.to_numpy())
    quantile_95 = np.quantile(loss_ddqn_2, 0.95)
    loss_ddqn_2 = np.arange(len(loss_ddqn_2[np.where(loss_ddqn_2 < quantile_95)])), loss_ddqn_2[np.where(loss_ddqn_2 < quantile_95)]
    np.savetxt("loss_ddqn_2.csv", loss_ddqn_2, delimiter=",")
    ddqn_2_actions_taken = np.array(ddqn_2.steps_taken)
    np.savetxt("ddqn_2_actions_taken.csv", ddqn_2_actions_taken, delimiter=",")
    ddqn_2_reward_eval = np.array(ddqn_2.reward_eval)
    np.savetxt("ddqn_2_reward_eval.csv", ddqn_2_reward_eval, delimiter=",")

    dqn_2 = Agent(model_type='dqn', betches_size=100, sync_models=50, per=True, env_number=2, is_stochastic=False)
    dqn_2.set_loger('huber_log_dqn_2.csv')
    dqn_2.train_agent(episodes=1, epsilon=0.6, discount_factor=0.9)
    ddqn_2.q_model.save(r'C:\Users\HP\Desktop\Finale\reinforcement_learning_final\weights\ex2_w\dqn_2')
    while (np.average(dqn_2.steps_taken[-5:]) < 50 and len(dqn_2.steps_taken) < episodes_to_learn):
        dqn_2.train_agent(episodes=5, epsilon=dqn_2.epsilon, discount_factor=dqn_2.discount_factor)
    print(f'Total episodes: {len(dqn_2.steps_taken)}')
    loss_dqn_2 = np.flip(pd.read_csv('huber_log_dqn_2.csv', delimiter=';').loss.to_numpy())
    quantile_95 = np.quantile(loss_dqn_2, 0.95)
    loss_dqn_2 = np.arange(len(loss_dqn_2[np.where(loss_dqn_2 < quantile_95)])), loss_dqn_2[
        np.where(loss_dqn_2 < quantile_95)]
    np.savetxt("loss_dqn_2.csv", loss_dqn_2, delimiter=",")
    dqn_2_actions_taken = np.array(dqn_2.steps_taken)
    np.savetxt("dqn_2_actions_taken.csv", dqn_2_actions_taken, delimiter=",")
    dqn_2_reward_eval = np.array(dqn_2.reward_eval)
    np.savetxt("dqn_2_reward_eval.csv", dqn_2_reward_eval, delimiter=",")

    ddqn_2=Agent(model_type='ddqn', betches_size=100, sync_models=50, per=False, env_number=2, is_stochastic=False)
    ddqn_2.set_loger('huber_log_ddqn_2_no_per.csv')
    ddqn_2.train_agent(episodes=1, epsilon=0.6, discount_factor=0.9)
    dqn_2.q_model.save(r'C:\Users\HP\Desktop\Finale\reinforcement_learning_final\weights\ex2_w\ddqn_2_no_per')
    while (np.average(ddqn_2.steps_taken[-5:]) < 50 and len(ddqn_2.steps_taken) < episodes_to_learn):
        ddqn_2.train_agent(episodes=5, epsilon=ddqn_2.epsilon, discount_factor=ddqn_2.discount_factor)
    print(f'Total episodes: {len(ddqn_2.steps_taken)}')
    loss_ddqn_2 = np.flip(pd.read_csv('huber_log_ddqn_2_no_per.csv', delimiter=';').loss.to_numpy())
    quantile_95 = np.quantile(loss_ddqn_2, 0.95)
    loss_ddqn_2 = np.arange(len(loss_ddqn_2[np.where(loss_ddqn_2 < quantile_95)])), loss_ddqn_2[np.where(loss_ddqn_2 < quantile_95)]
    np.savetxt("loss_ddqn_2_no_per.csv", loss_ddqn_2, delimiter=",")
    ddqn_2_actions_taken = np.array(ddqn_2.steps_taken)
    np.savetxt("ddqn_2_actions_taken_no_per.csv", ddqn_2_actions_taken, delimiter=",")
    ddqn_2_reward_eval = np.array(ddqn_2.reward_eval)
    np.savetxt("ddqn_2_reward_eval_no_per.csv", ddqn_2_reward_eval, delimiter=",")

    dqn_2 = Agent(model_type='dqn', betches_size=100, sync_models=50, per=False, env_number=2, is_stochastic=False)
    dqn_2.set_loger('huber_log_dqn_2_no_per.csv')
    dqn_2.train_agent(episodes=1, epsilon=0.6, discount_factor=0.9)
    dqn_2.q_model.save(r'C:\Users\HP\Desktop\Finale\reinforcement_learning_final\weights\ex2_w\dqn_2_no_per')
    while (np.average(dqn_2.steps_taken[-5:]) < 50 and len(dqn_2.steps_taken) < episodes_to_learn):
        dqn_2.train_agent(episodes=5, epsilon=dqn_2.epsilon, discount_factor=dqn_2.discount_factor)
    print(f'Total episodes: {len(dqn_2.steps_taken)}')
    loss_dqn_2 = np.flip(pd.read_csv('huber_log_dqn_2_no_per.csv', delimiter=';').loss.to_numpy())
    quantile_95 = np.quantile(loss_dqn_2, 0.95)
    loss_dqn_2 = np.arange(len(loss_dqn_2[np.where(loss_dqn_2 < quantile_95)])), loss_dqn_2[
        np.where(loss_dqn_2 < quantile_95)]
    np.savetxt("loss_dqn_2_no_per.csv", loss_dqn_2, delimiter=",")
    dqn_2_actions_taken = np.array(dqn_2.steps_taken)
    np.savetxt("dqn_2_actions_taken_no_per.csv", dqn_2_actions_taken, delimiter=",")
    dqn_2_reward_eval = np.array(dqn_2.reward_eval)
    np.savetxt("dqn_2_reward_eval_no_per.csv", dqn_2_reward_eval, delimiter=",")



    #
    # env_dict = {0: "highway-fast-v0",
    #             1: "merge-v0",
    #             2: "roundabout-v0"}
    # multi_env_agent_1 = Agent(model_type='ddqn', betches_size=100, sync_models=50, per=True, env_number=3)
    # dqn_1.set_loger('multi_env_agent_log_1.csv')
    # multi_env_agent_1.train_agent(episodes=1, epsilon=0.6, discount_factor=0.9)
    # while (np.average(multi_env_agent_1.steps_taken[-5:]) < 50 and len(
    #         multi_env_agent_1.steps_taken) < episodes_to_learn):
    #     multi_env_agent_1.train_agent(episodes=5, epsilon=multi_env_agent_1.epsilon,
    #                                   discount_factor=multi_env_agent_1.discount_factor)
    #     multi_env_agent_1.env_name = env_dict[len(multi_env_agent_1.steps_taken) % 3]
    #     multi_env_agent_1.init_env()
    #
    # loss_dqn_1 = np.flip(pd.read_csv('multi_env_agent_log_1.csv', delimiter=';').loss.to_numpy())
    # quantile_95 = np.quantile(loss_dqn_1, 0.95)
    # loss_dqn_1 = np.arange(len(loss_dqn_1[np.where(loss_dqn_1 < quantile_95)])), loss_dqn_1[
    #     np.where(loss_dqn_1 < quantile_95)]
    # np.savetxt("loss_multi_env_agent_1.csv", loss_dqn_1, delimiter=",")
    # dqn_1_actions_taken = np.array(multi_env_agent_1.steps_taken)
    # np.savetxt("multi_env_agent_1_actions_taken.csv", dqn_1_actions_taken, delimiter=",")
    # dqn_1_reward_eval = np.array(dqn_1.reward_eval)
    # np.savetxt("multi_env_agent_1_reward_eval.csv", dqn_1_reward_eval, delimiter=",")
    #
    # highway_env_agent = Agent(model_type='ddqn', betches_size=100, sync_models=50, per=True, env_number=3,
    #                           env_name="highway-fast-v0")
    # merge_env_agent = Agent(model_type='ddqn', betches_size=100, sync_models=50, per=True, env_number=3,
    #                         env_name="merge-v0")
    # roundabout_env_agent = Agent(model_type='ddqn', betches_size=100, sync_models=50, per=True, env_number=3,
    #                              env_name="roundabout-v0", max_steps=50)
    # multi_env_agent_2 = Agent(model_type='ddqn', betches_size=100, sync_models=50, per=True, env_number=3)
    # multi_env_agent_2.set_loger('multi_env_agent_log_2.csv')
    #
    # # multi_env_agent.train_agent(episodes=1, epsilon=0.6, discount_factor=0.9)
    # highway_env_agent.q_model = multi_env_agent_2.q_model
    # merge_env_agent.q_model = multi_env_agent_2.q_model
    # roundabout_env_agent.q_model = multi_env_agent_2.q_model
    # overall_episodes = 0
    # while (overall_episodes < episodes_to_learn):
    #     highway_env_agent.train_agent(episodes=1, epsilon=highway_env_agent.epsilon,
    #                                   discount_factor=multi_env_agent_2.discount_factor)
    #     merge_env_agent.train_agent(episodes=1, epsilon=merge_env_agent.epsilon,
    #                                 discount_factor=multi_env_agent_2.discount_factor)
    #     roundabout_env_agent.train_agent(episodes=1, epsilon=roundabout_env_agent.epsilon,
    #                                      discount_factor=multi_env_agent_2.discount_factor)
    #     if overall_episodes % 5 == 0:
    #         high_way_weight_diffs = [x - y for x, y in zip(multi_env_agent_2.q_model.get_weights(),
    #                                                        highway_env_agent.q_model.get_weights())]
    #         merge_weight_diffs = [x - y for x, y in
    #                               zip(multi_env_agent_2.q_model.get_weights(), merge_env_agent.q_model.get_weights())]
    #         roundabout_weight_diffs = [x - y for x, y in zip(multi_env_agent_2.q_model.get_weights(),
    #                                                          roundabout_env_agent.q_model.get_weights())]
    #
    #         weight_diffs = [x + y for x, y in zip(merge_weight_diffs, high_way_weight_diffs)]
    #         weight_diffs = [x + y for x, y in zip(weight_diffs, roundabout_weight_diffs)]
    #         weight_diffs = [x / 3.0 for x, y in zip(weight_diffs, weight_diffs)]
    #         updated_weight = [x - y for x, y in zip(weight_diffs, multi_env_agent_2.q_model.get_weights())]
    #
    #         highway_env_agent.q_model.set_weights(updated_weight)
    #         merge_env_agent.q_model.set_weights(updated_weight)
    #         roundabout_env_agent.q_model.set_weights(updated_weight)
    #         multi_env_agent_2.q_model.set_weights(updated_weight)
    #     overall_episodes += 1
    #
    # dqn_1_actions_taken = np.array(roundabout_env_agent.steps_taken)
    # np.savetxt("multi_env_agent_2_roundabout_actions_taken.csv", dqn_1_actions_taken, delimiter=",")
    # dqn_1_reward_eval = np.array(roundabout_env_agent.reward_eval)
    # np.savetxt("multi_env_agent_2_roundabout_reward_eval.csv", dqn_1_reward_eval, delimiter=",")
    #
    # dqn_1_actions_taken = np.array(merge_env_agent.steps_taken)
    # np.savetxt("multi_env_agent_2_merge_actions_taken.csv", dqn_1_actions_taken, delimiter=",")
    # dqn_1_reward_eval = np.array(merge_env_agent.reward_eval)
    # np.savetxt("multi_env_agent_2_merge_reward_eval.csv", dqn_1_reward_eval, delimiter=",")
    #
    # dqn_1_actions_taken = np.array(highway_env_agent.steps_taken)
    # np.savetxt("multi_env_agent_2_highway_actions_taken.csv", dqn_1_actions_taken, delimiter=",")
    # dqn_1_reward_eval = np.array(highway_env_agent.reward_eval)
    # np.savetxt("multi_env_agent_2_highway_reward_eval.csv", dqn_1_reward_eval, delimiter=",")

