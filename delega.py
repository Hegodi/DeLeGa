################################################################################
################################################################################
####                                                                        ####
####                    Deep Learning for Games (DeLeGa)                    ####
####                                                                        ####
####   Author:                                                              ####
####           Diego Gonzalez Herrero                                       ####
####           diegonher@gmail.com                                          ####
####                                                                        ####
################################################################################
################################################################################

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import numpy as np
import pygame
import time
import os
from matplotlib import pyplot as plt 


################################################################################
class TrainingSettings():
    def __init__(self):
        self.FPS = 30.0
        self.n_games = 500
        self.max_duration_game = 15.0

        self.lock_fps = False
        self.show_plot = False

        self.result_name = "test03"  # Result names
        self.save_model_period = 20  # Number of episodes between model saves
        self.save_log_period = 20       # Number of episodes between log saves
        self.gamma = 0.99            # Discounted reward factor
        self.lr = 1.0e-3             # Learning rate               
        self.epsilon_ini = 1.0       # Exploration factor: start
        self.epsilon_dec = 5.0e-4    # Exploration factor: decrement
        self.epsilon_min = 1.0e-2    # Exploration factor: min value
        self.update_nn_period = 10   # Number of frames between NN update
        self.replace = 100           # Number of NN trainings before swaping the two NN
        self.batch_size = 64         # Size of training batches
        self.mem_size = 100000       # Max memory size for replay buffer

    def get_info_text(self):
        text = "TRAINING SETTINGS\n\n"
        text += "result name: " + self.result_name + "\n" 
        text += "FPS: %d \n" % self.FPS 
        text += "n_games: %d \n" % self.n_games 
        text += "max_duration_game: %.1f \n" % self.max_duration_game 
        text += "save_period: %d \n" % self.save_period 
        text += "gamma: %.4f \n" % self.gamma 
        text += "lr: %.5f \n" % self.lr 
        text += "epsilon_ini: %.5f \n" % self.epsilon_ini 
        text += "epsilon_dec: %.5f \n" % self.epsilon_dec 
        text += "epsilon_min: %.5f \n" % self.epsilon_min 
        text += "update_nn_period: %d \n" % self.update_nn_period 
        text += "replace: %d \n" % self.replace 
        text += "batch_size: %d \n" % self.batch_size 
        text += "mem_size: %d \n" % self.mem_size 
        text += "\n\n"
        return text

################################################################################
class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_count = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.next_state_memory = np.zeros((self.mem_size, *input_shape),dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_count % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.terminal_memory[index] = done

        self.mem_count += 1        
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_count, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        next_states = self.next_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, next_states, dones

################################################################################
class AgentBase():

    def load(self, modelpath):
        self.q_eval = load_model(modelpath)

    def chose_action(self, state):
        state = np.array([state])
        actions_advantages = self.q_eval(state)
        action = tf.math.argmax(actions_advantages, axis=1).numpy()[0]
        return action

    def compile_model(self, lr):
        pass

################################################################################
class AgentTrainer():
    def __init__(self, agent, n_inputs, n_outputs, lr, gamma, epsilon, batch_size, eps_dec=1e-3, eps_min=0.01, 
                mem_size=100000, replace=100):
        self.agent = agent
        self.action_space = [i for i in range(n_outputs)]
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.replace = replace

        self.learn_step_counter = 0
        self.memory = ReplayBuffer(mem_size, n_inputs)
        self.q_next = tf.keras.models.clone_model(self.agent.q_eval)

        self.agent.compile_model(lr)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def chose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([state])
            actions_values = self.agent.q_eval(state)
            action = tf.math.argmax(actions_values, axis=1).numpy()[0]
        return action

    def learn(self):
        if (self.memory.mem_count < self.batch_size):
            return

        if (self.learn_step_counter % self.replace == 0):
            self.q_next.set_weights(self.agent.q_eval.get_weights())
        
        states, actions, rewards, next_states, dones = self.memory.sample_buffer(self.batch_size)

        q_pred = self.agent.q_eval(states)
        q_next = self.q_next(next_states)
        q_target = q_pred.numpy()
        max_actions = tf.math.argmax(self.agent.q_eval(next_states), axis=1)
        for idx, terminal in enumerate(dones):
            q_target[idx, actions[idx]] = rewards[idx] + self.gamma*q_next[idx, max_actions[idx]] * (1 - int(dones[idx]))
        self.agent.q_eval.train_on_batch(states, q_target)

        self.learn_step_counter += 1

    def save_model(self, foldername):
        self.agent.q_eval.save(foldername)
        print("Model saved")

    def update_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min


################################################################################
class GameCore:
    def __init__(self, numPlayers, numActions):
        pygame.init()
        pygame.font.init()
        self.numPlayers = numPlayers
        self.numActions = numActions
        self.playerActions = np.zeros(self.numActions)
        self.scores = np.zeros(self.numPlayers)

    def set_action_player(self, indPlayer, indAction):
        self.playerActions[indPlayer] = indAction

    def get_state_size(self):
        pass

    def get_number_actions(self):
        pass

    def reset(self):
        pass

    def get_score():
        pass

    
################################################################################
def Training(settings, game, agent):
    os.mkdir(settings.result_name)
    clock = pygame.time.Clock()
    state_size = game.get_state_size()
    number_actions = game.get_number_actions()
    agentTrain = AgentTrainer(agent=agent, gamma=settings.gamma, lr=settings.lr, epsilon=settings.epsilon_ini, eps_dec=settings.epsilon_dec, eps_min=settings.epsilon_min,
                  mem_size=settings.mem_size, batch_size=settings.batch_size, replace=settings.replace,
                  n_inputs=state_size, n_outputs=number_actions)
    
    myFile = open(settings.result_name + "\log.txt", mode="w")
    myFile.write("#Episode  Epsilon     Score   ScoreAvg\n")
    myFile.close()

    myFile = open(settings.result_name + "\settings.txt", mode="w")
    myFile.write(settings.get_info_text())
    agent.q_eval.summary(print_fn=lambda x: myFile.write(x + '\n'))
    myFile.close()

    myFile = open(settings.result_name + "\model.json", mode="w")
    myFile.write(agent.q_eval.to_json())
    myFile.close()

    log = []
    scores = []

    t_start = time.time()
    for i in range(settings.n_games):
        done = False
        game.reset()
        state = game.get_state()
        score = 0
        t = 0
        counter = 0
        t0 = time.time()
        while not done:
            action = agentTrain.chose_action(state)
            game.set_action_player(0, action)

            if (settings.lock_fps):
                dt = clock.get_time()/1000.0
            else:
                dt = 1.0/settings.FPS

            t += dt
            game.update(dt)
            game.render()

            done = t >= settings.max_duration_game
            next_state = game.get_state()
            reward = game.get_score() - score

            agentTrain.store_transition(state, action, reward, next_state, done)

            state = next_state
            score = game.get_score()

            if (settings.lock_fps):
                clock.tick(settings.FPS)

            counter +=1
            if (counter % settings.update_nn_period == 0):
                agentTrain.learn() 

        scores.append(score)
        kernel_size = 10
        avg_score = np.mean(scores[-kernel_size:])
        duration = time.time() - t0
        log.append([i, agentTrain.epsilon, score, avg_score])

        print("episode ", i, " score %.1f" % score, ' average score %.1f' % avg_score, " epsilon %.5f" % agentTrain.epsilon, " [%.1f s]" % duration)

        if ((i % settings.save_model_period == 0) and (i > 0)) or i == (settings.n_games-1):
            agentTrain.save_model(settings.result_name + "\model_" + str(i))

        if ((i % settings.save_log_period == 0) and (i > 0)) or i == (settings.n_games-1):
            myFile = open(settings.result_name + "\log.txt", mode="a")
            for line in log:
                txt = '%8d  ' % line[0] + '%8.6f  ' %  line[1] +  '%8.3f  ' % line[2] + '%8.4f  ' % line[3] + '\n'
                myFile.write(txt)
            myFile.close()
            kernel = np.ones(kernel_size) / kernel_size
            scores_convolved = np.convolve(scores, kernel, mode='same') 
            generate_plots(scores, settings.result_name + "\scores_history.png")
            generate_plots(scores_convolved, settings.result_name + "\scores_smooth_history.png")
            log.clear()

        agentTrain.update_epsilon()

    t_end = time.time()
    totalTime = (t_end - t_start)/60
    print("\n\n Training Finished\n Total time: %.1f min\n" % totalTime)



################################################################################
def generate_plots(scores, filename, show = False, save = True):
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    x = np.linspace(0, len(scores))
    ax.plot(scores)
    ax.set_xlabel("episodes")
    ax.set_ylabel("score")
    if (save):
        plt.savefig(filename)
    if (show):
        plt.show()



