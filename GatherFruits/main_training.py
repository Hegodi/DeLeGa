from game_gatherfruits import Game
from agent_gatherfruits import Agent
import sys
sys.path.append("..\\")
from delega import TrainingSettings, Training

settings = TrainingSettings()
settings.result_name = "train_64"
settings.save_model_period = 50
settings.save_log_period = 25 
settings.n_games = 500
settings.epsilon_dec = 2.0e-3
settings.epsilon_min = 1e-4
settings.max_duration_game = 15.0
settings.update_nn_period = 10

if __name__ == '__main__':
    show_game = False
    game = Game(show_game)
    agent = Agent(game.get_state_size(), game.get_number_actions())
    Training(settings, game, agent)
