from game_gatherfruits import Game
from agent_gatherfruits import Agent
import sys
sys.path.append("..\\")
from delega import TrainingSettings, Training

settings = TrainingSettings()
settings.result_name = "train02"
settings.save_model_period = 100
settings.save_log_period = 20
settings.n_games = 1000
settings.epsilon_dec = 1.0e-2
settings.epsilon_min = 1e-4
settings.max_duration_game = 10.0
settings.save_period = 100
settings.update_nn_period = 10

if __name__ == '__main__':
    show_game = False
    game = Game(show_game)
    agent = Agent(game.get_state_size(), game.get_number_actions())
    Training(settings, game, agent)
