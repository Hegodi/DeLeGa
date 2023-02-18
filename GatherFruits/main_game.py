import sys
import pygame

################################################################

from game_gatherfruits import Game
from agent_gatherfruits import Agent

################################################################

if __name__ == '__main__':

    playerInput = True
    game = Game()
    clock = pygame.time.Clock()

    agent = Agent(game.get_state_size(), 3)
    if (len(sys.argv) > 1):
        modelpath = sys.argv[1]
        agent.load(modelpath)
        playerInput = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Player Input
        if (playerInput):
            if pygame.key.get_pressed()[pygame.K_a] or pygame.key.get_pressed()[pygame.K_LEFT]:
                game.set_action_player(0, 1)
            elif pygame.key.get_pressed()[pygame.K_d] or pygame.key.get_pressed()[pygame.K_RIGHT] :
                game.set_action_player(0, 2)
            else:
                game.set_action_player(0, 0)
        else:
            action = agent.chose_action(game.get_state())
            game.set_action_player(0, action)

        dt = clock.get_time()/1000.0
        game.update(dt)
        game.render()
        clock.tick(30)
            