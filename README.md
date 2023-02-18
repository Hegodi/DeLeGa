    
    DEEP LEARNING IN GAMES 
    (DeLeGa)

        Diego Gonzalez Herrero
        February 2023

    Description:
        Simple Deep Learning framework to experiment with DL in video games.
        The module delega.py provides a simple framework for creating agents 
        and train them in games.

    Dependencies:
        (I higly recommended install Anaconda and create a new environment)
        Python >3.7
        tensorflow
        pygame
        numpy
        matplotlib

    Usage:
        Each project has to main python scripts:
            main_game.py
            main_training.py

        To run the game with a given model use:
            python main_game.py path/to/model
        
        To run play the game yourself just use:
            python main_game.py

        To train a model edit the settings in main_training.py and run
            python main_training.py

        You can change the model you are going to train in agent_*****.py
