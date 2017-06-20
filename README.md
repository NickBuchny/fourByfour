# fourByfour

This is in response to this stack overflow question https://stackoverflow.com/questions/44617398/predicting-board-values-using-cnn

Which is about the Frozen Lake challenge on open AI

- https://gym.openai.com/envs/FrozenLake-v0
- https://github.com/NickBuchny/FrozenLake_OneStep/blob/master/FrozenLake_DataCollection.ipynb

The file called exampleLearn.py takes as input a series of starting boards [n,16] and expected next boards [n,16] and
creates a prediction matrix using a CNN.

# Usage

    python exampleLearn.py
    
# Example output

The output is the estimated next board and the error

    [[0 4 1 1 1 2 1 2 1 1 1 2 2 1 1 3]
     [0 1 1 1 1 2 1 0 1 1 1 2 2 4 1 3]] 0.140438
    [[0 4 1 1 1 2 1 2 1 1 1 2 2 1 1 3]
     [0 1 1 1 1 2 1 0 1 1 1 2 2 4 1 3]] 0.10456
    [[0 4 1 1 1 2 1 2 1 1 1 2 2 1 1 3]
     [0 1 1 1 1 2 1 0 1 1 1 2 2 4 1 3]] 0.0787147
    [[0 4 1 1 1 2 1 2 1 1 1 2 2 1 1 3]
     [0 1 1 1 1 2 1 0 1 1 1 2 2 4 1 3]] 0.0597783
