# Halite III Bot

A Bot for Two Sigma's AI Competition.

## Contents

### Latest Bot

My final attempt at a deep learning powered Halite III bot.

#### Components
* `gather_data.py` was used to download matches from top players and convert
them into fake game state objects for training `MyBot.py`.
* `train_model.py` created and trained a model (shown below) to predict the actions
the bot should take for each game state. The model worked by taking as an observation
a game matrix `(64, 64, 7)` and a game state vector `(19, )` then producing an
action matrix `(64, 64, 6)`.
  * Model I/O: `(Game Matrix, Game State) -> Action Matrix`
    * `Game Matrix (64, 64, 7)`, where `GameMatrix[i, j, k]` represents the kth layer (layers are features such a dropoffs, ships, enemy ships, halite amount, etc) of row i, column j on the board.
    * `Game State (19, )`, where `GameState[i]` represents the ith global feature (meaning that the feature applies to all grid spaces and is not spacial feature). The vector contained information like total amount of halite stored, the number of ships, the turn #, etc
    * `Action Matrix (64, 64, 6)`, where `ActionMatrix[i, j, k]` represents the kth action designated for the unit located on row i, column j. The actions were encoded as a one-hot vector and model outputs were interpreted as a probability distribution. Predictions on grid spaces with no owned units were ignored.
  * Model Architecture
    * The model was designed to be very similar to [U-Net](https://arxiv.org/abs/1505.04597) such that the features where compressed down and the upscaled back to their original dimension with skip connections allowing for detailed features to be carried on to the output. The game state vector was injected in to the center of the model.
    * The final layer used softmax for predicting actions on every grid space and the loss function was calculated using categorical x-entropy with the loss to n/a units ignored.
* `MyBot.py` was the main runner for the bot. It contains two methods that decided how it will run.
    * `collect_data()` plays Halite while using another bot's moves (from an 'expert' or `gather_data.py`). This mode generates the training data for `train_model.py`.
    * `model_agent()` is the actual ML agent that uses the trained model created from `train_model.py` to play Halite.

![keras diagram](https://user-images.githubusercontent.com/6625384/51417197-e8df7800-1b4a-11e9-895a-89b7bfb0ce7f.png)

### Archive

Old Bots that I used to test against.

#### Bot G

This bot switched from the idea of having a separate (shared) model for
every unit to a single model that predicted a volume of actions (an action
vector was predicted for every pixel). The bot was trained with a
supervised dataset collected by an 'expert' bot playing against the archive.

#### Bot F

This bot was testing a mocking strategy where the reward was partially based
on how its actions were similar to an 'expert bot'.

#### Bot E

A non-ML bot that simply sends ships to/from areas of high halite concentration.

#### Bot D

I found this bot within the halite tools directory so I'm using it simply to test against.

#### Bot C

A refined version of ``Bot B`` with more parameters and more detail.

#### Bot B

Trying out an RL agent with basic game parameters as input and a discrete action as output
with the reward proportional to the halite gained each move.

#### Bot A

Randomly pick all actions, pretty much just the starter bot.

### Bin

Tools for running halite games.
