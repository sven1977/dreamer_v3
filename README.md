# DreamerV3
Implementation (TensorFlow/keras) of the "DreamerV3" model-based reinforcement learning
(RL) algorithm by Hafner et al. 2023

For algorithm details, please see:
https://arxiv.org/pdf/2301.04104v1.pdf

## Results



## Setup

```shell
# Install Atari env tools: 
pip install gymnasium[atari] gym==0.26.2 supersuit tinyscaler
# Clone this very repo.
git clone https://github.com/sven1977/dreamer_v3
cd dreamer_v3
# We use local imports, so make sure this dir is in your PYTHONPATH
export PYTHONPATH=[whatever dir the previous pwd command had output]

# Sorry, one more ugly hack: Supersuit does not seem to work yet with the
# new gymnasium vector envs. Hence, you will have to make one change to
# one of the supersuit source files:
pip show supersuit
# You should see something like:
# Name: SuperSuit
# Version: 3.7.1
# ..
# Location: [some path P ...]

# Edit the source file:
vim [some path P ...]/supersuit/lambda_wrappers/observation_lambda.py
# scroll down to the `class gym_observation_lambda` and simplify its `reset()`
# method to:
# def reset(self, seed=None, return_info=False, options=None):
#     observation, info = self.env.reset(
#         seed=seed, options=options
#     )
#     observation = self._modify_observation(observation)
#     return observation, info
# Save your changes and exit vim

# Run the Atari example.
python training_atari_world_model.py
```
