# DreamerV3
Implementation (TensorFlow/keras) of the "DreamerV3" model-based reinforcement learning
(RL) algorithm by Hafner et al. 2023

For algorithm details, please see:
https://arxiv.org/pdf/2301.04104v1.pdf

## Results



## Setup

```shell
# Install Atari env tools: 
pip install gymnasium[atari]==0.28.1 supersuit==3.8.0
# Clone this very repo.
git clone https://github.com/sven1977/dreamer_v3
cd dreamer_v3
# We use local imports, so make sure this dir is in your PYTHONPATH
export PYTHONPATH=$(pwd)

# Run the Atari example.
python run_experiment.py -c examples/atari_100k.yaml --env ALE/Pong-v5
```
