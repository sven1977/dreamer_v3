"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
"""
import tensorflow as tf

from models.components.cnn_atari import CNNAtari
from models.components.continue_predictor import ContinuePredictor
from models.components.conv_transpose_atari import ConvTransposeAtari
from models.components.dynamics_predictor import DynamicsPredictor
from models.components.representation_layer import RepresentationLayer
from models.components.reward_predictor import RewardPredictor
from models.components.sequence_model import SequenceModel


class WorldModel(tf.keras.Model):
    pass
