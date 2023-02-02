import tensorflow as tf

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

from models.atari_world_model import AtariWorldModel
from utils.env_runner import EnvRunner
from losses.world_model_losses import (
    world_model_dynamics_and_representation_loss,
    world_model_prediction_loss,
)


config = (
    AlgorithmConfig()
        .environment("ALE/MsPacman-v5")
        .rollouts(num_envs_per_worker=2, rollout_fragment_length=200)
)
env_runner = EnvRunner(model=None, config=config, max_seq_len=64)
world_model = AtariWorldModel(action_space=env_runner.env.single_action_space, batch_length_T=64)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-8)
last_h = None

for _ in range(1):
    obs, next_obs, actions, rewards, terminateds, truncateds, mask = (
        env_runner.sample(random_actions=True)
    )
    obs_tensor = tf.convert_to_tensor(obs)
    actions_tensor = tf.convert_to_tensor(actions)
    rewards_tensor = tf.convert_to_tensor(rewards)
    terminateds_tensor = tf.convert_to_tensor(terminateds)
    truncateds_tensor = tf.convert_to_tensor(truncateds)
    forward_train_outs = world_model(
        inputs=obs_tensor,
        actions=actions_tensor,
        initial_h=None if last_h is None else last_h,
    )
    last_h = forward_train_outs["h_tp1"]
    L_pred = world_model_prediction_loss(
        observations=obs_tensor,
        rewards=rewards_tensor,
        terminateds=terminateds_tensor,
        truncateds=truncateds_tensor,
        forward_train_outs=forward_train_outs,
    )
    L_dyn, L_rep = world_model_dynamics_and_representation_loss(
        forward_train_outs=forward_train_outs
    )
    L_total = L_pred + 0.5 * L_dyn + 0.1 * L_rep
    #optimizer.compute_gradients(L_total)
    print(L_total)
