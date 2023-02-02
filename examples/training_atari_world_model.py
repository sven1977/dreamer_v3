import tensorflow as tf

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

from models.world_model_atari import WorldModelAtari
from utils.env_runner import EnvRunner
from losses.world_model_losses import (
    world_model_dynamics_and_representation_loss,
    world_model_prediction_losses,
)

# EnvRunner config (an RLlib algorithm config).
config = (
    AlgorithmConfig()
        .environment("ALE/MsPacman-v5")
        .rollouts(num_envs_per_worker=2, rollout_fragment_length=200)
)
# The vectorized gymnasium EnvRunner to collect samples of shape (B, T, ...).
env_runner = EnvRunner(model=None, config=config, max_seq_len=64)
# Our DreamerV3 world model.
world_model = WorldModelAtari(action_space=env_runner.env.single_action_space, batch_length_T=64)
# Use an Adam optimizer.
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-8)
# World model grad clipping according to [1].
grad_clip = 1000.0

last_h = None
for epoch in range(1):
    # Sample one round.
    obs, next_obs, actions, rewards, terminateds, truncateds, mask = (
        env_runner.sample(random_actions=True)
    )
    # Convert samples (numpy) to tensors.
    B, T = rewards.shape
    obs_tensor = tf.convert_to_tensor(obs)
    actions_tensor = tf.convert_to_tensor(actions)
    rewards_tensor = tf.convert_to_tensor(rewards)
    terminateds_tensor = tf.convert_to_tensor(terminateds)
    truncateds_tensor = tf.convert_to_tensor(truncateds)
    mask_tensor = tf.convert_to_tensor(mask)

    # Compute losses.
    with tf.GradientTape() as tape:
        # Compute forward values.
        forward_train_outs = world_model(
            inputs=obs_tensor,
            actions=actions_tensor,
            initial_h=None if last_h is None else last_h,
        )

        prediction_losses = world_model_prediction_losses(
            observations=obs_tensor,
            rewards=rewards_tensor,
            terminateds=terminateds_tensor,
            truncateds=truncateds_tensor,
            forward_train_outs=forward_train_outs,
        )
        L_pred = prediction_losses["total_loss"]

        L_dyn, L_rep = world_model_dynamics_and_representation_loss(
            forward_train_outs=forward_train_outs
        )
        L_total = 1.0 * L_pred + 0.5 * L_dyn + 0.1 * L_rep

        # Sum up timesteps, and average over batch (eq. 4 in [1]).
        L_total = tf.reshape(L_total, shape=(B, T))
        # Mask out invalid timesteps (episode terminated/truncated).
        L_total = L_total * mask_tensor
        L_total = tf.reduce_mean(tf.reduce_sum(L_total, axis=-1))

    # Get the gradients from the tape.
    gradients = tape.gradient(L_total, world_model.trainable_variables)
    # Clip all gradients.
    clipped_gradients = []
    for grad in gradients:
        clipped_gradients.append(tf.clip_by_value(grad, -grad_clip, grad_clip))
    # Apply gradients to our model.
    optimizer.apply_gradients(zip(clipped_gradients, world_model.trainable_variables))

    # Remember last deterministic (GRU) states for next epoch.
    last_h = forward_train_outs["h_tp1"]

    print(
        f"Epoch {epoch}) L_total={L_total.numpy()} "
        f"(L_pred={L_pred.numpy()}; L_dyn={L_dyn.numpy()}; L_rep={L_rep.numpy()} B={B} T={T})"
    )
