"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
"""

import os
import tree  # pip install dm_tree
import tensorflow as tf

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

from losses.world_model_losses import (
    world_model_dynamics_and_representation_loss,
    world_model_prediction_losses,
)
from models.world_model_atari import WorldModelAtari
from utils.env_runner import EnvRunner
from utils.replay_buffer import ReplayBuffer
from utils.symlog import inverse_symlog

# Create the checkpoint path, if it doesn't exist yet.
os.makedirs("checkpoints", exist_ok=True)
# Create the tensorboard summary data dir.
os.makedirs("tensorboard", exist_ok=True)
tb_writer = tf.summary.create_file_writer("tensorboard")
# Every how many training steps do we write data to TB?
summary_frequency = 50
# Every how many (main) iterations (sample + N train steps) do we save our model?
model_save_frequency = 10

# Set batch size and -length according to [1]:
batch_size_B = 16
batch_length_T = 64

# EnvRunner config (an RLlib algorithm config).
config = (
    AlgorithmConfig()
    .environment("ALE/MsPacman-v5", env_config={
        # DreamerV3 paper does not specify, whether Atari100k is run
        # w/ or w/o sticky actions, just that frameskip=4.
        "frameskip": 4,
        "repeat_action_probability": 0.0,
    })
    .rollouts(num_envs_per_worker=1, rollout_fragment_length=64)
)
# The vectorized gymnasium EnvRunner to collect samples of shape (B, T, ...).
env_runner = EnvRunner(model=None, config=config, max_seq_len=batch_length_T)

# Our DreamerV3 world model.
from_checkpoint = None
# Uncomment this next line to load from a saved model.
#from_checkpoint = "checkpoints/world_model_0"
if from_checkpoint is not None:
    world_model = tf.keras.models.load_model(from_checkpoint)
else:
    world_model = WorldModelAtari(
        model_dimension="S",
        action_space=env_runner.env.single_action_space,
        batch_length_T=batch_length_T,
    )
# TODO: ugly hack (resulting from the insane fact that you cannot know
#  an env's spaces prior to actually constructing an instance of it) :(
env_runner.model = world_model

# The replay buffer for storing actual env samples.
buffer = ReplayBuffer(capacity=int(1e6 / batch_length_T))
# Timesteps to put into the buffer before the first learning step.
warm_up_timesteps = 0

# Use an Adam optimizer.
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-8)

# World model grad clipping according to [1].
grad_clip = 1000.0

# Training ratio: Ratio of replayed steps over env steps.
training_ratio = 1024


@tf.function
def train_one_step(sample, step):
    tf.summary.histogram("sampled_rewards", sample["rewards"], step)

    # Compute losses.
    with tf.GradientTape() as tape:
        # Compute forward values.
        forward_train_outs = world_model(
            inputs=sample["obs"],
            actions=sample["actions"],
            initial_h=sample["h_states"],
        )
        predicted_images_b0 = tf.reshape(
            tf.cast(
                tf.clip_by_value(inverse_symlog(
                    forward_train_outs["obs_distribution"].loc[0:batch_length_T]
                ), 0.0, 255.0),
                dtype=tf.uint8,
            ),
            shape=(-1, 64, 64, 3),
        )
        # Concat sampled and predicted images along the height axis (2) such that
        # real images show on top of respective predicted ones.
        # (B, w, h, C)
        sampled_vs_predicted_images = tf.concat([predicted_images_b0, sample["obs"][0]], axis=1)
        tf.summary.image("sampled_vs_predicted_images[0]", sampled_vs_predicted_images, step)
        tf.summary.histogram(
            "predicted_rewards",
            tf.reshape(
                inverse_symlog(forward_train_outs["reward_distribution"].mean()),
                shape=(batch_size_B, batch_length_T),
            ),
            step,
        )

        prediction_losses = world_model_prediction_losses(
            observations=sample["obs"],
            rewards=sample["rewards"],
            terminateds=sample["terminateds"],
            truncateds=sample["truncateds"],
            mask=sample["mask"],
            B=tf.convert_to_tensor(batch_size_B),
            T=tf.convert_to_tensor(batch_length_T),
            forward_train_outs=forward_train_outs,
        )
        L_pred_BxT = prediction_losses["total_loss"]
        L_pred = tf.reduce_mean(tf.reduce_sum(L_pred_BxT, axis=-1))
        tf.summary.histogram("L_pred_BxT", L_pred_BxT, step)
        tf.summary.scalar("L_pred", L_pred, step)

        L_decoder_BxT = prediction_losses["decoder_loss"]
        L_decoder = tf.reduce_mean(tf.reduce_sum(L_decoder_BxT, axis=-1))
        tf.summary.histogram("L_decoder_BxT", L_decoder_BxT, step)
        tf.summary.scalar("L_decoder", L_decoder, step)

        L_reward_BxT = prediction_losses["reward_loss"]
        L_reward = tf.reduce_mean(tf.reduce_sum(L_reward_BxT, axis=-1))
        tf.summary.histogram("L_reward_BxT", L_reward_BxT, step)
        tf.summary.scalar("L_reward", L_reward, step)

        L_continue_BxT = prediction_losses["continue_loss"]
        L_continue = tf.reduce_mean(tf.reduce_sum(L_continue_BxT, axis=-1))
        tf.summary.histogram("L_continue_BxT", L_continue_BxT, step)
        tf.summary.scalar("L_continue", L_continue, step)

        L_dyn_BxT, L_rep_BxT = world_model_dynamics_and_representation_loss(
            mask=sample["mask"],
            B=tf.convert_to_tensor(batch_size_B),
            T=tf.convert_to_tensor(batch_length_T),
            forward_train_outs=forward_train_outs,
        )
        L_dyn = tf.reduce_mean(tf.reduce_sum(L_dyn_BxT, axis=-1))
        tf.summary.histogram("L_dyn_BxT", L_dyn_BxT, step)
        tf.summary.scalar("L_dyn", L_dyn, step)

        L_rep = tf.reduce_mean(tf.reduce_sum(L_rep_BxT, axis=-1))
        tf.summary.histogram("L_rep_BxT", L_rep_BxT, step)
        tf.summary.scalar("L_rep", L_rep, step)

        L_total_BxT = 1.0 * L_pred_BxT + 0.5 * L_dyn_BxT + 0.1 * L_rep_BxT
        tf.summary.histogram("L_total_BxT", L_total_BxT, step)

        # Sum up timesteps, and average over batch (see eq. 4 in [1]).
        L_total = tf.reduce_mean(tf.reduce_sum(L_total_BxT, axis=-1))
        tf.summary.scalar("L_total", L_total, step)

    # Get the gradients from the tape.
    gradients = tape.gradient(L_total, world_model.trainable_variables)
    # Clip all gradients.
    clipped_gradients = []
    for grad in gradients:
        clipped_gradients.append(tf.clip_by_value(grad, -grad_clip, grad_clip))
    # Apply gradients to our model.
    optimizer.apply_gradients(zip(clipped_gradients, world_model.trainable_variables))

    return L_total, L_pred, L_dyn, L_rep


total_env_steps = 0
total_replayed_steps = 0
total_train_steps = 0

for iteration in range(1000):
    # Push enough samples into buffer initially before we start training.
    env_steps = env_steps_last_sample = 0
    while True:
        # Sample one round.
        # TODO: random_actions=False; right now, we act randomly, but perform a
        #  world-model forward pass using the random actions (in order to compute
        #  the h-states). Note that a world-model forward pass does NOT compute any
        #  new actions. This is covered by the Actor network.
        (
            obs,
            next_obs,
            actions,
            rewards,
            terminateds,
            truncateds,
            h_states,
            mask,
        ) = env_runner.sample(random_actions=True)

        # We took B x T env steps.
        env_steps_last_sample = rewards.shape[0] * rewards.shape[1]
        env_steps += env_steps_last_sample

        buffer.add({
            "obs": obs,
            "next_obs": next_obs,
            "actions": actions,
            "rewards": rewards,
            "terminateds": terminateds,
            "truncateds": truncateds,
            "mask": mask,
            "h_states": h_states,
        })
        print(f"Sampled env-steps={env_steps}; buffer-size={len(buffer)}")

        if (
            # Got to have more timesteps than warm up setting.
            len(buffer) * batch_length_T > warm_up_timesteps
            # But also more episodes (rows) than the batch size B.
            and len(buffer) >= batch_size_B
        ):
            break

    total_env_steps += env_steps

    replayed_steps = 0

    sub_iter = 0
    while replayed_steps / env_steps_last_sample < training_ratio:
        # Draw a sample from the replay buffer.
        sample = buffer.sample(num_items=batch_size_B)
        replayed_steps += batch_size_B * batch_length_T

        # Convert samples (numpy) to tensors.
        sample = tree.map_structure(lambda v: tf.convert_to_tensor(v), sample)
        total_train_steps_tensor = tf.convert_to_tensor(
            total_train_steps, dtype=tf.int64
        )

        # Perform one training step.
        if total_train_steps % summary_frequency == 0:
            with tb_writer.as_default():
                L_total, L_pred, L_dyn, L_rep = train_one_step(sample, total_train_steps_tensor)
        else:
            L_total, L_pred, L_dyn, L_rep = train_one_step(sample, total_train_steps_tensor)

        print(
            f"Iter {iteration}/{sub_iter}) L_total={L_total.numpy()} "
            f"(L_pred={L_pred.numpy()}; L_dyn={L_dyn.numpy()}; L_rep={L_rep.numpy()})"
        )
        sub_iter += 1
        total_train_steps += 1

    # Save the model every N iterations (but not after the very first).
    if iteration != 0 and iteration % model_save_frequency == 0:
        world_model.save(f"checkpoints/world_model_{iteration}")

    total_replayed_steps += replayed_steps
    print(
        f"\treplayed-steps: {total_replayed_steps}; env-steps: {total_env_steps}"
    )
