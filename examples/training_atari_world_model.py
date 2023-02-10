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
from models.components.cnn_atari import CNNAtari
from models.components.conv_transpose_atari import ConvTransposeAtari
from models.dreamer_model import DreamerModel
from models.world_model import WorldModel
from utils.env_runner import EnvRunner
from utils.continuous_episode_replay_buffer import ContinuousEpisodeReplayBuffer
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
# The number of timesteps we use to "initialize" (burn-in) a dream_trajectory run.
# For this many timesteps, the posterior (actual observation data) will be used
# to compute z, after that, only the prior (dynamics network) will be used.
burn_in_T = 5

# EnvRunner config (an RLlib algorithm config).
config = (
    AlgorithmConfig()
    #.environment("ALE/MontezumaRevenge-v5", env_config={
    .environment("ALE/MsPacman-v5", env_config={
        # [2]: "We follow the evaluation protocol of Machado et al. (2018) with 200M
        # environment steps, action repeat of 4, a time limit of 108,000 steps per
        # episode that correspond to 30 minutes of game play, no access to life
        # information, full action space, and sticky actions. Because the world model
        # integrates information over time, DreamerV2 does not use frame stacking.
        # The experiments use a single-task setup where a separate agent is trained
        # for each game. Moreover, each agent uses only a single environment instance.
        # already done by MaxAndSkip wrapper "frameskip": 4,  # "action repeat" (frameskip) == 4
        "repeat_action_probability": 0.25,  # "sticky actions"
        "full_action_space": True,  # "full action space"
    })
    .rollouts(
        num_envs_per_worker=batch_size_B,
        rollout_fragment_length=batch_length_T,
    )
)
# The vectorized gymnasium EnvRunner to collect samples of shape (B, T, ...).
env_runner = EnvRunner(
    model=None,
    config=config,
    max_seq_len=None,
    continuous_episodes=True,
)

# Our DreamerV3 world model.
from_checkpoint = None
# Uncomment this next line to load from a saved model.
#from_checkpoint = "checkpoints/dreamer_model_0"
if from_checkpoint is not None:
    dreamer_model = tf.keras.models.load_model(from_checkpoint)
else:
    model_dimension = "S"
    world_model = WorldModel(
        model_dimension=model_dimension,
        action_space=env_runner.env.single_action_space,
        batch_length_T=batch_length_T,
        encoder=CNNAtari(model_dimension=model_dimension),
        decoder=ConvTransposeAtari(
            model_dimension=model_dimension,
            gray_scaled=True,
        )
    )
    dreamer_model = DreamerModel(
        model_dimension=model_dimension,
        action_space=env_runner.env.single_action_space,
        world_model=world_model,
    )
# TODO: ugly hack (resulting from the insane fact that you cannot know
#  an env's spaces prior to actually constructing an instance of it) :(
env_runner.model = dreamer_model

# The replay buffer for storing actual env samples.
buffer = ContinuousEpisodeReplayBuffer(
    capacity=int(1e6 / batch_length_T),
    batch_size_B=batch_size_B,
)
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
        forward_train_outs = dreamer_model(
            inputs=sample["obs"],
            actions=sample["actions"],
            initial_h=sample["h_states"],
        )
        predicted_images = tf.reshape(
            tf.cast(
                tf.clip_by_value(inverse_symlog(
                    forward_train_outs["obs_distribution"].loc
                ), 0.0, 255.0),
                dtype=tf.uint8,
            ),
            shape=(-1, 64, 64),
        )
        # Concat sampled and predicted images along the height axis (2) such that
        # real images show on top of respective predicted ones.
        # (B, w, h, C)
        sampled_vs_predicted_images = tf.concat([predicted_images[0:batch_length_T], sample["obs"][0]], axis=1)
        tf.summary.image("sampled_vs_predicted(posterior)_images[0]", tf.expand_dims(sampled_vs_predicted_images, -1), step, max_outputs=20)

        #predicted_obs = tf.reshape(predicted_images, shape=(batch_size_B, batch_length_T) + predicted_images.shape[1:])
        mse_sampled_vs_predicted_obs = tf.losses.mse(
            tf.reshape(tf.cast(predicted_images, tf.float32), (batch_size_B, batch_length_T, -1)),
            tf.reshape(tf.cast(sample["obs"], tf.float32), (batch_size_B, batch_length_T, -1)),
        )
        # MSE difference between predicted and sampled observations: This must go to 0.
        diff_sampled_vs_predicted_obs = tf.reduce_mean(tf.reduce_sum(mse_sampled_vs_predicted_obs, axis=-1))
        tf.summary.scalar("MEAN(SUM(mse,T),B)_sampled_vs_predicted(posterior)_obs", diff_sampled_vs_predicted_obs, step)

        predicted_rewards = tf.reshape(
            inverse_symlog(forward_train_outs["reward_distribution"].mean()),
            shape=(batch_size_B, batch_length_T),
        )
        tf.summary.histogram("predicted_rewards", predicted_rewards, step)
        mse_sampled_vs_predicted_rewards = tf.losses.mse(predicted_rewards, sample["rewards"])
        # MSE difference between predicted and sampled rewards: This must go to 0.
        mse_sampled_vs_predicted_rewards = tf.reduce_mean(tf.reduce_sum(mse_sampled_vs_predicted_rewards, axis=-1))
        tf.summary.scalar("MEAN(SUM(mse,T),B)_sampled_vs_predicted(posterior)_rewards", mse_sampled_vs_predicted_rewards, step)

        prediction_losses = world_model_prediction_losses(
            observations=sample["obs"],
            rewards=sample["rewards"],
            terminateds=sample["terminateds"],
            truncateds=sample["truncateds"],
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

        # Two-hot reward loss.
        L_reward_two_hot_BxT = prediction_losses["reward_loss_two_hot"]
        L_reward_two_hot = tf.reduce_mean(tf.reduce_sum(L_reward_two_hot_BxT, axis=-1))
        tf.summary.histogram("L_reward_two_hot_BxT", L_reward_two_hot_BxT, step)
        tf.summary.scalar("L_reward_two_hot", L_reward_two_hot, step)
        # TEST: Out of interest, compare with simplge -log(p) loss for individual
        # rewards using the FiniteDiscrete distribution. This should be very close
        # to the two-hot reward loss.
        L_reward_logp_BxT = prediction_losses["reward_loss_logp"]
        L_reward_logp = tf.reduce_mean(tf.reduce_sum(L_reward_logp_BxT, axis=-1))
        tf.summary.histogram("L_reward_logp_BxT", L_reward_logp_BxT, step)
        tf.summary.scalar("L_reward_logp", L_reward_logp, step)

        L_continue_BxT = prediction_losses["continue_loss"]
        L_continue = tf.reduce_mean(tf.reduce_sum(L_continue_BxT, axis=-1))
        tf.summary.histogram("L_continue_BxT", L_continue_BxT, step)
        tf.summary.scalar("L_continue", L_continue, step)

        L_dyn_BxT, L_rep_BxT = world_model_dynamics_and_representation_loss(
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

        # Compute the actual total loss using fixed weights described in [1] eq. 4.
        L_total_BxT = 1.0 * L_pred_BxT + 0.5 * L_dyn_BxT + 0.1 * L_rep_BxT
        tf.summary.histogram("L_total_BxT", L_total_BxT, step)

        # Sum up timesteps, and average over batch (see eq. 4 in [1]).
        L_total = tf.reduce_mean(tf.reduce_sum(L_total_BxT, axis=-1))
        tf.summary.scalar("L_total", L_total, step)

    # Get the gradients from the tape.
    gradients = tape.gradient(L_total, dreamer_model.trainable_variables)
    # Clip all gradients.
    clipped_gradients = []
    for grad in gradients:
        clipped_gradients.append(tf.clip_by_value(grad, -grad_clip, grad_clip))
    # Apply gradients to our model.
    optimizer.apply_gradients(zip(clipped_gradients, dreamer_model.trainable_variables))

    return L_total, L_pred, L_dyn, L_rep


total_env_steps = 0
total_replayed_steps = 0
total_train_steps = 0


for iteration in range(1000):
    # Push enough samples into buffer initially before we start training.
    env_steps = env_steps_last_sample = 0
    #TEST: Put only a single row in the buffer and try to memorize it.
    #env_steps_last_sample = 64
    #while iteration == 0:
    #END TEST
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

                # EVALUATION:
                # Dream a trajectory using the samples from the buffer and compare obs, rewards,
                # continues to the actually observed trajectory.
                dreamed_T = batch_length_T - burn_in_T
                dream_data = dreamer_model.dream_trajectory(
                    observations=sample["obs"][:, :burn_in_T],  # use only first burn_in_T obs
                    actions=sample["actions"],  # use all actions from 0 to T (no actor)
                    initial_h=sample["h_states"],
                    timesteps=dreamed_T,  # dream for T-burn_in_T timesteps
                    use_sampled_actions=True,  # use all actions from 0 to T (no actor)
                )
                # Obs MSE.
                # Compute observations using h and z and the decoder net.
                # Note that the last h-state is NOT used here as it's already part of
                # a new trajectory.
                _, dreamed_obs_distr = dreamer_model.world_model.decoder(
                    h=tf.reshape(dream_data["h_states"][:, :-1], (batch_size_B * dreamed_T, -1)),
                    z=tf.reshape(dream_data["z_dreamed"],
                               (batch_size_B * dreamed_T) + dream_data["z_dreamed"].shape[2:]),
                )
                # Use mean() of the Gaussian, no sample!
                dreamed_obs = tf.reshape(dreamed_obs_distr.mean(), (batch_size_B, dreamed_T) + sample["obs"].shape[2:]).numpy()
                mse_sampled_vs_dreamed_obs = tf.losses.mse(
                    dreamed_obs,
                    tf.cast(sample["obs"][:,burn_in_T:], tf.float32),
                )
                mse_sampled_vs_dreamed_obs = tf.reduce_mean(tf.reduce_sum(mse_sampled_vs_dreamed_obs, axis=1))
                tf.summary.scalar("MEAN(SUM(mse,T),B)_sampled_vs_dreamed(prior)_obs", mse_sampled_vs_dreamed_obs, step=total_train_steps_tensor)

                # Reward MSE.
                mse_sampled_vs_dreamed_rewards = tf.losses.mse(
                    tf.expand_dims(dream_data["rewards_dreamed"], axis=-1),
                    tf.expand_dims(sample["rewards"][:,burn_in_T:], axis=-1),
                )
                mse_sampled_vs_dreamed_rewards = tf.reduce_mean(tf.reduce_sum(mse_sampled_vs_dreamed_rewards, axis=1))
                tf.summary.scalar("MEAN(SUM(mse,T),B)_sampled_vs_dreamed(prior)_rewards", mse_sampled_vs_dreamed_rewards, step=total_train_steps_tensor)
                # Continue MSE.
                mse_sampled_vs_dreamed_continues = tf.losses.mse(
                    tf.expand_dims(tf.cast(dream_data["continues_dreamed"], tf.float32), axis=-1),
                    tf.expand_dims(tf.cast(tf.logical_not(tf.logical_or(sample["terminateds"][:,burn_in_T:], sample["truncateds"][:,burn_in_T:])), tf.float32), axis=-1),
                )
                mse_sampled_vs_dreamed_continues = tf.reduce_mean(tf.reduce_sum(mse_sampled_vs_dreamed_continues, axis=1))
                tf.summary.scalar("MEAN(SUM(mse,T),B)_sampled_vs_dreamed(prior)_continues", mse_sampled_vs_dreamed_continues, step=total_train_steps_tensor)

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
        dreamer_model.save(f"checkpoints/dreamer_model_{iteration}")

    total_replayed_steps += replayed_steps
    print(
        f"\treplayed-steps: {total_replayed_steps}; env-steps: {total_env_steps}"
    )
