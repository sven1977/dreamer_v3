"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf

[2] Mastering Atari with Discrete World Models - 2021
D. Hafner, T. Lillicrap, M. Norouzi, J. Ba
https://arxiv.org/pdf/2010.02193.pdf
"""

import os

import numpy as np
import tree  # pip install dm_tree
import tensorflow as tf

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

from models.components.cnn_atari import CNNAtari
from models.components.conv_transpose_atari import ConvTransposeAtari
from models.dreamer_model import DreamerModel
from models.world_model import WorldModel
from training.train_one_step import (
    train_actor_and_critic_one_step,
    train_world_model_one_step,
)
from utils.env_runner import EnvRunner
from utils.continuous_episode_replay_buffer import ContinuousEpisodeReplayBuffer
from utils.tensorboard import (
    summarize_dreamed_trajectory_vs_samples,
    summarize_forward_train_outs_vs_samples,
    summarize_world_model_losses,
)

# Create the checkpoint path, if it doesn't exist yet.
os.makedirs("checkpoints", exist_ok=True)
# Create the tensorboard summary data dir.
os.makedirs("tensorboard", exist_ok=True)
tb_writer = tf.summary.create_file_writer("tensorboard")
# Every how many training steps do we write data to TB?
summary_frequency_train_steps = 5
# Every how many main iterations do we evaluate?
evaluation_frequency_main_iters = 1
evaluation_num_episodes = 10
# Every how many (main) iterations (sample + N train steps) do we save our model?
model_save_frequency_main_iters = 10000

# Set batch size and -length according to [1]:
batch_size_B = 16
batch_length_T = 64
# The number of timesteps we use to "initialize" (burn-in) a dream_trajectory run.
# For this many timesteps, the posterior (actual observation data) will be used
# to compute z, after that, only the prior (dynamics network) will be used.
burn_in_T = 5
horizon_H = 15

# Actor/critic hyperparameters.
discount_gamma = 0.997  # [1] eq. 7.
gae_lambda = 0.95  # [1] eq. 7.
entropy_scale = 3e-4  # [1] eq. 11.
return_normalization_decay = 0.99  # [1] eq. 11 and 12.


# EnvRunner config (an RLlib algorithm config).
config = (
    AlgorithmConfig()
    #.environment("ALE/MontezumaRevenge-v5", env_config={
    .environment("ALE/Pong-v5", env_config={
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
        num_envs_per_worker=1,
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
env_runner_evaluation = EnvRunner(
    model=None,
    config=config,
    max_seq_len=None,
    continuous_episodes=True,
)

# Whether to o nly train the world model (not the critic and actor networks).
train_world_model_only = False
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

#TEST: OOM
print("current mem:", tf.config.experimental.get_memory_info('GPU:0')['current'])

# TODO: ugly hack (resulting from the insane fact that you cannot know
#  an env's spaces prior to actually constructing an instance of it) :(
env_runner.model = dreamer_model
env_runner_evaluation.model = dreamer_model

# The replay buffer for storing actual env samples.
buffer = ContinuousEpisodeReplayBuffer(
    capacity=int(1e6 / batch_length_T),
    # Only add (B=1, T=...)-style data at a time.
    num_data_tracks=1,
)
# Timesteps to put into the buffer before the first learning step.
warm_up_timesteps = 0

# Use an Adam optimizer.
world_model_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-8)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-5)
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-5)

# World model grad clipping according to [1] Appendix W.
world_model_grad_clip = 1000.0
# Critic grad clipping according to [1] Appendix W.
critic_grad_clip = 100.0
# Actor grad clipping according to [1] Appendix W.
actor_grad_clip = 100.0

# Training ratio: Ratio of replayed steps over env steps.
training_ratio = 1024

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
        (
            obs,
            next_obs,
            actions,
            rewards,
            terminateds,
            truncateds,
            h_states,
        ) = env_runner.sample(random_actions=False)

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
            # But also at least as many trajectories as the batch size B.
            and len(buffer) >= batch_size_B
        ):
            break

    # Summarize actual environment interaction data.
    metrics = env_runner.get_metrics()
    with tb_writer.as_default(step=total_train_steps):
        # Summarize buffer length.
        tf.summary.scalar("buffer_size_num_trajectories", len(buffer))
        tf.summary.scalar("buffer_size_timesteps", len(buffer) * batch_length_T)
        # Summarize episode returns.
        if metrics["episode_returns"]:
            episode_return_mean = np.mean(metrics["episode_returns"])
            tf.summary.scalar("ENV_episode_return_mean", episode_return_mean)
        # Summarize actions taken.
        tf.summary.histogram("ENV_actions_taken", actions)

    total_env_steps += env_steps

    replayed_steps = 0

    sub_iter = 0
    while replayed_steps / env_steps_last_sample < training_ratio:
        # Enable TB summaries this step?
        tb_ctx = None
        if total_train_steps % summary_frequency_train_steps == 0:
            tb_ctx = tb_writer.as_default(step=total_train_steps)
            tb_ctx.__enter__()

        # Draw a sample from the replay buffer.
        sample = buffer.sample(num_items=batch_size_B)
        replayed_steps += batch_size_B * batch_length_T

        # Convert samples (numpy) to tensors.
        sample = tree.map_structure(lambda v: tf.convert_to_tensor(v), sample)

        # Perform one world-model training step.
        world_model_train_results = train_world_model_one_step(
            sample=sample,
            batch_size_B=batch_size_B,
            batch_length_T=batch_length_T,
            grad_clip=world_model_grad_clip,
            world_model=world_model,
            optimizer=world_model_optimizer,
        )
        # TEST: OOM
        print("\tafter world-model train:", tf.config.experimental.get_memory_info('GPU:0')['current'])

        summarize_forward_train_outs_vs_samples(
            forward_train_outs=world_model_train_results["forward_train_outs"],
            sample=sample,
            batch_size_B=batch_size_B,
            batch_length_T=batch_length_T,
        )
        summarize_world_model_losses(world_model_train_results)

        print(f"Iter {iteration}/{sub_iter})")
        print(
            f"\tL_world_model_total={world_model_train_results['L_world_model_total'].numpy():.5f} ("
            f"L_pred={world_model_train_results['L_pred'].numpy():.5f}; "
            f"L_dyn={world_model_train_results['L_dyn'].numpy():.5f}; "
            f"L_rep={world_model_train_results['L_rep'].numpy():.5f})"
        )

        # Train critic and actor.
        if not train_world_model_only:
            forward_train_outs = world_model_train_results["forward_train_outs"]

            # Build critic model first, so we can initialize EMA weights.
            if not dreamer_model.critic.trainable_variables:
                # Forward pass for fast critic.
                dreamer_model.critic(
                    h=forward_train_outs["h_states"],
                    z=forward_train_outs["z_states"],
                    return_distribution=True,
                )
                # Forward pass for EMA-weights critic.
                dreamer_model.critic(
                    h=forward_train_outs["h_states"],
                    z=forward_train_outs["z_states"],
                    return_distribution=False,
                    use_ema=True,
                )
                dreamer_model.critic.init_ema()
                # TEST: OOM
                print("\tafter critic EMA-init:",
                    tf.config.experimental.get_memory_info('GPU:0')['current'])

            actor_critic_train_results = train_actor_and_critic_one_step(
                forward_train_outs=forward_train_outs,
                horizon_H=horizon_H,
                gamma=discount_gamma,
                lambda_=gae_lambda,
                actor_grad_clip=actor_grad_clip,
                critic_grad_clip=critic_grad_clip,
                dreamer_model=dreamer_model,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                entropy_scale=entropy_scale,
                return_normalization_decay=return_normalization_decay,
            )
            # TEST: OOM
            print("\tafter actor/critic update:",
                  tf.config.experimental.get_memory_info('GPU:0')['current'])

            # Summarize actor-critic loss stats.
            L_critic = actor_critic_train_results["L_critic"]
            tf.summary.scalar("L_critic", L_critic)
            L_actor = actor_critic_train_results["L_actor"]
            tf.summary.scalar("L_actor", L_actor)
            tf.summary.scalar("L_actor_action_entropy", actor_critic_train_results["action_entropy"])
            tf.summary.histogram("L_actor_scaled_value_targets_B_H", actor_critic_train_results["scaled_value_targets_B_H"])
            tf.summary.histogram("L_actor_logp_loss_B_H", actor_critic_train_results["logp_loss_B_H"])

            print(
                f"\tL_actor={L_actor.numpy():.5f} L_critic={L_critic.numpy():.5f}"
            )

        sub_iter += 1
        total_train_steps += 1

        if tb_ctx is not None:
            tb_ctx.__exit__(None, None, None)

    # EVALUATION.
    if total_train_steps % evaluation_frequency_main_iters == 0:
        print("\nEVALUATION:")
        with tb_writer.as_default(step=total_train_steps):
            # Dream a trajectory using the samples from the buffer and compare obs,
            # rewards, continues to the actually observed trajectory.
            dreamed_T = horizon_H
            print(f"Dreaming trajectories (H={dreamed_T}) from all 1st timesteps drawn from buffer ...")
            dream_data = dreamer_model.dream_trajectory_with_burn_in(
                observations=sample["obs"][:, :burn_in_T],  # use only first burn_in_T obs
                actions=sample["actions"][:, :burn_in_T + dreamed_T],  # use all actions from 0 to T (no actor)
                initial_h=sample["h_states"],
                timesteps=dreamed_T,  # dream for n timesteps
                use_sampled_actions=True,  # use sampled actions, not the actor
            )
            # TEST: OOM
            print("\tafter eval dream w/ burn-in:",
                  tf.config.experimental.get_memory_info('GPU:0')['current'])

            mse_sampled_vs_dreamed_obs = summarize_dreamed_trajectory_vs_samples(
                dream_data,
                sample,
                batch_size_B=batch_size_B,
                burn_in_T=burn_in_T,
                dreamed_T=dreamed_T,
                dreamer_model=dreamer_model,
            )
            print(f"\tMSE sampled vs dreamed obs (B={batch_size_B} T/H={dreamed_T}): {mse_sampled_vs_dreamed_obs:.6f}")

            # Run n episodes in an actual env and report mean episode returns.
            print(f"Running {evaluation_num_episodes} episodes in env for evaluation ...")
            _, _, _, eval_rewards, _, _ = env_runner_evaluation.sample_episodes(
                num_episodes=evaluation_num_episodes, random_actions=False
            )
            mean_episode_return = np.mean([np.sum(rs) for rs in eval_rewards])
            print(f"\tMean episode return: {mean_episode_return:.4f}")
            tf.summary.scalar("EVAL_mean_episode_return", mean_episode_return)

    # Save the model every N iterations (but not after the very first).
    if iteration != 0 and iteration % model_save_frequency_main_iters == 0:
        dreamer_model.save(f"checkpoints/dreamer_model_{iteration}")
        print("\tafter model save:",
              tf.config.experimental.get_memory_info('GPU:0')['current'])

    total_replayed_steps += replayed_steps
    print(
        f"\treplayed-steps: {total_replayed_steps}; env-steps: {total_env_steps}"
    )
