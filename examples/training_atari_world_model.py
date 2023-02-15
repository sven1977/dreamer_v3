"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf

[2] Mastering Atari with Discrete World Models - 2021
D. Hafner, T. Lillicrap, M. Norouzi, J. Ba
https://arxiv.org/pdf/2010.02193.pdf
"""

import os
import tree  # pip install dm_tree
import tensorflow as tf

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

from models.components.cnn_atari import CNNAtari
from models.components.conv_transpose_atari import ConvTransposeAtari
from models.dreamer_model import DreamerModel
from models.world_model import WorldModel
from training.train_one_step import train_one_step
from utils.env_runner import EnvRunner
from utils.continuous_episode_replay_buffer import ContinuousEpisodeReplayBuffer
from utils.tensorboard import (
    summarize_dreamed_trajectory_vs_samples,
)

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
    num_data_tracks=1,
)
# Timesteps to put into the buffer before the first learning step.
warm_up_timesteps = 0

# Use an Adam optimizer.
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-8)

# World model grad clipping according to [1].
grad_clip = 1000.0

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
                L_total, L_pred, L_dyn, L_rep = train_one_step(
                    sample, total_train_steps_tensor
                )

                # EVALUATION:
                # Dream a trajectory using the samples from the buffer and compare obs,
                # rewards, continues to the actually observed trajectory.
                dreamed_T = batch_length_T - burn_in_T
                dream_data = dreamer_model.dream_trajectory(
                    observations=sample["obs"][:, :burn_in_T],  # use only first burn_in_T obs
                    actions=sample["actions"],  # use all actions from 0 to T (no actor)
                    initial_h=sample["h_states"],
                    timesteps=dreamed_T,  # dream for T-burn_in_T timesteps
                    use_sampled_actions=True,  # use all actions from 0 to T (no actor)
                )

                summarize_dreamed_trajectory_vs_samples(
                    dream_data,
                    sample,
                    batch_size_B=batch_size_B,
                    burn_in_T=burn_in_T,
                    dreamed_T=dreamed_T,
                    dreamer_model=dreamer_model,
                    step=total_train_steps_tensor,
                )

        else:
            L_world_model_total, L_pred, L_dyn, L_rep = train_one_step(sample, total_train_steps_tensor)

        print(
            f"Iter {iteration}/{sub_iter}) L_world_model_total={L_world_model_total.numpy()} "
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
