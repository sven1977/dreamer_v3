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

from models.components.mlp import MLP
from models.components.vector_decoder import VectorDecoder
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
summary_frequency = 5
# Every how many (main) iterations (sample + N train steps) do we save our model?
model_save_frequency = 10

# Set batch size and -length according to [1]:
batch_size_B = 16
batch_length_T = 64
# The number of timesteps we use to "initialize" (burn-in) a dream_trajectory run.
# For this many timesteps, the posterior (actual observation data) will be used
# to compute z, after that, only the prior (dynamics network) will be used.
burn_in_T = 5
horizon_H = 15

# Actor/critic Hyperparameters.
discount_gamma = 0.997
gae_lambda = 0.95

# EnvRunner config (an RLlib algorithm config).
config = (
    AlgorithmConfig()
    .environment("CartPole-v1")
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

# Whether to o nly train the world model (not the critic and actor networks).
train_world_model_only = False
# Our DreamerV3 world model.
from_checkpoint = None
# Uncomment this next line to load from a saved model.
#from_checkpoint = "checkpoints/dreamer_model_0"
if from_checkpoint is not None:
    dreamer_model = tf.keras.models.load_model(from_checkpoint)
else:
    model_dimension = "micro"
    world_model = WorldModel(
        model_dimension=model_dimension,
        action_space=env_runner.env.single_action_space,
        batch_length_T=batch_length_T,
        encoder=MLP(model_dimension=model_dimension),
        decoder=VectorDecoder(
            model_dimension=model_dimension,
            observation_space=env_runner.env.single_observation_space,
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

        # Enable TB summaries this step?
        tb_ctx = None
        if total_train_steps % summary_frequency == 0:
            tb_ctx = tb_writer.as_default(step=total_train_steps)
            tb_ctx.__enter__()

        # Perform one world-model training step.
        world_model_train_results = train_world_model_one_step(
            sample=sample,
            batch_size_B=batch_size_B,
            batch_length_T=batch_length_T,
            grad_clip=world_model_grad_clip,
            dreamer_model=dreamer_model,
            optimizer=world_model_optimizer,
        )
        summarize_forward_train_outs_vs_samples(
            forward_train_outs=world_model_train_results["forward_train_outs"],
            sample=sample,
            batch_size_B=batch_size_B,
            batch_length_T=batch_length_T,
        )
        summarize_world_model_losses(world_model_train_results)

        print(
            f"Iter {iteration}/{sub_iter}) "
            f"L_world_model_total={world_model_train_results['L_total'].numpy()} "
            f"(L_pred={world_model_train_results['L_pred'].numpy()}; "
            f"L_dyn={world_model_train_results['L_dyn'].numpy()}; "
            f"L_rep={world_model_train_results['L_rep'].numpy()})"
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
            )
            # Summarize critic loss stats.
            L_critic = actor_critic_train_results["L_critic"]
            tf.summary.scalar("L_critic", L_critic)

            print(
                f"\tL_critic={L_critic.numpy()} L_actor=TODO"
            )

        # EVALUATION:
        if total_train_steps % summary_frequency == 0:
            # Dream a trajectory using the samples from the buffer and compare obs,
            # rewards, continues to the actually observed trajectory.
            dreamed_T = horizon_H
            dream_data = dreamer_model.dream_trajectory_with_burn_in(
                observations=sample["obs"][:, :burn_in_T],  # use only first burn_in_T obs
                actions=sample["actions"][:, :burn_in_T + horizon_H],  # use all actions from 0 to T (no actor)
                initial_h=sample["h_states"],
                timesteps=dreamed_T,  # dream for n timesteps
                use_sampled_actions=True,  # use sampled actions,not the actor
            )

            summarize_dreamed_trajectory_vs_samples(
                dream_data,
                sample,
                batch_size_B=batch_size_B,
                burn_in_T=burn_in_T,
                dreamed_T=dreamed_T,
                dreamer_model=dreamer_model,
            )

        sub_iter += 1
        total_train_steps += 1

        if tb_ctx is not None:
            tb_ctx.__exit__(None, None, None)

    # Save the model every N iterations (but not after the very first).
    if iteration != 0 and iteration % model_save_frequency == 0:
        dreamer_model.save(f"checkpoints/dreamer_model_{iteration}")

    total_replayed_steps += replayed_steps
    print(
        f"\treplayed-steps: {total_replayed_steps}; env-steps: {total_env_steps}"
    )
