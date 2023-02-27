"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf

[2] Mastering Atari with Discrete World Models - 2021
D. Hafner, T. Lillicrap, M. Norouzi, J. Ba
https://arxiv.org/pdf/2010.02193.pdf
"""

import gc
import os
import yaml

import numpy as np
import tree  # pip install dm_tree
import tensorflow as tf

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

from models.components.cnn_atari import CNNAtari
from models.components.conv_transpose_atari import ConvTransposeAtari
from models.components.mlp import MLP
from models.components.vector_decoder import VectorDecoder
from models.dreamer_model import DreamerModel
from models.world_model import WorldModel
from training.train_one_step import (
    train_actor_and_critic_one_step,
    train_world_model_one_step,
)
from utils.env_runner_v2 import EnvRunnerV2
from utils.episode_replay_buffer import EpisodeReplayBuffer
from utils.cartpole_debug import CartPoleDebug  # import registers `CartPoleDebug-v0`
from utils.tensorboard import (
    summarize_dreamed_trajectory_vs_samples,
    summarize_forward_train_outs_vs_samples,
    reconstruct_obs_from_h_and_z,
    summarize_world_model_losses,
)

with open("atari_pong.yaml", "r") as f:
    options = yaml.safe_load(f)
    assert len(options) == 1
    options = next(iter(options.values()))

# Create the checkpoint path, if it doesn't exist yet.
os.makedirs("checkpoints", exist_ok=True)
# Create the tensorboard summary data dir.
os.makedirs("tensorboard", exist_ok=True)
tb_writer = tf.summary.create_file_writer("tensorboard")
# Every how many training steps do we write data to TB?
summary_frequency_train_steps = options["summary_frequency_train_steps"]
# Every how many main iterations do we evaluate?
evaluation_frequency_main_iters = options["evaluation_frequency_main_iters"]
evaluation_num_episodes = options["evaluation_num_episodes"]
# Every how many (main) iterations (sample + N train steps) do we save our model?
model_save_frequency_main_iters = options["model_save_frequency_main_iters"]

# Set batch size and -length according to [1]:
batch_size_B = 16
batch_length_T = 64
# The number of timesteps we use to "initialize" (burn-in) a dream_trajectory run.
# For this many timesteps, the posterior (actual observation data) will be used
# to compute z, after that, only the prior (dynamics network) will be used.
burn_in_T = 5
horizon_H = 15

# Whether to symlog the observations or not.
symlog_obs = not options.get("is_atari", False)

# Actor/critic hyperparameters.
discount_gamma = options.get("discount_gamma", 0.997)  # [1] eq. 7.
gae_lambda = options.get("gae_lambda", 0.95)  # [1] eq. 7.
entropy_scale = 3e-4  # [1] eq. 11.
return_normalization_decay = 0.99  # [1] eq. 11 and 12.


# EnvRunner config (an RLlib algorithm config).
config = (
    AlgorithmConfig()
    .environment(options["env"], env_config={
        # [2]: "We follow the evaluation protocol of Machado et al. (2018) with 200M
        # environment steps, action repeat of 4, a time limit of 108,000 steps per
        # episode that correspond to 30 minutes of game play, no access to life
        # information, full action space, and sticky actions. Because the world model
        # integrates information over time, DreamerV2 does not use frame stacking.
        # The experiments use a single-task setup where a separate agent is trained
        # for each game. Moreover, each agent uses only a single environment instance.
        "repeat_action_probability": 0.25,  # "sticky actions"
        "full_action_space": True,  # "full action space"
        "frameskip": 1,  # already done by MaxAndSkip wrapper: "action repeat" == 4
    } if options["is_atari"] else options.get("env_config", {}))
    .rollouts(
        num_envs_per_worker=1,
        rollout_fragment_length=batch_length_T,
    )
)
# The vectorized gymnasium EnvRunner to collect samples of shape (B, T, ...).
env_runner = EnvRunnerV2(model=None, config=config)
env_runner_evaluation = EnvRunnerV2(model=None, config=config)

# Whether to o nly train the world model (not the critic and actor networks).
train_critic = options.get("train_critic", True)
train_actor = options.get("train_actor", True)
# Cannot train actor w/o critic.
assert not (train_actor and not train_critic)

# Our DreamerV3 world model.
from_checkpoint = None
# Uncomment this next line to load from a saved model.
#from_checkpoint = "checkpoints/dreamer_model_0"
if from_checkpoint is not None:
    dreamer_model = tf.keras.models.load_model(from_checkpoint)
else:
    model_dimension = options["model_dimension"]
    world_model = WorldModel(
        model_dimension=model_dimension,
        action_space=env_runner.env.single_action_space,
        batch_length_T=batch_length_T,
        num_gru_units=options.get("num_gru_units"),
        encoder=CNNAtari(model_dimension=model_dimension) if options["is_atari"] else MLP(model_dimension=model_dimension),
        decoder=ConvTransposeAtari(
            model_dimension=model_dimension,
            gray_scaled=True,
        ) if options["is_atari"] else VectorDecoder(
            model_dimension=model_dimension,
            observation_space=env_runner.env.single_observation_space,
        ),
        symlog_obs=symlog_obs,
    )
    dreamer_model = DreamerModel(
        model_dimension=model_dimension,
        action_space=env_runner.env.single_action_space,
        world_model=world_model,
    )

#TEST: OOM
#print("current mem:", tf.config.experimental.get_memory_info('GPU:0')['current'])

# TODO: ugly hack (resulting from the insane fact that you cannot know
#  an env's spaces prior to actually constructing an instance of it) :(
env_runner.model = dreamer_model
env_runner_evaluation.model = dreamer_model

# The replay buffer for storing actual env samples.
buffer = EpisodeReplayBuffer(capacity=int(1e6))
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
training_ratio = options["training_ratio"]

total_env_steps = 0
total_replayed_steps = 0
total_train_steps = 0

for iteration in range(1000):
    print(f"Main iteration {iteration}")
    # Push enough samples into buffer initially before we start training.
    env_steps = env_steps_last_sample = 0
    #TEST: Put only a single row in the buffer and try to memorize it.
    #env_steps_last_sample = 64
    #while iteration == 0:
    #END TEST
    while True:
        # Sample one round.
        done_episodes, ongoing_episodes = env_runner.sample(random_actions=False)

        # We took B x T env steps.
        env_steps_last_sample = sum(
            len(eps) for eps in done_episodes + ongoing_episodes
        )
        env_steps += env_steps_last_sample

        buffer.add(episodes=done_episodes + ongoing_episodes)
        episodes_in_buffer = buffer.get_num_episodes()
        ts_in_buffer = buffer.get_num_timesteps()
        print(
            f"\tsampled env-steps={env_steps}; "
            f"buffer size (ts)={ts_in_buffer}; "
            f"buffer size (episodes)={episodes_in_buffer}"
        )

        if (
            # Got to have more timesteps than warm up setting.
            ts_in_buffer > warm_up_timesteps
            # and more timesteps than BxT.
            and ts_in_buffer >= batch_size_B * batch_length_T
            # But also at least as many episodes as the batch size B.
            and episodes_in_buffer >= batch_size_B
        ):
            break

    # Summarize actual environment interaction data.
    metrics = env_runner.get_metrics()
    with tb_writer.as_default(step=total_env_steps):
        # Summarize buffer length.
        tf.summary.scalar("buffer_size_num_episodes", episodes_in_buffer)
        tf.summary.scalar("buffer_size_timesteps", ts_in_buffer)
        # Summarize episode returns.
        if metrics.get("episode_returns"):
            episode_return_mean = np.mean(metrics["episode_returns"])
            tf.summary.scalar("ENV_episode_return_mean", episode_return_mean)
        # Summarize actions taken.
        actions = np.concatenate(
            [eps.actions for eps in done_episodes + ongoing_episodes],
            axis=0,
        )
        tf.summary.histogram("ENV_actions_taken", actions)

    total_env_steps += env_steps

    print(
        f"\treplayed-steps learned: {total_replayed_steps}; "
        f"env-steps taken: {total_env_steps}"
    )

    replayed_steps = 0

    #TEST: re-use same sample.
    #sample = buffer.sample(num_items=batch_size_B)
    #sample = tree.map_structure(lambda v: tf.convert_to_tensor(v), sample)
    #END TEST

    print()

    sub_iter = 0
    while replayed_steps / env_steps_last_sample < training_ratio:
        print(f"\tSub-iteration {iteration}/{sub_iter})")

        # Enable TB summaries this step?
        tb_ctx = None
        if total_train_steps % summary_frequency_train_steps == 0:
            tb_ctx = tb_writer.as_default(step=total_env_steps)
            tb_ctx.__enter__()

        # Draw a new sample from the replay buffer.
        sample = buffer.sample(batch_size_B=batch_size_B, batch_length_T=batch_length_T)
        replayed_steps += batch_size_B * batch_length_T

        # Convert samples (numpy) to tensors.
        sample = tree.map_structure(lambda v: tf.convert_to_tensor(v), sample)

        # Perform one world-model training step.
        world_model_train_results = train_world_model_one_step(
            sample=sample,
            batch_size_B=tf.convert_to_tensor(batch_size_B),
            batch_length_T=tf.convert_to_tensor(batch_length_T),
            grad_clip=tf.convert_to_tensor(world_model_grad_clip),
            world_model=world_model,
            optimizer=world_model_optimizer,
        )
        forward_train_outs = world_model_train_results["forward_train_outs"]

        # Update h_states in buffer after the world model (sequential model)
        # forward pass.
        h_BxT = forward_train_outs["h_states_BxT"]
        h_B_t2_to_Tp1 = tf.concat([tf.reshape(
            h_BxT,
            shape=(batch_size_B, batch_length_T) + h_BxT.shape[1:],
        )[:, 1:], tf.expand_dims(forward_train_outs["h_B_Tp1"], axis=1)], axis=1)
        buffer.update_h_states(h_B_t2_to_Tp1.numpy(), sample["indices"].numpy())

        # Summarize world model.
        if iteration == 0 and sub_iter == 0:
            # Dummy forward pass to be able to produce summary.
            world_model(
                sample["obs"][:, 0],
                sample["actions"][:, 0],
                sample["h_states"][:, 0],
            )
            world_model.summary()

        if total_train_steps % summary_frequency_train_steps == 0:
            summarize_forward_train_outs_vs_samples(
                forward_train_outs=forward_train_outs,
                sample=sample,
                batch_size_B=batch_size_B,
                batch_length_T=batch_length_T,
                symlog_obs=symlog_obs,
            )
            summarize_world_model_losses(world_model_train_results)

        print(
            f"\t\tL_world_model_total={world_model_train_results['L_world_model_total'].numpy():.5f} ("
            f"L_pred={world_model_train_results['L_pred'].numpy():.5f}; "
            f"L_dyn={world_model_train_results['L_dyn'].numpy():.5f}; "
            f"L_rep={world_model_train_results['L_rep'].numpy():.5f})"
        )

        # Train critic and actor.
        if train_critic:
            # Build critic model first, so we can initialize EMA weights.
            if not dreamer_model.critic.trainable_variables:
                # Forward pass for fast critic.
                dreamer_model.critic(
                    h=forward_train_outs["h_states_BxT"],
                    z=forward_train_outs["z_states_BxT"],
                    return_distribution=True,
                )
                # Forward pass for EMA-weights critic.
                dreamer_model.critic(
                    h=forward_train_outs["h_states_BxT"],
                    z=forward_train_outs["z_states_BxT"],
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
                entropy_scale=entropy_scale,
                return_normalization_decay=return_normalization_decay,
                train_actor=train_actor,
            )
            L_critic = actor_critic_train_results["L_critic"]
            if train_actor:
                L_actor = actor_critic_train_results["L_actor"]

            # Summarize actor/critic models.
            if iteration == 0 and sub_iter == 0:
                # Dummy forward pass to be able to produce summary.
                if train_actor:
                    dreamer_model.actor(
                        actor_critic_train_results["dream_data"]["h_states_t1_to_Hp1"][:, 0],
                        actor_critic_train_results["dream_data"]["z_states_prior_t1_to_H"][:, 0],
                    )
                    dreamer_model.actor.summary()
                dreamer_model.critic(
                    actor_critic_train_results["dream_data"]["h_states_t1_to_Hp1"][:, 0],
                    actor_critic_train_results["dream_data"]["z_states_prior_t1_to_H"][:, 0],
                )
                dreamer_model.critic.summary()

            # Analyze generated dream data for its suitability in training the critic
            # and actors.
            if total_train_steps % summary_frequency_train_steps == 0:
                #TODO: put all of this block into tensorboard module.
                #TODO: Make this work with any renderable env.
                if env_runner.config.env in ["CartPoleDebug-v0", "CartPole-v1", "FrozenLake-v1"]:
                    from utils.cartpole_debug import create_cartpole_dream_image, create_frozenlake_dream_image
                    dream_data = actor_critic_train_results["dream_data"]
                    dreamed_obs_B_H = reconstruct_obs_from_h_and_z(
                        h_t1_to_Tp1=dream_data["h_states_t1_to_Hp1"],
                        z_t1_to_T=dream_data["z_states_prior_t1_to_H"],
                        dreamer_model=dreamer_model,
                        obs_dims_shape=sample["obs"].shape[2:],
                    )
                    # Take 0th dreamed trajectory and produce series of images.
                    for t in range(len(dreamed_obs_B_H[0])):
                        func = create_cartpole_dream_image if env_runner.config.env.startswith("CartPole") else create_frozenlake_dream_image
                        img = func(
                            dreamed_obs=dreamed_obs_B_H[0][t],
                            dreamed_V=dream_data["values_dreamed_t1_to_Hp1"][0][t],
                            dreamed_a=dream_data["actions_dreamed_t1_to_H"][0][t],
                            dreamed_r_tp1=dream_data["rewards_dreamed_t1_to_Hp1"][0][t+1],
                            dreamed_c_tp1=dream_data["continues_dreamed_t1_to_Hp1"][0][t+1],
                            value_target=actor_critic_train_results["value_targets_B_H"][0][t],
                            initial_h=dream_data["h_states_t1_to_Hp1"][0][t],
                            as_tensor=True,
                        )
                        tf.summary.image(
                            f"dreamed_trajectories_for_critic_actor_learning[B=0,T={t}]",
                            tf.expand_dims(img, axis=0),
                        )

                # Summarize actor-critic loss stats.
                tf.summary.scalar("L_critic", L_critic)
                tf.summary.histogram("L_critic_value_targets", actor_critic_train_results["value_targets_B_H"])
                tf.summary.scalar("L_critic_neg_logp_target", actor_critic_train_results["L_critic_neg_logp_target"])
                tf.summary.histogram("L_critic_neg_logp_target_B_H", actor_critic_train_results["L_critic_neg_logp_target_B_H"])
                tf.summary.scalar("L_critic_ema_regularization", actor_critic_train_results["L_critic_ema_regularization"])
                tf.summary.histogram("L_critic_ema_regularization_B_H", actor_critic_train_results["L_critic_ema_regularization_B_H"])

                if train_actor:
                    tf.summary.scalar("L_actor", L_actor)
                    tf.summary.scalar("L_actor_action_entropy", actor_critic_train_results["action_entropy"])
                    tf.summary.scalar("L_actor_reinforce_term", actor_critic_train_results["L_actor_reinforce_term"])
                    tf.summary.scalar("L_actor_action_entropy_term", actor_critic_train_results["L_actor_action_entropy_term"])
                    tf.summary.histogram("L_actor_scaled_value_targets_B_H", actor_critic_train_results["scaled_value_targets_B_H"])
                    tf.summary.histogram("L_actor_logp_loss_B_H", actor_critic_train_results["logp_loss_B_H"])
                    tf.summary.scalar("L_actor_ema_value_target_pct95", actor_critic_train_results["L_actor_ema_value_target_pct95"])
                    tf.summary.scalar("L_actor_ema_value_target_pct5", actor_critic_train_results["L_actor_ema_value_target_pct5"])

            print(
                f"\t\tL_actor={L_actor.numpy() if train_actor else 0.0:.5f} L_critic={L_critic.numpy():.5f}"
            )

        sub_iter += 1
        total_train_steps += 1

        if tb_ctx is not None:
            tb_ctx.__exit__(None, None, None)

    total_replayed_steps += replayed_steps

    # EVALUATION.
    if total_train_steps % evaluation_frequency_main_iters == 0:
        print("\nEVALUATION:")
        with tb_writer.as_default(step=total_env_steps):
            # Dream a trajectory using the samples from the buffer and compare obs,
            # rewards, continues to the actually observed trajectory.
            dreamed_T = horizon_H
            print(f"\tDreaming trajectories (H={dreamed_T}) from all 1st timesteps drawn from buffer ...")
            dream_data = dreamer_model.dream_trajectory_with_burn_in(
                observations=sample["obs"][:, :burn_in_T],  # use only first burn_in_T obs
                actions=sample["actions"][:, :burn_in_T + dreamed_T],  # use all actions from 0 to T (no actor)
                initial_h=sample["h_states"][:, 0],  # use initial T=0 h-states
                timesteps=dreamed_T,  # dream for n timesteps
                use_sampled_actions=True,  # use sampled actions, not the actor
            )

            mse_sampled_vs_dreamed_obs = summarize_dreamed_trajectory_vs_samples(
                dream_data,
                sample,
                batch_size_B=batch_size_B,
                burn_in_T=burn_in_T,
                dreamed_T=dreamed_T,
                dreamer_model=dreamer_model,
                symlog_obs=symlog_obs,
            )
            print(f"\tMSE sampled vs dreamed obs (B={batch_size_B} T/H={dreamed_T}): {mse_sampled_vs_dreamed_obs:.6f}")

            # Run n episodes in an actual env and report mean episode returns.
            print(f"Running {evaluation_num_episodes} episodes in env for evaluation ...")
            episodes = env_runner_evaluation.sample_episodes(
                num_episodes=evaluation_num_episodes, random_actions=False
            )
            mean_episode_len = np.mean([len(eps) for eps in episodes])
            mean_episode_return = np.mean([eps.get_return() for eps in episodes])
            print(f"\tMean episode return: {mean_episode_return:.4f}; mean len: {mean_episode_len:.1f}")
            tf.summary.scalar("EVAL_mean_episode_return", mean_episode_return)
            tf.summary.scalar("EVAL_mean_episode_length", mean_episode_len)

    # Save the model every N iterations (but not after the very first).
    if iteration != 0 and iteration % model_save_frequency_main_iters == 0:
        dreamer_model.save(f"checkpoints/dreamer_model_{iteration}")

    #TEST: try trick from https://medium.com/dive-into-ml-ai/dealing-with-memory-leak-issue-in-keras-model-training-e703907a6501
    gc.collect()
    #tf.keras.backend.clear_session()

    try:
        gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
        print(f"\nMEM (GPU) consumption: {gpu_memory['current']}")
        with tb_writer.as_default(step=total_env_steps):
            tf.summary.scalar("MEM_gpu_memory_used", gpu_memory['current'])
            tf.summary.scalar("MEM_gpu_memory_peak", gpu_memory['peak'])
    except ValueError:
        pass

    # Main iteration done.
    print("\n")
