"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf

[2] Mastering Atari with Discrete World Models - 2021
D. Hafner, T. Lillicrap, M. Norouzi, J. Ba
https://arxiv.org/pdf/2010.02193.pdf
"""

import argparse
import gc
import os
import yaml
from pprint import pprint

import numpy as np
import tree  # pip install dm_tree
import tensorflow as tf
from tensorboardX import SummaryWriter

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

import examples.debug_img_env  # to trigger DebugImgEnv import and registration
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
from utils.episode import Episode
from utils.cartpole_debug import CartPoleDebug  # import registers `CartPoleDebug-v0`
from utils.tensorboard import (
    summarize_actor_losses,
    summarize_critic_losses,
    summarize_dreamed_eval_trajectory_vs_samples,
    summarize_forward_train_outs_vs_samples,
    reconstruct_obs_from_h_and_z,
    summarize_world_model_losses,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    "-c",
    type=str,
    default="atari_pong.yaml",
    help="The config yaml file for the experiment.",
)
args = parser.parse_args()

print(f"Trying to open config file {args.config} ...")
with open(args.config, "r") as f:
    config = yaml.safe_load(f)
print(f"Running with the following config:")
pprint(config)
assert len(config) == 1, "Only one experiment allowed in config yaml!"
config = next(iter(config.values()))

# Create the checkpoint path, if it doesn't exist yet.
os.makedirs("checkpoints", exist_ok=True)
# Create the tensorboard summary data dir.
os.makedirs("tensorboard", exist_ok=True)
tbx_writer = SummaryWriter("tensorboard")
# How many iterations do we pre-train?
num_pretrain_iterations = config.get("num_pretrain_iterations", 0)
# Every how many training steps do we write data to TB?
summary_frequency_train_steps = config.get("summary_frequency_train_steps", 20)
# Every how many training steps do we collect garbage
gc_frequency_train_steps = config.get("gc_frequency_train_steps", 100)
# Every how many main iterations do we evaluate?
evaluation_frequency_main_iters = config.get("evaluation_frequency_main_iters", 0)
evaluation_num_episodes = config["evaluation_num_episodes"]
# Every how many (main) iterations (sample + N train steps) do we save our model?
model_save_frequency_main_iters = config.get("model_save_frequency_main_iters", 0)

# Set batch size and -length according to [1]:
batch_size_B = config.get("batch_size_B", 16)
batch_length_T = config.get("batch_length_T", 64)
# The number of timesteps we use to "initialize" (burn-in) a dream_trajectory run.
# For this many timesteps, the posterior (actual observation data) will be used
# to compute z, after that, only the prior (dynamics network) will be used.
burn_in_T = 5
horizon_H = config.get("horizon_H", 15)

# Whether to symlog the observations or not.
symlog_obs = not config.get("is_atari", False)

# Actor/critic hyperparameters.
discount_gamma = config.get("discount_gamma", 0.997)  # [1] eq. 7.
gae_lambda = config.get("gae_lambda", 0.95)  # [1] eq. 7.
entropy_scale = 3e-4  # [1] eq. 11.
return_normalization_decay = 0.99  # [1] eq. 11 and 12.


# EnvRunner config (an RLlib algorithm config).
algo_config = (
    AlgorithmConfig()
    .environment(config["env"], env_config={
        # [2]: "We follow the evaluation protocol of Machado et al. (2018) with 200M
        # environment steps, action repeat of 4, a time limit of 108,000 steps per
        # episode that correspond to 30 minutes of game play, no access to life
        # information, full action space, and sticky actions. Because the world model
        # integrates information over time, DreamerV2 does not use frame stacking.
        # The experiments use a single-task setup where a separate agent is trained
        # for each game. Moreover, each agent uses only a single environment instance.
        "repeat_action_probability": 0.0,#25,  # "sticky actions" but not according to Dani's 100k configs
        "full_action_space": False,#True,  # "full action space" but not according to Dani's 100k configs
        "frameskip": 1,  # already done by MaxAndSkip wrapper: "action repeat" == 4
    } if config["is_atari"] else config.get("env_config", {}))
    .rollouts(
        num_envs_per_worker=1,
        rollout_fragment_length=1,#TESTbatch_length_T,
    )
)
# The vectorized gymnasium EnvRunner to collect samples of shape (B, T, ...).
env_runner = EnvRunnerV2(model=None, config=algo_config)
env_runner_evaluation = EnvRunnerV2(model=None, config=algo_config)

# Whether to o nly train the world model (not the critic and actor networks).
train_critic = config.get("train_critic", True)
train_actor = config.get("train_actor", True)
# Cannot train actor w/o critic.
assert not (train_actor and not train_critic)

# Our DreamerV3 world model.
from_checkpoint = None
# Uncomment this next line to load from a saved model.
#from_checkpoint = "checkpoints/dreamer_model_0"
if from_checkpoint is not None:
    dreamer_model = tf.keras.models.load_model(from_checkpoint)
else:
    model_dimension = config["model_dimension"]
    img_space = config["is_atari"] or config["env"] == "DebugImgEnv-v0"
    world_model = WorldModel(
        model_dimension=model_dimension,
        action_space=env_runner.env.single_action_space,
        batch_length_T=batch_length_T,
        num_gru_units=config.get("num_gru_units"),
        encoder=CNNAtari(model_dimension=model_dimension) if img_space else MLP(model_dimension=model_dimension),
        decoder=ConvTransposeAtari(
            model_dimension=model_dimension,
            gray_scaled=False,
        ) if img_space else VectorDecoder(
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
training_ratio = config["training_ratio"]

total_env_steps = 0
total_replayed_steps = 0
total_train_steps = 0

if num_pretrain_iterations > 0:
    # 1) Initialize dataset
    import d3rlpy
    if config["env"] == "CartPole-v0":
        dataset, _ = d3rlpy.datasets.get_cartpole()
    elif config["env"] == "Pendulum-v0":
        dataset, _ = d3rlpy.datasets.get_pendulum()
    elif config["is_atari"] == True:
        dataset, _ = d3rlpy.datasets.get_atari(config["offline_dataset"])
    else:
        raise ValueError("Unknown offline environment.")
    
    print("Loading episodes from d3rlpy to dreamer_v3")
    episodes = []
    for eps in dataset:
        eps_ = Episode()
        eps_.observations = np.concatenate(
            [eps.observations, np.array([eps.observations[-1]])], axis=0
        )
        eps_.actions = eps.actions
        eps_.rewards = eps.rewards
        eps_.is_terminated = eps.terminal == 1.0
        initial_h = dreamer_model._get_initial_h(1).numpy().astype(np.float32)
        eps_.h_states = np.repeat(initial_h, len(eps_.rewards), axis = 0)
        eps_.validate()
        buffer.add(eps_)

    assert buffer.get_num_episodes() == len(dataset)
    assert buffer.get_num_timesteps() == dataset.rewards.shape[0]

    print("Loaded d3rlpy dataset into replay buffer:")
    print(f"{dataset.size()} episodes {dataset.rewards.shape[0]} steps")
    print("Pretraining world model")

    # 2) Pretrain world model on offline data for n iterations.
    for iteration in range(num_pretrain_iterations):
        print(f"Offline training iteration {iteration}")

        sample = buffer.sample(batch_size_B=batch_size_B, batch_length_T=batch_length_T)
        total_replayed_steps += batch_size_B * batch_length_T

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
        #h_BxT = forward_train_outs["h_states_BxT"]
        #h_B_t2_to_Tp1 = tf.concat([tf.reshape(
        #    h_BxT,
        #    shape=(batch_size_B, batch_length_T) + h_BxT.shape[1:],
        #)[:, 1:], tf.expand_dims(h_states_training, axis=1)], axis=1)
        #buffer.update_h_states(h_B_t2_to_Tp1.numpy(), sample["indices"].numpy())

        # Summarize world model.
        if iteration == 0:
            # Dummy forward pass to be able to produce summary.
            world_model(
                sample["obs"][:, 0],
                sample["actions"][:, 0],
                sample["h_states"][:, 0],
            )
            world_model.summary()

        if summary_frequency_train_steps and iteration % summary_frequency_train_steps:
            summarize_forward_train_outs_vs_samples(
                tbx_writer=tbx_writer,
                step=total_env_steps,
                forward_train_outs=forward_train_outs,
                sample=sample,
                batch_size_B=batch_size_B,
                batch_length_T=batch_length_T,
                symlog_obs=symlog_obs,
            )
            summarize_world_model_losses(
                tbx_writer=tbx_writer,
                step=total_env_steps,
                world_model_train_results=world_model_train_results,
            )

        print(
            f"\t\tL_world_model_total={world_model_train_results['L_world_model_total'].numpy():.5f} ("
            f"L_pred={world_model_train_results['L_pred'].numpy():.5f} ("
            f"decoder/obs={world_model_train_results['L_decoder'].numpy()} "
            f"reward(two-hot)={world_model_train_results['L_reward_two_hot'].numpy()} "
            f"cont={world_model_train_results['L_continue'].numpy()}"
            "); "
            f"L_dyn={world_model_train_results['L_dyn'].numpy():.5f}; "
            f"L_rep={world_model_train_results['L_rep'].numpy():.5f})"
        )

    print()
    print("Pretraining offline completed ... switching to online training and evaluation")

for iteration in range(1000000):
    print(f"Online training main iteration {iteration}")
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

        # Add ongoing and finished episodes into buffer. The buffer will automatically
        # take care of properly concatenating (by episode IDs) the different chunks of
        # the same episodes, even if they come in in separate `add()` calls.
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
            ## But also at least as many episodes as the batch size B.
            ## Actually: This is not useful for longer episode envs, such as Atari.
            ## Too much initial data goes into the buffer, then.
            #and episodes_in_buffer >= batch_size_B
        ):
            break

    # Summarize actual environment interaction data.
    metrics = env_runner.get_metrics()
    assert not env_runner.get_metrics()  # make sure purges env-runner buffers

    # Summarize buffer length.
    tbx_writer.add_scalar(
        "buffer_size_num_episodes", episodes_in_buffer, global_step=total_env_steps
    )
    tbx_writer.add_scalar(
        "buffer_size_timesteps", ts_in_buffer, global_step=total_env_steps
    )
    # Summarize episode returns.
    if metrics.get("episode_returns"):
        episode_return_mean = np.mean(metrics["episode_returns"])
        print(f"\tFinished sampling episodes R={list(metrics['episode_returns'])}")
        tbx_writer.add_scalar(
            "ENV_episode_return_mean", episode_return_mean, global_step=total_env_steps
        )

    # Summarize actions taken.
    actions = np.concatenate(
        [eps.actions for eps in done_episodes + ongoing_episodes],
        axis=0,
    )
    tbx_writer.add_histogram(
        "ENV_actions_taken", actions, global_step=total_env_steps
    )

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

        # Draw a new sample from the replay buffer.
        sample = buffer.sample(batch_size_B=batch_size_B, batch_length_T=batch_length_T)
        replayed_steps += batch_size_B * batch_length_T

        # Convert samples (numpy) to tensors.
        sample = tree.map_structure(lambda v: tf.convert_to_tensor(v), sample)
        # Do some other conversions.
        sample["is_first"] = tf.cast(sample["is_first"], tf.float32)
        sample["is_last"] = tf.cast(sample["is_last"], tf.float32)
        sample["is_terminated"] = tf.cast(sample["is_terminated"], tf.float32)
        sample["actions_one_hot"] = tf.one_hot(
            sample["actions"], depth=env_runner.env.single_action_space.n
        )

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
        #h_BxT = forward_train_outs["h_states_BxT"]
        #h_B_t2_to_Tp1 = tf.concat([tf.reshape(
        #    h_BxT,
        #    shape=(batch_size_B, batch_length_T) + h_BxT.shape[1:],
        #)[:, 1:], tf.expand_dims(h_states_training, axis=1)], axis=1)
        #buffer.update_h_states(h_B_t2_to_Tp1.numpy(), sample["indices"].numpy())

        # Summarize world model.
        if iteration == 0 and sub_iter == 0 and num_pretrain_iterations == 0:
            # Dummy forward pass to be able to produce summary.
            world_model(
                {
                    "h": forward_train_outs["h_states_BxT"][:batch_size_B],
                    "z": forward_train_outs["z_states_BxT"][:batch_size_B],
                    "a_one_hot": sample["actions_one_hot"][:, 0],
                },
                sample["obs"][:, 0],
                is_first=sample["is_first"][:, 0],
            )
            world_model.summary()

        if summary_frequency_train_steps and (
                total_train_steps % summary_frequency_train_steps == 0
        ):
            summarize_forward_train_outs_vs_samples(
                tbx_writer=tbx_writer,
                step=total_env_steps,
                forward_train_outs=forward_train_outs,
                sample=sample,
                batch_size_B=batch_size_B,
                batch_length_T=batch_length_T,
                symlog_obs=symlog_obs,
            )
            summarize_world_model_losses(
                tbx_writer=tbx_writer,
                step=total_env_steps,
                world_model_train_results=world_model_train_results,
            )

        print(
            f"\t\tL_world_model_total={world_model_train_results['L_world_model_total'].numpy():.5f} ("
            f"L_pred={world_model_train_results['L_pred'].numpy():.5f} ("
            f"decoder/obs={world_model_train_results['L_decoder'].numpy()} "
            f"reward(two-hot)={world_model_train_results['L_reward_two_hot'].numpy()} "
            f"cont={world_model_train_results['L_continue'].numpy()}"
            "); "
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
                    return_logits=True,
                )
                # Forward pass for EMA-weights critic.
                dreamer_model.critic(
                    h=forward_train_outs["h_states_BxT"],
                    z=forward_train_outs["z_states_BxT"],
                    return_logits=False,
                    use_ema=True,
                )
                dreamer_model.critic.init_ema()

            actor_critic_train_results = train_actor_and_critic_one_step(
                forward_train_outs=forward_train_outs,
                is_terminated=tf.reshape(sample["is_terminated"], [-1]),
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
            dream_data = actor_critic_train_results["dream_data"]

            # Summarize actor/critic models.
            if iteration == 0 and sub_iter == 0:
                # Dummy forward pass to be able to produce summary.
                if train_actor:
                    dreamer_model.actor(
                        dream_data["h_states_t0_to_H_B"][0],
                        dream_data["z_states_prior_t0_to_H_B"][0],
                    )
                    dreamer_model.actor.summary()
                dreamer_model.critic(
                    dream_data["h_states_t0_to_H_B"][0],
                    dream_data["z_states_prior_t0_to_H_B"][0],
                )
                dreamer_model.critic.summary()

            # Analyze generated dream data for its suitability in training the critic
            # and actors.
            if summary_frequency_train_steps and (
                total_train_steps % summary_frequency_train_steps == 0
            ):
                #TODO: put all of this block into tensorboard module.
                #TODO: Make this work with any renderable env.
                if env_runner.config.env in ["CartPoleDebug-v0", "CartPole-v1", "FrozenLake-v1"]:
                    from utils.cartpole_debug import create_cartpole_dream_image, create_frozenlake_dream_image
                    dreamed_obs_H_B = reconstruct_obs_from_h_and_z(
                        h_t0_to_H=dream_data["h_states_t0_to_H_B"],
                        z_t0_to_H=dream_data["z_states_prior_t0_to_H_B"],
                        dreamer_model=dreamer_model,
                        obs_dims_shape=sample["obs"].shape[2:],
                    )
                    # Take 0th dreamed trajectory and produce series of images.
                    for t in range(len(dreamed_obs_H_B) - 1):
                        func = create_cartpole_dream_image if env_runner.config.env.startswith("CartPole") else create_frozenlake_dream_image
                        img = func(
                            dreamed_obs=dreamed_obs_H_B[t][0],
                            dreamed_V=dream_data["values_dreamed_t0_to_H_B"][t][0],
                            dreamed_a=dream_data["actions_dreamed_t0_to_H_B"][t][0],
                            dreamed_r_tp1=dream_data["rewards_dreamed_t0_to_H_B"][t+1][0],
                            dreamed_c_tp1=dream_data["continues_dreamed_t0_to_H_B"][t+1][0],
                            value_target=actor_critic_train_results["value_targets_H_B"][t][0],
                            initial_h=dream_data["h_states_t0_to_H_B"][t][0],
                            as_tensor=True,
                        )
                        tbx_writer.add_images(
                            f"dreamed_trajectories_for_critic_actor_learning[T={t},B=0]",
                            tf.expand_dims(img, axis=0).numpy(),
                            dataformats="NHWC",
                            global_step=total_env_steps,
                        )

                # Summarize actor-critic loss stats.
                summarize_critic_losses(
                    tbx_writer = tbx_writer,
                    step=total_env_steps,
                    actor_critic_train_results = actor_critic_train_results,
                )

                if train_actor:
                    summarize_actor_losses(
                        tbx_writer=tbx_writer,
                        step=total_env_steps,
                        actor_critic_train_results=actor_critic_train_results,
                    )

            print(
                f"\t\tL_actor={L_actor.numpy() if train_actor else 0.0:.5f} L_critic={L_critic.numpy():.5f}"
            )

        sub_iter += 1
        total_train_steps += 1

    total_replayed_steps += replayed_steps

    # EVALUATION.
    if evaluation_frequency_main_iters and (
            total_train_steps % evaluation_frequency_main_iters == 0
    ):
        print("\nEVALUATION:")
        # Dream a trajectory using the samples from the buffer and compare obs,
        # rewards, continues to the actually observed trajectory.
        dreamed_T = horizon_H
        print(f"\tDreaming trajectories (burn-in={burn_in_T}; H={dreamed_T}) from all 1st timesteps drawn from buffer ...")
        dream_data = dreamer_model.dream_trajectory_with_burn_in(
            observations=sample["obs"][:, :burn_in_T],  # use only first burn_in_T obs
            actions=sample["actions"][:, :burn_in_T + dreamed_T],  # use all actions from 0 to T (no actor)
            initial_h=sample["h_states"][:, 0],  # use initial T=0 h-states
            timesteps=dreamed_T,  # dream for n timesteps
            use_sampled_actions=True,  # use sampled actions, not the actor
        )

        mse_sampled_vs_dreamed_obs = summarize_dreamed_eval_trajectory_vs_samples(
            tbx_writer=tbx_writer,
            step=total_env_steps,
            dream_data=dream_data,
            sample=sample,
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
            num_episodes=evaluation_num_episodes,
            random_actions=False,
            with_render_data=True,
        )
        mean_episode_len = np.mean([len(eps) for eps in episodes])
        mean_episode_return = np.mean([eps.get_return() for eps in episodes])
        print(
            f"\tMean episode return: {mean_episode_return:.4f}; "
            f"mean len: {mean_episode_len:.1f}"
        )
        tbx_writer.add_scalar(
            "EVAL_mean_episode_return", mean_episode_return, global_step=total_env_steps
        )
        tbx_writer.add_scalar(
            "EVAL_mean_episode_length", mean_episode_len, global_step=total_env_steps
        )
        # Summarize (best and worst) evaluation episodes.
        sorted_episodes = sorted(episodes, key=lambda e: e.get_return())
        tbx_writer.add_video(
            f"EVAL_episode_video" + ("_best" if len(sorted_episodes) > 1 else ""),
            np.expand_dims(sorted_episodes[-1].render_images, axis=0),
            global_step=total_env_steps,
            fps=10,
            dataformats="NTHWC",
        )
        if len(sorted_episodes) > 1:
            tbx_writer.add_video(
                f"EVAL_episode_video_worst",
                np.expand_dims(sorted_episodes[0].render_images, axis=0),
                global_step=total_env_steps,
                fps=10,
                dataformats="NTHWC",
            )

    # Save the model every N iterations (but not after the very first).
    if iteration != 0 and model_save_frequency_main_iters and (
        iteration % model_save_frequency_main_iters == 0
    ):
        dreamer_model.save(f"checkpoints/dreamer_model_{iteration}")

    # Try trick from https://medium.com/dive-into-ml-ai/dealing-with-memory-leak-
    # issue-in-keras-model-training-e703907a6501
    if gc_frequency_train_steps and (
        total_train_steps % gc_frequency_train_steps == 0
    ):
        gc.collect()
        # tf.keras.backend.clear_session()  # <- this seems to be not needed.

    # Log GPU memory consumption.
    try:
        gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
        print(f"\nMEM (GPU) consumption: {gpu_memory['current']}")
        tbx_writer.add_scalar(
            "MEM_gpu_memory_used", gpu_memory['current'], global_step=total_env_steps
        )
        tbx_writer.add_scalar(
            "MEM_gpu_memory_peak", gpu_memory['peak'], global_step=total_env_steps
        )
    # No GPU? No problem.
    except ValueError:
        pass

    # Main iteration done.
    print("\n")
