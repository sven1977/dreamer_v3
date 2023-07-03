"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf

[2] Mastering Atari with Discrete World Models - 2021
D. Hafner, T. Lillicrap, M. Norouzi, J. Ba
https://arxiv.org/pdf/2010.02193.pdf
"""

import argparse
import datetime
import gc
import os
from pprint import pprint
import re
import time
import yaml

import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
import tensorflow as tf
import wandb

from ray.rllib.algorithms.dreamerv3.dreamerv3 import DreamerV3Config
from ray.rllib.algorithms.dreamerv3.tf.dreamerv3_tf_learner import DreamerV3TfLearner
from ray.rllib.algorithms.dreamerv3.utils.env_runner import DreamerV3EnvRunner
from ray.rllib.core.learner.learner import FrameworkHyperparameters
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
from ray.rllib.utils.replay_buffers.episode_replay_buffer import EpisodeReplayBuffer

import examples.debug_img_env  # to trigger DebugImgEnv import and registration
from utils.cartpole_debug import CartPoleDebug  # import registers `CartPoleDebug-v0`
from utils.tensorboard import (
    summarize_actor_train_results,
    summarize_critic_train_results,
    summarize_disagree_train_results,
    summarize_dreamed_eval_trajectory_vs_samples,
    summarize_dreamed_trajectory,
    summarize_forward_train_outs_vs_samples,
    summarize_sampling_and_replay_buffer,
    summarize_world_model_train_results,
)

now = datetime.datetime.now().strftime("%m-%d-%y-%H-%M-%S")

# Set GPU to grow in memory (so tf does not block all GPU mem).
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    "-c",
    type=str,
    default="examples/atari_100k.yaml",
    help="The config yaml file for the experiment.",
)
parser.add_argument(
    "--env",
    "-e",
    type=str,
    default="ALE/Pong-v5",
    help="The env to use for the experiment.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Checkpoint dir to load dreamer model model from."
)
args = parser.parse_args()
print(f"Trying to open config file {args.config} ...")
with open(args.config, "r") as f:
    config = yaml.safe_load(f)
assert len(config) == 1, "Only one experiment allowed in config yaml!"
experiment_name = next(iter(config.keys()))
config = config[experiment_name]
if args.env and not config.get("env"):
    config["env"] = args.env
    experiment_name += "-" + re.sub("^ALE/|-v\\d$", "", args.env).lower()

print(f"Running experiment {experiment_name} with the following config:")
pprint(config)

# Handle some paths for this experiment.
experiment_path = os.path.join(
    "experiments",
    experiment_name,
    now,
)
checkpoint_path = os.path.join(experiment_path, "checkpoints")
tensorboard_path = os.path.join(experiment_path, "tensorboard")
wandb_path = os.path.join(experiment_path, "wandb")
# Create the checkpoint path, if it doesn't exist yet.
os.makedirs(checkpoint_path)
# Create the wandb data dir.
os.makedirs(wandb_path)
wandb.init(
    project=re.sub("/", "-", experiment_name),
    name=now,
    dir=wandb_path,
    config=config,
)

# How many iterations do we pre-train?
num_pretrain_iterations = config.get("num_pretrain_iterations", 0)
# Every how many training steps do we write data to TB?
summary_frequency_train_steps = config.get("summary_frequency_train_steps", 20)
# Whether to summarize histograms as well.
summary_include_histograms = config.get("summary_include_histograms", False)
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
# to compute z, after that, only the prior (dynamics network) will be used (to compute
# z^).
burn_in_T = config.get("burn_in_T", 5)
horizon_H = config.get("horizon_H", 15)
assert burn_in_T + horizon_H <= batch_length_T, (
    f"ERROR: burn_in_T ({burn_in_T}) + horizon_H ({horizon_H}) must be <= "
    f"batch_length_T ({batch_length_T})!"
)

# Actor/critic hyperparameters.
discount_gamma = config.get("discount_gamma", 0.997)  # [1] eq. 7.
gae_lambda = config.get("gae_lambda", 0.95)  # [1] eq. 7.
entropy_scale = 3e-4  # [1] eq. 11.
return_normalization_decay = 0.99  # [1] eq. 11 and 12.

# EnvRunner config (an RLlib algorithm config).
algo_config = (
    DreamerV3Config()
    .environment(config["env"], env_config=config.get("env_config", {}))
    .training(
        model={
            "batch_length_T": batch_length_T,
            "horizon_H": horizon_H,
            "model_size": config["model_dimension"],
            "gamma": discount_gamma,
            "symlog_obs": False,
        },
        batch_size_B=batch_size_B,
        batch_length_T=batch_length_T,
        horizon_H=horizon_H,
        model_size=config["model_dimension"],
        gamma=discount_gamma,
        gae_lambda=gae_lambda,
        #burn_in_T=burn_in_T,
    )
    .reporting(report_individual_batch_item_stats=True)
    #.rl_module(rl_module_spec=dummy_spec)
    .rollouts(
        remote_worker_envs=False,
        num_envs_per_worker=config.get("num_envs_per_worker", 1),
        rollout_fragment_length=1,
    )
)
# The vectorized gymnasium EnvRunner to collect samples of shape (B, T, ...).
env_runner = DreamerV3EnvRunner(config=algo_config)

# Whether we have an image observation space or not.
is_img_space = len(env_runner.env.single_observation_space.shape) in [2, 3]
# Whether to symlog the observations or not.
symlog_obs = config.get("symlog_obs", not is_img_space)

action_space = env_runner.env.single_action_space

# Whether to o nly train the world model (not the critic and actor networks).
train_critic = config.get("train_critic", True)
train_actor = config.get("train_actor", True)
# Cannot train actor w/o critic.
assert not (train_actor and not train_critic)
# Whether to use the disagree-networks to compute intrinsic rewards for the dreamed
# data that critic and actor learn from.
use_curiosity = config.get("use_curiosity", False)
intrinsic_rewards_scale = config.get("intrinsic_rewards_scale", 0.1)

learner = DreamerV3TfLearner(
    module=env_runner.rl_module,
    learner_hyperparameters=algo_config.get_learner_hyperparameters(),
    framework_hyperparameters=FrameworkHyperparameters(eager_tracing=True),
)
learner.build()

# The replay buffer for storing actual env samples.
buffer = EpisodeReplayBuffer(capacity=int(1e6))
# Timesteps to put into the buffer before the first learning step.
warm_up_timesteps = 0

# Throughput metrics (EMA'd).
throughput_env_ts_per_second = None
throughput_train_ts_per_second = None

total_env_steps = 0
total_replayed_steps = 0
total_train_steps = 0
training_iteration_start = 0

# World model grad (by global norm) clipping according to [1] Appendix W.
world_model_grad_clip = 1000.0
# Critic grad (by global norm) clipping according to [1] Appendix W.
critic_grad_clip = 100.0
# Actor grad (by global norm) clipping according to [1] Appendix W.
actor_grad_clip = 100.0
# Disagree nets grad clipping according to Danijar's code.
disagree_grad_clip = 100.0

# Training ratio: Ratio of replayed steps over env steps.
training_ratio = config["training_ratio"]


for iteration in range(training_iteration_start, 1000000):
    t0 = time.time()

    print(f"Online training main iteration {iteration}")
    # Push enough samples into buffer initially before we start training.
    env_steps = env_steps_last_sample = 0

    if iteration == training_iteration_start:
        print(
            "Filling replay buffer so it contains at least "
            f"{batch_size_B * batch_length_T} ts (required for a single train batch)."
        )

    while True:
        # Sample one round.
        done_episodes, ongoing_episodes = env_runner.sample(random_actions=False)

        # We took B x T env steps.
        env_steps_last_sample = sum(
            len(eps) for eps in done_episodes + ongoing_episodes
        )
        env_steps += env_steps_last_sample
        total_env_steps += env_steps_last_sample

        # Add ongoing and finished episodes into buffer. The buffer will automatically
        # take care of properly concatenating (by episode IDs) the different chunks of
        # the same episodes, even if they come in via separate `add()` calls.
        buffer.add(episodes=done_episodes + ongoing_episodes)

        ts_in_buffer = buffer.get_num_timesteps()
        if (
            # Got to have more timesteps than warm up setting.
            ts_in_buffer > warm_up_timesteps
            # More timesteps than BxT.
            and ts_in_buffer >= batch_size_B * batch_length_T
            # And enough timesteps for the next train batch to not exceed
            # the training_ratio.
            and total_replayed_steps / total_env_steps < training_ratio
            ## But also at least as many episodes as the batch size B.
            ## Actually: This is not useful for longer episode envs, such as Atari.
            ## Too much initial data goes into the buffer, then.
            #and episodes_in_buffer >= batch_size_B
        ):
            # Summarize environment interaction and buffer data.
            summarize_sampling_and_replay_buffer(
                step=total_env_steps,
                replay_buffer=buffer,
                sampler_metrics=env_runner.get_metrics(),
                print_=True,
            )
            break

    replayed_steps = 0

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
        if isinstance(action_space, gym.spaces.Discrete):
            sample["actions_ints"] = sample["actions"]
            sample["actions"] = tf.one_hot(
                sample["actions_ints"], depth=action_space.n
            )

        results = learner.update(
            batch=MultiAgentBatch(
                {"default_policy": SampleBatch(sample)},
                env_steps=len(sample["obs"]),
            ),
        )
        results = results["default_policy"]
        learner.additional_update(timestep=0)  # timestep shouldn't matter

        if summary_frequency_train_steps and (
                total_train_steps % summary_frequency_train_steps == 0
        ):
            summarize_world_model_train_results(
                world_model_train_results=results,
                include_histograms=summary_include_histograms,
            )
            # Summarize actor-critic loss stats.
            if train_critic:
                summarize_critic_train_results(
                    actor_critic_train_results=results,
                    include_histograms=summary_include_histograms,
                )
            if train_actor:
                summarize_actor_train_results(
                    actor_critic_train_results=results,
                    include_histograms=summary_include_histograms,
                )
            if use_curiosity:
                summarize_disagree_train_results(
                    actor_critic_train_results=results,
                    include_histograms=summary_include_histograms,
                )
            # TODO: Make this work with any renderable env.
            if env_runner.config.env in [
                "CartPoleDebug-v0", "CartPole-v1", "FrozenLake-v1"
            ]:
                summarize_dreamed_trajectory(
                    dream_data=results["dream_data"],
                    actor_critic_train_results=results,
                    env=env_runner.config.env,
                    dreamer_model=learner.module.dreamer_model,
                    obs_dims_shape=sample["obs"].shape[2:],
                    desc="for_actor_critic_learning",
                )

        print(
            "\t\tWORLD_MODEL_L_total="
            f"{results['WORLD_MODEL_L_total']:.5f} ("
            "L_pred="
            f"{results['WORLD_MODEL_L_prediction']:.5f} ("
            f"dec/obs={results['WORLD_MODEL_L_decoder']} "
            f"rew(two-hot)={results['WORLD_MODEL_L_reward']} "
            f"cont={results['WORLD_MODEL_L_continue']}"
            "); "
            f"L_dyn={results['WORLD_MODEL_L_dynamics']:.5f}; "
            "L_rep="
            f"{results['WORLD_MODEL_L_representation']:.5f})"
        )
        print("\t\t", end="")
        if train_actor:
            L_actor = results["ACTOR_L_total"]
            print(f"L_actor={L_actor if train_actor else 0.0:.5f} ", end="")
        if train_critic:
            L_critic = results["CRITIC_L_total"]
            print(f"L_critic={L_critic:.5f} ", end="")
        if use_curiosity:
            L_disagree = results["DISAGREE_L_total"]
            print(f"L_disagree={L_disagree:.5f}", end="")
        print()

        sub_iter += 1
        total_train_steps += 1

    total_replayed_steps += replayed_steps

    # Try trick from https://medium.com/dive-into-ml-ai/dealing-with-memory-leak-
    # issue-in-keras-model-training-e703907a6501
    if gc_frequency_train_steps and (
        total_train_steps % gc_frequency_train_steps == 0
    ):
        gc.collect()

    t1 = time.time()
    ema = 0.98
    env_ts_per_sec = env_steps / (t1 - t0)
    train_ts_per_sec = replayed_steps / (t1 - t0)
    # Init.
    if throughput_train_ts_per_second is None:
        throughput_env_ts_per_second = env_ts_per_sec
        throughput_train_ts_per_second = train_ts_per_sec
    # EMA.
    else:
        throughput_env_ts_per_second = ema * throughput_env_ts_per_second + (1.0 - ema) * env_ts_per_sec
        throughput_train_ts_per_second = ema * throughput_train_ts_per_second + (1.0 - ema) * train_ts_per_sec

    # Final wandb (env) step commit.
    wandb.log({
        "THROUGHPUT_env_ts_per_sec": throughput_env_ts_per_second,
        "THROUGHPUT_train_ts_per_sec": throughput_train_ts_per_second,
    }, step=total_env_steps, commit=True)

    # Main iteration done.
    print()
