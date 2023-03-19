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
import yaml

import gymnasium as gym
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
    summarize_actor_train_results,
    summarize_critic_train_results,
    summarize_disagree_train_results,
    summarize_dreamed_eval_trajectory_vs_samples,
    summarize_dreamed_trajectory,
    summarize_forward_train_outs_vs_samples,
    summarize_sampling_and_replay_buffer,
    summarize_world_model_train_results,
)

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

print(f"Running with the following config:")
pprint(config)

# Handle some paths for this experiment.
experiment_path = os.path.join(
    "experiments",
    experiment_name,
    datetime.datetime.now().strftime("%m-%d-%y-%H-%M-%S"),
)
checkpoint_path = os.path.join(experiment_path, "checkpoints")
tensorboard_path = os.path.join(experiment_path, "tensorboard")
# Create the checkpoint path, if it doesn't exist yet.
os.makedirs(checkpoint_path)
# Create the tensorboard summary data dir.
os.makedirs(tensorboard_path)
tbx_writer = SummaryWriter(tensorboard_path)

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
    AlgorithmConfig()
    .environment(config["env"], env_config=config.get("env_config", {}))
    .rollouts(
        num_envs_per_worker=config.get("num_envs_per_worker", 1),
        rollout_fragment_length=1,
    )
)
# The vectorized gymnasium EnvRunner to collect samples of shape (B, T, ...).
env_runner = EnvRunnerV2(model=None, config=algo_config)
env_runner_evaluation = EnvRunnerV2(model=None, config=algo_config)

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

# Our DreamerV3 world model.
from_checkpoint = None
# Uncomment this next line to load from a saved model.
#from_checkpoint = "checkpoints/dreamer_model_0"
if from_checkpoint is not None:
    dreamer_model = tf.keras.models.load_model(from_checkpoint)
else:
    model_dimension = config["model_dimension"]
    gray_scaled = is_img_space and len(env_runner.env.single_observation_space.shape) == 2
    world_model = WorldModel(
        model_dimension=model_dimension,
        action_space=action_space,
        batch_length_T=batch_length_T,
        num_gru_units=config.get("num_gru_units"),
        encoder=(
            CNNAtari(model_dimension=model_dimension) if is_img_space
            else MLP(model_dimension=model_dimension)
        ),
        decoder=ConvTransposeAtari(
            model_dimension=model_dimension,
            gray_scaled=gray_scaled,
        ) if is_img_space else VectorDecoder(
            model_dimension=model_dimension,
            observation_space=env_runner.env.single_observation_space,
        ),
        symlog_obs=symlog_obs,
    )
    dreamer_model = DreamerModel(
        model_dimension=model_dimension,
        action_space=action_space,
        world_model=world_model,
        use_curiosity=use_curiosity,
        intrinsic_rewards_scale=intrinsic_rewards_scale,
    )

# TODO: ugly hack (resulting from the insane fact that you cannot know
#  an env's spaces prior to actually constructing an instance of it) :(
env_runner.model = dreamer_model
env_runner_evaluation.model = dreamer_model

# The replay buffer for storing actual env samples.
buffer = EpisodeReplayBuffer(capacity=int(1e6))
# Timesteps to put into the buffer before the first learning step.
warm_up_timesteps = 0

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

total_env_steps = 0
total_replayed_steps = 0
total_train_steps = 0

if num_pretrain_iterations > 0:
    # 1) Initialize dataset
    import d3rlpy
    if config["env"].startswith("ALE/"):
        dataset, _ = d3rlpy.datasets.get_atari(config["offline_dataset"])
    elif config["env"] == "CartPole-v0":
        dataset, _ = d3rlpy.datasets.get_cartpole()
    elif config["env"] == "Pendulum-v0":
        dataset, _ = d3rlpy.datasets.get_pendulum()
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
        #if iteration == 0:
        #    # Dummy forward pass to be able to produce summary.
        #    world_model(
        #        sample["obs"][:, 0],
        #        sample["actions"][:, 0],
        #        sample["h_states"][:, 0],
        #    )
        #    world_model.summary()

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
            summarize_world_model_train_results(
                tbx_writer=tbx_writer,
                step=total_env_steps,
                world_model_train_results=world_model_train_results,
            )

        print(
            "\t\tL_world_model_total="
            f"{world_model_train_results['L_world_model_total'].numpy():.5f} ("
            f"L_pred={world_model_train_results['L_pred'].numpy():.5f} ("
            f"decoder/obs={world_model_train_results['L_decoder'].numpy()} "
            f"reward(two-hot)={world_model_train_results['L_reward_two_hot'].numpy()} "
            f"cont={world_model_train_results['L_continue'].numpy()}"
            "); "
            f"L_dyn={world_model_train_results['L_dyn'].numpy():.5f}; "
            f"L_rep={world_model_train_results['L_rep'].numpy():.5f})"
        )

    print()
    print(
        "Pretraining offline completed ... switching to online training and evaluation"
    )


for iteration in range(1000000):
    print(f"Online training main iteration {iteration}")
    # Push enough samples into buffer initially before we start training.
    env_steps = env_steps_last_sample = 0
    #TEST: Put only a single row in the buffer and try to memorize it.
    #env_steps_last_sample = 64
    #while iteration == 0:
    #END TEST

    if iteration == 0:
        print(
            "Pre-filling replay buffer so it contains at least "
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
        # the same episodes, even if they come in in separate `add()` calls.
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
                tbx_writer=tbx_writer,
                step=total_env_steps,
                replay_buffer=buffer,
                sampler_metrics=env_runner.get_metrics(),
                print_=True,
            )
            break

    replayed_steps = 0

    #TEST: re-use same sample.
    #sample = buffer.sample(num_items=batch_size_B)
    #sample = tree.map_structure(lambda v: tf.convert_to_tensor(v), sample)
    #END TEST

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

        # Perform one world-model training step.
        world_model_train_results = train_world_model_one_step(
            sample=sample,
            batch_size_B=tf.convert_to_tensor(batch_size_B),
            batch_length_T=tf.convert_to_tensor(batch_length_T),
            grad_clip=tf.convert_to_tensor(world_model_grad_clip),
            world_model=world_model,
        )
        world_model_forward_train_outs = (
            world_model_train_results["WORLD_MODEL_forward_train_outs"]
        )

        # Update h_states in buffer after the world model (sequential model)
        # forward pass.
        #h_BxT = forward_train_outs["h_states_BxT"]
        #h_B_t2_to_Tp1 = tf.concat([tf.reshape(
        #    h_BxT,
        #    shape=(batch_size_B, batch_length_T) + h_BxT.shape[1:],
        #)[:, 1:], tf.expand_dims(h_states_training, axis=1)], axis=1)
        #buffer.update_h_states(h_B_t2_to_Tp1.numpy(), sample["indices"].numpy())

        # Summarize world model.
        #if iteration == 0 and sub_iter == 0 and num_pretrain_iterations == 0:
            # Dummy forward pass to be able to produce summary.
            #world_model(
            #    {
            #        "h": world_model_forward_train_outs["h_states_BxT"][:4],
            #        "z": world_model_forward_train_outs["z_states_BxT"][:4],
            #        "a": sample["actions"][:4, 0],
            #    },
            #    sample["obs"][:4, 0],
            #    sample["is_first"][:4, 0],
            #)
            #world_model.summary()

        if summary_frequency_train_steps and (
                total_train_steps % summary_frequency_train_steps == 0
        ):
            summarize_forward_train_outs_vs_samples(
                tbx_writer=tbx_writer,
                step=total_env_steps,
                forward_train_outs=world_model_forward_train_outs,
                sample=sample,
                batch_size_B=batch_size_B,
                batch_length_T=batch_length_T,
                symlog_obs=symlog_obs,
            )
            summarize_world_model_train_results(
                tbx_writer=tbx_writer,
                step=total_env_steps,
                world_model_train_results=world_model_train_results,
            )

        print(
            "\t\tWORLD_MODEL_L_total="
            f"{world_model_train_results['WORLD_MODEL_L_total'].numpy():.5f} ("
            "L_pred="
            f"{world_model_train_results['WORLD_MODEL_L_prediction'].numpy():.5f} ("
            f"dec/obs={world_model_train_results['WORLD_MODEL_L_decoder'].numpy()} "
            f"rew(two-hot)={world_model_train_results['WORLD_MODEL_L_reward'].numpy()} "
            f"cont={world_model_train_results['WORLD_MODEL_L_continue'].numpy()}"
            "); "
            f"L_dyn={world_model_train_results['WORLD_MODEL_L_dynamics'].numpy():.5f}; "
            "L_rep="
            f"{world_model_train_results['WORLD_MODEL_L_representation'].numpy():.5f})"
        )

        # Train critic and actor.
        if train_critic:
            # Build critic model first, so we can initialize EMA weights.
            if not dreamer_model.critic.trainable_variables:
                # Forward pass for fast critic.
                dreamer_model.critic(
                    h=world_model_forward_train_outs["h_states_BxT"],
                    z=world_model_forward_train_outs["z_states_BxT"],
                    return_logits=True,
                )
                # Forward pass for EMA-weights critic.
                dreamer_model.critic(
                    h=world_model_forward_train_outs["h_states_BxT"],
                    z=world_model_forward_train_outs["z_states_BxT"],
                    return_logits=False,
                    use_ema=True,
                )
                dreamer_model.critic.init_ema()
                # Summarize critic models.
                #dreamer_model.critic.summary()

            actor_critic_train_results = train_actor_and_critic_one_step(
                world_model_forward_train_outs=world_model_forward_train_outs,
                is_terminated=tf.reshape(sample["is_terminated"], [-1]),
                horizon_H=horizon_H,
                gamma=discount_gamma,
                lambda_=gae_lambda,
                actor_grad_clip=actor_grad_clip,
                critic_grad_clip=critic_grad_clip,
                disagree_grad_clip=disagree_grad_clip,
                dreamer_model=dreamer_model,
                entropy_scale=entropy_scale,
                return_normalization_decay=return_normalization_decay,
                train_actor=train_actor,
                use_curiosity=use_curiosity,
            )
            L_critic = actor_critic_train_results["CRITIC_L_total"]
            if train_actor:
                L_actor = actor_critic_train_results["ACTOR_L_total"]
            if use_curiosity:
                L_disagree = actor_critic_train_results["DISAGREE_L_total"]
            dream_data = actor_critic_train_results["dream_data"]

            # Summarize critic models.
            #if iteration == 0 and sub_iter == 0:
                # Dummy forward pass to be able to produce summary.
                #if train_actor:
                #    dreamer_model.actor.summary()
                #if use_curiosity:
                #    dreamer_model.disagree_nets(
                #        dream_data["h_states_t0_to_H_B"][0],
                #        z=dream_data["z_states_prior_t0_to_H_B"][0],
                #        a=dream_data["actions_dreamed_t0_to_H_B"][0],
                #    )
                #    #dreamer_model.disagree_nets.summary()

            # Analyze generated dream data for its suitability in training the critic
            # and actors.
            if summary_frequency_train_steps and (
                total_train_steps % summary_frequency_train_steps == 0
            ):
                # Summarize actor-critic loss stats.
                summarize_critic_train_results(
                    tbx_writer = tbx_writer,
                    step=total_env_steps,
                    actor_critic_train_results = actor_critic_train_results,
                )

                if train_actor:
                    summarize_actor_train_results(
                        tbx_writer=tbx_writer,
                        step=total_env_steps,
                        actor_critic_train_results=actor_critic_train_results,
                    )
                if use_curiosity:
                    summarize_disagree_train_results(
                        tbx_writer=tbx_writer,
                        step=total_env_steps,
                        actor_critic_train_results=actor_critic_train_results,
                    )

                # TODO: Make this work with any renderable env.
                if env_runner.config.env in [
                    "CartPoleDebug-v0", "CartPole-v1", "FrozenLake-v1"
                ]:
                    summarize_dreamed_trajectory(
                        tbx_writer=tbx_writer,
                        dream_data=dream_data,
                        actor_critic_train_results=actor_critic_train_results,
                        env=env_runner.config.env,
                        dreamer_model=dreamer_model,
                        obs_dims_shape=sample["obs"].shape[2:],
                        step=total_env_steps,
                        desc="for_actor_critic_learning",
                    )

            print(
                f"\t\tL_actor={L_actor.numpy() if train_actor else 0.0:.5f} "
                f"L_critic={L_critic.numpy():.5f}"
            )

        sub_iter += 1
        total_train_steps += 1

    total_replayed_steps += replayed_steps

    # EVALUATION.
    if evaluation_frequency_main_iters and (
            total_train_steps % evaluation_frequency_main_iters == 0
    ):
        print("\nEVALUATION:")

        # Special debug evaluation for intrinsic rewards -> Roll out a special
        # episode and draw the intrinsic rewards in the rendered images so we can
        # check, whether curiosity is actually producing the correct rewards.
        #if use_curiosity and env_runner.config.env == "FrozenLake-v1":
        #    start_states = dreamer_model.get_initial_state(batch_size_B=batch_size_B)
        #    dream_data = dreamer_model.dream_trajectory_with_burn_in(
        #        start_states=start_states,
        #        timesteps_burn_in=0,
        #        timesteps_H=horizon_H,
        #        # Initial state observation (start state of grid world).
        #        observations=np.array([[[1.0] + [0.0] * 15]]),  # [B=1, T=1, 16]
        #        actions=np.one_hot(np.array([[  # B=1, T=
        #            2, 2, 1, 1, 1, 2
        #        ]]), depth=4),
        #        use_sampled_actions_in_dream=True,
        #        use_random_actions_in_dream=False,
        #    )
        #    summarize_dreamed_trajectory(
        #        tbx_writer=tbx_writer,
        #        dream_data=dream_data,
        #        actor_critic_train_results=actor_critic_train_results,
        #        env=env_runner.config.env,
        #        dreamer_model=dreamer_model,
        #        obs_dims_shape=sample["obs"].shape[2:],
        #        step=total_env_steps,
        #        desc="for_intrinsic_reward_debugging",
        #    )

        # Dream a trajectory using the samples from the buffer and compare obs,
        # rewards, continues to the actually observed trajectory (from the real env).
        # Use a burn-in window where we compute posterior states (using the actual
        # observations), then a dream window, where we compute prior states, but still
        # use the actions from the real env (to be able to compare with the sampled
        # trajectory).
        dreamed_T = horizon_H
        print(
            f"\tDreaming trajectories (burn-in={burn_in_T}; H={dreamed_T}) starting "
            "from all 1st timesteps drawn from buffer to compare with sampled data "
            "(using the same actions as in the sampling) ..."
        )
        start_states = dreamer_model.get_initial_state(batch_size_B=batch_size_B)
        dream_data = dreamer_model.dream_trajectory_with_burn_in(
            start_states=start_states,
            timesteps_burn_in=burn_in_T,
            timesteps_H=horizon_H,
            # Use only first burn_in_T obs.
            observations=sample["obs"][:, :burn_in_T],
            # Use all actions from 0 to T (no actor).
            actions=sample["actions"][:, :burn_in_T + dreamed_T],
            # Use sampled actions, not the actor.
            use_sampled_actions_in_dream=True,
            use_random_actions_in_dream=False,
        )
        mse_sampled_vs_dreamed_obs = summarize_dreamed_eval_trajectory_vs_samples(
            tbx_writer=tbx_writer,
            step=total_env_steps,
            dream_data=dream_data,
            sample=sample,
            burn_in_T=burn_in_T,
            dreamed_T=dreamed_T,
            dreamer_model=dreamer_model,
            symlog_obs=symlog_obs,
        )
        print(
            f"\tMSE sampled vs dreamed obs (B={batch_size_B} T/H={dreamed_T}): "
            f"{mse_sampled_vs_dreamed_obs:.6f}"
        )

        # Run n episodes in an actual env and report mean episode returns.
        print(f"Running {evaluation_num_episodes} episodes in env for evaluation ...")
        episodes = env_runner_evaluation.sample_episodes(
            num_episodes=evaluation_num_episodes,
            random_actions=False,
            with_render_data=True,
        )
        metrics = env_runner_evaluation.get_metrics()
        if "episode_returns" in metrics:
            print(
                f"\tMean episode return: {np.mean(metrics['episode_returns']):.4f}; "
                f"mean len: {np.mean(metrics['episode_lengths']):.1f}"
            )
            tbx_writer.add_scalar(
                "EVALUATION_mean_episode_return",
                np.mean(metrics['episode_returns']),
                global_step=total_env_steps,
            )
            tbx_writer.add_scalar(
                "EVALUATION_mean_episode_length",
                np.mean(metrics['episode_lengths']),
                global_step=total_env_steps,
            )
        # Summarize (best and worst) evaluation episodes.
        sorted_episodes = sorted(episodes, key=lambda e: e.get_return())
        tbx_writer.add_video(
            f"EVALUATION_episode_video" + ("_best" if len(sorted_episodes) > 1 else ""),
            np.expand_dims(sorted_episodes[-1].render_images, axis=0),
            global_step=total_env_steps,
            fps=15,
            dataformats="NTHWC",
        )
        if len(sorted_episodes) > 1:
            tbx_writer.add_video(
                f"EVALUATION_episode_video_worst",
                np.expand_dims(sorted_episodes[0].render_images, axis=0),
                global_step=total_env_steps,
                fps=15,
                dataformats="NTHWC",
            )

    # Save the model every N iterations.
    if model_save_frequency_main_iters and (
        iteration % model_save_frequency_main_iters == 0
    ):
        #try:
        print("\nSAVING model ...")
        dreamer_model.save(f"{checkpoint_path}/dreamer_model_{iteration}")
        #except Exception as e:
        #    print(f"ERROR: Trying to save DreamerModel!!\nError is {e}")

    # Try trick from https://medium.com/dive-into-ml-ai/dealing-with-memory-leak-
    # issue-in-keras-model-training-e703907a6501
    if gc_frequency_train_steps and (
        total_train_steps % gc_frequency_train_steps == 0
    ):
        gc.collect()
        #tf.keras.backend.clear_session()  # <- this seems to be not needed.

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
    print()
