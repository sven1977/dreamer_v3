import tree  # pip install dm_tree
import tensorflow as tf

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

from models.world_model_atari import WorldModelAtari
from utils.env_runner import EnvRunner
from utils.replay_buffer import ReplayBuffer
from losses.world_model_losses import (
    world_model_dynamics_and_representation_loss,
    world_model_prediction_losses,
)


batch_size_B = 16
batch_length_T = 64

# EnvRunner config (an RLlib algorithm config).
config = (
    AlgorithmConfig()
    .environment("ALE/MsPacman-v5")
    .rollouts(num_envs_per_worker=2, rollout_fragment_length=200)
)
# The vectorized gymnasium EnvRunner to collect samples of shape (B, T, ...).
env_runner = EnvRunner(model=None, config=config, max_seq_len=batch_length_T)

# Our DreamerV3 world model.
world_model = WorldModelAtari(
    action_space=env_runner.env.single_action_space,
    batch_length_T=batch_length_T,
)
# TODO: ugly hack (resulting from the insane fact that you cannot know
#  an env's spaces prior to actually constructing an instance of it :(
env_runner.model = world_model

# The replay buffer for storing actual env samples.
buffer = ReplayBuffer(capacity=10000)

# Use an Adam optimizer.
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-8)

# World model grad clipping according to [1].
grad_clip = 1000.0

for iteration in range(10):
    # Push enough samples into buffer initially before we start training.
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
            mask,
        # TODO: random_actions=False; right now, we act randomly, but perform a
        #  world-model forward pass using the random actions (in order to compute
        #  the h-states).
        ) = env_runner.sample(random_actions=True)
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
        if len(buffer) >= batch_size_B:
            break

    # Draw a sample from the replay buffer.
    sample = buffer.sample(num_items=batch_size_B)
    # Convert samples (numpy) to tensors.
    sample = tree.map_structure(lambda v: tf.convert_to_tensor(v), sample)

    # Compute losses.
    with tf.GradientTape() as tape:
        # Compute forward values.
        forward_train_outs = world_model(
            inputs=sample["obs"],
            actions=sample["actions"],
            initial_h=sample["h_states"],
        )

        prediction_losses = world_model_prediction_losses(
            observations=sample["obs"],
            rewards=sample["rewards"],
            terminateds=sample["terminateds"],
            truncateds=sample["truncateds"],
            forward_train_outs=forward_train_outs,
        )
        L_pred = prediction_losses["total_loss"]

        L_dyn, L_rep = world_model_dynamics_and_representation_loss(
            forward_train_outs=forward_train_outs
        )
        L_total = 1.0 * L_pred + 0.5 * L_dyn + 0.1 * L_rep

        # Bring back into (B, T)-shape.
        L_total = tf.reshape(L_total, shape=(batch_size_B, batch_length_T))
        # Mask out invalid timesteps (episode terminated/truncated).
        L_total = L_total * sample["mask"]
        # Sum up timesteps, and average over batch (see eq. 4 in [1]).
        L_total = tf.reduce_mean(tf.reduce_sum(L_total, axis=-1))

    # Get the gradients from the tape.
    gradients = tape.gradient(L_total, world_model.trainable_variables)
    # Clip all gradients.
    clipped_gradients = []
    for grad in gradients:
        clipped_gradients.append(tf.clip_by_value(grad, -grad_clip, grad_clip))
    # Apply gradients to our model.
    optimizer.apply_gradients(zip(clipped_gradients, world_model.trainable_variables))

    print(
        f"Iter {iteration}) L_total={L_total.numpy()} "
        f"(L_pred={L_pred.numpy()}; L_dyn={L_dyn.numpy()}; L_rep={L_rep.numpy()})"
    )
