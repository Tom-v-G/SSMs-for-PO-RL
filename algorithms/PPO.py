# A clean single file implementation of PPO in Equinox
# Largely inspired by cleanRL (https://github.com/vwxyzjn/cleanrl) 
# and PureJaxRL (https://github.com/luchris429/purejaxrl)

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx
import chex
import optax
import gymnax
import distrax
from typing import NamedTuple, List, Tuple
from dataclasses import replace
from omegaconf import OmegaConf

import os
import argparse
import time
import sys
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from utils import create_save_dir, load_yaml_config, log_metrics, cfg2dict
from models.selector import initialize_model
# from RLkoen.util.wrappers import LogWrapper
from envs.wrappers import LogWrapper, AliasPrevActionV2
from envs.registration import make
from optimizer import initialize_optimizer

import numpy as np

# logging 
import wandb

@chex.dataclass(frozen=True)
class PPOConfig:
    ssm_lr: float= 1e-4
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 100.0
    clip_coef: float = 0.2
    clip_coef_vf: float = 10.0 # Depends on the reward scaling !
    ent_coef: float = 0.01
    vf_coef: float = 0.25

    total_timesteps: int = 1e6
    num_envs: int = 6
    num_steps: int = 128 # steps per environment
    num_minibatches: int = 4 # Number of mini-batches
    update_epochs: int = 4 # K epochs to update the policy
    # to be filled in runtime:
    batch_size: int = 0 # batch size (num_envs * num_steps)
    minibatch_size: int = 0 # mini-batch size (batch_size / num_minibatches)
    num_iterations: int = 0 # number of iterations (total_timesteps / num_steps / num_envs)

    test_envs: int = 8
    seed: int = 4
    dpo_loss: bool = False
    debug: bool = False
    wandb: bool = False
    evaluate_deterministically: bool = False

# Define a simple tuple to hold the state of the environment. 
# This is the format we will use to store transitions in our buffer.
@chex.dataclass(frozen=True)
class Transition:
    observation: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    value: chex.Array
    log_prob: chex.Array
    info: chex.Array

class TrainState(NamedTuple):
    dynamic_model: eqx.Module
    optimizer_state: optax.OptState
    key: jax.Array

def reset_hidden_states(x, dones=None):
    """
    Resets hidden states based on the provided done signals
    note: x has shape (num_envs, res_layers, H, N)
    """
    for index, done in enumerate(dones):
        h = done * jnp.zeros_like(x[0]) + (1-done) * x[index]
        x = x.at[index].set(h)
    return x

# Jit the returned function, not this function
def create_ppo_train_object(
        env,
        env_params,
        cfg,
        logger
    ):

    jax.debug.print("Start PPO initilialisation")
    # setup env (wrappers) and config
    
    if cfg.env_action_embedding:
        env = AliasPrevActionV2(env)
    env = LogWrapper(env)

    # set PPO config
    PPO_config = PPOConfig(**cfg.PPOConfig)

    # setting runtime parameters
    num_iterations = int(float(PPO_config.total_timesteps)) // int(PPO_config.num_steps) // int(PPO_config.num_envs)
    minibatch_size = int(PPO_config.num_envs) * int(PPO_config.num_steps) // int(PPO_config.num_minibatches)
    batch_size = int(minibatch_size) * int(PPO_config.num_minibatches)
    PPO_config = replace(
        PPO_config,
        num_iterations=num_iterations,
        minibatch_size=minibatch_size,
        batch_size=batch_size
    )

    # rng keys
    rng = jax.random.PRNGKey(PPO_config.seed)
    rng, network_key, reset_key, train_key = jax.random.split(rng, 4)

    if PPO_config.debug:
        jax.debug.print(f"Initializing {cfg.model} model")
    model, filter_spec = initialize_model(cfg, env, env_params, network_key)
    # if PPO_config.debug:
    #     jax.debug.print(f"{model}")

    # Create dynamic and static model parts for both training and inference
    dynamic, static = eqx.partition(model, filter_spec)
    static_inference = eqx.nn.inference_mode(static)
    static_model = {
        'training': static,
        'inference': static_inference
    }

    # Initialize Optimizer
    optimizer = initialize_optimizer(model, cfg)
    optimizer_state = optimizer.init(eqx.filter(model, filter_spec))

    # Create Trainstate object
    # Note: supply only dynamic model parts
    train_state = TrainState(
        dynamic_model=dynamic,
        optimizer_state=optimizer_state,
        key=train_key
    )

    # Initialise Environments, hidden states and done signals
    rng, key = jax.random.split(rng)
    reset_key = jax.random.split(key, PPO_config.num_envs)

    obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_key, env_params)
    # TODO vmap hidden state initialization over number of environments
    x_init = model.initialize_hidden_state(PPO_config.num_envs)
    done = jnp.zeros(shape=(PPO_config.num_envs,), dtype=jnp.bool)

    if PPO_config.debug:
        jax.debug.print(f"PPO initilialisation completed")
        jax.debug.print(f'Parameter count: {sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, filter_spec)))}')

    def eval_func(train_state, rng):
        
        model = eqx.combine(train_state.dynamic_model, static_model['inference'])

        @eqx.filter_jit
        def step_envs(carry):
            
            rng, last_obs, env_state, last_done, episode_reward, x = carry
            rng, model_key, action_key, step_key = jax.random.split(rng, 4)
            model_keys = jax.random.split(model_key, last_obs.shape[0])
            
            # run model
            action_dist, _, x = jax.vmap(model)(last_obs, x, last_done, model_keys)

            # Step environment
            if PPO_config.evaluate_deterministically:
                action = jnp.argmax(action_dist.logits, axis=-1)
            else:
                action = action_dist.sample(seed=action_key)
            action = jnp.squeeze(action)

            rng, key = jax.random.split(rng)
            step_key = jax.random.split(key, PPO_config.test_envs)
            obs, env_state, reward, done, _ = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(step_key, env_state, action, env_params)
            
            # Update reward
            new_reward = episode_reward + reward
            
            # Update carry
            episode_reward = jnp.where(last_done, episode_reward, new_reward)
            done = jnp.where(last_done, last_done, done)

            return (rng, obs, env_state, done, episode_reward, x)
        
        def cond_func(carry):
            _, _, _, done, _, _ = carry
            return jnp.any(jnp.logical_not(done))
        
        rng, reset_key = jax.random.split(rng)
        reset_keys = jax.random.split(reset_key, PPO_config.test_envs)

        obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, env_params)
        x_init = model.initialize_hidden_state(PPO_config.test_envs)
        done = jnp.zeros(shape=(PPO_config.test_envs,), dtype=jnp.bool)
        episode_reward = jnp.zeros(shape=(PPO_config.test_envs,), dtype=jnp.float32)

        rng, obs, env_state, done, episode_reward, x = jax.lax.while_loop(cond_func, step_envs, (rng, obs, env_state, done, episode_reward, x_init))

        return episode_reward

    def train_func(rng=rng, static_model=static_model):
        
        def scan_env(runner_state, static_model):

            # jax.debug.print("Starting environment scan")

            def _env_step(runner_state, _):
                train_state, env_state, last_obs, last_x, last_done ,rng = runner_state
                rng, sample_key, step_key = jax.random.split(rng, 3)
                # Last_obs shape: (num envs, L)
                
                # Recombine models
                model = eqx.combine(train_state.dynamic_model, static_model['training'])

                # select an action and ascertain value
                rng, model_key = jax.random.split(rng, 2)
                model_keys = jax.random.split(model_key, last_obs.shape[0])
                action_dist, value, x = jax.vmap(model, in_axes=(0, 0, 0, 0))(last_obs, last_x, last_done, model_keys)
                action, log_prob = action_dist.sample_and_log_prob(seed=sample_key)
                
                # Remove redundant dimension (sequence length = 1), ensure all values are (at least) 1D arrays (single environment)
                value, action, log_prob = jnp.squeeze(value), jnp.squeeze(action), jnp.squeeze(log_prob)
                value, action, log_prob = jnp.array(value, ndmin=1), jnp.array(action, ndmin=1), jnp.array(log_prob, ndmin=1)

                # Take a step in the environment
                rng, key = jax.random.split(rng)
                step_key = jax.random.split(key, PPO_config.num_envs)
                obs, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(step_key, env_state, action, env_params)

                x = reset_hidden_states(x, done)

                # Build a single transition. Jax.lax.scan will build the batch
                # returning num_steps transitions.
                transition = Transition(
                    observation=last_obs,
                    action=action,
                    reward=reward,
                    done=last_done,
                    value=value,
                    log_prob=log_prob,
                    info=info
                )
                runner_state = (train_state, env_state, obs, x, done, rng)
                return runner_state, transition
        
            runner_state, trajectory_batch = jax.lax.scan(
                    _env_step, runner_state, None, PPO_config.num_steps
                )
            return runner_state, trajectory_batch

        def scan_epoch(update_state, static_model):

            def _update_epoch(update_state, _):
                """ Do one epoch of update"""

                # @jax.value_and_grad(has_aux=True)
                def __ppo_los_fn(dynamic, x_init, trajectory_minibatch, advantages, returns, key):
                    # Recombine static and differentiable parameters
                    model = eqx.combine(dynamic, static_model['training'])
                    # Map networks over observations
                    # Note: minibatches can contain multiple transitions
                    key, model_key = jax.random.split(key, 2)
                    model_keys = jax.random.split(model_key, trajectory_minibatch.observation.shape[0])
                    
                    action_dist, value, _ = jax.vmap(model)(trajectory_minibatch.observation, x_init, trajectory_minibatch.done, model_keys)
                    log_prob = action_dist.log_prob(trajectory_minibatch.action)
                    entropy = action_dist.entropy().mean()

                    # jax.debug.breakpoint()

                    def ___ppo_actor_los():
                        # actor loss 
                        ratio = jnp.exp(log_prob - trajectory_minibatch.log_prob)
                        _advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                        actor_loss1 = _advantages * ratio
                        actor_loss2 = (
                            jnp.clip(
                                ratio, 1.0 - PPO_config.clip_coef, 1.0 + PPO_config.clip_coef
                            ) * _advantages
                        )
                        actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()
                        return actor_loss
                
                    actor_loss = ___ppo_actor_los() 

                    value_pred_clipped = trajectory_minibatch.value + (
                        jnp.clip(
                            value - trajectory_minibatch.value, -PPO_config.clip_coef_vf, PPO_config.clip_coef_vf
                        )
                    )
                    value_losses = jnp.square(value - returns)
                    value_losses_clipped = jnp.square(value_pred_clipped - returns)
                    value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()

                    # Total loss
                    total_loss = (
                        actor_loss 
                        + PPO_config.vf_coef * value_loss
                        - PPO_config.ent_coef * entropy
                    )
                    
                    return total_loss, (actor_loss, value_loss, entropy)
            
                def __update_over_minibatch(train_state: TrainState, minibatch):
                    x_init, trajectory_mb, advantages_mb, returns_mb = minibatch

                    key = train_state.key
                    key, ppo_los_key = jax.random.split(key, 2)

                    (total_loss, _), grads = jax.value_and_grad(__ppo_los_fn, argnums=0, has_aux=True)(
                        train_state.dynamic_model, x_init, trajectory_mb, advantages_mb, returns_mb, ppo_los_key
                    )

                    updates, optimizer_state = optimizer.update(grads, train_state.optimizer_state, eqx.filter(train_state.dynamic_model, filter_spec))
                    
                    new_dynamic = eqx.apply_updates(train_state.dynamic_model, updates)
        
                    train_state = TrainState(
                        dynamic_model = new_dynamic,
                        optimizer_state=optimizer_state,
                        key=key
                    )
                    return train_state, total_loss
            
                train_state, x_init, trajectory_batch, advantages, returns, rng = update_state
                rng, key = jax.random.split(rng)

                # Convert trajectory batch, advantages and returns to batch dim first
                swapped_trajectory_batch = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(x, 1, 0),
                    trajectory_batch
                )

                swapped_advantages = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(x, 1, 0),
                    advantages
                )

                swapped_returns = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(x, 1, 0),
                    returns
                )

                # Permute order of environments
                batch_idx = jax.random.permutation(key, PPO_config.num_envs)
                batch = (x_init, swapped_trajectory_batch, swapped_advantages, swapped_returns)

                # take from the batch in a new order (the order of the randomized batch_idx)
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, batch_idx, axis=0), batch
                )
                
                # Create minibatches (note: num_minibatches should divide total envs)
                # TODO: figure out how to deal with multiple transitions in a single minibatch

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, shape = [PPO_config.num_minibatches, -1] + list(x.shape[1:])                     
                    ), 
                    shuffled_batch
                )

                # jax.debug.breakpoint()


                # update over minibatches
                rng, key = jax.random.split(rng)
                train_state, total_loss = jax.lax.scan(
                    __update_over_minibatch, train_state, minibatches #shuffled_batch
                )

                update_state = (train_state, x_init, trajectory_batch, advantages, returns, rng)

                return update_state, total_loss
            
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, PPO_config.update_epochs
            )
            return update_state, loss_info

        def scan_train_step(runner_state, static_model):
            
            def train_step(runner_state, _):
                
                # Store starting hidden state
                x_init = runner_state[3]

                # Do rollout of single trajactory
                runner_state, trajectory_batch = scan_env(runner_state, static_model)

                # calculate gae
                train_state, env_state, last_obs, x, last_done, rng = runner_state

                model = eqx.combine(train_state.dynamic_model, static_model['training'])
                rng, model_key = jax.random.split(rng, 2)
                model_keys = jax.random.split(model_key, last_obs.shape[0])
                _, last_value, _ = jax.vmap(model, in_axes=(0, 0, 0, 0))(last_obs, x, last_done, model_keys)
                
                last_value = jnp.array(jnp.squeeze(last_value), ndmin=1)
                
                def _calculate_gae(carry, transition):
                    gae, next_value, next_done = carry
                    value, reward, done = transition.value, transition.reward, transition.done
                    delta = reward + PPO_config.gamma * next_value * (1 - next_done) - value
                    gae = delta + PPO_config.gamma * PPO_config.gae_lambda * (1 - next_done) * gae

                    return (gae, value, done), (gae, gae + value)

                _, (advantages, returns) = jax.lax.scan(
                    _calculate_gae,
                    (jnp.zeros_like(last_value), last_value, last_done),
                    trajectory_batch,
                    reverse=True,
                    unroll=16
                )
        
                # Do an epoch of updates
                update_state = (train_state, x_init, trajectory_batch, advantages, returns, rng)
                update_state, loss_info = scan_epoch(update_state, static_model)
                
                # Update metrics
                train_state = update_state[0]
                metric = trajectory_batch.info
                metric["loss_info"] = loss_info
                rng = update_state[-1]

                rng, eval_key = jax.random.split(rng)
                eval_rewards = eval_func(train_state, eval_key)
                metric["eval_rewards"] = eval_rewards

                def wandb_callback(info):
                    logger.log({
                        "timestep": info["timestep"][-1][0] * PPO_config.num_envs,
                        "loss_info": jnp.mean(info["loss_info"]),
                        "eval_rewards": info["eval_rewards"]
                    })

                def print_callback(info):
                    print(f'timestep={(info["timestep"][-1][0] * PPO_config.num_envs)}, eval rewards={jnp.mean(info["eval_rewards"])}')
                
                if PPO_config.debug:
                    jax.debug.callback(print_callback, metric)
                if PPO_config.wandb:
                    jax.debug.callback(wandb_callback, metric)

                runner_state = (train_state, env_state, last_obs, x, last_done, rng)
                return runner_state, metric 
            
            runner_state, metrics = jax.lax.scan(
                train_step, runner_state, None, PPO_config.num_iterations
            )
            # runner_state, metrics = train_step(runner_state, None)

            return runner_state, metrics
        
        rng, key = jax.random.split(rng)
        runner_state = (train_state, env_state, obs, x_init, done, key)
        runner_state, metrics = scan_train_step(runner_state, static_model)
        
        return  {
                "runner_state": runner_state, 
                "metrics": metrics,
                "static_model": static_model['training']
        } 

    return train_func

def train(cfg):
    
    # Create environment
    if cfg.env_name == "CartPole-v1":
        env, env_params = gymnax.make(cfg.env_name)
    else: 
        env, env_params = make(cfg.env_name)

    # Create save directory
    save_dir = create_save_dir(cfg)

    # Seed check
    if cfg.PPOConfig.seed is None:
        cfg.PPOConfig.seed = np.random.randint(low=0, high=999999)

    # Store configuration (with seed) in log folder
    OmegaConf.save(config=cfg, f=f"{save_dir}/config.yaml", resolve=True)
    if cfg.PPOConfig.wandb:
        logger = wandb.init(
            entity = "...",
            project = "Thesis",
            config = wandb.helper.parse_config(
                cfg2dict(cfg),
                exclude = [
                "logging",
                "checkpoint",
                "num_reps",
                "PPOConfig.debug"
                ]),
            group = f"{cfg.uid}"
        )
    else:
        logger = None

    start = time.time()
    train_func = create_ppo_train_object(
        env,
        env_params,
        cfg,
        logger
    )

    train_func_jit = eqx.filter_jit(train_func, backend="gpu")

    compile_time = time.time() - start
    if cfg.PPOConfig.debug:     
        jax.debug.print(f"Compile time: {compile_time}")
    out = train_func_jit()

    # Storing metrics
    out["metrics"]["runtime"] = time.time()-start
    out["metrics"]["compile_time"] = compile_time
    if cfg.PPOConfig.debug:
        jax.debug.print(f"Runtime: {out["metrics"]["runtime"]}")
    log_metrics(out["metrics"], save_dir)
    if cfg.PPOConfig.wandb:
        logger.log({
            "runtime": out["metrics"]["runtime"]
        })
        logger.finish()

    # Storing model 
    if cfg.store_model:
        dynamic = out["runner_state"][0].dynamic_model
        static = out["static_model"]
        model = eqx.combine(dynamic, static)
        eqx.tree_serialise_leaves(
            f"{save_dir}/model.eqx", model
        )


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="SSM Suite")
    parser.add_argument(
        "-p", "-cfg", "-path" , type=str, default="", help="Path to yaml config file"
    )
    parser.add_argument(
        "-v", "--verbose", default=False, action=argparse.BooleanOptionalAction, help="Toggles verbose logging mode"
    )
    args = parser.parse_args()    

    for subdir, dirs, files in os.walk(args.p):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(f".yaml"):
                cfg = load_yaml_config(filepath)
                for i in range(cfg.num_reps):
                    cfg = load_yaml_config(filepath) # resets seeding
                    jax.debug.print(f"{i+1}-th run of config {filepath}")
                    train(cfg)

    # cfg = load_yaml_config(args.p)
    # for i in range(cfg.num_reps):
    #     cfg = load_yaml_config(args.p) # resets seeding
    #     train(cfg)
