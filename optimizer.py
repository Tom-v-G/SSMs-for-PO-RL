import optax
from optax.transforms._combining import partition
import jax.tree_util as jtu
import equinox as eqx

from models.core.S5 import ResidualS5BlockRL
from models.core.LSSLf_diag import ResidualLSSLfDiagBlock

def set_optimizer(cfg):
    if cfg.optimizer.reduce_on_plateau.enabled:
            optim = optax.chain(
                optax.adam(float(cfg.optimizer.learning_rate)),
                optax.contrib.reduce_on_plateau(
                    patience=cfg.optimizer.reduce_on_plateau.patience,
                    cooldown=cfg.optimizer.reduce_on_plateau.cooldown,
                    factor=float(cfg.optimizer.reduce_on_plateau.factor),
                    rtol=float(cfg.optimizer.reduce_on_plateau.rtol),
                    accumulation_size=cfg.optimizer.reduce_on_plateau.accumulation_size,
                    min_scale=float(cfg.optimizer.reduce_on_plateau.min_scale)
                    )
            )
    else:
        optim = optax.chain(optax.adam(cfg.optimizer.learning_rate))
    return optim


def assign_optimizer_labels(model, cfg):
    """
    Create model specific optimizer labels for optax.
    """
    
    if cfg.model == "S5":
        is_res_block = lambda x: isinstance(x, ResidualS5BlockRL)

        optimizer_labels = jtu.tree_map(lambda _: "trainable" if eqx.is_array(_) else "non-trainable", model)
        for idx, layer in enumerate(optimizer_labels.residuallayers):
            if is_res_block(layer):
                new_layer = eqx.tree_at(
                    lambda tree: (tree.SSM.Lambda, tree.SSM.deltas, tree.SSM.B, tree.SSM.C),
                    layer,
                    replace=("Lambda", "deltas", "B", "C"), 
                )
                optimizer_labels.residuallayers[idx] = new_layer

    elif cfg.model == "LSSLf-diag":
        is_res_block = lambda x: isinstance(x, ResidualLSSLfDiagBlock)

        optimizer_labels = jtu.tree_map(lambda _: "trainable" if eqx.is_array(_) else "non-trainable", model)
        for idx, layer in enumerate(optimizer_labels.residuallayers):
            if is_res_block(layer):
                new_layer = eqx.tree_at(
                    lambda tree: (tree.SSM.Lambdas, tree.SSM.B_mats),
                    layer,
                    replace=("non-trainable", "non-trainable"), 
                )
                optimizer_labels.residuallayers[idx] = new_layer
                
    return optimizer_labels


def initialize_optimizer(model, cfg):
    if cfg.PPOConfig.anneal_lr:
        num_iterations = int(float(cfg.PPOConfig.total_timesteps)) // int(cfg.PPOConfig.num_steps) // int(cfg.PPOConfig.num_envs)
        schedule = optax.schedules.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=1.0,
            warmup_steps=(cfg.PPOConfig.num_minibatches * cfg.PPOConfig.update_epochs),
            decay_steps= num_iterations * (cfg.PPOConfig.num_minibatches * cfg.PPOConfig.update_epochs), # correct for batches
            end_value=0
        )
    else:
        schedule = optax.schedules.constant_schedule(1) # No learning rate annealing
    
    if cfg.model in ["GRU", "FF", "minGRU"]:
        optim = optax.chain(
            optax.clip_by_global_norm(cfg.PPOConfig.max_grad_norm),
            optax.adam(cfg.PPOConfig.learning_rate),
            optax.scale_by_schedule(schedule),
        )

        return optim

    param_labels = assign_optimizer_labels(model, cfg)
    
    if cfg.model == "S5":
       
        delta_transform = optax.chain(
            optax.clip_by_global_norm(cfg.PPOConfig.max_grad_norm),  # Clip total gradient norm to max of 1.0
            optax.adam(cfg.PPOConfig.ssm_lr),
            optax.scale_by_schedule(schedule),
            optax.keep_params_nonnegative()
            
        )

        Lambda_transform = optax.chain(
            optax.clip_by_global_norm(cfg.PPOConfig.max_grad_norm), # Clip total gradient norm to max of 1.0
            optax.adam(cfg.PPOConfig.ssm_lr),
            optax.scale_by_schedule(schedule),
            optax.keep_params_negative()
        )

        B_transform = optax.chain(
            optax.clip_by_global_norm(cfg.PPOConfig.max_grad_norm),  # Clip total gradient norm to max of 1.0
            optax.adam(cfg.PPOConfig.ssm_lr),
            optax.scale_by_schedule(schedule)
        )

        C_transform = optax.chain(
            optax.clip_by_global_norm(cfg.PPOConfig.max_grad_norm),  # Clip total gradient norm to max of 1.0
            optax.adamw(cfg.PPOConfig.learning_rate,
                        weight_decay=0.01),
            optax.scale_by_schedule(schedule)
        )

        train_transform =  optax.chain(
            optax.clip_by_global_norm(cfg.PPOConfig.max_grad_norm), # Clip total gradient norm to max of 1.0
            optax.adamw(cfg.PPOConfig.learning_rate,
                        weight_decay=0.01),
            optax.scale_by_schedule(schedule)
        )

        non_train_transform = optax.chain(
            optax.sgd(learning_rate=0.0)
        )

        optim = partition(
            {"Lambda": Lambda_transform,
            "deltas": delta_transform,
            "B": B_transform,
            "C": C_transform,
            "trainable": train_transform,
            "non-trainable": non_train_transform
            }, 
            param_labels
        )
    
    elif cfg.model == "LSSLf-diag":

        train_transform =  optax.chain(
            optax.clip_by_global_norm(cfg.PPOConfig.max_grad_norm),  # Clip total gradient norm to max of 1.0
            optax.adamw(cfg.PPOConfig.learning_rate,
                        weight_decay=0.01),
            optax.scale_by_schedule(schedule)
        )

        non_train_transform = optax.chain(
            optax.sgd(learning_rate=0.0)
        )

        optim = partition(
            {
            "trainable": train_transform,
            "non-trainable": non_train_transform
            }, 
            param_labels
        )
        
    else:
        raise NotImplementedError
    
    return optim
