import jax
import jax.tree_util as jtu
import equinox as eqx

import sys
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

# from models.LSSLf_RL import ResidualLSSLfBlockRL, LSSLDiagActorCriticNetwork, LSSLfDiagActorCriticNetwork
from models.FF import FFActorCriticNetwork
from models.GRU import GRUActorCriticNetwork
from models.S5_RL import S5ActorCriticNetwork
from models.minGRU_RL import minGRUActorCriticNetwork

# TODO: fix model initialiser and filter spec for LSSLf-diag model
class LSSLfDiagActorCriticNetwork:
    temp: bool = False
class ResidualLSSLfBlockRL:
    temp: bool = False

def create_filter_spec(model, **modelkwargs):

    if not isinstance(model, LSSLfDiagActorCriticNetwork):
        return eqx.is_array

    is_res_block = lambda x: isinstance(x, ResidualLSSLfBlockRL)

    def _create_ResidualLSSLBlock_filter_spec(res_model):
        filter_spec = jtu.tree_map(lambda _: False, res_model)
        filter_spec = eqx.tree_at(
            lambda tree: (tree.LSSLf.C_mats, tree.LSSLf.D_mats),
            filter_spec,
            replace=(True, modelkwargs['use_feedthrough']), 
        )
        filter_spec = eqx.tree_at(
                    lambda m: (m.linear.weight, m.linear.bias), 
                    filter_spec,
                    replace = (True, True)
                )
        filter_spec = eqx.tree_at(
                    lambda m: (m.layernorm.weight, m.layernorm.bias), 
                    filter_spec,
                    replace = (True, True)
                )
        return filter_spec
    
    temp = eqx.filter_eval_shape(ResidualLSSLfBlockRL, jax.random.PRNGKey(0), **modelkwargs) 
    res_filter_spec = _create_ResidualLSSLBlock_filter_spec(temp)

    filter_spec = jtu.tree_map(lambda _: eqx.is_array(_), model)
    for idx, layer in enumerate(filter_spec.residuallayers):
        if is_res_block(layer):
            filter_spec.residuallayers[idx] = res_filter_spec

    return filter_spec

def initialize_model(cfg, env, env_params, key):
    model_name = cfg.model

    observation_space = env.observation_space(env_params)
    action_space = env.action_space(env_params)
    num_actions = action_space.n
    
    residuallayers = cfg.modelparameters.residuallayers
    actor_hiddenlayers = cfg.modelparameters.actor_hiddenlayers
    critic_hiddenlayers = cfg.modelparameters.critic_hiddenlayers

    # Bit of a hacky way to make sure that H is always a multiple of the amount of features in an observation
    cfg.modelparameters.kernelparameters['H'] = int((cfg.modelparameters.kernelparameters['H'] // observation_space.shape[0]) * observation_space.shape[0])

    if model_name == "LSSLf":
        return NotImplementedError
    
    elif model_name == "LSSLf-diag":
        model = LSSLfDiagActorCriticNetwork(
            key=key, 
            obs_features=observation_space.shape[0],
            num_actions=num_actions,
            num_blocks=residuallayers,
            residualblockparams=cfg.modelparameters.kernelparameters,
            actor_hiddenlayers=actor_hiddenlayers,
            critic_hiddenlayers=critic_hiddenlayers 
        )
        filter_spec = create_filter_spec(model, **cfg.modelparameters.kernelparameters)
            
    elif model_name == "LSSL-diag":
        model = LSSLDiagActorCriticNetwork(
            key=key, 
            obs_features=observation_space.shape[0],
            num_actions=num_actions,
            num_blocks=residuallayers,
            residualblockparams=cfg.modelparameters.kernelparameters,
            actor_hiddenlayers=actor_hiddenlayers,
            critic_hiddenlayers=critic_hiddenlayers 
        )
        filter_spec = create_filter_spec(model, **cfg.modelparameters.kernelparameters)

    elif model_name == "FF":
        model = FFActorCriticNetwork(
            key=key, 
            obs_features=observation_space.shape[0],
            num_actions=num_actions,
            actor_hiddenlayers=actor_hiddenlayers,
            critic_hiddenlayers=critic_hiddenlayers 
        )
        filter_spec = create_filter_spec(model, **cfg.modelparameters.kernelparameters)
    
    elif model_name == "GRU":
        model = GRUActorCriticNetwork(
            key=key, 
            obs_features=observation_space.shape[0],
            num_actions=num_actions,
            num_blocks=residuallayers,
            residualblockparams=cfg.modelparameters.kernelparameters,
            actor_hiddenlayers=actor_hiddenlayers,
            critic_hiddenlayers=critic_hiddenlayers 
        )
        filter_spec = create_filter_spec(model, **cfg.modelparameters.kernelparameters)

    elif model_name == "S5":
        model = S5ActorCriticNetwork(
            key=key, 
            obs_features=observation_space.shape[0],
            num_actions=num_actions,
            num_blocks=residuallayers,
            residualblockparams=cfg.modelparameters.kernelparameters,
            actor_hiddenlayers=actor_hiddenlayers,
            critic_hiddenlayers=critic_hiddenlayers 
        )
        filter_spec = create_filter_spec(model, **cfg.modelparameters.kernelparameters)

    elif model_name == "minGRU":
        model = minGRUActorCriticNetwork(
            key=key, 
            obs_features=observation_space.shape[0],
            num_actions=num_actions,
            num_blocks=residuallayers,
            residualblockparams=cfg.modelparameters.kernelparameters,
            actor_hiddenlayers=actor_hiddenlayers,
            critic_hiddenlayers=critic_hiddenlayers 
        )
        filter_spec = create_filter_spec(model, **cfg.modelparameters.kernelparameters)

    else:
        raise NotImplementedError(f"Modelname {model_name} not found")

    return model, filter_spec