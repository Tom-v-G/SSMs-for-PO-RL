# SSMs-for-PO-RL
The repository contains equinox based JAX implementations of S5 (and LSSLf) SSMs, together with feedforward, GRU and minGRU baselines, which can be run on partially observable reinforcement learning environments. The codebase includes a PPO algorithm suitable for recurrent architectures, environments from the popjaxRL suite and a JAX implementation of the memory corridor environment. The notebooks folder contains the code used to create the figures for my thesis with examples of how data from runs can be retrieved and analyzed. 

## Running the code
To codebase uses `.yaml` configuration files to store hyperparameters. The codebase is set up to run batches of these configurations from (sub)folders in the `configs` directory. 
Below is an example of a bash command runnning PPO for all models, environments and hyperparameters found in the `exampleconfigs` folder

```bash 
python algorithms/PPO.py -cfg configs/exampleconfigs
```


# :warning: Read Before Running :warning:
In the current version of the code the parameter labels of equinox models created by optax are callables, whereas optax should treat them as non-callable. 
To run the code minor changes to optax are required. 

regel 233 en 254 van transforms._combining.py (equinox modellen tellen als callable, terwijl dit niet de bedoeling is).
Ook heb ik een keep_params_negative functie toegevoegd in transforms._constraining.py.

Line 233 of `optax.transforms._combining.py` should be commented out
```python
 def init_fn(params):
    # labels = param_labels(params) if callable(param_labels) else param_labels
    labels = param_labels
    label_set = set(jax.tree.leaves(labels))
    ...
```

The same for line 254
```python
 def update_fn(updates, state, params=None, **extra_args):
    # labels = param_labels(updates) if callable(param_labels) else param_labels
    labels = param_labels
    new_inner_state = {}
    ...
```

Also, I have added a function to constrain the parameters of the state matrix to the negative real plane in `optax.transforms._constraining.py`. A smarter solution (which is left as a todo) is to apply an activation function with a restricted domain (e.g. $$\sigma(x) = -e^{x}$$) to the matrix values before using the matrix in parallel scan computations.  

```python
def keep_params_negative() -> base.GradientTransformation:
  """Modifies the updates to keep parameters negative, i.e. <= 0.

  This transformation ensures that parameters after the update will be
  smaller than or equal to zero.
  In a chain of transformations, this should be the last one.

  Returns:
    A :class:`optax.GradientTransformation` object.

  .. warning::
    The transformation expects input params to be negative.
    When params is non-negative the transformed update will move them to 0.
  """

  def init_fn(params):
    del params
    return NonNegativeParamsState()

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)

    updates = jax.tree.map(
        lambda p, u: None if p is None else jnp.where((p + u) > 0.0, -p - 0.000001, u),
        params,
        updates,
        is_leaf=lambda x: x is None,
    )
    return updates, state

  return base.GradientTransformation(init_fn, update_fn)
```
