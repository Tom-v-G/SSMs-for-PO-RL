import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax  # https://github.com/deepmind/optax
import torch  # https://pytorch.org
import torchvision  # https://pytorch.org
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping
import hippox

import time
import sys
import logging
import json
import os
from functools import partial

#Dataset and plotting
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

# global logger
logger = logging.getLogger(__name__)

def GBT_discretization(A, B, delta):
    I = jnp.identity(A.shape[0])
    inv = jnp.linalg.inv(I - 0.5 * delta * A)
    A = inv @ (I + 0.5 * delta * A)
    B = delta * inv @ B
    return A, B

# TODO: add ZOH discretization
class LSSLfixed(eqx.Module):
    C_mats: jax.Array
    D_mats: jax.Array
    K_mats : jax.Array

    def __init__(self, key, N=128, H=128, dt_log_bounds=(-3, -1), L=784, **kwargs):
        """
        Implementation of the LSSL-f model, assuming single channel (M=1)
        N: Hidden state size 
        H amount of features
        delta_bounds: tuple containing $\Delta t_{\text{min}}$ and $\Delta t_\text{max}}
        sequence_length: Length of the sequence fed to the model (L) 

        initialises with
        C_mats: C matrices for each feature; size (H, N)
        D_mats: D matrices for each feature; size (H,)
        K_mats: Precomputed Krylov matrices; size (H, N, L)
        """

        hippo_params = hippox.basis_measures.hippo_legs(N)
        A = hippo_params.state_matrix
        B = hippo_params.input_matrix
        deltas = jnp.logspace(dt_log_bounds[0], dt_log_bounds[1], H)

        A_mats, B_mats = jax.vmap(lambda delta: GBT_discretization(A, B, delta))(deltas)

        def compute_krylov_matrix(A, B, L):
            mat = jnp.tile(B, (L, 1))
            # mat = mat.astype(jnp.float64)
            _, krylov_matrix = jax.lax.scan(lambda x, y: (A@x, x @ y), jnp.identity(A.shape[0]), mat)
            return krylov_matrix
        
        # Initialise Krylov Matrix
        self.K_mats = jax.vmap(lambda A, B: compute_krylov_matrix(A, B, L=L), out_axes=(1))(A_mats, B_mats)
        # Initialise C and D matrix
        Ckey, Dkey = jax.random.split(key, 2)
        self.C_mats = jax.random.normal(Ckey, (H, N))
        self.D_mats = jax.random.normal(Dkey, (H))

    def __call__(self, u):
        """
        input: 
            u (L, H)
        output:
            y (L, H)
        parameters: 
            K_mats: (L, H, N)
            C_mats: (H, N)
            D_mats: (H,)
        """
        Kl_mats = jax.vmap(lambda C, Km: jax.vmap(lambda K: C @ K)(Km), in_axes=(0, 1), out_axes=(1))(self.C_mats, self.K_mats)
        y = jax.vmap(lambda x, Kl, D, : jnp.convolve(Kl, x, 'same') + D * x,
                           in_axes = (0, 1, 0), 
                           out_axes=(0))(u, Kl_mats, self.D_mats)
        return y   

class Broadcast(eqx.Module):
    dim: int

    def __init__(self, dim):
        self.dim = dim
    
    def __call__(self, x):
        return jnp.broadcast_to(x, (self.dim, *x.shape))
    
class Dropout(eqx.nn.Dropout):
    def __init__(self, *params):
        super().__init__(*params)
    
    def __call__(self, x, key):
        return jax.lax.cond(self.inference,
                     x,
                    super().__call__(x, key=key)
        )

class ResidualLSSLBlock(eqx.Module):
    LSSLf: LSSLfixed
    linear: eqx.nn.Linear
    layernorm: eqx.nn.LayerNorm
    # dropout: Dropout
    inference: bool
    p_dropout: float = eqx.field(static=True)

    def __init__(self, key,  N=128, H=128, dt_log_bounds=(-3, -1), L=784, p_dropout=0.2, use_layernorm=True, use_feedthrough=True, **kwargs):
        """
        Implementation of a LSSLf layer with gelu activation, dropout, layer normalisation, and a residual connection.
        """

        LSSLkey, linearkey = jax.random.split(key, 2)
        
        self.LSSLf = LSSLfixed(key = LSSLkey, N=N, H=H, dt_log_bounds=dt_log_bounds, L=L)
        self.linear = eqx.nn.Linear(H, H, key=linearkey)
        if use_layernorm:
            self.layernorm = eqx.nn.LayerNorm(shape=(H))
        else:
            self.layernorm = eqx.nn.Identity()
        self.inference = False
        self.p_dropout = p_dropout

    # @partial(jax.jit, static_argnames=['inference'])
    def __call__(self, x, key, inference=None):
        h = self.LSSLf(x) # x stored as residual 
        h = jax.nn.gelu(h)

        # Dropout per feature channel
        
        def _dropout(h):
            q = 1 - jax.lax.stop_gradient(self.p_dropout)
            mask = jax.random.bernoulli(key, q, h.shape[0])
            h = jax.vmap(lambda row, mask_row: jnp.where(mask_row, row / q, jax.lax.stop_gradient(jnp.zeros(h.shape[-1]))), in_axes=(0,0))(h, mask)
            return h
        
        def _identity(h):
            return h
        
        if self.inference:
            h = _identity(h)
        
        else:
            h = _dropout(h)

        # h = jax.lax.cond(self.inference,
        #                 _identity,
        #                 _dropout,
        #                 h)

        # if inference is None or self.p_dropout == 0:
        #     inference = self.inference

        # if inference:
        #     h = _dropout(h)
        
        
        # q = 1 - jax.lax.stop_gradient(self.p_dropout)
        # mask = jax.random.bernoulli(key, q, h.shape[0])
        # h = jax.vmap(lambda row, mask_row: jnp.where(mask_row, row / q, jax.lax.stop_gradient(jnp.zeros(h.shape[-1]))), in_axes=(0,0))(h, mask)

        #     # Dropout on element level
        #     q = 1 - jax.lax.stop_gradient(self.dropout_p)
        #     mask = jax.random.bernoulli(key, q, h.shape)
        #     h = jnp.where(mask, h / q, 0)

        # Linear layer
        h = jax.vmap(self.linear, in_axes=(-1), out_axes=(-1))(h)
        # Layernorm
        h = jax.vmap(self.layernorm, in_axes=(-1), out_axes=(-1))(h)

        return h + x

class LSSLClassifier(eqx.Module):

    broadcast: Broadcast
    residuallayers: list
    linear: eqx.nn.Linear
    
    def __init__(self, key, num_blocks, residualblockparams): 
        key, linearkey = jax.random.split(key, 2)
        reskeys = jax.random.split(key, num_blocks)

        self.broadcast = Broadcast(residualblockparams.H)
        self.residuallayers = [ResidualLSSLBlock(reskey, **residualblockparams) for reskey in reskeys]
        self.linear = eqx.nn.Linear(residualblockparams.H, 10, key=linearkey)
    
    def __call__(self, x, key):

        x = self.broadcast(x)

        for i, layer in enumerate(self.residuallayers):
            layerkey, key = jax.random.split(key)
            x = layer(x, layerkey) # (H, L) 

        x = jnp.mean(x, axis=1) # Meanpooling (H,)
        # x = x[:, -1] # Last element (H,)
        x = self.linear(x) # (10)
        x = jax.nn.log_softmax(x)
        return x

class LSSLClassifierTrainer():
    def __init__(self, save_dir, cfg):
        self.epochs = cfg.hyperparameters.epochs
        self.print_every = cfg.logging.print_every
        self.save_dir = save_dir
    
    def train(
        self,
        model,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        optim: optax.GradientTransformation,
        filter_spec, # Specifies which gradients are propagated
        key: jax.Array,
    ):

        # Metrics
        metrics = {
            "train_acc_list": [],
            "train_loss_list": [],
            "test_acc_list":[],
            "test_loss_list":[]
        }

        # Filter optimizer state according to filespec
        opt_state = optim.init(eqx.filter(model, filter_spec))

        @eqx.filter_jit
        def make_step(
            model,
            opt_state: PyTree,
            x: Float[Array, " batch sequencelength "],
            y: Int[Array, " batch"],
            key: jax.Array
        ):
            diff_model, static_model = eqx.partition(model, filter_spec)
            (loss_value, acc_value), grads = jax.value_and_grad(loss, has_aux=True)(diff_model, static_model, x, y, key)

            updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, filter_spec), value=loss_value)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss_value, acc_value

        start = time.time()
        for step in range(self.epochs):
            
            mean_train_acc = []
            mean_train_loss = []
            for i, (x, y) in enumerate(trainloader):
                # PyTorch dataloaders give PyTorch tensors by default,
                # so convert them to NumPy arrays.                
                print(i)
                if i > 10:
                    break
                x = x.numpy()
                y = y.numpy()
                subkey, key = jax.random.split(key, 2)
                model, opt_state, train_loss, train_acc = make_step(model, opt_state, x, y, subkey)
                mean_train_acc.append(train_acc.item())
                mean_train_loss.append(train_loss.item())      
            metrics["train_acc_list"].append(np.mean(mean_train_acc))
            metrics["train_loss_list"].append(np.mean(mean_train_loss))

            # Evaluation
            if (step % self.print_every) == 0 or (step == self.epochs - 1):
                subkey, key = jax.random.split(key, 2)
                # breakpoint()
                model = eqx.nn.inference_mode(model)
                # breakpoint()
                diff_model, static_model = eqx.partition(model, filter_spec)

                x, y = next(iter(testloader))
                x = x.numpy()
                y = y.numpy()

                test_loss, test_accuracy = evaluate(diff_model, static_model, x, y, subkey)
                metrics["test_acc_list"].append(test_accuracy.item())
                metrics["test_loss_list"].append(test_loss.item())
                model = eqx.nn.inference_mode(model, value=False)
                lr_scale = optax.tree_utils.tree_get(opt_state, "scale")
                if lr_scale is None: lr_scale = 1
                
                # Store best model and (all) metrics
                if test_accuracy == np.max(metrics["test_acc_list"]) or (step == self.epochs - 1): 
                    eqx.tree_serialise_leaves(f"{self.save_dir}/LSSLClassifier_{step}.eqx", model)
                json.dump(metrics, open(f"{self.save_dir}/model_metrics.json", 'w'))

                logger.info(
                    f"{step=}, Mean train loss={train_loss.item():.3f}, Mean train accuracy={train_acc.item():.3f}, "
                    f"test_loss={test_loss.item():.3f}, test_accuracy={test_accuracy.item():.3f}, lr scale={lr_scale:.3E}, runtime={time.strftime("%H:%M:%S", time.gmtime(time.time()-start))}"
                )

        return model, metrics
    

def create_filter_spec(model,  N=128, H=128, dt_log_bounds=(-3, -1), L=784, p_dropout=0.2, use_layernorm=True, use_feedthrough=True, **kwargs):
    
    is_res_block = lambda x: isinstance(x, ResidualLSSLBlock)

    def _create_ResidualLSSLBlock_filter_spec(res_model):
        filter_spec = jtu.tree_map(lambda _: False, res_model)
        filter_spec = eqx.tree_at(
            lambda tree: (tree.LSSLf.C_mats, tree.LSSLf.D_mats),
            filter_spec,
            replace=(True, True), 
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
    
    temp = eqx.filter_eval_shape(ResidualLSSLBlock, jax.random.PRNGKey(0), N, H, dt_log_bounds, L, p_dropout, use_layernorm) 
    res_filter_spec = _create_ResidualLSSLBlock_filter_spec(temp)

    filter_spec = jtu.tree_map(lambda _: False, model)
    for idx, layer in enumerate(filter_spec.residuallayers):
        if is_res_block(layer):
            filter_spec.residuallayers[idx] = res_filter_spec
    filter_spec = eqx.tree_at(
                    lambda m: (m.linear.weight, m.linear.bias), 
                    filter_spec,
                    replace = (True, True)
                )
    return filter_spec

def initialise_dataloaders(TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, root_path="data"):

    normalise_sequentialise_data = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize((0.5,), (0.5,)),
            torchvision.transforms.Lambda(lambda x: x.view(-1))
        ]
    )

    training_data = datasets.MNIST(
        root=root_path,
        train=True,
        download=True,
        transform=normalise_sequentialise_data
    )

    test_data = datasets.MNIST(
        root=root_path,
        train=False,
        download=True,
        transform=normalise_sequentialise_data
    )

    trainloader = torch.utils.data.DataLoader(
        training_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True
    )
    testloader = torch.utils.data.DataLoader(
        test_data, batch_size=TEST_BATCH_SIZE, shuffle=True, drop_last=True
    )

    return trainloader, testloader

def cross_entropy(y, pred_y):
        pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
        # print(f"{pred_y}")
        return -jnp.mean(jnp.log(pred_y))

def loss(diff_model, static_model, x, y, key):
    model = eqx.combine(diff_model, static_model)
    predkeys = jax.random.split(key, x.shape[0])
    pred_y = jax.vmap(model, in_axes=(0, 0))(x, predkeys) # Batch dimension at the front

    loss = cross_entropy(y, pred_y)
    accuracy = jnp.mean(y == jnp.argmax(pred_y, axis=1))
    return loss, accuracy

def accuracy(diff_model, static_model, x , y, key):
    model = eqx.combine(diff_model, static_model)
    predkeys = jax.random.split(key, x.shape[0])
    pred_y = jax.vmap(model)(x, predkeys)
    pred_y = jnp.argmax(pred_y, axis=1)
    return jnp.mean(y == pred_y)

@eqx.filter_jit
def evaluate(diff_model, static_model, x, y, key):
    """This function evaluates the model on the test dataset,
    computing both the average loss and the average accuracy.
    """

    return loss(diff_model, static_model, x, y, key)

def train(
    model,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    optim: optax.GradientTransformation,
    steps: int,
    print_every: int,
    filter_spec, # Specifies which gradients are propagated
    key: jax.Array,
    DIR: str # working directory
):

    # Metrics
    metrics = {
        "train_acc_list": [],
        "train_loss_list": [],
        "test_acc_list":[],
        "test_loss_list":[]
    }

    # Filter optimizer state according to filespec
    opt_state = optim.init(eqx.filter(model, filter_spec))

    @eqx.filter_jit
    def make_step(
        model,
        opt_state: PyTree,
        x: Float[Array, " batch sequencelength "],
        y: Int[Array, " batch"],
        key: jax.Array
    ):
        diff_model, static_model = eqx.partition(model, filter_spec)
        (loss_value, acc_value), grads = jax.value_and_grad(loss, has_aux=True)(diff_model, static_model, x, y, key)

        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, filter_spec), value=loss_value)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value, acc_value

    start = time.time()
    for step in range(steps):
        
        mean_train_acc = []
        mean_train_loss = []
        for i, (x, y) in enumerate(trainloader):
            # PyTorch dataloaders give PyTorch tensors by default,
            # so convert them to NumPy arrays.
            x = x.numpy()
            y = y.numpy()
            if i > 3:
                break
            subkey, key = jax.random.split(key, 2)
            model, opt_state, train_loss, train_acc = make_step(model, opt_state, x, y, subkey)
            mean_train_acc.append(train_acc.item())
            mean_train_loss.append(train_loss.item())      
        metrics["train_acc_list"].append(np.mean(mean_train_acc))
        metrics["train_loss_list"].append(np.mean(mean_train_loss))

        # Evaluation
        if (step % print_every) == 0 or (step == steps - 1):
            subkey, key = jax.random.split(key, 2)
            # breakpoint()
            model = eqx.nn.inference_mode(model)
            # breakpoint()
            diff_model, static_model = eqx.partition(model, filter_spec)

            x, y = next(iter(testloader))
            x = x.numpy()
            y = y.numpy()
            test_loss, test_accuracy = evaluate(diff_model, static_model, x, y, subkey)
            metrics["test_acc_list"].append(test_accuracy.item())
            metrics["test_loss_list"].append(test_loss.item())
            model = eqx.nn.inference_mode(model, value=False)
            lr_scale = optax.tree_utils.tree_get(opt_state, "scale")
            if lr_scale is None: lr_scale = 1
            
            # Store best model and (all) metrics
            if test_accuracy == np.max(metrics["test_acc_list"]): 
                eqx.tree_serialise_leaves(f"{DIR}/LSSLClassifier_{step}.eqx", model)
            json.dump(metrics, open(f"{DIR}/model_metrics.json", 'w'))

            print(
                f"{step=}, Mean train loss={train_loss.item():.3f}, Mean train accuracy={train_acc.item():.3f}, "
                f"test_loss={test_loss.item():.3f}, test_accuracy={test_accuracy.item():.3f}, lr scale={lr_scale:.3E}, runtime={time.strftime("%H:%M:%S", time.gmtime(time.time()-start))}"
            )

    return model, metrics

def load_model(key, N=128, H=128, dt_log_bounds=(-3, -1), L=784, p_dropout=0.2, res_layers=6, load_path=None):
        if load_path is None:
            model= LSSLClassifier(key, N, H, dt_log_bounds, L, p_dropout, res_layers)
        else:
            model = eqx.filter_eval_shape(LSSLClassifier, key, N, H, dt_log_bounds, L, p_dropout)
            model = eqx.tree_deserialise_leaves(load_path, model)
        return model


def main():
    # Optimizer parameters
    LEARNING_RATE = 4e-3
    PATIENCE = 10
    COOLDOWN = 5
    FACTOR = 0.2
    RTOL = 1e-4
    ACCUMULATION_SIZE = 1
    MIN_SCALE = 1e-6

    # Dataloader parameters
    TRAIN_BATCH_SIZE = 50
    TEST_BATCH_SIZE = 100

    # LSSLf parameters
    HIDDEN_DIM = 128
    FEATURES = 128
    DELTA_BOUNDS = (-3, -1)
    SEQUENCE_LENGTH = 784
    DROPOUT = 0.2

    # Other hyperparameters
    SEED = 420
    STEPS = 200
    PRINT_EVERY = 1
    # CHECKPOINT = "/home/tom/Documents/Master/Thesis/models/LSSLClassifier.eqx"
    CHECKPOINT = None

    # File paths
    SAVE_PATH = "/home/tom/Documents/Master/Thesis/models"
    SAVE_DIR = time.strftime("%c", time.gmtime(time.time()))

    try: 
        os.mkdir(f"{SAVE_PATH}/{SAVE_DIR}")
    except FileExistsError:
        print(f"Directory '{SAVE_DIR}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{SAVE_DIR}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Initialisation
    key = jax.random.key(SEED)
    print('Initialising dataloaders')
    trainloader, testloader = initialise_dataloaders(TRAIN_BATCH_SIZE, TEST_BATCH_SIZE)

    print('Loading model')
    modelkey, trainkey = jax.random.split(key, 2)
    model = load_model(modelkey, HIDDEN_DIM, FEATURES, DELTA_BOUNDS, SEQUENCE_LENGTH, DROPOUT, load_path=CHECKPOINT)

    model_filter_spec = create_filter_spec(model, HIDDEN_DIM, FEATURES, DELTA_BOUNDS, SEQUENCE_LENGTH, DROPOUT)

    print(model_filter_spec)
    print(eqx.filter(model, model_filter_spec))
    print(f'Parameter count: {sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, model_filter_spec)))}')
    print('Initialising optimizer')
    return 
    # optim = optax.chain(
    #     optax.adamw(
    #         learning_rate=optax.schedules.cosine_decay_schedule(
    #             LEARNING_RATE_START,
    #             STEPS,
    #             LEARNING_RATE_END,
    #         )
    #     )
    # )

    optim = optax.chain(
        optax.adam(LEARNING_RATE),
        # optax.contrib.reduce_on_plateau(
        #     patience=PATIENCE,
        #     cooldown=COOLDOWN,
        #     factor=FACTOR,
        #     rtol=RTOL,
        #     accumulation_size=ACCUMULATION_SIZE,
        #     min_scale=MIN_SCALE
        # ),
    )

    # Training
    print('Starting Training')
    model, metrics = train(model, trainloader, testloader, optim, STEPS, PRINT_EVERY, model_filter_spec, trainkey, f"{SAVE_PATH}/{SAVE_DIR}")

    eqx.tree_serialise_leaves(f"{SAVE_PATH}/{SAVE_DIR}/LSSLClassifier.eqx", model)
    json.dump(metrics, open(f"{SAVE_PATH}/{SAVE_DIR}/model_metrics.json", 'w'))

    # # Testing 
    # x, y = next(iter(testloader))
    # x = x.numpy()
    # y = y.numpy()

    # model = eqx.nn.inference_mode(model, True)
    # predkeys = jax.random.split(trainkey, x.shape[0])
    # pred_y = jax.vmap(model)(x, predkeys)
    # print(jnp.argmax(pred_y, axis=1))
    # print(y)
    # print(jnp.sum(y == jnp.argmax(pred_y, axis=1))/len(y))

if __name__ == "__main__":
    main()