import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm
from flax.training.train_state import TrainState
from models import ResNet
from dataset import ChessDataset, DataLoader

dataset = ChessDataset()
dataloader = DataLoader(dataset, batch_size=32)

model = ResNet(18, outputs=64*144)
rng = jax.random.PRNGKey(0)
params = model.init(rng, jnp.ones((1, 8, 8), dtype=jnp.uint8))
optimizer = optax.adam(1e-3)
state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        inputs, _, targets = batch
        preds = model.apply(params, inputs)
        targets = jnp.reshape(targets, (-1, 64*144))
        preds = jnp.reshape(preds, (-1, 64*144))
        loss = optax.softmax_cross_entropy(preds, targets).mean()
        return loss, loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, loss), grads = grad_fn(state.params)
    return state.apply_gradients(grads=grads), loss

# Training loop
for epoch in range(10):
    for batch in tqdm(dataloader):
        state, loss = train_step(state, batch)
    print(f'Epoch {epoch+1}, Loss: {loss}')
