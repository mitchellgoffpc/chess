import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm
from pathlib import Path
from flax.training.train_state import TrainState
from flax.training.checkpoints import save_checkpoint, restore_checkpoint

from models import ResNet
from dataset import ChessDataset, DataLoader

dataset = ChessDataset()
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8)

model = ResNet(50, outputs=64*144)
rng = jax.random.PRNGKey(0)
params = model.init(rng, jnp.ones((1, 8, 8), dtype=jnp.uint8))
optimizer = optax.adam(3e-4)
state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

checkpoint_dir = Path(__file__).parent / 'checkpoints'
checkpoint_dir.mkdir(exist_ok=True)

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
running_loss = None
for epoch in range(10):
  for batch in (pbar := tqdm(dataloader)):
    state, loss = train_step(state, batch)
    running_loss = .99*(running_loss or loss) + .01*loss
    pbar.set_description(f"Epoch {epoch+1} | Loss: {running_loss:.3f}")

  save_checkpoint(ckpt_dir=checkpoint_dir, target=state, step=epoch)
