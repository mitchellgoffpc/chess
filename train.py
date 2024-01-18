import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm
from pathlib import Path
from flax.training.train_state import TrainState
from flax.training.checkpoints import save_checkpoint, restore_checkpoint

from models import ChessModel
from dataset import ChessDataset, DataLoader

dataset = ChessDataset()
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8)

rng = jax.random.PRNGKey(0)
params = model.init(rng, jnp.ones((1, 8, 8), dtype=jnp.uint8))
optimizer = optax.adam(3e-4)
state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

checkpoint_dir = Path(__file__).parent / 'checkpoints'
checkpoint_dir.mkdir(exist_ok=True)

@jax.jit
def train_step(state, batch):
  def loss_fn(params):
    inputs, _, actions, values = batch
    value_preds, policy_preds = model.apply(params, inputs)
    actions = jnp.reshape(actions, (-1, 64*144))
    values = jnp.reshape(values, (-1, 1))
    value_preds = jnp.reshape(value_preds, (-1, 1))
    policy_preds = jnp.reshape(policy_preds, (-1, 64*144))

    # ugh, need the double tf.where trick to avoid nans in gradient
    value_mask = ~jnp.isnan(values)
    value_loss = optax.l2_loss(value_preds, jnp.where(value_mask, values, 0))
    value_loss = jnp.where(value_mask, value_loss, 0)
    policy_loss = optax.softmax_cross_entropy(policy_preds, actions)
    loss = value_loss.mean() + policy_loss.mean()
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
