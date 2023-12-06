import chess.pgn
import numpy as np
from pathlib import Path
from jax.tree_util import tree_map
from torch.utils.data import Dataset, DataLoader as TorchDataLoader, default_collate

def numpy_collate(batch):
  return tree_map(np.asarray, default_collate(batch))

class DataLoader(TorchDataLoader):
  def __init__(self, *args, **kwargs):
    kwargs['collate_fn'] = numpy_collate
    super().__init__(*args, **kwargs)

class ChessDataset(Dataset):
  def __init__(self):
    pgn_path = Path(__file__).parent / 'data/fics/2022.pgn'
    self.games = []
    with open(pgn_path) as f:
      while (game := chess.pgn.read_game(f)) is not None:
        self.games.append(game)

  def __len__(self):
    return 1 # sum(len(game) for game in games)

  def __getitem__(self, idx):
    return np.array(10, dtype=np.float32)


if __name__ == '__main__':
  from tqdm import tqdm
  ds = ChessDataset()
  ldr = DataLoader(ds, batch_size=32)

  for x in tqdm(ldr):
    print(x.shape)
