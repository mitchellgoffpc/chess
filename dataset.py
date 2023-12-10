import chess
import numpy as np
from tqdm import tqdm
from pathlib import Path
from jax.tree_util import tree_map
from torch.utils.data import Dataset, DataLoader as TorchDataLoader, default_collate

COLORS = [chess.WHITE, chess.BLACK]
PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

PIECE_NUMBERS = {
  (color, piece): i*6+j+1
  for i, color in enumerate(COLORS)
  for j, piece in enumerate(PIECES)}

PROMOTIONS = [
  (square, piece)
  for square in chess.SQUARES if chess.square_rank(square) in (0, 7)
  for piece in PIECES if piece is not chess.PAWN]
NON_PROMOTIONS = [(square, None) for square in chess.SQUARES]
TO_SQUARES = {square: i for i, square in enumerate(NON_PROMOTIONS + PROMOTIONS)}


def numpy_collate(batch):
  return tree_map(np.asarray, default_collate(batch))

class DataLoader(TorchDataLoader):
  def __init__(self, *args, **kwargs):
    kwargs['collate_fn'] = numpy_collate
    super().__init__(*args, **kwargs)

class ChessDataset(Dataset):
  def __init__(self):
    self.fen_paths = list((Path(__file__).parent / 'data/fics/fens').iterdir())
    self.game_indices = []
    self.game_start_indices = []
    for game_idx, fn in enumerate(tqdm(self.fen_paths, desc='Loading FENs')):
      self.game_start_indices.append(len(self.game_indices))
      with open(fn) as f:
        self.game_indices.extend([game_idx] * int(f.readline()[:-1]))

  def __len__(self):
    return len(self.game_indices)

  def __getitem__(self, idx):
    game_idx = self.game_indices[idx]
    game_start_idx = self.game_start_indices[game_idx]
    with open(self.fen_paths[game_idx]) as f:
      lines = f.read().split('\n')[1:]
      fen, uci = lines[idx - game_start_idx].split(' | ')
      board = chess.Board(fen)
      move = chess.Move.from_uci(uci)

    board_np = np.zeros((8, 8), dtype=np.uint8)
    for square, piece in board.piece_map().items():
      board_np[square // 8, square % 8] = PIECE_NUMBERS[piece.color, piece.piece_type]
    legal_moves = np.zeros((64, len(TO_SQUARES)), dtype=bool)
    for legal_move in board.legal_moves:
      legal_moves[legal_move.from_square, TO_SQUARES[legal_move.to_square, legal_move.promotion]] = True
    gt_move = np.zeros((64, len(TO_SQUARES)), dtype=bool)
    gt_move[move.from_square, TO_SQUARES[move.to_square, move.promotion]] = True

    return board_np, legal_moves, gt_move


if __name__ == '__main__':
  ds = ChessDataset()
  ldr = DataLoader(ds, batch_size=32)

  x, y, z = ds[0]
  print(x.shape, y.shape, z.shape)

  for x, y, z in tqdm(ldr):
    pass
