import io
import sys
import chess
import chess.pgn
import multiprocessing
from tqdm import tqdm
from pathlib import Path

def get_value(board, outcome):
  if outcome and outcome.termination is chess.Termination.CHECKMATE:
    return 1 if board.turn is outcome.winner else -1
  elif outcome:
    return 0
  else:
    return 'nan'

def save_game_data(args):
  output_path, pgn = args
  game = chess.pgn.read_game(io.StringIO(pgn))
  boards, moves = [], []
  while True:
    board = game.board()
    game = game.next()
    if game is None:
      break
    boards.append(board)
    moves.append(game.move)

  outcome = board.outcome()
  with open(output_path, 'w') as f:
    f.write(f"{len(boards)}\n")
    for board, move in zip(boards, moves):
      f.write(f"{board.fen()} | {move.uci()} | {get_value(board, outcome)}\n")

  return None

def read_pgns(pgn_fn):
  output_path = Path(pgn_fn).parent / 'fens'
  output_path.mkdir(exist_ok=True)
  game = []
  game_id = 0

  with open(pgn_fn) as f:
    for line in f:
      game.append(line)
      if line.startswith('1.'):
        yield str(output_path / f'{game_id:05d}.fen'), ''.join(game)
        game = []
        game_id += 1


if __name__ == '__main__':
  assert len(sys.argv) >= 2, "usage: preprocess_data.py <pgn-path>"
  with multiprocessing.Pool(24) as pool:
    list(tqdm(pool.imap(save_game_data, read_pgns(sys.argv[1]))))
