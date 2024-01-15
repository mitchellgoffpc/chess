import sys
import chess
import chess.engine

def main():
  engine1 = chess.engine.SimpleEngine.popen_uci(sys.argv[1], timeout=20)
  engine2 = chess.engine.SimpleEngine.popen_uci(sys.argv[2], timeout=20)
  board = chess.Board()

  while not board.is_game_over():
    result = engine1.play(board, chess.engine.Limit(time=1.0))
    board.push(result.move)
    result = engine2.play(board, chess.engine.Limit(time=1.0))
    board.push(result.move)

  print(board.outcome())
  engine1.quit()
  engine2.quit()


if __name__ == "__main__":
  main()
