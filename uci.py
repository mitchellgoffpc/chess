#!/usr/bin/env python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import chess
import jax.numpy as jnp
from pathlib import Path
from flax.training.checkpoints import restore_checkpoint
from dataset import get_board_state, get_move_for_action, get_legal_moves_mask
from models import ResNet


def main():
  board = chess.Board()
  model = ResNet(50, outputs=64*144)
  checkpoint = restore_checkpoint(ckpt_dir=Path(__file__).parent / 'checkpoints', target=None)
  model.apply(checkpoint['params'], jnp.zeros((1, 8, 8), dtype=jnp.uint8))

  while True:
    args = input().split()

    if not args:
      continue

    elif args[0] in ("quit", "stop"):
      break

    elif args[0] == "uci":
      print("uciok")

    elif args[0] == "isready":
      print("readyok")

    elif args[0] == "ucinewgame":
      board = chess.Board()

    elif args[:2] == ["position", "startpos"]:
      board = chess.Board()
      for arg in args[3:]:
        board.push(chess.Move.from_uci(arg))

    elif args[:2] == ["position", "fen"]:
      board = chess.Board(fen=' '.join(args[2:8]))

    elif args[0] == "go":
      logits = model.apply(checkpoint['params'], get_board_state(board)[None])[0]
      logits = jnp.exp(logits) * get_legal_moves_mask(board)
      move = get_move_for_action(logits.argmax())
      board.push(move)
      print("bestmove", move.uci())

    else:
      raise NotImplementedError(args)


if __name__ == "__main__":
  main()
