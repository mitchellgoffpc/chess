#!/usr/bin/env python3
import io
import chess
import chess.svg
import chess.engine
import tkinter as tk
from pathlib import Path
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

class ChessGui(tk.Frame):
  def __init__(self, parent, engine1, engine2, *args, **kwargs):
    tk.Frame.__init__(self, parent, *args, **kwargs)
    self.parent = parent
    self.engine1 = engine1
    self.engine2 = engine2
    self.grid()
    self.chessboard = chess.Board()
    self.canvas = tk.Canvas(self, width=510, height=510, bg="grey")
    self.canvas.grid(row=0, column=0)
    self.canvas.bind("<Button-1>", self.click)
    self.selected_position = None
    self.draw_board()
    self.parent.after(1000, self.move)

  def draw_board(self):
    board_svg = chess.svg.board(board=self.chessboard, fill={self.selected_position: '#cc0000cc'}, size=510)
    drawing = svg2rlg(io.StringIO(board_svg))
    imgdata = io.BytesIO()
    renderPM.drawToFile(drawing, imgdata, fmt='png')
    imgdata.seek(0)
    self.board_photo = tk.PhotoImage(data=imgdata.read())
    self.canvas.create_image(0, 0, anchor='nw', image=self.board_photo)

  def is_human_turn(self):
    return (
      (self.chessboard.turn is chess.WHITE and self.engine1 is None) or
      (self.chessboard.turn is chess.BLACK and self.engine2 is None))

  def move(self):
    if self.chessboard.is_game_over():
      return
    if self.chessboard.turn is chess.WHITE and self.engine1 is not None:
      self.move_engine(self.engine1)
    elif self.chessboard.turn is chess.BLACK and self.engine2 is not None:
      self.move_engine(self.engine2)
    
  def move_engine(self, engine):
    result = engine.play(self.chessboard, chess.engine.Limit(time=1.0))
    self.chessboard.push(result.move)
    self.draw_board()
    self.parent.after(100, self.move)

  def click(self, event):
    x = (event.x - 15) // 60
    y = 7 - (event.y - 15) // 60
    position = chess.square(x, y)
    piece = self.chessboard.piece_at(position)

    if not self.is_human_turn() or self.chessboard.is_game_over() or not (0 <= x < 8 and 0 <= y < 8):
      return
    elif piece and position == self.selected_position:
      self.selected_position = None
    elif self.selected_position is not None:
      move = chess.Move.from_uci('{}{}'.format(chess.square_name(self.selected_position), chess.square_name(position)))
      if move in self.chessboard.legal_moves:
        self.chessboard.push(move)
        self.selected_position = None
      elif piece:
        self.selected_position = position
    elif piece:
      self.selected_position = position

    self.draw_board()
    if not self.is_human_turn():
      self.parent.after(100, self.move)


if __name__ == "__main__":
  import sys

  engine1, engine2 = None, None
  if sys.argv[1] != 'human':
    engine1 = chess.engine.SimpleEngine.popen_uci(sys.argv[1], timeout=20)
  if sys.argv[2] != 'human':
    engine2 = chess.engine.SimpleEngine.popen_uci(sys.argv[2], timeout=20)

  root = tk.Tk()
  ChessGui(root, engine1, engine2)
  root.mainloop()

  if engine1 is not None:
    engine1.quit()
  if engine2 is not None:
    engine2.quit()
