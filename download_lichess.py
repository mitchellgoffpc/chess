#!/usr/bin/env python
import sys
import re
import requests
from tqdm import tqdm
from pathlib import Path
from zstandard import ZstdDecompressor


def download_lichess(output_fn, year, month):
  zstd_fn = Path(f'{output_fn}.zst')
  output_fn.parent.mkdir(parents=True, exist_ok=True)

  if not zstd_fn.exists():
    data_url = f'https://database.lichess.org/standard/lichess_db_standard_rated_{year}-{month}.pgn.zst'
    r = requests.get(data_url, stream=True)
    r.raise_for_status()
    file_size = int(r.headers['content-length'])
    chunk_size = 1000  # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
    with open(zstd_fn, 'wb') as f:
      with tqdm(desc="Fetching " + data_url, total=file_size, unit_scale=True) as pbar:
        for chunk in r.iter_content(chunk_size=chunk_size):
          f.write(chunk)
          pbar.update(chunk_size)

  if not output_fn.exists():
    print("Extracting...")
    ctx = ZstdDecompressor()
    with open(zstd_fn, 'rb') as f_in, open(output_fn, 'wb') as f_out:
      ctx.copy_stream(f_in, f_out)

  print("Done")


if __name__ == '__main__':
  if len(sys.argv) < 3:
    raise RuntimeError("usage: download_lichess.py <year> <month>")
  year, month = sys.argv[1], sys.argv[2]
  download_lichess(Path(__file__).parent / f'data/lichess/{year}-{month}.pgn', year, month)
