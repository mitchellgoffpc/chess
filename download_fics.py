#!/usr/bin/env python
import bz2
import re
import requests
from tqdm import tqdm
from pathlib import Path


def download_fics(output_fn):
  bz2_fn = Path(f'{output_fn}.bz2')
  output_fn.parent.mkdir(parents=True, exist_ok=True)
  
  data = {
    'gametype': '2',
    'player': '',
    'year': '2022',
    'month': '0',
    'movetimes': '0',
    'download': 'Download',
  }

  headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
    'Referrer': 'https://www.figsgames.org/download.html',
  }

  if not bz2_fn.exists():
    r = requests.post('https://www.ficsgames.org/cgi-bin/download.cgi', data=data, headers=headers)
    r.raise_for_status()

    match = re.search('<a href="(/dl/.*?)">', str(r.content))
    if not match:
      raise Exception(f"Link not found in response: {r.content}")
    data_url = 'https://www.ficsgames.org' + match.group(1)

    r = requests.get(data_url, stream=True)
    r.raise_for_status()
    file_size = int(r.headers['content-length'])
    chunk_size = 1000  # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
    with open(bz2_fn, 'wb') as f:
      with tqdm(desc="Fetching " + data_url, total=file_size, unit_scale=True) as pbar:
        for chunk in r.iter_content(chunk_size=chunk_size):
          f.write(chunk)
          pbar.update(chunk_size)

  if not output_fn.exists():
    print("Extracting...")
    with open(bz2_fn, 'rb') as f:
      data = bz2.decompress(f.read())
    with open(output_fn, 'wb') as f:
      f.write(data)
    print("Done")


if __name__ == '__main__':
  download_fics(Path(__file__).parent / 'data/fics/2022.pgn')
