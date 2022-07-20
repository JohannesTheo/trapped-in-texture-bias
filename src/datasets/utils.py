import requests
from tqdm import tqdm

# download large files with a tqdm progress bar
# https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51
def download(url, fname, params={}):

    resp = requests.get(url, params = params, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
            desc=fname,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
