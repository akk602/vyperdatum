"""Seekable read-only file over HTTP range requests, so zipfile can list/extract single members."""
import io, sys, urllib.request, zipfile

URL = "https://zenodo.org/records/18893503/files/vyper_grids.zip?download=1"

class HttpFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos
    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        if n == 0 or self.pos >= self.size: return b''
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={'Range': f'bytes={self.pos}-{end}'})
        data = urllib.request.urlopen(req).read()
        self.pos += len(data)
        return data
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)

def open_zip():
    head = urllib.request.urlopen(urllib.request.Request(URL, method='HEAD'))
    size = int(head.headers['Content-Length'])
    return zipfile.ZipFile(io.BufferedReader(HttpFile(URL, size), buffer_size=1 << 20))

if __name__ == '__main__':
    z = open_zip()
    for i in z.infolist():
        print(f'{i.file_size/1e6:10.1f} MB  {i.compress_size/1e6:10.1f} MB zipped  {i.filename}')
