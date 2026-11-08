import os, shutil, sys
sys.path.insert(0, os.path.dirname(__file__))
from remotezip import open_zip

out = sys.argv[1]
names = sys.argv[2:]
z = open_zip()
for name in names:
    member = 'vyper_grids/' + name
    dest = os.path.join(out, name)
    if os.path.exists(dest):
        print('have', name); continue
    with z.open(member) as src, open(dest + '.part', 'wb') as dst:
        shutil.copyfileobj(src, dst, 8 << 20)
    os.replace(dest + '.part', dest)
    print('got ', name, os.path.getsize(dest))
