import gzip
import glob
import base64
from io import BytesIO

for filename in glob.glob('*.7z*'):
    print('encoding {} ...'.format(filename))
    out = BytesIO()
    with open(filename, 'rb') as fr:
        with gzip.GzipFile(fileobj=out, mode="wb") as f:
            f.write(fr.read())
    message = out.getvalue()
    with open('{}.coded'.format(filename), 'wb') as f:
        f.write(base64.b64encode(message))
