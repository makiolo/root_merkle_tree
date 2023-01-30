import gzip
import glob
import base64
from io import BytesIO

ext = 'coded'

if __name__ == '__main__':
    for filename in glob.glob('*.7z*'):
        if filename.find(ext) != -1:
            # skip
            continue
        print('encoding {} ...'.format(filename))
        out = BytesIO()
        with open(filename, 'rb') as fr:
            with gzip.GzipFile(fileobj=out, mode="wb") as f:
                f.write(fr.read())
        message = out.getvalue()
        with open('{}.{}'.format(filename, ext), 'wb') as f:
            f.write(base64.b64encode(message))
