import gzip
import base64
import glob
from io import BytesIO
from gzip_test_encode import ext

if __name__ == '__main__':

    for filename in glob.glob('*.7z.{}'.format(ext)):
        with open(filename, 'rb') as f:
            payload = f.read()
        message_gzip = base64.b64decode(payload)
        instr = BytesIO(message_gzip)
        with gzip.GzipFile(fileobj=instr, mode="rb") as f:
            message_raw = f.read()
        restored_filename = '{}.7z'.format(filename)
        print('restored {} ...'.format(restored_filename))
        with open(restored_filename, 'wb') as f:
            f.write(message_raw)
