import gzip
import base64
from StringIO import StringIO

out = StringIO()
with open('red_pandas.7z', 'rb') as fr:
    with gzip.GzipFile(fileobj=out, mode="wb") as f:
        f.write(fr.read())
message = out.getvalue()
with open('red_pandas.7z.encoded', 'wb') as f:
    f.write(base64.b64encode(message))
