import gzip
import base64
from StringIO import StringIO

with open('jenkins.7z.encoded', 'rb') as f:
    payload = f.read()
message_gzip = base64.b64decode(payload)
instr = StringIO(message_gzip)
with gzip.GzipFile(fileobj=instr, mode="rb") as f:
    message_raw = f.read()
with open('jenkins_restored.7z', 'wb') as f:
    f.write(message_raw)
