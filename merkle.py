'''

Generate root merkle tree hash in python.

I use https://github.com/bitcoin/bitcoin as reference:
    BlockBuildMerkleTree --> Satoshi implmentation
    BlockMerkleRoot ---> new bitcoin core implementation

'''
import pandas as pd
from hashlib import sha256
from io import StringIO

# h( h(1) + h(2) )
# 0df4085b3a65bd26ca6ab608c0f70c41213f77e56bc5b33bd9899db5d39a7cd8

# h( h(3) + h(4) )
# b26c7b49a69fe9a789facdaaad0af0bac4cd588db345d297f03359a5e40d73d2

# h( h( h(1) + h(2) ) + h( h(3) + h(4) ) )
# 93b46a24b0a418c5f6c31b4058dc5d0f3338a30951d3b4b5a74e9072f145c766

dataset = StringIO("""\
transaction1_serialized_A_B_3
transaction2_serialized_B_C_1
transaction3_serialized_D_E_2
transaction4_serialized_E_B_1
transaction5_serialized_C_B_2
transaction6_serialized_D_A_1
""")

df = pd.read_csv(dataset, encoding='utf-8', header=None)
hashes = df.iloc[:, 0].apply(lambda x: sha256(x.encode('utf-8')).hexdigest()).tolist()

while len(hashes) > 1:

    if len(hashes) % 2 != 0:
        hashes.append(hashes[-1])

    i = 0
    j = 0
    while i + 1 < len(hashes):
        hashes[j] = sha256(str(hashes[i] + hashes[i + 1]).encode('utf-8')).hexdigest()
        i += 2
        j += 1

    hashes = hashes[:int(len(hashes) / 2)]


# tree condensed in a hash
print(hashes[0])
