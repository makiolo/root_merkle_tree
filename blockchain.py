'''
requirements:

base58==1.0.0
ecdsa==0.13

'''
import os
import hashlib
import binascii
import base58
import ecdsa
import time
import itertools
from collections import defaultdict


def generate_private_key():
    return binascii.hexlify(os.urandom(32)).decode('utf-8')


def private_key_to_WIF(private_key):
    var80 = "80" + str(private_key)
    var = hashlib.sha256(binascii.unhexlify(hashlib.sha256(binascii.unhexlify(var80)).hexdigest())).hexdigest()
    return str(base58.b58encode(binascii.unhexlify(str(var80) + str(var[0:8]))), 'utf-8')


def private_key_to_public_key(private_key):
    sign = ecdsa.SigningKey.from_string(binascii.unhexlify(private_key), curve = ecdsa.SECP256k1)
    return '04' + binascii.hexlify(sign.verifying_key.to_string()).decode('utf-8')


def public_key_to_address(public_key):
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    count = 0; val = 0
    var = hashlib.new('ripemd160')
    var.update(hashlib.sha256(binascii.unhexlify(public_key.encode())).digest())
    doublehash = hashlib.sha256(hashlib.sha256(binascii.unhexlify(('00' + var.hexdigest()).encode())).digest()).hexdigest()
    address = '00' + var.hexdigest() + doublehash[0:8]
    for char in address:
        if (char != '0'):
            break
        count += 1
    count = count // 2
    n = int(address, 16)
    output = []
    while (n > 0):
        n, remainder = divmod (n, 58)
        output.append(alphabet[remainder])
    while (val < count):
        output.append(alphabet[0])
        val += 1
    return ''.join(output[::-1])


class Blockchain:
    def __init__(self, seed, bits):
        self.blocks = []
        self.make_genesis(seed, bits=bits)

    def __iter__(self):
        return iter(self.blocks)

    def make_genesis(self, seed, bits):
        genesis = Block(self.hash(), '0', [], seed, bits=bits)
        self.blocks.append(genesis)

    def last(self):
        return self.blocks[-1]

    def hash(self):
        hashes = []
        for block in self.blocks:
            hashes.append(block.hash())
        return Blockchain.calculate_root_merkle_hash(hashes)

    def make_block(self, facts, bits):
        return Block(self.hash(), self.last().hash(), facts, 0, bits=bits)

    def accept_block(self, block):
        self.blocks.append(block)

    @staticmethod
    def calculate_root_merkle_hash(hashes):
        while len(hashes) > 1:
            if len(hashes) % 2 != 0:
                hashes.append(hashes[-1])
            i = 0
            j = 0
            while i + 1 < len(hashes):
                hashes[j] = hashlib.sha256(str(hashes[i] + hashes[i + 1]).encode('utf-8')).hexdigest()
                i += 2
                j += 1
            hashes = hashes[:int(len(hashes) / 2)]
        if len(hashes) > 0:
            return hashes[0]
        else:
            return '{:064d}'.format(0)

    def valid(self):
        '''
        check order and integrity

        :return:
        '''
        i = 0
        while i + 1 < len(self.blocks):
            if self.blocks[i].version != self.blocks[i+1].version:
                return False
            if self.blocks[i].hash() != self.blocks[i+1].prev_hash:
                return False
            if self.blocks[i].merkle_root == self.blocks[i+1].merkle_root:
                return False
            if self.blocks[i].timestamp >= self.blocks[i + 1].timestamp:
                return False
            if not self.blocks[i].hash().startswith(self.blocks[i].target):
                return False
            if not self.blocks[i+1].hash().startswith(self.blocks[i+1].target):
                return False
            i += 2
        return True


class Block:
    def __init__(self, merkle_root, prev_hash, transactions, nonce, bits):
        self.version = 0x02000000
        self.prev_hash = prev_hash
        self.merkle_root = merkle_root
        self.target = "0" * bits
        self.nonce = nonce
        self.timestamp = time.time()
        self.transactions = transactions

    def __str__(self):
        return f'{self.version}{self.prev_hash}{self.merkle_root}{self.target}{self.nonce}{self.transactions}'

    def hash(self):
        return hashlib.sha256(self.__str__().encode()).hexdigest()

    def check_nonce(self, nonce, bits):
        self.nonce = nonce
        guess_hash = self.hash()
        return guess_hash[:bits] == self.target

    def pow(self, bits):
        '''
        calculate nonce

        :param bits:
        :return:
        '''
        for nonce in itertools.count():  # inifinite generator by brute force
            if self.check_nonce(nonce, bits):
                return nonce


def sign_message(private_key, message):
    sk = ecdsa.SigningKey.from_string(binascii.unhexlify(private_key), curve=ecdsa.SECP256k1)
    vk = sk.get_verifying_key()
    message = message.encode('utf-8')
    sig = sk.sign(message)
    sign = binascii.hexlify(sig)
    result = vk.verify(binascii.unhexlify(sign), message)  # True
    if result:
        return sign.decode('utf-8')
    else:
        raise Exception('Error signing "{}".'.format(message))


def verify_message(public_key, message, sign):
    assert(public_key[:2] == '04')  # uncompressed public key
    vk = ecdsa.VerifyingKey.from_string(binascii.unhexlify(public_key[2:]), curve=ecdsa.SECP256k1)
    result = vk.verify(binascii.unhexlify(sign), message)
    return result


class Transaction:
    def __init__(self, from_, to_, qty_):
        self._from = from_
        self._to = to_
        self._qty = qty_

    def __str__(self):
        return '<data from="{}" to="{}" qty="{:.8f}" />'.format(self._from, self._to, self._qty)


class Wallet:
    def __init__(self, address):
        self.address = address

    def endpoint(self):
        return self.address


class BusBlock:
    def __init__(self):
        self.transactions = []

    def send(self, from_wallet, to_wallet, qty_):
        trans = str(Transaction(from_wallet.endpoint(), to_wallet.endpoint(), qty_))
        self.transactions.append((trans, sign_message(from_wallet.private, trans), from_wallet.public))

    def flush(self, from_wallet):
        '''
        Generate block data
        '''
        message = ''
        for trans, sign, public in self.transactions:
            message += '<transaction>\n'
            message += '{}\n'.format(trans)
            message += '<sign>{}</sign>\n'.format(sign)
            message += '<public>{}</public>\n'.format(public)
            message += '</transaction>\n'
        message = message.encode('utf-8')
        return '''<transactions>
    <body>
{}
    </body>
</transactions>
        '''.format(message.decode('utf-8'))

    def validate(self, blockchain):
        '''
        TODO
        '''
        # check balances positives (after apply transactions)
        accounts = defaultdict(float)
        for block in blockchain.blocks:
            print('###############')
            # UNSERIALIZE
            # assert all balance >= 0
            # check transaction is valid (using criptography)
            print(block.transactions)
            print('###############')


class LocalWallet(Wallet):
    def __init__(self):
        self.private = generate_private_key()
        self.public = private_key_to_public_key(self.private)
        self.import_key = private_key_to_WIF(self.private)
        super().__init__(public_key_to_address(self.public))

    def sign(self, message):
        return sign_message(self.private, message)

    def verify(self, message, sign):
        return verify_message(self.public, message, sign)


if __name__ == '__main__':

    seed = 77127
    difficulty = 4  # TODO: calculate automatically for let create blocks each 2 mins
    blockchain = Blockchain(seed, difficulty)
    prev_block = blockchain.last()
    print(prev_block.pow(difficulty))
    assert(prev_block.pow(difficulty) == seed)

    bob = LocalWallet()
    maria = Wallet("1HzMPPEhDLojTBzH4r5NFWRteg4jqpxsLm")
    antonio = Wallet("1p8pxYrueuPkSiWMV89LoEHwvS4kMcXyX")

    for _ in range(20):

        bus = BusBlock()
        bus.send(bob, maria, 5)
        bus.send(bob, antonio, 3)
        bus.validate(blockchain)

        new_block = blockchain.make_block(bus.flush(bob), difficulty)
        begin = time.time()
        # avoid DDoS and control blockchain growth
        nonce = new_block.pow(difficulty)
        print('add block calculated pow {} found with 5 bits in {:.2f} secs.'.format(nonce, time.time() - begin))
        blockchain.accept_block(new_block)
        prev_block = new_block

    print('---- Blockchain fingerprint ---- (root merkle tree): {}'.format(blockchain.hash()))
    for i, block in enumerate(blockchain):
        print('{:02d} - {} at {} merkle ({})'.format(i, block.hash(), block.timestamp, block.merkle_root))

    if blockchain.valid():
        print('valid blockchain!')
    else:
        print('Error en el blockchain!')

    '''
    Notas random:

    Eventos que afectan a un minero:

    - Recibes una transacción nueva. La guardas en la mempool.
    - Estas en la blockchain equivocada. Actualiza a la blockchain honesta. Liberas las transacciones minadas en la falsedad.
    - Otro minero ha encontrado el próximo bloque. Empieza el nuevo reto. Seleccionas de la mempool las que más fee dan.
    - Has minado un bloque. Añadelo a la blockchain, e informa via broadcast.
    '''
