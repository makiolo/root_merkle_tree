import hashlib
import time
import itertools


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


if __name__ == '__main__':

    seed = 661279
    difficulty = 5  # TODO: calculate automatically for let create blocks each 2 mins
    blockchain = Blockchain(seed, difficulty)
    prev_block = blockchain.last()
    print(prev_block.pow(difficulty))
    assert(prev_block.pow(difficulty) == seed)

    for _ in range(20):
        # facts to register
        facts = ['X-->Y 3', 'Y-->Z 4', 'Z-->R 6']
        new_block = blockchain.make_block(facts, difficulty)
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
