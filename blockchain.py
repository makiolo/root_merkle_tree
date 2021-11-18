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
    def __init__(self, genesis_wallet, seed, bits, supply):
        self.genesis_wallet = genesis_wallet
        self.blocks = []
        self.bits = bits
        self._null_wallet = NullWallet()
        self.make_genesis(seed, bits)
        self.make_money(genesis_wallet, bits, supply)
        
    def __str__(self):
        message = '<blockchain hash="{}">\n'.format(self.hash())
        for i, block in enumerate(blockchain):
            message += '\t<block order="{}" hash="{}" timestamp="{}">\n'.format(i, block.hash(), block.timestamp)
            message += str(block)
            message += '\t</block>\n'
        message += '</blockchain>'
        return message

    def __iter__(self):
        return iter(self.blocks)

    def make_genesis(self, seed, bits):
        genesis_bus = BusBlock()
        genesis = Block(self.hash(), seed, genesis_bus, 0, bits=bits)
        genesis.pow(bits)
        self.accept_block(genesis)
        
    def make_money(self, genesis_wallet, bits, supply):
        supply_bus = BusBlock()
        supply_bus.send(self._null_wallet, genesis_wallet, supply)
        block = self.make_block(supply_bus, bits)
        block.pow(bits)
        self.accept_block(block)

    def last(self):
        return self.blocks[-1]

    def hash(self):
        hashes = []
        for block in self.blocks:
            hashes.append(block.hash())
        return Blockchain.calculate_root_merkle_hash(hashes)

    def make_block(self, busblock, bits):
        return Block(self.hash(), self.last().hash(), busblock, 0, bits=bits)

    def accept_block(self, block):
        self.blocks.append(block)

    def append (self, bus, driver):
        # TODO: how calculate salary
        salary = 0.5
        if self.genesis_wallet.balance(self) >= salary:
            # pay commission
            bus.send(self.genesis_wallet, driver, salary)
        new_block = self.make_block ( bus, self.bits )
        new_block.pow ( self.bits )
        self.accept_block ( new_block )
        if self.valid():
            return new_block
        else:
            self.blocks.pop()
            raise Exception('Error appending BusBlock.')

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

        # check positive balances
        publics = []
        for block in self.blocks:
            for transaction in block.transactions:
                if transaction.data._from not in publics:
                    publics.append( transaction.data._from )
                if transaction.data._to not in publics:
                    publics.append( transaction.data._to )
        # skip null address
        for public_address in publics:
            if public_address == self._null_wallet.address:
                continue
            wallet = PublicWallet( public_address )
            balan = wallet.balance(self)
            if balan < 0.0:
                # Invalid wallet
                return False
        
        # check integrity
        # TODO: check difficulty expected
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
        message = ''
        message += '\t\t<version>{}</version>\n'.format(self.version)
        message += '\t\t<prev_hash>{}</prev_hash>\n'.format(self.prev_hash)
        message += '\t\t<merkle_root>{}</merkle_root>\n'.format(self.merkle_root)
        message += '\t\t<target>{}</target>\n'.format(self.target)
        message += '\t\t<nonce>{}</nonce>\n'.format(self.nonce)
        if len(self.transactions) > 0:
            message += str(self.transactions)
        return message

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
    result = vk.verify(binascii.unhexlify(sign.encode('utf-8')), message.encode('utf-8'))
    return result


class Transaction:
    def __init__(self, from_, to_, qty_, *args, **kwargs):
        self._from = from_
        self._to = to_
        self._qty = qty_
        self._args = args
        self._kwargs = kwargs

    def __str__(self):
        return '<data from="{}" to="{}" qty="{:.8f}" />'.format(self._from, self._to, self._qty)


class TransactionWrap:
    def __init__( self, data, sign, public ):
        self.data = data
        self.sign = sign
        self.public = public

    def __str__(self):
        message = ''
        # mensaje
        message += '\t\t\t\t{}\n'.format( self.data )
        # firma
        message += '\t\t\t\t<sign>{}</sign>\n'.format(self.sign)
        # remitente
        message += '\t\t\t\t<public>{}</public>\n'.format(self.public)
        return message


class BusBlock:
    def __init__(self):
        self.transactions = []
        
    def __iter__(self):
        return iter(self.transactions)
    
    def __len__(self):
        return len(self.transactions)

    def send(self, from_wallet, to_wallet, qty_):
        transaction = Transaction(from_wallet.endpoint(), to_wallet.endpoint(), qty_)
        message = str(transaction)
        sign = sign_message(from_wallet.private, message)
        valid_sign = verify_message(from_wallet.public, message, sign)
        if not valid_sign:
            raise Exception('Error generating sign.')
        transaction_wrap = TransactionWrap(transaction, sign, from_wallet.public)
        self.transactions.append(transaction_wrap)

    def __str__(self):
        '''
        Generate block data
        '''
        if len(self.transactions) > 0:
            message = ''
            for i, transaction in enumerate(self.transactions):
                message += '\t\t\t<transaction order="{}">\n'.format(i)
                message += str(transaction)
                message += '\t\t\t</transaction>\n'
            return '\t\t<transactions>\n{}\t\t</transactions>\n'.format(message)
        else:
            return ''


class PublicWallet:
    def __init__(self, address):
        self.address = address

    def endpoint(self):
        return self.address

    def balance( self, blockchain ):
        return self.balance_income(blockchain) - self.balance_expenses(blockchain)

    def balance_expenses( self, blockchain ):
        balan = 0.0
        for block in blockchain.blocks:
            for transaction in block.transactions:
                if transaction.data._from == self.address:
                    # withdraw to other
                    balan += transaction.data._qty
        return balan
    
    def balance_income( self, blockchain ):
        balan = 0.0
        for block in blockchain.blocks:
            for transaction in block.transactions:
                if transaction.data._to == self.address:
                    # deposit from other
                    balan += transaction.data._qty
        return balan


class PrivateWallet( PublicWallet ):
    def __init__(self, private_key=None):
        if private_key is None:
            self.private = generate_private_key()
        else:
            self.private = binascii.hexlify(private_key.encode('utf-8')).decode('utf-8')
        self.public = private_key_to_public_key(self.private)
        self.import_key = private_key_to_WIF(self.private)
        super().__init__(public_key_to_address(self.public))

    def sign(self, message):
        return sign_message(self.private, message)

    def verify(self, message, sign):
        return verify_message(self.public, message, sign)


class HashWallet(PrivateWallet):
    def __init__(self, word):
        super().__init__(hashlib.sha256(word.encode()).hexdigest()[:32])


class NullWallet( PublicWallet ):
    def __init__(self):
        self.private = binascii.hexlify(('0'*32).encode('utf-8')).decode('utf-8')
        self.public = private_key_to_public_key(self.private)
        self.import_key = private_key_to_WIF(self.private)
        super().__init__(public_key_to_address(self.public))


if __name__ == '__main__':

    seed = 1234
    difficulty = 4
    supply = 21000000
    capital = HashWallet( 'Supply total de la coin' )
    blockchain = Blockchain( capital, seed, difficulty, supply )
    sathosi = HashWallet( 'Sathosi' )
    hal_finney = HashWallet( 'Hal Finney' )

    bus = BusBlock()
    bus.send( capital, sathosi, 10.0 )
    bus.send( sathosi, hal_finney, 10.0 )
    blockchain.append( bus, sathosi )  # sathosi receive prize for append
    
    print(blockchain)

    print("capital balance: {}".format( capital.balance( blockchain ) ) )
    print("Sathosi balance: {}".format( sathosi.balance( blockchain ) ) )
    print("Hal Finney balance: {}".format( hal_finney.balance( blockchain ) ) )

    '''
    Notas random:

    Eventos que afectan a un minero:

    - Recibes una transacción nueva. La guardas en la mempool.
    - Estas en la blockchain equivocada. Actualiza a la blockchain honesta. Liberas las transacciones minadas en la falsedad.
    - Otro minero ha encontrado el próximo bloque. Empieza el nuevo reto. Seleccionas de la mempool las que más fee dan.
    - Has minado un bloque. Añadelo a la blockchain, e informa via broadcast.
    '''
    
