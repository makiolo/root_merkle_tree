'''
requirements:

base58==1.0.0
ecdsa==0.13
mnemonic==0.20
bip32utils
flask==2.0.2
flask-crontab==0.1.2
flask-marshmallow==0.14.0
xmltodict==0.12.0

https://flask-marshmallow.readthedocs.io/en/latest/
'''
import contextlib
import os
import hashlib
import binascii
import traceback
from collections import defaultdict

import base58
import ecdsa
import time
import itertools
import pprint
import binascii
import mnemonic
import bip32utils

import sys
import re
from time import sleep

import xmltodict

try:  # if is python3
    from urllib.request import urlopen
except ImportError:  # if is python2
    from urllib2 import urlopen

from flask import Flask, jsonify


def check_balance ( address ):
    #Modify the value of the variable below to False if you do not want Bell Sound when the Software finds balance.
    SONG_BELL = True

    #Add time different of 0 if you need more security on the checks
    WARN_WAIT_TIME = 0

    blockchain_tags_json = [ 'total_received', 'final_balance', ]

    SATOSHIS_PER_BTC = 1e+8

    check_address = address

    parse_address_structure = re.match ( r' *([a-zA-Z1-9]{1,34})', check_address )
    if (parse_address_structure is not None):
        check_address = parse_address_structure.group ( 1 )
    else:
        print ( "\nThis Bitcoin Address is invalid" + check_address )
        exit ( 1 )

    #Read info from Blockchain about the Address
    reading_state = 1
    while (reading_state):
        try:
            htmlfile = urlopen ( "https://blockchain.info/address/%s?format=json" % check_address, timeout = 10 )
            htmltext = htmlfile.read ( ).decode ( 'utf-8' )
            reading_state = 0
        except:
            print(traceback.format_exc())
            reading_state += 1
            print ( "Checking... " + str ( reading_state ) )
            sleep (1)

    print ( "\nBitcoin Address = " + check_address )

    blockchain_info_array = [ ]
    tag = ''
    try:
        for tag in blockchain_tags_json:
            blockchain_info_array.append ( float ( re.search ( r'%s":(\d+),' % tag, htmltext ).group ( 1 ) ) )
    except:
        print ( "Error '%s'." % tag )
        exit ( 1 )

    for i, btc_tokens in enumerate ( blockchain_info_array ):

        sys.stdout.write ( "%s \t= " % blockchain_tags_json [ i ] )
        if btc_tokens > 0.0:
            print ( "%.8f Bitcoin" % (btc_tokens / SATOSHIS_PER_BTC) )
        else:
            print ( "0 Bitcoin" )

        if (SONG_BELL and blockchain_tags_json [ i ] == 'final_balance' and btc_tokens > 0.0):

            #If you have a balance greater than 0 you will hear the bell
            sys.stdout.write ( '\a\a\a' )
            sys.stdout.flush ( )

            return btc_tokens/SATOSHIS_PER_BTC


def bip39(mnemonic_words):
    # https://bitcoin.stackexchange.com/questions/76655/how-to-generate-public-and-private-key-pairs-from-the-12-seed-words-in-python
    mobj = mnemonic.Mnemonic("english")
    seed = mobj.to_seed(mnemonic_words)

    bip32_root_key_obj = bip32utils.BIP32Key.fromEntropy(seed)
    # fromEntropy
    # fromExtendedKey
    bip32_child_key_obj = bip32_root_key_obj.ChildKey(
        44 + bip32utils.BIP32_HARDEN
    ).ChildKey(
        0 + bip32utils.BIP32_HARDEN
    ).ChildKey(
        0 + bip32utils.BIP32_HARDEN
    ).ChildKey(0).ChildKey(0)

    return {
        'mnemonic_words': mnemonic_words,
        'addr': bip32_child_key_obj.Address(),
        'publickey': binascii.hexlify(bip32_child_key_obj.PublicKey()).decode(),
        'privatekey': bip32_child_key_obj.WalletImportFormat(),
    }


def generate_private_key():
    return binascii.hexlify(os.urandom(32)).decode('utf-8')


def private_key_to_WIF(private_key):
    var80 = "80" + str(private_key)
    var = hashlib.sha256(binascii.unhexlify(hashlib.sha256(binascii.unhexlify(var80)).hexdigest())).hexdigest()
    return str(base58.b58encode(binascii.unhexlify(str(var80) + str(var[0:8]))), 'utf-8')


def private_key_to_public_key(private_key):
    sign = ecdsa.SigningKey.from_string(binascii.unhexlify(private_key), curve = ecdsa.SECP256k1)
    return '04' + binascii.hexlify(sign.verifying_key.to_string()).decode('utf-8')


def public_key_to_address(public_key, unit):
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
    return ''.join(output[::-1]) + '_' + unit


class Blockchain:
    def __init__( self, seed, difficulty ):
        self.blocks = []
        self.difficulty = difficulty
        self.deposits = defaultdict(float)
        self.withdraws = defaultdict(float)
        self.qty_before = defaultdict( float )
        self._null_wallet = NullWallet()
        self.make_genesis( seed, difficulty )
        
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
        genesis = Block( self.hash(), seed, genesis_bus, 0, difficulty =bits )
        genesis.pow()
        self.accept_block(genesis)
        
    def make_money( self, foundation_wallet, supply, unit ):
        supply_bus = BusBlock()
        supply_bus.send( self._null_wallet, foundation_wallet, supply, unit )
        block = self.make_block( supply_bus )
        block.pow()
        self.accept_block(block)

    def last(self):
        return self.blocks[-1]

    def hash(self):
        hashes = []
        for block in self.blocks:
            hashes.append(block.hash())
        return Blockchain.calculate_root_merkle_hash(hashes)

    def make_block( self, busblock ):
        return Block( self.hash(), self.last().hash(), busblock, 0, difficulty=self.difficulty )
    
    def make_foundation_wallet( self, hash, unit, supply ):
        wallet = FoundationWallet(hash, unit)
        blockchain.make_money( wallet, supply, unit )
        return wallet
    
    def make_client_wallet( self, hash, unit ):
        wallet = HashWallet( hash, unit )
        self.init_profit( wallet, unit )
        return wallet

    def accept_block(self, block):
        if not block.solved:
            raise Exception('Block is not solved')
        self.blocks.append(block)

    def append (self, bus):
        new_block = self.make_block ( bus )
        new_block.pow ( )
        self.accept_block ( new_block )
        if self.valid():
            return new_block
        else:
            self.blocks.pop()
            raise Exception('Error appending BusBlock.')
        
    def exchange ( self, from_wallet, from_supply, baseQty, baseUnit, to_supply, to_wallet, quoteQty, quoteUnit ):
        bus = BusBlock()
        bus.doble_send (
                from_wallet, from_supply, baseQty,
                baseUnit, to_supply, to_wallet, quoteQty, quoteUnit)
        self.append( bus )

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
        # check positive balances and blockchain integrity
        publics = []
        for block in self.blocks:
            for transaction in block.transactions:
                # la null wallet no puede recibir dinero de nadie
                if transaction.data._to == self._null_wallet.address:
                    return False
                
                # una wallet de tipo foundation, solo recibe de la null wallet
                if transaction.data._to.endswith('FOUNDATION'):
                    if transaction.data._from != self._null_wallet.address:
                        return False
                
                # cada wallet foundation es única por nombre
                # la transferencia desde la null wallet a dicho simbolo de foundation debe ser
                # singleton, TODO:
                
                if transaction.data._from not in publics:
                    publics.append( (transaction.data._from, transaction.data._unit) )
                if transaction.data._to not in publics:
                    publics.append( (transaction.data._to, transaction.data._unit) )
                
        # skip null address
        for public_address, unit in publics:
            if public_address == self._null_wallet.address:
                # no check null wallet balance
                continue
            wallet = PublicWallet( public_address )
            balan = wallet.balance(self, unit)
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

    def init_profit( self, wallet, unit ):
        self.deposits[(wallet, unit)] = 0
        self.withdraws[(wallet, unit)] = 0
        self.qty_before[unit ] = wallet.balance( self, unit )

    def get_profit( self, wallet, unit ):
        qty_after = wallet.balance( blockchain, unit )
        profit = qty_after - self.qty_before[unit] + self.withdraws[(wallet, unit)] - self.deposits[(wallet, unit)]
        return profit

    def transfer( self, from_wallet, to_wallet, qty, unit ):
        bus = BusBlock()
        bus.send( from_wallet, to_wallet, qty, unit )
        self.append( bus )
        self.withdraws[(from_wallet, unit)] = self.withdraws[(from_wallet, unit)] + qty
        self.deposits[(to_wallet, unit)] = self.deposits[(to_wallet, unit)] + qty


class Block:
    def __init__( self, merkle_root, prev_hash, transactions, nonce, difficulty ):
        self.version = 0x02000000
        self.prev_hash = prev_hash
        self.merkle_root = merkle_root
        self.difficulty = difficulty
        self.target = "0" * difficulty
        self.nonce = nonce
        self.timestamp = time.time()
        self.transactions = transactions
        self.solved = False

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

    def pow(self):
        '''
        calculate nonce

        :param bits:
        :return:
        '''
        for nonce in itertools.count():  # inifinite generator by brute force
            if self.check_nonce(nonce, self.difficulty):
                self.solved = True
                return nonce


class Transaction:
    def __init__(self, from_, to_, qty_, unit_):
        self._from = from_
        self._to = to_
        self._qty = qty_
        self._unit = unit_

    def __str__(self):
        return '<data from="{}" to="{}" qty="{:.8f}" unit="{}" />'.format(self._from, self._to, self._qty, self._unit)

    def hash(self):
        return hashlib.sha256(self.__str__().encode()).hexdigest()


class TransactionWrap:
    def __init__( self, data, sign, public ):
        self.data = data
        self.sign = sign
        self.public = public

    def __str__(self):
        message = ''
        # mensaje
        message += '\t\t\t\t{}\n'.format(self.data)
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

    def sign_message(self, private_key, message):
        sk = ecdsa.SigningKey.from_string(binascii.unhexlify(private_key), curve=ecdsa.SECP256k1)
        message = message.encode('utf-8')
        sig = sk.sign(message)
        sign = binascii.hexlify(sig)
        return sign.decode('utf-8')


    def verify_message(self, public_key, message, sign):
        assert(public_key[:2] == '04')  # uncompressed public key
        vk = ecdsa.VerifyingKey.from_string(binascii.unhexlify(public_key[2:]), curve=ecdsa.SECP256k1)
        result = vk.verify(binascii.unhexlify(sign.encode('utf-8')), message.encode('utf-8'))
        return result


    def doble_send ( self, from_wallet, from_supply, baseQty, baseUnit, to_supply, to_wallet, quoteQty, quoteUnit ):
        self.send ( from_wallet, from_supply, baseQty, baseUnit)
        self.send ( to_supply, to_wallet, quoteQty, quoteUnit )

    def send(self, from_wallet, to_wallet, qty_, unit_):
        transaction = Transaction(from_wallet.endpoint(), to_wallet.endpoint(), qty_, unit_)
        message = str(transaction)
        sign = self.sign_message(from_wallet.private, message)
        valid_sign = self.verify_message(from_wallet.public, message, sign)
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
                message += '\t\t\t<transaction order="{}" hash="{}">\n'.format(i, transaction.data.hash())
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

    def balance( self, blockchain, unit, format=False ):
        if format:
            return '{} {}'.format(self.balance_income(blockchain, unit) - self.balance_expenses(blockchain, unit), unit)
        else:
            return self.balance_income(blockchain, unit) - self.balance_expenses(blockchain, unit)

    def balance_expenses( self, blockchain, unit ):
        balan = 0.0
        for block in blockchain.blocks:
            for transaction in block.transactions:
                if transaction.data._from == self.address and transaction.data._unit == unit:
                    # withdraw to other
                    balan += transaction.data._qty
        return balan
    
    def balance_income( self, blockchain, unit ):
        balan = 0.0
        for block in blockchain.blocks:
            for transaction in block.transactions:
                if transaction.data._to == self.address and transaction.data._unit == unit:
                    # deposit from other
                    balan += transaction.data._qty
        return balan


class PrivateWallet( PublicWallet ):
    def __init__(self, unit, private_key=None):
        if private_key is None:
            self.private = generate_private_key()
        else:
            self.private = binascii.hexlify(private_key.encode('utf-8')).decode('utf-8')
        self.public = private_key_to_public_key(self.private)
        self.import_key = private_key_to_WIF(self.private)
        super().__init__(public_key_to_address(self.public, unit))


class HashWallet(PrivateWallet):
    def __init__(self, name, unit):
        self.name = name
        super().__init__(unit, hashlib.sha256(name.encode()).hexdigest()[:32])


class NullWallet( PublicWallet ):
    def __init__(self):
        self.private = binascii.hexlify(('0'*32).encode('utf-8')).decode('utf-8')
        self.public = private_key_to_public_key(self.private)
        self.import_key = private_key_to_WIF(self.private)
        super().__init__(public_key_to_address(self.public, 'NULL'))



class FoundationWallet( PrivateWallet ):
    def __init__(self, name, unit):
        self.name = name
        self.unit = unit
        super().__init__('{}-FOUNDATION'.format(unit), hashlib.sha256(name.encode()).hexdigest()[:32])


seed = 1234
difficulty = 4
blockchain = Blockchain( seed, difficulty )

# crear distintas criptomonedas dentro de la red
capital_eur = blockchain.make_foundation_wallet('Central deposit EUROS', 'EUR', 21000000)
capital_vechain = blockchain.make_foundation_wallet('Central deposit VECHAIN', 'VET', 21000000)
test_nft = blockchain.make_foundation_wallet('Central deposit NFT', 'NFT-1234', 1)

# crear wallets
person = blockchain.make_client_wallet( 'Personal Ricardo', 'EUR' )
vechain = blockchain.make_client_wallet( 'Personal Vechain', 'VET' )
person2 = blockchain.make_client_wallet( 'Personal Ricardo 2', 'EUR' )
vechain2 = blockchain.make_client_wallet( 'Personal Vechain 2', 'VET' )

# reparto inicial (utilizando el dinero de la fundación)
blockchain.transfer( capital_eur, person, 35.0, 'EUR' )
blockchain.transfer( capital_eur, person2, 5.0, 'EUR' )
blockchain.transfer( capital_vechain, vechain, 0.0, 'VET' )
blockchain.transfer( capital_vechain, vechain2, 1478.0, 'VET' )

# send unique nft
assert(person.balance(blockchain, 'NFT-1234') == 0.0)
blockchain.transfer( test_nft, person, 1, 'NFT-1234')
assert(person.balance(blockchain, 'NFT-1234') == 1.0)

# compra VET/EUR
blockchain.exchange ( 
        person, person2, 35, 'EUR',
        vechain2, vechain, 1478, 'VET')

# venta VET/EUR (beneficio 5 euros)
blockchain.exchange (
        person2, person, 40, 'EUR',
        vechain, vechain2, 1478, 'VET')

# transferencia (no cuenta en el calculo del profit)
blockchain.transfer( person, person2, 33.0, 'EUR' )

# deberia dar +5 y -5 respectivamente
profit = blockchain.get_profit( person, 'EUR' )
profit2 = blockchain.get_profit( person2, 'EUR' )

print("profit: {}".format(profit))
print("profit2: {}".format(profit2))

print("Ricardo balance: {}".format( person.balance( blockchain, 'EUR', True ) ) )
print("Vechain balance: {}".format( vechain.balance( blockchain, 'VET', True ) ) )
print("Ricardo2 balance: {}".format( person2.balance( blockchain, 'EUR', True ) ) )
print("Vechain2 balance: {}".format( vechain2.balance( blockchain, 'VET', True ) ) )


app = Flask(__name__)


@app.route("/", methods=['GET'])
def get_blockchain():
    return jsonify(xmltodict.parse(str(blockchain)))


def on_event_transaction():
    # - Recibes una transacción nueva. La guardas en la mempool.
    pass


def check_blockchain():
    # - Si Estas en la blockchain equivocada. Actualiza a la blockchain honesta. Liberas las transacciones minadas en la falsedad.
    pass


def on_event_block_mined():
    pass
    # - Otro minero ha encontrado el próximo bloque. Empieza el nuevo reto. Seleccionas de la mempool las que más fee dan.


def mined_block():
    # - Has minado un bloque. Añadelo a la blockchain, e informa via broadcast.
    pass


# Eventos periodicos
# cada minuto buscar peers vecinos


# Desplegar una red p2p
# https://pypi.org/project/pyp2p/


app.run('127.0.0.1', 5000)
