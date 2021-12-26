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
import math
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
import pandas as pd

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
        self.foundation_wallets = {}
        self.client_wallets = {}
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
        supply_bus.send(self, self._null_wallet, foundation_wallet, supply, unit )
        block = self.make_block( supply_bus )
        block.pow()
        self.accept_block(block)

    def build_foundation_wallet(self, hash_wallet, unit, supply):
        key = hash_wallet
        if key not in self.foundation_wallets:
            wallet = blockchain.make_foundation_wallet(hash_wallet, unit, supply)
            self.foundation_wallets[key] = wallet
        return self.foundation_wallets[key]

    def build_client_wallet(self, hash_wallet, unit):
        key = hash_wallet
        if key not in self.client_wallets:
            wallet = blockchain.make_client_wallet(hash_wallet, unit)
            self.client_wallets[key] = wallet
        return self.client_wallets[key]

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
    
    def send(self, from_wallet, to_wallet, qty_, unit_):
        bus = BusBlock()
        bus.send(self, from_wallet, to_wallet, qty_, unit_ )
        blockchain.append( bus )

    def accept_block(self, block):
        if not block.solved:
            raise Exception('Block is not solved')
        self.blocks.append(block)

    # def search_next_txfrom(self, i, j, txto):
    #     first = True
    #     for block in self.blocks[i:]:
    #         if first:
    #             trans = block.transactions[j:]
    #             first = False
    #         else:
    #             trans = block.transactions
    #         for transaction in trans:
    #             for txfrom in transaction.data._froms:
    #                 if txfrom == txto:
    #                     return txfrom
                    
    def is_unspent( self, transaction, txto ):
        txid = transaction.hash()
        for block in blockchain.blocks:
            for trans in block.transactions:
                for txfrom in trans.data._froms:
                    if      txfrom.txid == txid and \
                            txfrom.wallet == txto.wallet and \
                            txfrom.unit == txto.unit and \
                            txfrom.qty == txto.qty:
                        return False
        return True
                    
    def get_unspent_2_0( self, wallet, qty, unit ):
        total_qty = qty
        candidates = []
        for block in blockchain.blocks:
            for transaction in block.transactions:
                if total_qty > 0.0:
                    for txto in transaction.data._tos:
                        if txto.wallet == wallet and txto.unit == unit:
                            if self.is_unspent(transaction.data, txto):
                                total_qty -= txto.qty
                                candidates.append((transaction.data, txto))
        return candidates

    def append (self, bus):
        '''
        TODO:
        me recorro el bus block y lo convierto
        en un "TransactionTree"
        '''
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
        bus.doble_send ( self, 
                from_wallet, from_supply, baseQty, baseUnit, 
                to_supply, to_wallet, quoteQty, quoteUnit)
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
                # if transaction.data._to == self._null_wallet.address:
                #     raise Exception("NullWallet can't receive transactions.")
                
                # una wallet de tipo foundation, solo recibe de la null wallet
                # if transaction.data._to.endswith('FOUNDATION'):
                #     if transaction.data._from != self._null_wallet.address:
                #         raise Exception("Foundation wallet only receive from NullWallet")
                
                # cada wallet foundation es única por nombre
                # la transferencia desde la null wallet a dicho simbolo de foundation debe ser
                # singleton, TODO:
                
                for txfrom in transaction.data._froms:
                    element = (txfrom.wallet, txfrom.unit)
                    if element not in publics:
                        publics.append( element )
                for txto in transaction.data._tos:
                    element = (txto.wallet, txto.unit)
                    if element not in publics:
                        publics.append( element )
                
        # skip null address
        # for public_address, unit in publics:
        #     if public_address == self._null_wallet.address:
        #         continue
        #     wallet = PublicWallet( public_address )
        #     balan = wallet.balance(self, unit)
        #     if balan < 0.0:
        #         raise Exception('Wallet {} with invalid balance {}.'.format(public_address, balan))
        
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
        bus.send( self, from_wallet, to_wallet, qty, unit )
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

class Tx:
    def __init__(self, wallet, qty, unit):
        self.wallet = wallet
        self.qty = qty
        self.unit = unit

    def __eq__(self, other):
        return self.wallet == other.wallet and \
               self.qty == other.qty and \
               self.unit == other.qty


class TxFrom(Tx):
    def __init__(self, txid, wallet, qty, unit):
        self.txid = txid
        super().__init__(wallet, qty, unit)
        
    def __str__(self):
        return '<from txid="{}" wallet="{}" qty="{:.8f}" unit="{}" />\n'.format(self.txid, self.wallet, self.qty, self.unit)

    def hash(self):
        return hashlib.sha256(self.__str__().encode()).hexdigest()


class TxTo(Tx):
    def __init__(self, wallet, qty, unit):
        super().__init__(wallet, qty, unit)

    def __str__(self):
        return '<to wallet="{}" qty="{:.8f}" unit="{}" />\n'.format(self.wallet, self.qty, self.unit)

    def hash(self):
        return hashlib.sha256(self.__str__().encode()).hexdigest()


class Transaction:
    def __init__(self):
        self.timestamp = time.time()
        self._froms = []
        self._tos = []
        
    def add_from( self, txid, wallet, qty, unit):
        self._froms.append(TxFrom(txid, wallet, qty, unit))

    def add_to( self, wallet, qty, unit ):
        self._tos.append(TxTo(wallet, qty, unit))

    def __str__(self):
        message = '\t\t\t<transaction_content timestamp="{}">\n'.format(self.timestamp)
        for txfro in self._froms:
            message += '\t\t\t\t{}\n'.format(txfro)
        for txto in self._tos:
            message += '\t\t\t\t{}\n'.format(txto)
        message += '\t\t\t</transaction_content>\n'
        return message
    
    def total_input( self, unit ):
        qty = 0.0
        for txfrom in self._froms:
            if txfrom.unit == unit:
                qty += txfrom.qty
        return qty
    
    def total_output( self, unit ):
        qty = 0.0
        for txto in self._tos:
            if txto.unit == unit:
                qty += txto.qty
        return qty
    
    def fee( self, unit ):
        return self.total_input(unit) - self.total_output(unit)

    def hash(self):
        message = str(self.timestamp) + '\n'
        for txfro in self._froms:
            message += '{}\n'.format(txfro.hash())
        for txto in self._tos:
            message += '{}\n'.format(txto.hash())
        return hashlib.sha256(message.encode()).hexdigest()


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
    
    def __getitem__(self, item):
        return self.transactions[item]

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


    def doble_send ( self, blockchain, from_wallet, from_supply, baseQty, baseUnit, to_supply, to_wallet, quoteQty, quoteUnit ):
        self.send ( blockchain, from_wallet, from_supply, baseQty, baseUnit)
        self.send ( blockchain, to_supply, to_wallet, quoteQty, quoteUnit )

    def send(self, blockchain, from_wallet, to_wallet, qty_, unit_):
        # buscar en todas las transacciones de "from_wallet"
        # elegir por FIFO, las primeras no gastadas (UTXO)
        # nos recorremos un arbol en 
        # 
        # Chat
        # 
        # 
        # Profile picture of Villar Robledillo Cesar.
        # Villar Robledillo Cesar
        # 
        # 
        #   
        # 
        # Settingsprofundidad
        # los nodos hoja son los UTXOS (dinero no gastado)
        
        transaction = Transaction()
        if from_wallet != blockchain._null_wallet:
            txtos = blockchain.get_unspent_2_0(from_wallet, qty_, unit_)
            for trans, txto in txtos:
                transaction.add_from(trans.hash(), txto.wallet, txto.qty, txto.unit)
            spent_qty = qty_
            for _, txto in txtos:
                if spent_qty < txto.qty:
                    transaction.add_to(to_wallet, spent_qty, txto.unit)
                    transaction.add_to(from_wallet, txto.qty - spent_qty, txto.unit)
                    spent_qty = 0.0
                else:
                    transaction.add_to(to_wallet, txto.qty, txto.unit)
                    spent_qty -= txto.qty
                if spent_qty == 0.0:
                    break
            if(spent_qty != 0.0):
                print("error {}: {} {} {} {}".format(spent_qty, from_wallet, to_wallet, qty_, unit_))
        else:
            # txto foundation
            transaction.add_to(to_wallet, qty_, unit_)
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
                message += '\t\t\t<transaction order="{}" txid="{}">\n'.format(i, transaction.data.hash())
                message += str(transaction)
                message += '\t\t\t</transaction>\n'
            return '\t\t<transactions>\n{}\t\t</transactions>\n'.format(message)
        else:
            return ''


class PublicWallet:
    def __init__(self, address):
        self.address = address
        
    def __str__(self):
        return self.address
    
    def __eq__(self, other):
        if isinstance(other, str):
            return self.address == other
        else:
            return self.address == other.address

    def __hash__(self):
        return hash(self.address)

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
                for txfrom in transaction.data._froms:
                    if txfrom.wallet == self.address and txfrom.unit == unit:
                        # withdraw to other
                        balan += txfrom.qty
        return balan
    
    def balance_income( self, blockchain, unit ):
        balan = 0.0
        for block in blockchain.blocks:
            for transaction in block.transactions:
                for txto in transaction.data._tos:
                    if txto.wallet == self.address and txto.unit == unit:
                        # deposit from other
                        balan += txto.qty
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


dataset_csv = r"C:\Users\x335336\OneDrive - Santander Office 365\Documents\crypto_transactions_record_20211213_195307.csv"
print(dataset_csv)
df = pd.read_csv(dataset_csv)

seed = 1234
difficulty = 2
blockchain = Blockchain( seed, difficulty )

'''
'Timestamp (UTC)', 
'Transaction Description', 
'Currency', 
'Amount',
'To Currency', 
'To Amount', 
'Native Currency', 
'Native Amount',
'Native Amount (in USD)', 
'Transaction Kind'
'''
currencies = []
for index, row in df.iterrows():
    currency = row[ 'Currency' ]
    amount = row[ 'Amount' ]
    native_currency = row[ 'Native Currency' ]
    native_amount = row[ 'Native Amount' ]
    to_currency = row['To Currency']
    to_amount = row['To Amount']
    if currency not in currencies:
        if isinstance(currency, str) or not math.isnan(currency):
            currencies.append(currency)
    if to_currency not in currencies:
        if isinstance(currency, str) or not math.isnan(to_currency):
            currencies.append(to_currency)

    super_supply = 200000000
    
    if(row['Transaction Kind'] == 'crypto_withdrawal') or (row['Transaction Kind'] == 'crypto_wallet_swap_debited') or (row['Transaction Kind'] == 'card_top_up'):

        from_foundation_wallet = blockchain.build_foundation_wallet('Central deposit {}'.format(currency), currency, super_supply)
        from_personal_wallet = blockchain.build_client_wallet( 'Personal {}'.format(currency), currency )
        
        blockchain.transfer(from_personal_wallet, from_foundation_wallet, -amount, currency)
        
    elif(row['Transaction Kind'] == 'crypto_deposit') or (row['Transaction Kind'] == 'crypto_wallet_swap_credited'):

        from_foundation_wallet = blockchain.build_foundation_wallet('Central deposit {}'.format(currency), currency, super_supply)
        from_personal_wallet = blockchain.build_client_wallet( 'Personal {}'.format(currency), currency )
        
        blockchain.transfer(from_foundation_wallet, from_personal_wallet, amount, currency)

    elif(row['Transaction Kind'] == 'dynamic_coin_swap_debited') or (row['Transaction Kind'] == 'dynamic_coin_swap_credited') or (row['Transaction Kind'] == 'dynamic_coin_swap_bonus_exchange_deposit') or (row['Transaction Kind'] == 'lockup_lock'):
        pass
        
    elif (row['Transaction Kind'] == 'crypto_exchange') or (row['Transaction Kind'] == 'crypto_viban_exchange'):

        from_foundation_wallet = blockchain.build_foundation_wallet('Central deposit {}'.format(currency), currency, super_supply)
        from_personal_wallet = blockchain.build_client_wallet( 'Personal {}'.format(currency), currency )

        to_foundation_wallet = blockchain.build_foundation_wallet('Central deposit {}'.format(to_currency), to_currency, super_supply)
        to_personal_wallet = blockchain.build_client_wallet( 'Personal {}'.format(to_currency), to_currency )

        blockchain.exchange (
                from_personal_wallet, from_foundation_wallet, -amount, currency,
                to_foundation_wallet, to_personal_wallet, to_amount, to_currency)
        
    elif(row['Transaction Kind'] == 'reimbursement'):

        from_foundation_wallet = blockchain.build_foundation_wallet('Central deposit {}'.format(currency), currency, super_supply)
        from_personal_wallet = blockchain.build_client_wallet( 'Personal {}'.format(currency), currency )

        blockchain.send( from_foundation_wallet, from_personal_wallet, amount, currency )
        
    elif(row['Transaction Kind'] == 'viban_purchase'):
        
        from_foundation_wallet = blockchain.build_foundation_wallet('Central deposit {}'.format(currency), currency, super_supply)
        from_personal_wallet = blockchain.build_client_wallet( 'Personal {}'.format(currency), currency )

        to_foundation_wallet = blockchain.build_foundation_wallet('Central deposit {}'.format(to_currency), to_currency, super_supply)
        to_personal_wallet = blockchain.build_client_wallet( 'Personal {}'.format(to_currency), to_currency )

        if currency == native_currency:
            blockchain.transfer (from_foundation_wallet, from_personal_wallet, -amount, currency)
        blockchain.exchange (
                from_personal_wallet, from_foundation_wallet, -amount, currency,
                to_foundation_wallet, to_personal_wallet, to_amount, to_currency)
        
    else:
        print(row['Transaction Kind'] + ' --')
        print(row)

currency = 'EUR'
wallet = blockchain.build_client_wallet( 'Personal {}'.format(currency), currency )
print("profit: {}".format(blockchain.get_profit(wallet, currency)))
 
print('-- balances --')
for currency in currencies:
    if isinstance(currency, str):
        wallet = blockchain.build_client_wallet( 'Personal {}'.format(currency), currency )
        print('{} balance: {}'.format(currency, wallet.balance(blockchain, currency, True)))

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
