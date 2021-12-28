'''
requirements:

base58==1.0.0
ecdsa==0.13
flask==2.0.2
flask-crontab==0.1.2
flask-marshmallow==0.14.0
xmltodict==0.12.0

https://flask-marshmallow.readthedocs.io/en/latest/
'''
import os
import math
import hashlib
import base58
import ecdsa
import itertools
import binascii
import pandas as pd
import xmltodict
from collections import defaultdict
from datetime import datetime
from flask import Flask, jsonify

try:  # if is python3
    from urllib.request import urlopen
except ImportError:  # if is python2
    from urllib2 import urlopen


size_bits = 32
cripto_curve = ecdsa.SECP256k1
address_algorithm = 'ripemd160'

# size_bits = 48
# cripto_curve = ecdsa.BRAINPOOLP384r1
# address_algorithm = 'sha384'

precision = 12
threshold = 10 ** -precision
threshold2 = 10 ** -7
super_supply = 200000000


def generate_private_key():
    return binascii.hexlify(os.urandom(size_bits)).decode('utf-8')


def private_key_to_WIF(private_key):
    var80 = "80" + str(private_key)
    var = hashlib.sha256(binascii.unhexlify(hashlib.sha256(binascii.unhexlify(var80)).hexdigest())).hexdigest()
    return str(base58.b58encode(binascii.unhexlify(str(var80) + str(var[0:8]))), 'utf-8')


def private_key_to_public_key(private_key):
    sign = ecdsa.SigningKey.from_string(binascii.unhexlify(private_key), curve=cripto_curve)
    return '04' + binascii.hexlify(sign.verifying_key.to_string()).decode('utf-8')


def public_key_to_address(public_key, unit):
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    count = 0
    val = 0
    var = hashlib.new(address_algorithm)
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
        n, remainder = divmod(n, 58)
        output.append(alphabet[remainder])
    while (val < count):
        output.append(alphabet[0])
        val += 1
    return ''.join(output[::-1]) + '_' + unit


def sign_message(private_key, message):
    sk = ecdsa.SigningKey.from_string(binascii.unhexlify(private_key), curve=cripto_curve)
    message = message.encode('utf-8')
    sig = sk.sign(message)
    sign = binascii.hexlify(sig)
    return sign.decode('utf-8')

def verify_message(public_key, message, sign):
    assert (public_key[:2] == '04')  # uncompressed public key
    vk = ecdsa.VerifyingKey.from_string(binascii.unhexlify(public_key[2:]), curve=cripto_curve)
    result = vk.verify(binascii.unhexlify(sign.encode('utf-8')), message.encode('utf-8'))
    return result


class Blockchain:
    def __init__(self, seed, difficulty):
        self.blocks = []
        self.difficulty = difficulty
        self.deposits = defaultdict(float)
        self.withdraws = defaultdict(float)
        self.qty_before = defaultdict(float)
        self._null_wallet = NullWallet()
        self.foundation_wallets = {}
        self.client_wallets = {}
        self.make_genesis(seed, difficulty)
        self.cached_txtos = {}

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
        genesis = Block(self.hash(), seed, genesis_bus, 0, difficulty=bits)
        genesis.pow()
        self.accept_block(genesis)

    def make_money(self, foundation_wallet, supply, unit):
        supply_bus = BusBlock()
        supply_bus.send(self, self._null_wallet, foundation_wallet, supply, unit, description='{} FOUNDATION'.format(unit))
        block = self.make_block(supply_bus)
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

    def make_block(self, busblock):
        return Block(self.hash(), self.last().hash(), busblock, 0, difficulty=self.difficulty)

    def make_foundation_wallet(self, hash, unit, supply):
        wallet = FoundationWallet(hash, unit)
        blockchain.make_money(wallet, supply, unit)
        return wallet

    def make_client_wallet(self, hash, unit):
        wallet = HashWallet(hash, unit)
        self.init_profit(wallet, unit)
        return wallet

    def accept_block(self, block):
        if not block.solved:
            raise Exception('Block is not solved')
        # register txtos in caches
        for trans in block.transactions:
            txid = trans.data.hash()
            for vout, txto in enumerate(trans.data._tos):
                self.cached_txtos[(txid, vout)] = txto
        for trans in block.transactions:
            for txfrom in trans.data._froms:
                txto = self.cached_txtos[(txfrom.txid, txfrom.vout)]
                valid_sign = verify_message(txto.public_key, txto.hash(), txfrom.signature)
                if not valid_sign:
                    raise Exception('Verify lock / unlock scripts are invalid.')
        self.blocks.append(block)

    def get_txto(self, txid, vout):
        '''
        cached
        '''
        return self.cached_txtos[(txid, vout)]

    def is_unspent(self, transaction, txto):
        txid = transaction.data.hash()
        for block in blockchain.blocks:
            for trans in block.transactions:
                if transaction.data.unit == trans.data.unit:
                    for txfrom in trans.data._froms:
                        txto_solved = self.get_txto(txfrom.txid, txfrom.vout)
                        if      txfrom.txid == txid and \
                                txto_solved.public_key == txto.public_key and \
                                txto_solved.qty == txto.qty:
                            return False
        return True

    def calculate_balance(self, public_key, unit):
        balance = 0.0
        for block in blockchain.blocks:
            for transaction in block.transactions:
                if transaction.data.unit == unit:
                    for txto in transaction.data._tos:
                        if txto.public_key == public_key:
                            if self.is_unspent(transaction, txto):
                                balance += txto.qty
        return balance

    def calculate_unspent(self, public_key, qty, unit):
        '''
        use FIFO
        '''
        total_qty = qty
        candidates = []
        for block in blockchain.blocks:
            for transaction in block.transactions:
                if transaction.data.unit == unit:
                    for vout, txto in enumerate(transaction.data._tos):
                        if total_qty <= threshold:
                            break
                        if txto.public_key == public_key:
                            if self.is_unspent(transaction, txto):
                                total_qty -= txto.qty
                                candidates.append((transaction.data, txto, vout))
        return candidates

    def append(self, bus):
        new_block = self.make_block(bus)
        new_block.pow()
        self.accept_block(new_block)
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
        # check integrity
        i = 0
        while i < len(self.blocks) - 1:
            if self.blocks[i].version != self.blocks[i + 1].version:
                return False
            if self.blocks[i].hash() != self.blocks[i + 1].prev_hash:
                return False
            if self.blocks[i].merkle_root == self.blocks[i + 1].merkle_root:
                return False
            if self.blocks[i].timestamp >= self.blocks[i + 1].timestamp:
                return False
            if not self.blocks[i].hash().startswith(self.blocks[i].target):
                return False
            for trans in self.blocks[i].transactions:
                if len(trans.data._froms) > 0:
                    if abs(trans.data.total_input(self, trans.data.unit) - trans.data.total_output(self, trans.data.unit)) > threshold2:
                        print('warning {} {} vs {} {}'.format(  trans.data.total_input(trans.data.unit), trans.data.unit,
                                                                trans.data.total_output(trans.data.unit), trans.data.unit))
                        return False
            i += 1
        return True

    def init_profit(self, wallet, unit):
        self.deposits[(wallet, unit)] = 0
        self.withdraws[(wallet, unit)] = 0
        self.qty_before[unit] = wallet.balance(self, unit)

    def get_profit(self, wallet, unit):
        qty_after = wallet.balance(self, unit)
        profit = qty_after - self.qty_before[unit] + self.withdraws[(wallet, unit)] - self.deposits[(wallet, unit)]
        return profit

    def exchange(self, from_wallet, from_supply, baseQty, baseUnit, to_supply, to_wallet, quoteQty, quoteUnit, timestamp=None, description=None, native_currency=None, native_amount=None):
        bus = BusBlock()
        bus.doble_send(self,
                       from_wallet, from_supply, baseQty, baseUnit,
                       to_supply, to_wallet, quoteQty, quoteUnit, timestamp=timestamp, description=description, native_currency=native_currency, native_amount=native_amount)
        self.append(bus)

    def transfer(self, from_wallet, to_wallet, qty, unit, timestamp=None, description=None, native_currency=None, native_amount=None):
        bus = BusBlock()
        bus.send(self, from_wallet, to_wallet, qty, unit, timestamp=timestamp, description=description, native_currency=native_currency, native_amount=native_amount)
        self.append(bus)
        self.withdraws[(from_wallet, unit)] = self.withdraws[(from_wallet, unit)] + qty
        self.deposits[(to_wallet, unit)] = self.deposits[(to_wallet, unit)] + qty

    def send(self, from_wallet, to_wallet, qty_, unit_, timestamp=None, description=None, native_currency=None, native_amount=None):
        bus = BusBlock()
        bus.send(self, from_wallet, to_wallet, qty_, unit_, timestamp=timestamp, description=description, native_currency=native_currency, native_amount=native_amount)
        self.append(bus)


class Block:
    def __init__(self, merkle_root, prev_hash, transactions, nonce, difficulty):
        self.version = 0x02000000
        self.prev_hash = prev_hash
        self.merkle_root = merkle_root
        self.difficulty = difficulty
        self.target = "0" * difficulty
        self.nonce = nonce
        self.timestamp = datetime.now()
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
        for nonce in itertools.count():  # inifinite generator by brute force
            if self.check_nonce(nonce, self.difficulty):
                self.solved = True
                return nonce


class TxFrom:
    def __init__(self, txid, vout, signature):
        self.txid = txid
        self.vout = vout
        self.signature = signature

    def __str__(self):
        return '<from txid="{}" vout="{}" signature="{}" />\n'.format(self.txid, self.vout, self.signature)

    def hash(self):
        return hashlib.sha256(self.__str__().encode()).hexdigest()


class TxTo:
    def __init__(self, qty, public_key):
        self.qty = round(qty, precision)
        self.public_key = public_key

    def __str__(self):
        return '<to qty="{0:.{2}g}" public_key="{1}" />\n'.format(self.qty, self.public_key, precision)

    def hash(self):
        return hashlib.sha256(self.__str__().encode()).hexdigest()


class Transaction:
    def __init__(self, qty, unit, timestamp=None, description=None, native_currency=None, native_amount=None):
        self.qty = qty
        self.unit = unit
        if timestamp is None:
            self.timestamp = datetime.now()
        else:
            self.timestamp = timestamp
        if description is None:
            self.description = ""
        else:
            self.description = description
        self.native_currency = native_currency
        self.native_amount = native_amount
        self._froms = []
        self._tos = []

    def add_from(self, txid, vout, signature):
        self._froms.append(TxFrom(txid, vout, signature))

    def add_to(self, qty, public_key):
        self._tos.append(TxTo(qty, public_key))

    def __str__(self):
        try:
            extra = ' price="{}"'.format(float(self.native_amount) / float(self.qty))
        except TypeError:
            extra = ''
        message = '\t\t\t<transaction_content timestamp="{}" description="{}" qty="{}" unit="{}" native_amount="{}" native_currency="{}"{}>\n'.format(self.timestamp, self.description, self.qty, self.unit, self.native_amount, self.native_currency, extra)
        for txfro in self._froms:
            message += '\t\t\t\t{}\n'.format(txfro)
        for txto in self._tos:
            message += '\t\t\t\t{}\n'.format(txto)
        message += '\t\t\t</transaction_content>\n'
        return message

    def total_input(self, blockchain, unit):
        qty = 0.0
        for txfrom in self._froms:
            txto_solved = blockchain.get_txto(txfrom.txid, txfrom.vout)
            if self.unit == unit:
                qty += txto_solved.qty
        return qty

    def total_output(self, blockchain, unit):
        qty = 0.0
        for txto in self._tos:
            if self.unit == unit:
                qty += txto.qty
        return qty

    def hash(self):
        message = str(self.timestamp) + '\n'
        for txfro in self._froms:
            message += '{}\n'.format(txfro.hash())
        for txto in self._tos:
            message += '{}\n'.format(txto.hash())
        return hashlib.sha256(message.encode()).hexdigest()


class TransactionWrap:
    def __init__(self, data, sign, public):
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

    def doble_send(self, blockchain, from_wallet, from_supply, baseQty, baseUnit, to_supply, to_wallet, quoteQty, quoteUnit, timestamp=None, description=None, native_currency=None, native_amount=None):
        self.send(blockchain, from_wallet, from_supply, baseQty, baseUnit, timestamp=timestamp, description='1/2 ' + description, native_currency=native_currency, native_amount=native_amount)
        self.send(blockchain, to_supply, to_wallet, quoteQty, quoteUnit, timestamp=timestamp, description='2/2 ' + description, native_currency=native_currency, native_amount=native_amount)

    def send(self, blockchain, from_wallet, to_wallet, qty_, unit_, timestamp=None, description=None, native_currency=None, native_amount=None):

        transaction = Transaction(qty_, unit_, timestamp=timestamp, description=description, native_currency=native_currency, native_amount=native_amount)
        if from_wallet != blockchain._null_wallet:

            txtos = blockchain.calculate_unspent(from_wallet.public, qty_, unit_)

            for trans, txto, vout in txtos:
                serialized_txto = txto.hash()
                signature = sign_message(from_wallet.private, serialized_txto)
                transaction.add_from(trans.hash(), vout, signature)

            # deposit EUR -> JUV . viban_purchase -53.38 -> 5.0

            spent_qty = qty_
            group_txtos = defaultdict(float)
            for _, txto, _ in txtos:
                # el monto es mayor que la cantidad necesaria, devuelveme el cambio
                if txto.qty > spent_qty:
                    group_txtos[to_wallet.public] = group_txtos[to_wallet.public] + spent_qty
                    change = txto.qty - spent_qty
                    if change > threshold:
                        group_txtos[from_wallet.public] = group_txtos[from_wallet.public] + change
                    spent_qty -= spent_qty
                else:
                    group_txtos[to_wallet.public] = group_txtos[to_wallet.public] + txto.qty
                    spent_qty -= txto.qty
                if abs(spent_qty) <= threshold:
                    break

            '''
            Agrupar todos los TxTo que van a la misma dirección
            minimizar el número de salidas para ahorrar bytes
            '''
            for k, v in group_txtos.items():
                transaction.add_to(v, k)
            
            if abs(spent_qty) > threshold:
                print("error {}: {} {} {} {}".format(spent_qty, from_wallet, to_wallet, qty_, unit_))
        else:
            # txto foundation
            transaction.add_to(qty_, to_wallet.public)
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
                message += '\t\t\t<transaction order="{}" txid="{}">\n'.format(i, transaction.data.hash())
                message += str(transaction)
                message += '\t\t\t</transaction>\n'
            return '\t\t<transactions>\n{}\t\t</transactions>\n'.format(message)
        else:
            return ''


class PublicWallet:
    def __init__(self, unit, private_key, suffix=None):
        self.public = private_key_to_public_key(private_key)
        self.import_key = private_key_to_WIF(private_key)
        if suffix is not None:
            unit = unit + suffix
        self.address = public_key_to_address(self.public, unit)

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

    def balance(self, blockchain, unit, format=False):
        '''
        txto = income
        txfrom = expenses
        balance = sum(txto) - sum(txfrom)
        '''
        if format:
            return '{} {}'.format(blockchain.calculate_balance(self.public, unit), unit)
        else:
            return blockchain.calculate_balance(self.public, unit)


class PrivateWallet(PublicWallet):
    def __init__(self, unit, private_key=None, suffix=None):
        if private_key is None:
            self.private = generate_private_key()
        else:
            self.private = binascii.hexlify(private_key.encode('utf-8')).decode('utf-8')
        super().__init__(unit, self.private, suffix=suffix)


class NullWallet(PublicWallet):
    def __init__(self):
        self.unit = 'NULL'
        self.private = binascii.hexlify(('0'*size_bits).encode('utf-8')).decode('utf-8')
        super().__init__(self.unit, self.private)


class HashWallet(PrivateWallet):
    def __init__(self, name, unit):
        self.name = name
        self.unit = unit
        private = hashlib.sha256(name.encode()).hexdigest()[:size_bits]
        super().__init__(unit, private)


class FoundationWallet(PrivateWallet):
    def __init__(self, name, unit):
        self.name = name
        self.unit = unit
        private = hashlib.sha256(name.encode()).hexdigest()[:size_bits]
        super().__init__(self.unit, private, suffix="_FOUNDATION")


# dataset_csv = r"C:\Users\x335336\OneDrive - Santander Office 365\Documents\crypto_transactions_record_20211213_195307.csv"
dataset_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crypto_transactions_record_20211213_195307.csv")
df = pd.read_csv(dataset_csv)

seed = 1234
difficulty = 2
blockchain = Blockchain(seed, difficulty)
df.index = pd.to_datetime(df['Timestamp (UTC)'])
df.drop('Timestamp (UTC)', axis=1, inplace=True)
df = df.sort_index()

currencies = []
for index, row in df.iterrows():
    timestamp = index.to_pydatetime()
    currency = row['Currency']
    amount = row['Amount']
    native_currency = row['Native Currency']
    native_amount = row['Native Amount']
    to_currency = row['To Currency']
    to_amount = row['To Amount']
    kind = row['Transaction Kind']
    description = '{} . {} {} -> {}'.format(row['Transaction Description'], kind, amount, to_amount)

    if currency not in currencies:
        if isinstance(currency, str) or not math.isnan(currency):
            currencies.append(currency)
    if to_currency not in currencies:
        if isinstance(currency, str) or not math.isnan(to_currency):
            currencies.append(to_currency)

    if (kind == 'crypto_withdrawal') or (kind == 'crypto_wallet_swap_debited') or (kind == 'card_top_up'):

        from_foundation_wallet = blockchain.build_foundation_wallet('Central deposit {}'.format(currency), currency, super_supply)
        from_personal_wallet = blockchain.build_client_wallet('Personal {}'.format(currency), currency)

        blockchain.transfer(from_personal_wallet, from_foundation_wallet, -amount, currency, timestamp=timestamp, description=description, native_currency=native_currency, native_amount=native_amount)

    elif (kind == 'crypto_deposit') or (kind == 'crypto_wallet_swap_credited'):

        from_foundation_wallet = blockchain.build_foundation_wallet('Central deposit {}'.format(currency), currency, super_supply)
        from_personal_wallet = blockchain.build_client_wallet('Personal {}'.format(currency), currency)

        blockchain.transfer(from_foundation_wallet, from_personal_wallet, amount, currency, timestamp=timestamp, description=description, native_currency=native_currency, native_amount=native_amount)

    elif (kind == 'dynamic_coin_swap_debited') or (
            kind == 'dynamic_coin_swap_credited') or (
            kind == 'dynamic_coin_swap_bonus_exchange_deposit') or (
            kind == 'lockup_lock'):
        pass

    elif (kind == 'crypto_exchange') or (kind == 'crypto_viban_exchange'):

        from_foundation_wallet = blockchain.build_foundation_wallet('Central deposit {}'.format(currency), currency, super_supply)
        from_personal_wallet = blockchain.build_client_wallet('Personal {}'.format(currency), currency)

        to_foundation_wallet = blockchain.build_foundation_wallet('Central deposit {}'.format(to_currency), to_currency, super_supply)
        to_personal_wallet = blockchain.build_client_wallet('Personal {}'.format(to_currency), to_currency)

        blockchain.exchange(
            from_personal_wallet, from_foundation_wallet, -amount, currency,
            to_foundation_wallet, to_personal_wallet, to_amount, to_currency, timestamp=timestamp, description=description, native_currency=native_currency, native_amount=native_amount)

    elif (kind == 'reimbursement'):

        from_foundation_wallet = blockchain.build_foundation_wallet('Central deposit {}'.format(currency), currency, super_supply)
        from_personal_wallet = blockchain.build_client_wallet('Personal {}'.format(currency), currency)

        blockchain.send(from_foundation_wallet, from_personal_wallet, amount, currency, timestamp=timestamp, description=description, native_currency=native_currency, native_amount=native_amount)

    elif (kind == 'viban_purchase'):

        from_foundation_wallet = blockchain.build_foundation_wallet('Central deposit {}'.format(currency), currency, super_supply)
        from_personal_wallet = blockchain.build_client_wallet('Personal {}'.format(currency), currency)

        to_foundation_wallet = blockchain.build_foundation_wallet('Central deposit {}'.format(to_currency), to_currency, super_supply)
        to_personal_wallet = blockchain.build_client_wallet('Personal {}'.format(to_currency), to_currency)

        if currency == native_currency:
            blockchain.transfer(from_foundation_wallet, from_personal_wallet, -amount, currency, timestamp=timestamp, description='deposit ' + description, native_currency=native_currency, native_amount=native_amount)

        blockchain.exchange(
            from_personal_wallet, from_foundation_wallet, -amount, currency,
            to_foundation_wallet, to_personal_wallet, to_amount, to_currency, timestamp=timestamp, description=description, native_currency=native_currency, native_amount=native_amount)

    else:
        print(kind + ' --')
        print(row)

        
print('-- balances --')
for currency in currencies:
    if isinstance(currency, str):
        wallet = blockchain.build_client_wallet('Personal {}'.format(currency), currency)
        print('{} - balance: {} - profit: {}'.format(currency, wallet.balance(blockchain, currency, True), blockchain.get_profit(wallet, currency)))


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


'''


revisar:
1/2 EUR -> SOL . viban_purchase -219.8 -> 1.1788


'''

