import functools
import json
import base64
import hashlib
import numpy as np 
import binascii


# RSA calculator: https://www.cs.drexel.edu/~jpopyack/IntroCS/HW/RSAWorksheet.html
# TODO:
# ECDSA: https://kjur.github.io/jsrsasign/sample/sample-ecdsa.html

def public_key_to_address ( e, n ):
    address_algorithm = 'ripemd160'
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    count = 0
    val = 0
    var = hashlib.new ( address_algorithm )
    var.update ( hashlib.sha256 ( str((e, n)) ).digest ( ) )
    doublehash = hashlib.sha256 ( hashlib.sha256 ( binascii.unhexlify ( ('00' + var.hexdigest ( )).encode ( ) ) ).digest ( ) ).hexdigest ( )
    address = '00' + var.hexdigest ( ) + doublehash [ 0:8 ]
    for char in address:
        if (char != '0'):
            break
        count += 1
    count = count // 2
    n = int ( address, 16 )
    output = [ ]
    while (n > 0):
        n, remainder = divmod ( n, 58 )
        output.append ( alphabet [ remainder ] )
    while (val < count):
        output.append ( alphabet [ 0 ] )
        val += 1
    return ''.join ( output [ ::-1 ] )[:9]

def hash_data( message ):
    h = hashlib.md5(message)
    return h.digest().encode('base64')[:6]

def sign_message ( message, d, n ):
    hash = hash_data( message )
    sign = encript( hash, d, n )
    sign_str = key_to_str(sign)
    return sign_str

def verify_message ( message, sign_str, e, n ):
    sign = str_to_key(sign_str)
    return decrypt( sign, e, n ) == hash_data( message )

def generate_random_primes ( seed, len=2 ):
    generated = 0
    np.random.seed ( seed )
    st0 = np.random.get_state ( )
    try:
        n = 10000 + np.random.randint(100001, 80000001)
        while True:
            # numeros de Marsenne
            # candidate = bin_pow(2, n, msb=True) - 1
            if n%2!=0 and n%3!=0 and n%5!=0:
                candidate = n
                if isPrime(candidate):
                    generated += 1
                    yield candidate
                    if generated >= len:
                        break
            n += np.random.randint(1100, 810000000)
    finally:
        np.random.set_state ( st0 )
        
def search_candidates_1_mod_r( r ):
    n = r + 1
    for _ in range(30):
        yield n
        n += r

def isPrime(n, k=5): # miller-rabin
    from random import randint
    if n < 2: return False
    for p in [2,3,5,7,11,13,17,19,23,29]:
        if n % p == 0: return n == p
    s, d = 0, n-1
    while d % 2 == 0:
        s, d = s+1, d/2
    for i in range(k):
        x = pow(randint(2, n-1), d, n)
        if x == 1 or x == n-1: continue
        for r in range(1, s):
            x = (x * x) % n
            if x == 1: return False
            if x == n-1: break
        else: return False
    return True


def factors(n, b2=-1, b1=10000): # 2,3,5-wheel, then rho
    def gcd(a,b): # euclid's algorithm
        if b == 0: return a
        return gcd(b, a%b)
    def insertSorted(x, xs): # linear search
        i, ln = 0, len(xs)
        while i < ln and xs[i] < x: i += 1
        xs.insert(i,x)
        return xs
    if -1 <= n <= 1: return [n]
    if n < -1: return [-1] + factors(-n)
    wheel = [1,2,2,4,2,4,2,4,6,2,6]
    w, f, fs = 0, 2, []
    while f*f <= n and f < b1:
        while n % f == 0:
            fs.append(f)
            n /= f
        f, w = f + wheel[w], w+1
        if w == 11: w = 3
    if n == 1: return fs
    h, t, g, c = 1, 1, 1, 1
    while not isPrime(n):
        while b2 <> 0 and g == 1:
            h = (h*h+c)%n # the hare runs
            h = (h*h+c)%n # twice as fast
            t = (t*t+c)%n # as the tortoise
            g = gcd(t-h, n); b2 -= 1
        if b2 == 0: return fs
        if isPrime(g):
           while n % g == 0:
                fs = insertSorted(g, fs)
                n /= g
        h, t, g, c = 1, 1, 1, c+1
    return insertSorted(n, fs)


def fastmodpow( base, exp, m ):
   result = 1
   while (exp > 0):
      if ((exp & 1) > 0):
          result = (result * base) % m
      exp >>= 1
      base = (base * base) % m
   return result

def encript( m, d, n ):
    m = [ fastmodpow( ord( x ), d, n ) for x in m ]
    return m

def decrypt( c, e, n ):
    m = []
    for i in c:
        m.append( chr( fastmodpow( i, e, n ) ) )
    return ''.join(m)

def key_to_str(key):
    return base64.b64encode(
        json.dumps(key).encode('utf-8')
    )

def str_to_key(text):
    return json.loads(
        base64.b64decode(text).encode('utf-8')
    )

generate_count = 100000  # for test collisions
wallets = {}
for seed in range(generate_count):
    print('seed {} ...'.format(seed))
    primes = list( generate_random_primes( seed ) )
    print(primes)
    p = primes[0]
    q = primes[1]
    n = p * q
    phi = (p-1) * (q-1)

    e = None
    d = None
    found = False
    candidates = list( search_candidates_1_mod_r( phi ) )
    while len(candidates) > 0:
        lastone = candidates.pop()
        facts = list(factors(lastone))
        if len(facts) > 1:
            others = facts[:-1]
            e = functools.reduce(lambda x,y: x * y, others)
            d = facts[-1]
            found = True
            break

    if not found:
        raise Exception('Not found d and e')

    assert((e*d) % phi == 1)

    address = public_key_to_address(e, n)

    print('wallet address: {}'.format( address ))
    print('private key: {}'.format( d ))
    
    if address not in wallets:
        wallets[address] = (e, d, n)
    else:
        raise Exception('Collision on {}, {}, {}.'.format(e, d, n))

    message = '''
    JUAN -> ANTONIO 1 BTC
    PEPE -> JUAN 2 BTC
    JUAN -> ANTONIO 1 BTC
    PEPE -> JUAN 2 BTC
    '''

    sign = sign_message(message, d, n)
    print(message)
    print(sign)

    if(verify_message(message, sign, e, n)):
        print('Firma valida')
    else:
        print('Mensaje con firma invalida')
        raise Exception('Firma invalida')

assert(len(wallets) == generate_count)
