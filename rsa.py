import functools
import json
import base64
import math
import hashlib
from paramiko.util import mod_inverse


def hash_data( message ):
    h = hashlib.md5(str(message))
    return h.digest().encode('base64')[:6]

def sign_message ( message, d, n ):
    hash = hash_data( message )
    sign = encript( hash, d, n )
    sign_str = key_to_str(sign)
    return sign_str

def verify_message ( message, sign_str, e, n ):
    sign = str_to_key(sign_str)
    return decrypt( sign, e, n ) == hash_data( message )

def is_prime(n):
    fac = list(factors(n))
    return len(fac) == 1
    
def generate_random_primes ( seed, len=2 ):
    i = 0
    n = seed
    while True:
        if is_prime(n):
            i += 1
            yield n
            if i >= len:
                break
            n += 1000
        else:
            n += 1
        
def search_candidates_1_mod_r( r ):
    i = 3
    while True:
        if r%i == 1:
            yield i
        if i > 30*r:
            break
        i += 1

def factors(n):
    j = 2
    while n > 1:
        for i in xrange(j, int(math.sqrt(n+0.05)) + 1):
            if n % i == 0:
                n /= i ; j = i
                yield i
                break
        else:
            if n > 1:
                yield n; break

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

# RSA calculator: https://www.cs.drexel.edu/~jpopyack/IntroCS/HW/RSAWorksheet.html
seed = int(1234)
primes = list( generate_random_primes( seed ) )
p = primes[0]
q = primes[1]
n = p * q
phi = (p-1) * (q-1)

candidates = list( search_candidates_1_mod_r( phi ) )
facts = list(factors(max(candidates)))
others = facts[:-1]
if len(others) > 1:
    e = functools.reduce(lambda x,y: x + y, others)
elif len(others) == 1:
    e = others[0]
else:
    e = 65537
d = mod_inverse(e, phi) # private key

print('wallet address: {}'.format( hash_data( e ) ))
print('private key: {}'.format( d ))

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
