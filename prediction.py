# https://www.youtube.com/watch?v=xUuWgrPZedo
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
from scipy.stats import norm, lognorm
import yfinance as yf
import sympy as sp
import sympy.core
from scipy.interpolate import CubicSpline


ticker = 'NZDUSD=X'
model = 'gbm'  # gbm # jump
paths = 200
multiplier = 1
steps = 255 * multiplier
t0 = 0.0
T = 10.0
lambda_ = 20.0 / 365.0  # rate ocurrence of a jump
mu_j = 0.89
sigma_j = 0.77
dt = 1.0 / steps
total_steps = int((T - t0) * steps)


close = yf.download(ticker, '2019-01-01', '2023-12-31', progress=False)['Close']
# returns = np.log(close / close.shift(1))
returns = np.diff(np.log(close))

S0 = close[0]
sigma = (returns.std() / np.sqrt(multiplier)) * (1.0 / np.sqrt(dt))
mu = ((np.mean(returns) / multiplier) * (1.0 / dt))

print('-- real --')
print(mu)
print(sigma)



def pdf(x, mu, sigma):
    # analizas un stock y encajas su mejor "pdf"
    # return lamb * sp.exp(-lamb * x)
    # return (1.0 / (sigma * sp.sqrt(2 * sp.pi))) * sp.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return norm.pdf(x, mu, sigma)


def cdf(x, mu, sigma):
    return norm.cdf(x, mu, sigma)


xx, yy = sp.symbols('x y')
# pdf_numpy = sp.lambdify(xx, pdf(xx, mu, sigma), 'numpy')
pdf_numpy = lambda xxx: pdf(xxx, mu, sigma)
cdf_numpy = lambda xxx: cdf(xxx, mu, sigma)
# cdf_integrated = sp.integrate(pdf(xx, mu, sigma), xx)
# cdf_sympy = sp.lambdify((xx, mu, sigma), cdf_numpy, 'sympy')
# cdf_numpy = cdf_integrated

x = np.linspace(-1.5, +1.5, 100000)
f = pdf_numpy(x)
F = cdf_numpy(x)

# Inverse por formula
# cdf_inverse3 = norm.ppf(F, 0, 1)

# Inversa por analisis
# eq = sp.Eq(cdf_sympy(yy), xx)
# soluciones = sp.solve(eq, yy)
# print(soluciones)
# cdf_inverse3 = sp.lambdify(xx, soluciones[1], 'numpy')(x)

# Inversa numerica
# ver: https://www.youtube.com/watch?v=U00Kseb6SB4
cdf_inverse3 = x[np.searchsorted(F[:-1], x)]

# F1 = cdf_inverse2(x)
# print(F1)
# F2 = cdf_inverse3(x)
# print(F2)

plt.figure(figsize=(8, 3))
plt.plot(x, f, label=r'$f(x)$')
plt.plot(x, F, label=r'$F(x)$')
# plt.plot(x, F1, label=r'$F^-1(x)$')
plt.plot(x, cdf_inverse3, label=r'$F^-1(y)$')
plt.legend()
plt.show()

Z = np.random.rand(100000)
cdf_inverse_Z2 = CubicSpline(x, cdf_inverse3)(Z)
plt.figure(figsize=(8, 3))
plt.plot(x, f, label=r'$f(x)$')
plt.hist(cdf_inverse_Z2, histtype='step', color='green', density='norm', bins=100, label=r'$F^-1(z)$')
plt.legend()
plt.show()


#### simulation

prices = np.zeros((paths, total_steps))
prices[:, 0] = S0

# utilizas "cdf_inverse2" para simular un precio con su propio CDF inverso

for i in range(1, int(total_steps)):
    if model == 'gbm':
        Z = np.random.rand(paths)
        Z = (CubicSpline(x, cdf_inverse3)(Z) - mu) / sigma
        # Z = np.random.normal(0, 1, paths)
        # GBM
        prices[:, i] = prices[:, i - 1] * np.exp((mu - sigma ** 2.0 / 2.0) * dt + sigma * np.sqrt(dt) * Z)
    elif model == 'jump':
        # Jump diffusion
        Z = np.random.rand(paths)
        Z = (CubicSpline(x, cdf_inverse3)(Z) - mu) / sigma
        # Z = np.random.normal(0, 1, paths)
        # random numbers for the jump
        J = np.random.normal(mu_j, sigma_j, paths)
        prices[:, i] = prices[:, i - 1] * np.exp(
            (mu - sigma ** 2 / 2 - lambda_ * (np.exp(mu_j * sigma_j ** 2 / 2) - 1)) * dt + sigma * np.sqrt(dt) * Z)
    else:
        raise Exception('Invalid {}'.format(model))

plot(prices.T)
show()

returns = np.diff(np.log(prices))
sigma = (returns.std() / np.sqrt(multiplier)) * (1.0 / np.sqrt(dt))
mu = ((np.mean(returns) / multiplier) * (1.0 / dt)) - ((sigma ** 2) / 2)
print('-- simulado --')
print(mu)
print(sigma)

