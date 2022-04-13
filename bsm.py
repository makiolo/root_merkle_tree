'''
requirements.txt:
yfinance
pandas
matplotlib
seaborn
pandas-ta
arch
datapackage
numpy
QuantLib
'''
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import QuantLib as ql
import pandas_ta as ta
import yfinance as yf
from datetime import datetime, timedelta
from scipy.stats import norm, lognorm
from arch import arch_model


def get_marketdata(ticker):
    path = '{}.csv'.format(ticker.lower().replace('^', ''))

    tick = yf.Ticker(ticker.upper())
    now = datetime.now()
    year_ago = now - timedelta(days=365)
    mask = (year_ago < tick.dividends.index) & (tick.dividends.index <= now)
    dividends = tick.dividends.loc[mask]

    '''
    print(f'tick.cashflow = {tick.cashflow}')
    print(f'tick.quarterly_cashflow = {tick.quarterly_cashflow}')
    print(f'tick.earnings = {tick.earnings}')
    print(f'tick.quarterly_earnings = {tick.quarterly_earnings}')
    '''

    if not os.path.exists(path):
        print('download {} ...'.format(path))
        df = yf.download(ticker.upper(), period="1mo", interval='60m', proxy=os.getenv('PROXY_HTTPS_SANTANDER'))
        df.to_csv(path)
    else:
        df = pd.read_csv(path, index_col=0)

    dividends_rate = dividends.sum() / df.Close[-1]

    return df, dividends_rate


def geo_paths(S, T, q, sigma, steps, N, r):
    """
    Inputs
    #S = Current stock Price
    #K = Strike Price
    #T = Time to maturity 1 year = 1, 1 months = 1/12
    #q = dividend yield
    #sigma = volatility
    #r = risk free interest rate

    Output
    # [steps,N] Matrix of asset paths
    """
    dt = T / steps
    # S_{T} = ln(S_{0})+\int_{0}^T(\mu-\frac{\sigma^2}{2})dt+\int_{0}^T \sigma dW(t)
    ST = np.log(S) + np.cumsum(((r - q - sigma ** 2 / 2) * dt + \
                                sigma * np.sqrt(dt) * \
                                np.random.normal(size=(steps, N))), axis=0)

    return np.exp(ST)


def d1(S, K, r, q, Vol, T):
    return (np.log(S / K) + (r - q + Vol ** 2 / 2) * T) / Vol * np.sqrt(T)

def d2(d1, Vol, T):
    return d1 - Vol * np.sqrt(T)

def BSMCall(S, K, r, q, Vol, T):
    d_uno = d1(S, K, r, q, Vol, T)
    d_dos = d2(d_uno, Vol, T)
    return (S * np.exp(-q * T) * norm.cdf(d_uno)) - (K * np.exp(-r * T) * norm.cdf(d_dos))

def BSMPut(S, K, r, q, Vol, T):
    d_uno = d1(S, K, r, q, Vol, T)
    d_dos = d2(d_uno, Vol, T)
    return (K * np.exp(-r * T) * norm.cdf(-d_dos)) - (S * np.exp(-q * T) * norm.cdf(-d_uno))

def BSMCallGreeks(S, K, r, q, Vol, T):
    d_uno = d1(S, K, r, q, Vol, T)
    d_dos = d2(d_uno, Vol, T)
    Delta = norm.cdf(d_uno)
    Gamma = norm.pdf(d_uno) / (S * Vol * np.sqrt(T))
    Theta = -(S * Vol * norm.pdf(d_uno)) / (2 * np.sqrt(T) - r*K*np.exp(-r*T)*norm.cdf(d_dos))
    Vega = S * np.sqrt(T) * norm.pdf(d_uno)
    Rho = K * T * np.exp(-r*T) * norm.cdf(d_dos)
    return Delta, Gamma, Theta, Vega , Rho

def BSMPutGreeks(S, K, r, q, Vol, T):
    d_uno = d1(S, K, r, q, Vol, T)
    d_dos = d2(d_uno, Vol, T)
    Delta = norm.cdf(-d_uno)
    Gamma = norm.pdf(d_uno) / (S * Vol * np.sqrt(T))
    Theta = -(S*Vol*norm.pdf(d_uno)) / (2*np.sqrt(T))+ r*K * np.exp(-r*T) * norm.cdf(-d_uno)
    Vega = S * np.sqrt(T) * norm.pdf(d_uno)
    Rho = -K*T*np.exp(-r*T) * norm.cdf(-d_dos)
    return Delta, Gamma, Theta, Vega, Rho

if __name__ == '__main__':
    df_free, _ = get_marketdata('^tnx')

    df_sp500 = pd.read_csv('sp500.csv')
    tickers = []
    for index, row in df_sp500.iterrows():
        tickers.append(row['Ticker'])

    scheme = ['name', 'dividends',
              'quantlib_call', 'quantlib_put', 'bsm_call', 'bsm_put', 'mc_call', 'mc_put',
              'quantlib_call_delta', 'quantlib_call_gamma', 'quantlib_call_theta', 'quantlib_call_vega', 'quantlib_call_rho',
              'quantlib_put_delta', 'quantlib_put_gamma', 'quantlib_put_theta', 'quantlib_put_vega', 'quantlib_put_rho',
              'bsm_call_delta', 'bsm_call_gamma', 'bsm_call_theta', 'bsm_call_vega', 'bsm_call_rho',
              'bsm_put_delta', 'bsm_put_gamma', 'bsm_put_theta', 'bsm_put_vega', 'bsm_put_rho',
              ]
    dataset = []

    try:
        for ticker in tickers:

            row = []

            try:
                df, dividends_rate = get_marketdata(ticker)
            except TypeError as e:
                continue
            except IndexError as e:
                continue

            row.append(ticker)
            row.append(dividends_rate)

            '''
            # df = df[-(24 * 5 * 2):]
            # df = df[1:len(df)-1]

            df['S_+'] = np.log(df.Close.shift(+1) / df['Close'])
            df['S_-'] = np.log(df['Close'] / df.Close.shift(-1))

            df.dropna(inplace=True)

            mu = df['S_+'].mean()
            # print(df['S_-'].mean())
            std = df['S_+'].std()
            # print(df['S_-'].std())

            n = len(df)

            x = np.log(np.random.lognormal(mu, std, n))
            sns.distplot(x, color='g')  # log normal distribution

            am = arch_model(df['S_+'])
            res = am.fit()
            print(res)

            # sns.displot(df, x='S_-', y='S_+')
            sns.distplot(df['S_+'], color='r')
            # sns.distplot(df['S_-'])
            plt.legend()
            plt.show()
            '''

            # https://www.codearmo.com/blog/pricing-options-monte-carlo-simulation-python

            # browniano
            periods = 252
            S = df.Close[-1]
            T = 1.0
            q = dividends_rate  # annual dividend rate
            log_returns = ta.log_return(close=df.Close)
            sigma = np.sqrt(periods) * log_returns.std()  # annual volatility in %
            steps = 1000  # time steps
            N = 10000  # number of trials
            # esperanza matematica
            r = df_free.Close[-1] / 100.0
            nominal = 10
            K = S
            print(f'<{ticker}>: price: {S}, volatility: {sigma}, risk_free: {100.0*r}, dividend: {100.0*q}.')

            paths = geo_paths(S, T, q, sigma, steps, N, r)

            '''
            plt.plot(paths)
            plt.xlabel("Time Increments")
            plt.ylabel("Stock Price")
            plt.title("Geometric Brownian Motion")
            plt.show()
            '''

            # ultima columna de resultados
            last = paths[-1]
            # diferencia respecto al strike
            diff = last - K
            # parte postiva
            positive_payoffs = np.maximum(diff, 0)
            negative_payoffs = np.minimum(diff, 0)
            # precio medio (esperanza matematica)
            mean_positive_payoffs = np.mean(positive_payoffs)
            mean_negative_payoffs = np.mean(negative_payoffs)
            # traer a presente con interes compuesto continuo
            call_option_price = mean_positive_payoffs * np.exp(-r * T)  # discounting back to present value
            put_option_price = -mean_negative_payoffs * np.exp(-r * T)  # discounting back to present value

            ############################

            day_count = ql.Actual365Fixed()
            initialValue = ql.QuoteHandle(ql.SimpleQuote(S))
            today = ql.Date().todaysDate()
            riskFreeTS = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count))
            dividendTS = ql.YieldTermStructureHandle(ql.FlatForward(today, q, day_count))
            volatility = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), sigma, day_count))
            process = ql.BlackScholesMertonProcess(initialValue, dividendTS, riskFreeTS, volatility)

            # steps = 2
            # rng = "pseudorandom" # could use "lowdiscrepancy"
            # numPaths = 100000
            # engine = ql.MCEuropeanEngine(process, rng, steps, requiredSamples=numPaths)
            engine = ql.AnalyticEuropeanEngine(process)
            # tGrid, xGrid = 2000, 200
            # engine = ql.FdBlackScholesVanillaEngine(process, tGrid, xGrid)

            # option data
            calculation_date = ql.Date(1, 1, 2022)
            maturity_date = ql.Date(31, 12, 2022)

            calendar = ql.TARGET()

            # schedule = ql.MakeSchedule(calculation_date, maturity_date, ql.Period('1d'))
            # print(len(list(schedule)))

            ql.Settings.instance().evaluationDate = calculation_date

            # construct the European Option
            payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
            exercise = ql.EuropeanExercise(maturity_date)
            european_option_call = ql.VanillaOption(payoff, exercise)
            european_option_call.setPricingEngine(engine)
            bs_price = european_option_call.NPV() * nominal
            print("Quantlib CALL option price is ", bs_price)
            row.append(bs_price)

            # construct the European Option
            payoff = ql.PlainVanillaPayoff(ql.Option.Put, K)
            exercise = ql.EuropeanExercise(maturity_date)
            european_option_put = ql.VanillaOption(payoff, exercise)
            european_option_put.setPricingEngine(engine)
            bs_price = european_option_put.NPV() * nominal
            print("Quantlib PUT option price is ", bs_price)
            row.append(bs_price)


            print(f"Black Scholes CALL option price is {BSMCall(S, K, r, q, sigma, T) * nominal}")
            print(f"Black Scholes PUT option price is {BSMPut(S, K, r, q, sigma, T) * nominal}")
            print(f"Montecarlo CALL option price is {call_option_price * nominal}")
            print(f"Montecarlo PUT option price is {put_option_price * nominal}")
            row.append(BSMCall(S, K, r, q, sigma, T) * nominal)
            row.append(BSMPut(S, K, r, q, sigma, T) * nominal)
            row.append(call_option_price * nominal)
            row.append(put_option_price * nominal)


            print(f'    Quantlib CALL Delta: {european_option_call.delta()}')
            print(f'    Quantlib CALL Gamma: {european_option_call.gamma()}')
            print(f'    Quantlib CALL Theta: {european_option_call.theta()}')
            print(f'    Quantlib CALL Vega: {european_option_call.vega()}')
            print(f'    Quantlib CALL Rho: {european_option_call.rho()}')
            print(f'    Quantlib PUT Delta: {european_option_put.delta()}')
            print(f'    Quantlib PUT Gamma: {european_option_put.gamma()}')
            print(f'    Quantlib PUT Theta: {european_option_put.theta()}')
            print(f'    Quantlib PUT Vega: {european_option_put.vega()}')
            print(f'    Quantlib PUT Rho: {european_option_put.rho()}')

            row.append(european_option_call.delta())
            row.append(european_option_call.gamma())
            row.append(european_option_call.theta())
            row.append(european_option_call.vega())
            row.append(european_option_call.rho())
            row.append(european_option_put.delta())
            row.append(european_option_put.gamma())
            row.append(european_option_put.theta())
            row.append(european_option_put.vega())
            row.append(european_option_put.rho())

            Delta, Gamma, Theta, Vega, Rho = BSMCallGreeks(S, K, r, q, sigma, T)
            print(f'    Formula CALL Delta: {Delta}')
            print(f'    Formula CALL Gamma: {Gamma}')
            print(f'    Formula CALL Theta: {Theta}')
            print(f'    Formula CALL Vega: {Vega}')
            print(f'    Formula CALL Rho: {Rho}')
            row.append(Delta)
            row.append(Gamma)
            row.append(Theta)
            row.append(Vega)
            row.append(Rho)

            Delta, Gamma, Theta, Vega, Rho = BSMPutGreeks(S, K, r, q, sigma, T)
            print(f'    Formula PUT Delta: {Delta}')
            print(f'    Formula PUT Gamma: {Gamma}')
            print(f'    Formula PUT Theta: {Theta}')
            print(f'    Formula PUT Vega: {Vega}')
            print(f'    Formula PUT Rho: {Rho}')
            row.append(Delta)
            row.append(Gamma)
            row.append(Theta)
            row.append(Vega)
            row.append(Rho)

            dataset.append(row)
    except KeyboardInterrupt as e:
        print('Interrupted by user ...')

    result = pd.DataFrame(dataset, columns=scheme)
    result.to_csv('result.csv')
