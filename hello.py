import os
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import QuantLib as ql
import pandas_ta as ta
import yfinance as yf
from datetime import datetime, timedelta
from arch import arch_model
from arch.univariate import ConstantMean, GARCH, Normal, StudentsT, HARX
from sklearn.metrics import mean_squared_error


def pdf(x, m=0, v=1):
    return math.exp(-((x - m) ** 2) / (2 * (v**2))) / (v * math.sqrt(2 * np.pi))


def cdf(x, m=0, v=1):
    return (math.erf((math.sqrt(2) * (x - m)) / (2 * v)) / 2) + 0.5


def implied_vol(S, K, r, q, Vol_guess, T, market_price, flag='c', tol=0.00001, lr=10):
    """Compute the implied volatility of a European Option
        S0: initial stock price
        K:  strike price
        T:  maturity
        r:  risk-free rate
        market_price: market observed price
        tol: user choosen tolerance
    """
    max_iter = 2000  #max number of iterations
    vol_old = Vol_guess #initial guess
    for k in range(max_iter):
        if flag == 'c':
            bs_price = BSMCall(S, K, r, q, vol_old, T)
        else:
            bs_price = BSMPut(S, K, r, q, vol_old, T)

        # calculate vega
        d_uno = d1(S, K, r, q, vol_old, T)
        Cprime = (S * math.sqrt(T) * pdf(d_uno)) * lr

        C = bs_price - market_price
        vol_new = vol_old - C/Cprime
        if vol_new < 0.0:
            return vol_old
        if flag == 'c':
            bs_new = BSMCall(S, K, r, q, vol_new, T)
        else:
            bs_new = BSMPut(S, K, r, q, vol_new, T)
        if abs(bs_new - market_price) < tol:
            break
        vol_old = vol_new

    implied_vol = vol_old
    return implied_vol


def get_marketdata(ticker):
    path = os.path.join('csv', '{}.csv'.format(ticker.lower().replace('^', '')))

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
        df = yf.download(ticker.upper(), period="5y", interval='1d', proxy=os.getenv('PROXY_HTTPS_SANTANDER'))
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
    return (S * np.exp(-q * T) * cdf(d_uno)) - (K * np.exp(-r * T) * cdf(d_dos))

def BSMPut(S, K, r, q, Vol, T):
    d_uno = d1(S, K, r, q, Vol, T)
    d_dos = d2(d_uno, Vol, T)
    return (K * np.exp(-r * T) * cdf(-d_dos)) - (S * np.exp(-q * T) * cdf(-d_uno))

def BSMCallGreeks(S, K, r, q, Vol, T):
    d_uno = d1(S, K, r, q, Vol, T)
    d_dos = d2(d_uno, Vol, T)
    Delta = cdf(d_uno)
    Gamma = pdf(d_uno) / (S * Vol * np.sqrt(T))
    Theta = -(S * Vol * pdf(d_uno)) / (2 * np.sqrt(T) - r*K*np.exp(-r*T) * cdf(d_dos))
    Vega = S * np.sqrt(T) * pdf(d_uno)
    Rho = K * T * np.exp(-r*T) * cdf(d_dos)
    return Delta, Gamma, Theta, Vega , Rho

def BSMPutGreeks(S, K, r, q, Vol, T):
    d_uno = d1(S, K, r, q, Vol, T)
    d_dos = d2(d_uno, Vol, T)
    Delta = cdf(-d_uno)
    Gamma = pdf(d_uno) / (S * Vol * np.sqrt(T))
    Theta = -(S*Vol*pdf(d_uno)) / (2*np.sqrt(T))+ r*K * np.exp(-r*T) * cdf(-d_uno)
    Vega = S * np.sqrt(T) * pdf(d_uno)
    Rho = -K*T*np.exp(-r*T) * cdf(-d_dos)
    return Delta, Gamma, Theta, Vega, Rho


if __name__ == '__main__':
    df_free, _ = get_marketdata('^tnx')

    # df_sp500 = pd.read_csv('sp500.csv')
    # tickers = []
    # for index, row in df_sp500.iterrows():
    #     tickers.append(row['Ticker'])
    tickers = ['TSLA', ]
    call_market_price = 0.02
    put_market_price = 27.93
    tol = 0.5
    strike_price = 305

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

            # df = df[-(24 * 5 * 2):]
            # df = df[1:len(df)-1]

            df['S_+'] = np.log(df.Close.shift(+1) / df['Close'])
            df['S_-'] = np.log(df['Close'] / df.Close.shift(-1))

            df.dropna(inplace=True)

            S0 = df.Close[0]
            S = df.Close[-1]
            T = 1.0
            periods = 252
            dt = T / periods
            # log_returns = ta.log_return(close=df.Close)
            df.index = pd.to_datetime(df.index)
            returns = df.Close.pct_change().dropna()
            sigma = np.sqrt(periods) * returns.std()  # annual volatility in %

            # mu = (np.log(S0) + (log_returns.mean() - (log_returns.var() / 2.0))) * dt
            mu = df['S_+'].mean()
            # std = log_returns.var() * dt
            std = df['S_+'].std(ddof=0)

            n = len(df)

            # Model price
            params = {
                "mean": 'AR',
                # "lags": 1,
                "vol": 'harch',
                # "p": [1, 3, 22],
            }
            # index = log_returns.index
            scaled_returns = 100.0 * returns
            am = arch_model(scaled_returns, **params)
            end = '2019-12-01'
            last_obs = end

            # res = am.fit()
            # res = am.fix(params=res.params)
            res = am.fit(disp="off", first_obs=0, last_obs=600, )
            # fig = res.hedgehog_plot()
            # plt.show()

            sim = arch_model(None, **params)
            sim_df = sim.simulate(res.params, n)
            x = sim_df.data
            # check model

            '''
            print(returns.describe())
            sns.distplot(x, color='g')
            sns.distplot(scaled_returns, color='r')
            plt.legend()
            plt.show()
            '''

            mse = mean_squared_error(scaled_returns, x[:-1])
            print(f'mse: {mse}')

            # https://www.codearmo.com/blog/pricing-options-monte-carlo-simulation-python

            # browniano
            q = dividends_rate  # annual dividend rate
            steps = periods  # time steps
            N = 10000  # number of trials
            r = df_free.Close[-1] / 100.0
            nominal = 1
            K = strike_price
            print(f'<{ticker}>: price: {S}, strike: {K} volatility (historic): {100.0*sigma}, risk_free: {100.0*r}, dividend: {100.0*q}.')

            paths = geo_paths(S, T, q, sigma, steps, N, r)

            # forecasts = res.forecast(horizon=3, reindex=False, simulations=2)
            # print('-- mean --')
            # print(forecasts.mean)
            # print('-- residual variance --')
            # print(forecasts.residual_variance)
            # print('-- variance --')
            # print(forecasts.variance)

            '''
            dt = T / steps
            r = forecasts.mean
            sigma = np.sqrt(forecasts.variance)
            ST = np.log(S) + np.cumsum(((r - sigma ** 2 / 2) * dt + \
                                        sigma * np.sqrt(dt) * \
                                        forecasts.residual_variance), axis=0)
            paths2 = np.exp(ST)
            paths = paths2.T.columns.tolist()
            '''

            # view paths
            # plt.plot(paths)
            # plt.xlabel("Time Increments")
            # plt.ylabel("Stock Price")
            # plt.title("Geometric Brownian Motion")
            # plt.show()

            # ultima columna de resultados
            last = paths[-1]
            # diferencia respecto al strike
            # parte postiva
            positive_payoffs = np.maximum(last - K, 0)
            negative_payoffs = np.minimum(last - K, 0)
            negative_payoffs2 = -np.maximum(K - last, 0)
            assert(np.array_equal(negative_payoffs, negative_payoffs2))
            # precio medio (esperanza matematica)
            mean_positive_payoffs = np.mean(positive_payoffs)
            mean_negative_payoffs = np.mean(negative_payoffs)
            # traer a presente con interes compuesto continuo
            call_option_price = mean_positive_payoffs * np.exp(-r * T)  # discounting back to present value
            put_option_price = -mean_negative_payoffs * np.exp(-r * T)  # discounting back to present value

            print(f"Montecarlo CALL option price is {call_option_price * nominal}")
            print(f"Montecarlo PUT option price is {put_option_price * nominal}")

            implied_vol_call = implied_vol(S, K, r, q, sigma, T, call_market_price, flag='c', tol=tol)
            print("Montecarlo CALL implied vol is ", implied_vol_call)

            implied_vol_put = implied_vol(S, K, r, q, sigma, T, put_market_price, flag='p', tol=tol)
            print("Montecarlo PUT implied vol is ", implied_vol_put)

            ############################

            day_count = ql.Business252()
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
            implied_vol_call = implied_vol(S, K, r, q, sigma, T, call_market_price, flag='c', tol=tol)
            print("Quantlib CALL implied vol is ", implied_vol_call)


            # construct the European Option
            payoff = ql.PlainVanillaPayoff(ql.Option.Put, K)
            exercise = ql.EuropeanExercise(maturity_date)
            european_option_put = ql.VanillaOption(payoff, exercise)
            european_option_put.setPricingEngine(engine)
            bs_price = european_option_put.NPV() * nominal
            print("Quantlib PUT option price is ", bs_price)
            row.append(bs_price)
            implied_vol_put = implied_vol(S, K, r, q, sigma, T, put_market_price, flag='p', tol=tol)
            print("Quantlib PUT implied vol is ", implied_vol_put)

            print(f"Black Scholes CALL option price is {BSMCall(S, K, r, q, sigma, T) * nominal}")
            print(f"Black Scholes PUT option price is {BSMPut(S, K, r, q, sigma, T) * nominal}")
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
    
    
