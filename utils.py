from datetime import datetime

import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import pandas_datareader.data as web
from retry.api import retry
from cache_to_disk import cache_to_disk

from config import n_output, n, interest_rates, reverse_rates, interest_full
from config_base import bb_length, bb_std, ema_fast, ema_slow

tries = 0


@retry(AttributeError, tries=10, delay=1, backoff=2, max_delay=4)
def read_from_metatrader(ticket, period, backbars, step=5000, min_bars=24):
    global tries
    if tries > 1:
        backbars -= (step * tries)
    if backbars < min_bars:
        backbars = min_bars
    print(f'Ticket {ticket}, reading {backbars} bars ...')
    tries += 1
    rates = mt5.copy_rates_from_pos(ticket, period, 0, backbars)
    df = pd.DataFrame(rates)
    df.index = df.time.apply(lambda x: datetime.utcfromtimestamp(x))
    df = df.loc[:, ['open', 'close', 'high', 'low', 'tick_volume']]
    df['volume'] = df['tick_volume']
    df['adj_close'] = df['close']
    df.drop(['tick_volume', ], axis=1, inplace=True)
    return df


def download_dataset(tickets, period, backbars, predict=False, technical=True):
    global tries

    dataframes = []

    for i, ticket in enumerate(tickets):

        tries = 0
        df = read_from_metatrader(ticket, period, backbars)

        if i == 0:
            df['DayOfWeek'] = df.index.dayofweek
            df['Hour'] = df.index.hour

        # decomp = sea.seasonal_decompose(df['close'], model='additive', period=5 * 24)
        # df[ticket + '_Trend'] = decomp.trend
        # df[ticket + '_Seasonal'] = decomp.seasonal
        # df[ticket + '_Resid'] = decomp.resid

        if technical:
            # hour to daily
            # df_daily = df.resample('D').mean()
            df.ta.atr(append=True, suffix=ticket)
            df.ta.bbands(close='close', suffix=ticket, append=True, length=bb_length, std=bb_std)
            # df_daily = df_daily.dropna()
            # df_daily = df_daily.resample('60min').mean()
            # df_daily = df_daily.fillna(method='ffill')

            middle = f'BBM_{bb_length}_{bb_std:.1f}_{ticket}'
            upper = f'BBU_{bb_length}_{bb_std:.1f}_{ticket}'
            lower = f'BBL_{bb_length}_{bb_std:.1f}_{ticket}'
            df[ticket + '_UPPER_DIFF'] = df[upper] - df['high']
            df[ticket + '_MIDDLE_DIFF'] = df[middle] - df['close']
            df[ticket + '_LOWER_DIFF'] = df[lower] - df['low']
            df[ticket + '_UPPER_ATR_TIMES'] = df[ticket + '_UPPER_DIFF'] / df['ATRr_14_' + ticket]
            df[ticket + '_MIDDLE_ATR_TIMES'] = df[ticket + '_MIDDLE_DIFF'] / df['ATRr_14_' + ticket]
            df[ticket + '_LOWER_ATR_TIMES'] = df[ticket + '_LOWER_DIFF'] / df['ATRr_14_' + ticket]

            df.ta.natr(append=True, suffix=ticket)
            df.ta.rsi(append=True, suffix=ticket)
            df.ta.adx(append=True, suffix=ticket)
            df.ta.macd(append=True, suffix=ticket)
            df.ta.stoch(append=True, suffix=ticket)
            df.ta.mfi(append=True, suffix=ticket)
            df.ta.nvi(append=True, suffix=ticket)
            df.ta.pvi(append=True, suffix=ticket)
            df.ta.ebsw(append=True, suffix=ticket)
            df.ta.zscore(append=True, suffix=ticket)

            df['EMA_4_NATR_14_' + ticket] = ta.ema(df['NATR_14_' + ticket], length=ema_fast)
            df['EMA_20_NATR_14_' + ticket] = ta.ema(df['NATR_14_' + ticket], length=ema_slow)

        for idx in n:
            df[f'{ticket}_Past_{idx}'] = df['close'].shift(idx)
            df[f'{ticket}_PastVolume_{idx}'] = df['volume'].shift(idx)
            df[f'{ticket}_HighPast_{idx}'] = df['high'].shift(idx)
            df[f'{ticket}_LowPast_{idx}'] = df['low'].shift(idx)
            if not predict:
                df[f'{ticket}_Future_{idx}'] = df['close'].shift(-idx)
                df[f'{ticket}_HighFuture_{idx}'] = df['high'].shift(-idx)
                df[f'{ticket}_LowFuture_{idx}'] = df['low'].shift(-idx)

        df[f'{ticket}_HighDiff'] = df['close'] - df[f'{ticket}_HighPast_1']
        df[f'{ticket}_HighDiff_d2'] = df[f'{ticket}_HighDiff'] - (df[f'{ticket}_HighPast_1'] - df[f'{ticket}_HighPast_2'])

        df[f'{ticket}_LowDiff'] = df['close'] - df[f'{ticket}_LowPast_1']
        df[f'{ticket}_LowDiff_d2'] = df[f'{ticket}_LowDiff'] - (df[f'{ticket}_LowPast_1'] - df[f'{ticket}_LowPast_2'])

        for idx in n:
            # Medidas absolutas

            pasts = []
            for j in range(1, idx + 1):
                pasts.append(f'{ticket}_PastVolume_{j}')
            df.loc[:, f'{ticket}_Volume_CUM_{idx}'] = df[[*pasts]].sum(axis=1)

            # df[f'{ticket}_Volume_EMA_{idx}'] = ta.ema(df['volume'], length=12)

            df[f'{ticket}_InputReturn_{idx}'] = (df['close'] / df[f'{ticket}_Past_{idx}']) - 1.0
            df[f'{ticket}_HighInputReturn_{idx}'] = (df['high'] / df[f'{ticket}_HighPast_{idx}']) - 1.0
            df[f'{ticket}_LowInputReturn_{idx}'] = (df['low'] / df[f'{ticket}_LowPast_{idx}']) - 1.0

            # df[f'{ticket}_Diff_Volume_{idx}'] = df[f'{ticket}_Volume_EMA_{idx}'].diff(idx)
            # df[f'{ticket}_Diff_Volume_d2_{idx}'] = df[f'{ticket}_Diff_Volume_{idx}'].diff(idx)
            #
            # df[f'{ticket}_InputReturn_Volume_{idx}'] = ta.percent_return(close=df[f'{ticket}_Volume_EMA_{idx}'], length=idx)
            # df[f'{ticket}_InputReturn_Volume_d2_{idx}'] = ta.percent_return(close=df[f'{ticket}_InputReturn_Volume_{idx}'], length=idx)

            # percent_return
            # log_return

        df = df.dropna()

        if not predict:
            for i, idx in enumerate(n):
                df.loc[:, f'{ticket}_OutputReturn_{idx}'] = (df.loc[:, f'{ticket}_Future_{idx}'] / df.loc[:, 'close']) - 1.0
                df.loc[:, f'{ticket}_HighOutputReturn_{idx}'] = (df.loc[:, f'{ticket}_HighFuture_{idx}'] / df.loc[:, 'high']) - 1.0
                df.loc[:, f'{ticket}_LowOutputReturn_{idx}'] = (df.loc[:, f'{ticket}_LowFuture_{idx}'] / df.loc[:, 'low']) - 1.0

            for i, idx in enumerate(n_output):
                high_futures = []
                low_futures = []
                for j in range(1, idx + 1):
                    high_futures.append(f'{ticket}_HighOutputReturn_{j}')
                    low_futures.append(f'{ticket}_LowOutputReturn_{j}')
                df.loc[:, f'{ticket}_MaxOutputReturn_{idx}'] = df[[*high_futures]].max(axis=1)
                df.loc[:, f'{ticket}_MinOutputReturn_{idx}'] = -df[[*low_futures]].min(axis=1)
                if df[f'{ticket}_MaxOutputReturn_{idx}'].abs().max() >= df[f'{ticket}_MinOutputReturn_{idx}'].abs().max():
                    # buy
                    df.loc[:, f'{ticket}_WinOutputReturn_{idx}'] = df.loc[:, f'{ticket}_MaxOutputReturn_{idx}']
                    df.loc[:, f'{ticket}_LossOutputReturn_{idx}'] = -df.loc[:, f'{ticket}_MinOutputReturn_{idx}']
                    # rr buy = win% / loss%
                    df.loc[:, f'{ticket}_RewardRatio_{idx}'] = (
                            df.loc[:, f'{ticket}_WinOutputReturn_{idx}'] / df.loc[:,
                                                                                f'{ticket}_LossOutputReturn_{idx}'])
                else:
                    # sell
                    df.loc[:, f'{ticket}_WinOutputReturn_{idx}'] = -df.loc[:, f'{ticket}_MinOutputReturn_{idx}']
                    df.loc[:, f'{ticket}_LossOutputReturn_{idx}'] = df.loc[:, f'{ticket}_MaxOutputReturn_{idx}']
                    # rr sell = -win% / loss%
                    df.loc[:, f'{ticket}_RewardRatio_{idx}'] = (
                            -df.loc[:, f'{ticket}_WinOutputReturn_{idx}'] / df.loc[:,
                                                                            f'{ticket}_LossOutputReturn_{idx}'])

        df = df.rename(columns={'close': ticket + '_Close',
                                'tick_volume': ticket + '_Volume'
                                })
        df.drop(['open', 'high', 'low', 'volume', 'adj_close'], inplace=True, axis=1)
        print(df)
        dataframes.append(df)

    df_prev = None
    df_current = None
    for df in dataframes:
        df_prev = df_current
        df_current = df
        if df_prev is not None:
            df_current = df_prev.merge(df_current, left_index=True, right_index=True)

    print('from date: {}'.format(df_current.index[0]))
    print('to date: {}'.format(df_current.index[-1]))
    print(f'new fred columns: {reverse_rates.values()}')
    start = df_current.index[0]
    end = df_current.index[-1]
    if not predict:
        fred_dataset = web.DataReader([*interest_rates.values()], 'fred', start, end)
        fred_dataset = fred_dataset.rename(columns=reverse_rates)
        fred_dataset.to_csv(interest_full)
    else:
        fred_dataset = pd.read_csv(interest_full, index_col=0, parse_dates=True)

    if len(fred_dataset) > 0:

        fred_dataset['time'] = pd.to_timedelta(0, unit='h')
        fred_dataset.index = fred_dataset.index + fred_dataset.time
        fred_dataset = fred_dataset.drop('time', axis=1)
        fred_dataset = fred_dataset.fillna(method='ffill')

        print(fred_dataset)

        df_current = df_current.merge(fred_dataset, left_index=True, right_index=True, how='left')

        for currency in reverse_rates.values():
            df_current[currency] = df_current[currency].fillna(method='ffill')

        # _ArrayMemoryError ?
        # df_current = df_current.dropna(axis=1)

    return df_current

