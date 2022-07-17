# https://nbviewer.org/github/dcajasn/Riskfolio-Lib/blob/master/examples/Tutorial%206.ipynb

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import vectorbt as vbt
import riskfolio as rp
from matplotlib import pyplot as plt
from vectorbt.portfolio.nb import order_nb, sort_call_seq_nb
from vectorbt.portfolio.enums import SizeType, Direction

warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.4%}'.format

# Date range
start = '2022-01-01'
end = '2022-07-17'

# Tickers of assets

assets = []
assets_inverse = []
industries = []
currencies = ['EUR', 'GBP', 'AUD', 'NZD', 'USD', 'CAD', 'CHF', 'JPY']
industry_weights = [1.0] * len(currencies)
ticket_combinations = []
for mode in ['LONG', 'SHORT']:
    for n_clusters in range(len(currencies)):
        currency1 = currencies[n_clusters]
        for currency2 in currencies[n_clusters + 1:]:
            ticket = '{}{}=X'.format(currency1, currency2)
            if mode == 'LONG':
                assets.append(f'{ticket}')
                industries.append(f'{currency1}')
            else:
                assets_inverse.append(f'{ticket}_{mode}')
                industries.append(f'{currency2}_{mode}')
# assets.sort()

# Downloading data
data = yf.download(assets, start=start, end=end)
data = data.loc[:, ['Adj Close']]
data.columns = assets
for asset_inverse in assets_inverse:
    asset_long = asset_inverse.split('_')[0]
    data[asset_inverse] = -data[asset_long]

Y = data.pct_change().dropna()
print(Y)

# price = data['USDJPY=X']
# fast_ma = vbt.MA.run(price, 20)
# slow_ma = vbt.MA.run(price, 50)
# entries = fast_ma.ma_crossed_above(slow_ma)
# exits = fast_ma.ma_crossed_below(slow_ma)
#
# pf = vbt.Portfolio.from_signals(price, entries, exits, init_cash=5000)
# print(pf.total_profit())
# print(pf.stats())

port = rp.Portfolio(returns=Y)

method_mu = 'hist'  # Method to estimate expected returns based on historical data.
method_cov = 'hist'  # Method to estimate covariance matrix based on historical data.
port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

# Estimate optimal portfolio:

model = 'Classic'  # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
rm = 'MV'  # Risk measure used, this time will be variance
obj = 'Sharpe'  # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
hist = True  # Use historical scenarios for risk measures that depend on scenarios
rf = 0  # Risk free rate
l = 0  # Risk aversion factor, only useful when obj is 'Utility'

# calculate optimal weights
w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
print(w.T)
#
# # plot portfolio
# ax = rp.plot_pie(w=w, title='Sharpe Mean Variance', others=0.05, nrow=25, cmap = "tab20",
#                  height=6, width=10, ax=None)
# plt.show()
#
# # calculate frontier
# points = 50 # Number of points of the frontier
#
# frontier = port.efficient_frontier(model=model, rm=rm, points=points, rf=rf, hist=hist)

# Plotting the efficient frontier
#
# label = 'Max Risk Adjusted Return Portfolio' # Title of point
# mu = port.mu # Expected returns
# cov = port.cov # Covariance matrix
# returns = port.returns # Returns of the assets
#
# ax = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm=rm,
#                       rf=rf, alpha=0.05, cmap='viridis', w=w, label=label,
#                       marker='*', s=16, c='r', height=6, width=10, ax=None)
# plt.show()
#
# # Plotting efficient frontier composition
#
# ax = rp.plot_frontier_area(w_frontier=frontier, cmap="tab20", height=6, width=10, ax=None)
# plt.show()


# # Calculate CVaR
# rm = 'CVaR' # Risk measure
# w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
#
# ax = rp.plot_pie(w=w, title='Sharpe Mean CVaR', others=0.05, nrow=25, cmap = "tab20",
#                  height=6, width=10, ax=None)
# plt.show()
#
# points = 50 # Number of points of the frontier
#
# frontier = port.efficient_frontier(model=model, rm=rm, points=points, rf=rf, hist=hist)
#
# label = 'Max Risk Adjusted Return Portfolio' # Title of point
#
# ax = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm=rm,
#                       rf=rf, alpha=0.05, cmap='viridis', w=w, label=label,
#                       marker='*', s=16, c='r', height=6, width=10, ax=None)
#
# plt.show()
#
# # Plotting efficient frontier composition
#
# ax = rp.plot_frontier_area(w_frontier=frontier, cmap="tab20", height=6, width=10, ax=None)
# plt.show()

# # multiples measure risks
#
# rms = ['MV', 'MAD', 'MSV', 'FLPM', 'SLPM', 'CVaR',
#        'EVaR', 'WR', 'MDD', 'ADD', 'CDaR', 'UCI', 'EDaR']
#
# w_s = pd.DataFrame([])
#
# for i in rms:
#     w = port.optimization(model=model, rm=i, obj=obj, rf=rf, l=l, hist=hist)
#     w_s = pd.concat([w_s, w], axis=1)
#
# w_s.columns = rms
# w_s.style.format("{:.2%}").background_gradient(cmap='YlGn')
# print(w_s)


# Plotting a comparison of assets weights for each portfolio

# fig = plt.gcf()
# fig.set_figwidth(14)
# fig.set_figheight(6)
# ax = fig.subplots(nrows=1, ncols=1)
#
# w_s.plot.bar(ax=ax)
# plt.show()


asset_classes = {'Assets': assets + assets_inverse,
                 'Industry': industries
                 }
asset_classes = pd.DataFrame(asset_classes)
# asset_classes = asset_classes.sort_values(by=['Assets'])

constraints = {'Disabled': [False] * len(currencies),
               'Type': ['Classes'] * len(currencies),
               'Set': ['Industry'] * len(currencies),
               'Position': currencies,
               'Sign': ['<='] * len(currencies),
               'Weight': max_weight,
               'Type Relative': [''] * len(currencies),
               'Relative Set': [''] * len(currencies),
               'Relative': [''] * len(currencies),
               'Factor': [''] * len(currencies)}

constraints = pd.DataFrame(constraints)
A, B = rp.assets_constraints(constraints, asset_classes)

port.ainequality = A
port.binequality = B

rm = 'MV'  # Risk measure used, this time will be variance
obj = 'Sharpe'  # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
hist = True  # Use historical scenarios for risk measures that depend on scenarios
rf = 0  # Risk free rate
l = 0  # Risk aversion factor, only useful when obj is 'Utility'

model = 'Classic'
# obj = 'Sharpe'
# rf = 0

rms = ['MV', 'MAD', 'MSV', 'FLPM', 'SLPM', 'CVaR',
       'EVaR', 'WR', 'MDD', 'ADD', 'CDaR', 'UCI', 'EDaR']
ws = []

w_s = pd.DataFrame([])
w_class = pd.DataFrame([])

for i in rms:
    w = port.optimization(model=model, rm=i, obj=obj, rf=rf, l=l, hist=hist)
    ws.append(w)
    classes = pd.concat([asset_classes.set_index('Assets'), w], axis=1)
    industry_groups = classes.groupby(['Industry']).sum()
    w_s = pd.concat([w_s, w], axis=1)
    w_class = pd.concat([w_class, industry_groups], axis=1)

w_s.columns = rms
w_class.columns = rms

print('Pesos segun medida de riesgo')
print(w_s)
fig = plt.gcf()
fig.set_figwidth(14)
fig.set_figheight(6)
ax = fig.subplots(nrows=1, ncols=1)
w_s.plot.bar(ax=ax)
plt.margins()
plt.tight_layout()
plt.show()

print('Pesos por industria')
print(w_class)
fig = plt.gcf()
fig.set_figwidth(14)
fig.set_figheight(6)
ax = fig.subplots(nrows=1, ncols=1)
w_class.plot.bar(ax=ax)
plt.margins()
plt.tight_layout()
plt.show()


# vbt.settings.returns['year_freq'] = '30 days'
# vbt.settings['plotting']['layout']['width'] = 900
# vbt.settings['plotting']['layout']['height'] = 400
#
# num_tests = 2000
# ann_factor = data.vbt.returns(freq='D').ann_factor
#
#
# def pre_sim_func_nb(sc, every_nth):
#     # Define rebalancing days
#     sc.segment_mask[:, :] = False
#     sc.segment_mask[every_nth::every_nth, :] = True
#     return ()
#
#
# def pre_segment_func_nb(sc, find_weights_nb, rm, history_len, ann_factor, num_tests, srb_sharpe):
#     if history_len == -1:
#         # Look back at the entire time period
#         close = sc.close[:sc.i, sc.from_col:sc.to_col]
#     else:
#         # Look back at a fixed time period
#         if sc.i - history_len <= 0:
#             return (np.full(sc.group_len, np.nan),)  # insufficient data
#         close = sc.close[sc.i - history_len:sc.i, sc.from_col:sc.to_col]
#
#     # Find optimal weights
#     best_sharpe_ratio, weights = find_weights_nb(sc, rm, close, num_tests)
#     srb_sharpe[sc.i] = best_sharpe_ratio
#
#     # Update valuation price and reorder orders
#     size_type = np.full(sc.group_len, SizeType.TargetPercent)
#     direction = np.full(sc.group_len, Direction.LongOnly)
#     temp_float_arr = np.empty(sc.group_len, dtype=np.float_)
#     for k in range(sc.group_len):
#         col = sc.from_col + k
#         sc.last_val_price[col] = sc.close[sc.i, col]
#     sort_call_seq_nb(sc, weights, size_type, direction, temp_float_arr)
#
#     return (weights,)
#
#
# def order_func_nb(oc, weights):
#     col_i = oc.call_seq_now[oc.call_idx]
#     return order_nb(
#         weights[col_i],
#         oc.close[oc.i, oc.col],
#         size_type=SizeType.TargetPercent,
#     )
#
#
# def plot_allocation(rb_pf):
#     # Plot weights development of the portfolio
#     rb_asset_value = rb_pf.asset_value(group_by=False)
#     rb_value = rb_pf.value()
#     rb_idxs = np.flatnonzero((rb_pf.asset_flow() != 0).any(axis=1))
#     rb_dates = rb_pf.wrapper.index[rb_idxs]
#     fig = (rb_asset_value.vbt / rb_value).vbt.plot(
#         trace_names=assets,
#         trace_kwargs=dict(
#             stackgroup='one'
#         )
#     )
#     for rb_date in rb_dates:
#         fig.add_shape(
#             dict(
#                 xref='x',
#                 yref='paper',
#                 x0=rb_date,
#                 x1=rb_date,
#                 y0=0,
#                 y1=1,
#                 line_color=fig.layout.template.layout.plot_bgcolor
#             )
#         )
#     fig.show_svg()
#
#
# i = 0
# sharpe_0 = np.full(data.shape[0], np.nan)
# portfolio_vbt = vbt.Portfolio.from_order_func(
#         data,
#         order_func_nb,
#         pre_sim_func_nb=pre_sim_func_nb,
#         pre_sim_args=(30,),
#         pre_segment_func_nb=pre_segment_func_nb,
#         pre_segment_args=(ws[0], i, 252*4, ann_factor, num_tests, sharpe_0),
#         cash_sharing=True,
#         group_by=True,
#         use_numba=False,
#     )
# plot_allocation(portfolio_vbt)
