# coding=utf-8
import sys
import math
import matplotlib.pyplot as plt
import random
import numpy as np

# Input from data
probability_of_one_oportunity = 3.0 / 24.0
deal_duration_max = 24 * 30

# Inputs
num_simulations = 1000
weeks = 4
timesteps = ((24 - 9) + 24 + 24 + 24 + (24 - 6)) * weeks
balance_init = 800.0
min_risk_per_operation = 0.5 / 100.0
max_risk_per_operation = 2.0 / 100.0
eur_each_slot = 57
loss_atr_times = 6
mu_velocity = 0.01
reward_risk_ratio = 3.0


def te_2_tn(te, n=12):
    return (((1 + te) ** (1.0/n)) - 1) * n

def tn_2_te(tn, n=12):
    return ((1 + (tn / n)) ** n) - 1

def interes_compuesto(c0, r, t, n=1):
    """
    n: reinversiones al año
    """
    if n is not None:
        return c0 * (1 + (r / n)) ** (n * t)
    else:
        return c0 * math.exp(r * t)

def interes_simple(c0, r, t):
    return c0 * (1 + r * t)

def interes_simple_a_compuesto( c0, r_simple, t ):
    r_compose = np.log( 1 + r_simple * t ) / t
    return r_compose
    
print(interes_simple(1000, 0.1, 12))
r = interes_simple_a_compuesto(1000, 0.1, 12)
print('r {} simple is same that r compose {}'.format(0.1, r))
print(interes_compuesto(1000, r, 12, n=99999))

print('----')

print(interes_compuesto(1000, 0.2, 2, 12))
print(interes_compuesto(1000, 0.2, 2, 6))
print(interes_compuesto(1000, 0.2, 2, 4))
print(interes_compuesto(1000, 0.2, 2, 2))
print(interes_compuesto(1000, 0.2, 2, 1))
print('----')
print(interes_compuesto(1000, tn_2_te(0.2, n=12), 2))
print(interes_compuesto(1000, tn_2_te(0.2, n=6), 2))
print(interes_compuesto(1000, tn_2_te(0.2, n=4), 2))
print(interes_compuesto(1000, tn_2_te(0.2, n=2), 2))
print(interes_compuesto(1000, tn_2_te(0.2, n=1), 2))
print('----')


# interes compuesto
# final = interes_compuesto(1000, 0.12, 12, None) - 1000
# print(final)
# final = interes_compuesto(1000 * 20, 0.012, 12, None) - (1000 * 20)
# print(final)
# print(interes_compuesto(1000, 0.14, 20, 1))
# print(interes_compuesto(1000, 0.14, 20, 20))
# print(interes_compuesto(1000, 0.14, 20, None))

#interes simple
# beneficio_simple = interes_simple(1000, 0.14, 1) - 1000
# for _ in range(20):
#     beneficio_simple = interes_simple(1000 + beneficio_simple, 0.14, 1) - 1000
# print(1000 + beneficio_simple)

# print(1000 * (1 + te_2_tn(0.14, n=20)) ** 20)
# print(1000 * (1 + tn_2_te(0.14, n=20)) ** 20)


# random multivariate_normal
# np.random.multivariate_normal([10, 500], [[1.0, 0.2], [0.2, 1.0]], (100, 1000)).shape


# Tracking
win_probability = []
end_balance = []
win_streaks = []
loss_streaks = []


# Creating Figure for Simulation Balances
fig = plt.figure(figsize = (20, 16), dpi=90)
plt.title("MonteCarlo Trading strategy [" + str(num_simulations) + " simulations]")
plt.xlabel("Timesteps")
plt.ylabel("Balance [$]")
# plt.xlim([0, timesteps])

# reward_ratio = 4
# simular un browniano: https://bukowskydrunk.medium.com/movimiento-browniano-implementaci%C3%B3n-num%C3%A9rica-en-python-aa39804ccb83

# For loop to run for the number of simulations desired
for i in range(num_simulations):
    balance = [balance_init]
    equity = [balance_init]
    steps = [ 0 ]
    # Run until the player has rolled 1,000 times
    win_streak = 0
    loss_streak = 0

    max_slots = round(balance[-1] / eur_each_slot)
    active_deals_end_timestep = []
    semaphore_count = max_slots
    
    # PAra cada camino
    while steps[-1 ] < timesteps:
        
        timestep = steps[-1]
        diff_balance = 0
        # diff_equity = 0

        if semaphore_count > 0:
            if np.random.uniform ( 0, 1 ) < probability_of_one_oportunity:
                # open deal
                bet = round(random.uniform(min_risk_per_operation, max_risk_per_operation) * balance[-1], 2)
                estimated_duration = deal_duration_max
                
                # digits = 5
                # atr = 600
                # loss_pips = loss_atr_times * atr
                # volume = bet / loss_pips
                
                profits = []
                profit = 0.0
                i = timestep
                while i <= timestep + estimated_duration:
                    profits.append(profit)
                    profit = np.random.normal(profit + mu_velocity, bet / loss_atr_times)  # volatilidad del profit, depende del volumen
                    i += 1
                active_deals_end_timestep.append( (timestep, timestep + estimated_duration, bet, profits) )
                semaphore_count -= 1

        end_timesteps_to_remove = []
        for begin_timestep, end_timestep, bet, profits in active_deals_end_timestep:

            if(begin_timestep <= timestep < timestep + estimated_duration):
                diff_step = timestep - begin_timestep
                profit = profits[diff_step]
            else:
                profit = profits[-1]
            win_factor = profit / (reward_risk_ratio * bet)
            loss_factor = profit / -bet
            

            if win_factor >= 1.0 or loss_factor >= 1.0:

                # close operation
                end_timesteps_to_remove.append( (begin_timestep, end_timestep, bet, profits) )

                # expectance_value = np.random.uniform( -1, reward_risk_ratio )
                if profit > 0:
                    expectance_value = win_factor * reward_risk_ratio
                else:
                    expectance_value = loss_factor * -1.0
                    
                win_probability.append( (expectance_value + 1) / (reward_risk_ratio + 1) )
                # Result if the dice are the same number
                if expectance_value >= 0:
                    win_streak += 1
                    loss_streak = 0
                else:
                    win_streak = 0
                    loss_streak += 1
                if win_streak > 0:
                    win_streaks.append(win_streak)
                if loss_streak > 0:
                    loss_streaks.append(loss_streak)
                    
                diff_balance += (expectance_value * bet)
                semaphore_count += 1
                
        for begin_timestep, end_timestep, bet, profits in end_timesteps_to_remove:
            active_deals_end_timestep.remove( (begin_timestep, end_timestep, bet, profits) )

        # for begin_timestep, end_timestep, bet, profits in active_deals_end_timestep:
        #     assert(begin_timestep <= timestep < end_timestep)
        #     diff_step = timestep - begin_timestep
        #     profit = profits[diff_step]
        #     diff_equity += profit

        balance.append(balance[-1] + diff_balance)
        # equity.append(balance[-1] + diff_equity)
        steps.append( timestep + 1 )
            
    end_balance.append(balance[-1])
    
    rng = np.arange(len(balance))
    plt.plot(rng, balance)
    # plt.plot(rng, equity, 'r--')

plt.tight_layout()
plt.show()


# Averaging win probability and end balance
overall_win_probability = sum(win_probability)/len(win_probability)
overall_end_balance = sum(end_balance)/len(end_balance)

loss_probability = 1.0 - overall_win_probability
diff_probability = overall_win_probability - loss_probability
deals_to_ruin = int(balance_init / (balance_init * max_risk_per_operation))
ruin_probability = ((1 - diff_probability) / (1 + diff_probability)) ** deals_to_ruin

# Displaying the averages
print("Average win probability after " + str(num_simulations) + " runs: " + str(overall_win_probability))
print("Max balance after " + str(num_simulations) + " runs: $" + str(max(end_balance)))
print("Min balance after " + str(num_simulations) + " runs: $" + str(min(end_balance)))
print('Max win streak: {}'.format(max(win_streaks)))
print('Max loss streak: {}'.format(max(loss_streaks)))
print('Percentil 50 win streak: {}'.format(np.percentile(np.array(win_streaks), 50)))
print('Percentil 50 loss streak: {}'.format(np.percentile(np.array(loss_streaks), 50)))
print('Percentil 95 win streak: {}'.format(np.percentile(np.array(win_streaks), 95)))
print('Percentil 95 loss streak: {}'.format(np.percentile(np.array(loss_streaks), 95)))
print("Performance: {:.3f}%".format(100.0 * ((overall_end_balance / balance_init) - 1.0)))
print("Ruin probability: {:.3f}".format(100.0 * ruin_probability))
if overall_win_probability > (1.0 / (reward_risk_ratio + 1)):
    rentable = overall_win_probability - (1.0 / (reward_risk_ratio + 1))
    print('Estrategia rentable: {}'.format(rentable))
else:
    print('La estrategia no es rentable.')
print("Average ending balance after " + str(num_simulations) + " runs: $" + str(overall_end_balance))
print("Average profit after " + str(num_simulations) + " runs: $" + str(overall_end_balance - balance_init))
