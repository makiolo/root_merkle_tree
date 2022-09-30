import cPickle
import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer


def learning(model=None, serialize_to=None, debug=True, **kwargs):

    class decorator:

        def __init__(self, functor):
            self.functor = functor
            self.serialize_to = serialize_to
            self.data = {}
            self.debug = debug
            self.model = None
            self.n_inputs = None
            self.n_outputs = None
            self.scaler_inputs = StandardScaler()
            self.scaler_outputs = StandardScaler()
            if self.serialize_to and os.path.isfile(self.serialize_to):
                with open(self.serialize_to, 'rb') as f:
                    key_map, mdl, inputs, outputs = cPickle.load(f)
                self.data[key_map] = (mdl, inputs, outputs)
            else:
                self.model = model(**kwargs)

        def train(self, dataframe, inputs, outputs):
            xx = dataframe.loc[:, inputs]
            yy = dataframe.loc[:, outputs]

            # from sklearn import utils
            # utils.type_of_target(xx)

            self.scaler_inputs.fit(xx)
            self.scaler_outputs.fit(yy)

            xx_scaled = self.scaler_inputs.transform(xx)
            yy_scaled = self.scaler_outputs.transform(yy)

            self.model.fit(xx_scaled, yy_scaled)

            return self.model

        def predict(self, model, dataframe, inputs, outputs):
            self.n_inputs = len(inputs)
            self.n_outputs = len(outputs)
            xx = dataframe.loc[:, inputs]
            xx_scaled = self.scaler_inputs.transform(xx)
            yy_scaled = model.predict(xx_scaled)
            yy = self.scaler_outputs.inverse_transform(yy_scaled)

            df2 = pd.DataFrame(yy, columns=outputs)
            if self.debug:
                df2.rename(columns=dict((x, 'model_{}'.format(x)) for x in outputs), inplace=True)
                df1 = pd.DataFrame(xx, columns=inputs)
                df = pd.concat([df1, df2], axis=1)
                self.functor(df)
                df.rename(columns=dict((x, 'expected_{}'.format(x)) for x in outputs), inplace=True)
                for x in outputs:
                    df['error_{}'.format(x)] = (df['expected_{}'.format(x)] - df['model_{}'.format(x)])
                return df
            else:
                df1 = pd.DataFrame(xx, columns=inputs)
                df = pd.concat([df1, df2], axis=1)
                return df

        def __call__(self, dataframe):
            key_map = tuple(dataframe.columns.values + [self.functor.__name__])

            if key_map in self.data:
                # predict
                model, inputs, outputs = self.data[key_map]
                return self.predict(model, dataframe, inputs, outputs)
            else:
                # 1. prepare data
                inputs = list(dataframe.columns.values)
                outputs = self.functor(dataframe)
                # 2. train
                model = self.train(dataframe, inputs, outputs)
                model_data = (key_map, model, inputs, outputs)
                if self.serialize_to:
                    with open(self.serialize_to, 'wb') as f:
                        cPickle.dump(model_data, f)
                self.data[key_map] = (model, inputs, outputs)
                # 3. predict
                return self.predict(model, dataframe, inputs, outputs)

    return decorator

from sklearn import preprocessing
from sklearn import utils

# Regressors
#       single output
from sklearn.svm                    import SVR
from sklearn.svm                    import LinearSVR
from sklearn.svm                    import NuSVR
from sklearn.linear_model           import LinearRegression
from sklearn.linear_model           import SGDRegressor
from sklearn.linear_model           import BayesianRidge
from sklearn.linear_model           import LassoLars
from sklearn.linear_model           import ARDRegression
from sklearn.linear_model           import PassiveAggressiveRegressor
from sklearn.linear_model           import TheilSenRegressor
from sklearn.linear_model           import SGDRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
#       multiple output
from sklearn.linear_model           import MultiTaskLasso


# Classifiers ?
from sklearn.svm                    import SVC
from sklearn.linear_model           import LogisticRegression
from sklearn.tree                   import DecisionTreeRegressor
from sklearn.tree                   import DecisionTreeClassifier
from sklearn.neighbors              import KNeighborsClassifier
from sklearn.discriminant_analysis  import LinearDiscriminantAnalysis
from sklearn.naive_bayes            import GaussianNB

# neuronal net
from sklearn.neural_network import MLPRegressor

# Distribution
from sklearn.preprocessing import PowerTransformer


# @learning(model=linear_model.LinearRegression, debug=True)
# @learning(model=LassoLars, debug=True, alpha=.1, normalize=False)
# @learning(model=MultiTaskLasso, debug=True, alpha=1.0)
# @learning(model=SVR, debug=True, kernel='poly', degree=2, epsilon=0.2)
# @learning(model=DecisionTreeRegressor, debug=True, max_depth=16)
# @learning(model=AdaBoostRegressor, debug=True, base_estimator=DecisionTreeRegressor(max_depth=32), n_estimators=900, random_state=np.random.RandomState(1))
# @learning(model=LogisticRegression, debug=True, random_state=0, solver='lbfgs', multi_class='multinomial')
@learning(model=MLPRegressor, debug=True, hidden_layer_sizes=(200, 150, 200, 150, 200), random_state=np.random.RandomState(1), max_iter=2000, activation='relu')
def calculate_outputs(data):
    # polynomial
    # data['output1'] = 2*data['input2'] ** 3 + 5*data['input2']**2 + 10*data['input1'] * 5 - 1
    # stochastic
    data['output1'] = data['max_+72_0']
    data['output2'] = data['min_+72_0']

    # define new outputs
    return ['output1', 'output2']


def prepare_inputs(df):
    df['log_ret_-24_0'] = (calculate_returns(df.input1, 24))
    df['log_ret_-48_0'] = (calculate_returns(df.input1, 48))
    df['log_ret_-72_0'] = (calculate_returns(df.input1, 72))
    df['log_ret_+24_0'] = (calculate_returns(df.input1, -24))
    df['log_ret_+48_0'] = (calculate_returns(df.input1, -48))
    df['log_ret_+72_0'] = (calculate_returns(df.input1, -72))

    df['log_ret_-24_1'] = (calculate_returns(df.input2, 24))
    df['log_ret_-48_1'] = (calculate_returns(df.input2, 48))
    df['log_ret_-72_1'] = (calculate_returns(df.input2, 72))
    df['log_ret_+24_1'] = (calculate_returns(df.input2, -24))
    df['log_ret_+48_1'] = (calculate_returns(df.input2, -48))
    df['log_ret_+72_1'] = (calculate_returns(df.input2, -72))

    df['max_-24_0'] = (calculate_max(df.input1, 24))
    df['max_-48_0'] = (calculate_max(df.input1, 48))
    df['max_-72_0'] = (calculate_max(df.input1, 72))
    df['max_+24_0'] = (calculate_max(df.input1, -24))
    df['max_+48_0'] = (calculate_max(df.input1, -48))
    df['max_+72_0'] = (calculate_max(df.input1, -72))

    df['max_-24_1'] = (calculate_max(df.input2, 24))
    df['max_-48_1'] = (calculate_max(df.input2, 48))
    df['max_-72_1'] = (calculate_max(df.input2, 72))
    df['max_+24_1'] = (calculate_max(df.input2, -24))
    df['max_+48_1'] = (calculate_max(df.input2, -48))
    df['max_+72_1'] = (calculate_max(df.input2, -72))

    df['min_-24_0'] = (calculate_min(df.input1, 24))
    df['min_-48_0'] = (calculate_min(df.input1, 48))
    df['min_-72_0'] = (calculate_min(df.input1, 72))
    df['min_+24_0'] = (calculate_min(df.input1, -24))
    df['min_+48_0'] = (calculate_min(df.input1, -48))
    df['min_+72_0'] = (calculate_min(df.input1, -72))

    df['min_-24_1'] = (calculate_min(df.input2, 24))
    df['min_-48_1'] = (calculate_min(df.input2, 48))
    df['min_-72_1'] = (calculate_min(df.input2, 72))
    df['min_+24_1'] = (calculate_min(df.input2, -24))
    df['min_+48_1'] = (calculate_min(df.input2, -48))
    df['min_+72_1'] = (calculate_min(df.input2, -72))

    df.dropna(inplace=True)
    return df


def make_gaussian(serie):
    return PowerTransformer(standardize=True).fit_transform(np.array(serie).reshape(-1, 1))


def calculate_returns(serie, n=1, log=False):
    if log:
        if n > 0:
            return np.log(serie / serie.shift(n))
        else:
            return np.log(serie.shift(n) / serie)
    else:
        if n > 0:
            return serie / serie.shift(n) - 1.0
        else:
            return serie.shift(n) / serie - 1.0


def calculate_max(serie, n=1):
    if n > 0:
        return serie.rolling(n).max()
    else:
        return serie.shift(n).rolling(-n).max()


def calculate_min(serie, n=1):
    if n > 0:
        return serie.rolling(n).min()
    else:
        return serie.shift(n).rolling(-n).min()


def geo_paths(S, T, q, sigma, steps, N, r):
    dt = T / steps
    ST = np.cumsum(((r - q - sigma ** 2 / 2) * dt + \
                                sigma * np.sqrt(dt) * \
                                np.random.normal(size=(steps, N))), axis=0)
    return S + np.exp(ST)


S = 100
T = 1.0
q = 0.0
sigma = 0.22
steps = 20000
N = 4
r = 0.06
paths = geo_paths(S, T, q, sigma, steps, N, r)

############
print('Train ...')
dataset = {
    'input1': paths.T[0],
    'input2': paths.T[1],
}
df = pd.DataFrame(dataset)
df = prepare_inputs(df)
df = calculate_outputs(df)
print(df)

############

sigma = 2
mu = 102

print('Predict ...')
dataset = {
    'input1': paths.T[2],
    'input2': paths.T[3],
}
df = pd.DataFrame(dataset)
df = prepare_inputs(df)
df = calculate_outputs(df)


if calculate_outputs.debug:

    print('Error mean:')
    print(df['expected_output1'].mean() - df['model_output1'].mean())
    print('Error std:')
    print(df['expected_output1'].std() - df['model_output1'].std())

    fig, axs = plt.subplots(2, 2, figsize=(15, 9))

    def animate(i):
        axs[0,0].cla()  # clear the previous image
        axs[1,0].cla()  # clear the previous image

        n = 20

        axs[0,0].plot(df.input1[:i-n], df.model_output1[:i-n], color='#EBA2A6', marker='o', linestyle='')  # plot the line
        axs[0,0].plot(df.input1[:i-n], df.expected_output1[:i-n], color='#B5FFC8', marker='x', linestyle='')  # plot the line
        axs[0,0].plot(df.input1[i-n:i], df.model_output1[i-n:i], color='#9C1F26', marker='o', linestyle='-')  # plot the line
        axs[0,0].plot(df.input1[i-n:i], df.expected_output1[i-n:i], color='#2F9C4A', marker='x', linestyle='-')  # plot the line

        axs[1, 0].plot(df.input1[:i-n], df.error_output1[:i-n], color='#EBA2A6', label='abs. error 1.', marker='x', linestyle='')
        axs[1, 0].plot(df.input1[i-n:i], df.error_output1[i-n:i], color='#9C1F26', label='abs. error 1.', marker='x', linestyle='-')

        if True:
            axs[0, 1].cla()  # clear the previous image
            axs[1, 1].cla()  # clear the previous image

            axs[0, 1].plot(df.input1[:i - n], df.model_output2[:i - n], color='#EBA2A6', marker='o', linestyle='')  # plot the line
            axs[0, 1].plot(df.input1[:i - n], df.expected_output2[:i - n], color='#B5FFC8', marker='x', linestyle='')  # plot the line
            axs[0, 1].plot(df.input1[i - n:i], df.model_output2[i - n:i], color='#9C1F26', marker='o', linestyle='-')  # plot the line
            axs[0, 1].plot(df.input1[i - n:i], df.expected_output2[i - n:i], color='#2F9C4A', marker='x', linestyle='-')  # plot the line

            axs[1, 1].plot(df.input1[:i - n], df.error_output2[:i - n], color='#EBA2A6', label='abs. error 1.', marker='x', linestyle='')
            axs[1, 1].plot(df.input1[i - n:i], df.error_output2[i - n:i], color='#9C1F26', label='abs. error 1.', marker='x', linestyle='-')

    anim = animation.FuncAnimation(fig, animate, frames=len(df.input1) + 1, interval=0.1, blit=False)
    # axs[0,0].legend()
    # axs[1,0].legend()
    # axs[0, 1].legend()
    # axs[1, 1].legend()
    plt.tight_layout()
    plt.show()
else:
    print(df.columns)

    fig, axs = plt.subplots(1, 2, figsize=(15, 9))

    axs[0].plot(df.input1, df.output1, label='output1 (model).', marker='x', linestyle='')
    axs[0].legend()

    if True:
        axs[1].plot(df.input1, df.output2, label='output2 (model).',marker='x', linestyle='')
        axs[1].legend()

    plt.tight_layout()
    plt.show()

