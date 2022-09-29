import cPickle
import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
                # return df[(-100 < df['error_output1']) & (df['error_output1'] < 100)]
                return df
            else:
                return df2

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
    sigma = 2
    mu = 4
    data['serie'] = pd.Series(sigma * np.random.randn(1000) + mu)
    data['output1'] = data['serie'].rolling(window=60).quantile(0.01)
    data['output2'] = data['serie'].rolling(window=60).quantile(0.99)
    data.dropna(inplace=True)
    # define new outputs
    return ['serie', 'output1', 'output2']

############
print('Train ...')
dataset = {
    'input1': np.linspace(-10000, 10000, 1000),
    'input2': np.linspace(-10000, 10000, 1000),
}
df = pd.DataFrame(dataset)
df = calculate_outputs(df)
print(df)

############

print('Predict ...')
dataset = {
    'input1': np.linspace(-10000, 10000, 1000),
    'input2': np.linspace(-10000, 10000, 1000),
}
df = pd.DataFrame(dataset)
df = calculate_outputs(df)
print(df)


if calculate_outputs.debug:
    fig, axs = plt.subplots(2, 2, figsize=(15, 9))

    axs[0,0].plot(df.input1, df.expected_serie, label='serie (real).', marker='o', linestyle='-')

    axs[0,0].plot(df.input1, df.expected_output1, label='output1 (real).', marker='x', linestyle='-')
    axs[0,0].plot(df.input1, df.model_output1, label='output1 (model).', marker='o', linestyle='-')
    axs[0,0].legend()

    axs[1,0].plot(df.input1, df.error_output1, label='abs. error 1.', marker='o', linestyle='-')
    axs[1,0].legend()

    if True:
        axs[0,0].plot(df.input2, df.expected_output2, label='output2 (real).',marker='x',linestyle='-')
        axs[0,0].plot(df.input2, df.model_output2, label='output2 (model).',marker='o',linestyle='-')
        axs[0,0].legend()

        axs[1,0].plot(df.input2, df.error_output2, label='abs. error 2.', marker='o',linestyle='-')
        axs[1,0].legend()

    plt.tight_layout()
    plt.show()
else:
    fig, axs = plt.subplots(1, 2, figsize=(15, 9))

    axs[0].plot(dataset['input1'], df.output1, label='output1 (model).', marker='o', linestyle='-')
    axs[0].legend()

    if True:
        axs[1].plot(dataset['input2'], df.output2, label='output2 (model).',marker='o',linestyle='-')
        axs[1].legend()

    plt.tight_layout()
    plt.show()

