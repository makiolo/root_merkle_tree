import cPickle
import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def learning(model=None, serialize_to=None, return_inputs=True, **kwargs):

    class decorator:

        def __init__(self, functor):
            self.functor = functor
            self.serialize_to = serialize_to
            self.data = {}
            self.return_inputs = return_inputs
            self.model = None
            if self.serialize_to and os.path.isfile(self.serialize_to):
                with open(self.serialize_to, 'rb') as f:
                    key_map, mdl, inputs, outputs = cPickle.load(f)
                self.data[key_map] = (mdl, inputs, outputs)
            else:
                self.model = model(**kwargs)

        def train(self, dataframe, inputs, outputs):
            xx = dataframe.loc[:, inputs]
            yy = dataframe.loc[:, outputs]
            self.model.fit(xx, yy)
            return self.model

        def predict(self, model, dataframe, inputs, outputs):
            xx = dataframe.loc[:, inputs]
            yy = model.predict(xx)
            df2 = pd.DataFrame(yy, columns=outputs)
            if self.return_inputs:
                df2.rename(columns=dict((x, 'model_{}'.format(x)) for x in outputs), inplace=True)
                df1 = pd.DataFrame(xx, columns=inputs)
                df = pd.concat([df1, df2], axis=1)
                self.functor(df)
                df.rename(columns=dict((x, 'expected_{}'.format(x)) for x in outputs), inplace=True)
                for x in outputs:
                    df['error_{}'.format(x)] = pow(df['expected_{}'.format(x)] - df['model_{}'.format(x)], 2)
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
                self.functor(dataframe)
                outputs = list(set(dataframe.columns.values) - set(inputs))
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


from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import MultiTaskLasso
from sklearn.svm import SVR


@learning(model=linear_model.LinearRegression, return_inputs=True)
# @learning(model=linear_model.LassoLars, return_inputs=True, alpha=.1, normalize=False)
# @learning(model=DecisionTreeRegressor, return_inputs=True, max_depth=4)
# @learning(model=MultiTaskLasso, return_inputs=True, alpha=1.0)
# @learning(model=SVR, return_inputs=True, kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)
def calculate_outputs(data):
    #
    data['output1'] = data['input1'] * 2 + 2

############

dataset = {
    'input1': np.linspace(-1000, 1000, 10),
    'input2': np.linspace(-1000, 1000, 10),
}
df = pd.DataFrame(dataset)
df = calculate_outputs(df)
print(df)

############

dataset = {
    'input1': np.linspace(5, 10, 5),
    'input2': np.linspace(5, 10, 5),
}
df = pd.DataFrame(dataset)
df = calculate_outputs(df)
print(df)

fig, axs = plt.subplots(2, 2, figsize=(15, 9))

axs[0,0].plot(df.input1, df.expected_output1, label='expected 1.', marker='o', linestyle='-')
axs[0,0].plot(df.input1, df.model_output1, label='model output 1.', marker='o', linestyle='-')
axs[0,0].legend()

# axs[0,1].plot(df.input2, df.expected_output2, label='expected 2.',marker='o',linestyle='-')
# axs[0,1].plot(df.input2, df.model_output2, label='model output 2.',marker='o',linestyle='-')
# axs[0,1].legend()

axs[1,0].plot(df.input1, df.error_output1, label='error output 1.', marker='o', linestyle='-')
axs[1,0].legend()

# axs[1,1].plot(df.input2, df.error_output2, label='error output 2.', marker='o',linestyle='-')
# axs[1,1].legend()

plt.tight_layout()
plt.show()
