import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = [ ]
for _ in range ( 200 ):
    dataset.append ( (
        int ( np.random.normal ( 32, 30 ) ), 
        round ( np.random.normal ( 170, 20 ), 2 ), 
        True if np.random.normal() < 0.5 else False
    ) )

df = pd.DataFrame(dataset, columns = ('age', 'height', 'live?') )
sns.barplot(x='age', y='height', hue='live?', data=df)
plt.show()


iris = sns.load_dataset('iris')

setosa = iris.loc[ iris.species == "setosa" ]
virginica = iris.loc[ iris.species == "virginica" ]
ax = sns.kdeplot( setosa.sepal_width, setosa.sepal_length, cmap="Reds", shade=True, shade_lowest=False )
ax = sns.kdeplot( virginica.sepal_width, virginica.sepal_length, cmap="Blues", shade=True, shade_lowest=False )
plt.show()
