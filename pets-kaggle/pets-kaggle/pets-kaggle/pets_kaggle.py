import pandas as pd

train = pd.read_csv('C:/Users/User/Desktop/source/repos/KAGGLE/pets-kaggle/train/train.csv', engine='python')
test = pd.read_csv('C:/Users/User/Desktop/source/repos/KAGGLE/pets-kaggle/test/test.csv', engine='python')
train = pd.DataFrame(train)
test = pd.DataFrame(test)

pd.set_option('display.width',400)
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows', 100)
pd.options.display.float_format = '{:,.0f}'.format

train.drop(['Name','RescuerID','VideoAmt','PhotoAmt','Description'], axis=1, inplace=True)
test.drop(['Name','RescuerID','VideoAmt','PhotoAmt','Description'], axis=1, inplace=True)
from sklearn.utils import shuffle
train = shuffle(train)
test = shuffle(test)
train.fillna(train.mean())
test.fillna(test.mean())

import cats


