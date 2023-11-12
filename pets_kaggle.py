import pandas as pd
from sklearn.utils import shuffle

train = pd.read_csv('train/train.csv', engine='python')
test = pd.read_csv('test/test.csv', engine='python')
train = pd.DataFrame(train)
test = pd.DataFrame(test)

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.options.display.float_format = '{:,.0f}'.format

train.drop(['Name', 'RescuerID', 'VideoAmt', 'PhotoAmt', 'Description'], axis=1, inplace=True)
test.drop(['Name', 'RescuerID', 'VideoAmt', 'PhotoAmt', 'Description'], axis=1, inplace=True)

train = shuffle(train)
test = shuffle(test)
