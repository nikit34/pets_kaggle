import pets_kaggle 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

train2 = pd.DataFrame(pets_kaggle.train[pets_kaggle.train['Type'] % 2 == 0])
test2 = pd.DataFrame(pets_kaggle.test[pets_kaggle.test['Type'] % 2 == 0])

Adopttrain = pd.DataFrame(train2[['AdoptionSpeed']])
ID = pd.DataFrame(test2[['PetID']])
test2.drop(['PetID'],axis=1, inplace=True)
train2 = train2.drop(['PetID', 'AdoptionSpeed'], axis=1)


parameter_grid = {
            'criterion': ['entropy'],
            'max_depth': [30,33,36,40,43,46,50],
            'n_estimators': [40,43,46,50,53,56,60],
            'min_samples_leaf': [5]
        }
clf = RandomForestClassifier(n_jobs=-1)
grid_searcher = GridSearchCV(clf, parameter_grid, verbose=2)
grid_searcher.fit(train2, Adopttrain.values.ravel())
clf_best = grid_searcher.best_estimator_
clf_best.fit(train2, Adopttrain.values.ravel())
predictions = pd.DataFrame(clf_best.predict(test2))

predictions.columns = [' AdoptionSpeed']

cattest = ID.join(predictions)


