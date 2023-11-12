import pets_kaggle
import cats
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

train1 = pd.DataFrame(pets_kaggle.train[pets_kaggle.train['Type'] % 2 != 0])
test1 = pd.DataFrame(pets_kaggle.test[pets_kaggle.test['Type'] % 2 != 0])

Adopttrain = pd.DataFrame(train1[['AdoptionSpeed']])
ID = pd.DataFrame(test1[['PetID']])
test1.drop(['PetID'], axis=1, inplace=True)
train1 = train1.drop(['PetID', 'AdoptionSpeed'], 1)

parameter_grid = {
    'criterion': ['entropy'],
    'max_depth': [100, 103, 106, 110, 113, 116, 120],
    'n_estimators': [70, 73, 76, 80, 83, 86, 90],
    'min_samples_leaf': [4]

}
clf = RandomForestClassifier(n_jobs=-1)
grid_searcher = GridSearchCV(clf, parameter_grid, verbose=2)
grid_searcher.fit(train1, Adopttrain.values.ravel())
clf_best = grid_searcher.best_estimator_
clf_best.fit(train1, Adopttrain.values.ravel())
predictions = pd.DataFrame(clf_best.predict(test1))

predictions.columns = [' AdoptionSpeed']

dogtest = ID.join(predictions)

file = pd.concat([dogtest, cats.cattest])
print(file)
file.to_csv('sample_submission.csv', index=False)
