import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
X,y = load_iris(return_X_y=True)
cif = RandomForestClassifier()
cif.fit(X,y)
joblib.dump(cif, 'random_forest_model.pkl')