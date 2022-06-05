#EDA
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as style

#Preprocessing
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import RepeatedStratifiedKFold

from imblearn.pipeline import Pipeline

#Machine Learning
from sklearn.linear_model import LogisticRegression
from causallift import CausalLift
#import xgboost
#import lightgbm as lgb
#from lightgbm import LGBMClassifier

#Evaluation
#from sklearn.model_selection import RepeatedStratifiedKFold
#from sklearn.model_selection import GridSearchCV

#Uplift
from sklift.models import TwoModels
#from sklift.models import SoloModel
#from sklift.models import ClassTransformation
from sklift.metrics import qini_auc_score
from sklift.viz import plot_qini_curve
from sklift.viz import plot_uplift_curve
from sklift.viz import plot_uplift_by_percentile

import pickle
from joblib import dump

#import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
#%matplotlib inline

#Read the dataset
#df = pd.read_csv("criteo-uplift-data.csv")
df_ml = pd.read_csv("exposed.csv")


#Scale the feature set
scaled = preprocessing.StandardScaler()
df_sca = scaled.fit_transform(df_ml.drop(['conversion','exposure'], axis=1))

df_sca = pd.DataFrame(data=scaled, columns = df_ml.drop(['conversion','exposure'], axis = 1).columns)
df_exposure = df_ml['exposure']
df_conversion = df_ml['conversion']

#Exposed flag is our new Treatment indicator
X   = pd.DataFrame(df_sca.tolist())
treatment = df_ml['exposure']
y   = df_ml['conversion']

#Split dataset into a Train & Test set, stratified over Treatment (i.e. Exposed flag)
X_train, X_test, treatment_train, treatment_test, y_train, y_test = train_test_split(X,treatment,y, random_state=23, stratify=treatment, test_size=0.33)

treatment_model = LogisticRegression(C= 0.00001, penalty= 'elasticnet', solver= 'saga', l1_ratio=0.1, random_state=23)
control_model = LogisticRegression(C= 0.00001, penalty= 'elasticnet', solver= 'saga', l1_ratio=0.1, random_state=23)
LogReg = TwoModels(estimator_trmnt = treatment_model, estimator_ctrl = control_model, method='vanilla')

#Train the model
logreg_tm = LogReg.fit(X_train, y_train, treatment_train)
#uplift_logreg = logreg_tm.predict(X_test)

# save the model as pkl
filename = 'uplift_model.pkl'
pickle.dump(logreg_tm, open(filename, 'wb')

# save model as joblib
dump(logreg_tm, 'uplift_model.joblib')



