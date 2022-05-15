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

#import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
#%matplotlib inline

#Read the dataset
df = pd.read_csv("criteo-uplift-data.csv")

#Scale the feature set
scaled = preprocessing.StandardScaler()
df_sca = scaled.fit_transform(df_ml.drop(['conversion','exposure'], axis=1))

#df_sca = pd.DataFrame(data=scaled, columns = df_ml.drop(['conversion','exposure'], axis = 1).columns)
df_exposure = df_ml['exposure']
df_conversion = df_ml['conversion']

#Exposed flag is our new Treatment indicator
X   = pd.DataFrame(df_sca.tolist())
treatment = df_ml['exposure']
y   = df_ml['conversion']

