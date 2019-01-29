import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from lightgbm import LGBMClassifier


data=pd.read_csv("fuc.csv")
xtrain,xtest,ytrain,ytest=train_test_split(data.ix[:,data.columns!=9000],data.ix[:,9000])


lg=LGBMClassifier().fit(xtrain,ytrain)


joblib.dump(lg,'save.sav')



