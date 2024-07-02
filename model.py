import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

wine = pd.read_csv("winequality-red.csv")
wine.head(4)

x = wine.drop("quality",axis = 1)

y = wine.iloc[:,-1]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
wine_regression = LinearRegression()

wine_regression.fit(x_train,y_train)

import pickle
pickle_out = open("wine_regression.pkl","wb")
pickle.dump(wine_regression,pickle_out)
pickle_out.close()
