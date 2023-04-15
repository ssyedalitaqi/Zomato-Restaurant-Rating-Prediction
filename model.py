import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
import warnings
warnings.filterwarnings('ignore')
data=pd.read_csv('Zomato_df.csv')
x=data.drop(['Unnamed: 0','rate'],axis=1)
y=data['rate']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=1)

et_model=ExtraTreesRegressor(n_estimators=120)
et_model.fit(x_train,y_train)
test_predict=et_model.predict(x_test)

import pickle
pickle.dump(et_model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print(test_predict)
