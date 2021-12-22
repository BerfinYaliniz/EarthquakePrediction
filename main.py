import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
veri=pd.read_csv(('earthquake.csv'), encoding='utf-8', engine='python',sep=',',error_bad_lines=False)
data=pd.read_csv("earthquake.csv",sep=",")
veri=veri[['date','time','lat','city','long','depth','xm']]
data=veri.head(10)
veri=veri[['lat','long','depth','xm']]
data=veri.head(10)
print(data)
y=np.array(data['xm'])
X=np.array(data.drop('xm',axis=1))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)
linear=LinearRegression()
linear.fit(X_train,y_train) #Eğitim verileri
predict_data=np.array([[40.05,34.07,1.0]])
data2=linear.predict(predict_data)
print("TAHMİN EDİLEN DEĞER:",data2)
f,ax = plt.subplots(figsize=(10, 10))
sn.heatmap(veri.corr(), annot=True, linewidths=.9, fmt= '.2f',ax=ax)
plt.show()
veri.depth.plot(kind="line",grid=True,label="Derinlik",linestyle=":",color="r")
veri.lat.plot(kind="line",grid=True,label="Richter",linestyle="-",color="g")
veri.xm.plot(kind="line",grid=True,label="Büyüklük",linestyle="--",color="b")
plt.legend(loc="best")
plt.title("derinlik-richter-büyüklük")
plt.show()
