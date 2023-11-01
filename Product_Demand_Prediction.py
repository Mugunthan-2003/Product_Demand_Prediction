import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor,plot_tree

data = pd.read_csv('C:\\Users\\Mugun\\Desktop\\Dataset\\PoductDemand.csv')
print("Description of columns : \n")
print(data.describe())  #decription of each column
print("\nNo.of Null Columns :\n",data.isnull().sum())  #count of null values in columns
data = data.dropna()    #to remove null data

print("\nHeatmap to show correlation between attributes :-")
correlation=data.corr(method='pearson')
sns.heatmap(correlation, cmap="coolwarm",annot=True)
plt.show()  #To show correlation between attributes

x=data[["Total Price","Base Price"]]
y=data["Units Sold"]

xtr,xte,ytr,yte=train_test_split(x,y,test_size=0.2,random_state=42)
model = DecisionTreeRegressor()
model.fit(xtr,ytr)

features = np.array([[100, 120]])
us=model.predict(features)
print("Total Price : Rs.",features[0][0])
print("Base Price : Rs.",features[0][1])
print("Predicted value for Unit Sold : ",us[0])
