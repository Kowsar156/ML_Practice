import pandas as pd
from sklearn.tree import DecisionTreeRegressor

home_data = pd.read_csv('train.csv')
#print(home_data.columns)

#SalePrice is the target column
y = home_data.SalePrice
feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_names]

model = DecisionTreeRegressor(random_state=5)
model.fit(X,y)

prediction = model.predict(X)
print(prediction)