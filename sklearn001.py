import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
#read csv file
data = pd.read_csv("E:\\mystuffs\\archive\\train.csv")
#init classes
lav_sex = LabelEncoder()
imputer = SimpleImputer()
model = DecisionTreeClassifier(random_state=0)
#data preprocessing
data['new_sex'] = lav_sex.fit_transform(data['Sex'])
#data = data.drop(['PassengerId','Name'],axis='columns')
features = ['Pclass','Age','Fare','new_sex']
X = data[features]
y = data.Survived
train_X,check_X,train_y,check_y = train_test_split(X,y,random_state=0)
imputed_X_train = pd.DataFrame(imputer.fit_transform(train_X))
imputed_X_check = pd.DataFrame(imputer.transform(check_X))
imputed_X_train.columns = train_X.columns
imputed_X_check.columns = check_X.columns
# print(check_y.head(10))
#model_training
trained_model = model.fit(imputed_X_train,train_y)
pred = trained_model.predict(check_X)
print(pred)
