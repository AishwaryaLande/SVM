import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

sal_test = pd.read_csv("C:/Users/ADMIN/Desktop/Data_Science_Assig/SVM/SalaryData_Test(1).csv")

sal_train= pd.read_csv("C:\\Users\\ADMIN\\Desktop\\Data_Science_Assig\\SVM\\SalaryData_Train(1).csv")

string_columns = ["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]

from sklearn import preprocessing
num = preprocessing.LabelEncoder()
for i in string_columns:
    sal_train[i] = num.fit_transform(sal_train[i])
    sal_test[i] = num.fit_transform(sal_test[i])
    
colnames = sal_train.columns
len(colnames[0:13])

sal_test.drop_duplicates(keep='first',inplace=True)
sal_train.drop_duplicates(keep='first',inplace=True)
sal_test.head()
sal_test.describe()
sal_test.columns
sal_test.shape
sal_test.isnull().sum()


sal_train.head()
sal_train.describe()
sal_train.columns
sal_train.shape
sal_train.isnull().sum()

plt.hist(sal_train['age']);plt.xlabel('age');plt.ylabel('Salary');plt.title('histogram of age')
plt.hist(sal_train['education']);plt.xlabel('education');plt.ylabel('Salary');plt.title('histogram of education')
plt.hist(sal_train['sex']);plt.xlabel('sex');plt.ylabel('Salary');plt.title('histogram of sex')
plt.hist(sal_test['age']);plt.xlabel('age');plt.ylabel('Salary');plt.title('histogram of age')
plt.hist(sal_test['education']);plt.xlabel('education');plt.ylabel('Salary');plt.title('histogram of education')
plt.hist(sal_test['sex']);plt.xlabel('sex');plt.ylabel('Salary');plt.title('histogram of sex')

sns.pairplot((sal_train),hue='Salary')
sns.pairplot((sal_test),hue='Salary')

#Q-plot
plt.plot(sal_train);plt.legend(list(sal_train.columns))
plt.plot(sal_test);plt.legend(list(sal_test.columns))

from scipy import stats
corr = sal_train.corr()
corr1 = sal_test.corr()

sns.heatmap(corr, annot=True)
sns.heatmap(corr1, annot=True)

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

#train,test = train_test_split(data,test_size = 0.3)
sal_train.head()
sal_test.head()

trainX = sal_train[colnames[0:13]]
trainY = sal_train[colnames[13]]
testX  = sal_test[colnames[0:13]]
testY  = sal_test[colnames[13]]

#kernel_linear
model_linear = SVC(kernel = "linear")
model_linear.fit(trainX,trainY)
pred_test_linear = model_linear.predict(testX)
np.mean(pred_test_linear==testY)

model_poly = SVC(kernel = "poly")
model_poly.fit(trainX,trainY)
pred_test_poly = model_poly.predict(testX)
np.mean(pred_test_poly==testY)

model_rbf = SVC(kernel = "rbf")
model_rbf.fit(trainX,trainY)
pred_test_rbf = model_rbf.predict(testX)
np.mean(pred_test_rbf==testY)
