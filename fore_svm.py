import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fore = pd.read_csv("C:\\Users\\ADMIN\\Desktop\\Data_Science_Assig\\SVM\\forestfires.csv")

fore.head()
fore.describe()
fore.columns
fore.shape
fore.drop_duplicates(keep='first',inplace=True)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
fore['size_category']=le.fit_transform(fore['size_category'])
fore
fore['size_category'].unique()

fore.drop(["month","day","dayfri","daymon","daysat","daysun","daythu","daytue","daywed","monthapr","monthaug","monthdec","monthfeb","monthjan","monthjul","monthjun","monthmar","monthmay","monthnov","monthoct","monthsep"],axis=1,inplace=True) # Dropping the uncessary column
fore.size_category.value_counts()

fore.shape
fore.head()
fore.describe()
fore.columns
np.mean(fore)
np.median(fore)

y = y = fore["size_category"]
y=y.astype('int')

plt.hist(fore['FFMC']);plt.xlabel('FFMC');plt.ylabel('y');plt.title('histogram of FFMC')
plt.hist(fore['DMC']);plt.xlabel('DMC');plt.ylabel('y');plt.title('histogram of DMC')
plt.hist(fore['DC']);plt.xlabel('DC');plt.ylabel('y');plt.title('histogram of DC')
plt.hist(fore['ISI']);plt.xlabel('ISI');plt.ylabel('y');plt.title('histogram of ISI')
plt.hist(fore['RH']);plt.xlabel('RH');plt.ylabel('y');plt.title('histogram of RH')
plt.hist(fore['area']);plt.xlabel('area');plt.ylabel('y');plt.title('histogram of area')
plt.hist(fore['temp']);plt.xlabel('temp');plt.ylabel('y');plt.title('histogram of temp')
plt.hist(fore['rain']);plt.xlabel('rain');plt.ylabel('y');plt.title('histogram of rain')
plt.hist(fore['wind']);plt.xlabel('wind');plt.ylabel('y');plt.title('histogram of wind')

from scipy.stats import skew, kurtosis
skew(fore)
kurtosis(fore)

sns.boxplot(fore.size_category)
sns.boxplot(fore.temp)
sns.boxplot(fore.rain)
sns.boxplot(fore.wind)
sns.boxplot(fore.FFMC)
sns.boxplot(fore.DMC)
sns.boxplot(fore.DC)
sns.boxplot(fore.ISI)
sns.boxplot(fore.RH)
sns.boxplot(fore.area)

sns.pairplot((fore),hue='size_category')

#Q-plot
plt.plot(fore);plt.legend(list(fore.columns))
FFMC= np.array(fore['FFMC'])
DMC = np.array(fore['DMC'])
DC = np.array(fore['DC'])
ISI = np.array(fore['ISI'])
temp = np.array(fore['temp'])
RH= np.array(fore['RH'])
wind = np.array(fore['wind'])
rain = np.array(fore['rain'])
area = np.array(fore['area'])
size_category = np.array(fore['size_category'])

from scipy import stats
stats.probplot(FFMC, dist='norm', plot=plt);plt.title('Probability Plot of FFMC')
stats.probplot(DMC, dist='norm', plot=plt);plt.title('Probability Plot of DMC')
stats.probplot(DC, dist='norm', plot=plt);plt.title('Probability Plot of DC')
stats.probplot(ISI, dist='norm', plot=plt);plt.title('Probability Plot of ISI')
stats.probplot(temp, dist='norm', plot=plt);plt.title('Probability Plot of temp')
stats.probplot(RH, dist='norm', plot=plt);plt.title('Probability Plot of RH')
stats.probplot(wind, dist='norm', plot=plt);plt.title('Probability Plot of wind')
stats.probplot(rain, dist='norm', plot=plt);plt.title('Probability Plot of rain')
stats.probplot(area, dist='norm', plot=plt);plt.title('Probability Plot of area')
stats.probplot(size_category, dist='norm', plot=plt);plt.title('Probability Plot of size_category')

corr = fore.corr()
corr
sns.heatmap(corr, annot=True)

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
train,test = train_test_split(fore,test_size = 0.3)
train.head()
test.head()
train_X = train.iloc[:,:-1]
train_y = train.iloc[:,-1]
test_X  = test.iloc[:,:-1]
test_y  = test.iloc[:,-1]

model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)
np.mean(pred_test_linear==test_y)

model_poly = SVC(kernel = "poly")
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)
np.mean(pred_test_poly==test_y)

model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)
np.mean(pred_test_rbf==test_y)
