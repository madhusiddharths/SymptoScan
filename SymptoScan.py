import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, roc_auc_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# dataset cleaning and eda not included in this file
# dividing main data into 4 separate datasets, 1 for each disease
# allergy
df = pd.read_csv('disease.csv')
def feature(x):
    if x == 'ALLERGY':
        return 1
    else:
        return 0
df['TYPE'] = df.TYPE.apply(feature)
# df.to_csv('allergy.csv', index = False)

# cold
df = pd.read_csv('disease.csv')
def feature(x):
    if x == 'COLD':
        return 1
    else:
        return 0
df['TYPE'] = df.TYPE.apply(feature)
# duplicating values due to data imbalance
temp = df[df.TYPE == 1]
for i in range(5):
    df = pd.concat([df, temp], axis = 0)
df = df.sample(frac = 1).reset_index(drop = True)
# df.to_csv('cold.csv', index = False)

# covid
df = pd.read_csv('disease.csv')
def feature(x):
    if x == 'COVID':
        return 1
    else:
        return 0
df['TYPE'] = df.TYPE.apply(feature)
temp = df[df.TYPE == 1]
# duplicating values due to class imbalance
for i in range(5):
    df = pd.concat([df, temp], axis = 0)
df = df.sample(frac = 1).reset_index(drop = True)
# df.to_csv('covid.csv',index = False)

# flu
df = pd.read_csv('disease.csv')
def feature(x):
    if x == 'FLU':
        return 1
    else:
        return 0
df['TYPE'] = df.TYPE.apply(feature)
# df.to_csv('flu.csv', index=False)

# model for allergy
allergy = pd.read_csv('allergy.csv')
x_train = allergy.drop(['TYPE'], axis = 1)
y_train = allergy['TYPE']
tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5)
tree.fit(x_train, y_train)
pred = tree.predict(x_train)
metrics_allergy = [precision_score(y_train, pred), f1_score(y_train, pred), recall_score(y_train, pred), accuracy_score(y_train, pred), roc_auc_score(y_train, pred)]
metrics_allergy = [round(i, 4) for i in metrics_allergy]

# model for cold
cold = pd.read_csv('cold.csv')
x_train = cold.drop(['TYPE'], axis = 1)
y_train = cold['TYPE']
gboost = GradientBoostingClassifier(max_depth=10, min_samples_leaf=5, n_estimators=50, learning_rate=0.15)
gboost.fit(x_train, y_train)
pred = gboost.predict(x_train)
metrics_cold = [precision_score(y_train, pred), f1_score(y_train, pred), recall_score(y_train, pred), accuracy_score(y_train, pred), roc_auc_score(y_train, pred)]
metrics_cold = [round(i, 4) for i in metrics_cold]

# model for flu
flu = pd.read_csv('flu.csv')
x_train = flu.drop(['TYPE'], axis = 1)
y_train = flu['TYPE']
svc = LinearSVC(C = 0.01)
svc.fit(x_train, y_train)
pred = svc.predict(x_train)
metrics_flu = [precision_score(y_train, pred), f1_score(y_train, pred), recall_score(y_train, pred), accuracy_score(y_train, pred), roc_auc_score(y_train, pred)]
metrics_flu = [round(i, 4) for i in metrics_flu]

# model for covid
covid = pd.read_csv('covid.csv')
x_train = covid.drop(['TYPE'], axis = 1)
y_train = covid['TYPE']
rf = RandomForestClassifier(max_depth=10, min_samples_leaf=5, n_estimators = 100)
rf.fit(x_train, y_train)
pred = rf.predict(x_train)
metrics_covid = [precision_score(y_train, pred), f1_score(y_train, pred), recall_score(y_train, pred), accuracy_score(y_train, pred), roc_auc_score(y_train, pred)]
metrics_covid = [round(i, 4) for i in metrics_covid]

# checking how well train data has been fit
data = [metrics_cold, metrics_allergy, metrics_flu, metrics_covid]
columns = ['cold', 'allergy', 'flu', 'covid']
index = ['precision_score', 'f1-score', 'recall_score', 'accuracy_score', 'roc_auc_score']
# print(pd.DataFrame(data, columns=columns, index=index))
for i in data:
    print(i)

# meta classifier
df = pd.read_csv('disease.csv')
encoding = LabelEncoder()
df['TYPE'] = encoding.fit_transform(df['TYPE'])
df['TYPE'].value_counts()
x_train, x_test, y_train, y_test = train_test_split(df.loc[:, df.columns!='TYPE'], df['TYPE'], test_size=0.15)
data_train = pd.DataFrame(tree.predict(x_train), columns = ['Allergy'])
data_train['Cold'] = gboost.predict(x_train)
data_train['Covid'] = rf.predict(x_train)
data_train['Flu'] = svc.predict(x_train)
meta_model = SVC(kernel='rbf', C=0.1)
meta_model.fit(data_train, y_train)

# creating testing data
data_test = pd.DataFrame(list(zip(tree.predict(x_test), gboost.predict(x_test), rf.predict(x_test), svc.predict(x_test))), columns=['Allergy','Cold','Covid','Flu'])
pred = meta_model.predict(data_test)

# model evaluation
print("model evaluation")
print("precision score : ",precision_score(y_test, pred, average='micro'))
print("f1 score : ",f1_score(y_test, pred, average='micro'))
print("recall score : ",recall_score(y_test, pred, average='micro'))
print("accuracy score : ",accuracy_score(y_test, pred))
print(pd.DataFrame(classification_report(y_test, pred, output_dict=True, target_names=encoding.classes_)).T)

# printing it as a heat map
plt.figure(figsize = (18,8))
sns.heatmap(confusion_matrix(y_test, pred), annot = True, xticklabels = y_test.unique(), yticklabels = y_test.unique(), cmap = 'viridis')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()