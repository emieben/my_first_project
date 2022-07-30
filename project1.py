import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import  RandomForestRegressor
sns.set()

# LOADING DATA
file = '/Users/eben/Library/Containers/com.microsoft.Excel/Data/Downloads/employee_promotion.csv'
df = pd.read_csv(file, encoding='Windows-1252')
print(df)
print("\n\n")
print(df.head(3))
print("\n\n")

# DATA_DESCRIPTION
print(f'number of rows {df.shape[0]}')
print(f'number of cols {df.shape[1]}')
print("\n\n")
print(df.describe())
print("\n\n")
print(df.info())
print("\n\n")


# DATA_PREPROCESSING
df1 = df.copy()
print((df1.isna().sum() / len(df1)) * 100)
print("\n\n")

# Fill NA
df1.age.fillna(df1['age'].median(), inplace=True )
df1.previous_year_rating.fillna(df1['previous_year_rating'].median(),  inplace=True )
df1.avg_training_score.fillna(df1['avg_training_score'].median(), inplace=True )
df1.education.fillna("Bachelor's", inplace=True )
df1.drop('employee_id', axis=1, inplace=True)
df1['gender'] = df1['gender'].map({'f': 0, 'm': 1}).astype(int)
print("\n\n")

# Split Numerical and Categorical
df_num = df1.select_dtypes(include=['int64', 'float64'])
df_cat = df1.select_dtypes(exclude=['int64', 'float64'])
print("\n\n")
"""EXPLORATORY DATA ANALYSIS"""

# UNIVARIATED ANALYSIS

plt.figure(figsize=(15,6))
sns.countplot(x='is_promoted', data=df1)
plt.show()


# Bivariate Analysis
plt.figure(figsize=(15,6))
sns.histplot(x='length_of_service', hue='is_promoted', data=df1, bins=30, kde=True)
plt.show()

plt.figure(figsize=(15,6))
sns.countplot(x='education', hue='is_promoted', data=df1)
plt.show()

plt.figure(figsize=(15,6))
sns.histplot(x='age', hue='is_promoted', data=df1, bins=20, kde=True)
plt.show()



# Multivariate Analysis
plt.figure(figsize=(15,6))
correlation = df_num.corr( method='pearson' )
sns.heatmap( correlation, annot=True );
plt.show()


# DATA PREPARATION
dfcat = df_cat.copy()
dfnum = df_num.drop('is_promoted', axis=1)
OHE = ce.OneHotEncoder(cols=['department',
                             'region',
                             'education',
                             'recruitment_channel'],use_cat_names=True)

dfcat = OHE.fit_transform(dfcat)
df_all = pd.concat([dfnum, dfcat], axis= 1)
X = df_all.copy()
y = df1['is_promoted']
print("\n\n")


# Scaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns = df_all.columns)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")

    elif train == False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")



print("Desicion tree")


# Decision Tree

from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
print_score(tree_clf, X_train, y_train, X_test, y_test, train=True)
print_score(tree_clf, X_train, y_train, X_test, y_test, train=False)
print("\n\n")

print("Random forest")

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=100)
rf_clf.fit(X_train, y_train)
print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
print_score(rf_clf,X_train, y_train, X_test, y_test, train=False)
