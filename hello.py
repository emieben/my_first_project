import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
file = '/Users/eben/Library/Containers/com.microsoft.Excel/Data/Downloads/kl.csv'
df = pd.read_csv(file, encoding=' Windows-1252')
print(df)
print("\n\n")

print("\n\n")
df = df.drop(columns=['Loaned From','LS',	'ST',	'RS', 'LW',	'LF',	'CF',	'RF',	'RW',	'LAM', 'CAM',	'RAM',	'LM',	'LCM',	'CM',	'RCM',	'RM',	'LWB',	'LDM',	'CDM',
         'RDM',	'RWB',	'LB',	'LCB',	'CB',	'RCB',	'RB'],axis=1)
print("\n\n")
print(df.info())
print("\n\n")
print(df.head())
print("\n\n")
print(df.tail())
print("\n\n")
# import chardet
# with open(file, 'rb') as rawdata:
#     result = chardet.detect(rawdata.read(100000))
# print(result)
print(np.shape(df))
print("\n\n")
print(df.describe())
print("\n\n")

print("\n\n")
nulls = df.isnull().sum()
print(nulls)
print("\n\n")
nulls_percentage = nulls[nulls!=0]/df.shape[0]*100
print('the percentage value of null:\n')
print(round(nulls_percentage,2))
print("\n\n")
Height_cm = []
for i in list(df['Height'].values):
    try:
        Height_cm.append((float(str(i)[0]) * 12.0 + float(str(i)[2:])) * 2.54)
    except(ValueError):

        Height_cm.append(np.nan)

df['Height_cm'] = Height_cm
df.dropna(inplace = True)
print(df['Height_cm'].head())
print("\n\n")

print("mean : ", df['Height_cm'].mean())
print("standerd diviation :", df['Height_cm'].std())

df['Weight_kg'] = df['Weight'].str[:3].astype(float)/2.20462
print(df.head())
print("\n\n")
print("mean : ", df['Weight_kg'].mean())
print("standerd diviation :", df['Weight_kg'].std())


# df1=df.select_dtypes(include='number').nunique()
# print(df1)
# df['International Reputation']=df.International_Reputation.astype(str)
# df['Weak Foot']=df.Weak_Foot.astype(str)
# df['Skill Moves']=df.Skil_Moves.astype(str)
# print("\n\n")
# uniques = df.select_dtypes(exclude='number').nunique()
# print(uniques)
print("\n\n")


plt.figure(figsize=[10,7])
plt.hist(df['Age'])
plt.title("Distribution of age")
plt.show()

print(df['Nationality'].value_counts().head(40).plot(kind='bar', figsize=[15,5]))
plt.show()

football = df.copy()


def str2float(euros):
    if euros[-1] == 'M':
        return float(euros[1:-1])*1000000
    elif euros[-1] == 'K':
        return float(euros[1:-1])*1000

    else:
        return float(euros[1:])


football['Value'] = football['Value'].apply(lambda x: str2float(x))

football['Wage'] = football['Wage'].apply(lambda x: str2float(x))


print(football[['Name', 'Value', 'Wage']])

plt.hist(football['Value'])
plt.show()

plt.hist(football['Wage'])
plt.show()

sns.distplot(football['Age'])
plt.show()

sns.distplot(football['Value'])
plt.show()

sns.distplot(football['Wage'])
plt.show()

print("\n\n")
print("value")
print(football.sort_values(by='Value', ascending=False)[['Name','Value','Wage']][:15])
print("\n\n")
print("wage")
print(football.sort_values(by='Wage', ascending=False)[['Name','Value','Wage']][:15])

new_df = football.sort_values(by='Wage', ascending=False)
new_df1 = football.sort_values(by='Value', ascending=False)


plt.figure(figsize=(8,5))
wages = new_df["Wage"].head(5)
names = new_df["Name"].head(5)
plt.bar(names,wages,color='r')
plt.title("higest wage")
plt.show()

plt.figure(figsize=(8,5))
Value = new_df1["Value"].head(5)
name1 = new_df1["Name"].head(5)
plt.bar(name1,Value,color='b')
plt.title("Higest Value")
plt.show()


