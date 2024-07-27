import pandas as pd
file_path = "/Users/fsa/Desktop/data/titanic/tested.csv"
# Verisetini yükleyin
df = pd.read_csv(file_path)

# Veri çerçevesinin ilk birkaç satırını görüntüleyin
print(df.head())

# Verisetinin özet bilgisi
print(df.info())

# Özet istatistikler
print(df.describe())

# Eksik değerleri kontrol etme
print(df.isnull().sum())

# Eksik değerleri doldurma
"""
Veri setimizde bazı sütünlara ait veriler eksik. 
Eğer eksik veri sayısı çok değilse mod veya median ile veya 
kendimizin belirleyeceği bir sayı ile doldurabiliriz. 
Fakat "Cabin" sütunu gibi çok fazla değer boş ise o sütun veriden çıkarılır.
Age - 86/418
Embarked - 
"""
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.drop(columns=['Cabin'])

# Eksik değerlerin olmadığını doğrulama
print(df.isnull().sum())

# Kategorik değişkenleri kodlama
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# İlk birkaç satırı görüntüleme
print(df.head())

# Özellik ölçeklendirme
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Ölçeklendirmeyi doğrulama
print(df[['Age', 'Fare']].describe())

# Veri görselleştirme
import seaborn as sns
import matplotlib.pyplot as plt

# 'Age' sütununun dağılım grafiği
sns.histplot(df['Age'], kde=True)
plt.title('Age Distribution')
plt.show()

# Cinsiyete göre hayatta kalma oranı
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival Rate by Sex')
plt.show()

# Yolcu sınıfına göre hayatta kalma oranı
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Pclass')
plt.show()
print("Kod Çalıştı")




