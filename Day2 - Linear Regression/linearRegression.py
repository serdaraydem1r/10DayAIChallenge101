import pandas as pd
file_path = "/Users/fsa/Desktop/data/bostonHousingDay2/bostonHousing.csv"
df = pd.read_csv(file_path)

# Veriyi İnceleme
print(df.head())
print(df.info())
print(df.describe())
print(df.corr(method="pearson"))
print(df.isnull().sum())

# rm değişkeninde 5 adet boş değer var.
print(df["rm"].head())
print(df["rm"].describe())
# ortalama ile doldurucaz.
df["rm"] = df["rm"].fillna(df["rm"].mean())
print(df["rm"].isnull().sum())

from sklearn.model_selection import train_test_split
X = df.drop(columns="medv")
y = df["medv"]

print(X.shape, y.shape) # (506,13) (506,)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

print(f"Katsayılar: ", lr.coef_)
print(f"Sabit: ", lr.intercept_)

from sklearn.metrics import mean_squared_error, r2_score

# Test seti üzerinde tahmin yapma
y_pred = lr.predict(X_test)

# Performans metriklerini hesaplama
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)


import matplotlib.pyplot as plt

# Gerçek ve tahmin edilen değerleri karşılaştırma grafiği
plt.scatter(y_test, y_pred)
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahmin Edilen Değerler")
plt.title("Gerçek vs Tahmin Edilen Değerler")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.show()