#%%
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["species"] = iris.target
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df["species"].value_counts())
print(df["species"].unique())
#%%
import seaborn as sns
import matplotlib.pyplot as plt

df["species"] = df["species"].map({0:"setosa",1:"versicolor",2:"virginica"})
sns.set(style="whitegrid", color_codes=True)
#özelliklerin birbiriyle olan ilişkileri
sns.pairplot(df,hue="species",markers=["o","s","D"])
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()

#%%
#Boxplot (Özelliklerin dağılımlarını türlere göre karşılaştırma)
plt.figure(figsize=(10, 6))
sns.boxplot(x='species', y='sepal length (cm)', data=df)
plt.title('Boxplot of Sepal Length by Species')
plt.show()

#%%
#Violin plot (Özelliklerin dağılımlarını ve yoğunluklarını türlere göre karşılaştırma)
plt.figure(figsize=(10, 6))
sns.violinplot(x='species', y='petal length (cm)', data=df)
plt.title('Violin Plot of Petal Length by Species')
plt.show()

#%%
#Heatmap (Özellikler arasındaki korelasyonu gösterme)
plt.figure(figsize=(8, 6))
corr_matrix = df.iloc[:, :-1].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Matrix of Iris Features')
plt.show()

"""
	1.	Pairplot: Iris veri setindeki tüm özelliklerin birbirlerine karşı dağılımını gösterir. 
	Ayrıca, farklı türleri farklı renkte göstermek, türler arasındaki farkları görmeyi kolaylaştırır.
	2.	Boxplot: Her tür için sepal length (cm) özelliğinin dağılımını gösterir. 
	Kutunun içindeki çizgi medyan değeri temsil eder, kutu ise verinin %50’sini içerir (alt ve üst çeyrekler). 
	Bu grafik, türler arasındaki boyut farklarını görsel olarak karşılaştırmayı kolaylaştırır.
	3.	Violin Plot: Benzer şekilde, petal length (cm) özelliğinin her tür için dağılımını ve yoğunluğunu gösterir. 
	Yoğunluğun daha yüksek olduğu yerler, daha fazla veri noktasının bulunduğunu gösterir.
	4.	Heatmap (Korelasyon Matrisi): Özellikler arasındaki ilişkiyi gösterir. 
	Pozitif korelasyonlar (bir özellik artarken diğerinin de arttığı durumlar) kırmızıya yakın renkte, 
	negatif korelasyonlar (bir özellik artarken diğerinin azaldığı durumlar) maviye yakın renkte gösterilir. 
	Bu grafik, hangi özelliklerin birbirleriyle en çok ilişkili olduğunu anlamak için kullanılır.
"""

#%%
from sklearn.model_selection import train_test_split
X = df.drop(columns=["species"])
y = df["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)
print(f"X_train.shape: {X_train.shape},X_test.shape: {X_test.shape}")
print(f"y_train.shape: {y_train.shape},y_test.shape: {y_test.shape}")

#%% Lojistik Regresyon
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
param_grid = {
    "C":[0.001, 0.01, 0.1, 1,10,100],
    "solver":["lbfgs","sgd","liblinear"],
    "penalty":["l1","l2"]
}
model = LogisticRegression()
grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train_scaled, y_train)
print(f"Best C:", grid_search.best_params_)
print(f"Best score:", grid_search.best_score_)

"""
Best C: {'C': 10, 'penalty': 'l2', 'solver': 'lbfgs'}
Best score: 0.95
"""
#%%
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n {classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n {confusion_matrix(y_test, y_pred)}")

#%% KNN
from sklearn.neighbors import KNeighborsClassifier
param_grid_knn={
    "n_neighbors":[3,5,7,9],
    "weights":["uniform","distance"],
    "algorithm":["ball_tree","kd_tree","brute"],
    "metric":["euclidean","manhattan"]
}
model_knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(model_knn, param_grid_knn, cv=5, scoring="accuracy")
grid_search_knn.fit(X_train_scaled, y_train)
print(f"Best C: {grid_search_knn.best_params_}")
print(f"Best score: {grid_search_knn.best_score_}")
"""
Best C: {'algorithm': 'ball_tree', 'metric': 'euclidean', 'n_neighbors': 3, 'weights': 'uniform'}
Best score: 0.9583333333333334
"""
y_pred = grid_search_knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n {classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n {confusion_matrix(y_test, y_pred)}")