#%%
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

#%%
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
print(df.head())
print(df['target'].value_counts())

#%%
plt.figure(figsize=(10,10))
sns.countplot(x='target', data=df)
plt.xlabel('Sınıf')
plt.ylabel('Sayı')
plt.title('Hedef Değişkenin Dağılımı')
plt.xticks(ticks=[0,1], labels=['Malignant','Benign'])
plt.show()

#%%
corr_matrix = df.drop('target', axis=1).corr()

plt.figure(figsize=(15, 12))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Özellikler Arası Korelasyon Matrisi')
plt.show()

#%%
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%%
models = {
    "Logistic Regression":{
        "model":LogisticRegression(max_iter=1000),
        "params":{
            "C":[0.01,0.1,1,10],
            "solver":["liblinear","lbfgs"]
        }
    },
    "K-NN":{
        "model":KNeighborsClassifier(),
        "params":{
            "n_neighbors":[3,5,7,9],
            "weights":["uniform","distance"]
        }
    },
    "SVC":{
        "model":SVC(),
        "params":{
            "kernel":["rbf","sigmoid","linear"],
            "C":[0.01,0.1,1,10],
            "gamma":["auto","scale"]

        }
    },
    "Random Forest":{
        "model":RandomForestClassifier(),
        "params":{
            "n_estimators":[50,100,200],
            "max_depth":[5,10,15],
            "criterion":["gini","entropy"],
            "min_samples_split":[2,5,10],
        }
    },
    "Gradient Boosting":{
        "model":GradientBoostingClassifier(),
        "params":{
            "n_estimators":[50,100,200],
            "max_depth":[5,10,15],
            "learning_rate":[0.1,0.2,0.3,0.4],
        }
    }
}

best_models = {}

for model_name, mp in models.items():
    clf = GridSearchCV(mp["model"],mp["params"],cv=5,scoring="accuracy")
    clf.fit(X_train,y_train)
    best_models[model_name] = clf.best_estimator_
    print(f"En iyi {model_name} modeli: {clf.best_estimator_}")
    print(f"En iyi {model_name} skoru: {clf.best_score_}\n")


for model_name, model in best_models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{model_name} Test Doğruluğu: {acc}")
    print(f"{model_name} Sınıflandırma Raporu:")
    print(classification_report(y_test, y_pred))
    print(f"{model_name} Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n")

#%%
from sklearn.metrics import confusion_matrix,roc_curve,auc,precision_recall_curve

# En iyi modellerin ROC AUC ve Precision-Recall Curve grafikleri
def plot_evaluation_curves(models, X_test, y_test):
    plt.figure(figsize=(15, 5))

    # ROC AUC Curve
    plt.subplot(1, 2, 1)
    for model_name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC AUC Curve')
    plt.legend(loc='lower right')

    # Precision-Recall Curve
    plt.subplot(1, 2, 2)
    for model_name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        plt.plot(recall, precision, label=f'{model_name}')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')

    plt.show()

# Her model için Confusion Matrix çizdirme
def plot_confusion_matrices(models, X_test, y_test):
    plt.figure(figsize=(15, 10))
    for i, (model_name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        plt.subplot(2, 3, i + 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()
