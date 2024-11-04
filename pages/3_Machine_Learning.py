import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.cluster import KMeans
import os

st.set_page_config(page_title='Modelos de Machine Learning', layout='wide')

st.title("Modelos de Machine Learning")

caminho_dataset = os.path.join(os.path.dirname(__file__), '..', 'df_final.csv')
df_test = pd.read_csv(caminho_dataset, sep='\t')  # Ajuste conforme necessário

## ML 1 - PREDIÇÃO DE COMPRAS

st.header("Predição de Compras")

st.subheader("Preparação dos Dados")
X = df_test.drop('NumStorePurchases', axis=1)
y = df_test['NumStorePurchases']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

st.subheader("Modelo XGBoost")
categorical_cols = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

xgb_model = xgb.XGBClassifier(
    colsample_bytree=0.8,
    gamma=0.2,
    learning_rate=0.1,
    max_depth=7,
    n_estimators=50,
    subsample=1.0,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)

xgb_model.fit(X_train, y_train_encoded)

y_pred_xgb = xgb_model.predict(X_test)

st.write("Matriz de Confusão - XGBoost")
conf_matrix = confusion_matrix(y_test_encoded, y_pred_xgb)
st.write(conf_matrix)

st.write("Relatório de Classificação - XGBoost")
st.write(classification_report(y_test_encoded, y_pred_xgb))

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

st.subheader("Importância das Características")

fig, ax = plt.subplots(figsize=(10, 8))
xgb.plot_importance(xgb_model, importance_type='weight', max_num_features=10, ax=ax)
ax.set_title('Importância das Características - XGBoost')
st.pyplot(fig)

## ML 2 - CLASSIFICAÇÃO POR GRUPOS

st.header("Classificação por Grupos")

st.subheader("K-Means Clustering")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)

kmeans.fit(X_scaled)

clusters_kmeans = kmeans.labels_

df_test['Cluster_KMeans'] = clusters_kmeans

st.write(df_test[['NumWebPurchases', 'Cluster_KMeans']].head(20))

st.subheader("Contagem de Clusters")
st.write(df_test['Cluster_KMeans'].value_counts())

st.subheader("Resumo dos Clusters")
cluster_summary = df_test.groupby('Cluster_KMeans').mean()
st.write(cluster_summary)

st.subheader("Coeficiente de Silhueta")
silhouette_avg = silhouette_score(X_scaled, clusters_kmeans)
st.write(f"Coeficiente de Silhueta: {silhouette_avg}")

st.subheader("Médias das Variáveis por Cluster")
cluster_summary = df_test.groupby('Cluster_KMeans').mean().drop(columns=['Income', 'Age']).reset_index()
cluster_summary_melted = cluster_summary.melt(id_vars='Cluster_KMeans', var_name='Variável', value_name='Média')

fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(data=cluster_summary_melted, x='Variável', y='Média', hue='Cluster_KMeans', palette='viridis', ax=ax)
plt.title('Médias das Variáveis por Cluster (Excluindo Income e Age)')
plt.xticks(rotation=45)
plt.xlabel('Variável')
plt.ylabel('Média')
plt.legend(title='Cluster')
st.pyplot(fig)

st.subheader("Média de Income por Cluster")
income_summary = df_test.groupby('Cluster_KMeans')['Income'].mean().reset_index()

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=income_summary, x='Cluster_KMeans', y='Income', palette='viridis', ax=ax)
plt.title('Média de Income por Cluster')
plt.xlabel('Cluster')
plt.ylabel('Média de Income')
st.pyplot(fig)

st.subheader("Média de Age por Cluster")
age_summary = df_test.groupby('Cluster_KMeans')['Age'].mean().reset_index()

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=age_summary, x='Cluster_KMeans', y='Age', palette='viridis', ax=ax)
plt.title('Média de Age por Cluster')
plt.xlabel('Cluster')
plt.ylabel('Média de Age')
st.pyplot(fig)

st.subheader("Visualização de Clusters - Compras na Web vs Idade")
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=df_test, x='Age', y='NumWebPurchases', hue='Cluster_KMeans', palette='viridis', ax=ax)
plt.title('Clusters de Compras na Web vs Idade')
st.pyplot(fig)

st.subheader("Visualização de Clusters - Compras em Lojas vs Idade")
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=df_test, x='Age', y='NumStorePurchases', hue='Cluster_KMeans', palette='viridis', ax=ax)
plt.title('Compras em Lojas vs Idade')
st.pyplot(fig)
