import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

st.set_page_config(page_title='Visualização e Análise de Dados', layout='wide')

st.title("Visualização e Análise dos Dados")

caminho_dataset = os.path.join(os.path.dirname(__file__), '..', 'df_pre_processado.csv')
df_test = pd.read_csv(caminho_dataset, sep='\t')  # Ajuste conforme necessário

st.header("Distribuição de Compras por Faixa Etária")
fig, ax = plt.subplots(2, 1, figsize=(12, 12))
sns.boxplot(x=pd.cut(df_test['Age'], bins=5), y='NumStorePurchases', data=df_test, ax=ax[0])
ax[0].set_title('Distribuição de Compras em Loja por Faixa Etária')
ax[0].set_xlabel('Faixa Etária')
ax[0].set_ylabel('Compras em Loja')

sns.boxplot(x=pd.cut(df_test['Age'], bins=5), y='NumWebPurchases', data=df_test, ax=ax[1])
ax[1].set_title('Distribuição de Compras na Web por Faixa Etária')
ax[1].set_xlabel('Faixa Etária')
ax[1].set_ylabel('Compras na Web')
st.pyplot(fig)

st.subheader("Média de Compras por Faixa Etária")
age_bins = pd.cut(df_test['Age'], bins=[20, 30, 40, 50, 60, 70, 80, 90])
store_purchases_by_age = df_test.groupby(age_bins)['NumStorePurchases'].mean()
web_purchases_by_age = df_test.groupby(age_bins)['NumWebPurchases'].mean()
st.write("Média de Compras em Loja por Faixa Etária:")
st.write(store_purchases_by_age)
st.write("Média de Compras na Web por Faixa Etária:")
st.write(web_purchases_by_age)

st.subheader("Média de Compras por Estado Civil")
store_purchases_by_marital_status = df_test.groupby('Marital_Status')['NumStorePurchases'].mean()
web_purchases_by_marital_status = df_test.groupby('Marital_Status')['NumWebPurchases'].mean()
st.write("Média de Compras em Loja por Estado Civil:")
st.write(store_purchases_by_marital_status)
st.write("Média de Compras na Web por Estado Civil:")
st.write(web_purchases_by_marital_status)

st.subheader("Gráficos de Barras por Estado Civil")
df_grouped = df_test.groupby('Marital_Status').agg({
    'NumStorePurchases': 'mean',
    'NumWebPurchases': 'mean'
}).reset_index()

fig, ax = plt.subplots(2, 1, figsize=(10, 10))
sns.barplot(x='Marital_Status', y='NumStorePurchases', data=df_grouped, palette='Blues_d', ax=ax[0])
ax[0].set_title('Média de Compras em Loja por Estado Civil')
ax[0].set_xlabel('Estado Civil')
ax[0].set_ylabel('Média de Compras em Loja')
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)

sns.barplot(x='Marital_Status', y='NumWebPurchases', data=df_grouped, palette='Greens_d', ax=ax[1])
ax[1].set_title('Média de Compras na Web por Estado Civil')
ax[1].set_xlabel('Estado Civil')
ax[1].set_ylabel('Média de Compras na Web')
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)
st.pyplot(fig)

st.subheader("Gráficos de Barras por Faixa de Renda")
bins = [0, 20000, 40000, 60000, 80000, 100000]
labels = ['0-20k', '20k-40k', '40k-60k', '60k-80k', '80k-100k']
df_test['Income_Group'] = pd.cut(df_test['Income'], bins=bins, labels=labels, right=False)

income_grouped = df_test.groupby('Income_Group').agg({
    'NumStorePurchases': 'mean',
    'NumWebPurchases': 'mean'
}).reset_index()

fig, ax = plt.subplots(2, 1, figsize=(10, 10))
sns.barplot(x='Income_Group', y='NumStorePurchases', data=income_grouped, palette='Blues_d', ax=ax[0])
ax[0].set_title('Média de Compras em Loja por Faixa de Renda')
ax[0].set_xlabel('Faixa de Renda')
ax[0].set_ylabel('Média de Compras em Loja')

sns.barplot(x='Income_Group', y='NumWebPurchases', data=income_grouped, palette='Greens_d', ax=ax[1])
ax[1].set_title('Média de Compras na Web por Faixa de Renda')
ax[1].set_xlabel('Faixa de Renda')
ax[1].set_ylabel('Média de Compras na Web')
st.pyplot(fig)

st.subheader("Gráficos de Barras por Nível de Educação")
education_grouped = df_test.groupby('Education').agg({
    'NumStorePurchases': 'mean',
    'NumWebPurchases': 'mean'
}).reset_index()

fig, ax = plt.subplots(2, 1, figsize=(10, 10))
sns.barplot(x='Education', y='NumStorePurchases', data=education_grouped, palette='Blues_d', ax=ax[0])
ax[0].set_title('Média de Compras em Loja por Nível de Educação')
ax[0].set_xlabel('Nível de Educação')
ax[0].set_ylabel('Média de Compras em Loja')
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)

sns.barplot(x='Education', y='NumWebPurchases', data=education_grouped, palette='Greens_d', ax=ax[1])
ax[1].set_title('Média de Compras na Web por Nível de Educação')
ax[1].set_xlabel('Nível de Educação')
ax[1].set_ylabel('Média de Compras na Web')
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)
st.pyplot(fig)

st.subheader("Gráficos de Barras por Idade")
df_grouped_age = df_test.groupby('Age').agg({
    'NumStorePurchases': 'mean',
    'NumWebPurchases': 'mean'
}).reset_index()

fig, ax = plt.subplots(2, 1, figsize=(10, 10))
sns.barplot(x='Age', y='NumStorePurchases', data=df_grouped_age, palette='Blues_d', ax=ax[0])
ax[0].set_title('Média de Compras em Loja por Idade')
ax[0].set_xlabel('Idade')
ax[0].set_ylabel('Média de Compras em Loja')
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)

sns.barplot(x='Age', y='NumWebPurchases', data=df_grouped_age, palette='Greens_d', ax=ax[1])
ax[1].set_title('Média de Compras na Web por Idade')
ax[1].set_xlabel('Idade')
ax[1].set_ylabel('Média de Compras na Web')
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)
st.pyplot(fig)

st.subheader("Criação de Grupos de Idade")
bins = [18, 30, 55, 80, 100]
labels = [1, 2, 3, 4]
df_test['Age_Group'] = pd.cut(df_test['Age'], bins=bins, labels=labels, right=False)
df_test['Age_Group'] = df_test['Age_Group'].cat.add_categories([0]).fillna(0).astype(int)
st.write(df_test[['Age', 'Age_Group']].head(20))

st.subheader("Criação de Grupos de Renda")
bins = [0, 20000, 40000, 60000, 80000, 100000, float('inf')]
labels = [1, 2, 3, 4, 5, 6]
df_test['Income_Group'] = pd.cut(df_test['Income'], bins=bins, labels=labels, right=False)
df_test['Income_Group'] = df_test['Income_Group'].astype(float).fillna(0).astype(int)
st.write(df_test[['Income', 'Income_Group']].head(20))

st.subheader("Criação de Novas Features")
df_test['TotalPurchases'] = df_test['NumWebPurchases'] + df_test['NumStorePurchases']
df_test['WebToStoreRatio'] = df_test['NumWebPurchases'] / (df_test['NumStorePurchases'] + 1)
st.write(df_test[['NumWebPurchases', 'NumStorePurchases', 'TotalPurchases', 'WebToStoreRatio']].head())

st.header("Dataset Final")
st.write(df_test.head(20))

caminho_saida = os.path.join(os.path.dirname(__file__), '..', 'df_final.csv')
df_test.to_csv(caminho_saida, index=False, sep='\t')