import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime
import os
import io

st.set_page_config(page_title='Pré-processamento de Dados', layout='wide')

caminho_dataset = os.path.join(os.path.dirname(__file__), '..', 'marketing_campaign.csv')
df = pd.read_csv(caminho_dataset, sep='\t')
pd.set_option('display.max_columns', 29)

st.title("Pré-Processamento dos Dados")

st.header("Visualização Inicial dos Dados")
st.write(df.head(10))

st.subheader("Remoção de Colunas Desnecessárias")
df_sub = df.drop(columns=["ID", "Recency", 'Dt_Customer', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
                          'MntSweetProducts', 'AcceptedCmp3', 'MntGoldProds', 'AcceptedCmp4', 'AcceptedCmp5',
                          'AcceptedCmp2', 'AcceptedCmp1', 'Response', 'Z_CostContact', 'Z_Revenue'])
st.write(df_sub.head(10))

st.subheader("Valores Únicos em Colunas Categóricas")
st.write("Valores únicos em Education:", df_sub['Education'].unique())
st.write("Valores únicos em Marital_Status:", df_sub['Marital_Status'].unique())

st.subheader("Mapeamento de Valores Categóricos")
education_mapping = {'Basic': 1, '2n Cycle': 2, 'Graduation': 3, 'Master': 4, 'PhD': 5}
marital_status_mapping = {'Single': 1, 'Married': 2, 'Together': 3, 'Divorced': 4, 'Widow': 5, 'Alone': 6, 'Absurd': 7, 'YOLO': 8}
df_sub['Education'] = df_sub['Education'].map(education_mapping)
df_sub['Marital_Status'] = df_sub['Marital_Status'].map(marital_status_mapping)
st.write(df_sub[['Education', 'Marital_Status']].head(10))

st.subheader("Informações do DataFrame")
buffer = io.StringIO()
df_sub.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

st.subheader("Estatísticas Descritivas")
st.write(df_sub.describe())

st.subheader("Cálculo da Idade")
year = datetime.today().year
age = year - df_sub['Year_Birth']
df_sub['Age'] = age
df_sub_2 = df_sub.drop(columns=['Year_Birth'])
st.write(df_sub_2.head())

st.subheader("Histograma dos Dados")
fig, ax = plt.subplots(figsize=(20, 15))
df_sub_2.hist(bins=30, ax=ax)
st.pyplot(fig)

st.subheader("Boxplot dos Dados")
fig, ax = plt.subplots(figsize=(15, 10))
df_sub_2.plot(kind='box', subplots=True, layout=(4, 4), ax=ax)
st.pyplot(fig)

st.subheader("Consulta de Dados")
st.write(df_sub_2.query("Income > 140000"))

df_sub_3 = df_sub_2.drop([164, 617, 655, 687, 1300, 1653, 2132, 2233])

st.subheader("Verificação de Valores Nulos")
st.write(df_sub_3.isnull().sum())

df_sub_3['Income'].fillna(df_sub_3['Income'].mean(), inplace=True)

st.subheader("Heatmap de Valores Nulos")
fig, ax = plt.subplots()
sns.heatmap(df_sub_3.isnull(), ax=ax)
st.pyplot(fig)

st.subheader("Boxplot de Idade")
grafico = px.box(df_sub_3, y='Age')
st.plotly_chart(grafico)

st.subheader("Consulta de Idade")
st.write(df_sub_3.loc[df_sub_3['Age'] > 81])

df_sub_4 = df_sub_3.drop([192, 239, 339])

st.subheader("Criação de Novas Colunas")
df_sub_4['Young'] = df_sub_4.apply(lambda x: x['Teenhome'] + x['Kidhome'], axis=1)
df_test = df_sub_4.drop(columns=['Kidhome', 'Teenhome'])
st.write(df_test.head(2))

st.subheader("Contagem de Jovens")
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(y="Young", data=df_test, order=df_test["Young"].value_counts().index, ax=ax)
ax.set_title("Números de Crianças")
st.pyplot(fig)

st.subheader("Correlação entre Variáveis")
correlations = df_test.corr()
fig, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(correlations, annot=True, ax=ax)
st.pyplot(fig)

caminho_saida = os.path.join(os.path.dirname(__file__), '..', 'df_pre_processado.csv')
df_test.to_csv(caminho_saida, index=False, sep='\t')