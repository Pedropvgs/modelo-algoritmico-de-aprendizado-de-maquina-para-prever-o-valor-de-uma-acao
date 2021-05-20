#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importanto as bibliotecas Pandas, Scikit-learn (klearn) e Matplotlib
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df_ITSA4 = pd.read_csv("C://Users//pvgs_//Desktop//Faculdade//TCC/ITSA4.SA.csv", sep=",")


# In[3]:


#verificando informações sobre o formato dos dados do Dataframe
df_ITSA4.info()


# In[4]:


df_ITSA4.head(10)


# In[5]:


#criando médias móveis de 5 e 21 dias

df_ITSA4['mmd5'] = df_ITSA4['Close'].rolling(7).mean()
df_ITSA4['mmd21'] = df_ITSA4['Close'].rolling(21).mean()


# In[6]:


df_ITSA4.head(22)


# In[7]:


#obter os valores previstos de fechamento d+1
df_ITSA4['Close'] = df_ITSA4['Close'].shift(-1)
df_ITSA4.head()


# In[8]:


#retirar valores nulos
df_ITSA4.dropna(inplace=True)
df_ITSA4


# In[10]:


#separando linhas do data frame para teste, treinamento e validação
qtd_linhas = len(df_ITSA4)
qtd_linhas_treinamento = qtd_linhas -1260 
qtd_linhas_teste = qtd_linhas - 39

qtd_linhas_validacao = qtd_linhas_treinamento - qtd_linhas_teste  

info =(
    f"linhas treino 0:{qtd_linhas_treinamento}"
    f" linhas teste = {qtd_linhas_treinamento}:{qtd_linhas_teste}"
    f" linhas validacao={qtd_linhas_teste}:{qtd_linhas}"
)
info


# In[11]:


df_ITSA4 =df_ITSA4.reset_index(drop=True)
df_ITSA4


# In[12]:


#separando as features e labels
features = df_ITSA4.drop(['Date','Close'],1)
labels =df_ITSA4['Close']


# In[13]:


#utilizando o método SelectKBest para selecionar as melhores features para nosso modelo de treinamento
features_lista = ('Open','High','Low','Close','Adj Close','Volume.','mmd5','mmd21')

k_best_features =  SelectKBest(k='all')
k_best_features.fit_transform(features,labels)
k_best_features_scores = k_best_features.scores_
raw_pairs = zip(features_lista[1:],k_best_features_scores)
ordened_pairs = list(reversed(sorted(raw_pairs,key=lambda x: x[1])))

k_best_features_final = dict(ordened_pairs[:15])
best_features = k_best_features_final.keys()
print('')
print("Melhores features")
print(k_best_features_final)


# In[14]:


feautures = df_ITSA4.loc[:,['Low','Close','High','mmd5']]


# In[15]:


#Normalizando os dados das features realizando um balancemamento para melhor previsão, método min trata os dados como pesos de 0 e 1
#fazendo com que haja um balanceamento adequado pois sabemos que existem valores altos para não haver discrepancia 
scaler = MinMaxScaler().fit(features)
features_scale = scaler.transform(features)

print('Features: ',features_scale.shape)
print(features_scale)#normalizando dados de entrada


# In[16]:


#separando os dados de treino, teste e validação
xtreino = features_scale[:qtd_linhas_treinamento]
xteste = features_scale[qtd_linhas_treinamento:qtd_linhas_teste]

ytreino = labels[:qtd_linhas_treinamento]
yteste = labels[qtd_linhas_treinamento:qtd_linhas_teste]

print(len(xtreino), len(ytreino))
print(len(xteste),len(yteste))


# In[17]:


#treinamento usando regressão linear
lr = linear_model.LinearRegression()
lr.fit(xtreino,ytreino)
predicao=lr.predict(xteste)
cd = r2_score(yteste,predicao)

f'coeficiente de determinação:{cd * 100:.2f}'


# In[18]:


#rede neural

redeneural = MLPRegressor(max_iter=2000)

redeneural.fit(xtreino,ytreino)
predicao = redeneural.predict(xteste)

cd = redeneural.score(xteste,yteste)

f'coeficiente de determinação:{cd * 100:.2f}'


# In[19]:


#executando a previsao pelo modelo de regressão linear

previsao = features_scale[qtd_linhas_teste:qtd_linhas]

Date_full = df_ITSA4['Date']
Date = Date_full[qtd_linhas_teste:qtd_linhas]

Close_full = df_ITSA4['Close']
Close = Close_full[qtd_linhas_teste:qtd_linhas]

predicao = lr.predict(previsao)

df=pd.DataFrame({'Date':Date,'real':Close,'previsao':predicao})
df['real']=df['real'].shift(+1)

df.set_index('Date',inplace=True)

print(df)


# In[27]:


#plotando a previsão
plt.figure(figsize=(30,15))
plt.title('Previsão preço da ação da ITAUSA')
plt.plot(df['real'],label="real",color='blue',marker='o')
plt.plot(df['previsao'],label="previsao",color='red',marker='o')
plt.xlabel('Date')
plt.ylabel('Close')
leg = plt.legend()


# In[ ]:




