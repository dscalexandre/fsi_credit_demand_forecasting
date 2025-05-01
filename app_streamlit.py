import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from datetime import timedelta
import matplotlib.pyplot as plt

# Título
st.title("Hiring Demand Forecasting")

# Barra lateral
st.sidebar.header("Bem-Vindo(a)")
st.sidebar.info('Hiring Demand Forecasting')

# Ajuste da barra lateral
st.sidebar.markdown("&#8203;" * 100)

# Decorador para melhorar a performance da aplicação
@st.cache_resource
def load_model_and_scaler():
    """Carrega o modelo LSTM e o scaler."""
    try:
        model = tf.keras.models.load_model('models/lstm_model.keras')
        scaler = joblib.load('models/scaler.gz')
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Erro ao carregar o modelo ou o scaler: {e}")
        return None, None

@st.cache_data
def load_data():
    """Carrega os dados históricos."""
    try:
        df = pd.read_csv('data/clean_data.csv')
        df['data'] = pd.to_datetime(df['data'], format='%Y-%m-%d')
        df.set_index('data', inplace=True)
        df = df.asfreq('D')  
        return df
    except FileNotFoundError as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return None

model, scaler = load_model_and_scaler()
df_historico = load_data()

if model is None or scaler is None or df_historico is None:
    st.stop()

# Módulo de seleção de página
def seleciona_pagina():
    """Cria um menu lateral para selecionar a página."""
    option = st.sidebar.selectbox(
        'Selecionar opção:',
        ['Visualizar Gráficos', 'Visualizar Tabela de Dados', 'Fazer Previsões']
    )
    return option

def indicadores_tecnicos():
    """Exibe gráficos"""
    st.subheader("Visualizar Gráficos")
    st.info("Série Temporal.")
    # Exemplo de plot dos dados históricos
    if df_historico is not None:
        st.line_chart(df_historico['contratacoes'])

def imprime_tabela():
    """Exibe a tabela de dados."""
    st.subheader("Visualizar Tabela de Dados")
    if df_historico is not None:
        num_rows = st.sidebar.slider("Número de Registros para Visualizar", 5, len(df_historico), 20)
        st.dataframe(df_historico.tail(num_rows))

        # Download dos dados
        st.sidebar.download_button(
            label="Baixar Dados em CSV",
            data=df_historico.to_csv().encode('utf-8'),
            file_name='dados_historicos.csv',
            mime='text/csv',
        )

def previsoes():
    """Módulo para fazer previsões a partir dos dados."""
    st.subheader("Fazer Previsões")

    if df_historico is not None and model is not None and scaler is not None:
        look_back = 30
        n_dias_prever = st.slider("Número de dias para prever", 30, 365, 90)

        if st.button("Gerar Previsão"):
            contratacoes = df_historico['contratacoes'].values
            last_30_days = contratacoes[-look_back:].reshape(-1, 1)
            entrada_normalizada = scaler.transform(last_30_days)
            entrada = entrada_normalizada.copy()
            previsoes_normalizadas = []

            for _ in range(n_dias_prever):
                entrada_modelo = entrada[-look_back:].reshape((1, look_back, 1))
                pred_normalizada = model.predict(entrada_modelo, verbose=0)
                previsoes_normalizadas.append(pred_normalizada[0, 0])
                entrada = np.append(entrada, pred_normalizada, axis=0)

            previsoes = scaler.inverse_transform(np.array(previsoes_normalizadas).reshape(-1, 1)).flatten()

            data_inicio_previsao = df_historico.index[-1] + timedelta(days=1)
            datas_futuras = pd.date_range(start=data_inicio_previsao, periods=n_dias_prever, freq='D')
            df_previsao = pd.DataFrame({'data': datas_futuras, 'previsao_contratacoes': previsoes})
            df_previsao.set_index('data', inplace=True)

            st.subheader("Previsão de Contratações")
            st.dataframe(df_previsao)

            # Plotar os últimos dias históricos + previsão
            num_dias_historico_plot = st.slider("Visualizar Histórico (últimos dias)", 30, 180, 90)
            plt.figure(figsize=(10, 4))
            plt.plot(df_historico.index[-num_dias_historico_plot:], df_historico['contratacoes'].iloc[-num_dias_historico_plot:], label='Histórico', color='blue')
            plt.plot(df_previsao.index, df_previsao['previsao_contratacoes'], label='Previsão', color='red', linestyle='--')
            plt.title('Previsão de Contratações')
            plt.xlabel('Data')
            plt.ylabel('Número de Contratações')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)
    else:
        st.error("Erro ao carregar dados, modelo ou scaler. Impossível fazer previsões.")

# Seleciona a página com base na escolha do usuário na barra lateral
pagina_selecionada = seleciona_pagina()

if pagina_selecionada == 'Visualizar Gráficos':
    indicadores_tecnicos()
elif pagina_selecionada == 'Visualizar Tabela de Dados':
    imprime_tabela()
elif pagina_selecionada == 'Fazer Previsões':
    previsoes()

# Rodapé 
st.caption('Autor Alexandre Rodrigues')



