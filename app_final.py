import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
from datetime import datetime

# 1. Cargar la Capa Silver y Entrenar la IA
try:
    df = pd.read_csv('silver_historico.csv')
    columnas_modelo = ['tmo_historico_avg', 'fcr_historico_avg', 'total_micro_ausentismos', 'nota_calidad_avg', 'test_estres', 'test_empatia']
    X = df[columnas_modelo].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    # === EL TRUCO MAESTRO: AUTO-ORDENAMIENTO DE CLÚSTERES ===
    # Obtenemos los promedios reales de cada grupo que armó la IA
    centros = scaler.inverse_transform(kmeans.cluster_centers_)
    df_centros = pd.DataFrame(centros, columns=columnas_modelo)
    
    # Le damos una "Nota de Deseabilidad" a cada grupo
    # Sumamos lo bueno (FCR, Calidad, Empatía*10, Estrés*10) y restamos lo malo (Ausentismos*5)
    df_centros['puntaje_ideal'] = (df_centros['fcr_historico_avg'] + df_centros['nota_calidad_avg'] + 
                                   (df_centros['test_empatia'] * 10) + (df_centros['test_estres'] * 10) - 
                                   (df_centros['total_micro_ausentismos'] * 5))
    
    # Ordenamos: El puntaje más alto será el Nivel 0, el más bajo el Nivel 5
    orden_correcto = df_centros.sort_values(by='puntaje_ideal', ascending=False).index.tolist()
    
    # Creamos un diccionario traductor (ej: si el mejor grupo fue el 4, ahora será el 0)
    mapa_niveles = {id_original: nivel_logico for nivel_logico, id_original in enumerate(orden_correcto)}
    
    modelo_listo = True
except Exception as e:
    modelo_listo = False
    error_msg = str(e)

# ================= DISEÑO DEL APLICATIVO WEB =================
st.set_page_config(page_title="IA de Contratación - COSO ERM", page_icon="🛡️", layout="wide")

st.title("🛡️ Sistema Predictivo de Selección y Riesgos (Marco COSO)")
st.markdown("Este modelo evalúa a los postulantes comparándolos con la **Capa Silver** (histórico) y guarda las decisiones en la **Capa Gold**.")

if not modelo_listo:
    st.error(f"Falta subir el archivo 'silver_historico.csv' al repositorio. Error: {error_msg}")
else:
    # SECCIÓN DE INGRESO DE DATOS
    st.sidebar.header("📝 Datos del Postulante")
    nombre = st.sidebar.text_input("Nombre del Candidato", "Ej. Nazaret Perea")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Proyección Operativa")
        tmo = st.slider("TMO Esperado (min)", 3.0, 15.0, 7.5)
        fcr = st.slider("Resolución FCR (%)", 30.0, 100.0, 75.0)
    with col2:
        st.subheader("Riesgo de Desgaste")
        ausentismos = st.slider("Micro-ausentismos proyectados", 0, 50, 5)
        calidad = st.slider("Nota de Calidad esperada", 0.0, 100.0, 85.0)
    with col3:
        st.subheader("Perfil Psicológico")
        estres = st.slider("Tolerancia al Estrés (1-10)", 1.0, 10.0, 7.0)
        empatia = st.slider("Nivel de Empatía (1-10)", 1.0, 10.0, 8.0)

    st.markdown("---")
    
    # LÓGICA DE PREDICCIÓN Y GUARDADO
    if st.button("🧠 Procesar Evaluación de Riesgo", use_container_width=True):
        datos_candidato = [[tmo, fcr, ausentismos, calidad, estres, empatia]]
        datos_scaled = scaler.transform(datos_candidato)
        
        # Predicción cruda de la IA
        raw_cluster = kmeans.predict(datos_scaled)[0]
        # Traducción lógica con nuestro mapa
        cluster_final = mapa_niveles[raw_cluster]
        
        # Asignar nivel COSO asegurado
        if cluster_final == 0: decision_texto = "🟢 NIVEL 1: Sobresaliente / Embajador"
        elif cluster_final == 1: decision_texto = "🟩 NIVEL 2: Apto / Estable"
        elif cluster_final == 2: decision_texto = "🟡 NIVEL 3: Brecha Técnica / Inseguro"
        elif cluster_final == 3: decision_texto = "🟠 NIVEL 4: Fatiga Temprana / Burnout"
        elif cluster_final == 4: decision_texto = "🔴 NIVEL 5: Riesgo Operativo Crítico"
        else: decision_texto = "⚫ NIVEL 6: Riesgo Ético / Normativo"

        st.success(f"**Resultado para {nombre}:** {decision_texto}")

        # Guardar en la Capa Gold 
        nuevo_registro = pd.DataFrame([{
            'Fecha_Evaluacion': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Candidato': nombre,
            'TMO': tmo,
            'FCR': fcr,
            'Nivel_Riesgo_IA': cluster_final + 1,
            'Decision_COSO': decision_texto
        }])
        
        archivo_gold = 'gold_evaluaciones.csv'
        if not os.path.isfile(archivo_gold):
            nuevo_registro.to_csv(archivo_gold, index=False)
        else:
            nuevo_registro.to_csv(archivo_gold, mode='a', header=False, index=False)

    # ================= LA CAPA GOLD VISUAL =================
    st.markdown("---")
    st.title("🏆 CAPA GOLD: Base de Datos de Decisiones")
    
    archivo_gold = 'gold_evaluaciones.csv'
    if os.path.isfile(archivo_gold):
        df_gold = pd.read_csv(archivo_gold)
        st.dataframe(df_gold, use_container_width=True)
        
        csv = df_gold.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar Tabla de la Capa Gold (CSV)",
            data=csv,
            file_name='capa_gold_final.csv',
            mime='text/csv',
        )
    else:
        st.info("⚠️ La Capa Gold está vacía. Haz clic en 'Procesar Evaluación de Riesgo' para registrar al primer candidato.")
