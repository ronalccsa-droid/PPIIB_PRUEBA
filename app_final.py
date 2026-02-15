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
    
    # Seleccionamos las variables clave
    columnas_modelo = [
        'tmo_historico_avg', 'fcr_historico_avg', 'total_micro_ausentismos', 
        'nota_calidad_avg', 'test_estres', 'test_empatia'
    ]
    
    X = df[columnas_modelo].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Entrenamos con los 6 niveles de la Matriz COSO
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    modelo_listo = True
except Exception as e:
    modelo_listo = False
    error_msg = str(e)

# ================= DISEÑO DEL APLICATIVO WEB =================
st.set_page_config(page_title="IA de Contratación - COSO ERM", page_icon="🛡️", layout="wide")

st.title("🛡️ Sistema Predictivo de Selección y Riesgos (Marco COSO)")
st.markdown("Este modelo de Machine Learning evalúa a los postulantes comparándolos con el histórico de la **Capa Silver** (2 años) y guarda las decisiones en la **Capa Gold**.")

if not modelo_listo:
    st.error(f"Falta subir el archivo 'silver_historico.csv' a Colab. Error: {error_msg}")
else:
    st.sidebar.header("📝 Datos del Postulante")
    nombre = st.sidebar.text_input("Nombre del Candidato", "Ej. Carlos Mendoza")
    
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
    
    if st.button("🧠 Procesar Evaluación de Riesgo", use_container_width=True):
        # Predecir el nivel
        datos_candidato = [[tmo, fcr, ausentismos, calidad, estres, empatia]]
        datos_scaled = scaler.transform(datos_candidato)
        cluster = kmeans.predict(datos_scaled)[0]
        
        st.header(f"Resultados para: {nombre}")
        
        # Lógica de los 6 Niveles COSO
        if cluster == 0:
            decision_texto = "🟢 NIVEL 1: Sobresaliente / Embajador (Riesgo Nulo)"
            st.success(decision_texto)
            st.write("**Decisión:** Contratación Inmediata. Perfil estable con alta resolución y empatía.")
        elif cluster == 1:
            decision_texto = "🟩 NIVEL 2: Apto / Estable (Riesgo Bajo)"
            st.info(decision_texto)
            st.write("**Decisión:** Contratar. Cumple con los protocolos operativos de forma consistente.")
        elif cluster == 2:
            decision_texto = "🟡 NIVEL 3: Brecha Técnica / Inseguro (Riesgo Moderado)"
            st.warning(decision_texto)
            st.write("**Decisión:** Apto Condicionado. Requiere capacitación técnica urgente para mejorar sus tiempos.")
        elif cluster == 3:
            decision_texto = "🟠 NIVEL 4: Fatiga Temprana / Burnout (Riesgo Medio-Alto)"
            st.warning(decision_texto)
            st.write("**Decisión:** Alerta. Históricamente estos perfiles evaden llamadas. Ingreso directo a programa de seguimiento.")
        elif cluster == 4:
            decision_texto = "🔴 NIVEL 5: Riesgo Operativo Crítico (Falsos Rápidos)"
            st.error(decision_texto)
            st.write("**Decisión:** No Apto. Cortan rápido pero no resuelven problemas, generando sobrecostos a la empresa.")
        elif cluster == 5:
            decision_texto = "⚫ NIVEL 6: Riesgo Ético / Normativo (Riesgo Extremo)"
            st.error(decision_texto)
            st.write("**Decisión:** Bloqueo Definitivo. Baja empatía cruzada con pésima calidad, riesgo de maltrato al asegurado.")

        # ================= GUARDADO EN LA CAPA GOLD =================
        nuevo_registro = pd.DataFrame([{
            'Fecha': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Candidato': nombre,
            'TMO': tmo,
            'FCR': fcr,
            'Nivel_IA': cluster,
            'Decision_COSO': decision_texto
        }])
        
        archivo_gold = 'gold_evaluaciones.csv'
        if not os.path.isfile(archivo_gold):
            nuevo_registro.to_csv(archivo_gold, index=False)
        else:
            nuevo_registro.to_csv(archivo_gold, mode='a', header=False, index=False)
            
        st.success("💾 Evaluación guardada exitosamente en la Capa Gold.")

    # Mostrar la Capa Gold en pantalla para la presentación
    st.markdown("---")
    st.subheader("📂 Registros de la Capa Gold (Historial de Evaluaciones)")
    if os.path.isfile('gold_evaluaciones.csv'):
        df_gold = pd.read_csv('gold_evaluaciones.csv')
        st.dataframe(df_gold, use_container_width=True)
    else:
        st.info("Aún no hay evaluaciones registradas. Evalúa a un candidato para crear la Capa Gold.")
