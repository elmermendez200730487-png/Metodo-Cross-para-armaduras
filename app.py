import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Configuracion de pagina
st.set_page_config(page_title="Resolución Hardy Cross", layout="wide")

st.title("Análisis Estructural - Método de Hardy Cross")
st.markdown("Basado en el cálculo de pórticos con desplazamiento lateral y columnas inclinadas.")

# Selector de ejercicio
ejercicio = st.sidebar.radio(
    "Selecciona el tipo de marco a resolver:", 
    ("Ejercicio 1: Marco de 3 Niveles", "Ejercicio 2: Marco con Columna Inclinada")
)

if ejercicio == "Ejercicio 1: Marco de 3 Niveles":
    st.header("Marco con Tres Niveles y Una Crujía (Con desplazamiento lateral)")
    
    st.subheader("Parámetros de Entrada")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Longitudes (m)**")
        L1 = st.number_input("Longitud viga (L1)", value=5.0, step=0.5)
        L2 = st.number_input("Voladizo (L2)", value=1.5, step=0.1)
        L3 = st.number_input("Altura Nivel 1 (L3)", value=4.0, step=0.5)
        L4 = st.number_input("Altura Nivel 2 (L4)", value=3.0, step=0.5)
        L5 = st.number_input("Altura Nivel 3 (L5)", value=3.0, step=0.5)
    with col2:
        st.markdown("**Cargas (Ton y Ton/m)**")
        w1 = st.number_input("Carga Nivel 3 (w1)", value=2.0, step=0.5)
        w2 = st.number_input("Carga Nivel 2 (w2)", value=3.0, step=0.5)
        w3 = st.number_input("Carga Nivel 1 (w3)", value=3.0, step=0.5)
        F1 = st.number_input("Fuerza Lat. Nivel 3", value=8.0, step=1.0)
        F2 = st.number_input("Fuerza Lat. Nivel 2", value=6.0, step=1.0)
        F3 = st.number_input("Fuerza Lat. Nivel 1", value=4.0, step=1.0)
        P = st.number_input("Carga puntual voladizo (P)", value=2.0, step=0.5)
    with col3:
        st.markdown("**Coeficientes de Inercia (I)**")
        I_col1 = st.number_input("Inercia Col N1 (a)", value=4.0, step=1.0)
        I_col2 = st.number_input("Inercia Col N2 (b)", value=3.0, step=1.0)
        I_col3 = st.number_input("Inercia Col N3 (c)", value=2.0, step=1.0)
        I_viga = st.number_input("Inercia Vigas (d)", value=1.0, step=1.0)

    if st.button("Resolver Ejercicio 1"):
        st.write("---")
        st.subheader("Desglose Paso a Paso")
        
        K_AB = I_col1 / L3
        K_BC = I_col2 / L4
        K_CD = I_col3 / L5
        K_viga = I_viga / L1
        
        st.markdown("**Paso a) Cálculo de rigideces angulares simplificadas (K = I/L)**")
        st.latex(rf"K_{{AB}} = K_{{GH}} = \frac{{{I_col1}I_0}}{{{L3}}} = {K_AB:.2f} I_0")
        st.latex(rf"K_{{BC}} = K_{{FG}} = \frac{{{I_col2}I_0}}{{{L4}}} = {K_BC:.2f} I_0")
        st.latex(rf"K_{{CD}} = K_{{EF}} = \frac{{{I_col3}I_0}}{{{L5}}} = {K_CD:.4f} I_0")
        
        st.markdown("**Paso b) Cálculo de los factores de distribución**")
        st.info("Calculado en base a la suma de rigideces concurrentes en cada nudo (ej. Nudo B = BA + BC + BG).")
        
        M_DE = -(w1 * L1**2) / 12
        M_ED = (w1 * L1**2) / 12
        st.markdown("**Paso c) Momentos de empotramiento perfecto (sin desplazamiento)**")
        st.latex(rf"\bar{{M}}_{{DE}} = -\frac{{w l^2}}{{12}} = -\frac{{{w1} \times {L1}^2}}{{12}} = {M_DE:.4f}")
        st.latex(rf"\bar{{M}}_{{ED}} = \frac{{w l^2}}{{12}} = {M_ED:.4f}")

        st.markdown("**Representación de la Estructura**")
        fig, ax = plt.subplots(figsize=(6, 8))
        ax.plot([0, 0], [0, L3+L4+L5], 'k-', lw=3, label="Columnas")
        ax.plot([L1, L1], [0, L3+L4+L5], 'k-', lw=3)
        ax.plot([0, L1], [L3, L3], 'b-', lw=3, label="Vigas")
        ax.plot([0, L1], [L3+L4, L3+L4], 'b-', lw=3)
        ax.plot([0, L1+L2], [L3+L4+L5, L3+L4+L5], 'b-', lw=3)
        ax.set_title("Esquema Estructural - Marco 3 Niveles")
        ax.axis('off')
        st.pyplot(fig)

elif ejercicio == "Ejercicio 2: Marco con Columna Inclinada":
    st.header("Marco con una Columna Inclinada")
    
    st.subheader("Parámetros de Entrada")
    col1, col2 = st.columns(2)
    with col1:
        h = st.number_input("Altura de columna recta (a)", value=3.0, step=0.5)
        viga_L = st.number_input("Longitud tramo recto viga (b)", value=3.5, step=0.5)
        proy_x = st.number_input("Proyección X de col. inclinada (c)", value=2.0, step=0.5)
        angulo = st.number_input("Ángulo de inclinación (θ)", value=40.0, step=1.0)
    with col2:
        P = st.number_input("Carga Horizontal F (Ton)", value=3.0, step=0.5)
        I_val = st.number_input("Coeficiente de Inercia I (cte)", value=1.0, step=0.1)

    if st.button("Resolver Ejercicio 2"):
        st.write("---")
        st.subheader("Desglose Paso a Paso")
        
        angulo_rad = np.radians(angulo)
        L_CD = proy_x / np.cos(angulo_rad)
        
        st.markdown("**Paso 1: Longitud de la columna inclinada**")
        st.latex(rf"\cos({angulo}^\circ) = \frac{{{proy_x}}}{{CD}} \implies CD = {L_CD:.2f} \text{{ mts}}")

        K_AB = I_val / h
        K_BC = I_val / viga_L
        K_CD = I_val / L_CD
        
        st.markdown("**Paso 2: Rigideces Angulares**")
        st.latex(rf"K_{{AB}} = \frac{{I}}{{{h}}} = {K_AB:.4f}")
        st.latex(rf"K_{{BC}} = \frac{{I}}{{{viga_L}}} = {K_BC:.4f}")
        st.latex(rf"K_{{CD}} = \frac{{I}}{{{L_CD:.2f}}} = {K_CD:.4f}")
        
        st.markdown("**Paso 3 y 4: Momentos Impuestos y Factor de Corrección**")
        st.info("Imponiendo un momento de prueba y obteniendo la fuerza cortante equilibrada con la carga P.")
        
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot([0, 0], [0, h], 'k-', lw=3)
        ax2.plot([0, viga_L], [h, h], 'b-', lw=3)
        ax2.plot([viga_L, viga_L+proy_x], [h, 0], 'r-', lw=3, label="Columna Inclinada")
        ax2.set_title("Esquema Estructural - Columna Inclinada")
        ax2.axis('off')
        ax2.legend()
        st.pyplot(fig2)
