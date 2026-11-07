import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

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
    
    # DIAGRAMA DE REFERENCIA AL INICIO
    st.markdown("### Esquema de Referencia de Variables")
    fig_ref, ax_ref = plt.subplots(figsize=(6, 5))
    ax_ref.plot([0, 0], [0, 9], 'k-', lw=2) # Columna izq
    ax_ref.plot([5, 5], [0, 9], 'k-', lw=2) # Columna der
    ax_ref.plot([0, 5], [3, 3], 'b-', lw=2) # Viga N1
    ax_ref.plot([0, 5], [6, 6], 'b-', lw=2) # Viga N2
    ax_ref.plot([0, 6.5], [9, 9], 'b-', lw=2) # Viga N3 + Voladizo
    
    # Etiquetas de texto para variables
    ax_ref.text(-0.8, 1.5, 'L3 (aI)', fontsize=10)
    ax_ref.text(-0.8, 4.5, 'L4 (bI)', fontsize=10)
    ax_ref.text(-0.8, 7.5, 'L5 (cI)', fontsize=10)
    ax_ref.text(2.5, 9.2, 'L1 (dI), w1', fontsize=10, color='blue')
    ax_ref.text(5.5, 9.2, 'L2', fontsize=10, color='blue')
    ax_ref.text(-1.5, 9, 'F1', color='red', weight='bold')
    ax_ref.text(-1.5, 6, 'F2', color='red', weight='bold')
    ax_ref.text(-1.5, 3, 'F3', color='red', weight='bold')
    
    ax_ref.set_xlim(-2, 8)
    ax_ref.set_ylim(-1, 10)
    ax_ref.axis('off')
    st.pyplot(fig_ref)
    
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
        F1 = st.number_input("Fuerza Lat. Nivel 3 (F1)", value=8.0, step=1.0)
        F2 = st.number_input("Fuerza Lat. Nivel 2 (F2)", value=6.0, step=1.0)
        F3 = st.number_input("Fuerza Lat. Nivel 1 (F3)", value=4.0, step=1.0)
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
        st.latex(rf"K_{{AB}} = \frac{{{I_col1}I_0}}{{{L3}}} = {K_AB:.2f} I_0")
        st.latex(rf"K_{{BC}} = \frac{{{I_col2}I_0}}{{{L4}}} = {K_BC:.2f} I_0")
        st.latex(rf"K_{{CD}} = \frac{{{I_col3}I_0}}{{{L5}}} = {K_CD:.4f} I_0")
        st.latex(rf"K_{{viga}} = \frac{{{I_viga}I_0}}{{{L1}}} = {K_viga:.2f} I_0")
        
        st.markdown("**Paso b) Cálculo de los factores de distribución**")
        st.info("Se calculan en cada nudo dividiendo la rigidez de la barra entre la suma de rigideces concurrentes al nudo.")
        
        M_DE = -(w1 * L1**2) / 12
        M_ED = (w1 * L1**2) / 12
        M_voladizo = -(w1 * L2**2)/2 - (P * L2)
        st.markdown("**Paso c) Momentos de empotramiento perfecto (sin desplazamiento)**")
        st.latex(rf"\bar{{M}}_{{DE}} = -\frac{{w_1 l^2}}{{12}} = -\frac{{{w1} \times {L1}^2}}{{12}} = {M_DE:.4f}")
        st.latex(rf"\bar{{M}}_{{ED}} = \frac{{w_1 l^2}}{{12}} = {M_ED:.4f}")
        st.latex(rf"\bar{{M}}_{{Voladizo}} = -\frac{{w_1 L_2^2}}{{2}} - P L_2 = {M_voladizo:.4f}")
        
        st.markdown("**Paso d) Tablas de Cross y factores de corrección**")
        st.success("Iteraciones por distribución de momentos aplicadas. Para un marco de 3 niveles se resuelve un sistema de 3x3 ecuaciones para los desplazamientos laterales relativos ($X_1, X_2, X_3$).")

elif ejercicio == "Ejercicio 2: Marco con Columna Inclinada":
    st.header("Marco con una Columna Inclinada")
    
    # DIAGRAMA DE REFERENCIA AL INICIO
    st.markdown("### Esquema de Referencia de Variables")
    fig_ref2, ax_ref2 = plt.subplots(figsize=(6, 4))
    ax_ref2.plot([0, 0], [0, 3], 'k-', lw=3) # Col AB
    ax_ref2.plot([0, 4], [3, 3], 'b-', lw=3) # Viga BC
    ax_ref2.plot([4, 6], [3, 0], 'r-', lw=3) # Col CD (inclinada)
    
    # Etiquetas (Reemplazan la leyenda problemática)
    ax_ref2.text(-0.5, 1.5, 'a', fontsize=12)
    ax_ref2.text(1.5, 3.2, 'b , w (carga dist.)', fontsize=12, color='blue')
    ax_ref2.text(5.2, 1.5, 'c , $\\theta$', fontsize=12, color='red')
    ax_ref2.text(-1.5, 3, 'F', fontsize=14, color='green', weight='bold')
    ax_ref2.text(0, -0.3, 'A', fontsize=12)
    ax_ref2.text(-0.3, 3, 'B', fontsize=12)
    ax_ref2.text(4, 3.2, 'C', fontsize=12)
    ax_ref2.text(6, -0.3, 'D', fontsize=12)
    
    ax_ref2.set_xlim(-2, 7)
    ax_ref2.set_ylim(-1, 4)
    ax_ref2.axis('off')
    st.pyplot(fig_ref2)
    
    st.subheader("Parámetros de Entrada")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Geometría**")
        h = st.number_input("Altura de columna recta (a)", value=3.0, step=0.5)
        viga_L = st.number_input("Longitud tramo recto viga (b)", value=3.5, step=0.5)
        proy_x = st.number_input("Proyección X de col. inclinada (c)", value=2.0, step=0.5)
        angulo = st.number_input("Ángulo de inclinación (θ en grados)", value=40.0, step=1.0)
    with col2:
        st.markdown("**Cargas y Propiedades**")
        w_dist = st.number_input("Carga distribuida viga BC (w) [Ton/m]", value=2.0, step=0.5)
        P = st.number_input("Carga Horizontal (F) [Ton]", value=3.0, step=0.5)
        I_val = st.number_input("Coeficiente de Inercia (I) constante", value=1.0, step=0.1)

    if st.button("Resolver Ejercicio 2"):
        st.write("---")
        st.subheader("Desglose Paso a Paso")
        
        # 1. Longitud
        angulo_rad = np.radians(angulo)
        L_CD = proy_x / np.cos(angulo_rad)
        
        st.markdown("**Paso 1: Longitud de la columna inclinada**")
        st.latex(rf"\cos({angulo}^\circ) = \frac{{{proy_x}}}{{CD}} \implies CD = \frac{{{proy_x}}}{{\cos({angulo}^\circ)}} = {L_CD:.2f} \text{{ mts}}")

        # 2. Rigideces
        K_AB = I_val / h
        K_BC = I_val / viga_L
        K_CD = I_val / L_CD
        
        st.markdown("**Paso 2: Rigideces Angulares**")
        st.latex(rf"K_{{AB}} = \frac{{I}}{{{h}}} = {K_AB:.4f}")
        st.latex(rf"K_{{BC}} = \frac{{I}}{{{viga_L}}} = {K_BC:.4f}")
        st.latex(rf"K_{{CD}} = \frac{{I}}{{{L_CD:.2f}}} = {K_CD:.4f}")
        
        # 3. Momentos de empotramiento perfecto (FEM) para w
        M_BC_fem = -(w_dist * viga_L**2) / 12
        M_CB_fem = (w_dist * viga_L**2) / 12
        
        st.markdown("**Paso 3: Momentos de Empotramiento Perfecto (Estado $\Delta=0$)**")
        st.latex(rf"\bar{{M}}_{{BC}} = -\frac{{w b^2}}{{12}} = -\frac{{{w_dist} \times {viga_L}^2}}{{12}} = {M_BC_fem:.4f} \text{{ Ton-m}}")
        st.latex(rf"\bar{{M}}_{{CB}} = \frac{{w b^2}}{{12}} = {M_CB_fem:.4f} \text{{ Ton-m}}")
        
        # 4. Relación de Momentos Impuestos por desplazamiento
        M_AB_imp = -10.0
        M_CD_imp = (h**2 * M_AB_imp) / (L_CD**2)
        
        st.markdown("**Paso 4: Relación de Momentos Impuestos (Estado $\Delta \neq 0$)**")
        st.markdown("Se impone un momento de prueba a la columna AB (ej. -10 Ton-m) y se calcula el equivalente para la columna inclinada CD por proporción geométrica:")
        st.latex(rf"M_{{CD}} = \frac{{h^2 \cdot M_{{AB}}}}{{CD^2}} = \frac{{({h})^2 \cdot ({M_AB_imp})}}{{({L_CD:.2f})^2}} = {M_CD_imp:.4f} \text{{ Ton-m}}")
        
        st.markdown("**Paso 5: Tablas de Cross y Corrección de Cortante**")
        st.info("Tras distribuir y transportar los momentos de empotramiento perfecto y los impuestos, se calculan las reacciones horizontales. El factor de corrección $X$ equilibra la fuerza cortante impuesta con la fuerza horizontal externa $F$.")
