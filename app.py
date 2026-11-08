import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Resolución Hardy Cross Completo", layout="wide")

st.title("Análisis Estructural - Método de Hardy Cross Completo")
st.markdown("Resolución iterativa incluyendo estados de desplazamiento lateral, superposición y diagramas de elementos mecánicos.")

# --- CONFIGURACIÓN DE UNIDADES ---
st.sidebar.header("Configuración de Unidades")
u_fuerza = st.sidebar.selectbox("Unidad de Fuerza", ["ton", "N", "lb"])
u_long = st.sidebar.selectbox("Unidad de Longitud", ["m", "ft", "in"])
u_ang = st.sidebar.selectbox("Unidad de Ángulo", ["Grados", "Radianes"])

# --- FUNCIÓN DE DISTRIBUCIÓN DE HARDY CROSS CORREGIDA ---
def hardy_cross_iterations(nodos, barras, FEM, iteraciones=15):
    M = FEM.copy()
    for n1, n2 in barras:
        if f"{n1}{n2}" not in M: M[f"{n1}{n2}"] = 0.0
        if f"{n2}{n1}" not in M: M[f"{n2}{n1}"] = 0.0
    for nodo, conexiones in nodos.items():
        for vecino in conexiones.keys():
            if f"{nodo}{vecino}" not in M: 
                M[f"{nodo}{vecino}"] = 0.0

    for _ in range(iteraciones):
        desequilibrios = {}
        for nodo, conexiones in nodos.items():
            desequilibrios[nodo] = sum(M[f"{nodo}{vecino}"] for vecino in conexiones.keys())
            
        M_dist = {k: 0.0 for k in M.keys()}
        for nodo, conexiones in nodos.items():
            for vecino, FD in conexiones.items():
                M_dist[f"{nodo}{vecino}"] = -desequilibrios[nodo] * FD
                
        for k in M.keys():
            M[k] += M_dist[k]
            
        M_transporte = {k: 0.0 for k in M.keys()}
        for n1, n2 in barras:
            M_transporte[f"{n2}{n1}"] += M_dist[f"{n1}{n2}"] * 0.5
            M_transporte[f"{n1}{n2}"] += M_dist[f"{n2}{n1}"] * 0.5
            
        for k in M.keys():
            M[k] += M_transporte[k]
            
    return M

# --- SELECTOR DE EJERCICIO ---
ejercicio = st.sidebar.radio(
    "Selecciona el tipo de marco a resolver:", 
    ("Ejercicio 1: Marco de 3 Niveles", "Ejercicio 2: Marco con Columna Inclinada")
)

if ejercicio == "Ejercicio 1: Marco de 3 Niveles":
    st.header("Marco con Tres Niveles y Una Crujía")
    
    st.markdown("### Esquema de Referencia de Variables")
    fig_ref, ax_ref = plt.subplots(figsize=(8, 6))
    
    # Dibujar la estructura base
    ax_ref.plot([0, 0], [0, 9], 'k-', lw=2) 
    ax_ref.plot([6, 6], [0, 9], 'k-', lw=2) 
    ax_ref.plot([0, 8], [3, 3], 'b-', lw=2) 
    ax_ref.plot([0, 8], [6, 6], 'b-', lw=2) 
    ax_ref.plot([0, 8], [9, 9], 'b-', lw=2) 
    
    # Dibujar empotramientos (Soportes fijos base)
    ax_ref.plot([-0.4, 0.4], [0, 0], 'k-', lw=3)
    for i in np.linspace(-0.4, 0.3, 6): ax_ref.plot([i, i+0.1], [0, -0.3], 'k-', lw=1)
    
    ax_ref.plot([5.6, 6.4], [0, 0], 'k-', lw=3)
    for i in np.linspace(5.6, 6.3, 6): ax_ref.plot([i, i+0.1], [0, -0.3], 'k-', lw=1)
    
    # Simular cargas distribuidas (w)
    for y_val in [3, 6, 9]:
        ax_ref.plot([0, 8], [y_val+0.3, y_val+0.3], 'g-', lw=1)
        for x_val in np.linspace(0, 8, 17):
            ax_ref.plot([x_val, x_val], [y_val, y_val+0.3], 'g-', lw=1)
            
    # Etiquetas de nodos
    ax_ref.text(-0.5, 0.2, 'A', fontsize=11, weight='bold')
    ax_ref.text(6.3, 0.2, 'H', fontsize=11, weight='bold')
    ax_ref.text(-0.4, 2.7, 'B', fontsize=11, weight='bold')
    ax_ref.text(6.2, 2.7, 'G', fontsize=11, weight='bold')
    ax_ref.text(-0.4, 5.7, 'C', fontsize=11, weight='bold')
    ax_ref.text(6.2, 5.7, 'F', fontsize=11, weight='bold')
    ax_ref.text(-0.4, 8.7, 'D', fontsize=11, weight='bold')
    ax_ref.text(6.2, 8.7, 'E', fontsize=11, weight='bold')

    # INERCIAS EN LOS TRAMOS
    ax_ref.text(0.2, 1.5, 'Ia', color='black', fontsize=10, weight='bold')
    ax_ref.text(6.2, 1.5, 'Ia', color='black', fontsize=10, weight='bold')
    ax_ref.text(0.2, 4.5, 'Ib', color='black', fontsize=10, weight='bold')
    ax_ref.text(6.2, 4.5, 'Ib', color='black', fontsize=10, weight='bold')
    ax_ref.text(0.2, 7.5, 'Ic', color='black', fontsize=10, weight='bold')
    ax_ref.text(6.2, 7.5, 'Ic', color='black', fontsize=10, weight='bold')
    ax_ref.text(3, 2.7, 'Id (Viga)', color='blue', fontsize=10, weight='bold', ha='center')
    ax_ref.text(3, 5.7, 'Id (Viga)', color='blue', fontsize=10, weight='bold', ha='center')
    ax_ref.text(3, 8.7, 'Id (Viga)', color='blue', fontsize=10, weight='bold', ha='center')

    # Añadir flechas y etiquetas de Fuerzas Laterales (F) - de izquierda a derecha
    ax_ref.annotate('F1 (N3)', xy=(0, 9), xytext=(-2.5, 9), arrowprops=dict(facecolor='red', shrink=0, width=2, headwidth=8), fontsize=10, color='red', va='center')
    ax_ref.annotate('F2 (N2)', xy=(0, 6), xytext=(-2.5, 6), arrowprops=dict(facecolor='red', shrink=0, width=2, headwidth=8), fontsize=10, color='red', va='center')
    ax_ref.annotate('F3 (N1)', xy=(0, 3), xytext=(-2.5, 3), arrowprops=dict(facecolor='red', shrink=0, width=2, headwidth=8), fontsize=10, color='red', va='center')
    
    # Etiquetas de Cargas Distribuidas (w)
    ax_ref.text(3, 9.5, 'w1', color='green', ha='center', fontsize=10, weight='bold')
    ax_ref.text(3, 6.5, 'w2', color='green', ha='center', fontsize=10, weight='bold')
    ax_ref.text(3, 3.5, 'w3', color='green', ha='center', fontsize=10, weight='bold')
    
    # Añadir flechas de Cargas Puntuales en Voladizo (P)
    ax_ref.annotate('P1', xy=(8, 9), xytext=(8, 10.5), arrowprops=dict(facecolor='purple', shrink=0, width=2, headwidth=8), fontsize=10, color='purple', ha='center')
    ax_ref.annotate('P2', xy=(8, 6), xytext=(8, 7.5), arrowprops=dict(facecolor='purple', shrink=0, width=2, headwidth=8), fontsize=10, color='purple', ha='center')
    ax_ref.annotate('P3', xy=(8, 3), xytext=(8, 4.5), arrowprops=dict(facecolor='purple', shrink=0, width=2, headwidth=8), fontsize=10, color='purple', ha='center')
    
    ax_ref.set_xlim(-3, 10)
    ax_ref.set_ylim(-1, 11)
    ax_ref.axis('off')
    st.pyplot(fig_ref)
    plt.close(fig_ref)
    
    st.subheader("Parámetros de Entrada")
    col1, col2, col3 = st.columns(3)
    with col1:
        L1 = st.number_input(f"Longitud viga L1 ({u_long})", value=5.0)
        L2 = st.number_input(f"Voladizo L2 ({u_long})", value=1.5)
        L3 = st.number_input(f"Altura N1 L3 ({u_long})", value=4.0)
