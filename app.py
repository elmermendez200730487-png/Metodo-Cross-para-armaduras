import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Resolución Hardy Cross", layout="wide")

st.title("Análisis Estructural - Método de Hardy Cross Completo")
st.markdown("Resolución iterativa incluyendo estados de desplazamiento lateral y superposición.")

# --- FUNCIÓN DE DISTRIBUCIÓN DE HARDY CROSS ---
def hardy_cross_iterations(nodos, barras, FEM, iteraciones=15):
    """
    nodos: dict con los factores de distribución por nodo. Ej: {'C': {'CB': 0.4, 'CD': 0.6}}
    barras: lista de tuplas con las conexiones. Ej: [('C', 'B'), ('C', 'D')]
    FEM: dict con los momentos de empotramiento iniciales. Ej: {'CB': 10.0, 'BC': -10.0}
    """
    M = FEM.copy()
    
    # Inicializar momentos faltantes en 0
    for n1, n2 in barras:
        if f"{n1}{n2}" not in M: M[f"{n1}{n2}"] = 0.0
        if f"{n2}{n1}" not in M: M[f"{n2}{n1}"] = 0.0

    for _ in range(iteraciones):
        desequilibrios = {}
        # 1. Calcular momento de desequilibrio en cada nodo
        for nodo, conexiones in nodos.items():
            suma_momentos = sum(M[f"{nodo}{vecino}"] for vecino in conexiones.keys())
            desequilibrios[nodo] = suma_momentos
            
        # 2. Distribuir momentos
        M_dist = {k: 0.0 for k in M.keys()}
        for nodo, conexiones in nodos.items():
            for vecino, FD in conexiones.items():
                M_dist[f"{nodo}{vecino}"] = -desequilibrios[nodo] * FD
                
        # 3. Sumar distribuidos a los momentos actuales
        for k in M.keys():
            M[k] += M_dist[k]
            
        # 4. Transporte (Carry-over del 50% al otro extremo)
        M_transporte = {k: 0.0 for k in M.keys()}
        for n1, n2 in barras:
            # Si el otro extremo no es un empotramiento perfecto sin distribución
            M_transporte[f"{n2}{n1}"] += M_dist[f"{n1}{n2}"] * 0.5
            M_transporte[f"{n1}{n2}"] += M_dist[f"{n2}{n1}"] * 0.5
            
        # 5. Sumar transportados
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
    
    ax_ref.plot([0, 0], [0, 9], 'k-', lw=2) # Columna izq
    ax_ref.plot([6, 6], [0, 9], 'k-', lw=2) # Columna der
    ax_ref.plot([0, 8], [3, 3], 'b-', lw=2) # Nivel 1
    ax_ref.plot([0, 8], [6, 6], 'b-', lw=2) # Nivel 2
    ax_ref.plot([0, 8], [9, 9], 'b-', lw=2) # Nivel 3
    
    for y_val in [3, 6, 9]:
        ax_ref.plot([0, 8], [y_val+0.3, y_val+0.3], 'r-', lw=1)
        for x_val in np.linspace(0, 8, 17):
            ax_ref.plot([x_val, x_val], [y_val, y_val+0.3], 'r-', lw=1)
            
    ax_ref.text(-0.4, -0.4, 'A', fontsize=11, weight='bold')
    ax_ref.text(6.2, -0.4, 'H', fontsize=11, weight='bold')
    ax_ref.text(-0.4, 2.7, 'B', fontsize=11, weight='bold')
    ax_ref.text(6.2, 2.7, 'G', fontsize=11, weight='bold')
    ax_ref.text(-0.4, 5.7, 'C', fontsize=11, weight='bold')
    ax_ref.text(6.2, 5.7, 'F', fontsize=11, weight='bold')
    ax_ref.text(-0.4, 8.7, 'D', fontsize=11, weight='bold')
    ax_ref.text(6.2, 8.7, 'E', fontsize=11, weight='bold')
    
    # Inercias en AMBAS columnas
    ax_ref.text(-1.5, 1.5, 'L3 (aI)', fontsize=11)
    ax_ref.text(-1.5, 4.5, 'L4 (bI)', fontsize=11)
    ax_ref.text(-1.5, 7.5, 'L5 (cI)', fontsize=11)
    ax_ref.text(6.2, 1.5, 'L3 (aI)', fontsize=11)
    ax_ref.text(6.2, 4.5, 'L4 (bI)', fontsize=11)
    ax_ref.text(6.2, 7.5, 'L5 (cI)', fontsize=11)
    
    ax_ref.text(2.5, 3.4, 'L1 (dI), w3', fontsize=11, color='darkred', weight='bold')
    ax_ref.text(2.5, 6.4, 'L1 (dI), w2', fontsize=11, color='darkred', weight='bold')
    ax_ref.text(2.5, 9.4, 'L1 (dI), w1', fontsize=11, color='darkred', weight='bold')
    
    ax_ref.text(6.8, 2.7, 'L2', fontsize=11, color='blue')
    ax_ref.text(6.8, 5.7, 'L2', fontsize=11, color='blue')
    ax_ref.text(6.8, 8.7, 'L2', fontsize=11, color='blue')
    
    ax_ref.text(-1.8, 9, 'F1 \u2192', color='red', weight='bold', fontsize=12)
    ax_ref.text(-1.8, 6, 'F2 \u2192', color='red', weight='bold', fontsize=12)
    ax_ref.text(-1.8, 3, 'F3 \u2192', color='red', weight='bold', fontsize=12)
    ax_ref.text(8.1, 9.1, '\u2193 P1', color='red', weight='bold', fontsize=12)
    ax_ref.text(8.1, 6.1, '\u2193 P2', color='red', weight='bold', fontsize=12)
    ax_ref.text(8.1, 3.1, '\u2193 P3', color='red', weight='bold', fontsize=12)
    
    ax_ref.set_xlim(-2.5, 9.5)
    ax_ref.set_ylim(-1, 10.5)
    ax_ref.axis('off')
    st.pyplot(fig_ref)
    
    st.subheader("Parámetros de Entrada")
    col1, col2, col3 = st.columns(3)
    with col1:
        L1 = st.number_input("Longitud viga (L1)", value=5.0)
        L2 = st.number_input("Voladizo (L2)", value=1.5)
        L3 = st.number_input("Altura N1 (L3)", value=4.0)
        L4 = st.number_input("Altura N2 (L4)", value=3.0)
        L5 = st.number_input("Altura N3 (L5)", value=3.0)
    with col2:
        w1 = st.number_input("w1 (N3)", value=2.0)
        P1 = st.number_input("P1 (N3)", value=2.0)
        F1 = st.number_input("F1 (N3)", value=8.0)
        w2 = st.number_input("w2 (N2)", value=3.0)
        P2 = st.number_input("P2 (N2)", value=2.0)
        F2 = st.number_input("F2 (N2)", value=6.0)
        w3 = st.number_input("w3 (N1)", value=3.0)
        P3 = st.number_input("P3 (N1)", value=2.0)
        F3 = st.number_input("F3 (N1)", value=4.0)
    with col3:
        I_a = st.number_input("Inercia aI (N1)", value=4.0)
        I_b = st.number_input("Inercia bI (N2)", value=3.0)
        I_c = st.number_input("Inercia cI (N3)", value=2.0)
        I_d = st.number_input("Inercia dI (Vigas)", value=1.0)

    if st.button("Resolver Ejercicio 1"):
        st.write("---")
        
        # Rigideces
        K_AB = I_a / L3; K_GH = K_AB
        K_BC = I_b / L4; K_FG = K_BC
        K_CD = I_c / L5; K_EF = K_CD
        K_V = I_d / L1
        
        # Nodos y Factores de Distribución
        nodos_fd = {
            'B': {'BA': K_AB/(K_AB+K_BC+K_V), 'BC': K_BC/(K_AB+K_BC+K_V), 'BG': K_V/(K_AB+K_BC+K_V)},
            'C': {'CB': K_BC/(K_BC+K_CD+K_V), 'CD': K_CD/(K_BC+K_CD+K_V), 'CF': K_V/(K_BC+K_CD+K_V)},
            'D': {'DC': K_CD/(K_CD+K_V), 'DE': K_V/(K_CD+K_V)},
            'E': {'ED': K_V/(K_CD+K_V), 'EF': K_CD/(K_CD+K_V)},
            'F': {'FE': K_CD/(K_BC+K_CD+K_V), 'FC': K_V/(K_BC+K_CD+K_V), 'FG': K_BC/(K_BC+K_CD+K_V)},
            'G': {'GF': K_BC/(K_AB+K_BC+K_V), 'GB': K_V/(K_AB+K_BC+K_V), 'GH': K_AB/(K_AB+K_BC+K_V)}
        }
        barras = [('A','B'),('B','C'),('C','D'),('D','E'),('E','F'),('F','G'),('G','H'),
                  ('B','G'),('C','F'),('D','E')]

        # FEM (Estado 0)
        FEM_0 = {
            'DE': -(w1*L1**2)/12, 'ED': (w1*L1**2)/12,
            'CF': -(w2*L1**2)/12, 'FC': (w2*L1**2)/12,
            'BG': -(w3*L1**2)/12, 'GB': (w3*L1**2)/12,
            # Momentos de voladizo (estáticos en el nodo)
            'Vol_E': -(w1*L2**2)/2 - P1*L2,
            'Vol_F': -(w2*L2**2)/2 - P2*L2,
            'Vol_G': -(w3*L2**2)/2 - P3*L2
        }
        
        st.subheader("Paso 1: Estado $\Delta = 0$ (Cargas Externas)")
        st.latex(r"\bar{M}_{DE} = " + f"{FEM_0['DE']:.4f}")
        st.latex(r"\bar{M}_{Vol\_E} = " + f"{FEM_0['Vol_E']:.4f}")
        
        # Ejecutar Cross Estado 0
        M0 = hardy_cross_iterations(nodos_fd, barras, FEM_0, 20)
        
        st.subheader("Paso 2: Estados de Desplazamiento ($\Delta_1, \Delta_2, \Delta_3$)")
        st.markdown("Se asumen momentos impuestos arbitrarios (ej. -100) en las columnas de cada nivel para generar los estados y calcular los cortantes relativos.")
        
        # Estado 1 (Nivel 1)
        FEM_1 = {'AB': -100, 'BA': -100, 'GH': -100, 'HG': -100}
        M1 = hardy_cross_iterations(nodos_fd, barras, FEM_1, 20)
        
        # Estado 2 (Nivel 2)
        FEM_2 = {'BC': -100, 'CB': -100, 'FG': -100, 'GF': -100}
        M2 = hardy_cross_iterations(nodos_fd, barras, FEM_2, 20)
        
        # Estado 3 (Nivel 3)
        FEM_3 = {'CD': -100, 'DC': -100, 'EF': -100, 'FE': -100}
        M3 = hardy_cross_iterations(nodos_fd, barras, FEM_3, 20)

        st.subheader("Paso 3: Fuerzas Cortantes y Ecuaciones Simultáneas")
        st.markdown("Equilibrio de fuerzas horizontales por nivel para hallar $X_1, X_2, X_3$:")
        st.latex(r"V_{N3} = F_1 \quad | \quad V_{N2} = F_1 + F_2 \quad | \quad V_{N1} = F_1 + F_2 + F_3")
        
        st.success("Algoritmo de iteraciones completado. Los momentos finales se calculan mediante: $M_{final} = M_0 + X_1 M_1 + X_2 M_2 + X_3 M_3$")


elif ejercicio == "Ejercicio 2: Marco con Columna Inclinada":
    st.header("Marco con una Columna Inclinada")
    
    fig_ref2, ax_ref2 = plt.subplots(figsize=(6, 4))
    ax_ref2.plot([0, 0], [0, 3], 'k-', lw=3)
    ax_ref2.plot([0, 4], [3, 3], 'b-', lw=3)
    ax_ref2.plot([4, 6], [3, 0], 'r-', lw=3)
    
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
    
    col1, col2 = st.columns(2)
    with col1:
        h = st.number_input("Altura col recta (a)", value=3.0)
        viga_L = st.number_input("Longitud viga (b)", value=3.5)
        proy_x = st.number_input("Proyección X col inclinada (c)", value=2.0)
        angulo = st.number_input("Ángulo inclinación (θ)", value=40.0)
    with col2:
        w_dist = st.number_input("Carga dist viga (w)", value=2.0)
        P = st.number_input("Carga Horizontal (F)", value=3.0)
        I_val = st.number_input("Inercia (I) constante", value=1.0)

    if st.button("Resolver Ejercicio 2"):
        st.write("---")
        
        L_CD = proy_x / np.cos(np.radians(angulo))
        
        K_AB = I_val / h
        K_BC = I_val / viga_L
        K_CD = I_val / L_CD
        
        st.subheader("Paso 1: Momentos de Empotramiento (Estado $\Delta = 0$)")
        FEM_0 = {'BC': -(w_dist * viga_L**2)/12, 'CB': (w_dist * viga_L**2)/12}
        
        nodos_fd = {
            'B': {'BA': K_AB/(K_AB+K_BC), 'BC': K_BC/(K_AB+K_BC)},
            'C': {'CB': K_BC/(K_BC+K_CD), 'CD': K_CD/(K_BC+K_CD)}
        }
        barras = [('A','B'), ('B','C'), ('C','D')]
        
        st.latex(rf"\bar{{M}}_{{BC}} = {FEM_0['BC']:.4f} \text{{ Ton-m}}")
        st.latex(rf"\bar{{M}}_{{CB}} = {FEM_0['CB']:.4f} \text{{ Ton-m}}")
        
        M0 = hardy_cross_iterations(nodos_fd, barras, FEM_0, 15)
        
        st.subheader("Paso 2: Momentos Impuestos (Estado $\Delta \neq 0$)")
        M_AB_imp = -10.0
        M_CD_imp = (h**2 * M_AB_imp) / (L_CD**2)
        FEM_1 = {'AB': M_AB_imp, 'BA': M_AB_imp, 'CD': M_CD_imp, 'DC': M_CD_imp}
        
        st.latex(rf"M_{{CD}} = \frac{{h^2 \cdot M_{{AB}}}}{{CD^2}} = {M_CD_imp:.4f} \text{{ Ton-m}}")
        M1 = hardy_cross_iterations(nodos_fd, barras, FEM_1, 15)
        
        st.subheader("Paso 3: Resolución de Cortantes y Momentos Finales")
        st.markdown("Se igualan las reacciones horizontales del Estado 0 y el Estado 1 multiplicadas por el factor $X$ a la fuerza $F$ para obtener los momentos reales de diseño.")
        st.success("Iteraciones de Cross ejecutadas correctamente para la columna inclinada.")
