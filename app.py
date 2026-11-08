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
u_ang = st.sidebar.selectbox("Unidad de Ángulo (Ej. 2)", ["Grados", "Radianes"])

# --- FUNCIÓN DE DISTRIBUCIÓN DE HARDY CROSS CORREGIDA ---
def hardy_cross_iterations(nodos, barras, FEM, iteraciones=15):
    M = FEM.copy()
    
    # 1. Inicializar en 0 los momentos de las barras principales faltantes
    for n1, n2 in barras:
        if f"{n1}{n2}" not in M: M[f"{n1}{n2}"] = 0.0
        if f"{n2}{n1}" not in M: M[f"{n2}{n1}"] = 0.0
        
    # 2. Inicializar en 0 cualquier otra conexión
    for nodo, conexiones in nodos.items():
        for vecino in conexiones.keys():
            if f"{nodo}{vecino}" not in M: 
                M[f"{nodo}{vecino}"] = 0.0

    for _ in range(iteraciones):
        desequilibrios = {}
        # 3. Calcular desequilibrio
        for nodo, conexiones in nodos.items():
            desequilibrios[nodo] = sum(M[f"{nodo}{vecino}"] for vecino in conexiones.keys())
            
        M_dist = {k: 0.0 for k in M.keys()}
        
        # 4. Distribuir
        for nodo, conexiones in nodos.items():
            for vecino, FD in conexiones.items():
                M_dist[f"{nodo}{vecino}"] = -desequilibrios[nodo] * FD
                
        # 5. Aplicar
        for k in M.keys():
            M[k] += M_dist[k]
            
        M_transporte = {k: 0.0 for k in M.keys()}
        
        # 6. Transporte
        for n1, n2 in barras:
            M_transporte[f"{n2}{n1}"] += M_dist[f"{n1}{n2}"] * 0.5
            M_transporte[f"{n1}{n2}"] += M_dist[f"{n2}{n1}"] * 0.5
            
        # 7. Sumar transportados
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
    
    # Simular cargas distribuidas (w)
    for y_val in [3, 6, 9]:
        ax_ref.plot([0, 8], [y_val+0.3, y_val+0.3], 'g-', lw=1)
        for x_val in np.linspace(0, 8, 17):
            ax_ref.plot([x_val, x_val], [y_val, y_val+0.3], 'g-', lw=1)
            
    # Etiquetas de nodos
    ax_ref.text(-0.4, -0.4, 'A', fontsize=11, weight='bold')
    ax_ref.text(6.2, -0.4, 'H', fontsize=11, weight='bold')
    ax_ref.text(-0.4, 2.7, 'B', fontsize=11, weight='bold')
    ax_ref.text(6.2, 2.7, 'G', fontsize=11, weight='bold')
    ax_ref.text(-0.4, 5.7, 'C', fontsize=11, weight='bold')
    ax_ref.text(6.2, 5.7, 'F', fontsize=11, weight='bold')
    ax_ref.text(-0.4, 8.7, 'D', fontsize=11, weight='bold')
    ax_ref.text(6.2, 8.7, 'E', fontsize=11, weight='bold')

    # Añadir flechas y etiquetas de Fuerzas Laterales (F)
    ax_ref.annotate('F1 (N3)', xy=(0, 9), xytext=(-2.5, 9), arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8), fontsize=10, color='red', va='center')
    ax_ref.annotate('F2 (N2)', xy=(0, 6), xytext=(-2.5, 6), arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8), fontsize=10, color='red', va='center')
    ax_ref.annotate('F3 (N1)', xy=(0, 3), xytext=(-2.5, 3), arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8), fontsize=10, color='red', va='center')
    
    # Etiquetas de Cargas Distribuidas (w)
    ax_ref.text(3, 9.5, 'w1', color='green', ha='center', fontsize=10, weight='bold')
    ax_ref.text(3, 6.5, 'w2', color='green', ha='center', fontsize=10, weight='bold')
    ax_ref.text(3, 3.5, 'w3', color='green', ha='center', fontsize=10, weight='bold')
    
    # Añadir flechas de Cargas Puntuales en Voladizo (P)
    ax_ref.annotate('P1', xy=(8, 9), xytext=(8, 10.5), arrowprops=dict(facecolor='purple', shrink=0.05, width=2, headwidth=8), fontsize=10, color='purple', ha='center')
    ax_ref.annotate('P2', xy=(8, 6), xytext=(8, 7.5), arrowprops=dict(facecolor='purple', shrink=0.05, width=2, headwidth=8), fontsize=10, color='purple', ha='center')
    ax_ref.annotate('P3', xy=(8, 3), xytext=(8, 4.5), arrowprops=dict(facecolor='purple', shrink=0.05, width=2, headwidth=8), fontsize=10, color='purple', ha='center')
    
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
        L4 = st.number_input(f"Altura N2 L4 ({u_long})", value=3.0)
        L5 = st.number_input(f"Altura N3 L5 ({u_long})", value=3.0)
    with col2:
        w1 = st.number_input(f"w1 - Distribuida N3 ({u_fuerza}/{u_long})", value=2.0)
        P1 = st.number_input(f"P1 - Puntual Vol. N3 ({u_fuerza})", value=2.0)
        F1 = st.number_input(f"F1 - Lateral N3 ({u_fuerza})", value=8.0)
        w2 = st.number_input(f"w2 - Distribuida N2 ({u_fuerza}/{u_long})", value=3.0)
        P2 = st.number_input(f"P2 - Puntual Vol. N2 ({u_fuerza})", value=2.0)
        F2 = st.number_input(f"F2 - Lateral N2 ({u_fuerza})", value=6.0)
        w3 = st.number_input(f"w3 - Distribuida N1 ({u_fuerza}/{u_long})", value=3.0)
        P3 = st.number_input(f"P3 - Puntual Vol. N1 ({u_fuerza})", value=2.0)
        F3 = st.number_input(f"F3 - Lateral N1 ({u_fuerza})", value=4.0)
    with col3:
        I_a = st.number_input(f"Inercia aI ({u_long}^4)", value=4.0)
        I_b = st.number_input(f"Inercia bI ({u_long}^4)", value=3.0)
        I_c = st.number_input(f"Inercia cI ({u_long}^4)", value=2.0)
        I_d = st.number_input(f"Inercia dI ({u_long}^4)", value=1.0)

    if st.button("Resolver Ejercicio 1"):
        st.write("---")
        
        # Rigideces Angulares
        K_AB = I_a / L3; K_GH = K_AB
        K_BC = I_b / L4; K_FG = K_BC
        K_CD = I_c / L5; K_EF = K_CD
        K_V = I_d / L1
        
        st.subheader("Paso 1: Rigideces Angulares ($K = I/L$)")
        st.latex(rf"K_{{AB}} = K_{{GH}} = {K_AB:.4f}")
        st.latex(rf"K_{{BC}} = K_{{FG}} = {K_BC:.4f}")
        st.latex(rf"K_{{CD}} = K_{{EF}} = {K_CD:.4f}")
        st.latex(rf"K_{{Vigas}} = {K_V:.4f}")
        
        # Sumatorias en los nodos
        S_B = K_AB + K_BC + K_V
        S_C = K_BC + K_CD + K_V
        S_D = K_CD + K_V
        S_E = K_CD + K_V
        S_F = K_BC + K_CD + K_V
        S_G = K_AB + K_BC + K_V
        
        nodos_fd = {
            'B': {'A': K_AB/S_B, 'C': K_BC/S_B, 'G': K_V/S_B},
            'C': {'B': K_BC/S_C, 'D': K_CD/S_C, 'F': K_V/S_C},
            'D': {'C': K_CD/S_D, 'E': K_V/S_D},
            'E': {'D': K_V/S_E, 'F': K_CD/S_E, 'Vol': 0.0},
            'F': {'C': K_V/S_F, 'E': K_CD/S_F, 'G': K_BC/S_F, 'Vol': 0.0},
            'G': {'B': K_V/S_G, 'F': K_BC/S_G, 'H': K_AB/S_G, 'Vol': 0.0}
        }
        
        st.subheader("Paso 2: Factores de Distribución ($FD$)")
        st.markdown("**Nodo B:**")
        st.latex(rf"FD_{{BA}}={nodos_fd['B']['A']:.4f}, \; FD_{{BC}}={nodos_fd['B']['C']:.4f}, \; FD_{{BG}}={nodos_fd['B']['G']:.4f}")
        st.markdown("**Nodo C:**")
        st.latex(rf"FD_{{CB}}={nodos_fd['C']['B']:.4f}, \; FD_{{CD}}={nodos_fd['C']['D']:.4f}, \; FD_{{CF}}={nodos_fd['C']['F']:.4f}")
        st.markdown("**Nodo D:**")
        st.latex(rf"FD_{{DC}}={nodos_fd['D']['C']:.4f}, \; FD_{{DE}}={nodos_fd['D']['E']:.4f}")

        barras = [('A','B'), ('B','C'), ('C','D'), ('H','G'), ('G','F'), ('F','E'), ('B','G'), ('C','F'), ('D','E')]

        FEM_0 = {
            'DE': -(w1*L1**2)/12, 'ED': (w1*L1**2)/12,
            'CF': -(w2*L1**2)/12, 'FC': (w2*L1**2)/12,
            'BG': -(w3*L1**2)/12, 'GB': (w3*L1**2)/12,
            'EVol': -(w1*L2**2)/2 - P1*L2,
            'FVol': -(w2*L2**2)/2 - P2*L2,
            'GVol': -(w3*L2**2)/2 - P3*L2
        }
        
        st.subheader("Paso 3: Momentos de Empotramiento (Estado $\Delta = 0$)")
        st.latex(rf"\bar{{M}}_{{DE}} = {FEM_0['DE']:.4f} \quad \bar{{M}}_{{Vol\_E}} = {FEM_0['EVol']:.4f}")
        
        M0 = hardy_cross_iterations(nodos_fd, barras, FEM_0, 20)
        
        st.subheader("Paso 4: Estados de Desplazamiento ($\Delta_1, \Delta_2, \Delta_3$)")
        FEM_1 = {'AB': -100, 'BA': -100, 'GH': -100, 'HG': -100}
        M1 = hardy_cross_iterations(nodos_fd, barras, FEM_1, 20)
        
        FEM_2 = {'BC': -100, 'CB': -100, 'FG': -100, 'GF': -100}
        M2 = hardy_cross_iterations(nodos_fd, barras, FEM_2, 20)
        
        FEM_3 = {'CD': -100, 'DC': -100, 'EF': -100, 'FE': -100}
        M3 = hardy_cross_iterations(nodos_fd, barras, FEM_3, 20)
        
        df_M = pd.DataFrame([M0, M1, M2, M3], index=['Estado 0', 'Estado 1 (\u03941)', 'Estado 2 (\u03942)', 'Estado 3 (\u03943)']).T
        st.dataframe(df_M.style.format("{:.4f}"))

        st.subheader("Paso 5: Superposición y Factores de Corrección")
        def calc_cortantes(M_dict):
            V1 = -(M_dict.get('AB',0) + M_dict.get('BA',0) + M_dict.get('GH',0) + M_dict.get('HG',0)) / L3
            V2 = -(M_dict.get('BC',0) + M_dict.get('CB',0) + M_dict.get('FG',0) + M_dict.get('GF',0)) / L4
            V3 = -(M_dict.get('CD',0) + M_dict.get('DC',0) + M_dict.get('EF',0) + M_dict.get('FE',0)) / L5
            return np.array([V1, V2, V3])

        V0 = calc_cortantes(M0); V1 = calc_cortantes(M1)
        V2 = calc_cortantes(M2); V3 = calc_cortantes(M3)
        
        H_ext = np.array([F1 + F2 + F3, F1 + F2, F1])
        Matriz_V = np.column_stack((V1, V2, V3))
        Vector_F = H_ext - V0
        
        try:
            X = np.linalg.solve(Matriz_V, Vector_F)
            st.info(f"**Factores:** $X_1 = {X[0]:.4f}$, $X_2 = {X[1]:.4f}$, $X_3 = {X[2]:.4f}$")
            
            M_final = {k: M0[k] + X[0]*M1[k] + X[1]*M2[k] + X[2]*M3[k] for k in M0.keys()}
            df_final = pd.DataFrame([M_final], index=['M Final']).T
            st.dataframe(df_final.style.format("{:.4f}"))
            
            # --- DIAGRAMAS EX 1 ---
            st.subheader("Paso 6: Diagramas Estructurales Aproximados")
            fig, ax = plt.subplots(1, 2, figsize=(14, 6))
            
            for i in range(2):
                ax[i].plot([0, 0], [0, L3+L4+L5], 'k-', lw=2)
                ax[i].plot([L1, L1], [0, L3+L4+L5], 'k-', lw=2)
                ax[i].plot([0, L1], [L3, L3], 'k-', lw=2)
                ax[i].plot([0, L1], [L3+L4, L3+L4], 'k-', lw=2)
                ax[i].plot([0, L1], [L3+L4+L5, L3+L4+L5], 'k-', lw=2)
            
            ax[0].set_title(f"Momentos Finales en Extremos ({u_fuerza}-{u_long})")
            ax[0].text(-0.5, 0.5, f"{M_final.get('AB',0):.1f}", color='red')
            ax[0].text(-0.5, L3-0.5, f"{M_final.get('BA',0):.1f}", color='red')
            ax[0].text(1, L3+0.2, f"{M_final.get('BG',0):.1f}", color='blue')
            ax[0].text(L1-1.5, L3+0.2, f"{M_final.get('GB',0):.1f}", color='blue')
            ax[0].text(-0.5, L3+L4+L5-0.5, f"{M_final.get('DC',0):.1f}", color='red')
            
            ax[1].set_title(f"Cortantes por Piso ({u_fuerza})")
            ax[1].text(-1.5, L3/2, f"V1={-(M_final.get('AB',0)+M_final.get('BA',0))/L3:.1f}", color='green')
            ax[1].text(-1.5, L3+L4/2, f"V2={-(M_final.get('BC',0)+M_final.get('CB',0))/L4:.1f}", color='green')
            ax[1].text(-1.5, L3+L4+L5/2, f"V3={-(M_final.get('CD',0)+M_final.get('DC',0))/L5:.1f}", color='green')
            
            st.pyplot(fig)
            plt.close(fig)

        except np.linalg.LinAlgError:
            st.error("Error en superposición.")


elif ejercicio == "Ejercicio 2: Marco con Columna Inclinada":
    st.header("Marco con una Columna Inclinada")
    
    st.markdown("### Esquema de Referencia de Variables")
    fig_ref2, ax_ref2 = plt.subplots(figsize=(7, 5))
    ax_ref2.plot([0, 0], [0, 3], 'k-', lw=3)
    ax_ref2.plot([0, 4], [3, 3], 'b-', lw=3)
    ax_ref2.plot([4, 6], [3, 0], 'r-', lw=3)
    
    # Etiquetas nodos
    ax_ref2.text(0, -0.3, 'A', fontsize=12, weight='bold'); ax_ref2.text(-0.3, 3, 'B', fontsize=12, weight='bold')
    ax_ref2.text(4, 3.2, 'C', fontsize=12, weight='bold'); ax_ref2.text(6, -0.3, 'D', fontsize=12, weight='bold')
    
    # Dibujar Carga Distribuida (w)
    ax_ref2.plot([0, 4], [3.15, 3.15], 'g-', lw=1)
    for x_val in np.linspace(0, 4, 15):
        ax_ref2.plot([x_val, x_val], [3, 3.15], 'g-', lw=1)
    ax_ref2.text(2, 3.4, 'w_dist (Carga Distribuida)', color='green', ha='center', fontsize=10, weight='bold')
    
    # Dibujar Carga Lateral (Fuerza P)
    ax_ref2.annotate('P (Fuerza Horizontal)', xy=(0, 3), xytext=(-2.5, 3), arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8), fontsize=10, color='red', va='center')
    
    ax_ref2.set_xlim(-3, 7)
    ax_ref2.set_ylim(-1, 4.5)
    ax_ref2.axis('off')
    st.pyplot(fig_ref2)
    plt.close(fig_ref2)
    
    col1, col2 = st.columns(2)
    with col1:
        h = st.number_input(f"Altura col recta a ({u_long})", value=3.0)
        viga_L = st.number_input(f"Longitud viga b ({u_long})", value=3.5)
        proy_x = st.number_input(f"Proyección X col inclinada c ({u_long})", value=2.0)
        ang_simbolo = "°" if u_ang == "Grados" else "rad"
        angulo = st.number_input(f"Ángulo inclinación θ ({ang_simbolo})", value=40.0)
    with col2:
        w_dist = st.number_input(f"Carga dist viga w_dist ({u_fuerza}/{u_long})", value=2.0)
        P = st.number_input(f"Carga Horizontal P ({u_fuerza})", value=3.0)
        I_val = st.number_input(f"Inercia I constante ({u_long}^4)", value=1.0)

    if st.button("Resolver Ejercicio 2"):
        st.write("---")
        
        st.subheader("Paso 1: Longitud de Columna CD")
        if u_ang == "Grados":
            L_CD = proy_x / np.cos(np.radians(angulo))
            st.latex(rf"L_{{CD}} = \frac{{{proy_x}}}{{\cos({angulo}^\circ)}} = {L_CD:.4f} \text{{ {u_long}}}")
        else:
            L_CD = proy_x / np.cos(angulo)
            st.latex(rf"L_{{CD}} = \frac{{{proy_x}}}{{\cos({angulo} \text{{ rad}})}} = {L_CD:.4f} \text{{ {u_long}}}")
        
        st.subheader("Paso 2: Rigideces Angulares ($K$)")
        K_AB = I_val / h
        K_BC = I_val / viga_L
        K_CD = I_val / L_CD
        st.latex(rf"K_{{AB}} = {K_AB:.4f} \quad K_{{BC}} = {K_BC:.4f} \quad K_{{CD}} = {K_CD:.4f}")
        
        st.subheader("Paso 3: Factores de Distribución ($FD$)")
        FD_BA = K_AB/(K_AB+K_BC); FD_BC = K_BC/(K_AB+K_BC)
        FD_CB = K_BC/(K_BC+K_CD); FD_CD = K_CD/(K_BC+K_CD)
        st.latex(rf"FD_{{BA}} = {FD_BA:.4f} \quad FD_{{BC}} = {FD_BC:.4f}")
        st.latex(rf"FD_{{CB}} = {FD_CB:.4f} \quad FD_{{CD}} = {FD_CD:.4f}")
        
        st.subheader("Paso 4: Momentos de Empotramiento (Estado $\Delta = 0$)")
        FEM_0 = {'BC': -(w_dist * viga_L**2)/12, 'CB': (w_dist * viga_L**2)/12}
        
        nodos_fd = {
            'B': {'A': FD_BA, 'C': FD_BC},
            'C': {'B': FD_CB, 'D': FD_CD}
        }
        barras = [('A','B'), ('B','C'), ('C','D')]
        
        M0 = hardy_cross_iterations(nodos_fd, barras, FEM_0, 15)
        
        st.subheader("Paso 5: Momentos Impuestos (Estado $\Delta \neq 0$)")
        M_AB_imp = -10.0
        M_CD_imp = (h**2 * M_AB_imp) / (L_CD**2)
        FEM_1 = {'AB': M_AB_imp, 'BA': M_AB_imp, 'CD': M_CD_imp, 'DC': M_CD_imp}
        
        M1 = hardy_cross_iterations(nodos_fd, barras, FEM_1, 15)
        
        df_M = pd.DataFrame([M0, M1], index=['Estado 0', 'Estado 1 (\u0394)']).T
        st.dataframe(df_M.style.format("{:.4f}"))

        st.subheader("Paso 6: Equilibrio de Cortante y Momentos Finales")
        V0 = -(M0.get('AB',0) + M0.get('BA',0))/h - (M0.get('CD',0) + M0.get('DC',0))/h
        V1 = -(M1.get('AB',0) + M1.get('BA',0))/h - (M1.get('CD',0) + M1.get('DC',0))/h
        
        if V1 != 0:
            X = (P - V0) / V1
            st.latex(rf"X = {X:.5f}")
            M_final = {k: M0[k] + X*M1[k] for k in M0.keys()}
            df_final = pd.DataFrame([M_final], index=['Momento Final']).T
            st.dataframe(df_final.style.format("{:.4f}"))
            
            # --- DIAGRAMAS EX 2 ---
            st.subheader("Paso 7: Diagramas Estructurales")
            fig2, ax2 = plt.subplots(1, 2, figsize=(14, 6))
            
            for i in range(2):
                ax2[i].plot([0, 0], [0, h], 'k-', lw=2)
                ax2[i].plot([0, viga_L], [h, h], 'k-', lw=2)
                ax2[i].plot([viga_L, viga_L+proy_x], [h, 0], 'k-', lw=2)
                
            ax2[0].set_title(f"Momentos Finales en Extremos ({u_fuerza}-{u_long})")
            ax2[0].text(-0.8, 0.5, f"{M_final.get('AB',0):.2f}", color='red')
            ax2[0].text(-0.8, h-0.5, f"{M_final.get('BA',0):.2f}", color='red')
            ax2[0].text(0.5, h+0.2, f"{M_final.get('BC',0):.2f}", color='blue')
            ax2[0].text(viga_L-1.5, h+0.2, f"{M_final.get('CB',0):.2f}", color='blue')
            ax2[0].text(viga_L+0.5, h-0.5, f"{M_final.get('CD',0):.2f}", color='purple')
            
            ax2[1].set_title(f"Cortantes de Columna ({u_fuerza})")
            V_AB = -(M_final.get('AB',0) + M_final.get('BA',0)) / h
            V_CD = -(M_final.get('CD',0) + M_final.get('DC',0)) / L_CD
            ax2[1].text(-1.5, h/2, f"V_AB={V_AB:.2f}", color='green')
            ax2[1].text(viga_L+proy_x/2 + 0.5, h/2, f"V_CD={V_CD:.2f}", color='green')

            st.pyplot(fig2)
            plt.close(fig2)
