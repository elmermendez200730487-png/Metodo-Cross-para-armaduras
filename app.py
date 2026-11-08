import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Resolución Hardy Cross Completo", layout="wide")

st.title("Análisis Estructural - Método de Hardy Cross Completo")
st.markdown("Resolución iterativa incluyendo tablas paso a paso, estados de desplazamiento lateral y diagramas de elementos mecánicos.")

# --- CONFIGURACIÓN DE UNIDADES ---
st.sidebar.header("Configuración de Unidades")
u_fuerza = st.sidebar.selectbox("Unidad de Fuerza", ["ton", "N", "lb"])
u_long = st.sidebar.selectbox("Unidad de Longitud", ["m", "ft", "in"])
u_ang = st.sidebar.selectbox("Unidad de Ángulo", ["Grados", "Radianes"])

# --- FUNCIÓN GENERADORA DE TABLAS ESTILO HARDY CROSS (Como en la imagen) ---
def tabla_cross_paso_a_paso(nodos_fd, barras, FEM, iteraciones, orden_miembros, nombre_m_inicial="M. IMPUESTO"):
    columnas_tuplas = [(m[0], m) for m in orden_miembros]
    columnas = pd.MultiIndex.from_tuples(columnas_tuplas, names=['NODO', 'MIEMBRO'])
    
    # 1. F.D.
    fd_row = []
    for m in orden_miembros:
        n1, n2 = m[0], m[1:]
        if len(m) > 2 and 'Vol' in m:
            fd_row.append(0.0)
        else:
            fd_row.append(nodos_fd.get(n1, {}).get(n2, 0.0))
            
    # 2. M. IMPUESTO
    m_row = [FEM.get(m, 0.0) for m in orden_miembros]
    
    filas = [fd_row, m_row]
    nombres_filas = ['F.D.', nombre_m_inicial]
    
    M_actual = {m: FEM.get(m, 0.0) for m in orden_miembros}
    
    def num_ordinal(n):
        if n == 1: return "1era."
        elif n == 2: return "2da."
        elif n == 3: return "3era."
        else: return f"{n}ta."
        
    def num_ordinal_transp(n):
        if n == 1: return "1er."
        elif n == 2: return "2do."
        elif n == 3: return "3er."
        else: return f"{n}to."
    
    for i in range(1, iteraciones + 1):
        # Distribución
        desequilibrios = {n: 0.0 for n in nodos_fd.keys()}
        for m in orden_miembros:
            n1 = m[0]
            if n1 in desequilibrios:
                desequilibrios[n1] += M_actual[m]
                
        dist_row_dict = {}
        for m in orden_miembros:
            n1, n2 = m[0], m[1:]
            if n1 in nodos_fd and n2 in nodos_fd[n1]:
                dist = -desequilibrios[n1] * nodos_fd[n1][n2]
                dist_row_dict[m] = dist
                M_actual[m] += dist
            else:
                dist_row_dict[m] = 0.0
                
        filas.append([dist_row_dict[m] for m in orden_miembros])
        nombres_filas.append(f"{num_ordinal(i)} Dist.")
        
        # Transporte
        trans_row_dict = {m: 0.0 for m in orden_miembros}
        for n1, n2 in barras:
            m_ida, m_vuelta = f"{n1}{n2}", f"{n2}{n1}"
            if m_ida in orden_miembros and m_vuelta in orden_miembros:
                t_ida = dist_row_dict.get(m_ida, 0.0) * 0.5
                t_vuelta = dist_row_dict.get(m_vuelta, 0.0) * 0.5
                trans_row_dict[m_ida] += t_vuelta
                trans_row_dict[m_vuelta] += t_ida
                M_actual[m_ida] += t_vuelta
                M_actual[m_vuelta] += t_ida
                
        filas.append([trans_row_dict[m] for m in orden_miembros])
        nombres_filas.append(f"{num_ordinal_transp(i)} Transp.")
        
    # Momentos Finales
    filas.append([M_actual[m] for m in orden_miembros])
    nombres_filas.append('Momentos Finales')
    
    df = pd.DataFrame(filas, index=nombres_filas, columns=columnas)
    return df, M_actual

# --- SELECTOR DE EJERCICIO ---
ejercicio = st.sidebar.radio(
    "Selecciona el tipo de marco a resolver:", 
    ("Ejercicio 1: Marco de 3 Niveles", "Ejercicio 2: Marco con Columna Inclinada")
)

if ejercicio == "Ejercicio 1: Marco de 3 Niveles":
    st.header("Marco con Tres Niveles y Una Crujía")
    
    st.markdown("### Esquema de Referencia de Variables")
    fig_ref, ax_ref = plt.subplots(figsize=(8, 6))
    
    ax_ref.plot([0, 0], [0, 9], 'k-', lw=2) 
    ax_ref.plot([6, 6], [0, 9], 'k-', lw=2) 
    ax_ref.plot([0, 8], [3, 3], 'b-', lw=2) 
    ax_ref.plot([0, 8], [6, 6], 'b-', lw=2) 
    ax_ref.plot([0, 8], [9, 9], 'b-', lw=2) 
    
    ax_ref.plot([-0.4, 0.4], [0, 0], 'k-', lw=3)
    for i in np.linspace(-0.4, 0.3, 6): ax_ref.plot([i, i+0.1], [0, -0.3], 'k-', lw=1)
    ax_ref.plot([5.6, 6.4], [0, 0], 'k-', lw=3)
    for i in np.linspace(5.6, 6.3, 6): ax_ref.plot([i, i+0.1], [0, -0.3], 'k-', lw=1)
    
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

    # Cotas de longitudes
    ax_ref.annotate('', xy=(-1.5, 0), xytext=(-1.5, 3), arrowprops=dict(arrowstyle='<->'))
    ax_ref.text(-2, 1.5, 'L3 (N1)', va='center')
    ax_ref.annotate('', xy=(-1.5, 3), xytext=(-1.5, 6), arrowprops=dict(arrowstyle='<->'))
    ax_ref.text(-2, 4.5, 'L4 (N2)', va='center')
    ax_ref.annotate('', xy=(-1.5, 6), xytext=(-1.5, 9), arrowprops=dict(arrowstyle='<->'))
    ax_ref.text(-2, 7.5, 'L5 (N3)', va='center')
    
    ax_ref.annotate('', xy=(0, 10.5), xytext=(6, 10.5), arrowprops=dict(arrowstyle='<->'))
    ax_ref.text(3, 10.7, 'L1 (Viga)', ha='center')
    ax_ref.annotate('', xy=(6, 10.5), xytext=(8, 10.5), arrowprops=dict(arrowstyle='<->'))
    ax_ref.text(7, 10.7, 'L2 (Vol.)', ha='center')
    
    ax_ref.set_xlim(-3, 10)
    ax_ref.set_ylim(-1, 11.5)
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
        I_a = st.number_input(f"Inercia Ia ({u_long}^4)", value=4.0)
        I_b = st.number_input(f"Inercia Ib ({u_long}^4)", value=3.0)
        I_c = st.number_input(f"Inercia Ic ({u_long}^4)", value=2.0)
        I_d = st.number_input(f"Inercia Id ({u_long}^4)", value=1.0)

    if st.button("Resolver Ejercicio 1"):
        st.write("---")
        
        K_AB = I_a / L3; K_GH = K_AB
        K_BC = I_b / L4; K_FG = K_BC
        K_CD = I_c / L5; K_EF = K_CD
        K_V = I_d / L1
        
        S_B = K_AB + K_BC + K_V; S_C = K_BC + K_CD + K_V; S_D = K_CD + K_V
        S_E = K_CD + K_V; S_F = K_BC + K_CD + K_V; S_G = K_AB + K_BC + K_V
        
        nodos_fd = {
            'A': {'B': 0}, 'H': {'G': 0},
            'B': {'A': K_AB/S_B, 'C': K_BC/S_B, 'G': K_V/S_B},
            'C': {'B': K_BC/S_C, 'D': K_CD/S_C, 'F': K_V/S_C},
            'D': {'C': K_CD/S_D, 'E': K_V/S_D},
            'E': {'D': K_V/S_E, 'F': K_CD/S_E, 'Vol': 0.0},
            'F': {'C': K_V/S_F, 'E': K_CD/S_F, 'G': K_BC/S_F, 'Vol': 0.0},
            'G': {'B': K_V/S_G, 'F': K_BC/S_G, 'H': K_AB/S_G, 'Vol': 0.0}
        }
        
        barras = [('A','B'), ('B','C'), ('C','D'), ('D','E'), ('C','F'), ('B','G'), ('E','F'), ('F','G'), ('G','H')]
        orden_miembros_ej1 = ['AB', 'BA', 'BC', 'CB', 'CD', 'DC', 'DE', 'ED', 'EVol', 'EF', 'FE', 'FVol', 'FC', 'CF', 'FG', 'GF', 'GVol', 'GB', 'BG', 'GH', 'HG']
        
        FEM_0 = {
            'DE': -(w1*L1**2)/12, 'ED': (w1*L1**2)/12,
            'CF': -(w2*L1**2)/12, 'FC': (w2*L1**2)/12,
            'BG': -(w3*L1**2)/12, 'GB': (w3*L1**2)/12,
            'EVol': -(w1*L2**2)/2 - P1*L2,
            'FVol': -(w2*L2**2)/2 - P2*L2,
            'GVol': -(w3*L2**2)/2 - P3*L2
        }
        
        st.subheader("Tablas de Iteración de Hardy Cross")
        
        df_estado0, M0 = tabla_cross_paso_a_paso(nodos_fd, barras, FEM_0, 15, orden_miembros_ej1, "M. INICIAL")
        with st.expander("Ver Tabla Iterativa - Estado Inicial (Δ = 0)", expanded=True):
            st.dataframe(df_estado0.style.format("{:.4f}"))
            
        FEM_1 = {'AB': -100, 'BA': -100, 'GH': -100, 'HG': -100}
        df_estado1, M1 = tabla_cross_paso_a_paso(nodos_fd, barras, FEM_1, 15, orden_miembros_ej1, "M. IMPUESTO (Δ1)")
        with st.expander("Ver Tabla Iterativa - Estado Δ1"):
            st.dataframe(df_estado1.style.format("{:.4f}"))
            
        FEM_2 = {'BC': -100, 'CB': -100, 'FG': -100, 'GF': -100}
        df_estado2, M2 = tabla_cross_paso_a_paso(nodos_fd, barras, FEM_2, 15, orden_miembros_ej1, "M. IMPUESTO (Δ2)")
        with st.expander("Ver Tabla Iterativa - Estado Δ2"):
            st.dataframe(df_estado2.style.format("{:.4f}"))
            
        FEM_3 = {'CD': -100, 'DC': -100, 'EF': -100, 'FE': -100}
        df_estado3, M3 = tabla_cross_paso_a_paso(nodos_fd, barras, FEM_3, 15, orden_miembros_ej1, "M. IMPUESTO (Δ3)")
        with st.expander("Ver Tabla Iterativa - Estado Δ3"):
            st.dataframe(df_estado3.style.format("{:.4f}"))
            
        st.subheader("Superposición y Momentos Finales")
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
            M_final = {k: M0[k] + X[0]*M1[k] + X[1]*M2[k] + X[2]*M3[k] for k in orden_miembros_ej1}
            
            # Formateando los Momentos Finales ordenados
            df_final = pd.DataFrame([M_final], index=['Momento Final'])
            columnas_tuplas = [(m[0], m) for m in orden_miembros_ej1]
            df_final.columns = pd.MultiIndex.from_tuples(columnas_tuplas, names=['NODO', 'MIEMBRO'])
            st.dataframe(df_final.style.format("{:.4f}"))

        except np.linalg.LinAlgError:
            st.error("Error en superposición.")

elif ejercicio == "Ejercicio 2: Marco con Columna Inclinada":
    st.header("Marco con una Columna Inclinada")
    
    st.markdown("### Esquema de Referencia de Variables")
    fig_ref2, ax_ref2 = plt.subplots(figsize=(7, 5))
    ax_ref2.plot([0, 0], [0, 3], 'k-', lw=3)
    ax_ref2.plot([0, 4], [3, 3], 'b-', lw=3)
    ax_ref2.plot([4, 6], [3, 0], 'r-', lw=3)
    
    ax_ref2.plot([-0.3, 0.3], [0, 0], 'k-', lw=3)
    for i in np.linspace(-0.3, 0.2, 5): ax_ref2.plot([i, i+0.1], [0, -0.2], 'k-', lw=1)
    ax_ref2.plot([5.7, 6.3], [0, 0], 'k-', lw=3)
    for i in np.linspace(5.7, 6.2, 5): ax_ref2.plot([i, i+0.1], [0, -0.2], 'k-', lw=1)
    
    ax_ref2.text(0.4, 0.2, 'A', fontsize=12, weight='bold'); ax_ref2.text(-0.3, 3, 'B', fontsize=12, weight='bold')
    ax_ref2.text(4, 3.2, 'C', fontsize=12, weight='bold'); ax_ref2.text(6.3, 0.2, 'D', fontsize=12, weight='bold')
    
    # Cotas de longitudes
    ax_ref2.annotate('', xy=(-1, 0), xytext=(-1, 3), arrowprops=dict(arrowstyle='<->'))
    ax_ref2.text(-1.5, 1.5, 'Altura "a"', va='center')
    
    ax_ref2.annotate('', xy=(0, 3.8), xytext=(4, 3.8), arrowprops=dict(arrowstyle='<->'))
    ax_ref2.text(2, 4, 'Longitud "b"', ha='center')
    
    ax_ref2.annotate('', xy=(4, -0.8), xytext=(6, -0.8), arrowprops=dict(arrowstyle='<->'))
    ax_ref2.text(5, -1.2, 'Proy. "c"', ha='center')
    
    # Dibujar Carga Distribuida (w)
    ax_ref2.plot([0, 4], [3.15, 3.15], 'g-', lw=1)
    for x_val in np.linspace(0, 4, 15): ax_ref2.plot([x_val, x_val], [3, 3.15], 'g-', lw=1)
    
    ax_ref2.set_xlim(-3, 7)
    ax_ref2.set_ylim(-1.5, 4.5)
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
        if u_ang == "Grados": L_CD = proy_x / np.cos(np.radians(angulo))
        else: L_CD = proy_x / np.cos(angulo)
        
        K_AB = I_val / h; K_BC = I_val / viga_L; K_CD = I_val / L_CD
        
        FD_BA = K_AB/(K_AB+K_BC); FD_BC = K_BC/(K_AB+K_BC)
        FD_CB = K_BC/(K_BC+K_CD); FD_CD = K_CD/(K_BC+K_CD)
        
        nodos_fd = {
            'A': {'B': 0.0}, 'D': {'C': 0.0},
            'B': {'A': FD_BA, 'C': FD_BC},
            'C': {'B': FD_CB, 'D': FD_CD}
        }
        barras = [('A','B'), ('B','C'), ('C','D')]
        orden_miembros_ej2 = ['AB', 'BA', 'BC', 'CB', 'CD', 'DC']
        
        FEM_0 = {'BC': -(w_dist * viga_L**2)/12, 'CB': (w_dist * viga_L**2)/12}
        
        st.subheader("Tablas de Iteración de Hardy Cross (Formato Detallado)")
        
        df_estado0, M0 = tabla_cross_paso_a_paso(nodos_fd, barras, FEM_0, 4, orden_miembros_ej2, "M. INICIAL")
        st.markdown("**Tabla Estado 0 (Cargas Externas sin desplazamiento)**")
        st.dataframe(df_estado0.style.format("{:.4f}"))
        
        M_AB_imp = -10.0
        M_CD_imp = (h**2 * M_AB_imp) / (L_CD**2)
        FEM_1 = {'AB': M_AB_imp, 'BA': M_AB_imp, 'CD': M_CD_imp, 'DC': M_CD_imp}
        
        df_estado1, M1 = tabla_cross_paso_a_paso(nodos_fd, barras, FEM_1, 4, orden_miembros_ej2, "M. IMPUESTO")
        st.markdown("**Tabla Estado 1 (Imposición de Momentos por Desplazamiento)**")
        st.dataframe(df_estado1.style.format("{:.4f}"))
        
        st.subheader("Equilibrio de Cortante y Momentos Finales")
        V0 = -(M0.get('AB',0) + M0.get('BA',0))/h - (M0.get('CD',0) + M0.get('DC',0))/h
        V1 = -(M1.get('AB',0) + M1.get('BA',0))/h - (M1.get('CD',0) + M1.get('DC',0))/h
        
        if V1 != 0:
            X = (P - V0) / V1
            st.latex(rf"X = {X:.5f}")
            M_final = {k: M0[k] + X*M1[k] for k in orden_miembros_ej2}
            
            # Formato de la tabla final
            df_final = pd.DataFrame([M_final], index=['Momentos Finales'])
            columnas_tuplas = [(m[0], m) for m in orden_miembros_ej2]
            df_final.columns = pd.MultiIndex.from_tuples(columnas_tuplas, names=['NODO', 'MIEMBRO'])
            st.dataframe(df_final.style.format("{:.4f}"))
