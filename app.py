import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

st.set_page_config(page_title="Resolución Hardy Cross Completo", layout="wide")

st.title("Análisis Estructural - Método de Hardy Cross Completo")
st.markdown("Resolución iterativa incluyendo esquemas, tablas paso a paso y diagramas de Cortante y Momento Flexionante.")

# --- CONFIGURACIÓN DE UNIDADES ---
st.sidebar.header("Configuración de Unidades")
u_fuerza = st.sidebar.selectbox("Unidad de Fuerza", ["ton", "N", "lb"])
u_long = st.sidebar.selectbox("Unidad de Longitud", ["m", "ft", "in"])
u_ang = st.sidebar.selectbox("Unidad de Ángulo", ["Grados", "Radianes"])

# ==========================================
# FUNCIONES DE DIBUJO DE DIAGRAMAS
# ==========================================
def draw_support(ax, x, y, nx=0, ny=-1):
    if nx == 0:
        ax.plot([x-0.4, x+0.4], [y, y], 'k-', lw=2)
        for i in np.linspace(-0.4, 0.3, 6):
            ax.plot([x+i, x+i+0.1], [y, y-0.2], 'k-', lw=1)
    else:
        ax.plot([x-0.4*ny, x+0.4*ny], [y+0.4*nx, y-0.4*nx], 'k-', lw=2)
        for i in np.linspace(-0.4, 0.3, 6):
            xi, yi = x + i*ny, y - i*nx
            ax.plot([xi, xi+0.1*nx], [yi, yi+0.1*ny], 'k-', lw=1)

def draw_diagram_member(ax, x1, y1, x2, y2, v1, v2, scale, color='blue', label1="", label2="", draw_member=True):
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L == 0: return
    nx, ny = -dy/L, dx/L
    
    px1, py1 = x1 + v1 * scale * nx, y1 + v1 * scale * ny
    px2, py2 = x2 + v2 * scale * nx, y2 + v2 * scale * ny
    
    if draw_member:
        ax.plot([x1, x2], [y1, y2], 'k-', lw=2.5)
        
    ax.plot([x1, px1, px2, x2], [y1, py1, py2, y2], color=color, lw=1.2)
    
    num_hatches = max(5, int(L * 8))
    for i in range(1, num_hatches):
        t = i / num_hatches
        bx, by = x1 + t*dx, y1 + t*dy
        vt = v1 + t*(v2 - v1)
        tx, ty = bx + vt * scale * nx, by + vt * scale * ny
        ax.plot([bx, tx], [by, ty], color=color, lw=0.6, alpha=0.6)
        
    if abs(v1) > 0.001: ax.text(px1 + 0.3*nx, py1 + 0.3*ny, label1, color='black', fontsize=9, weight='bold', ha='center', va='center')
    if abs(v2) > 0.001: ax.text(px2 + 0.3*nx, py2 + 0.3*ny, label2, color='black', fontsize=9, weight='bold', ha='center', va='center')

# ==========================================
# FUNCIÓN GENERADORA DE TABLAS
# ==========================================
def tabla_cross_paso_a_paso(nodos_fd, barras, FEM, iteraciones, orden_miembros, nombre_m_inicial="M. IMPUESTO"):
    columnas_tuplas = [(m[0], m) for m in orden_miembros]
    columnas = pd.MultiIndex.from_tuples(columnas_tuplas, names=['NODO', 'MIEMBRO'])
    
    fd_row = []
    for m in orden_miembros:
        n1, n2 = m[0], m[1:]
        if len(m) > 2 and 'Vol' in m: fd_row.append(0.0)
        else: fd_row.append(nodos_fd.get(n1, {}).get(n2, 0.0))
            
    m_row = [FEM.get(m, 0.0) for m in orden_miembros]
    filas = [fd_row, m_row]
    nombres_filas = ['F.D.', nombre_m_inicial]
    
    M_actual = {m: FEM.get(m, 0.0) for m in orden_miembros}
    
    for i in range(1, iteraciones + 1):
        desequilibrios = {n: 0.0 for n in nodos_fd.keys()}
        for m in orden_miembros:
            if m[0] in desequilibrios: desequilibrios[m[0]] += M_actual[m]
                
        dist_row_dict = {}
        for m in orden_miembros:
            n1, n2 = m[0], m[1:]
            if n1 in nodos_fd and n2 in nodos_fd[n1]:
                dist = -desequilibrios[n1] * nodos_fd[n1][n2]
                dist_row_dict[m] = dist
                M_actual[m] += dist
            else: dist_row_dict[m] = 0.0
                
        filas.append([dist_row_dict[m] for m in orden_miembros])
        nombres_filas.append(f"{i}a. Dist.")
        
        trans_row_dict = {m: 0.0 for m in orden_miembros}
        for n1, n2 in barras:
            m_ida, m_vuelta = f"{n1}{n2}", f"{n2}{n1}"
            if m_ida in orden_miembros and m_vuelta in orden_miembros:
                trans_row_dict[m_ida] += dist_row_dict.get(m_vuelta, 0.0) * 0.5
                trans_row_dict[m_vuelta] += dist_row_dict.get(m_ida, 0.0) * 0.5
                M_actual[m_ida] += trans_row_dict[m_ida]
                M_actual[m_vuelta] += trans_row_dict[m_vuelta]
                
        filas.append([trans_row_dict[m] for m in orden_miembros])
        nombres_filas.append(f"{i}o. Transp.")
        
    filas.append([M_actual[m] for m in orden_miembros])
    nombres_filas.append('Momentos Finales')
    
    return pd.DataFrame(filas, index=nombres_filas, columns=columnas), M_actual

# ==========================================
# MENÚ DE SELECCIÓN
# ==========================================
ejercicio = st.sidebar.radio("Selecciona el tipo de marco:", ("Ejercicio 1: Marco de 3 Niveles", "Ejercicio 2: Marco con Columna Inclinada"))

if ejercicio == "Ejercicio 1: Marco de 3 Niveles":
    st.header("Marco con Tres Niveles y Una Crujía")
    
    # ESQUEMA INICIAL EJERCICIO 1
    st.markdown("### Esquema de Referencia con Variables")
    fig_ref1, ax_ref1 = plt.subplots(figsize=(6, 8))
    # Marco principal
    ax_ref1.plot([0, 0], [0, 10], 'k-', lw=3); ax_ref1.plot([4, 4], [0, 10], 'k-', lw=3)
    ax_ref1.plot([0, 4], [4, 4], 'k-', lw=3); ax_ref1.plot([0, 4], [7, 7], 'k-', lw=3); ax_ref1.plot([0, 4], [10, 10], 'k-', lw=3)
    # Voladizos
    ax_ref1.plot([4, 5.5], [4, 4], 'k-', lw=3); ax_ref1.plot([4, 5.5], [7, 7], 'k-', lw=3); ax_ref1.plot([4, 5.5], [10, 10], 'k-', lw=3)
    # Nodos
    ax_ref1.text(-0.5, 0, 'A', fontsize=12, weight='bold'); ax_ref1.text(4.2, 0, 'H', fontsize=12, weight='bold')
    ax_ref1.text(-0.5, 4, 'B', fontsize=12, weight='bold'); ax_ref1.text(-0.5, 7, 'C', fontsize=12, weight='bold'); ax_ref1.text(-0.5, 10, 'D', fontsize=12, weight='bold')
    ax_ref1.text(3.5, 4, 'G', fontsize=12, weight='bold'); ax_ref1.text(3.5, 7, 'F', fontsize=12, weight='bold'); ax_ref1.text(3.5, 10, 'E', fontsize=12, weight='bold')
    # Fuerzas laterales
    ax_ref1.annotate('F1', xy=(0, 10), xytext=(-2, 10), arrowprops=dict(facecolor='red', width=1, headwidth=5), color='red', va='center')
    ax_ref1.annotate('F2', xy=(0, 7), xytext=(-2, 7), arrowprops=dict(facecolor='red', width=1, headwidth=5), color='red', va='center')
    ax_ref1.annotate('F3', xy=(0, 4), xytext=(-2, 4), arrowprops=dict(facecolor='red', width=1, headwidth=5), color='red', va='center')
    # Cotas
    ax_ref1.text(-1, 2, 'L3', color='blue'); ax_ref1.text(-1, 5.5, 'L4', color='blue'); ax_ref1.text(-1, 8.5, 'L5', color='blue')
    ax_ref1.text(2, 10.5, 'w1, L1', color='green'); ax_ref1.text(4.5, 10.5, 'P1, L2', color='green')
    ax_ref1.set_xlim(-3, 7); ax_ref1.set_ylim(-1, 12); ax_ref1.axis('off')
    st.pyplot(fig_ref1)
    plt.close(fig_ref1)

    st.subheader("Parámetros de Entrada")
    col1, col2, col3 = st.columns(3)
    with col1:
        L1 = st.number_input(f"L1 (Viga) ({u_long})", value=5.0)
        L2 = st.number_input(f"L2 (Voladizo) ({u_long})", value=1.5)
        L3 = st.number_input(f"L3 (Altura N1) ({u_long})", value=4.0)
        L4 = st.number_input(f"L4 (Altura N2) ({u_long})", value=3.0)
        L5 = st.number_input(f"L5 (Altura N3) ({u_long})", value=3.0)
    with col2:
        w1 = st.number_input(f"w1 (Dist. N3) ({u_fuerza}/{u_long})", value=2.0)
        P1 = st.number_input(f"P1 (Puntual N3) ({u_fuerza})", value=2.0)
        F1 = st.number_input(f"F1 (Lateral N3) ({u_fuerza})", value=8.0)
        w2 = st.number_input(f"w2 (Dist. N2) ({u_fuerza}/{u_long})", value=3.0)
        P2 = st.number_input(f"P2 (Puntual N2) ({u_fuerza})", value=2.0)
        F2 = st.number_input(f"F2 (Lateral N2) ({u_fuerza})", value=6.0)
        w3 = st.number_input(f"w3 (Dist. N1) ({u_fuerza}/{u_long})", value=3.0)
        P3 = st.number_input(f"P3 (Puntual N1) ({u_fuerza})", value=2.0)
        F3 = st.number_input(f"F3 (Lateral N1) ({u_fuerza})", value=4.0)
    with col3:
        I_a = st.number_input(f"Ia (Col. N1) ({u_long}^4)", value=4.0)
        I_b = st.number_input(f"Ib (Col. N2) ({u_long}^4)", value=3.0)
        I_c = st.number_input(f"Ic (Col. N3) ({u_long}^4)", value=2.0)
        I_d = st.number_input(f"Id (Vigas) ({u_long}^4)", value=1.0)

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
        orden_miembros = ['AB', 'BA', 'BC', 'CB', 'CD', 'DC', 'DE', 'ED', 'EVol', 'EF', 'FE', 'FVol', 'FC', 'CF', 'FG', 'GF', 'GVol', 'GB', 'BG', 'GH', 'HG']
        
        FEM_0 = {
            'DE': -(w1*L1**2)/12, 'ED': (w1*L1**2)/12, 'CF': -(w2*L1**2)/12, 'FC': (w2*L1**2)/12,
            'BG': -(w3*L1**2)/12, 'GB': (w3*L1**2)/12, 'EVol': -(w1*L2**2)/2 - P1*L2,
            'FVol': -(w2*L2**2)/2 - P2*L2, 'GVol': -(w3*L2**2)/2 - P3*L2
        }
        
        st.subheader("Tablas de Iteración")
        df_estado0, M0 = tabla_cross_paso_a_paso(nodos_fd, barras, FEM_0, 15, orden_miembros, "M. INICIAL")
        st.markdown("**Estado 0 (Cargas Reales)**")
        st.dataframe(df_estado0.style.format("{:.3f}"))
        
        FEM_1 = {'AB': -100, 'BA': -100, 'GH': -100, 'HG': -100}
        df_estado1, M1 = tabla_cross_paso_a_paso(nodos_fd, barras, FEM_1, 15, orden_miembros, "M. IMPUESTO (\u03941)")
        st.markdown("**Estado 1 (Desplazamiento Nivel 1)**")
        st.dataframe(df_estado1.style.format("{:.3f}"))
        
        FEM_2 = {'BC': -100, 'CB': -100, 'FG': -100, 'GF': -100}
        df_estado2, M2 = tabla_cross_paso_a_paso(nodos_fd, barras, FEM_2, 15, orden_miembros, "M. IMPUESTO (\u03942)")
        st.markdown("**Estado 2 (Desplazamiento Nivel 2)**")
        st.dataframe(df_estado2.style.format("{:.3f}"))
        
        FEM_3 = {'CD': -100, 'DC': -100, 'EF': -100, 'FE': -100}
        df_estado3, M3 = tabla_cross_paso_a_paso(nodos_fd, barras, FEM_3, 15, orden_miembros, "M. IMPUESTO (\u03943)")
        st.markdown("**Estado 3 (Desplazamiento Nivel 3)**")
        st.dataframe(df_estado3.style.format("{:.3f}"))
        
        def calc_cortantes(M_dict):
            V1 = -(M_dict.get('AB',0) + M_dict.get('BA',0) + M_dict.get('GH',0) + M_dict.get('HG',0)) / L3
            V2 = -(M_dict.get('BC',0) + M_dict.get('CB',0) + M_dict.get('FG',0) + M_dict.get('GF',0)) / L4
            V3 = -(M_dict.get('CD',0) + M_dict.get('DC',0) + M_dict.get('EF',0) + M_dict.get('FE',0)) / L5
            return np.array([V1, V2, V3])

        V0, V1, V2, V3 = calc_cortantes(M0), calc_cortantes(M1), calc_cortantes(M2), calc_cortantes(M3)
        Matriz_V = np.column_stack((V1, V2, V3))
        Vector_F = np.array([F1 + F2 + F3, F1 + F2, F1]) - V0
        
        X = np.linalg.solve(Matriz_V, Vector_F)
        M_final = {k: M0[k] + X[0]*M1[k] + X[1]*M2[k] + X[2]*M3[k] for k in orden_miembros}
        
        st.subheader("Momentos Finales")
        df_final = pd.DataFrame([M_final], index=['Mom. Final'])
        df_final.columns = pd.MultiIndex.from_tuples([(m[0], m) for m in orden_miembros], names=['NODO', 'MIEMBRO'])
        st.dataframe(df_final.style.format("{:.4f}"))
        
        st.subheader("Diagramas de Cortante y Momento")
        fig, (ax_v, ax_m) = plt.subplots(1, 2, figsize=(16, 8))
        coords = {'A':(0,0), 'B':(0,L3), 'C':(0,L3+L4), 'D':(0,L3+L4+L5), 'H':(L1,0), 'G':(L1,L3), 'F':(L1,L3+L4), 'E':(L1,L3+L4+L5)}
        
        max_m = max([abs(v) for v in M_final.values()]) or 1.0
        scale_m = (L1 * 0.15) / max_m
        
        ax_m.set_title("Diagrama de Momento Flexionante")
        draw_support(ax_m, coords['A'][0], coords['A'][1]); draw_support(ax_m, coords['H'][0], coords['H'][1])
        tramos_m = [('A','B'), ('B','C'), ('C','D'), ('H','G'), ('G','F'), ('F','E'), ('B','G'), ('C','F'), ('D','E')]
        for n1, n2 in tramos_m:
            v1, v2 = M_final.get(n1+n2, 0), M_final.get(n2+n1, 0)
            draw_diagram_member(ax_m, coords[n1][0], coords[n1][1], coords[n2][0], coords[n2][1], v1, -v2, scale_m, color='purple', label1=f"{v1:.2f}", label2=f"{v2:.2f}")

        ax_v.set_title("Diagrama de Fuerza Cortante")
        draw_support(ax_v, coords['A'][0], coords['A'][1]); draw_support(ax_v, coords['H'][0], coords['H'][1])
        V_vals = {}
        for n1, n2, L, w in [('A','B',L3,0), ('B','C',L4,0), ('C','D',L5,0), ('H','G',L3,0), ('G','F',L4,0), ('F','E',L5,0), ('B','G',L1,w3), ('C','F',L1,w2), ('D','E',L1,w1)]:
            v_izq = w*L/2.0 - (M_final.get(n1+n2,0) + M_final.get(n2+n1,0))/L
            v_der = -w*L/2.0 - (M_final.get(n1+n2,0) + M_final.get(n2+n1,0))/L
            V_vals[n1+n2] = (v_izq, v_der)
        
        max_v = max([abs(v[0]) for v in V_vals.values()] + [abs(v[1]) for v in V_vals.values()]) or 1.0
        scale_v = (L1 * 0.15) / max_v
        
        for (n1, n2), (v_izq, v_der) in V_vals.items():
            draw_diagram_member(ax_v, coords[n1][0], coords[n1][1], coords[n2][0], coords[n2][1], v_izq, v_der, scale_v, color='green', label1=f"{v_izq:.2f}", label2=f"{abs(v_der):.2f}")
            
        for ax in (ax_v, ax_m): ax.axis('equal'); ax.axis('off')
        st.pyplot(fig)


elif ejercicio == "Ejercicio 2: Marco con Columna Inclinada":
    st.header("Marco con una Columna Inclinada")
    
    st.markdown("### Esquema de Referencia con Variables")
    fig_ref2, ax_ref2 = plt.subplots(figsize=(7, 5))
    ax_ref2.plot([0, 0], [0, 3], 'k-', lw=3)
    ax_ref2.plot([0, 4], [3, 3], 'b-', lw=3)
    ax_ref2.plot([4, 6], [3, 0], 'r-', lw=3)
    
    ax_ref2.plot([-0.3, 0.3], [0, 0], 'k-', lw=3)
    for i in np.linspace(-0.3, 0.2, 5): ax_ref2.plot([i, i+0.1], [0, -0.2], 'k-', lw=1)
    ax_ref2.plot([5.7, 6.3], [0, 0], 'k-', lw=3)
    for i in np.linspace(5.7, 6.2, 5): ax_ref2.plot([i, i+0.1], [0, -0.2], 'k-', lw=1)
    
    ax_ref2.text(0.4, 0.2, 'A', fontsize=12, weight='bold'); ax_ref2.text(-0.3, 3.2, 'B', fontsize=12, weight='bold')
    ax_ref2.text(4, 3.2, 'C', fontsize=12, weight='bold'); ax_ref2.text(6.3, 0.2, 'D', fontsize=12, weight='bold')
    
    ax_ref2.plot([0, 4], [3.2, 3.2], 'g-', lw=1)
    for x_val in np.linspace(0, 4, 15): ax_ref2.plot([x_val, x_val], [3, 3.2], 'g-', lw=1)
    ax_ref2.text(2, 3.4, 'w_dist, b (viga_L)', color='green', ha='center')
    
    # Textos de cotas
    ax_ref2.text(-0.8, 1.5, 'a (h)', color='blue', rotation=90)
    ax_ref2.text(5.5, 1.5, 'c (proy_x)', color='blue')
    
    # Fuerza P (Alineada hacia la derecha)
    ax_ref2.annotate('P', xy=(0, 3), xytext=(-2.5, 3), arrowprops=dict(facecolor='red', shrink=0, width=2, headwidth=8), fontsize=12, color='red', va='center')
    
    ax_ref2.set_xlim(-3, 7); ax_ref2.set_ylim(-1, 4.5); ax_ref2.axis('off')
    st.pyplot(fig_ref2)
    plt.close(fig_ref2)

    st.subheader("Parámetros de Entrada")
    col1, col2 = st.columns(2)
    with col1:
        h = st.number_input(f"Altura col recta a ({u_long})", value=3.0)
        viga_L = st.number_input(f"Longitud viga b ({u_long})", value=3.5)
        proy_x = st.number_input(f"Proyección X col incl. c ({u_long})", value=2.0)
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
        
        nodos_fd = {'A': {'B': 0.0}, 'D': {'C': 0.0}, 'B': {'A': FD_BA, 'C': FD_BC}, 'C': {'B': FD_CB, 'D': FD_CD}}
        barras = [('A','B'), ('B','C'), ('C','D')]; orden_miembros = ['AB', 'BA', 'BC', 'CB', 'CD', 'DC']
        
        st.subheader("Tablas de Iteración")
        FEM_0 = {'BC': -(w_dist * viga_L**2)/12, 'CB': (w_dist * viga_L**2)/12}
        df_estado0, M0 = tabla_cross_paso_a_paso(nodos_fd, barras, FEM_0, 4, orden_miembros, "M. INICIAL")
        st.markdown("**Estado 0 (Cargas Reales)**")
        st.dataframe(df_estado0.style.format("{:.3f}"))
        
        M_AB_imp = -10.0
        M_CD_imp = (h**2 * M_AB_imp) / (L_CD**2)
        FEM_1 = {'AB': M_AB_imp, 'BA': M_AB_imp, 'CD': M_CD_imp, 'DC': M_CD_imp}
        df_estado1, M1 = tabla_cross_paso_a_paso(nodos_fd, barras, FEM_1, 4, orden_miembros, "M. IMPUESTO")
        st.markdown("**Estado 1 (Desplazamiento Lateral)**")
        st.dataframe(df_estado1.style.format("{:.3f}"))
        
        V0 = -(M0.get('AB',0) + M0.get('BA',0))/h - (M0.get('CD',0) + M0.get('DC',0))/h
        V1 = -(M1.get('AB',0) + M1.get('BA',0))/h - (M1.get('CD',0) + M1.get('DC',0))/h
        
        if V1 != 0:
            X = (P - V0) / V1
            M_final = {k: M0[k] + X*M1[k] for k in orden_miembros}
            
            st.subheader("Momentos Finales")
            df_final = pd.DataFrame([M_final], index=['Mom. Final'])
            df_final.columns = pd.MultiIndex.from_tuples([(m[0], m) for m in orden_miembros], names=['NODO', 'MIEMBRO'])
            st.dataframe(df_final.style.format("{:.4f}"))
            
            st.subheader("Diagramas de Cortante y Momento")
            fig2, (ax_v, ax_m) = plt.subplots(1, 2, figsize=(14, 6))
            coords = {'A':(0,0), 'B':(0,h), 'C':(viga_L,h), 'D':(viga_L+proy_x,0)}
            
            max_m = max([abs(v) for v in M_final.values()]) or 1.0
            scale_m = (viga_L * 0.15) / max_m
            
            ax_m.set_title("Diagrama de Momento Flexionante")
            draw_support(ax_m, coords['A'][0], coords['A'][1]); draw_support(ax_m, coords['D'][0], coords['D'][1])
            for n1, n2 in [('A','B'), ('B','C'), ('C','D')]:
                v1, v2 = M_final.get(n1+n2, 0), M_final.get(n2+n1, 0)
                draw_diagram_member(ax_m, coords[n1][0], coords[n1][1], coords[n2][0], coords[n2][1], v1, -v2, scale_m, color='purple', label1=f"{v1:.4f}", label2=f"{v2:.4f}")
            
            ax_v.set_title("Diagrama de Fuerza Cortante")
            draw_support(ax_v, coords['A'][0], coords['A'][1]); draw_support(ax_v, coords['D'][0], coords['D'][1])
            V_vals = {}
            for n1, n2, L, w in [('A','B',h,0), ('B','C',viga_L,w_dist), ('C','D',L_CD,0)]:
                v_izq = w*L/2.0 - (M_final.get(n1+n2,0) + M_final.get(n2+n1,0))/L
                v_der = -w*L/2.0 - (M_final.get(n1+n2,0) + M_final.get(n2+n1,0))/L
                V_vals[n1+n2] = (v_izq, v_der)
            
            max_v = max([abs(v[0]) for v in V_vals.values()] + [abs(v[1]) for v in V_vals.values()]) or 1.0
            scale_v = (viga_L * 0.15) / max_v
            
            for (n1, n2), (v_izq, v_der) in V_vals.items():
                draw_diagram_member(ax_v, coords[n1][0], coords[n1][1], coords[n2][0], coords[n2][1], v_izq, v_der, scale_v, color='green', label1=f"{v_izq:.4f}", label2=f"{abs(v_der):.4f}")
            
            for ax in (ax_v, ax_m): ax.axis('equal'); ax.axis('off')
            st.pyplot(fig2)








































































































                                     
