import sys
sys.setrecursionlimit(5000)

import matplotlib
matplotlib.use('Agg')

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

st.set_page_config(page_title="Hardy Cross - Marcos", layout="wide")

# ==========================================
# FUNCIONES DE DIBUJO DE DIAGRAMAS
# ==========================================
def draw_support(ax, x, y, nx=0, ny=-1):
    hx, hy = [], []
    if nx == 0:
        ax.plot([x-0.4, x+0.4], [y, y], 'k-', lw=2)
        for i in np.linspace(-0.4, 0.3, 6):
            hx.extend([x+i, x+i+0.1, np.nan])
            hy.extend([y, y-0.2, np.nan])
    else:
        ax.plot([x-0.4*ny, x+0.4*ny], [y+0.4*nx, y-0.4*nx], 'k-', lw=2)
        for i in np.linspace(-0.4, 0.3, 6):
            xi, yi = x + i*ny, y - i*nx
            hx.extend([xi, xi+0.1*nx, np.nan])
            hy.extend([yi, yi+0.1*ny, np.nan])
            
    if hx: ax.plot(hx, hy, 'k-', lw=1)

def draw_distributed_load(ax, x1, x2, y, label):
    ax.plot([x1, x2], [y+0.5, y+0.5], 'r-', lw=1)
    for px in np.linspace(x1, x2, int((x2-x1)*3)+2):
        ax.annotate('', xy=(px, y), xytext=(px, y+0.5),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1))
    ax.text((x1+x2)/2, y+0.7, label, color='red', ha='center', weight='bold')

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
    ax.fill_between([x1, x2], [y1, y2], [py1, py2], color=color, alpha=0.2)
        
    if abs(v1) > 0.001: ax.text(px1 + 0.3*nx, py1 + 0.3*ny, label1, color='black', fontsize=9, weight='bold', ha='center', va='center')
    if abs(v2) > 0.001: ax.text(px2 + 0.3*nx, py2 + 0.3*ny, label2, color='black', fontsize=9, weight='bold', ha='center', va='center')

def draw_dimension(ax, x1, y1, x2, y2, label, offset_x=0, offset_y=0):
    """Dibuja una línea de cota con flechas en los extremos y texto centrado."""
    px1, py1 = x1 + offset_x, y1 + offset_y
    px2, py2 = x2 + offset_x, y2 + offset_y
    
    ax.annotate('', xy=(px1, py1), xytext=(px2, py2), 
                arrowprops=dict(arrowstyle='<|-|>', color='black', shrinkA=0, shrinkB=0, lw=1))
    
    ax.plot([x1, px1], [y1, py1], color='gray', lw=0.8, ls=':')
    ax.plot([x2, px2], [y2, py2], color='gray', lw=0.8, ls=':')
    
    tx, ty = (px1 + px2) / 2, (py1 + py2) / 2
    if offset_x != 0: tx += offset_x * 0.15
    if offset_y != 0: ty += offset_y * 0.15
    
    ax.text(tx, ty, label, ha='center', va='center', weight='bold', color='black', 
            fontsize=9, bbox=dict(facecolor='white', edgecolor='none', pad=2))

# ==========================================
# FUNCIÓN GENERADORA DE TABLAS CROSS
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

st.title("Análisis Estructural - Método de Hardy Cross")
st.sidebar.header("Configuración de Unidades")
u_fuerza = st.sidebar.selectbox("Unidad de Fuerza", ["ton", "N", "lb"])
u_long = st.sidebar.selectbox("Unidad de Longitud", ["m", "ft", "in"])
u_ang = st.sidebar.selectbox("Unidad de Ángulo", ["Grados", "Radianes"])

ejercicio = st.sidebar.radio("Selecciona el marco:", ("Ejercicio 1: 3 Niveles", "Ejercicio 2: Columna Inclinada"))

# ==============================================================================
# EJERCICIO 1
# ==============================================================================
if ejercicio == "Ejercicio 1: 3 Niveles":
    st.header("Marco con Tres Niveles y Una Crujía")
    
    fig_ref1, ax_ref1 = plt.subplots(figsize=(8, 7))
    ax_ref1.plot([0, 0], [0, 9], 'b-', lw=2); ax_ref1.plot([5, 5], [0, 9], 'b-', lw=2)
    ax_ref1.plot([0, 7], [3, 3], 'b-', lw=2); ax_ref1.plot([0, 7], [6, 6], 'b-', lw=2); ax_ref1.plot([0, 7], [9, 9], 'b-', lw=2)
    draw_support(ax_ref1, 0, 0); draw_support(ax_ref1, 5, 0)
    
    ax_ref1.text(-0.3, 0, 'A', weight='bold'); ax_ref1.text(5.2, 0, 'H', weight='bold')
    ax_ref1.text(-0.3, 3, 'B', weight='bold'); ax_ref1.text(-0.3, 6, 'C', weight='bold'); ax_ref1.text(-0.3, 9, 'D', weight='bold')
    ax_ref1.text(4.7, 3, 'G', weight='bold'); ax_ref1.text(4.7, 6, 'F', weight='bold'); ax_ref1.text(4.7, 9, 'E', weight='bold')
    
    ax_ref1.text(0.2, 1.5, 'aI', color='teal', weight='bold'); ax_ref1.text(5.2, 1.5, 'aI', color='teal', weight='bold')
    ax_ref1.text(0.2, 4.5, 'bI', color='teal', weight='bold'); ax_ref1.text(5.2, 4.5, 'bI', color='teal', weight='bold')
    ax_ref1.text(0.2, 7.5, 'cI', color='teal', weight='bold'); ax_ref1.text(5.2, 7.5, 'cI', color='teal', weight='bold')
    ax_ref1.text(2.5, 2.7, 'dI', color='teal', weight='bold'); ax_ref1.text(2.5, 5.7, 'dI', color='teal', weight='bold'); ax_ref1.text(2.5, 8.7, 'dI', color='teal', weight='bold')

    draw_distributed_load(ax_ref1, 0, 7, 9, '$w_1$'); draw_distributed_load(ax_ref1, 0, 7, 6, '$w_2$'); draw_distributed_load(ax_ref1, 0, 7, 3, '$w_3$')
    ax_ref1.annotate('$P_1$', xy=(7, 9), xytext=(7, 10), arrowprops=dict(facecolor='red', width=1, headwidth=5), color='red', ha='center')
    ax_ref1.annotate('$P_2$', xy=(7, 6), xytext=(7, 7), arrowprops=dict(facecolor='red', width=1, headwidth=5), color='red', ha='center')
    ax_ref1.annotate('$P_3$', xy=(7, 3), xytext=(7, 4), arrowprops=dict(facecolor='red', width=1, headwidth=5), color='red', ha='center')
    
    ax_ref1.annotate('$F_1$', xy=(0, 9), xytext=(-1.5, 9), arrowprops=dict(facecolor='red', width=1, headwidth=5), color='red', va='center')
    ax_ref1.annotate('$F_2$', xy=(0, 6), xytext=(-1.5, 6), arrowprops=dict(facecolor='red', width=1, headwidth=5), color='red', va='center')
    ax_ref1.annotate('$F_3$', xy=(0, 3), xytext=(-1.5, 3), arrowprops=dict(facecolor='red', width=1, headwidth=5), color='red', va='center')
    
    # NUEVAS LÍNEAS DE COTAS PARA EJERCICIO 1
    draw_dimension(ax_ref1, 0, 0, 0, 3, '$L_3$', offset_x=-2.2)
    draw_dimension(ax_ref1, 0, 3, 0, 6, '$L_4$', offset_x=-2.2)
    draw_dimension(ax_ref1, 0, 6, 0, 9, '$L_5$', offset_x=-2.2)
    draw_dimension(ax_ref1, 0, 0, 5, 0, '$L_1$', offset_y=-1.5)
    draw_dimension(ax_ref1, 5, 0, 7, 0, '$L_2$', offset_y=-1.5)
    
    ax_ref1.set_xlim(-4, 9); ax_ref1.set_ylim(-2.5, 11.5); ax_ref1.axis('off')
    st.pyplot(fig_ref1, clear_figure=True)
    plt.close(fig_ref1)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        L1 = st.number_input("L1 (m)", value=5.0); L2 = st.number_input("L2 (m)", value=1.5)
        L3 = st.number_input("L3 (m)", value=4.0); L4 = st.number_input("L4 (m)", value=3.0); L5 = st.number_input("L5 (m)", value=3.0)
    with col2:
        w1 = st.number_input("w1 (Ton/m)", value=2.0); P1 = st.number_input("P1 (Ton)", value=2.0); F1 = st.number_input("F1 (Ton)", value=8.0)
        w2 = st.number_input("w2 (Ton/m)", value=3.0); P2 = st.number_input("P2 (Ton)", value=2.0); F2 = st.number_input("F2 (Ton)", value=6.0)
        w3 = st.number_input("w3 (Ton/m)", value=3.0); P3 = st.number_input("P3 (Ton)", value=2.0); F3 = st.number_input("F3 (Ton)", value=4.0)
    with col3:
        a_coef = st.number_input("Coeficiente Inercia a", value=4.0)
        b_coef = st.number_input("Coeficiente Inercia b", value=3.0)
        c_coef = st.number_input("Coeficiente Inercia c", value=2.0)
        d_coef = st.number_input("Coeficiente Inercia d", value=1.0)

    if st.button("Resolver Ejercicio 1"):
        st.write("---")
        
        st.markdown("### Paso 1: Rigideces Angulares (K)")
        K_AB = a_coef / L3; K_GH = K_AB
        K_BC = b_coef / L4; K_FG = K_BC
        K_CD = c_coef / L5; K_EF = K_CD
        K_V = d_coef / L1
        
        st.latex(rf"K_{{AB}} = K_{{GH}} = {K_AB:.4f} \quad K_{{BC}} = K_{{FG}} = {K_BC:.4f}")
        st.latex(rf"K_{{CD}} = K_{{EF}} = {K_CD:.4f} \quad K_{{Vigas}} = {K_V:.4f}")
        
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
        
        st.markdown("### Paso 2: Factores de Distribución (FD)")
        st.latex(r"\text{Nodos A y H (Empotramientos):} \quad FD_{AB}=0.0000, \quad FD_{HG}=0.0000")
        st.latex(rf"\text{{Nodo B:}} \quad FD_{{BA}}={nodos_fd['B']['A']:.4f}, \quad FD_{{BC}}={nodos_fd['B']['C']:.4f}, \quad FD_{{BG}}={nodos_fd['B']['G']:.4f}")
        st.latex(rf"\text{{Nodo C:}} \quad FD_{{CB}}={nodos_fd['C']['B']:.4f}, \quad FD_{{CD}}={nodos_fd['C']['D']:.4f}, \quad FD_{{CF}}={nodos_fd['C']['F']:.4f}")
        st.latex(rf"\text{{Nodo D:}} \quad FD_{{DC}}={nodos_fd['D']['C']:.4f}, \quad FD_{{DE}}={nodos_fd['D']['E']:.4f}")
        st.latex(rf"\text{{Nodo E:}} \quad FD_{{ED}}={nodos_fd['E']['D']:.4f}, \quad FD_{{EF}}={nodos_fd['E']['F']:.4f}")
        st.latex(rf"\text{{Nodo F:}} \quad FD_{{FC}}={nodos_fd['F']['C']:.4f}, \quad FD_{{FE}}={nodos_fd['F']['E']:.4f}, \quad FD_{{FG}}={nodos_fd['F']['G']:.4f}")
        st.latex(rf"\text{{Nodo G:}} \quad FD_{{GB}}={nodos_fd['G']['B']:.4f}, \quad FD_{{GF}}={nodos_fd['G']['F']:.4f}, \quad FD_{{GH}}={nodos_fd['G']['H']:.4f}")
        
        barras = [('A','B'), ('B','C'), ('C','D'), ('D','E'), ('C','F'), ('B','G'), ('E','F'), ('F','G'), ('G','H')]
        orden_miembros = ['AB', 'BA', 'BC', 'CB', 'CD', 'DC', 'DE', 'ED', 'EVol', 'EF', 'FE', 'FVol', 'FC', 'CF', 'FG', 'GF', 'GVol', 'GB', 'BG', 'GH', 'HG']
        
        FEM_0 = {
            'DE': -(w1*L1**2)/12, 'ED': (w1*L1**2)/12, 'CF': -(w2*L1**2)/12, 'FC': (w2*L1**2)/12,
            'BG': -(w3*L1**2)/12, 'GB': (w3*L1**2)/12, 'EVol': -(w1*L2**2)/2 - P1*L2,
            'FVol': -(w2*L2**2)/2 - P2*L2, 'GVol': -(w3*L2**2)/2 - P3*L2
        }
        FEM_1 = {'AB': -100, 'BA': -100, 'GH': -100, 'HG': -100}
        FEM_2 = {'BC': -100, 'CB': -100, 'FG': -100, 'GF': -100}
        FEM_3 = {'CD': -100, 'DC': -100, 'EF': -100, 'FE': -100}
        
        st.markdown("### Paso 3: Momentos de Empotramiento (Iniciales e Impuestos)")
        st.write("**Estado 0 (Cargas Reales en Vigas y Voladizos):**")
        st.latex(rf"M_{{DE}} = {FEM_0['DE']:.2f}, \quad M_{{ED}} = {FEM_0['ED']:.2f}")
        st.latex(rf"M_{{CF}} = {FEM_0['CF']:.2f}, \quad M_{{FC}} = {FEM_0['FC']:.2f}")
        st.latex(rf"M_{{BG}} = {FEM_0['BG']:.2f}, \quad M_{{GB}} = {FEM_0['GB']:.2f}")
        st.latex(rf"M_{{EVol}} = {FEM_0['EVol']:.2f}, \quad M_{{FVol}} = {FEM_0['FVol']:.2f}, \quad M_{{GVol}} = {FEM_0['GVol']:.2f}")
        
        st.write("**Estados Impuestos (Desplazamientos Laterales \u03941, \u03942, \u03943):**")
        st.latex(rf"\Delta_1 (\text{{Nivel 1}}): M_{{AB}} = M_{{BA}} = M_{{GH}} = M_{{HG}} = {FEM_1['AB']:.2f}")
        st.latex(rf"\Delta_2 (\text{{Nivel 2}}): M_{{BC}} = M_{{CB}} = M_{{FG}} = M_{{GF}} = {FEM_2['BC']:.2f}")
        st.latex(rf"\Delta_3 (\text{{Nivel 3}}): M_{{CD}} = M_{{DC}} = M_{{EF}} = M_{{FE}} = {FEM_3['CD']:.2f}")
        
        st.markdown("### Paso 4: Desglose de Tablas de Iteración (Hardy Cross)")
        df_estado0, M0 = tabla_cross_paso_a_paso(nodos_fd, barras, FEM_0, 10, orden_miembros, "M. INICIAL")
        with st.expander("Ver Tabla Estado 0 (Cargas Reales)", expanded=True):
            st.dataframe(df_estado0.round(3))
        
        df_estado1, M1 = tabla_cross_paso_a_paso(nodos_fd, barras, FEM_1, 10, orden_miembros, "M. IMPUESTO (\u03941)")
        with st.expander("Ver Tabla Estado 1 (Desplazamiento Nivel 1)"):
            st.dataframe(df_estado1.round(3))
        
        df_estado2, M2 = tabla_cross_paso_a_paso(nodos_fd, barras, FEM_2, 10, orden_miembros, "M. IMPUESTO (\u03942)")
        with st.expander("Ver Tabla Estado 2 (Desplazamiento Nivel 2)"):
            st.dataframe(df_estado2.round(3))
        
        df_estado3, M3 = tabla_cross_paso_a_paso(nodos_fd, barras, FEM_3, 10, orden_miembros, "M. IMPUESTO (\u03943)")
        with st.expander("Ver Tabla Estado 3 (Desplazamiento Nivel 3)"):
            st.dataframe(df_estado3.round(3))
        
        st.markdown("### Paso 5: Factores de Corrección")
        def calc_cortantes(M_dict):
            V1 = -(M_dict.get('AB',0) + M_dict.get('BA',0) + M_dict.get('GH',0) + M_dict.get('HG',0)) / L3
            V2 = -(M_dict.get('BC',0) + M_dict.get('CB',0) + M_dict.get('FG',0) + M_dict.get('GF',0)) / L4
            V3 = -(M_dict.get('CD',0) + M_dict.get('DC',0) + M_dict.get('EF',0) + M_dict.get('FE',0)) / L5
            return np.array([V1, V2, V3])

        V0, V1, V2, V3 = calc_cortantes(M0), calc_cortantes(M1), calc_cortantes(M2), calc_cortantes(M3)
        Matriz_V = np.column_stack((V1, V2, V3))
        Vector_F = np.array([F1 + F2 + F3, F1 + F2, F1]) - V0
        
        X = np.linalg.solve(Matriz_V, Vector_F)
        st.latex(rf"X_1 = {X[0]:.4f}, \quad X_2 = {X[1]:.4f}, \quad X_3 = {X[2]:.4f}")
        M_final = {k: M0[k] + X[0]*M1[k] + X[1]*M2[k] + X[2]*M3[k] for k in orden_miembros}
        
        st.markdown("### Paso 6: Momentos Finales")
        df_final = pd.DataFrame([M_final], index=['Mom. Final'])
        df_final.columns = pd.MultiIndex.from_tuples([(m[0], m) for m in orden_miembros], names=['NODO', 'MIEMBRO'])
        st.dataframe(df_final.astype(float).round(4))
        
        st.markdown("### Paso 7: Reacciones Finales")
        R_Ax = -(M_final.get('AB',0) + M_final.get('BA',0)) / L3
        R_Hx = -(M_final.get('GH',0) + M_final.get('HG',0)) / L3
        
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("**En el Apoyo A:**")
            st.latex(rf"M_A = M_{{AB}} = {M_final.get('AB',0):.4f} \text{{ {u_fuerza}-{u_long}}}")
            st.latex(rf"R_{{Ax}} = {R_Ax:.4f} \text{{ {u_fuerza}}}")
        with col_r2:
            st.markdown("**En el Apoyo H:**")
            st.latex(rf"M_H = M_{{HG}} = {M_final.get('HG',0):.4f} \text{{ {u_fuerza}-{u_long}}}")
            st.latex(rf"R_{{Hx}} = {R_Hx:.4f} \text{{ {u_fuerza}}}")

        st.markdown("### Paso 8: Diagramas de Cortante y Momento")
        fig_res1, (ax_v, ax_m) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Coordenadas actualizadas (incluyendo extremos de los voladizos)
        coords = {
            'A':(0,0), 'B':(0,L3), 'C':(0,L3+L4), 'D':(0,L3+L4+L5), 
            'H':(L1,0), 'G':(L1,L3), 'F':(L1,L3+L4), 'E':(L1,L3+L4+L5),
            'E_vol':(L1+L2, L3+L4+L5), 'F_vol':(L1+L2, L3+L4), 'G_vol':(L1+L2, L3)
        }
        
        max_m = max([abs(v) for v in M_final.values() if not np.isnan(v)], default=0.0)
        scale_m = (L1 * 0.15) / max_m if max_m > 1e-4 else 1.0
        
        ax_m.set_title("Diagrama de Momento Flexionante")
        draw_support(ax_m, coords['A'][0], coords['A'][1]); draw_support(ax_m, coords['H'][0], coords['H'][1])
        tramos_m = [('A','B'), ('B','C'), ('C','D'), ('H','G'), ('G','F'), ('F','E'), ('B','G'), ('C','F'), ('D','E')]
        
        for n1, n2 in tramos_m:
            v1, v2 = M_final.get(n1+n2, 0), M_final.get(n2+n1, 0)
            draw_diagram_member(ax_m, coords[n1][0], coords[n1][1], coords[n2][0], coords[n2][1], v1, -v2, scale_m, color='purple', label1=f"{v1:.2f}", label2=f"{v2:.2f}")

        # Dibujar momentos en los voladizos
        for n_start, n_end, name_m in [('E', 'E_vol', 'EVol'), ('F', 'F_vol', 'FVol'), ('G', 'G_vol', 'GVol')]:
            v1 = M_final.get(name_m, 0)
            draw_diagram_member(ax_m, coords[n_start][0], coords[n_start][1], coords[n_end][0], coords[n_end][1], v1, 0, scale_m, color='purple', label1=f"{v1:.2f}", label2="0.00")

        ax_v.set_title("Diagrama de Fuerza Cortante")
        draw_support(ax_v, coords['A'][0], coords['A'][1]); draw_support(ax_v, coords['H'][0], coords['H'][1])
        
        V_vals = {}
        for n1, n2, L, w_val in [('A','B',L3,0), ('B','C',L4,0), ('C','D',L5,0), ('H','G',L3,0), ('G','F',L4,0), ('F','E',L5,0), ('B','G',L1,w3), ('C','F',L1,w2), ('D','E',L1,w1)]:
            v_izq = w_val*L/2.0 - (M_final.get(n1+n2,0) + M_final.get(n2+n1,0))/L
            v_der = -w_val*L/2.0 - (M_final.get(n1+n2,0) + M_final.get(n2+n1,0))/L
            V_vals[n1+n2] = (v_izq, v_der)
        
        # Información de cortantes en voladizos (v_apoyo, v_punta)
        vol_shear_info = [
            ('E', 'E_vol', w1*L2 + P1, P1), 
            ('F', 'F_vol', w2*L2 + P2, P2), 
            ('G', 'G_vol', w3*L2 + P3, P3)
        ]
        
        max_v_tramos = max([abs(v[0]) for v in V_vals.values()] + [abs(v[1]) for v in V_vals.values()], default=0.0)
        max_v_vol = max([abs(v[2]) for v in vol_shear_info] + [abs(v[3]) for v in vol_shear_info], default=0.0)
        max_v = max(max_v_tramos, max_v_vol)
        scale_v = (L1 * 0.15) / max_v if max_v > 1e-4 else 1.0
        
        # Dibujar cortante normal
        for (n1, n2), (v_izq, v_der) in V_vals.items():
            draw_diagram_member(ax_v, coords[n1][0], coords[n1][1], coords[n2][0], coords[n2][1], v_izq, v_der, scale_v, color='green', label1=f"{v_izq:.2f}", label2=f"{abs(v_der):.2f}")
            
        # Dibujar cortante en voladizos
        for n_start, n_end, v_izq, v_der in vol_shear_info:
            draw_diagram_member(ax_v, coords[n_start][0], coords[n_start][1], coords[n_end][0], coords[n_end][1], v_izq, v_der, scale_v, color='green', label1=f"{v_izq:.2f}", label2=f"{v_der:.2f}")
            
        ax_v.annotate(f"Rx = {R_Ax:.2f}", xy=(coords['A'][0], coords['A'][1]), xytext=(coords['A'][0]-2.0, coords['A'][1]-0.8), 
                      arrowprops=dict(arrowstyle="->", color='blue', lw=1.5), color='blue', weight='bold', ha='center')
        ax_v.annotate(f"Rx = {R_Hx:.2f}", xy=(coords['H'][0], coords['H'][1]), xytext=(coords['H'][0]+2.0, coords['H'][1]-0.8), 
                      arrowprops=dict(arrowstyle="->", color='blue', lw=1.5), color='blue', weight='bold', ha='center')
            
        for ax in (ax_v, ax_m): 
            ax.set_xlim(-4, L1+L2+3); ax.set_ylim(-1.5, L3+L4+L5+2)
            ax.set_xticks([]); ax.set_yticks([])
            ax.axis('off')
        st.pyplot(fig_res1, clear_figure=True)
        plt.close(fig_res1)

# ==============================================================================
# EJERCICIO 2
# ==============================================================================
elif ejercicio == "Ejercicio 2: Columna Inclinada":
    st.header("Marco con una Columna Inclinada")
    
    fig_ref2, ax_ref2 = plt.subplots(figsize=(8, 6))
    ax_ref2.plot([0, 0], [0, 4], 'b-', lw=2); ax_ref2.plot([0, 5], [4, 4], 'b-', lw=2); ax_ref2.plot([5, 8], [4, 0], 'b-', lw=2)
    draw_support(ax_ref2, 0, 0); draw_support(ax_ref2, 8, 0)
    
    ax_ref2.text(-0.3, 0, 'A', weight='bold'); ax_ref2.text(-0.3, 4, 'B', weight='bold')
    ax_ref2.text(5.2, 4, 'C', weight='bold'); ax_ref2.text(8.3, 0, 'D', weight='bold')
    ax_ref2.text(0.2, 2, 'I', color='teal', weight='bold'); ax_ref2.text(2.5, 3.7, 'I', color='teal', weight='bold'); ax_ref2.text(6.7, 2.2, 'I', color='teal', weight='bold')
    
    draw_distributed_load(ax_ref2, 0, 5, 4, '$w$')
    ax_ref2.annotate('$F$', xy=(0, 4), xytext=(-2.5, 4), arrowprops=dict(facecolor='red', width=1.5, headwidth=6), color='red', va='center')
    
    # NUEVAS LÍNEAS DE COTAS PARA EJERCICIO 2
    draw_dimension(ax_ref2, 0, 0, 0, 4, '$a$', offset_x=-1.5)
    draw_dimension(ax_ref2, 0, 4, 5, 4, '$b$', offset_y=1.5)
    draw_dimension(ax_ref2, 5, 4, 8, 4, '$c$', offset_y=1.5)
    
    ax_ref2.plot([7, 9], [0, 0], 'k-', lw=1)
    theta_arc = np.linspace(np.pi - np.arctan(4/3), np.pi, 20)
    ax_ref2.plot(8 + 0.8*np.cos(theta_arc), 0.8*np.sin(theta_arc), 'r-', lw=1.5)
    ax_ref2.text(7.3, 0.4, r'$\theta$', color='red', weight='bold')
    
    ax_ref2.set_xlim(-4, 10); ax_ref2.set_ylim(-1.5, 7.5); ax_ref2.axis('off')
    st.pyplot(fig_ref2, clear_figure=True)
    plt.close(fig_ref2)
    
    col1, col2 = st.columns(2)
    with col1:
        a = st.number_input("Altura columna a (m)", value=3.0)
        b = st.number_input("Longitud viga b (m)", value=3.5)
        c = st.number_input("Proyección columna c (m)", value=2.0)
        ang_simbolo = "°" if u_ang == "Grados" else "rad"
        angulo = st.number_input(f"Ángulo inclinación θ ({ang_simbolo})", value=40.0)
    with col2:
        w = st.number_input("Carga distribuida w (Ton/m)", value=2.0)
        F = st.number_input("Fuerza F (Ton)", value=3.0)

    if st.button("Resolver Ejercicio 2"):
        st.write("---")
        
        st.markdown("### Paso 1: Cálculo de Longitud CD (Marco Inclinado)")
        if u_ang == "Grados":
            L_CD = c / np.cos(np.radians(angulo))
            st.latex(rf"L_{{CD}} = \frac{{c}}{{\cos(\theta)}} = \frac{{{c}}}{{\cos({angulo}^\circ)}} = {L_CD:.4f}")
        else:
            L_CD = c / np.cos(angulo)
            st.latex(rf"L_{{CD}} = \frac{{c}}{{\cos(\theta)}} = \frac{{{c}}}{{\cos({angulo} \text{{ rad}})}} = {L_CD:.4f}")
            
        st.markdown("### Paso 2: Rigideces Angulares (K)")
        K_AB = 1.0 / a; K_BC = 1.0 / b; K_CD = 1.0 / L_CD
        st.latex(rf"K_{{AB}} = {K_AB:.4f}, \quad K_{{BC}} = {K_BC:.4f}, \quad K_{{CD}} = {K_CD:.4f}")
        
        st.markdown("### Paso 3: Factores de Distribución (FD)")
        FD_BA = K_AB/(K_AB+K_BC); FD_BC = K_BC/(K_AB+K_BC)
        FD_CB = K_BC/(K_BC+K_CD); FD_CD = K_CD/(K_BC+K_CD)
        st.latex(r"\text{Nodo A:} \quad FD_{AB}=0.0000 \quad \text{(Empotramiento)}")
        st.latex(rf"\text{{Nodo B:}} \quad FD_{{BA}}={FD_BA:.4f}, \quad FD_{{BC}}={FD_BC:.4f}")
        st.latex(rf"\text{{Nodo C:}} \quad FD_{{CB}}={FD_CB:.4f}, \quad FD_{{CD}}={FD_CD:.4f}")
        st.latex(r"\text{Nodo D:} \quad FD_{DC}=0.0000 \quad \text{(Empotramiento)}")
        
        st.markdown("### Paso 4: Momentos de Empotramiento (Iniciales e Impuestos)")
        FEM_0 = {'BC': -(w * b**2)/12, 'CB': (w * b**2)/12}
        M_AB_imp = -100.0
        M_CD_imp = (a**2 * M_AB_imp) / (L_CD**2)
        FEM_1 = {'AB': M_AB_imp, 'BA': M_AB_imp, 'CD': M_CD_imp, 'DC': M_CD_imp}
        st.write(f"Estado 0 (Real): $M_{{BC}} = {FEM_0['BC']:.2f}, M_{{CB}} = {FEM_0['CB']:.2f}$")
        st.write(rf"Estado 1 (Impuesto): $M_{{AB}} = M_{{BA}} = {FEM_1['AB']:.2f}, \quad M_{{CD}} = M_{{DC}} = {FEM_1['CD']:.2f}$")
        
        st.markdown("### Paso 5: Desglose de Tablas de Iteración (Hardy Cross)")
        nodos_fd = {'A': {'B': 0.0}, 'D': {'C': 0.0}, 'B': {'A': FD_BA, 'C': FD_BC}, 'C': {'B': FD_CB, 'D': FD_CD}}
        barras = [('A','B'), ('B','C'), ('C','D')]; orden_miembros = ['AB', 'BA', 'BC', 'CB', 'CD', 'DC']
        
        df_estado0, M0 = tabla_cross_paso_a_paso(nodos_fd, barras, FEM_0, 6, orden_miembros, "M. INICIAL")
        st.markdown("**Tabla Estado 0 (Cargas Reales)**")
        st.dataframe(df_estado0.round(3))
        
        df_estado1, M1 = tabla_cross_paso_a_paso(nodos_fd, barras, FEM_1, 6, orden_miembros, "M. IMPUESTO")
        st.markdown("**Tabla Estado 1 (Desplazamiento Lateral)**")
        st.dataframe(df_estado1.round(3))
        
        st.markdown("### Paso 6: Factor de Corrección por Desplazamiento")
        V0 = -(M0.get('AB',0) + M0.get('BA',0))/a - (M0.get('CD',0) + M0.get('DC',0))/a
        V1 = -(M1.get('AB',0) + M1.get('BA',0))/a - (M1.get('CD',0) + M1.get('DC',0))/a
        
        X = (F - V0) / V1
        st.latex(rf"X = \frac{{F - V_0}}{{V_1}} = {X:.4f}")
        
        st.markdown("### Paso 7: Momentos Finales")
        M_final = {k: M0[k] + X*M1[k] for k in orden_miembros}
        df_final = pd.DataFrame([M_final], index=['Mom. Final'])
        df_final.columns = pd.MultiIndex.from_tuples([(m[0], m) for m in orden_miembros], names=['NODO', 'MIEMBRO'])
        st.dataframe(df_final.astype(float).round(4))
        
        st.markdown("### Paso 8: Reacciones Finales")
        R_Ax = -(M_final.get('AB',0) + M_final.get('BA',0)) / a
        R_Dx = -F - R_Ax 
        
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("**En el Apoyo A:**")
            st.latex(rf"M_A = M_{{AB}} = {M_final.get('AB',0):.4f} \text{{ {u_fuerza}-{u_long}}}")
            st.latex(rf"R_{{Ax}} = {R_Ax:.4f} \text{{ {u_fuerza}}}")
        with col_r2:
            st.markdown("**En el Apoyo D:**")
            st.latex(rf"M_D = M_{{DC}} = {M_final.get('DC',0):.4f} \text{{ {u_fuerza}-{u_long}}}")
            st.latex(rf"R_{{Dx}} = {R_Dx:.4f} \text{{ {u_fuerza}}}")

        st.markdown("### Paso 9: Diagramas de Cortante y Momento")
        fig_res2, (ax_v, ax_m) = plt.subplots(1, 2, figsize=(14, 6))
        coords = {'A':(0,0), 'B':(0,a), 'C':(b,a), 'D':(b+c,0)}
        
        max_m = max([abs(v) for v in M_final.values() if not np.isnan(v)], default=0.0)
        scale_m = (b * 0.15) / max_m if max_m > 1e-4 else 1.0
        
        ax_m.set_title("Diagrama de Momento Flexionante")
        draw_support(ax_m, coords['A'][0], coords['A'][1]); draw_support(ax_m, coords['D'][0], coords['D'][1])
        for n1, n2 in [('A','B'), ('B','C'), ('C','D')]:
            v1, v2 = M_final.get(n1+n2, 0), M_final.get(n2+n1, 0)
            draw_diagram_member(ax_m, coords[n1][0], coords[n1][1], coords[n2][0], coords[n2][1], v1, -v2, scale_m, color='purple', label1=f"{v1:.4f}", label2=f"{v2:.4f}")
        
        ax_v.set_title("Diagrama de Fuerza Cortante")
        draw_support(ax_v, coords['A'][0], coords['A'][1]); draw_support(ax_v, coords['D'][0], coords['D'][1])
        V_vals = {}
        for n1, n2, L, load in [('A','B',a,0), ('B','C',b,w), ('C','D',L_CD,0)]:
            v_izq = load*L/2.0 - (M_final.get(n1+n2,0) + M_final.get(n2+n1,0))/L
            v_der = -load*L/2.0 - (M_final.get(n1+n2,0) + M_final.get(n2+n1,0))/L
            V_vals[n1+n2] = (v_izq, v_der)
        
        max_v = max([abs(v[0]) for v in V_vals.values()] + [abs(v[1]) for v in V_vals.values()], default=0.0)
        scale_v = (b * 0.15) / max_v if max_v > 1e-4 else 1.0
        
        for (n1, n2), (v_izq, v_der) in V_vals.items():
            draw_diagram_member(ax_v, coords[n1][0], coords[n1][1], coords[n2][0], coords[n2][1], v_izq, v_der, scale_v, color='green', label1=f"{v_izq:.4f}", label2=f"{abs(v_der):.4f}")
        
        ax_v.annotate(f"Rx = {R_Ax:.2f}", xy=(coords['A'][0], coords['A'][1]), xytext=(coords['A'][0]-1.5, coords['A'][1]-0.8), 
                      arrowprops=dict(arrowstyle="->", color='blue', lw=1.5), color='blue', weight='bold', ha='center')
        ax_v.annotate(f"Rx = {R_Dx:.2f}", xy=(coords['D'][0], coords['D'][1]), xytext=(coords['D'][0]+1.5, coords['D'][1]-0.8), 
                      arrowprops=dict(arrowstyle="->", color='blue', lw=1.5), color='blue', weight='bold', ha='center')
        
        for ax in (ax_v, ax_m): 
            ax.set_xlim(-3, b+c+4); ax.set_ylim(-1.5, a+2)
            ax.set_xticks([]); ax.set_yticks([])
            ax.axis('off')
        st.pyplot(fig_res2, clear_figure=True)
        plt.close(fig_res2)
