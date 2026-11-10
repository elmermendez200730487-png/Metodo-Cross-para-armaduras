import sys
sys.setrecursionlimit(5000)

import matplotlib
matplotlib.use('Agg')

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import math

st.set_page_config(page_title="Hardy Cross - Marcos", layout="wide")

def reset_state():
    st.session_state.resuelto = False

if 'resuelto' not in st.session_state:
    st.session_state.resuelto = False

# ==========================================
# FUNCIONES DE DIBUJO DE DIAGRAMAS
# ==========================================
def draw_support(ax, x, y, nx=0, ny=-1, is_pin=False):
    if is_pin:
        ax.plot([x-0.3, x+0.3], [y-0.4, y-0.4], 'k-', lw=2)
        ax.plot([x, x-0.3, x+0.3, x], [y, y-0.4, y-0.4, y], 'k-', lw=1.5)
    else:
        hx, hy = [], []
        ax.plot([x-0.4, x+0.4], [y, y], 'k-', lw=2)
        for i in np.linspace(-0.4, 0.3, 6):
            hx.extend([x+i, x+i+0.1, np.nan])
            hy.extend([y, y-0.2, np.nan])
        if hx: ax.plot(hx, hy, 'k-', lw=1)

def draw_distributed_load(ax, x1, x2, y, label, is_vertical=True):
    if is_vertical:
        ax.plot([x1, x2], [y+0.5, y+0.5], 'r-', lw=1)
        for px in np.linspace(x1, x2, int((x2-x1)*3)+2):
            ax.annotate('', xy=(px, y), xytext=(px, y+0.5), arrowprops=dict(arrowstyle='->', color='red', lw=1))
        ax.text((x1+x2)/2, y+0.7, label, color='red', ha='center', weight='bold')

def draw_diagram_member(ax, x1, y1, x2, y2, v1, v2, scale, color='blue', label1="", label2="", draw_member=True):
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L == 0: return
    nx, ny = -dy/L, dx/L
    
    px1, py1 = x1 + v1 * scale * nx, y1 + v1 * scale * ny
    px2, py2 = x2 + v2 * scale * nx, y2 + v2 * scale * ny
    
    if draw_member: ax.plot([x1, x2], [y1, y2], 'k-', lw=2.5)
    ax.plot([x1, px1, px2, x2], [y1, py1, py2, y2], color=color, lw=1.2)
    ax.fill_between([x1, x2], [y1, y2], [py1, py2], color=color, alpha=0.2)
        
    if abs(v1) > 0.001: ax.text(px1 + 0.4*nx, py1 + 0.4*ny, label1, color='black', fontsize=9, weight='bold', ha='center', va='center')
    if abs(v2) > 0.001: ax.text(px2 + 0.4*nx, py2 + 0.4*ny, label2, color='black', fontsize=9, weight='bold', ha='center', va='center')

def tabla_cross_paso_a_paso(nodos_fd, barras, FEM, iteraciones, orden_miembros, nombre_m_inicial):
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

st.sidebar.header("Tipo de Ejercicio")
tipo_ejercicio = st.sidebar.radio("Selecciona el modelo:", ["Ejercicio 1: Marco Ortogonal Dinámico", "Ejercicio 2: Marco Inclinado"], on_change=reset_state)

st.sidebar.header("Unidades")
u_fuerza = st.sidebar.selectbox("Fuerza", ["ton", "N", "lb", "kN"])
u_long = st.sidebar.selectbox("Longitud", ["m", "ft", "in"])

# ==============================================================================
# EJERCICIO 1: MARCO ORTOGONAL DINÁMICO
# ==============================================================================
if tipo_ejercicio == "Ejercicio 1: Marco Ortogonal Dinámico":
    num_niveles = st.sidebar.radio("Número de niveles:", (1, 2, 3), index=2, on_change=reset_state)
    
    st.sidebar.subheader("Voladizos Activos (Derecha)")
    vol_activos = {}
    for i in range(1, num_niveles + 1):
        vol_activos[i] = st.sidebar.checkbox(f"Voladizo en Nivel {i}", value=True, on_change=reset_state)

    st.header(f"Marco de {num_niveles} Nivel(es)")

    fig_ref, ax_ref = plt.subplots(figsize=(8, 7))
    ax_ref.plot([0, 0], [0, num_niveles*3], 'b-', lw=2)
    ax_ref.plot([5, 5], [0, num_niveles*3], 'b-', lw=2)

    draw_support(ax_ref, 0, 0); draw_support(ax_ref, 5, 0)
    ax_ref.text(-0.3, 0, 'A', weight='bold'); ax_ref.text(5.2, 0, 'H', weight='bold')

    for i in range(1, num_niveles + 1):
        y = i * 3
        ax_ref.plot([0, 5], [y, y], 'b-', lw=2)
        ax_ref.text(2.5, y - 0.3, 'L', color='black', ha='center', style='italic')
        ax_ref.text(2.5, y + 0.2, 'Iv', color='blue', ha='center', style='italic', weight='bold')
        
        ax_ref.annotate('', xy=(-0.8, y-3), xytext=(-0.8, y), arrowprops=dict(arrowstyle='<->', color='green'))
        ax_ref.text(-1.0, y - 1.5, f'H{i}', color='green', ha='right', va='center', weight='bold')
        ax_ref.text(-0.2, y - 1.5, f'Ic{i}', color='blue', ha='right', va='center', style='italic', weight='bold')
        ax_ref.text(5.2, y - 1.5, f'Ic{i}', color='blue', ha='left', va='center', style='italic', weight='bold')
        
        draw_distributed_load(ax_ref, 0, 5, y, f"w{i}")
        ax_ref.annotate('', xy=(0, y), xytext=(-1.5, y), arrowprops=dict(arrowstyle='->', color='purple', lw=2))
        ax_ref.text(-1.7, y, f'F{i}', color='purple', ha='right', va='center', weight='bold')

        if vol_activos[i]:
            ax_ref.plot([5, 7], [y, y], 'b-', lw=2) 
            ax_ref.annotate('', xy=(7, y+1), xytext=(7, y), arrowprops=dict(arrowstyle='<-', color='orange', lw=2))
            ax_ref.text(7, y+1.2, f'P{i}', color='orange', ha='center', weight='bold')
            ax_ref.text(6, y - 0.3, f'Lvol_{i}', color='black', ha='center', style='italic')
            
        if i == 1:
            ax_ref.text(-0.3, y, 'B', weight='bold'); ax_ref.text(4.7, y, 'G', weight='bold')
        elif i == 2:
            ax_ref.text(-0.3, y, 'C', weight='bold'); ax_ref.text(4.7, y, 'F', weight='bold')
        elif i == 3:
            ax_ref.text(-0.3, y, 'D', weight='bold'); ax_ref.text(4.7, y, 'E', weight='bold')

    ax_ref.set_xlim(-3, 9); ax_ref.set_ylim(-1.5, num_niveles*3 + 2.5); ax_ref.axis('off')
    st.pyplot(fig_ref, clear_figure=True)
    plt.close(fig_ref)

    st.subheader("Geometría e Inercia")
    colG1, colG2 = st.columns(2)
    with colG1:
        L_viga = st.number_input("Long. Viga Central (L)", value=5.0, on_change=reset_state)
        coef_viga = st.number_input("Inercia Vigas (Iv)", value=1.0, on_change=reset_state)
    with colG2:
        H_niveles = []; coef_cols = []; L_vols = {}
        for i in range(1, num_niveles + 1):
            H_niveles.append(st.number_input(f"Altura Nivel {i} (H{i})", value=3.0, key=f"H{i}", on_change=reset_state))
            coef_cols.append(st.number_input(f"Inercia Col Nivel {i} (Ic{i})", value=4.0 if i==1 else (3.0 if i==2 else 2.0), key=f"I{i}", on_change=reset_state))
            if vol_activos[i]:
                L_vols[i] = st.number_input(f"Long. Voladizo Nivel {i} (Lvol_{i})", value=1.5, key=f"Lvol{i}", on_change=reset_state)
            else:
                L_vols[i] = 0.0

    st.subheader("Cargas (0 para anular)")
    cargas_W = []; cargas_P = []; cargas_F = []
    for i in range(1, num_niveles + 1):
        st.markdown(f"**Nivel {i}:**")
        colC1, colC2, colC3 = st.columns(3)
        with colC1: cargas_W.append(st.number_input(f"Distribuida (w{i})", value=2.0, key=f"w{i}", on_change=reset_state))
        with colC2: cargas_P.append(st.number_input(f"Puntual en voladizo (P{i})", value=2.0 if vol_activos[i] else 0.0, disabled=not vol_activos[i], key=f"p{i}", on_change=reset_state))
        with colC3: cargas_F.append(st.number_input(f"Fuerza Horizontal (F{i})", value=8.0 if i==3 else (6.0 if i==2 else 4.0), key=f"f{i}", on_change=reset_state))

    if st.button("Resolver Ejercicio 1"):
        st.session_state.resuelto = True

    if st.session_state.resuelto:
        st.write("---")
        K_cols = [coef_cols[i] / H_niveles[i] for i in range(num_niveles)]
        K_viga = coef_viga / L_viga
        
        nodos_activos = ['A', 'H', 'B', 'G']
        barras_activas = [('A','B'), ('B','G'), ('G','H')]
        orden_miembros = ['AB', 'BA', 'BG', 'GB', 'GH', 'HG']
        if vol_activos[1]: orden_miembros.insert(4, 'GVol')
        
        if num_niveles >= 2:
            nodos_activos.extend(['C', 'F'])
            barras_activas.extend([('B','C'), ('C','F'), ('F','G')])
            orden_miembros.extend(['BC', 'CB', 'CF', 'FC', 'FG', 'GF'])
            if vol_activos[2]: orden_miembros.insert(orden_miembros.index('FC')+1, 'FVol')
            
        if num_niveles == 3:
            nodos_activos.extend(['D', 'E'])
            barras_activas.extend([('C','D'), ('D','E'), ('E','F')])
            orden_miembros.extend(['CD', 'DC', 'DE', 'ED', 'EF', 'FE'])
            if vol_activos[3]: orden_miembros.insert(orden_miembros.index('ED')+1, 'EVol')

        nodos_fd = {n: {} for n in nodos_activos}
        nodos_fd['A']['B'] = 0; nodos_fd['H']['G'] = 0
        S_nodos = {n: 0.0 for n in nodos_activos if n not in ['A','H']}
        
        nodos_fd['B']['A'] = K_cols[0]; nodos_fd['B']['G'] = K_viga; S_nodos['B'] += K_cols[0] + K_viga
        nodos_fd['G']['H'] = K_cols[0]; nodos_fd['G']['B'] = K_viga; S_nodos['G'] += K_cols[0] + K_viga
        if num_niveles >= 2:
            nodos_fd['B']['C'] = K_cols[1]; S_nodos['B'] += K_cols[1]
            nodos_fd['G']['F'] = K_cols[1]; S_nodos['G'] += K_cols[1]
            nodos_fd['C']['B'] = K_cols[1]; nodos_fd['C']['F'] = K_viga; S_nodos['C'] += K_cols[1] + K_viga
            nodos_fd['F']['G'] = K_cols[1]; nodos_fd['F']['C'] = K_viga; S_nodos['F'] += K_cols[1] + K_viga
            if num_niveles == 3:
                nodos_fd['C']['D'] = K_cols[2]; S_nodos['C'] += K_cols[2]
                nodos_fd['F']['E'] = K_cols[2]; S_nodos['F'] += K_cols[2]
                nodos_fd['D']['C'] = K_cols[2]; nodos_fd['D']['E'] = K_viga; S_nodos['D'] += K_cols[2] + K_viga
                nodos_fd['E']['F'] = K_cols[2]; nodos_fd['E']['D'] = K_viga; S_nodos['E'] += K_cols[2] + K_viga

        st.markdown("### Paso 1: Factores de Distribución (FD)")
        col_fd1, col_fd2 = st.columns(2)
        for n in S_nodos:
            for adj in nodos_fd[n]:
                val = nodos_fd[n][adj] / S_nodos[n]
                nodos_fd[n][adj] = val
                col_fd1.latex(rf"FD_{{{n}{adj}}} = \frac{{K_{{{n}{adj}}}}}{{ \Sigma K_{{{n}}} }} = {val:.3f}")
            if vol_activos.get(list(S_nodos.keys()).index(n)//2 + 1, False) and n in ['G','F','E']:
                nodos_fd[n]['Vol'] = 0.0
                col_fd2.latex(rf"FD_{{{n}Vol}} = 0.000 \text{{ (Voladizo determinado)}}")

        st.markdown("### Paso 2: Momentos de Empotramiento (FEM)")
        FEM_0 = {}
        vigas_por_nivel = {1: ('B','G'), 2: ('C','F'), 3: ('D','E')}
        for i in range(1, num_niveles + 1):
            n1, n2 = vigas_por_nivel[i]
            w = cargas_W[i-1]
            val = (w * L_viga**2)/12
            FEM_0[f"{n1}{n2}"] = -val; FEM_0[f"{n2}{n1}"] = val
            st.latex(rf"FEM_{{{n1}{n2}}} = -\frac{{w L^2}}{{12}} = -\frac{{{w}({L_viga})^2}}{{12}} = {-val:.2f}")
            
            if vol_activos[i]:
                P = cargas_P[i-1]; Lvol = L_vols[i]
                val_vol = -(w * Lvol**2)/2 - P*Lvol
                FEM_0[f"{n2}Vol"] = val_vol
                st.latex(rf"FEM_{{{n2}Vol}} = -\frac{{w L_{{vol}}^2}}{{2}} - P L_{{vol}} = {val_vol:.2f}")

        FEM_sway = []
        cols_por_nivel = {1: ('AB','BA','GH','HG'), 2: ('BC','CB','FG','GF'), 3: ('CD','DC','EF','FE')}
        for i in range(1, num_niveles + 1):
            f_sway = {c: -100.0 for c in cols_por_nivel[i]}
            FEM_sway.append(f_sway)

        st.markdown("### Paso 3: Tablas de Hardy Cross")
        df_0, M0 = tabla_cross_paso_a_paso(nodos_fd, barras_activas, FEM_0, 10, orden_miembros, "M. INICIAL")
        with st.expander("Tabla Estado 0 (Cargas Reales)", expanded=True): st.dataframe(df_0.round(3))
            
        M_sways = []
        for i in range(num_niveles):
            df_sway, m_sway = tabla_cross_paso_a_paso(nodos_fd, barras_activas, FEM_sway[i], 10, orden_miembros, f"M. Ladeo {i+1}")
            M_sways.append(m_sway)
            with st.expander(f"Tabla Estado Ladeo Nivel {i+1}"): st.dataframe(df_sway.round(3))

        st.markdown("### Paso 4: Ecuaciones de Cortante (Ladeos)")
        def calc_cortante_nivel(M_dict, nivel):
            c = cols_por_nivel[nivel]
            return -(M_dict.get(c[0],0) + M_dict.get(c[1],0) + M_dict.get(c[2],0) + M_dict.get(c[3],0)) / H_niveles[nivel-1]

        V_matrix = np.zeros((num_niveles, num_niveles)); F_vector = np.zeros(num_niveles)
        for fila in range(num_niveles):
            corte_idx = fila + 1
            f_actuante = sum(cargas_F[corte_idx-1:])
            v0 = calc_cortante_nivel(M0, corte_idx)
            F_vector[fila] = f_actuante - v0
            st.latex(rf"\Sigma F_{{x, nivel {corte_idx}}} = 0 \Rightarrow {f_actuante} - V_{{0, {corte_idx}}} = {F_vector[fila]:.3f}")
            for col in range(num_niveles):
                V_matrix[fila, col] = calc_cortante_nivel(M_sways[col], corte_idx)
                
        X_correcciones = np.linalg.solve(V_matrix, F_vector)
        for i, x in enumerate(X_correcciones):
            st.latex(rf"X_{i+1} = {x:.4f}")

        st.markdown("### Paso 5: Momentos Finales")
        M_final = {k: M0.get(k,0) for k in orden_miembros}
        for i in range(num_niveles):
            for k in orden_miembros: M_final[k] += X_correcciones[i] * M_sways[i].get(k, 0)
        df_final = pd.DataFrame([M_final], index=['Mom. Final'])
        df_final.columns = pd.MultiIndex.from_tuples([(m[0], m) for m in orden_miembros], names=['NODO', 'MIEMBRO'])
        st.dataframe(df_final.astype(float).round(4))

        st.markdown("### Paso 6: Reacciones Finales")
        R_Ax = -(M_final.get('AB',0) + M_final.get('BA',0)) / H_niveles[0]
        R_Hx = -(M_final.get('GH',0) + M_final.get('HG',0)) / H_niveles[0]
        
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("**En el Apoyo A (Izquierda):**")
            st.latex(rf"M_A = M_{{AB}} = {M_final.get('AB',0):.3f} \text{{ {u_fuerza}-{u_long}}}")
            st.latex(rf"R_{{Ax}} = {R_Ax:.3f} \text{{ {u_fuerza}}}")
        with col_r2:
            st.markdown("**En el Apoyo H (Derecha):**")
            st.latex(rf"M_H = M_{{HG}} = {M_final.get('HG',0):.3f} \text{{ {u_fuerza}-{u_long}}}")
            st.latex(rf"R_{{Hx}} = {R_Hx:.3f} \text{{ {u_fuerza}}}")

        st.markdown("### Paso 7: Diagramas de Cortante y Momento")
        fig_res, (ax_v, ax_m) = plt.subplots(1, 2, figsize=(16, max(6, num_niveles*3)))
        coords = {'A':(0,0), 'H':(L_viga, 0)}
        y_acum = 0
        for i in range(1, num_niveles + 1):
            y_acum += H_niveles[i-1]
            nI, nD = vigas_por_nivel[i]
            coords[nI] = (0, y_acum); coords[nD] = (L_viga, y_acum)
            if vol_activos[i]: coords[f"{nD}Vol"] = (L_viga + L_vols[i], y_acum)

        max_m = max([abs(v) for v in M_final.values() if not np.isnan(v)], default=0.0)
        scale_m = (L_viga * 0.15) / max_m if max_m > 1e-4 else 1.0

        ax_m.set_title("Momento Flexionante")
        draw_support(ax_m, coords['A'][0], coords['A'][1]); draw_support(ax_m, coords['H'][0], coords['H'][1])
        for c in barras_activas:
            v1, v2 = M_final.get(c[0]+c[1], 0), M_final.get(c[1]+c[0], 0)
            draw_diagram_member(ax_m, coords[c[0]][0], coords[c[0]][1], coords[c[1]][0], coords[c[1]][1], v1, -v2, scale_m, color='purple', label1=f"{v1:.2f}", label2=f"{v2:.2f}")
        for i in range(1, num_niveles + 1):
            if vol_activos[i]:
                nD = vigas_por_nivel[i][1]
                v1 = M_final.get(f"{nD}Vol", 0)
                draw_diagram_member(ax_m, coords[nD][0], coords[nD][1], coords[f"{nD}Vol"][0], coords[f"{nD}Vol"][1], v1, 0, scale_m, color='purple', label1=f"{v1:.2f}")

        ax_v.set_title("Fuerza Cortante")
        draw_support(ax_v, coords['A'][0], coords['A'][1]); draw_support(ax_v, coords['H'][0], coords['H'][1])
        V_vals = {}
        for i in range(1, num_niveles + 1):
            nI, nD = vigas_por_nivel[i]
            w_val = cargas_W[i-1]
            v_izq = w_val*L_viga/2.0 - (M_final.get(nI+nD,0) + M_final.get(nD+nI,0))/L_viga
            v_der = -w_val*L_viga/2.0 - (M_final.get(nI+nD,0) + M_final.get(nD+nI,0))/L_viga
            V_vals[nI+nD] = (v_izq, v_der)
            if vol_activos[i]:
                P_val = cargas_P[i-1]
                V_vals[f"{nD}Vol"] = (w_val*L_vols[i] + P_val, P_val)
        for i in range(1, num_niveles + 1):
            h = H_niveles[i-1]; c = cols_por_nivel[i]
            for col in [c[:2], c[2:]]:
                v_izq = -(M_final.get(col[0],0) + M_final.get(col[1],0))/h
                V_vals[col[0]] = (v_izq, v_izq)

        max_v = max([abs(v[0]) for v in V_vals.values()] + [abs(v[1]) for v in V_vals.values()], default=0.0)
        scale_v = (L_viga * 0.15) / max_v if max_v > 1e-4 else 1.0

        for mem, (v_izq, v_der) in V_vals.items():
            if "Vol" in mem:
                nD = mem.replace("Vol","")
                draw_diagram_member(ax_v, coords[nD][0], coords[nD][1], coords[mem][0], coords[mem][1], v_izq, v_der, scale_v, color='green', label1=f"{v_izq:.2f}", label2=f"{v_der:.2f}")
            else:
                n1, n2 = mem[0], mem[1]
                draw_diagram_member(ax_v, coords[n1][0], coords[n1][1], coords[n2][0], coords[n2][1], v_izq, v_der, scale_v, color='green', label1=f"{v_izq:.2f}", label2=f"{abs(v_der):.2f}")

        for ax in (ax_v, ax_m): ax.set_xticks([]); ax.set_yticks([]); ax.axis('off')
        st.pyplot(fig_res, clear_figure=True)
        plt.close(fig_res)

# ==============================================================================
# EJERCICIO 2: MARCO INCLINADO INTERACTIVO
# ==============================================================================
elif tipo_ejercicio == "Ejercicio 2: Marco Inclinado":
    st.header("Marco con Pata Inclinada Asimétrica")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Geometría y Cargas")
        h_ab = st.number_input("Altura Columna AB", value=4.0, on_change=reset_state)
        l_bc = st.number_input("Longitud Viga BC", value=5.0, on_change=reset_state)
        h_cd = st.number_input("Altura Columna CD (Vertical)", value=4.0, on_change=reset_state)
        
        tipo_cd = st.radio("Definir inclinación CD mediante:", ["Ángulo", "Distancia Horizontal Base"])
        if tipo_cd == "Ángulo":
            angulo_d = st.number_input("Ángulo de inclinación CD (grados)", value=55.0, min_value=10.0, max_value=85.0, on_change=reset_state)
            base_d = h_cd / math.tan(math.radians(angulo_d))
        else:
            base_d = st.number_input("Distancia horizontal CD (Base)", value=2.8, min_value=0.1, on_change=reset_state)
            angulo_d = math.degrees(math.atan(h_cd / base_d))
            
        w = st.number_input(f"Carga distribuida w en BC ({u_fuerza}/{u_long})", value=12.0, on_change=reset_state)
        P = st.number_input(f"Carga horizontal P en Nodo B ({u_fuerza})", value=20.0, on_change=reset_state)
    with col2:
        st.subheader("Rigideces (I)")
        i_ab = st.number_input("I de AB", value=1.0, on_change=reset_state)
        i_bc = st.number_input("I de BC", value=2.0, on_change=reset_state)
        i_cd = st.number_input("I de CD", value=1.0, on_change=reset_state)

    # Cálculos derivados
    L_cd = math.hypot(base_d, h_cd)
    y_d = h_ab - h_cd  # Elevación (y) del apoyo D respecto al apoyo A
    
    # Dibujar marco dinámico
    fig_ref, ax_ref = plt.subplots(figsize=(8, 5))
    ax_ref.plot([0, 0], [0, h_ab], 'b-', lw=2) # Columna AB
    ax_ref.plot([0, l_bc], [h_ab, h_ab], 'b-', lw=2) # Viga BC
    ax_ref.plot([l_bc, l_bc + base_d], [h_ab, y_d], 'b-', lw=2) # Pata inclinada CD
    
    # Soportes fijos
    draw_support(ax_ref, 0, 0, is_pin=False)
    draw_support(ax_ref, l_bc + base_d, y_d, is_pin=False)
    
    # Etiquetas de Nodos e Inercias
    ax_ref.text(-0.4, 0, 'A', weight='bold')
    ax_ref.text(-0.4, h_ab + 0.2, 'B', weight='bold')
    ax_ref.text(l_bc + 0.2, h_ab + 0.2, 'C', weight='bold')
    ax_ref.text(l_bc + base_d + 0.3, y_d, 'D', weight='bold')
    
    ax_ref.text(0.2, h_ab/2, f'I={i_ab}', color='teal', weight='bold')
    ax_ref.text(l_bc/2, h_ab + 0.2, f'I={i_bc}', color='teal', weight='bold')
    ax_ref.text(l_bc + base_d/2 + 0.3, h_ab - h_cd/2, f'I={i_cd}', color='teal', weight='bold')
    
    # Acotación de la base CD
    ax_ref.annotate('', xy=(l_bc, y_d - 0.6), xytext=(l_bc + base_d, y_d - 0.6), arrowprops=dict(arrowstyle='<->', color='green'))
    ax_ref.text(l_bc + base_d/2, y_d - 1.0, f'L base = {base_d:.2f}', color='green', ha='center', weight='bold')
    ax_ref.plot([l_bc, l_bc], [h_ab, y_d - 0.8], 'k--', lw=1, alpha=0.3)
    ax_ref.plot([l_bc + base_d, l_bc + base_d], [y_d, y_d - 0.8], 'k--', lw=1, alpha=0.3)
    
    # Dibujar el arco del ángulo
    radio_arco = min(h_cd, base_d) * 0.4
    arc = patches.Arc((l_bc + base_d, y_d), radio_arco*2, radio_arco*2, angle=0, theta1=180-angulo_d, theta2=180, color='green', lw=2)
    ax_ref.add_patch(arc)
    ax_ref.text(l_bc + base_d - radio_arco - 0.2, y_d + 0.2, f"{angulo_d:.1f}°", color='green', weight='bold')

    if w > 0:
        draw_distributed_load(ax_ref, 0, l_bc, h_ab, f"w={w}")
    if P != 0:
        ax_ref.annotate('', xy=(0, h_ab), xytext=(-1.5, h_ab), arrowprops=dict(arrowstyle='->', color='purple', lw=2))
        ax_ref.text(-1.7, h_ab, f'P={P}', color='purple', ha='right', va='center', weight='bold')
    
    y_min_plot = min(0, y_d) - 1.5
    y_max_plot = max(h_ab, y_d) + 2
    ax_ref.set_xlim(-2.5, l_bc + base_d + 2); ax_ref.set_ylim(y_min_plot, y_max_plot); ax_ref.axis('off')
    st.pyplot(fig_ref, clear_figure=True); plt.close(fig_ref)
        
    if st.button("Resolver Ejercicio 2"):
        st.write("---")
        st.markdown("### Paso 1: Factores de Distribución")
        k_ab = i_ab / h_ab; k_bc = i_bc / l_bc; k_cd = i_cd / L_cd 
        sum_B = k_ab + k_bc; sum_C = k_bc + k_cd
        fd_ba = k_ab / sum_B; fd_bc = k_bc / sum_B
        fd_cb = k_bc / sum_C; fd_cd = k_cd / sum_C
        
        colfd1, colfd2 = st.columns(2)
        colfd1.latex(rf"FD_{{BA}} = {fd_ba:.3f}, \quad FD_{{BC}} = {fd_bc:.3f}")
        colfd2.latex(rf"FD_{{CB}} = {fd_cb:.3f}, \quad FD_{{CD}} = {fd_cd:.3f}")
        
        st.markdown("### Paso 2: Momentos de Empotramiento (Estado 0)")
        fem_bc = -(w * l_bc**2)/12; fem_cb = (w * l_bc**2)/12
        fem_ab = 0.0; fem_ba = 0.0; fem_cd = 0.0; fem_dc = 0.0
        FEM_0 = {'AB': fem_ab, 'BA': fem_ba, 'BC': fem_bc, 'CB': fem_cb, 'CD': fem_cd, 'DC': fem_dc}
        st.latex(rf"FEM_{{BC}} = -\frac{{w L^2}}{{12}} = {fem_bc:.2f}, \quad FEM_{{AB}} = 0 \text{{ (Carga en nodo)}}")
        
        st.markdown("### Paso 3: Análisis de Ladeo (Sway)")
        fem_sway_ab = -100.0
        # Relación de momentos por deformación relativa tomando en cuenta h_cd y h_ab variables
        fem_sway_cd = fem_sway_ab * (i_cd / i_ab) * (h_cd * h_ab**2) / (L_cd**3)
        FEM_Sway = {'AB': fem_sway_ab, 'BA': fem_sway_ab, 'BC': 0.0, 'CB': 0.0, 'CD': fem_sway_cd, 'DC': fem_sway_cd}
        
        st.latex(rf"M'_{{AB}} = {fem_sway_ab:.1f}, \quad M'_{{CD}} = M'_{{AB}} \left(\frac{{I_{{CD}}}}{{I_{{AB}}}}\right) \left(\frac{{h_{{CD}} h_{{AB}}^2}}{{L_{{CD}}^3}}\right) = {fem_sway_cd:.1f}")

        st.markdown("### Paso 4: Tablas de Hardy Cross")
        nodos_fd = {'A': {'B': 0}, 'B': {'A': fd_ba, 'C': fd_bc}, 'C': {'B': fd_cb, 'D': fd_cd}, 'D': {'C': 0}}
        barras = [('A','B'), ('B','C'), ('C','D')]
        orden_miembros = ['AB', 'BA', 'BC', 'CB', 'CD', 'DC']
        
        df_0, M0 = tabla_cross_paso_a_paso(nodos_fd, barras, FEM_0, 10, orden_miembros, "M. INICIAL")
        df_sway, Msway = tabla_cross_paso_a_paso(nodos_fd, barras, FEM_Sway, 10, orden_miembros, "M. LADEO")
        
        with st.expander("Ver Tabla Estado 0 (Cargas Reales)", expanded=True): st.dataframe(df_0.round(3))
        with st.expander("Ver Tabla Estado Ladeo"): st.dataframe(df_sway.round(3))

        st.markdown("### Paso 5: Ecuación de Cortante Virtual y Momentos Finales")
        def calc_v_eq(M):
            # La ecuación de trabajo virtual ajustada para alturas asimétricas h_ab y h_cd
            return -((M.get('AB',0) + M.get('BA',0))/h_ab + (M.get('CD',0) + M.get('DC',0)) * h_cd / L_cd**2)

        v_eq_0 = calc_v_eq(M0)
        v_eq_sway = calc_v_eq(Msway)
        X = (P - v_eq_0) / v_eq_sway
        
        st.latex(rf"\Sigma F_x \text{{ (Virtual)}} \Rightarrow P = V_{{eq,0}} + X \cdot V_{{eq,sway}}")
        st.latex(rf"X = \frac{{{P} - ({v_eq_0:.3f})}}{{{v_eq_sway:.3f}}} = {X:.4f}")
        
        M_final = {k: M0.get(k,0) + X * Msway.get(k,0) for k in orden_miembros}
        df_final = pd.DataFrame([M_final], index=['Momento Final'])
        df_final.columns = pd.MultiIndex.from_tuples([(m[0], m) for m in orden_miembros], names=['NODO', 'MIEMBRO'])
        st.dataframe(df_final.astype(float).round(4))

        st.markdown("### Paso 6: Reacciones Finales")
        # Por equilibrio general del marco, asumiendo sumatoria de fuerzas horizontales = 0
        R_Ax = -(M_final.get('AB',0) + M_final.get('BA',0)) / h_ab
        R_Dx = -P - R_Ax 
        
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("**En el Apoyo A (Izquierda):**")
            st.latex(rf"M_A = M_{{AB}} = {M_final.get('AB',0):.3f} \text{{ {u_fuerza}-{u_long}}}")
            st.latex(rf"R_{{Ax}} = {R_Ax:.3f} \text{{ {u_fuerza}}}")
        with col_r2:
            st.markdown("**En el Apoyo D (Derecha):**")
            st.latex(rf"M_D = M_{{DC}} = {M_final.get('DC',0):.3f} \text{{ {u_fuerza}-{u_long}}}")
            st.latex(rf"R_{{Dx}} = {R_Dx:.3f} \text{{ {u_fuerza}}}")

        st.markdown("### Paso 7: Diagramas de Cortante y Momento")
        fig_res2, (ax_v2, ax_m2) = plt.subplots(1, 2, figsize=(14, 5))
        coords = {'A': (0,0), 'B': (0, h_ab), 'C': (l_bc, h_ab), 'D': (l_bc + base_d, y_d)}
        
        max_m = max([abs(v) for v in M_final.values() if not np.isnan(v)], default=0.0)
        scale_m = (l_bc * 0.15) / max_m if max_m > 1e-4 else 1.0

        # Diagrama de Momento
        ax_m2.set_title("Momento Flexionante")
        draw_support(ax_m2, coords['A'][0], coords['A'][1]); draw_support(ax_m2, coords['D'][0], coords['D'][1])
        for n1, n2 in barras:
            v1, v2 = M_final.get(n1+n2, 0), M_final.get(n2+n1, 0)
            draw_diagram_member(ax_m2, coords[n1][0], coords[n1][1], coords[n2][0], coords[n2][1], v1, -v2, scale_m, color='purple', label1=f"{v1:.2f}", label2=f"{v2:.2f}")

        # Diagrama de Cortante
        ax_v2.set_title("Fuerza Cortante")
        draw_support(ax_v2, coords['A'][0], coords['A'][1]); draw_support(ax_v2, coords['D'][0], coords['D'][1])
        
        V_vals = {}
        # AB y CD (constante a lo largo de su eje local)
        V_vals['AB'] = (-(M_final['AB'] + M_final['BA'])/h_ab, -(M_final['AB'] + M_final['BA'])/h_ab)
        V_vals['CD'] = (-(M_final['CD'] + M_final['DC'])/L_cd, -(M_final['CD'] + M_final['DC'])/L_cd)
        # BC (tiene carga distribuida w)
        V_vals['BC'] = (w*l_bc/2.0 - (M_final['BC'] + M_final['CB'])/l_bc, -w*l_bc/2.0 - (M_final['BC'] + M_final['CB'])/l_bc)

        max_v = max([abs(v[0]) for v in V_vals.values()] + [abs(v[1]) for v in V_vals.values()], default=0.0)
        scale_v = (l_bc * 0.15) / max_v if max_v > 1e-4 else 1.0

        for mem, (v_izq, v_der) in V_vals.items():
            n1, n2 = mem[0], mem[1]
            draw_diagram_member(ax_v2, coords[n1][0], coords[n1][1], coords[n2][0], coords[n2][1], v_izq, v_der, scale_v, color='green', label1=f"{v_izq:.2f}", label2=f"{v_der:.2f}")

        for ax in (ax_v2, ax_m2): ax.set_xlim(-2, l_bc + base_d + 2); ax.set_ylim(y_min_plot, y_max_plot); ax.set_xticks([]); ax.set_yticks([]); ax.axis('off')
        st.pyplot(fig_res2, clear_figure=True); plt.close(fig_res2)
