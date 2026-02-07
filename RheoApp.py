import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d, UnivariateSpline
from io import BytesIO

# --- CONFIGURATIE & STYLING ---
st.set_page_config(page_title="TPU Rheology Expert Tool", layout="wide")

# Custom CSS voor betere leesbaarheid van expert-notes
st.markdown("""
    <style>
    .reportview-container .main .block-container { padding-top: 2rem; }
    .expert-note { background-color: #f0f2f6; padding: 15px; border-left: 5px solid #ff4b4b; border-radius: 5px; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- FUNCTIES ---
def load_rheo_data(file):
    try:
        file.seek(0)
        raw_bytes = file.read()
        if raw_bytes[:2] == b'\xff\xfe': decoded_text = raw_bytes.decode('utf-16-le')
        elif raw_bytes[:3] == b'\xef\xbb\xbf': decoded_text = raw_bytes.decode('utf-8-sig')
        else:
            try: decoded_text = raw_bytes.decode('latin-1')
            except: decoded_text = raw_bytes.decode('utf-8')
    except Exception as e:
        st.error(f"Encoding error: {e}"); return pd.DataFrame()
    
    lines = decoded_text.splitlines()
    all_data = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if 'Interval data:' in line and 'Point No.' in line and 'Storage Modulus' in line:
            header_parts = line.split('\t')
            clean_headers = [p.strip() for p in header_parts if p.strip() and p.strip() != 'Interval data:']
            i += 3
            while i < len(lines):
                data_line = lines[i]
                if 'Result:' in data_line or 'Interval data:' in data_line: break
                if not data_line.strip(): i += 1; continue
                parts = data_line.split('\t')
                non_empty_parts = [p.strip() for p in parts if p.strip()]
                if len(non_empty_parts) >= 4:
                    row_dict = {clean_headers[idx]: non_empty_parts[idx] for idx in range(len(clean_headers)) if idx < len(non_empty_parts)}
                    if 'Temperature' in row_dict and 'Storage Modulus' in row_dict: all_data.append(row_dict)
                i += 1
        else: i += 1
    
    if not all_data: return pd.DataFrame()
    df = pd.DataFrame(all_data)
    df = df.rename(columns={'Temperature': 'T', 'Angular Frequency': 'omega', 'Storage Modulus': 'Gp', 'Loss Modulus': 'Gpp'})
    
    def safe_float(val):
        try: return float(str(val).replace(',', '.'))
        except: return np.nan
    
    for col in ['T', 'omega', 'Gp', 'Gpp']:
        if col in df.columns: df[col] = df[col].apply(safe_float)
    
    return df.dropna(subset=['T', 'omega', 'Gp']).query("Gp > 0 and omega > 0")

def to_excel(summary_df, shift_df, crossover_df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        shift_df.to_excel(writer, sheet_name='ShiftFactors', index=False)
        crossover_df.to_excel(writer, sheet_name='Crossovers', index=False)
    return output.getvalue()

# --- SIDEBAR ---
st.sidebar.title("🧪 Rheo-Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload Anton Paar CSV/TXT", type=['csv', 'txt'])

if uploaded_file:
    df = load_rheo_data(uploaded_file)
    if not df.empty:
        df['T_group'] = df['T'].round(0)
        temps = sorted(df['T_group'].unique())
        
        selected_temps = st.sidebar.multiselect("Selecteer Temperaturen", temps, default=temps)
        ref_temp = st.sidebar.selectbox("Referentie T (°C)", selected_temps, index=len(selected_temps)//2)
        cmap_opt = st.sidebar.selectbox("Kleurenschema", ["coolwarm", "viridis", "magma", "jet"])
        
        if 'shifts' not in st.session_state: st.session_state.shifts = {t: 0.0 for t in temps}
        if 'reset_id' not in st.session_state: st.session_state.reset_id = 0

        c_auto, c_reset = st.sidebar.columns(2)
        if c_reset.button("🔄 Reset"):
            for t in temps: st.session_state.shifts[t] = 0.0
            st.session_state.reset_id += 1; st.rerun()

        if c_auto.button("🚀 Auto-Align"):
            for t in selected_temps:
                if t == ref_temp: continue
                def objective(log_at):
                    ref_d, tgt_d = df[df['T_group'] == ref_temp], df[df['T_group'] == t]
                    f = interp1d(np.log10(ref_d['omega']), np.log10(ref_d['Gp']), bounds_error=False)
                    v = f(np.log10(tgt_d['omega']) + log_at)
                    m = ~np.isnan(v)
                    return np.sum((v[m] - np.log10(tgt_d['Gp'].values[m]))**2) if np.sum(m) >= 2 else 9999
                res = minimize(objective, x0=st.session_state.shifts[t], method='Nelder-Mead')
                st.session_state.shifts[t] = round(float(res.x[0]), 2)
            st.session_state.reset_id += 1; st.rerun()

        for t in selected_temps:
            st.session_state.shifts[t] = st.sidebar.slider(f"{int(t)}°C", -15.0, 15.0, float(st.session_state.shifts[t]), 0.1, key=f"{t}_{st.session_state.reset_id}")

        # --- DATA PREP ---
        color_map = plt.get_cmap(cmap_opt)
        colors = color_map(np.linspace(0, 0.9, len(selected_temps)))
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Master Curve", "🧪 Structuur", "🧬 Thermisch (Ea/WLF)", "🔬 Validatie", "💾 Export", "📊 Dashboard"])

        with tab1:
            st.subheader(f"Master Curve bij {ref_temp}°C")
            col_m1, col_m2 = st.columns([2, 1])
            with col_m1:
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                for t, color in zip(selected_temps, colors):
                    d = df[df['T_group'] == t].copy()
                    at = 10**st.session_state.shifts[t]
                    ax1.loglog(d['omega']*at, d['Gp'], 'o-', color=color, label=f"{int(t)}°C G'", markersize=4)
                    ax1.loglog(d['omega']*at, d['Gpp'], 'x--', color=color, alpha=0.3, markersize=3)
                ax1.set_xlabel("ω·aT (rad/s)"); ax1.set_ylabel("Modulus (Pa)"); ax1.legend(ncol=2, fontsize=8); ax1.grid(True, alpha=0.1)
                st.pyplot(fig1)
            with col_m2:
                st.write("**Shift Factor Trend**")
                t_plot = sorted(selected_temps)
                s_plot = [st.session_state.shifts[t] for t in t_plot]
                fig2, ax2 = plt.subplots(); ax2.plot(t_plot, s_plot, 's-', color='red'); ax2.set_xlabel("T (°C)"); ax2.set_ylabel("log(aT)"); st.pyplot(fig2)
                st.info("💡 Een lineaire trend wijst op Arrhenius gedrag; een sterke kromming op WLF.")

        with tab2:
            st.subheader("Van Gurp-Palmen (vGP) Analyse")
            st.markdown("""
            > **Expert Interpretatie:** Deze plot is de 'vingerafdruk' van de structuur. 
            > * **Overlappende lijnen:** Thermorheologisch eenvoudig (homogene smelt).
            > * **Spreiding van lijnen:** Thermorheologisch complex. Bij TPU duidt dit vaak op het smelten van hard-segment domeinen of fase-veranderingen.
            """)
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            for t, color in zip(selected_temps, colors):
                d = df[df['T_group'] == t]
                g_star = np.sqrt(d['Gp']**2 + d['Gpp']**2)
                delta = np.degrees(np.arctan2(d['Gpp'], d['Gp']))
                ax3.plot(g_star, delta, 'o-', color=color, label=f"{int(t)}°C")
            ax3.set_xscale('log'); ax3.set_xlabel("|G*| (Pa)"); ax3.set_ylabel("δ (°)"); ax3.grid(True, alpha=0.2); st.pyplot(fig3)

        with tab3:
            st.subheader("🧬 Arrhenius vs WLF Vergelijking")
            
            t_k = np.array([t + 273.15 for t in selected_temps])
            log_at = np.array([st.session_state.shifts[t] for t in selected_temps])
            tr_k = ref_temp + 273.15
            
            # Arrhenius
            inv_t = 1/t_k
            slope, intercept = np.polyfit(inv_t, log_at, 1)
            ea = abs(slope * 8.314 * np.log(10) / 1000)
            
            # WLF Fit
            def wlf_func(p, t, tr): return -p[0]*(t-tr) / (p[1] + (t-tr))
            def wlf_err(p): return np.sum((log_at - wlf_func(p, t_k, tr_k))**2)
            res_wlf = minimize(wlf_err, x0=[17, 50])
            c1, c2 = res_wlf.x

            col_t1, col_t2 = st.columns([2, 1])
            with col_t1:
                fig_t, ax_t = plt.subplots()
                ax_t.scatter(selected_temps, log_at, color='black', label='Data')
                ax_t.plot(selected_temps, slope*(1/(np.array(selected_temps)+273.15)) + intercept, 'r--', label='Arrhenius Fit')
                ax_t.plot(selected_temps, wlf_func([c1, c2], t_k, tr_k), 'b-', label='WLF Fit')
                ax_t.set_xlabel("T (°C)"); ax_t.set_ylabel("log(aT)"); ax_t.legend(); st.pyplot(fig_t)
            
            with col_t2:
                st.metric("Ea (Arrhenius)", f"{ea:.1f} kJ/mol")
                st.write(f"**WLF C1:** {c1:.2f}")
                st.write(f"**WLF C2:** {c2:.2f}")
                st.warning("""
                **Welke te volgen?**
                * Gebruik **Arrhenius** ($E_a$) als de TPU ver boven $T_g$ is (meestal in de smelt).
                * Gebruik **WLF** als je dicht bij de glasovergang meet ($T_g < T < T_g + 100^\circ\text{C}$).
                """)

        with tab4:
            st.subheader("🔬 Geavanceerde TTS Validatie")
            cv1, cv2 = st.columns(2)
            with cv1:
                st.write("**1. Han Plot ($G'$ vs $G''$)**")
                fig_h, ax_h = plt.subplots()
                for t, color in zip(selected_temps, colors):
                    d = df[df['T_group'] == t]
                    ax_h.loglog(d['Gpp'], d['Gp'], 'o', color=color, alpha=0.6)
                ax_h.set_xlabel("G'' (Pa)"); ax_h.set_ylabel("G' (Pa)"); st.pyplot(fig_h)
                st.caption("Gevaar: Als de lijnen spreiden, verandert de morfologie en is TTS ongeldig.")
            
            with cv2:
                st.write("**2. Cole-Cole Plot ($\eta''$ vs $\eta'$)**")
                fig_c, ax_c = plt.subplots()
                for t, color in zip(selected_temps, colors):
                    d = df[df['T_group'] == t]
                    ax_c.plot(d['Gpp']/d['omega'], d['Gp']/d['omega'], 'o-', color=color)
                ax_c.set_xlabel("η' (Pa·s)"); ax_c.set_ylabel("η'' (Pa·s)"); st.pyplot(fig_c)
                st.caption("Interpretatie: Een afgeplatte boog duidt op een brede molecuulgewichtsverdeling (MWD).")

        with tab5:
            st.subheader("💾 Smooth Export")
            # Spline logic
            m_list = []
            for t in selected_temps:
                d = df[df['T_group'] == t].copy()
                at = 10**st.session_state.shifts[t]
                d['w_s'] = d['omega'] * at
                d['eta_s'] = np.sqrt(d['Gp']**2 + d['Gpp']**2) / d['w_s']
                m_list.append(d)
            m_df = pd.concat(m_list).sort_values('w_s')
            s_val = st.slider("Smoothing Sterkte", 0.0, 2.0, 0.4)
            log_w, log_eta = np.log10(m_df['w_s']), np.log10(m_df['eta_s'])
            spl = UnivariateSpline(log_w, log_eta, s=s_val)
            w_new = np.logspace(log_w.min(), log_w.max(), 50)
            eta_new = 10**spl(np.log10(w_new))
            
            fig_s, ax_s = plt.subplots(); ax_s.loglog(m_df['w_s'], m_df['eta_s'], 'k.', alpha=0.1); ax_s.loglog(w_new, eta_new, 'r-'); st.pyplot(fig_s)
            
            csv = pd.DataFrame({'omega_shifted': w_new, 'eta_complex': eta_new}).to_csv(index=False).encode('utf-8')
            st.download_button("Download Smooth CSV", csv, "mastercurve.csv")

        with tab6:
            st.header("📊 TPU Expert Dashboard")
            
            # Crossover berekening
            co_data = []
            for t in selected_temps:
                d = df[df['T_group'] == t].sort_values('omega')
                if len(d) > 3:
                    try:
                        f_diff = interp1d(np.log10(d['omega']), np.log10(d['Gp']) - np.log10(d['Gpp']), bounds_error=False)
                        w_range = np.logspace(np.log10(d['omega'].min()), np.log10(d['omega'].max()), 500)
                        idx = np.nanargmin(np.abs(f_diff(np.log10(w_range))))
                        w_co = w_range[idx]
                        co_data.append({"T_C": int(t), "w_co": round(w_co, 2), "Lambda_s": round(1/w_co, 4)})
                    except: pass
            co_df = pd.DataFrame(co_data)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Flow Activation (Ea)", f"{ea:.1f} kJ/mol")
            if not co_df.empty:
                c2.metric("Max Relaxatietijd", f"{co_df['Lambda_s'].max():.2e} s")
            c3.metric("TTS Fit (R²)", f"{1 - (np.sum((log_at-(slope*inv_t+intercept))**2)/np.sum((log_at-np.mean(log_at))**2)):.3f}")
            
            st.divider()
            st.subheader("Crossover & Relaxatie Tabel")
            st.dataframe(co_df, use_container_width=True)
            
            # Excel Download
            summ_df = pd.DataFrame({'Parameter': ['Ea', 'WLF_C1', 'WLF_C2', 'Ref_T'], 'Waarde': [ea, c1, c2, ref_temp]})
            shift_df = pd.DataFrame({'Temp': selected_temps, 'log_at': [st.session_state.shifts[t] for t in selected_temps]})
            
            st.download_button(
                "📥 Download Geformatteerd Excel Rapport",
                data=to_excel(summ_df, shift_df, co_df),
                file_name="TPU_Rheology_Report.xlsx"
            )