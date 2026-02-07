import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d, UnivariateSpline

# --- CONFIGURATIE ---
st.set_page_config(page_title="TPU Rheology Expert Tool", layout="wide")

st.markdown("""
    <style>
    [data-testid="stSidebar"] { min-width: 400px; }
    .stAlert { font-size: 0.95rem; }
    </style>
    """, unsafe_allow_html=True)

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
        st.error(f"Encoding error: {e}")
        return pd.DataFrame()
    
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
                if not data_line.strip():
                    i += 1
                    continue
                parts = data_line.split('\t')
                non_empty_parts = [p.strip() for p in parts if p.strip()]
                if len(non_empty_parts) >= 4:
                    row_dict = {clean_headers[idx]: non_empty_parts[idx] for idx in range(len(clean_headers)) if idx < len(non_empty_parts)}
                    if 'Temperature' in row_dict and 'Storage Modulus' in row_dict:
                        all_data.append(row_dict)
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

# --- SIDEBAR ---
st.sidebar.title("🧪 Rheo-Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload Anton Paar CSV/TXT", type=['csv', 'txt'])

if uploaded_file:
    df = load_rheo_data(uploaded_file)
    if not df.empty:
        df['T_group'] = df['T'].round(0)
        temps = sorted(df['T_group'].unique())
        
        st.sidebar.header("1. Selectie & Kleur")
        selected_temps = st.sidebar.multiselect("Temperaturen", temps, default=temps)
        if not selected_temps: st.stop()
        
        ref_temp = st.sidebar.selectbox("Referentie T (°C)", selected_temps, index=len(selected_temps)//2)
        cmap_opt = st.sidebar.selectbox("Kleurenschema", ["coolwarm", "viridis", "magma", "jet"])
        
        if 'shifts' not in st.session_state: st.session_state.shifts = {t: 0.0 for t in temps}
        if 'reset_id' not in st.session_state: st.session_state.reset_id = 0

        c_auto, c_reset = st.sidebar.columns(2)
        if c_reset.button("🔄 Reset Shifts"):
            for t in temps: st.session_state.shifts[t] = 0.0
            st.session_state.reset_id += 1
            st.rerun()

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
            st.session_state.reset_id += 1
            st.rerun()

        st.sidebar.header("2. Handmatige Fine-tuning")
        for t in selected_temps:
            st.session_state.shifts[t] = st.sidebar.slider(f"{int(t)}°C", -10.0, 10.0, float(st.session_state.shifts[t]), 0.1, key=f"{t}_{st.session_state.reset_id}")

        # --- DATA PREP ---
        color_map = plt.get_cmap(cmap_opt)
        colors = color_map(np.linspace(0, 0.9, len(selected_temps)))
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Master Curve", "🧪 Structuur (vGP)", "🧬 Thermisch (Ea)", "🔬 TTS Validatie", "💾 Smooth Export"])

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
                ax1.set_xlabel("ω·aT (rad/s)"); ax1.set_ylabel("Modulus (Pa)"); ax1.legend(ncol=2, fontsize=8); ax1.grid(True, which="both", alpha=0.1)
                st.pyplot(fig1)
            with col_m2:
                st.write("**Shift Factor Trend**")
                t_list = sorted([t for t in selected_temps])
                s_list = [st.session_state.shifts[t] for t in t_list]
                fig2, ax2 = plt.subplots(); ax2.plot(t_list, s_list, 's-', color='red'); ax2.set_xlabel("T (°C)"); ax2.set_ylabel("log(aT)"); st.pyplot(fig2)
                st.info("💡 Een lineaire trend wijst op Arrhenius gedrag; een sterke kromming op WLF (nabij Tg).")

        with tab2:
            st.subheader("Van Gurp-Palmen (vGP) Analyse")
            st.markdown("> **Interpretatie:** Als de curves bij verschillende temperaturen niet samenvallen in deze plot, is het materiaal **thermorheologisch complex**. Voor TPU betekent dit vaak dat de verhouding tussen harde en zachte segmenten verandert (bijv. door het smelten van hard-segment domeinen).")
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            for t, color in zip(selected_temps, colors):
                d = df[df['T_group'] == t]
                g_star = np.sqrt(d['Gp']**2 + d['Gpp']**2)
                delta = np.degrees(np.arctan2(d['Gpp'], d['Gp']))
                ax3.plot(g_star, delta, 'o-', color=color, label=f"{int(t)}°C")
            ax3.set_xscale('log'); ax3.set_xlabel("|G*| (Pa)"); ax3.set_ylabel("δ (°)"); ax3.set_ylim(0, 95); ax3.grid(True, alpha=0.2); st.pyplot(fig3)

        with tab3:
            st.subheader("🧬 Activeringsenergie ($E_a$)")
            all_omegas = sorted(df['omega'].unique())
            target_w = st.select_slider("Selecteer frequentie voor analyse (rad/s)", options=all_omegas, value=all_omegas[len(all_omegas)//2])
            
            t_k = np.array([t + 273.15 for t in selected_temps])
            inv_t, log_at = 1/t_k, np.array([st.session_state.shifts[t] for t in selected_temps])
            
            # Ea berekening
            slope, intercept = np.polyfit(inv_t, log_at, 1)
            ea = abs(slope * 8.314 * np.log(10) / 1000)
            
            c1, c2 = st.columns([2, 1])
            with c1:
                fig_ea, ax_ea = plt.subplots(); ax_ea.scatter(inv_t, log_at, color='red'); ax_ea.plot(inv_t, slope*inv_t + intercept, 'k--')
                ax_ea.set_xlabel("1/T (1/K)"); ax_ea.set_ylabel("log(aT)"); st.pyplot(fig_ea)
            with c2:
                st.metric("Ea (Shift)", f"{ea:.1f} kJ/mol")
                st.warning("⚠️ Bij TPU's: Een plotselinge verandering in Ea kan duiden op het bereiken van de 'Order-Disorder Transition' (ODT).")

        with tab4:
            st.subheader("🔬 TTS Validatie & Crossovers")
            c_v1, c_v2 = st.columns(2)
            with c_v1:
                st.write("**Han Plot** ($G'$ vs $G''$)")
                fig_h, ax_h = plt.subplots(); [ax_h.loglog(df[df['T_group']==t]['Gpp'], df[df['T_group']==t]['Gp'], 'o', color=c, alpha=0.6) for t, c in zip(selected_temps, colors)]; st.pyplot(fig_h)
                st.caption("Gevaar: Als de lijnen spreiden, is TTS wiskundig gezien ongeldig voor dit temperatuurbereik.")
            with c_v2:
                st.write("**Cole-Cole** ($\eta''$ vs $\eta'$)")
                fig_c, ax_c = plt.subplots(); [ax_c.plot(df[df['T_group']==t]['Gpp']/t, df[df['T_group']==t]['Gp']/t, 'o-') for t in selected_temps]; st.pyplot(fig_c)

            st.write("**Cross-over Punten ($G' = G''$)**")
            co_list = []
            for t in selected_temps:
                d = df[df['T_group'] == t].sort_values('omega')
                if len(d) > 2:
                    f = interp1d(np.log10(d['Gp']) - np.log10(d['Gpp']), np.log10(d['omega']), bounds_error=False)
                    try: 
                        w_co = 10**float(f(0))
                        g_co = 10**float(interp1d(np.log10(d['omega']), np.log10(d['Gp']))(np.log10(w_co)))
                        co_list.append({"T (°C)": int(t), "ω_co (rad/s)": round(w_co, 2), "G_co (Pa)": round(g_co, 0)})
                    except: pass
            if co_list: st.table(pd.DataFrame(co_list))

        with tab5:
            st.subheader("💾 Smooth Export & Resampling")
            st.info("Deze functie creëert een ruisvrije curve die geschikt is voor simulatiesoftware.")
            
            # Master Data verzamelen voor spline
            m_list = []
            for t in selected_temps:
                d = df[df['T_group'] == t].copy()
                at = 10**st.session_state.shifts[t]
                d['w_s'] = d['omega'] * at
                d['eta_s'] = np.sqrt(d['Gp']**2 + d['Gpp']**2) / d['w_s']
                m_list.append(d)
            m_df = pd.concat(m_list).sort_values('w_s')

            s_val = st.slider("Smoothing Sterkte (0 = ruw, 1 = glad)", 0.0, 2.0, 0.4, 0.1)
            res = st.slider("Aantal datapunten", 20, 100, 50)
            
            log_w, log_eta = np.log10(m_df['w_s']), np.log10(m_df['eta_s'])
            spl = UnivariateSpline(log_w, log_eta, s=s_val)
            w_new = np.logspace(log_w.min(), log_w.max(), res)
            eta_new = 10**spl(np.log10(w_new))
            
            fig_s, ax_s = plt.subplots(figsize=(10, 4))
            ax_s.loglog(m_df['w_s'], m_df['eta_s'], 'k.', alpha=0.2, label="Origineel")
            ax_s.loglog(w_new, eta_new, 'r-', label="Smooth")
            ax_s.set_ylabel("η* (Pa·s)"); ax_s.set_xlabel("ω·aT (rad/s)"); ax_s.legend(); st.pyplot(fig_s)
            
            csv = pd.DataFrame({'omega_shifted': w_new, 'eta_complex': eta_new}).to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Smooth Master Curve", csv, "smooth_export.csv", "text/csv")