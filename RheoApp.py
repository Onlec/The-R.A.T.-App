import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d, UnivariateSpline

# --- CONFIGURATIE ---
st.set_page_config(page_title="TPU Rheology Expert Tool", layout="wide")

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
        if c_reset.button("🔄 Reset"):
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

        st.sidebar.header("2. Handmatige Shift")
        for t in selected_temps:
            st.session_state.shifts[t] = st.sidebar.slider(f"{int(t)}°C", -15.0, 15.0, float(st.session_state.shifts[t]), 0.1, key=f"{t}_{st.session_state.reset_id}")

        color_map = plt.get_cmap(cmap_opt)
        colors = color_map(np.linspace(0, 0.9, len(selected_temps)))
        
        # --- TAB DEFINITIES ---
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Master Curve", "🧪 Structuur (vGP)", "🧬 Thermisch (Ea)", 
            "🔬 TTS Validatie", "💾 Smooth Export", "📊 Summary Dashboard"
        ])

        # Bereken crossovers vooraf voor gebruik in Dashboard en Tab 4
        co_data = []
        for t in selected_temps:
            d = df[df['T_group'] == t].sort_values('omega')
            if len(d) > 3:
                try:
                    f_diff = interp1d(np.log10(d['omega']), np.log10(d['Gp']) - np.log10(d['Gpp']), bounds_error=False)
                    w_range = np.logspace(np.log10(d['omega'].min()), np.log10(d['omega'].max()), 500)
                    diffs = f_diff(np.log10(w_range))
                    idx_zero = np.nanargmin(np.abs(diffs))
                    if np.abs(diffs[idx_zero]) < 0.1:
                        w_co = w_range[idx_zero]
                        g_co = 10**float(interp1d(np.log10(d['omega']), np.log10(d['Gp']))(np.log10(w_co)))
                        co_data.append({
                            "T (°C)": int(t), 
                            "ω_co (rad/s)": round(w_co, 2), 
                            "G_co (Pa)": round(g_co, 0),
                            "λ (s)": round(1/w_co, 4)
                        })
                except: pass
        co_df = pd.DataFrame(co_data)

        # Ea berekening
        t_k = np.array([t + 273.15 for t in selected_temps])
        inv_t, log_at = 1/t_k, np.array([st.session_state.shifts[t] for t in selected_temps])
        slope, intercept = np.polyfit(inv_t, log_at, 1)
        r_squared = 1 - (np.sum((log_at - (slope*inv_t + intercept))**2) / np.sum((log_at - np.mean(log_at))**2))
        ea_val = abs(slope * 8.314 * np.log(10) / 1000)

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
                fig2, ax2 = plt.subplots(); ax2.plot(sorted(selected_temps), [st.session_state.shifts[t] for t in sorted(selected_temps)], 's-', color='red'); ax2.set_ylabel("log(aT)"); st.pyplot(fig2)

        with tab2:
            st.subheader("Van Gurp-Palmen Analyse")
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            for t, color in zip(selected_temps, colors):
                d = df[df['T_group'] == t]; g_star = np.sqrt(d['Gp']**2 + d['Gpp']**2); delta = np.degrees(np.arctan2(d['Gpp'], d['Gp']))
                ax3.plot(g_star, delta, 'o-', color=color)
            ax3.set_xscale('log'); ax3.set_xlabel("|G*| (Pa)"); ax3.set_ylabel("δ (°)"); st.pyplot(fig3)

        with tab3:
            st.subheader("🧬 Thermische Analyse")
            st.metric("Activeringsenergie (Ea)", f"{ea_val:.1f} kJ/mol")
            fig_ea, ax_ea = plt.subplots(); ax_ea.scatter(inv_t, log_at, color='red'); ax_ea.plot(inv_t, slope*inv_t + intercept, 'k--'); st.pyplot(fig_ea)

        with tab4:
            st.subheader("🔬 TTS Validatie Checks")
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Han Plot**")
                fig_h, ax_h = plt.subplots()
                for t, color in zip(selected_temps, colors):
                    d = df[df['T_group'] == t]
                    ax_h.loglog(d['Gpp'], d['Gp'], 'o', color=color, alpha=0.6)
                st.pyplot(fig_h)
            with c2:
                st.write("**Cole-Cole Plot**")
                fig_c, ax_c = plt.subplots()
                for t, color in zip(selected_temps, colors):
                    d = df[df['T_group'] == t]
                    ax_c.plot(d['Gpp']/d['omega'], d['Gp']/d['omega'], 'o-', color=color)
                st.pyplot(fig_c)
            st.write("**Crossover Tabel**")
            st.table(co_df)

        with tab5:
            st.subheader("💾 Smooth Export")
            m_df = pd.concat([df[df['T_group'] == t].assign(w_s=lambda x: x.omega * 10**st.session_state.shifts[t], eta_s=lambda x: np.sqrt(x.Gp**2 + x.Gpp**2)/(x.omega * 10**st.session_state.shifts[t])) for t in selected_temps]).sort_values('w_s')
            s_val = st.slider("Smoothing", 0.0, 2.0, 0.4)
            log_w, log_eta = np.log10(m_df['w_s']), np.log10(m_df['eta_s'])
            spl = UnivariateSpline(log_w, log_eta, s=s_val)
            w_new = np.logspace(log_w.min(), log_w.max(), 50)
            eta_new = 10**spl(np.log10(w_new))
            fig_s, ax_s = plt.subplots(); ax_s.loglog(m_df['w_s'], m_df['eta_s'], 'k.', alpha=0.1); ax_s.loglog(w_new, eta_new, 'r-'); st.pyplot(fig_s)
            st.download_button("Download CSV", pd.DataFrame({'w': w_new, 'eta': eta_new}).to_csv(index=False).encode('utf-8'))

        with tab6:
            st.header("📊 TPU Fingerprint Summary")
            col_s1, col_s2, col_s3 = st.columns(3)
            
            with col_s1:
                st.subheader("🔥 Thermisch")
                st.metric("Ea (Flow)", f"{ea_val:.1f} kJ/mol")
                st.write(f"**Fit Kwaliteit (R²):** {r_squared:.4f}")
                status_ea = "Hoog" if ea_val > 150 else "Normaal" if ea_val > 70 else "Laag"
                st.caption(f"Temperatuurgevoeligheid is **{status_ea}**.")

            with col_s2:
                st.subheader("🧬 Structuur")
                if not co_df.empty:
                    avg_lambda = co_df['λ (s)'].mean()
                    st.metric("Gem. Relaxatietijd", f"{avg_lambda:.4e} s")
                else:
                    st.write("Geen crossovers gevonden.")
                simplicity = "Hoog" if r_squared > 0.99 else "Matig" if r_squared > 0.95 else "Laag (Complex)"
                st.write(f"**TTS Geldigheid:** {simplicity}")

            with col_s3:
                st.subheader("⚙️ Processing")
                if not co_df.empty:
                    st.write(f"**Crossover bereik:**")
                    st.write(f"{co_df['ω_co (rad/s)'].min()} - {co_df['ω_co (rad/s)'].max()} rad/s")
                st.write("**Referentie T:**", f"{ref_temp} °C")

            st.divider()
            st.subheader("📝 Expert Conclusie")
            
            # Dynamische conclusie op basis van data
            if r_squared < 0.95:
                st.error("⚠️ **Thermorheologische Complexiteit gedetecteerd.** De Han-plot en vGP-plot suggereren dat de TPU structuur verandert binnen dit temperatuurbereik. De Master Curve is een indicatie, geen absolute waarheid.")
            else:
                st.success("✅ **TTS is valide.** Het materiaal gedraagt zich als een homogene smelt. De Master Curve kan betrouwbaar worden gebruikt voor simulaties.")
                
            if ea_val > 180:
                st.warning("❗ **Hoge Ea gedetecteerd.** Deze TPU is extreem gevoelig voor temperatuurvariaties tijdens extrusie/spuitgieten. Controleer je heater bands.")
            
            st.markdown("### Samenvattingstabel voor Rapportage")
            summary_table = co_df.copy()
            summary_table['log(aT)'] = [st.session_state.shifts[t] for t in co_df['T (°C)']]
            st.dataframe(summary_table, use_container_width=True)

    else: st.error("Geen data.")
else: st.info("👋 Upload data om te beginnen.")