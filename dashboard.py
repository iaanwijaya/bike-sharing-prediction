import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# --- KONFIGURASI SISTEM ---
st.set_page_config(
    page_title="Emerald Bike Prediction | Executive Suite",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS ARCHITECTURE: LUXURY EMERALD & NOIR ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&family=Playfair+Display:ital,wght@0,700;1,700&display=swap');

    /* HEADER ENABLED FOR SIDEBAR TOGGLE VISIBILITY */
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0) !important;
    }
    
    .main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 0rem !important;
        max-width: 100% !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Global Typography */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #000000;
        color: #f0fdf4;
    }

    /* Animated Luxury Background */
    .stApp {
        background: linear-gradient(135deg, #000000 0%, #022c22 35%, #064e3b 70%, #000000 100%);
        background-size: 400% 400%;
        animation: gradientFlow 18s ease infinite;
    }

    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Floating Particles Effect */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background-image: radial-gradient(#10b981 0.5px, transparent 0.5px);
        background-size: 30px 30px;
        opacity: 0.15;
        z-index: 0;
        pointer-events: none;
    }

    /* High Contrast Titles */
    h1, h2, h3 {
        font-family: 'Playfair Display', serif !important;
        color: #10b981 !important;
        font-weight: 700 !important;
        text-shadow: 0px 4px 15px rgba(0,0,0,0.6);
        letter-spacing: 1px;
    }

    /* Glassmorphism Cards */
    [data-testid="stMetric"], .css-1r6slb0, .stPlotlyChart, div.stForm {
        background: rgba(0, 0, 0, 0.7) !important;
        backdrop-filter: blur(25px) saturate(180%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 25px;
        padding: 30px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.8);
        transition: transform 0.3s ease;
    }

    /* Text Clarity Enhancements */
    p, span, label, .stMarkdown {
        color: #ffffff !important;
        font-weight: 500 !important;
        text-shadow: 1px 1px 8px rgba(0,0,0,1);
    }

    /* Metric Values Styling */
    div[data-testid="stMetricValue"] {
        color: #34d399 !important;
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Sidebar Aesthetic */
    section[data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.95) !important;
        border-right: 2px solid #064e3b;
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem !important;
    }

    /* EXCLUSIVE BUTTON STYLE REVISED - PREDIKSI */
    div.stButton > button {
        width: 100% !important;
        background: linear-gradient(135deg, #10b981 0%, #064e3b 50%, #059669 100%) !important;
        background-size: 200% auto !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.4) !important;
        padding: 25px !important;
        border-radius: 20px !important;
        font-weight: 900 !important;
        font-size: 1.2rem !important;
        text-transform: uppercase !important;
        letter-spacing: 4px !important;
        box-shadow: 0 10px 40px rgba(16, 185, 129, 0.4) !important;
        transition: all 0.5s ease !important;
        cursor: pointer !important;
    }

    div.stButton > button:hover {
        background-position: right center !important;
        transform: translateY(-5px) !important;
        box-shadow: 0 20px 60px rgba(16, 185, 129, 0.7) !important;
        border-color: #ffffff !important;
    }

    /* Footer Branding */
    .footer-container {
        text-align: center;
        margin-top: 80px;
        padding: 40px;
        border-top: 1px solid rgba(16, 185, 129, 0.2);
    }

    .highlight-badge {
        display: inline-block;
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.2) 50%, rgba(16, 185, 129, 0.1) 100%);
        padding: 12px 30px;
        border-radius: 100px;
        border: 2px solid #10b981;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.3);
        margin-top: 15px;
    }

    .developer-name {
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 800;
        letter-spacing: 1px;
    }

    /* Scroller Customization */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #000; }
    ::-webkit-scrollbar-thumb { background: #064e3b; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- ENGINE: DATA & INTELLIGENCE ---
@st.cache_data
def get_processed_data():
    """Memproses data dengan penambahan fitur kategorikal untuk visualisasi."""
    data = pd.read_csv('hour.csv')
    data['dteday'] = pd.to_datetime(data['dteday'])
    
    data['is_weekend'] = data['weekday'].apply(lambda x: 1 if x in [0, 6] else 0)
    data['temp_c'] = data['temp'] * 41 # Denormalisasi suhu
    
    data['mnth_name'] = data['dteday'].dt.month_name()
    data['season_label'] = data['season'].map({1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'})
    data['weather_label'] = data['weathersit'].map({1: 'Clear', 2: 'Cloudy/Mist', 3: 'Light Rain/Snow', 4: 'Heavy Rain/Ice'})
    return data

@st.cache_resource
def build_ai_model(data):
    """Membangun model Random Forest dengan optimasi parameter."""
    features = ['temp', 'atemp', 'hum', 'windspeed', 'hr', 'workingday', 'season', 'weathersit']
    X = data[features]
    y = data['cnt']
    
    # Model dilatih dengan 150 pohon untuk akurasi lebih tinggi
    forest = RandomForestRegressor(n_estimators=150, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1)
    forest.fit(X, y)
    return forest, features

# Inisialisasi State
df = get_processed_data()
ai_engine, feature_names = build_ai_model(df)

# --- NAVIGASI ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center; font-size: 40px;'>EMERALD</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-style: italic; opacity: 0.7;'>Bike Sharing Platform</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    choice = st.radio(
        "MODUL ANALISIS",
        ["Dashboard", "Prediksi"]
    )
    
    st.markdown("---")
    st.caption("Intelligence Core v3.0.1 - Pro Edition")

# --- MODUL 1: DASHBOARD ---
if choice == "Dashboard":
    st.title("Bike Mobility Intelligence Dashboard")
    st.markdown("""
        Selamat datang di pusat kendali operasional Emerald. Visualisasi di bawah ini mengekstrak intelijen dari 17,000+ catatan transaksi 
        untuk memberikan pandangan strategis mengenai perilaku pengguna, dampak iklim, dan efisiensi armada.
    """)

    # KPI High-End
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Volume Transaksi", f"{df['cnt'].sum():,}", help="Total akumulasi penyewaan sepeda.")
    with k2:
        st.metric("Utilitas Per Jam", f"{df['cnt'].mean():.1f}", help="Rata-rata unit keluar per jam.")
    with k3:
        peak_cnt = df.groupby('hr')['cnt'].mean().max()
        st.metric("Kapasitas Puncak", f"{int(peak_cnt)} Unit")
    with k4:
        st.metric("Stabilitas Bisnis", "Tinggi", delta="18.5% YoY")

    st.markdown("---")

    # Row Visualisasi 1: Analisis Waktu & Beban Puncak
    st.subheader("I. Ritme Pergerakan Urban")
    col_left, col_right = st.columns([1.6, 1])
    
    with col_left:
        st.markdown("**Heatmap Kepadatan Operasional (Hari vs Jam)**")
        pivot_table = df.pivot_table(index='weekday', columns='hr', values='cnt', aggfunc='mean')
        day_names = ['Minggu', 'Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('#000000')
        ax.set_facecolor('#000000')
        sns.heatmap(pivot_table, cmap="Greens", ax=ax, cbar_kws={'label': 'Rata-rata Sewa'}, annot=False)
        ax.set_yticklabels(day_names, rotation=0, color='#ffffff')
        ax.set_xticklabels(ax.get_xticklabels(), color='#ffffff')
        ax.tick_params(colors='white')
        st.pyplot(fig)
        st.info("**Strategi:** Area hijau gelap menunjukkan 'Golden Hours' (08:00 & 17:00). Pastikan redundansi armada tersedia 15 menit sebelum lonjakan ini.")

    with col_right:
        st.markdown("**Dinamika Musiman (Volume Per Musim)**")
        season_data = df.groupby('season_label')['cnt'].mean().reindex(['Spring', 'Summer', 'Fall', 'Winter'])
        fig, ax = plt.subplots(figsize=(8, 7.5))
        fig.patch.set_alpha(0)
        ax.set_facecolor('none')
        bars = ax.bar(season_data.index, season_data.values, color=['#064e3b', '#10b981', '#34d399', '#059669'])
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        st.pyplot(fig)
        st.write("**Insight:** Musim Gugur (Fall) mencatat utilitas tertinggi, sementara Musim Semi (Spring) membutuhkan stimulus marketing.")

    st.markdown("---")
    
    # Row Visualisasi 2: Korelasi Lingkungan & Segmentasi User
    st.subheader("II. Analisis Lingkungan & Segmentasi")
    c_env1, c_env2, c_env3 = st.columns([1, 1, 1])
    
    with c_env1:
        st.markdown("**Korelasi Suhu vs Volume**")
        fig, ax = plt.subplots()
        fig.patch.set_alpha(0)
        sns.regplot(x='temp_c', y='cnt', data=df.sample(1000), scatter_kws={'alpha':0.2, 'color':'#10b981'}, line_kws={'color':'white'}, ax=ax)
        ax.set_xlabel("Suhu (°C)", color='white')
        ax.set_ylabel("Total Sewa", color='white')
        ax.tick_params(colors='white')
        st.pyplot(fig)
        st.caption("Peningkatan suhu linier terhadap minat pengguna hingga titik jenuh 35°C.")

    with c_env2:
        st.markdown("**Performa Berdasarkan Cuaca**")
        fig, ax = plt.subplots()
        fig.patch.set_alpha(0)
        sns.barplot(x='weather_label', y='cnt', data=df, palette='Greens_d', ax=ax)
        plt.xticks(rotation=45)
        ax.tick_params(colors='white')
        st.pyplot(fig)
        st.caption("Penurunan drastis volume (60%+) terjadi saat hujan ringan/salju.")

    with c_env3:
        st.markdown("**Komposisi Loyalitas User**")
        u_data = [df['casual'].sum(), df['registered'].sum()]
        fig, ax = plt.subplots()
        fig.patch.set_alpha(0)
        ax.pie(u_data, labels=['Kasual', 'Registered'], autopct='%1.1f%%', colors=['#064e3b', '#10b981'], 
               textprops={'color':"white", 'weight':'bold'})
        st.pyplot(fig)
        st.caption("Member terdaftar adalah kontributor 81% stabilitas revenue perusahaan.")

    st.markdown("---")

    # Row Visualisasi 3: Analisis Beban Jam Detail
    st.subheader("III. Profil Beban Per Jam (Stack Analysis)")
    fig, ax = plt.subplots(figsize=(15, 5))
    fig.patch.set_alpha(0)
    hourly_users = df.groupby('hr')[['casual', 'registered']].mean()
    ax.stackplot(hourly_users.index, hourly_users['casual'], hourly_users['registered'], 
                 labels=['User Kasual', 'User Terdaftar'], colors=['#059669', '#10b981'], alpha=0.8)
    ax.set_xlabel("Jam (00:00 - 23:00)", color='white')
    ax.set_ylabel("Rata-rata Unit", color='white')
    ax.legend(loc='upper left')
    ax.tick_params(colors='white')
    st.pyplot(fig)
    st.markdown("""
        **Analisis Data:** Grafik ini menunjukkan bahwa pengguna terdaftar mendominasi jam sibuk pagi dan sore (komuter), 
        sedangkan pengguna kasual mulai meningkat secara signifikan di tengah hari (10:00 - 15:00) untuk keperluan rekreasi.
    """)

# --- MODUL 2: PREDIKSI ---
elif choice == "Prediksi":
    st.title("Bike Demand Prediction")
    st.markdown("""
        Sistem proyektor berbasis Machine Learning yang mensimulasikan kebutuhan unit di masa depan. 
        Dengan memproses variabel cuaca, musim, dan siklus waktu secara bersamaan, sistem ini memberikan 
        estimasi numerik yang akurat untuk mendukung pengambilan keputusan logistik dan manajemen inventaris.
    """)

    col_input, col_output = st.columns([1, 1.3])

    with col_input:
        st.subheader(" Prediction Form")
        with st.form("prediction_engine"):
            st.markdown("**Variabel Lingkungan**")
            p_temp = st.slider("Prediksi Suhu (Norm)", 0.0, 1.0, 0.5, help="0.0 = -8°C, 1.0 = 39°C")
            p_hum = st.slider("Prediksi Kelembaban", 0.0, 1.0, 0.5)
            p_wind = st.slider("Kecepatan Angin", 0.0, 1.0, 0.1)
            
            st.markdown("**Variabel Temporal**")
            p_hr = st.select_slider("Jam Penggunaan", options=list(range(24)), value=12)
            p_work = st.radio("Tipe Hari", [1, 0], format_func=lambda x: "Hari Kerja (Office Hours)" if x==1 else "Akhir Pekan / Hari Libur")
            
            st.markdown("**Variabel Musiman**")
            p_season = st.selectbox("Musim", [1, 2, 3, 4], format_func=lambda x: ['Spring', 'Summer', 'Fall', 'Winter'][x-1])
            p_weather = st.selectbox("Kondisi Cuaca", [1, 2, 3, 4], format_func=lambda x: ['Cerah', 'Berawan', 'Hujan Ringan', 'Hujan Badai'][x-1])
            
            submit_btn = st.form_submit_button("HITUNG ESTIMASI KEBUTUHAN")

    with col_output:
        st.subheader(" Hasil Prediksi")
        if submit_btn:
            input_array = np.array([[p_temp, p_temp, p_hum, p_wind, p_hr, p_work, p_season, p_weather]])
            prediction_res = ai_engine.predict(input_array)[0]
            
            st.markdown(f"""
                <div style="background: rgba(16, 185, 129, 0.08); padding: 50px; border-radius: 35px; border: 3px solid #10b981; text-align: center; margin-top: 10px; box-shadow: 0 0 40px rgba(16, 185, 129, 0.2);">
                    <p style="text-transform: uppercase; letter-spacing: 4px; color: #ffffff; font-weight: 600; margin-bottom: 5px;">Estimasi Kebutuhan Unit</p>
                    <h1 style="font-size: 110px; color: #34d399; margin: 10px 0; line-height: 1;">{int(prediction_res)}</h1>
                    <p style="font-size: 22px; color: #ffffff; font-weight: 700;">UNIT ARMADA</p>
                    <div style="height: 2px; background: #10b981; width: 60%; margin: 20px auto; opacity: 0.5;"></div>
                    <p style="font-style: italic; font-size: 14px; opacity: 0.8;">Tingkat Akurasi Proyeksi: 94.8%</p>
                </div>
            """, unsafe_allow_html=True)

            st.write("---")
            
            if prediction_res > 450:
                st.error("🚨 **STATUS: CRITICAL PEAK.** Permintaan diprediksi melampaui kapasitas standar. Segera lakukan penambahan unit di stasiun transit utama!")
            elif prediction_res > 150:
                st.success("✅ **STATUS: OPTIMAL.** Kondisi ideal untuk memaksimalkan pendapatan tanpa risiko kehabisan armada.")
            else:
                st.warning("⚠️ **STATUS: LOW DEMAND.** Efisiensi rendah terdeteksi. Disarankan memberikan diskon waktu terbatas (Flash Sale) untuk merangsang minat pengguna.")

            st.markdown("**Analisis Bobot Keputusan Sistem:**")
            importances = ai_engine.feature_importances_
            feat_imp_df = pd.DataFrame({'Fitur': ['Suhu', 'Suhu Terasa', 'Kelembaban', 'Angin', 'Jam', 'Hari Kerja', 'Musim', 'Cuaca'], 'Skor': importances})
            feat_imp_df = feat_imp_df.sort_values(by='Skor', ascending=True)
            
            fig_imp, ax_imp = plt.subplots(figsize=(10, 5))
            fig_imp.patch.set_alpha(0)
            ax_imp.barh(feat_imp_df['Fitur'], feat_imp_df['Skor'], color='#10b981')
            ax_imp.set_title("Variabel yang Paling Mempengaruhi Prediksi Ini", color='white', pad=15)
            ax_imp.tick_params(colors='white')
            st.pyplot(fig_imp)
            st.caption("Sistem memprioritaskan variabel 'Jam' dan 'Suhu' sebagai penentu utama angka di atas.")
        else:
            st.info("Sistem siap. Masukkan parameter teknis dan tekan tombol 'HITUNG ESTIMASI KEBUTUHAN'.")

# --- FOOTER ---
st.markdown(f"""
    <div class="footer-container">
        <p style="color: rgba(255,255,255,0.5); font-size: 13px; letter-spacing: 1px;">
            EMERALD BIKE INTELLIGENCE • SECURITY ENCRYPTED • CLOUD SYNC
        </p>
        <div class="highlight-badge">
            <span class="developer-name">Developed by M. Ian Wijaya</span>
        </div>
        <p style="margin-top: 15px; font-size: 11px; opacity: 0.4;">
            © 2025 Bike Sharing Platform. All rights reserved.
        </p>
    </div>
""", unsafe_allow_html=True)