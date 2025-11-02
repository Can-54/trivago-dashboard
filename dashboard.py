# dashboard.py

import streamlit as st
import pandas as pd
import sqlite3
import os
import requests
import time
from datetime import date
from typing import Tuple, List, Dict, Any, Optional

# Plotly import kontrolü
try:
    import plotly.graph_objects as go
    import plotly.express as px

    PLOTLY_AVAILABLE = True
except ImportError:
    st.error("Plotly kütüphanesi bulunamadı. Lütfen kurun: pip install plotly")
    PLOTLY_AVAILABLE = False
    px = None
    go = None

# ! YENİ (v5.0): Prophet kütüphanesi import kontrolü
try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# =============================================================================
# 0. SABİTLER (CONSTANTS)
# =============================================================================
DB_TR_FILE = "trivago_tr_fiyatlar.db"
DB_US_FILE = "trivago_usa_fiyatlar.db"
DB_DE_FILE = "trivago_de_fiyatlar.db"
DB_UK_FILE = "trivago_uk_fiyatlar.db"
API_URL_RATES = "https://api.frankfurter.app/latest?from=TRY"
FALLBACK_RATES = {"USD": 0.029, "EUR": 0.027, "GBP": 0.023}
STRATEGY_PERCENT_THRESHOLD = 10.0


# =============================================================================
# 1. SAYFA AYARLARI VE CSS
# =============================================================================
def setup_page():
    """Streamlit sayfa yapılandırmasını ayarlar."""
    st.set_page_config(
        page_title="Otel Gelir Yönetimi ve Fiyat Optimizasyon Sistemi",
        layout="wide",
        page_icon="🏨"
    )


def inject_css():
    """Özel CSS stillerini sayfaya enjekte eder."""
    # CSS
    st.markdown("""
    <style>
        .big-metric {
            font-size: 20px;
            font-weight: bold;
            padding: 12px;
            border-radius: 8px;
            text-align: center;
            margin: 5px 0;
        }
        .loss-metric {
            background-color: rgba(239, 83, 80, 0.15);
            color: #ef5350;
            border: 2px solid #ef5350;
        }
        .optimal-metric {
            background-color: rgba(102, 187, 106, 0.15);
            color: #66bb6a;
            border: 2px solid #66bb6a;
        }
        .warning-metric {
            background-color: rgba(255, 167, 38, 0.15);
            color: #ffa726;
            border: 2px solid #ffa726;
        }
        .profit-metric {
            background-color: rgba(66, 165, 245, 0.15);
            color: #42a5f5;
            border: 2px solid #42a5f5;
        }
        .info-box {
            background-color: rgba(33, 150, 243, 0.1);
            padding: 15px;
            border-radius: 8px;
            border-left: 5px solid #2196f3;
            margin: 15px 0;
        }
        .critical-alert {
            background-color: rgba(244, 67, 54, 0.15);
            padding: 20px;
            border-radius: 8px;
            border-left: 5px solid #f44336;
            margin: 15px 0;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.85; }
        }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# 2. VERİ YÜKLEME
# =============================================================================

def _load_single_db(db_file: str, market_name: str, price_col: str, currency_col: str, time_col: str,
                    note_col: str) -> Optional[pd.DataFrame]:
    """
    Tek bir SQLite veritabanı dosyasından veri yükler ve işler.
    """
    base_columns = "otel, checkin, fiyat, para_birimi, cekilme_zamani"
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(fiyatlar)")
        columns = [info[1] for info in cursor.fetchall()]

        if 'source_note' in columns:
            query = f"SELECT {base_columns}, source_note FROM fiyatlar"
        else:
            query = f"SELECT {base_columns}, 'N/A' as source_note FROM fiyatlar"

        df = pd.read_sql_query(query, conn)
        conn.close()

        df['otel'] = df['otel'].str.replace(f"({market_name})", "", regex=False).str.strip()

        df = df.rename(columns={
            'fiyat': price_col,
            'para_birimi': currency_col,
            'cekilme_zamani': time_col,
            'source_note': note_col
        })
        df[price_col] = pd.to_numeric(df[price_col], errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"{market_name} veritabanı ({db_file}) okunurken hata: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_data() -> Tuple[
    Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Dört ayrı pazarın veritabanlarından verileri çeker."""
    df_tr = _load_single_db(DB_TR_FILE, "TR", 'fiyat_tl', 'para_birimi_tl', 'cekilme_zamani_tr', 'source_note_tr')
    df_us = _load_single_db(DB_US_FILE, "USA", 'fiyat_usd', 'para_birimi_usd', 'cekilme_zamani_us', 'source_note_us')
    df_de = _load_single_db(DB_DE_FILE, "DE", 'fiyat_eur', 'para_birimi_eur', 'cekilme_zamani_de', 'source_note_de')
    df_uk = _load_single_db(DB_UK_FILE, "UK", 'fiyat_gbp', 'para_birimi_gbp', 'cekilme_zamani_uk', 'source_note_uk')

    return df_tr, df_us, df_de, df_uk


def merge_dataframes(df_tr: pd.DataFrame, df_us: pd.DataFrame, df_de: pd.DataFrame,
                     df_uk: pd.DataFrame) -> pd.DataFrame:
    """Yüklenen verileri tek bir DataFrame'de birleştirir."""
    df_merged = pd.merge(df_tr, df_us, on=['otel', 'checkin'], how='outer')
    df_merged = pd.merge(df_merged, df_de, on=['otel', 'checkin'], how='outer')
    df_merged = pd.merge(df_merged, df_uk, on=['otel', 'checkin'], how='outer')

    df_merged['checkin'] = pd.to_datetime(df_merged['checkin'], errors='coerce')
    df_merged = df_merged.sort_values(by=['otel', 'checkin'])

    # NaN temizleme
    df_merged['fiyat_tl'] = df_merged['fiyat_tl'].fillna(0)
    df_merged['fiyat_usd'] = df_merged['fiyat_usd'].fillna(0)
    df_merged['fiyat_eur'] = df_merged['fiyat_eur'].fillna(0)
    df_merged['fiyat_gbp'] = df_merged['fiyat_gbp'].fillna(0)

    return df_merged


# =============================================================================
# 3. KENAR ÇUBUĞU (SIDEBAR) VE FİLTRELER
# =============================================================================

@st.cache_data(ttl=21600, show_spinner=False)
def get_exchange_rates(today_date: date) -> Tuple[float, float, float]:
    """Frankfurter.app API'sini kullanarak güncel USD, EUR ve GBP kurlarını çeker."""
    try:
        response = requests.get(API_URL_RATES, timeout=5)
        response.raise_for_status()
        data = response.json()

        rate_usd_try = 1 / data['rates']['USD']
        rate_eur_try = 1 / data['rates']['EUR']
        rate_gbp_try = 1 / data['rates']['GBP']

        st.session_state['kur_kaynagi'] = f"✅ Güncel Kur ({data['date']})"
        st.session_state['kur_durum'] = "success"
        return rate_usd_try, rate_eur_try, rate_gbp_try

    except Exception as e:
        rate_usd_try = 1 / FALLBACK_RATES["USD"]
        rate_eur_try = 1 / FALLBACK_RATES["EUR"]
        rate_gbp_try = 1 / FALLBACK_RATES["GBP"]

        st.session_state['kur_kaynagi'] = f"⚠️ API Hatası: Varsayılan Kur Kullanılıyor"
        st.session_state['kur_durum'] = "warning"
        return rate_usd_try, rate_eur_try, rate_gbp_try


def build_sidebar(df_merged: pd.DataFrame) -> Tuple[str, str, str, float, float, float]:
    """Kenar çubuğunu oluşturur ve kullanıcı girdilerini döndürür."""
    st.sidebar.header("⚙️ Sistem Ayarları")

    # Strateji Seçimi
    st.sidebar.subheader("🎯 Optimizasyon Stratejisi")
    strateji = st.sidebar.radio(
        "Hangi stratejiyi kullanmak istersiniz?",
        options=[
            "📈 Maksimum Gelir (Premium)",
            "💰 Rekabetçi Fiyat (Volüm)",
            "⚖️ Dengeli Fiyat (Pazar Ortalaması)"
        ],
        help="""
        **Maksimum Gelir:** En yüksek fiyatı hedefler, diğer pazarları ona yükseltir.
        **Rekabetçi Fiyat:** En düşük fiyatı hedefler, pazarlarda rekabetçi kalır.
        **Dengeli Fiyat:** Fiyatı olan pazarların ortalamasını hedefler.
        """
    )
    if "Maksimum" in strateji:
        strateji_mod = "MAX"
    elif "Rekabetçi" in strateji:
        strateji_mod = "MIN"
    else:
        strateji_mod = "MEAN"

    st.sidebar.divider()

    # Otel Seçimi
    otel_listesi = ["Tümü"] + sorted(list(df_merged['otel'].unique()))
    secilen_otel = st.sidebar.selectbox("🏨 Otel Seçimi:", otel_listesi)

    # Kur Bilgisi
    st.sidebar.divider()
    st.sidebar.subheader("💱 Döviz Kuru")

    otomatik_kur_usd, otomatik_kur_eur, otomatik_kur_gbp = get_exchange_rates(date.today())

    if 'kur_kaynagi' in st.session_state:
        if st.session_state['kur_durum'] == "success":
            st.sidebar.success(st.session_state['kur_kaynagi'])
        else:
            st.sidebar.warning(st.session_state['kur_kaynagi'])

    col1, col2, col3 = st.sidebar.columns(3)
    col1.metric("USD/TRY", f"{otomatik_kur_usd:.2f}₺")
    col2.metric("EUR/TRY", f"{otomatik_kur_eur:.2f}₺")
    col3.metric("GBP/TRY", f"{otomatik_kur_gbp:.2f}₺")

    kur_usd_tl = otomatik_kur_usd
    kur_eur_tl = otomatik_kur_eur
    kur_gbp_tl = otomatik_kur_gbp

    # Veri Tazeleme Butonu
    st.sidebar.divider()
    st.sidebar.subheader("🔄 Veri Yenileme")
    if st.sidebar.button("Veritabanlarını Yeniden Yükle"):
        st.cache_data.clear()
        st.sidebar.success("Önbellek temizlendi!")
        st.sidebar.info("Sayfa 2 saniye içinde yeniden yüklenecek...")
        time.sleep(2)
        st.rerun()

    return strateji, strateji_mod, secilen_otel, kur_usd_tl, kur_eur_tl, kur_gbp_tl


# =============================================================================
# 4. TEMEL HESAPLAMALAR
# =============================================================================

def calculate_strategy_dataframe(df: pd.DataFrame, strateji_mod: str, kur_usd_tl: float, kur_eur_tl: float,
                                 kur_gbp_tl: float) -> Tuple[pd.DataFrame, str]:
    """
    Filtrelenmiş DataFrame'e strateji bazlı hesaplamaları (farklar, hedef fiyat) ekler.
    """
    df_calc = df.copy()

    # Döviz fiyatlarını TL'ye çevir
    df_calc['fiyat_usd_tl'] = df_calc['fiyat_usd'] * kur_usd_tl
    df_calc['fiyat_eur_tl'] = df_calc['fiyat_eur'] * kur_eur_tl
    df_calc['fiyat_gbp_tl'] = df_calc['fiyat_gbp'] * kur_gbp_tl

    # Sıfır olmayan fiyatları filtrele
    df_calc['fiyatlar_listesi'] = df_calc.apply(
        lambda row: [p for p in [
            row['fiyat_tl'],
            row['fiyat_usd_tl'],
            row['fiyat_eur_tl'],
            row['fiyat_gbp_tl']
        ] if p > 0],
        axis=1
    )

    # Stratejileri hesapla
    df_calc['max_fiyat_tl'] = df_calc['fiyatlar_listesi'].apply(lambda x: max(x) if len(x) > 0 else 0)
    df_calc['min_fiyat_tl'] = df_calc['fiyatlar_listesi'].apply(lambda x: min(x) if len(x) > 0 else 0)
    df_calc['mean_fiyat_tl'] = df_calc['fiyatlar_listesi'].apply(lambda x: pd.Series(x).mean() if len(x) > 0 else 0)

    # Seçilen stratejiye göre hedef fiyat
    if strateji_mod == "MAX":
        df_calc['hedef_fiyat_tl'] = df_calc['max_fiyat_tl']
        hedef_aciklama = "En Yüksek Pazar Fiyatı"
    elif strateji_mod == "MIN":
        df_calc['hedef_fiyat_tl'] = df_calc['min_fiyat_tl']
        hedef_aciklama = "En Düşük Pazar Fiyatı"
    else:  # "MEAN"
        df_calc['hedef_fiyat_tl'] = df_calc['mean_fiyat_tl']
        hedef_aciklama = "Pazar Ortalaması Fiyatı"

    # Kayıp/Kar hesapla
    df_calc['fark_tr'] = df_calc['fiyat_tl'] - df_calc['hedef_fiyat_tl']
    df_calc['fark_us'] = df_calc['fiyat_usd_tl'] - df_calc['hedef_fiyat_tl']
    df_calc['fark_de'] = df_calc['fiyat_eur_tl'] - df_calc['hedef_fiyat_tl']
    df_calc['fark_uk'] = df_calc['fiyat_gbp_tl'] - df_calc['hedef_fiyat_tl']

    # Yüzde hesapla
    def calculate_percent_diff(row, fark_col):
        if row['hedef_fiyat_tl'] > 0:
            return (row[fark_col] / row['hedef_fiyat_tl'] * 100)
        return 0

    df_calc['fark_tr_yuzde'] = df_calc.apply(calculate_percent_diff, fark_col='fark_tr', axis=1)
    df_calc['fark_us_yuzde'] = df_calc.apply(calculate_percent_diff, fark_col='fark_us', axis=1)
    df_calc['fark_de_yuzde'] = df_calc.apply(calculate_percent_diff, fark_col='fark_de', axis=1)
    df_calc['fark_uk_yuzde'] = df_calc.apply(calculate_percent_diff, fark_col='fark_uk', axis=1)

    return df_calc, hedef_aciklama

# =============================================================================
# 5. GÖSTERGE PANELİ (DASHBOARD) BİLEŞENLERİ
# =============================================================================

# -----------------------------------------------------------------------------
# 5.1. GENEL BAKIŞ SEKMESİ
# -----------------------------------------------------------------------------

def display_summary_metrics(df: pd.DataFrame, strateji: str, strateji_mod: str, hedef_aciklama: str):
    """Ana sayfadaki özet metrikleri (KPI) gösterir."""
    st.header(f"📊 Gelir Analizi - {strateji}")

    # Hesaplamalar
    fark_tr_neg = abs(df['fark_tr'][df['fark_tr'] < 0].sum())
    fark_us_neg = abs(df['fark_us'][df['fark_us'] < 0].sum())
    fark_de_neg = abs(df['fark_de'][df['fark_de'] < 0].sum())
    fark_uk_neg = abs(df['fark_uk'][df['fark_uk'] < 0].sum())
    toplam_kayip = fark_tr_neg + fark_us_neg + fark_de_neg + fark_uk_neg

    fark_tr_pos = df['fark_tr'][df['fark_tr'] > 0].sum()
    fark_us_pos = df['fark_us'][df['fark_us'] > 0].sum()
    fark_de_pos = df['fark_de'][df['fark_de'] > 0].sum()
    fark_uk_pos = df['fark_uk'][df['fark_uk'] > 0].sum()
    toplam_fazlalik = fark_tr_pos + fark_us_pos + fark_de_pos + fark_uk_pos

    col1, col2, col3, col4, col5 = st.columns(5)

    if strateji_mod == "MAX":
        tr_adet = len(df[df['fark_tr'] < 0])
        us_adet = len(df[df['fark_us'] < 0])
        de_adet = len(df[df['fark_de'] < 0])
        uk_adet = len(df[df['fark_uk'] < 0])
        toplam_adet = tr_adet + us_adet + de_adet + uk_adet

        with col1:
            st.metric("🇹🇷 Türkiye Potansiyel Kayıp", f"{fark_tr_neg:,.0f}₺", f"-{tr_adet} rezervasyon",
                      delta_color="inverse")
        with col2:
            st.metric("🇺🇸 ABD Potansiyel Kayıp", f"{fark_us_neg:,.0f}₺", f"-{us_adet} rezervasyon",
                      delta_color="inverse")
        with col3:
            st.metric("🇩🇪 Almanya Potansiyel Kayıp", f"{fark_de_neg:,.0f}₺", f"-{de_adet} rezervasyon",
                      delta_color="inverse")
        with col4:
            st.metric("🇬🇧 UK Potansiyel Kayıp", f"{fark_uk_neg:,.0f}₺", f"-{uk_adet} rezervasyon",
                      delta_color="inverse")
        with col5:
            st.metric("💰 Toplam Potansiyel Kayıp", f"{toplam_kayip:,.0f}₺", f"{toplam_adet} rezervasyonda kayıp",
                      delta_color="inverse")

        if toplam_kayip > 100:
            st.markdown(f"""
            <div class='critical-alert'>
            <h3>🚨 KRİTİK UYARI - Maksimum Gelir Stratejisi</h3>
            <p><b>{toplam_kayip:,.0f}₺</b> potansiyel gelir kaybı tespit edildi!</p>
            <p><b>{toplam_adet} rezervasyonda</b> fiyatlar optimal seviyenin altında.</p>
            <p>Tüm pazarlarda fiyatları <b>{hedef_aciklama}</b> seviyesine yükselterek bu kaybı önleyebilirsiniz.</p>
            </div>
            """, unsafe_allow_html=True)

    else:  # MIN veya MEAN
        tr_adet = len(df[df['fark_tr'] > 0])
        us_adet = len(df[df['fark_us'] > 0])
        de_adet = len(df[df['fark_de'] > 0])
        uk_adet = len(df[df['fark_uk'] > 0])
        toplam_adet = tr_adet + us_adet + de_adet + uk_adet

        mesaj = "Fiyat İndirimi" if strateji_mod == "MIN" else "Fiyat Fazlalığı (Ort. Üstü)"
        mesaj_toplam = "Toplam İndirim Potansiyeli" if strateji_mod == "MIN" else "Toplam Fiyat Fazlalığı"

        with col1:
            st.metric(f"🇹🇷 Türkiye {mesaj}", f"{fark_tr_pos:,.0f}₺", f"{tr_adet} rezervasyon", delta_color="normal")
        with col2:
            st.metric(f"🇺🇸 ABD {mesaj}", f"{fark_us_pos:,.0f}₺", f"{us_adet} rezervasyon", delta_color="normal")
        with col3:
            st.metric(f"🇩🇪 Almanya {mesaj}", f"{fark_de_pos:,.0f}₺", f"{de_adet} rezervasyon", delta_color="normal")
        with col4:
            st.metric(f"🇬🇧 UK {mesaj}", f"{fark_uk_pos:,.0f}₺", f"{uk_adet} rezervasyon", delta_color="normal")
        with col5:
            st.metric(f"💼 {mesaj_toplam}", f"{toplam_fazlalik:,.0f}₺", f"{toplam_adet} rezervasyon",
                      delta_color="normal")

    # Stratejiye özel açıklama kutusu
    gunluk_ortalama = 0.0
    if strateji_mod == "MAX":
        st.info(f"""
        💡 **Maksimum Gelir Stratejisi - Açıklama:**
        - **{toplam_kayip:,.0f}₺:** Fiyatlar optimal seviyeye çıkarılırsa kazanılabilecek toplam ek gelir
        - **{toplam_adet} rezervasyon:** Fiyat artışı önerilen rezervasyon sayısı
        - **Hedef:** Her pazarda en yüksek fiyatı hedefleyerek geliri maksimize edin
        - **Beklenen Sonuç:** Oda başı gelir artar, premium pozisyon güçlenir
        """)
        gunluk_ortalama = toplam_kayip / df['checkin'].nunique() if df['checkin'].nunique() > 0 else 0

    elif strateji_mod == "MIN":
        st.info(f"""
        💡 **Rekabetçi Fiyat Stratejisi - Açıklama:**
        - **{toplam_fazlalik:,.0f}₺:** Tüm pazarlarda yapılabilecek toplam fiyat indirimi
        - **{toplam_adet} rezervasyon:** Bu indirim önerilen rezervasyon sayısı
        - **Hedef:** En düşük pazar fiyatına uyum sağlayarak rekabetçi kalın
        - **Beklenen Sonuç:** Fiyat düşürülerek doluluk oranı artırılabilir
        """)
        gunluk_ortalama = toplam_fazlalik / df['checkin'].nunique() if df['checkin'].nunique() > 0 else 0

    else:  # MEAN
        st.info(f"""
        💡 **Dengeli Fiyat Stratejisi - Açıklama:**
        - **{toplam_fazlalik:,.0f}₺:** Fiyatı ortalamanın üzerinde olan rezervasyonlardaki toplam fazlalık.
        - **{toplam_kayip:,.0f}₺:** Fiyatı ortalamanın altında olan rezervasyonlardaki toplam kayıp.
        - **Net Etki:** {toplam_fazlalik - toplam_kayip:,.0f}₺
        - **Hedef:** Fiyatı pazar ortalamasına çekerek fiyat tutarlılığı sağlamak
        """)
        gunluk_ortalama = (toplam_fazlalik - toplam_kayip) / df['checkin'].nunique() if df[
                                                                                            'checkin'].nunique() > 0 else 0

    benzersiz_gunler = df['checkin'].nunique()
    st.info(f"""
    💡 **Tahmini Projeksiyonlar ({strateji}):**

    **Mevcut Veri Özeti:**
    - Analiz edilen tarih sayısı: **{benzersiz_gunler} gün**
    - Toplam rezervasyon: **{len(df)} adet**
    - Stratejinin Günlük Ortalama Etkisi: **{gunluk_ortalama:,.0f}₺**

    **Varsayımsal Tahminler** *(Aynı doluluk oranı ve pazar koşulları devam ederse)*:
    - **Aylık (30 gün):** {gunluk_ortalama * 30:,.0f}₺
    - **Yıllık (365 gün):** {gunluk_ortalama * 365:,.0f}₺

    **Not:** Bu tahminler, mevcut veri setindeki günlük ortalama üzerinden hesaplanmıştır.
    Gerçek sonuçlar sezon, doluluk oranı ve pazar koşullarına göre değişebilir.

    **Hedef Fiyat Bazı:** {hedef_aciklama}
    """)


def display_price_chart(df: pd.DataFrame, strateji: str, strateji_mod: str, secilen_otel: str, hedef_aciklama: str):
    """Plotly ile zaman serisi fiyat karşılaştırma grafiğini çizer."""
    if not PLOTLY_AVAILABLE: return

    st.subheader(f"📈 {secilen_otel} - Fiyat Karşılaştırma Grafiği")
    fig = go.Figure()

    # Traces
    fig.add_trace(go.Scatter(x=df['checkin'], y=df['fiyat_tl'], mode='lines+markers', name='🇹🇷 Türkiye (₺)',
                             line=dict(color='#ef5350', width=2), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=df['checkin'], y=df['fiyat_usd_tl'], mode='lines+markers', name='🇺🇸 ABD (₺)',
                             line=dict(color='#42a5f5', width=2), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=df['checkin'], y=df['fiyat_eur_tl'], mode='lines+markers', name='🇩🇪 Almanya (₺)',
                             line=dict(color='#ffa726', width=2), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=df['checkin'], y=df['fiyat_gbp_tl'], mode='lines+markers', name='🇬🇧 UK (₺)',
                             line=dict(color='#ab47bc', width=2), marker=dict(size=6)))

    # Hedef fiyat çizgisi
    if strateji_mod == "MAX":
        renk, etiket = '#66bb6a', '✅ Hedef: Maksimum Fiyat'
    elif strateji_mod == "MIN":
        renk, etiket = '#42a5f5', '💰 Hedef: Minimum Fiyat'
    else:  # MEAN
        renk, etiket = '#ffffff', '⚖️ Hedef: Ortalama Fiyat'

    fig.add_trace(go.Scatter(x=df['checkin'], y=df['hedef_fiyat_tl'], mode='lines', name=etiket,
                             line=dict(color=renk, width=3, dash='dash')))

    fig.update_layout(title=f'{secilen_otel} - Pazarlara Göre Fiyat Değişimi ({strateji})',
                      xaxis_title='Check-in Tarihi', yaxis_title='Fiyat (₺)', hovermode='x unified', height=500,
                      template='plotly_dark',
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)


def display_heatmap(df: pd.DataFrame, strateji_mod: str):
    """'Tümü' seçiliyken otellerin potansiyel kaybını/fazlalığını gösteren bir ısı haritası çizer."""
    if not PLOTLY_AVAILABLE: return

    st.subheader("🔥 Otel Fiyat Farklılıkları - Isı Haritası")

    df_hm_list = []

    if strateji_mod == "MAX":
        title_text = "Potansiyel Kayıp (₺) (Fiyat, Hedef Fiyattan Ne Kadar Düşük?)"
        color_scale = "Reds"
        for _, row in df.iterrows():
            kayip = sum(abs(row[f]) if row[f] < 0 else 0 for f in ['fark_tr', 'fark_us', 'fark_de', 'fark_uk'])
            df_hm_list.append({'otel': row['otel'], 'checkin': row['checkin'], 'deger': kayip})
    else:  # MIN veya MEAN
        title_text = "Fiyat Fazlalığı (₺) (Fiyat, Hedef Fiyattan Ne Kadar Yüksek?)"
        color_scale = "Blues"
        for _, row in df.iterrows():
            fazlalik = sum(row[f] if row[f] > 0 else 0 for f in ['fark_tr', 'fark_us', 'fark_de', 'fark_uk'])
            df_hm_list.append({'otel': row['otel'], 'checkin': row['checkin'], 'deger': fazlalik})

    df_hm = pd.DataFrame(df_hm_list)
    df_hm_pivot = df_hm.pivot_table(index='otel', columns='checkin', values='deger', aggfunc='sum').fillna(0)

    if df_hm_pivot.empty:
        st.warning("Isı haritası için veri bulunamadı.")
        return

    fig = px.imshow(
        df_hm_pivot,
        aspect="auto",
        labels=dict(x="Check-in Tarihi", y="Otel", color="Fark (₺)"),
        title=title_text,
        color_continuous_scale=color_scale,
        template="plotly_dark"
    )
    fig.update_traces(hovertemplate="Otel: %{y}<br>Tarih: %{x}<br>Fark: %{z:,.0f}₺<extra></extra>")
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    st.success(
        "💡 **Isı Haritası Yorumu:** Koyu renkler, o otelin o tarihte seçilen stratejiye göre en fazla saptığı yerleri gösterir.")


def display_day_of_week_analysis(df: pd.DataFrame, strateji_mod: str):
    """Haftanın günlerine göre fiyat farklarını analiz eden bir BASİT ÇUBUK GRAFİK çizer."""
    if not PLOTLY_AVAILABLE: return

    st.subheader("📅 Haftanın Günü Bazlı Analiz")

    df_dow = df.copy()
    df_dow['gun_adi'] = df_dow['checkin'].dt.day_name()

    days_tr = {
        'Monday': 'Pazartesi', 'Tuesday': 'Salı', 'Wednesday': 'Çarşamba',
        'Thursday': 'Perşembe', 'Friday': 'Cuma', 'Saturday': 'Cumartesi',
        'Sunday': 'Pazar'
    }
    df_dow['gun_adi'] = df_dow['gun_adi'].map(days_tr)
    day_order = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar']

    if strateji_mod == "MAX":
        df_dow['toplam_fark'] = df_dow.apply(
            lambda row: sum(abs(row[f]) if row[f] < 0 else 0 for f in ['fark_tr', 'fark_us', 'fark_de', 'fark_uk']),
            axis=1)
        title_text = "Haftanın Günlerine Göre Ortalama Potansiyel Kayıp"
        y_label = "Ortalama Potansiyel Kayıp (₺)"
    else:  # MIN veya MEAN
        df_dow['toplam_fark'] = df_dow.apply(
            lambda row: sum(row[f] if row[f] > 0 else 0 for f in ['fark_tr', 'fark_us', 'fark_de', 'fark_uk']), axis=1)
        title_text = "Haftanın Günlerine Göre Ortalama Fiyat Fazlalığı"
        y_label = "Ortalama Fiyat Fazlalığı (₺)"

    df_dow_filtered = df_dow[df_dow['toplam_fark'] > 0]

    if df_dow_filtered.empty:
        st.warning("Haftanın günü analizi için yeterli sapma verisi bulunamadı.")
        return

    df_dow_agg = df_dow_filtered.groupby('gun_adi')['toplam_fark'].mean().reset_index()
    df_dow_agg = df_dow_agg.set_index('gun_adi').reindex(day_order).reset_index()

    fig = px.bar(
        df_dow_agg,
        x='gun_adi',
        y='toplam_fark',
        title=title_text,
        labels={'gun_adi': 'Haftanın Günü', 'toplam_fark': y_label},
        template='plotly_dark'
    )
    fig.update_traces(hovertemplate="Gün: %{x}<br>Ortalama Fark: %{y:,.0f}₺<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)
    st.success(
        "💡 **Çubuk Grafik Yorumu:** Bu grafik, haftanın hangi günlerinin **ortalama olarak** stratejiden en çok saptığını gösterir.")


def display_overview_tab(df_analiz: pd.DataFrame, strateji: str, strateji_mod: str, secilen_otel: str,
                         hedef_aciklama: str):
    """Ana 'Genel Bakış' sekmesinin içeriğini yönetir."""

    display_summary_metrics(df_analiz, strateji, strateji_mod, hedef_aciklama)

    st.divider()

    tab_fiyat_analizi, tab_gun_analizi = st.tabs(
        ["📈 Fiyat Analizi (Zaman Serisi)", "📅 Haftanın Günü Analizi (Sezonsallık)"])

    with tab_fiyat_analizi:
        if secilen_otel == "Tümü":
            display_heatmap(df_analiz, strateji_mod)
        else:
            display_price_chart(df_analiz, strateji, strateji_mod, secilen_otel, hedef_aciklama)

    with tab_gun_analizi:
        display_day_of_week_analysis(df_analiz, strateji_mod)


# -----------------------------------------------------------------------------
# 5.2. STRATEJİ ÖNERİLERİ SEKMESİ
# -----------------------------------------------------------------------------

# ! 'st.data_editor' yerine 'st.dataframe' + Style eklemesi gerçekleşti
def display_styled_analysis_table(df: pd.DataFrame, strateji: str, strateji_mod: str):
    """TAB 1: Rezervasyon Bazlı Analiz sekmesini 'st.dataframe' (renkli) ile gösterir."""
    st.subheader(f"📊 Rezervasyon Bazlı Analiz ({strateji})")

    df_tab = df.copy()
    df_tab['Tarih'] = df_tab['checkin'].dt.strftime('%Y-%m-%d')

    columns_to_show = [
        'otel', 'Tarih', 'hedef_fiyat_tl',
        'fiyat_tl', 'fark_tr_yuzde',
        'fiyat_usd_tl', 'fark_us_yuzde',
        'fiyat_eur_tl', 'fark_de_yuzde',
        'fiyat_gbp_tl', 'fark_uk_yuzde'
    ]
    df_display = df_tab[columns_to_show].rename(columns={
        'otel': 'Otel',
        'hedef_fiyat_tl': 'Hedef Fiyat (₺)',
        'fiyat_tl': 'TR Fiyat (₺)',
        'fark_tr_yuzde': 'TR Fark (%)',
        'fiyat_usd_tl': 'US Fiyat (₺)',
        'fark_us_yuzde': 'US Fark (%)',
        'fiyat_eur_tl': 'DE Fiyat (₺)',
        'fark_de_yuzde': 'DE Fark (%)',
        'fiyat_gbp_tl': 'UK Fiyat (₺)',
        'fark_uk_yuzde': 'UK Fark (%)'
    })

    min_val = -50
    max_val = 50

    if strateji_mod == "MAX":
        cmap_color = 'Reds_r'
    else:
        cmap_color = 'Blues'

    fark_cols = ['TR Fark (%)', 'US Fark (%)', 'DE Fark (%)', 'UK Fark (%)']

    st.dataframe(
        df_display.style
        .format({
            'Hedef Fiyat (₺)': '{:,.0f}₺',
            'TR Fiyat (₺)': '{:,.0f}₺',
            'US Fiyat (₺)': '{:,.0f}₺',
            'DE Fiyat (₺)': '{:,.0f}₺',
            'UK Fiyat (₺)': '{:,.0f}₺',
            'TR Fark (%)': '{:,.1f}%',
            'US Fark (%)': '{:,.1f}%',
            'DE Fark (%)': '{:,.1f}%',
            'UK Fark (%)': '{:,.1f}%',
        })
        .background_gradient(cmap=cmap_color, subset=fark_cols, vmin=min_val, vmax=max_val),
        use_container_width=True,
        height=500
    )

def display_recommendations_tab(df: pd.DataFrame, strateji: str, strateji_mod: str, kur_usd_tl: float,
                                kur_eur_tl: float, kur_gbp_tl: float):
    """TAB 2: Strateji Önerileri sekmesini gösterir."""
    st.subheader(f"💡 {strateji} - Eylem Önerileri")

    if strateji_mod == "MAX":
        st.info(
            f"📈 **Maksimum Gelir Stratejisi:** Potansiyel kayıp {STRATEGY_PERCENT_THRESHOLD}%'den fazla olan rezervasyonlar için fiyat artışı önerileri")
        compare_op = lambda fark_yuzde: fark_yuzde < -STRATEGY_PERCENT_THRESHOLD
    elif strateji_mod == "MIN":
        st.info(
            f"💰 **Rekabetçi Fiyat Stratejisi:** {STRATEGY_PERCENT_THRESHOLD}%'den fazla pahalı olan rezervasyonlar için fiyat indirimi önerileri")
        compare_op = lambda fark_yuzde: fark_yuzde > STRATEGY_PERCENT_THRESHOLD
    else:  # MEAN
        st.info(
            f"⚖️ **Dengeli Fiyat Stratejisi:** Fiyatı ortalamadan {STRATEGY_PERCENT_THRESHOLD}%'den fazla sapan rezervasyonlar için öneriler")
        compare_op = lambda fark_yuzde: abs(fark_yuzde) > STRATEGY_PERCENT_THRESHOLD

    oneriler_listesi = []

    for _, row in df.iterrows():
        oneriler = []
        hedef_fiyat_tl = row['hedef_fiyat_tl']

        def create_recommendation(fark_yuzde, pazar_adi, fiyat_orj, kur, symbol, hedef_fiyat_tl):
            if compare_op(fark_yuzde):
                hedef_fiyat_orj = hedef_fiyat_tl / kur if kur > 0 else hedef_fiyat_tl
                if fark_yuzde < 0:
                    artis = hedef_fiyat_orj - fiyat_orj
                    return f"{pazar_adi}: Fiyatı `{fiyat_orj:,.0f}{symbol}` den `{hedef_fiyat_orj:,.0f}{symbol}` ye yükseltin (`+{artis:,.0f}{symbol}`, `+{abs(fark_yuzde):.1f}%`)"
                else:
                    azalis = fiyat_orj - hedef_fiyat_orj
                    return f"{pazar_adi}: Fiyatı `{fiyat_orj:,.0f}{symbol}` den `{hedef_fiyat_orj:,.0f}{symbol}` ye indirin (`-{azalis:,.0f}{symbol}`, `-{fark_yuzde:.1f}%`)"
            return None

        oneriler.append(
            create_recommendation(row['fark_tr_yuzde'], "🇹🇷 **Türkiye**", row['fiyat_tl'], 1.0, "₺", hedef_fiyat_tl))
        oneriler.append(create_recommendation(row['fark_us_yuzde'], "🇺🇸 **ABD**", row['fiyat_usd'], kur_usd_tl, "\$",
                                              hedef_fiyat_tl))
        oneriler.append(create_recommendation(row['fark_de_yuzde'], "🇩🇪 **Almanya**", row['fiyat_eur'], kur_eur_tl, "€",
                                              hedef_fiyat_tl))
        oneriler.append(
            create_recommendation(row['fark_uk_yuzde'], "🇬🇧 **UK**", row['fiyat_gbp'], kur_gbp_tl, "£", hedef_fiyat_tl))

        oneriler = [o for o in oneriler if o is not None]

        if oneriler:
            oneriler_listesi.append((row, oneriler))

    def get_total_diff_score(row_data: Tuple[pd.Series, List[str]]) -> float:
        row = row_data[0]
        return sum(abs(row[f]) if compare_op(row[f]) else 0 for f in
                   ['fark_tr_yuzde', 'fark_us_yuzde', 'fark_de_yuzde', 'fark_uk_yuzde'])

    oneriler_listesi.sort(key=get_total_diff_score, reverse=True)

    if not oneriler_listesi:
        st.success(
            f"✅ {strateji} için acil eylem gerekmiyor (Tüm fiyatlar +/- %{STRATEGY_PERCENT_THRESHOLD} toleransı içinde).")
    else:
        st.error(f"**{len(oneriler_listesi)}** adet eylem önerisi bulundu:")
        for row, oneriler in oneriler_listesi:
            toplam_etki_skoru = get_total_diff_score((row, oneriler))
            baslik = f"🔴 {row['otel']} ({row['checkin'].strftime('%d.%m.%Y')}) - Toplam Sapma Skoru: {toplam_etki_skoru:.0f} Puan"

            with st.expander(baslik):
                for oneri in oneriler:
                    st.markdown(oneri, unsafe_allow_html=True)


def display_data_table_tab(df: pd.DataFrame, strateji_mod: str, secilen_otel: str):
    """TAB 3: Detaylı Veri Tablosu sekmesini ve CSV indirme butonunu gösterir."""
    st.subheader("🗂️ Detaylı Veri Tablosu")

    df_gosterim = df[[
        'otel', 'checkin',
        'fiyat_tl', 'fark_tr', 'fark_tr_yuzde',
        'fiyat_usd', 'fiyat_usd_tl', 'fark_us', 'fark_us_yuzde',
        'fiyat_eur', 'fiyat_eur_tl', 'fark_de', 'fark_de_yuzde',
        'fiyat_gbp', 'fiyat_gbp_tl', 'fark_uk', 'fark_uk_yuzde',
        'hedef_fiyat_tl', 'max_fiyat_tl', 'min_fiyat_tl', 'mean_fiyat_tl'
    ]].copy()

    df_gosterim = df_gosterim.rename(columns={
        'otel': 'Otel Adı', 'checkin': 'Check-in',
        'fiyat_tl': 'TR (₺)', 'fark_tr': 'TR Fark (₺)', 'fark_tr_yuzde': 'TR Fark (%)',
        'fiyat_usd': 'US ($)', 'fiyat_usd_tl': 'US (₺)', 'fark_us': 'US Fark (₺)', 'fark_us_yuzde': 'US Fark (%)',
        'fiyat_eur': 'DE (€)', 'fiyat_eur_tl': 'DE (₺)', 'fark_de': 'DE Fark (₺)', 'fark_de_yuzde': 'DE Fark (%)',
        'fiyat_gbp': 'UK (£)', 'fiyat_gbp_tl': 'UK (₺)', 'fark_uk': 'UK Fark (₺)', 'fark_uk_yuzde': 'UK Fark (%)',
        'hedef_fiyat_tl': 'Hedef Fiyat (₺)', 'max_fiyat_tl': 'Max Fiyat (₺)', 'min_fiyat_tl': 'Min Fiyat (₺)',
        'mean_fiyat_tl': 'Ortalama Fiyat (₺)'
    })
    st.dataframe(df_gosterim, use_container_width=True, height=400)
    csv = df_gosterim.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 CSV Olarak İndir", data=csv,
        file_name=f"gelir_analizi_{strateji_mod}_{secilen_otel}_{date.today()}.csv",
        mime="text/csv"
    )

def display_strategy_tab(df_analiz: pd.DataFrame, strateji: str, strateji_mod: str, hedef_aciklama: str,
                         kur_usd_tl: float,
                         kur_eur_tl: float, kur_gbp_tl: float, secilen_otel: str):
    """Ana 'Strateji Önerileri' sekmesinin içeriğini yönetir."""

    st.info("Bu sekmedeki tüm analizler, aşağıda seçtiğiniz tarih aralığına göre filtrelenir.")

    min_tarih = df_analiz['checkin'].min().date()
    max_tarih = df_analiz['checkin'].max().date()

    secilen_aralik = st.date_input(
        "Analiz için Tarih Aralığı Seçin:",
        value=(min_tarih, max_tarih),
        min_value=min_tarih,
        max_value=max_tarih,
        key="strategy_date_filter"
    )

    df_filtrelenmis = df_analiz.copy()
    if len(secilen_aralik) == 2:
        df_filtrelenmis = df_analiz[
            (df_analiz['checkin'].dt.date >= secilen_aralik[0]) &
            (df_analiz['checkin'].dt.date <= secilen_aralik[1])
            ]

    if df_filtrelenmis.empty:
        st.warning(f"Seçilen tarih aralığı ({secilen_aralik[0]} - {secilen_aralik[1]}) için veri bulunamadı.")
        return

    tab_analiz, tab_oneri, tab_detay = st.tabs([
        f"📊 Renkli Analiz Tablosu ({len(df_filtrelenmis)} Kayıt)",
        f"💡 Eylem Önerileri ({len(df_filtrelenmis)} Kayıt)",
        "🗂️ Detaylı Veri Tablosu (CSV)"
    ])

    with tab_analiz:
        display_styled_analysis_table(df_filtrelenmis, strateji, strateji_mod)

    with tab_oneri:
        display_recommendations_tab(df_filtrelenmis, strateji, strateji_mod, kur_usd_tl, kur_eur_tl, kur_gbp_tl)

    with tab_detay:
        display_data_table_tab(df_filtrelenmis, strateji_mod, secilen_otel)


# -----------------------------------------------------------------------------
# 5.3. SİSTEM SAĞLIĞI SEKMESİ
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_health_data(df_tr: pd.DataFrame, df_us: pd.DataFrame, df_de: pd.DataFrame, df_uk: pd.DataFrame) -> pd.DataFrame:
    """4 ham dataframe'den source_note verilerini birleştirir."""

    dfs_to_concat = []

    if df_tr is not None:
        df_tr_notes = df_tr[['otel', 'checkin', 'source_note_tr']].rename(columns={'source_note_tr': 'source_note'})
        df_tr_notes['Pazar'] = 'TR'
        dfs_to_concat.append(df_tr_notes)

    if df_us is not None:
        df_us_notes = df_us[['otel', 'checkin', 'source_note_us']].rename(columns={'source_note_us': 'source_note'})
        df_us_notes['Pazar'] = 'US'
        dfs_to_concat.append(df_us_notes)

    if df_de is not None:
        df_de_notes = df_de[['otel', 'checkin', 'source_note_de']].rename(columns={'source_note_de': 'source_note'})
        df_de_notes['Pazar'] = 'DE'
        dfs_to_concat.append(df_de_notes)

    if df_uk is not None:
        df_uk_notes = df_uk[['otel', 'checkin', 'source_note_uk']].rename(columns={'source_note_uk': 'source_note'})
        df_uk_notes['Pazar'] = 'UK'
        dfs_to_concat.append(df_uk_notes)

    if not dfs_to_concat:
        return pd.DataFrame(columns=['otel', 'checkin', 'source_note', 'Pazar'])

    df_health = pd.concat(dfs_to_concat, ignore_index=True)
    df_health['source_note'] = df_health['source_note'].fillna('Bilinmiyor')
    return df_health

def display_raw_data_section(df_tr: pd.DataFrame, df_us: pd.DataFrame, df_de: pd.DataFrame, df_uk: pd.DataFrame):
    """Ham veritabanı verilerini bir checkbox ardında gösterir."""
    st.subheader("🔧 Ham Veritabanı Verileri")
    st.warning("⚠️ Bu bölüm teknik kullanıcılar içindir. Ana filtrelerden etkilenmez.")

    tab_tr, tab_us, tab_de, tab_uk = st.tabs(["TR Veritabanı", "US Veritabanı", "DE Veritabanı", "UK Veritabanı"])

    with tab_tr:
        if df_tr is not None and not df_tr.empty:
            st.dataframe(df_tr, use_container_width=True, height=300)
            st.caption(f"Toplam {len(df_tr)} kayıt")
        else:
            st.error("TR verisi yüklenemedi veya boş.")
    with tab_us:
        if df_us is not None and not df_us.empty:
            st.dataframe(df_us, use_container_width=True, height=300)
            st.caption(f"Toplam {len(df_us)} kayıt")
        else:
            st.error("US verisi yüklenemedi veya boş.")
    with tab_de:
        if df_de is not None and not df_de.empty:
            st.dataframe(df_de, use_container_width=True, height=300)
            st.caption(f"Toplam {len(df_de)} kayıt")
        else:
            st.error("DE verisi yüklenemedi veya boş.")
    with tab_uk:
        if df_uk is not None and not df_uk.empty:
            st.dataframe(df_uk, use_container_width=True, height=300)
            st.caption(f"Toplam {len(df_uk)} kayıt")
        else:
            st.error("UK verisi yüklenemedi veya boş.")

def display_health_tab(df_tr: pd.DataFrame, df_us: pd.DataFrame, df_de: pd.DataFrame, df_uk: pd.DataFrame):
    """Ana 'Sistem Sağlığı' sekmesinin içeriğini yönetir."""

    st.header("🩺 Scraper Sistem Sağlığı")
    st.info("Bu bölüm, 4 pazarın (TR, US, DE, UK) veritabanlarındaki `source_note` (kaynak notu) sütununu analiz eder.")

    df_health = get_health_data(df_tr, df_us, df_de, df_uk)

    if df_health.empty:
        st.warning("Sistem sağlığı verisi bulunamadı.")
        return

    success_notes = [
        'our_lowest_label', 'min_from_list', 'fallback_top_main_block',
        'en_dusuk_fiyatimiz_etiketi', 'min_from_main_block',
        'niedrigster_preis_etikett'
    ]

    error_notes = [
        'CRASH_OR_NOT_FOUND', 'CRASH_OR_TIMEOUT',
        'main_block_id_timeout', 'main_block_id_not_found',
        'main_block_find_error', 'not_found', 'Bilinmiyor', 'N/A'
    ]

    def categorize_note(note):
        if note in success_notes:
            return "Başarılı"
        elif note in error_notes:
            return "Veri Çekilemedi"
        else:
            if 'min_from_main_block' in note:
                success_notes.append(note)
                return "Başarılı"
            return "Diğer"

    df_health['Kategori'] = df_health['source_note'].apply(categorize_note)

    total_scrapes = len(df_health)
    total_success = (df_health['Kategori'] == 'Başarılı').sum()
    total_errors = (df_health['Kategori'] == 'Veri Çekilemedi').sum()

    success_rate = (total_success / total_scrapes) * 100 if total_scrapes > 0 else 0

    st.subheader("Genel Başarı Durumu")
    col1, col2, col3 = st.columns(3)
    col1.metric("Toplam Kayıt (Deneme)", f"{total_scrapes}")
    col2.metric("Başarılı Fiyat Çekme", f"{total_success}")
    col3.metric("Genel Başarı Oranı", f"{success_rate:.1f}%")

    if total_errors > 0:
        st.error(f"**{total_errors} adet kayıtta veri çekilemedi.** Detaylar için aşağıdaki tabloları inceleyin.")
    else:
        st.success("Tüm scrape işlemleri başarıyla tamamlanmış görünüyor!")

    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Başarı Durumu (Genel)")

        df_kategori_counts = df_health['Kategori'].value_counts().reset_index()
        df_kategori_counts.columns = ['Kategori', 'Sayı']

        if PLOTLY_AVAILABLE:
            fig = px.bar(
                df_kategori_counts, x='Kategori', y='Sayı', color='Kategori',
                color_discrete_map={'Başarılı': 'green', 'Veri Çekilemedi': 'red', 'Diğer': 'grey'},
                template='plotly_dark', title="Başarılı vs. Çekilemeyen Veri Sayısı"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(df_kategori_counts)

        with st.expander("Detaylı 'source_note' Dağılımını Gör (Pasta Grafik)"):
            note_counts = df_health['source_note'].value_counts().reset_index()
            note_counts.columns = ['Not', 'Sayı']
            note_counts['Kategori'] = note_counts['Not'].apply(categorize_note)

            if PLOTLY_AVAILABLE:
                fig_pie = px.pie(
                    note_counts, names='Not', values='Sayı', title='`source_note` Dağılımı',
                    color='Kategori',
                    color_discrete_map={'Başarılı': 'green', 'Veri Çekilemedi': 'red', 'Diğer': 'grey'},
                    template='plotly_dark'
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.dataframe(note_counts)

    with col2:
        st.subheader("Pazar Başına Başarı Oranı")
        pazar_counts = df_health['Pazar'].value_counts()
        pazar_errors = df_health[df_health['Kategori'] == 'Veri Çekilemedi']['Pazar'].value_counts()

        df_pazar_health = pd.DataFrame({'Toplam Kayıt': pazar_counts, 'Kayıp Sayısı': pazar_errors}).fillna(0).astype(
            int)
        df_pazar_health['Başarı Oranı (%)'] = (df_pazar_health['Toplam Kayıt'] - df_pazar_health['Kayıp Sayısı']) / \
                                              df_pazar_health['Toplam Kayıt'] * 100

        st.dataframe(
            df_pazar_health,
            column_config={
                "Başarı Oranı (%)": st.column_config.ProgressColumn(
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
            },
            use_container_width=True
        )

        st.info("""
        **`source_note` Anlamları:**
        - **Başarılı (Yeşil):** `our_lowest_label`, `min_from_list`, `en_dusuk...` vb. (Fiyat bir etiket veya listeden bulundu)
        - **Veri Çekilemedi (Kırmızı):**
            - `CRASH_OR_...`: Tarayıcı çöktü veya kritik hata.
            - `main_block_id_timeout`: VPN yavaş, sayfa yüklendi ama otel bloğu 15-20 saniyede gelmedi.
            - `not_found`: Sayfa yüklendi, otel bloğu bulundu, ancak içinde **hiçbir** fiyat bulunamadı (Otel dolu olabilir).
        """)

    with st.expander("Veri Çekilemeyen Kayıtların Dökümünü İncele"):
        df_hatalar = df_health[df_health['Kategori'] == 'Veri Çekilemedi']
        st.dataframe(df_hatalar, use_container_width=True)
        st.caption(f"Toplam {len(df_hatalar)} adet veri çekilemeyen kayıt bulundu.")

    st.divider()
    display_raw_data_section(df_tr, df_us, df_de, df_uk)

# -----------------------------------------------------------------------------
# 5.4. FİYAT TAHMİNLEMESİ SEKMESİ
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=True)
def get_price_forecast(df_otel: pd.DataFrame, days_to_forecast: int) -> Optional[pd.DataFrame]:
    """
    Prophet kütüphanesini kullanarak bir otelin Pazar Ortalaması (mean_fiyat_tl)
    fiyatı için gelecek 'days_to_forecast' gününü tahmin eder.
    """
    if not PROPHET_AVAILABLE:
        st.error("Tahminleme için 'prophet' kütüphanesi yüklenmemiş. (pip install prophet)")
        return None

    try:
        # 1. Veriyi Prophet formatına hazırla (ds, y)
        # Tahminleme için en stabil olan Pazar Ortalaması
        df_prophet = df_otel[['checkin', 'mean_fiyat_tl']].copy()
        df_prophet = df_prophet.rename(columns={'checkin': 'ds', 'mean_fiyat_tl': 'y'})

        # Sadece fiyatı 0'dan büyük olanları al (dolu günleri modele katma)
        df_prophet = df_prophet[df_prophet['y'] > 0]

        if len(df_prophet) < 7:
            st.warning(f"Tahmin modeli için yetersiz veri (en az 7 gün gerekli, {len(df_prophet)} gün bulundu).")
            return None

        # 2. Modeli Kur ve Eğit
        # Sadece haftalık sezonsallığı etkinleştir
        model = Prophet(
            weekly_seasonality=True,
            daily_seasonality=False,
            yearly_seasonality=False
        )
        model.fit(df_prophet)

        # 3. Gelecek için dataframe oluştur ve tahmin et
        future = model.make_future_dataframe(periods=days_to_forecast)
        forecast = model.predict(future)

        return forecast

    except Exception as e:
        st.error(f"Fiyat tahminleme modelinde hata oluştu: {e}")
        return None

def display_forecasting_tab(df_analiz: pd.DataFrame, secilen_otel: str):
    """Ana 'Fiyat Tahminlemesi' sekmesinin içeriğini yönetir."""

    st.header("🔮 Gelecek 7 Günlük Fiyat Tahminlemesi")

    if not PROPHET_AVAILABLE:
        st.error(
            "Bu özellik için `prophet` kütüphanesi gereklidir. Terminalden `pip install prophet` komutu ile yükleyebilirsiniz.")
        return

    if secilen_otel == "Tümü":
        st.info("Lütfen kenar çubuktan **tek bir otel** seçerek o otel için fiyat tahmini oluşturun.")
        return

    st.info(f"**{secilen_otel}** için Pazar Ortalaması (TL) fiyatı kullanılarak gelecek 7 günün tahmini yapılıyor...")

    days_to_forecast = 7

    df_analiz_copy = df_analiz.copy()
    df_analiz_copy['checkin'] = pd.to_datetime(df_analiz_copy['checkin'])

    forecast_data = get_price_forecast(df_analiz_copy, days_to_forecast)

    if forecast_data is None:
        st.error("Tahmin verisi oluşturulamadı. Lütfen 'Sistem Sağlığı' sekmesinden veri sayısını kontrol edin.")
        return

    if not PLOTLY_AVAILABLE: return

    # Geçmiş veriyi al (sadece 0'dan büyük olanlar)
    df_past = df_analiz_copy[df_analiz_copy['mean_fiyat_tl'] > 0]

    if df_past.empty:
        st.warning("Tahmin modeli için 'Gerçekleşen Fiyat' (eğitim verisi) bulunamadı.")
        return

    # Grafiği oluştur
    fig = go.Figure()

    # 1. Tahmin Güven Aralığı (yhat_lower, yhat_upper) - Mavi alan
    fig.add_trace(go.Scatter(
        x=forecast_data['ds'],
        y=forecast_data['yhat_upper'],
        mode='lines',
        line=dict(color='rgba(66, 165, 245, 0.3)'),
        name='Güven Aralığı (Üst)'
    ))
    fig.add_trace(go.Scatter(
        x=forecast_data['ds'],
        y=forecast_data['yhat_lower'],
        mode='lines',
        line=dict(color='rgba(66, 165, 245, 0.3)'),
        fill='tonexty',
        name='Güven Aralığı (Alt)',
    ))

    # 2. Tahmin Çizgisi (yhat) - Beyaz çizgi
    fig.add_trace(go.Scatter(
        x=forecast_data['ds'],
        y=forecast_data['yhat'],
        mode='lines',
        line=dict(color='white', width=3, dash='dash'),
        name='Tahmin Edilen Fiyat'
    ))

    # 3. Gerçekleşen Fiyat (Geçmiş) - Kırmızı noktalar
    fig.add_trace(go.Scatter(
        x=df_past['checkin'],
        y=df_past['mean_fiyat_tl'],
        mode='markers',
        marker=dict(color='red', size=8),
        name='Gerçekleşen Pazar Ortalaması'
    ))

    # 4. Tahmin başlangıç çizgisi
    last_known_date = df_past['checkin'].max()

    fig.add_vline(
        x=last_known_date,
        line_width=2,
        line_dash="dot",
        line_color="yellow"
    )

    y_pos = max(forecast_data['yhat_upper'].max(), df_past['mean_fiyat_tl'].max())

    fig.add_annotation(
        x=last_known_date,
        y=y_pos,
        yref="y",
        text="Tahmin Başlangıcı",
        font=dict(color="yellow", size=12),
        showarrow=False,
        yanchor="bottom",
        yshift=5
    )

    fig.update_layout(
        title=f"{secilen_otel} - Pazar Ortalaması Fiyat Tahmini (Gelecek {days_to_forecast} Gün)",
        xaxis_title='Tarih',
        yaxis_title='Tahmini Fiyat (₺)',
        hovermode='x unified',
        height=500,
        template='plotly_dark'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.success("""
    💡 **Tahmin Grafiği Yorumu:**
    - **Kırmızı Noktalar:** Veritabanındaki "gerçekleşmiş" pazar ortalaması fiyatlarıdır (Modelin eğitim verisi).
    - **Beyaz Kesikli Çizgi:** Modelin "olması gerektiğini" düşündüğü fiyattır (Geçmişe yönelik `fit` ve geleceğe yönelik `tahmin`).
    - **Mavi Alan (Güven Aralığı):** Fiyatın %95 olasılıkla bu bandın içinde kalacağını gösterir. Alan ne kadar genişse, tahmin o kadar belirsizdir.
    - **Sarı Çizgi:** Geçmiş verinin bittiği ve geleceğe yönelik "saf" tahminin başladığı yerdir.
    """)


# =============================================================================
# 6. EK BİLGİ VE HAM VERİ BÖLÜMLERİ
# =============================================================================

def display_about_section():
    """Hakkında bölümünü bir expander içinde gösterir."""
    st.divider()
    with st.expander("ℹ️ Sistem Hakkında Bilgi"):
        st.markdown("""
        ### 🎯 Projenin Amacı
        Bu sistem, otel işletmelerinin farklı dijital pazarlardaki fiyatlandırma stratejilerini
        **üç farklı yaklaşımla** analiz eder ve **öngörüsel tahminleme** yapar:

        1. **📈 Maksimum Gelir Stratejisi:** En yüksek pazar fiyatını hedefler, gelir kaybını minimize eder
        2. **💰 Rekabetçi Fiyat Stratejisi:** En düşük pazar fiyatını hedefler, doluluk oranını maksimize eder
        3. **⚖️ Dengeli Fiyat Stratejisi:** Fiyatı pazar ortalamasında tutarak tutarlılık sağlar.

        ### 📊 Nasıl Çalışır?
        1. **Veri Toplama:** 4 farklı pazardan (TR, US, DE, UK) otomatik fiyat verisi toplama
        2. **Kur Dönüşümü:** Güncel döviz kurları ile tüm fiyatlar TL'ye çevrilir
        3. **Strateji Analizi:** Maksimum, minimum ve ortalama fiyat hedefleri hesaplanır
        4. **Eylem Önerileri:** Seçilen stratejiye göre (`%10`'dan fazla sapma varsa) somut fiyat değişikliği önerileri
        5. **Sistem Sağlığı:** Scraper'ların başarı/hata oranını `source_note` üzerinden analiz eder.
        6. **Öngörüsel Analiz (Tahminleme):** `Prophet` zaman serisi modelini kullanarak gelecek 7 günün pazar ortalaması fiyatını tahmin eder.

        ### 🔬 Teknik Altyapı
        - **Programlama Dili:** Python 3.11+
        - **Framework:** Streamlit
        - **Veri İşleme:** Pandas
        - **Görselleştirme:** Plotly
        - **Tahminleme (ML):** Prophet (Meta)
        - **Veri Kaynağı:** SQLite (4 farklı pazar)
        - **API:** Frankfurter.app (gerçek zamanlı döviz kurları)
        """)


def display_footer():
    """Sayfanın en altına bir altbilgi (footer) ekler."""
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #888; padding: 20px;'>
        <p style='font-size: 1.2em; font-weight: bold;'>Otel Gelir Yönetimi ve Fiyat Optimizasyon Sistemi</p>
        <p><i>Tri-Strategy & Predictive Edition (4 Pazar + Tahminleme & Sistem Sağlığı)</i></p>
        <p>TÜBİTAK 2209-A/B Üniversite Öğrencileri Araştırma Projeleri | 2025</p>
        <p style='font-size: 12px;'>Bu sistem bilimsel araştırma amaçlı geliştirilmiştir.</p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# 7. ANA UYGULAMA AKIŞI (MAIN)
# =============================================================================

def main():
    """Ana Streamlit uygulama akışını yönetir."""

    # 1. Sayfa Ayarları ve Başlık
    setup_page()
    inject_css()
    st.title("🏨 Otel Gelir Yönetimi ve Fiyat Optimizasyon Sistemi")
    st.markdown("""
    <div class='info-box'>
    <b>📊 Tri-Stratejili & Öngörüsel Gelir Yönetim Sistemi</b><br>
    Bu sistem, otellerinizin farklı dijital pazarlardaki (TR, US, DE, UK) fiyat tutarsızlıklarını tespit eder, 
    <b>üç farklı optimizasyon stratejisi</b> sunar, <b>gelecek 7 gün için fiyat tahmini</b> yapar ve <b>scraper sistem sağlığını</b> izler.
    </div>
    """, unsafe_allow_html=True)

    # 2. Veri Yükleme ve Birleştirme
    with st.spinner('🔄 4 Pazardaki veriler ve sağlık kayıtları yükleniyor...'):
        df_tr, df_us, df_de, df_uk = load_data()

    if df_tr is None and df_us is None and df_de is None and df_uk is None:
        st.error(
            "⚠️ HİÇBİR VERİ KAYNAĞI YÜKLENEMEDİ. Scraper'ları çalıştırdığınızdan ve veritabanı yollarının doğru olduğundan emin olun.")
        st.stop()

    empty_df_tr = pd.DataFrame(
        columns=['otel', 'checkin', 'fiyat_tl', 'para_birimi_tl', 'cekilme_zamani_tr', 'source_note_tr'])
    empty_df_us = pd.DataFrame(
        columns=['otel', 'checkin', 'fiyat_usd', 'para_birimi_usd', 'cekilme_zamani_us', 'source_note_us'])
    empty_df_de = pd.DataFrame(
        columns=['otel', 'checkin', 'fiyat_eur', 'para_birimi_eur', 'cekilme_zamani_de', 'source_note_de'])
    empty_df_uk = pd.DataFrame(
        columns=['otel', 'checkin', 'fiyat_gbp', 'para_birimi_gbp', 'cekilme_zamani_uk', 'source_note_uk'])

    df_tr = df_tr if df_tr is not None else empty_df_tr
    df_us = df_us if df_us is not None else empty_df_us
    df_de = df_de if df_de is not None else empty_df_de
    df_uk = df_uk if df_uk is not None else empty_df_uk

    df_merged = merge_dataframes(df_tr, df_us, df_de, df_uk)

    if df_merged.empty or len(df_merged[df_merged['otel'].notna()]) == 0:
        st.error("⚠️ Veritabanları yüklendi ancak içlerinde hiç veri bulunamadı. Lütfen scraper'ları çalıştırın.")
        st.stop()

    # 3. Kenar Çubuğu ve Filtreler
    strateji, strateji_mod, secilen_otel, kur_usd_tl, kur_eur_tl, kur_gbp_tl = build_sidebar(df_merged)

    df_filtrelenmis = df_merged.copy()
    if secilen_otel != "Tümü":
        df_filtrelenmis = df_filtrelenmis[df_filtrelenmis['otel'] == secilen_otel]

    if df_filtrelenmis.empty:
        st.warning("⚠️ Seçilen otel için veri bulunamadı.")
        st.stop()

    # 4. Strateji Hesaplamaları
    df_analiz, hedef_aciklama = calculate_strategy_dataframe(
        df_filtrelenmis, strateji_mod, kur_usd_tl, kur_eur_tl, kur_gbp_tl
    )

    # 5. Dashboard Gösterimi

    tab_genel, tab_strateji, tab_saglik, tab_tahmin = st.tabs([
        "📈 Genel Bakış & KPI'lar",
        "💡 Strateji Önerileri",
        "🩺 Sistem Sağlığı & Ham Veri",
        "🔮 Fiyat Tahminlemesi (Prophet)"
    ])

    with tab_genel:
        display_overview_tab(df_analiz, strateji, strateji_mod, secilen_otel, hedef_aciklama)

    with tab_strateji:
        display_strategy_tab(
            df_analiz, strateji, strateji_mod, hedef_aciklama,
            kur_usd_tl, kur_eur_tl, kur_gbp_tl, secilen_otel
        )

    with tab_saglik:
        display_health_tab(df_tr, df_us, df_de, df_uk)

    with tab_tahmin:
        display_forecasting_tab(df_analiz, secilen_otel)

    # 6. Ek Bilgi ve Footer
    display_about_section()
    display_footer()


if __name__ == "__main__":
    if not PLOTLY_AVAILABLE:
        st.error("Kritik Hata: Plotly kütüphanesi bulunamadı. Dashboard başlatılamıyor.")
        st.info("Lütfen terminalden 'pip install plotly' komutunu çalıştırın.")
    else:

        main()
