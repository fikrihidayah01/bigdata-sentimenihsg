import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import ast
import difflib
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Dashboard Sentimen IHSG",
    layout="wide",
    page_icon="📈"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
    /* Signal badge */
    .signal-box {
        padding: 1.2rem 2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .signal-buy  { background: #e6f9f0; color: #1a7f4b; border: 2px solid #1a7f4b; }
    .signal-sell { background: #fdecea; color: #c0392b; border: 2px solid #c0392b; }
    .signal-neutral { background: #f0f0f0; color: #555; border: 2px solid #aaa; }

    /* Lag highlight card */
    .lag-card {
        background: #f8f9ff;
        border-left: 4px solid #4361ee;
        padding: 0.8rem 1.2rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }
    .lag-card b { color: #4361ee; }

    /* Metric override */
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 0. SENTIMENT LEXICONS (For WordCloud Filtering)
# ==========================================
positif_lex = {
    "cuan", "uptrend", "bullish", "rebound", "akumulasi", "serok",
    "terbang", "hijau", "naik", "untung", "bagus", "mantap", "roket",
    "hold", "dividen", "profit", "buy", "beli", "murah", "diskon",
    "potensi", "peluang", "aman"
}
negatif_lex = {
    "nyangkut", "cutloss", "cl", "turun", "longsor", "junam", "anjlok",
    "downtrend", "bearish", "distribusi", "merah", "rugi", "parah",
    "nyungsep", "arb", "bawah", "koreksi", "jatuh", "jual", "sell",
    "mahal", "hancur", "jebol", "gagal", "panik"
}

fuzzy_wc_cache = {}

def get_base_word(kata):
    if kata in fuzzy_wc_cache:
        return fuzzy_wc_cache[kata]
    if kata in positif_lex or kata in negatif_lex:
        fuzzy_wc_cache[kata] = kata
        return kata
    match_pos = difflib.get_close_matches(kata, positif_lex, n=1, cutoff=0.8)
    if match_pos:
        fuzzy_wc_cache[kata] = match_pos[0]
        return match_pos[0]
    match_neg = difflib.get_close_matches(kata, negatif_lex, n=1, cutoff=0.8)
    if match_neg:
        fuzzy_wc_cache[kata] = match_neg[0]
        return match_neg[0]
    fuzzy_wc_cache[kata] = None
    return None

# ==========================================
# 1. LOAD DATA
# ==========================================
@st.cache_data
def load_data():
    try:
        df_merge = pd.read_csv("output/final_merge.csv")
        df_merge["date"] = pd.to_datetime(df_merge["date"])

        df_sent = pd.read_csv("output/sentiment_result.csv")
        df_sent["timestamp"] = pd.to_datetime(df_sent["timestamp"])

        if df_sent["timestamp"].dt.tz is None:
            df_sent["timestamp"] = (
                df_sent["timestamp"]
                .dt.tz_localize("UTC")
                .dt.tz_convert("Asia/Jakarta")
            )
        else:
            df_sent["timestamp"] = df_sent["timestamp"].dt.tz_convert("Asia/Jakarta")

        df_sent["date"] = pd.to_datetime(df_sent["timestamp"].dt.date)
        return df_merge, df_sent

    except FileNotFoundError:
        st.error(
            "Data tidak ditemukan! Harap jalankan 'python run_pipeline.py' terlebih dahulu."
        )
        st.stop()

df_merge, df_sent = load_data()

# ==========================================
# 2. SIDEBAR FILTER
# ==========================================
st.sidebar.header("🗓️ Filter Tanggal")
min_date = df_merge["date"].min().date()
max_date = df_merge["date"].max().date()

start_date, end_date = st.sidebar.date_input(
    "Pilih Rentang Tanggal",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

mask = (df_merge["date"].dt.date >= start_date) & (
    df_merge["date"].dt.date <= end_date
)
filtered_merge = df_merge.loc[mask].copy()

mask_sent = (df_sent["date"].dt.date >= start_date) & (
    df_sent["date"].dt.date <= end_date
)
filtered_sent = df_sent.loc[mask_sent].copy()

# ==========================================
# 3. HEADER
# ==========================================
st.title("📈 Dashboard Analisis Sentimen Threads vs IHSG")
st.markdown(
    "Perbandingan langsung antara metode kamus kustom **Lexicon** "
    "dengan Machine Learning AI **IndoBERT** — dilengkapi analisis lag dan sinyal prediksi."
)

# ==========================================
# 4. TABS
# ==========================================
tab_lexicon, tab_indobert, tab_lag, tab_pred = st.tabs([
    "📖 Metode Lexicon",
    "🤖 Metode IndoBERT",
    "⏱️ Time-Lag Analysis",
    "🔮 Prediksi & Sinyal",
])


# ──────────────────────────────────────────
# HELPER: EXISTING DASHBOARD RENDERER
# ──────────────────────────────────────────
def render_dashboard(df_merge_tab, df_sent_tab, suffix):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Data Threads", f"{len(df_sent_tab):,}")
    with col2:
        avg_sent = df_merge_tab[f"sentiment_index{suffix}"].mean()
        st.metric("Rata-Rata Indeks Sentimen", f"{avg_sent:.2f}")
    with col3:
        avg_ret = df_merge_tab["return"].mean()
        st.metric("Rata-Rata Return IHSG", f"{avg_ret:.2f}%")
    with col4:
        korelasi = (
            df_merge_tab[[f"sentiment_index{suffix}", "return"]]
            .corr()
            .iloc[0, 1]
        )
        st.metric("Nilai Korelasi (Pearson)", f"{korelasi:.3f}")

    st.divider()
    st.subheader("Tren Sentimen vs Return IHSG")

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df_merge_tab["date"],
        y=df_merge_tab[f"sentiment_index{suffix}"],
        name="Sentiment Index",
        line=dict(color="blue", width=2),
    ))
    fig1.add_trace(go.Scatter(
        x=df_merge_tab["date"],
        y=df_merge_tab["return"],
        name="Return IHSG (%)",
        yaxis="y2",
        line=dict(color="red", width=2, dash="dot"),
    ))
    fig1.update_layout(
        xaxis_title="Tanggal",
        yaxis=dict(
            title=dict(text="Sentiment Index", font=dict(color="blue")),
            tickfont=dict(color="blue"),
        ),
        yaxis2=dict(
            title=dict(text="Return IHSG (%)", font=dict(color="red")),
            tickfont=dict(color="red"),
            overlaying="y",
            side="right",
        ),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig1, use_container_width=True)

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.subheader("Scatter Plot Korelasi")
        fig2 = px.scatter(
            df_merge_tab,
            x=f"sentiment_index{suffix}",
            y="return",
            hover_data=["date"],
            labels={
                f"sentiment_index{suffix}": "Indeks Sentimen",
                "return": "Return IHSG (%)",
            },
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_chart2:
        st.subheader("Distribusi Label Sentimen")
        sent_counts = (
            df_sent_tab[f"sentiment{suffix}"].value_counts().reset_index()
        )
        sent_counts.columns = ["sentiment", "count"]
        fig3 = px.pie(
            sent_counts,
            values="count",
            names="sentiment",
            color="sentiment",
            color_discrete_map={
                "Positif": "#2ca02c",
                "Negatif": "#d62728",
                "Netral": "#7f7f7f",
            },
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.subheader("Wordcloud Kata Populer")

    def generate_wordcloud(df, sentiment_label):
        subset = df[df[f"sentiment{suffix}"] == sentiment_label]
        if len(subset) == 0:
            return None
        all_words = []
        for tokens_str in subset["tokens"]:
            try:
                tokens_list = ast.literal_eval(tokens_str)
                all_words.extend(tokens_list)
            except Exception:
                pass
        valid_lexicon = positif_lex if sentiment_label == "Positif" else negatif_lex
        filtered_words = []
        for w in all_words:
            base_w = get_base_word(w.lower())
            if base_w in valid_lexicon:
                filtered_words.append(base_w)
        text = " ".join(filtered_words)
        if not text.strip():
            return None
        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap="viridis" if sentiment_label == "Positif" else "magma",
        ).generate(text)
        return wc

    col_wc1, col_wc2 = st.columns(2)
    with col_wc1:
        st.markdown("**Kata-Kata Sentimen Positif**")
        wc_pos = generate_wordcloud(df_sent_tab, "Positif")
        if wc_pos:
            fig_pos, ax_pos = plt.subplots()
            ax_pos.imshow(wc_pos, interpolation="bilinear")
            ax_pos.axis("off")
            st.pyplot(fig_pos)
        else:
            st.info("Tidak ada data sentimen positif di rentang ini.")

    with col_wc2:
        st.markdown("**Kata-Kata Sentimen Negatif**")
        wc_neg = generate_wordcloud(df_sent_tab, "Negatif")
        if wc_neg:
            fig_neg, ax_neg = plt.subplots()
            ax_neg.imshow(wc_neg, interpolation="bilinear")
            ax_neg.axis("off")
            st.pyplot(fig_neg)
        else:
            st.info("Tidak ada data sentimen negatif di rentang ini.")


# ──────────────────────────────────────────
# TAB 1 & 2: EXISTING
# ──────────────────────────────────────────
with tab_lexicon:
    render_dashboard(filtered_merge, filtered_sent, "_lexicon")

with tab_indobert:
    render_dashboard(filtered_merge, filtered_sent, "_indobert")


# ──────────────────────────────────────────
# TAB 3: TIME-LAG ANALYSIS
# ──────────────────────────────────────────
with tab_lag:
    st.header("⏱️ Time-Lag Analysis")
    st.markdown(
        "Apakah sentimen **hari ini** memprediksi return IHSG **beberapa hari ke depan**? "
        "Atau justru pasar bereaksi **setelah** pergerakan harga? "
        "Cross-correlation di bawah menjawabnya."
    )

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Pengaturan Lag")
    max_lag = st.sidebar.slider(
        "Maksimum Lag (hari)", min_value=3, max_value=14, value=7
    )

    def compute_cross_correlation(df, sentiment_col, lags):
        """
        Lag > 0 : sentimen N hari lalu vs return hari ini
                  (sentimen leading / prediktif)
        Lag < 0 : return N hari lalu vs sentimen hari ini
                  (harga leading / reaktif)
        Lag = 0 : same-day correlation
        """
        results = []
        df = df.sort_values("date").dropna(subset=[sentiment_col, "return"])
        for lag in lags:
            if lag >= 0:
                s = df[sentiment_col].shift(lag)
            else:
                s = df[sentiment_col].shift(lag)
            combined = pd.concat(
                [s.rename("sent"), df["return"].rename("ret")], axis=1
            ).dropna()
            if len(combined) < 5:
                results.append({"lag": lag, "correlation": np.nan, "p_value": np.nan})
                continue
            r, p = stats.pearsonr(combined["sent"], combined["ret"])
            results.append({"lag": lag, "correlation": r, "p_value": p})
        return pd.DataFrame(results)

    lags = list(range(-max_lag, max_lag + 1))

    df_lag_lex = compute_cross_correlation(filtered_merge, "sentiment_index_lexicon", lags)
    df_lag_bert = compute_cross_correlation(filtered_merge, "sentiment_index_indobert", lags)

    # --- Chart
    fig_lag = go.Figure()

    fig_lag.add_trace(go.Bar(
        x=df_lag_lex["lag"],
        y=df_lag_lex["correlation"],
        name="Lexicon",
        marker_color=[
            "#2ca02c" if v > 0 else "#d62728"
            for v in df_lag_lex["correlation"].fillna(0)
        ],
        opacity=0.75,
    ))
    fig_lag.add_trace(go.Scatter(
        x=df_lag_bert["lag"],
        y=df_lag_bert["correlation"],
        name="IndoBERT",
        mode="lines+markers",
        line=dict(color="#9467bd", width=2),
        marker=dict(size=7),
    ))

    # Significance threshold lines (±0.3 as rough guide)
    fig_lag.add_hline(y=0.3, line_dash="dash", line_color="gray",
                      annotation_text="r = +0.30", annotation_position="top right")
    fig_lag.add_hline(y=-0.3, line_dash="dash", line_color="gray",
                      annotation_text="r = -0.30", annotation_position="bottom right")
    fig_lag.add_vline(x=0, line_color="black", line_width=1)

    fig_lag.update_layout(
        title="Cross-Correlation: Sentimen vs Return IHSG per Lag",
        xaxis=dict(
            title="Lag (hari) — negatif: harga duluan | positif: sentimen duluan",
            tickmode="linear",
            dtick=1,
        ),
        yaxis=dict(title="Pearson r"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        bargap=0.1,
    )
    st.plotly_chart(fig_lag, use_container_width=True)

    # --- Interpretation cards
    st.subheader("📌 Temuan Utama")

    def best_lag_info(df_lag, name):
        valid = df_lag.dropna(subset=["correlation"])
        if valid.empty:
            return
        idx_best = valid["correlation"].abs().idxmax()
        best = valid.loc[idx_best]
        lag_val = int(best["lag"])
        r_val = best["correlation"]
        p_val = best["p_value"]
        sig = "✅ Signifikan (p < 0.05)" if p_val < 0.05 else "⚠️ Tidak signifikan (p ≥ 0.05)"

        if lag_val > 0:
            direction = f"Sentimen **{lag_val} hari sebelumnya** → return IHSG hari ini (sentimen bersifat **leading/prediktif**)"
        elif lag_val < 0:
            direction = f"Return IHSG **{abs(lag_val)} hari sebelumnya** → sentimen hari ini (pasar bersifat **leading/reaktif**)"
        else:
            direction = "Korelasi terkuat terjadi pada **hari yang sama** (no lead/lag)"

        st.markdown(f"""
<div class="lag-card">
    <b>{name}</b> — Lag terkuat: <b>Lag {lag_val}</b> &nbsp;|&nbsp; r = <b>{r_val:.3f}</b> &nbsp;|&nbsp; {sig}<br>
    {direction}
</div>
""", unsafe_allow_html=True)

    best_lag_info(df_lag_lex, "Lexicon")
    best_lag_info(df_lag_bert, "IndoBERT")

    # --- Tabel lengkap
    with st.expander("📊 Lihat Tabel Korelasi Lengkap"):
        df_lag_combined = df_lag_lex.rename(columns={
            "correlation": "r_lexicon", "p_value": "p_lexicon"
        }).merge(
            df_lag_bert.rename(columns={"correlation": "r_indobert", "p_value": "p_indobert"}),
            on="lag"
        )
        df_lag_combined = df_lag_combined.sort_values("lag").set_index("lag")

        def color_r(val):
            if pd.isna(val):
                return ""
            if val > 0.3:
                return "background-color: #c6efce; color: #276221"
            if val < -0.3:
                return "background-color: #ffc7ce; color: #9c0006"
            return ""

        styled = df_lag_combined.style.applymap(
            color_r, subset=["r_lexicon", "r_indobert"]
        ).format("{:.3f}", na_rep="-")

        st.dataframe(styled, use_container_width=True)

    st.info(
        "💡 **Cara baca:** Lag positif artinya sentimen X hari lalu berkorelasi dengan return hari ini. "
        "Jika lag +1 atau +2 memiliki r tinggi dan signifikan, sentimen threads bisa dijadikan sinyal "
        "**leading indicator** untuk IHSG."
    )


# ──────────────────────────────────────────
# TAB 4: PREDICTIVE ANALYSIS & TRADING SIGNAL
# ──────────────────────────────────────────
with tab_pred:
    st.header("🔮 Prediksi & Sinyal Trading")
    st.markdown(
        "Model logistic regression sederhana dilatih untuk memprediksi "
        "**arah return IHSG esok hari** (naik/turun) berdasarkan indeks sentimen hari ini."
    )

    # --- Settings
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Pengaturan Model")
    method = st.sidebar.radio(
        "Metode Sentimen",
        options=["Lexicon", "IndoBERT", "Keduanya (gabungan)"],
        index=2,
    )
    lag_pred = st.sidebar.slider(
        "Prediksi N Hari ke Depan", min_value=1, max_value=5, value=1
    )

    # --- Build features
    df_pred = filtered_merge.sort_values("date").copy()
    df_pred["next_return"] = df_pred["return"].shift(-lag_pred)
    df_pred["target"] = (df_pred["next_return"] > 0).astype(int)  # 1=naik, 0=turun

    if method == "Lexicon":
        feature_cols = ["sentiment_index_lexicon"]
    elif method == "IndoBERT":
        feature_cols = ["sentiment_index_indobert"]
    else:
        feature_cols = ["sentiment_index_lexicon", "sentiment_index_indobert"]

    df_pred_clean = df_pred.dropna(subset=feature_cols + ["target"])

    if len(df_pred_clean) < 20:
        st.warning(
            "Data terlalu sedikit untuk melatih model. "
            "Perluas rentang tanggal di sidebar."
        )
        st.stop()

    X = df_pred_clean[feature_cols].values
    y = df_pred_clean["target"].values

    # Simple walk-forward split (80% train / 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=500)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # ── KPI Row ──
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Akurasi Model", f"{acc*100:.1f}%",
                  help="Persentase prediksi arah yang benar pada data test.")
    with col_b:
        baseline = max(y_test.mean(), 1 - y_test.mean())
        st.metric("Baseline (naif)", f"{baseline*100:.1f}%",
                  help="Akurasi jika selalu prediksi kelas mayoritas.")
    with col_c:
        n_train = len(X_train)
        st.metric("Data Latih", f"{n_train} hari")
    with col_d:
        n_test = len(X_test)
        st.metric("Data Uji", f"{n_test} hari")

    st.divider()

    # ── LIVE SIGNAL ──
    st.subheader(f"📡 Sinyal Hari Ini (prediksi {lag_pred} hari ke depan)")

    latest_row = df_merge.sort_values("date").dropna(subset=feature_cols).iloc[-1]
    latest_date = latest_row["date"].strftime("%d %b %Y")
    latest_features = scaler.transform([latest_row[feature_cols].values])
    latest_prob = model.predict_proba(latest_features)[0][1]
    latest_pred = model.predict(latest_features)[0]

    sig_col1, sig_col2, sig_col3 = st.columns([1, 1, 2])

    with sig_col1:
        if latest_pred == 1:
            st.markdown(
                f'<div class="signal-box signal-buy">🟢 BUY SIGNAL<br>'
                f'<span style="font-size:0.9rem;font-weight:400">Prediksi: IHSG Naik</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="signal-box signal-sell">🔴 SELL SIGNAL<br>'
                f'<span style="font-size:0.9rem;font-weight:400">Prediksi: IHSG Turun</span></div>',
                unsafe_allow_html=True,
            )

    with sig_col2:
        st.metric("Probabilitas Naik", f"{latest_prob*100:.1f}%")
        st.caption(f"Berdasarkan data: {latest_date}")

    with sig_col3:
        lex_val = latest_row.get("sentiment_index_lexicon", np.nan)
        bert_val = latest_row.get("sentiment_index_indobert", np.nan)
        st.markdown("**Indeks sentimen terkini:**")
        st.markdown(f"- 📖 Lexicon: `{lex_val:.3f}`" if not pd.isna(lex_val) else "- 📖 Lexicon: N/A")
        st.markdown(f"- 🤖 IndoBERT: `{bert_val:.3f}`" if not pd.isna(bert_val) else "- 🤖 IndoBERT: N/A")
        st.caption(
            "⚠️ Sinyal ini hanya untuk tujuan riset, bukan rekomendasi investasi."
        )

    st.divider()

    # ── CHARTS ROW ──
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Confusion Matrix")
        labels = ["Turun (0)", "Naik (1)"]
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=labels, y=labels,
            color_continuous_scale="Blues",
            labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
        )
        fig_cm.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_right:
        st.subheader("Probabilitas Prediksi vs Return Aktual")
        df_test_plot = df_pred_clean.iloc[split_idx:].copy()
        df_test_plot["prob_naik"] = y_prob
        df_test_plot["arah_aktual"] = np.where(
            df_test_plot["next_return"] > 0, "Naik ✅", "Turun ❌"
        )

        fig_prob = px.scatter(
            df_test_plot,
            x="date",
            y="prob_naik",
            color="arah_aktual",
            color_discrete_map={"Naik ✅": "#2ca02c", "Turun ❌": "#d62728"},
            labels={"prob_naik": "Probabilitas Naik", "date": "Tanggal"},
            hover_data=["next_return"],
        )
        fig_prob.add_hline(y=0.5, line_dash="dash", line_color="gray",
                           annotation_text="threshold 0.5")
        st.plotly_chart(fig_prob, use_container_width=True)

    # ── BACKTEST ──
    st.subheader("📈 Simulasi Backtest — Ikuti Sinyal vs Buy & Hold")
    st.caption(
        "Strategi: jika model prediksi naik → masuk (long); "
        "jika prediksi turun → keluar (tidak pegang). Tidak ada leverage, tidak ada biaya transaksi."
    )

    df_bt = df_test_plot.copy()
    df_bt = df_bt.sort_values("date").dropna(subset=["next_return"])
    df_bt["signal_return"] = np.where(
        df_bt["prob_naik"] >= 0.5, df_bt["next_return"], 0
    )

    df_bt["equity_signal"] = (1 + df_bt["signal_return"] / 100).cumprod() * 100
    df_bt["equity_bh"] = (1 + df_bt["next_return"] / 100).cumprod() * 100

    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(
        x=df_bt["date"], y=df_bt["equity_signal"],
        name="Strategi Sinyal Sentimen",
        line=dict(color="#4361ee", width=2.5),
        fill="tozeroy", fillcolor="rgba(67,97,238,0.08)",
    ))
    fig_bt.add_trace(go.Scatter(
        x=df_bt["date"], y=df_bt["equity_bh"],
        name="Buy & Hold IHSG",
        line=dict(color="#ef233c", width=2, dash="dot"),
    ))
    fig_bt.update_layout(
        yaxis_title="Nilai Portofolio (mulai dari 100)",
        xaxis_title="Tanggal",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    # Backtest summary
    total_signal = df_bt["equity_signal"].iloc[-1] - 100
    total_bh = df_bt["equity_bh"].iloc[-1] - 100
    win_rate = (df_bt[df_bt["prob_naik"] >= 0.5]["signal_return"] > 0).mean() * 100
    n_trades = (df_bt["prob_naik"] >= 0.5).sum()

    bt_col1, bt_col2, bt_col3, bt_col4 = st.columns(4)
    with bt_col1:
        delta_str = f"{total_signal - total_bh:+.1f}%"
        st.metric("Return Strategi", f"{total_signal:+.1f}%", delta=delta_str)
    with bt_col2:
        st.metric("Return Buy & Hold", f"{total_bh:+.1f}%")
    with bt_col3:
        st.metric("Win Rate Sinyal", f"{win_rate:.1f}%")
    with bt_col4:
        st.metric("Jumlah Sinyal Masuk", f"{n_trades}")

    st.info(
        "💡 **Catatan:** Backtest ini bersifat **in-sample** pada periode data yang sama. "
        "Hasil tidak menjamin performa masa depan. "
        "Gunakan sebagai titik awal riset, bukan panduan investasi nyata."
    )

    # ── Classification report ──
    with st.expander("📋 Lihat Classification Report Lengkap"):
        report = classification_report(y_test, y_pred, target_names=["Turun", "Naik"], output_dict=True)
        df_report = pd.DataFrame(report).T
        st.dataframe(df_report.style.format("{:.2f}", na_rep="-"), use_container_width=True)

# ──────────────────────────────────────────
# FOOTER TABLE
# ──────────────────────────────────────────
st.divider()
st.subheader("📄 Data Tabel Final (Merged)")
st.dataframe(filtered_merge, use_container_width=True)