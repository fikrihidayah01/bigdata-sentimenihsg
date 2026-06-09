import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import ast
import os
import difflib

st.set_page_config(page_title="Dashboard Sentimen IHSG", layout="wide", page_icon="📈")

# ==========================================
# 0. SENTIMENT LEXICONS (For WordCloud Filtering)
# ==========================================
positif_lex = {"cuan", "uptrend", "bullish", "rebound", "akumulasi", "serok", "terbang", "hijau", "naik", "untung", "bagus", "mantap", "roket", "hold", "dividen", "profit", "buy", "beli", "murah", "diskon", "potensi", "peluang", "aman"}
negatif_lex = {"nyangkut", "cutloss", "cl", "turun", "longsor", "junam", "anjlok", "downtrend", "bearish", "distribusi", "merah", "rugi", "parah", "nyungsep", "arb", "bawah", "koreksi", "jatuh", "jual", "sell", "mahal", "hancur", "jebol", "gagal", "panik"}

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
def load_data():
    try:
        # Load final merge
        df_merge = pd.read_csv("output/final_merge.csv")
        df_merge["date"] = pd.to_datetime(df_merge["date"])
        
        # Load raw sentiment
        df_sent = pd.read_csv("output/sentiment_result.csv")
        df_sent["timestamp"] = pd.to_datetime(df_sent["timestamp"])
        
        # Konversi ke WIB agar sinkron dengan file merge
        if df_sent['timestamp'].dt.tz is None:
            df_sent['timestamp'] = df_sent['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Asia/Jakarta')
        else:
            df_sent['timestamp'] = df_sent['timestamp'].dt.tz_convert('Asia/Jakarta')
            
        df_sent["date"] = pd.to_datetime(df_sent["timestamp"].dt.date)
        
        return df_merge, df_sent
    except FileNotFoundError:
        st.error("Data tidak ditemukan! Harap jalankan 'python run_pipeline.py' terlebih dahulu.")
        st.stop()

df_merge, df_sent = load_data()

# ==========================================
# 2. SIDEBAR FILTER
# ==========================================
st.sidebar.header("Filter Tanggal")
min_date = df_merge["date"].min().date()
max_date = df_merge["date"].max().date()

start_date, end_date = st.sidebar.date_input(
    "Pilih Rentang Tanggal",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Filter dataframe
mask = (df_merge["date"].dt.date >= start_date) & (df_merge["date"].dt.date <= end_date)
filtered_merge = df_merge.loc[mask]

mask_sent = (df_sent["date"].dt.date >= start_date) & (df_sent["date"].dt.date <= end_date)
filtered_sent = df_sent.loc[mask_sent]

# ==========================================
# 3. HEADER
# ==========================================
st.title("📈 Dashboard Analisis Sentimen Threads vs IHSG")
st.markdown("Perbandingan langsung antara metode kamus kustom (Lexicon) dengan Machine Learning AI (IndoBERT).")

# ==========================================
# 4. TABS
# ==========================================
tab_lexicon, tab_indobert = st.tabs(["Metode Lexicon (Aturan Baku)", "Metode IndoBERT (AI)"])

def render_dashboard(df_merge_tab, df_sent_tab, suffix):
    """
    Fungsi untuk merender isi dashboard sesuai tab.
    suffix: '_lexicon' atau '_indobert'
    """
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Data Threads", f"{len(df_sent_tab):,}")
    with col2:
        avg_sent = df_merge_tab[f'sentiment_index{suffix}'].mean()
        st.metric("Rata-Rata Indeks Sentimen", f"{avg_sent:.2f}")
    with col3:
        avg_ret = df_merge_tab['return'].mean()
        st.metric("Rata-Rata Return IHSG", f"{avg_ret:.2f}%")
    with col4:
        korelasi = df_merge_tab[[f'sentiment_index{suffix}', 'return']].corr().iloc[0,1]
        st.metric("Nilai Korelasi (Pearson)", f"{korelasi:.3f}")

    st.divider()

    st.subheader("Tren Sentimen vs Return IHSG")

    fig1 = go.Figure()

    # Sentiment Line
    fig1.add_trace(go.Scatter(
        x=df_merge_tab["date"], y=df_merge_tab[f"sentiment_index{suffix}"],
        name="Sentiment Index", line=dict(color='blue', width=2)
    ))

    # Return Line (Secondary Y)
    fig1.add_trace(go.Scatter(
        x=df_merge_tab["date"], y=df_merge_tab["return"],
        name="Return IHSG (%)", yaxis="y2", line=dict(color='red', width=2, dash='dot')
    ))

    fig1.update_layout(
        xaxis_title="Tanggal",
        yaxis=dict(title=dict(text="Sentiment Index", font=dict(color="blue")), tickfont=dict(color="blue")),
        yaxis2=dict(title=dict(text="Return IHSG (%)", font=dict(color="red")), tickfont=dict(color="red"),
                    overlaying="y", side="right"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig1, use_container_width=True)

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Scatter Plot Korelasi")
        fig2 = px.scatter(
            df_merge_tab, x=f"sentiment_index{suffix}", y="return", 
            hover_data=["date"],
            labels={f"sentiment_index{suffix}": "Indeks Sentimen", "return": "Return IHSG (%)"}
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_chart2:
        st.subheader("Distribusi Label Sentimen")
        # Rename agar gampang plotting
        sent_counts = df_sent_tab[f'sentiment{suffix}'].value_counts().reset_index()
        sent_counts.columns = ['sentiment', 'count']
        fig3 = px.pie(sent_counts, values='count', names='sentiment', color='sentiment',
                      color_discrete_map={'Positif':'#2ca02c', 'Negatif':'#d62728', 'Netral':'#7f7f7f'})
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # Wordcloud (Hanya akurat jika suffix Lexicon, tapi kita render untuk keduanya)
    st.subheader("Wordcloud Kata Populer")

    def generate_wordcloud(df, sentiment_label):
        subset = df[df[f'sentiment{suffix}'] == sentiment_label]
        if len(subset) == 0:
            return None
            
        all_words = []
        for tokens_str in subset['tokens']:
            try:
                tokens_list = ast.literal_eval(tokens_str)
                all_words.extend(tokens_list)
            except:
                pass
                
        valid_lexicon = positif_lex if sentiment_label == 'Positif' else negatif_lex
        
        filtered_words = []
        for w in all_words:
            base_w = get_base_word(w.lower())
            if base_w in valid_lexicon:
                filtered_words.append(base_w)
                
        text = " ".join(filtered_words)
        
        if not text.strip():
            return None
            
        wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis' if sentiment_label=='Positif' else 'magma').generate(text)
        return wc

    col_wc1, col_wc2 = st.columns(2)

    with col_wc1:
        st.markdown("**Kata-Kata Sentimen Positif**")
        wc_pos = generate_wordcloud(df_sent_tab, "Positif")
        if wc_pos:
            fig_pos, ax_pos = plt.subplots()
            ax_pos.imshow(wc_pos, interpolation='bilinear')
            ax_pos.axis("off")
            st.pyplot(fig_pos)
        else:
            st.info("Tidak ada data sentimen positif di rentang ini.")

    with col_wc2:
        st.markdown("**Kata-Kata Sentimen Negatif**")
        wc_neg = generate_wordcloud(df_sent_tab, "Negatif")
        if wc_neg:
            fig_neg, ax_neg = plt.subplots()
            ax_neg.imshow(wc_neg, interpolation='bilinear')
            ax_neg.axis("off")
            st.pyplot(fig_neg)
        else:
            st.info("Tidak ada data sentimen negatif di rentang ini.")


with tab_lexicon:
    render_dashboard(filtered_merge, filtered_sent, "_lexicon")

with tab_indobert:
    render_dashboard(filtered_merge, filtered_sent, "_indobert")

st.divider()
st.subheader("Data Tabel Final (Merged)")
st.dataframe(filtered_merge, use_container_width=True)
