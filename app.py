import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import urllib.request

# ======================================================
# 1. KONFIGURASI HALAMAN & GAYA PLOT
# ======================================================
st.set_page_config(page_title="Analisis Kontras Citra", layout="wide", page_icon="🎨")
plt.style.use('seaborn-v0_8-whitegrid') # Menerapkan gaya plot yang lebih modern

# ======================================================
# 2. JUDUL & DESKRIPSI BARU YANG LEBIH ELEGAN
# ======================================================
st.title("🎨 Eksplorasi Citra: Dari Histogram ke Peningkatan Kontras")
st.markdown("Unggah sebuah citra dan lihat bagaimana histogram dapat digunakan untuk segmentasi biner dan peningkatan kontras secara otomatis dan interaktif.")

# ======================================================
# 3. SIDEBAR YANG LEBIH TERSTRUKTUR
# ======================================================
with st.sidebar:
    st.header("⚙️ Pengaturan Input")
    use_example = st.checkbox("Gunakan contoh gambar (Lena)", value=True)
    upload = st.file_uploader("Atau upload gambar Anda", type=["png", "jpg", "jpeg"])
    
    st.header("🛠️ Opsi Thresholding")
    # Opsi untuk memilih metode thresholding
    thresh_method = st.radio(
        "Pilih Metode Thresholding",
        ("Rata-rata 2 Puncak", "Otsu Otomatis"),
        help="Otsu adalah metode standar industri untuk thresholding otomatis."
    )
    manual_thresh = st.checkbox("Atur threshold manual", value=False)
    threshold_manual_val = st.slider("Nilai threshold manual", 0, 255, 128) if manual_thresh else None

# ======================================================
# 4. FUNGSI HELPER
# ======================================================
@st.cache_data
def load_image_from_bytes(image_bytes):
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Langsung konversi ke RGB

@st.cache_data
def load_image_from_url(url):
    with urllib.request.urlopen(url) as resp:
        image = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Langsung konversi ke RGB

# ======================================================
# 5. LOGIKA UTAMA & PEMROSESAN GAMBAR
# ======================================================
if upload is not None:
    image_bytes = upload.read()
    img_rgb = load_image_from_bytes(image_bytes)
elif use_example:
    example_url = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg'
    try:
        img_rgb = load_image_from_url(example_url)
    except Exception as e:
        st.error(f"Gagal mengunduh contoh gambar: {e}")
        img_rgb = None
else:
    st.info("👈 Silakan upload gambar di sidebar atau centang 'Gunakan contoh gambar' untuk memulai.")
    img_rgb = None

if img_rgb is not None:
    # --- Proses Awal ---
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hist_gray_values = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()

    # --- Tampilkan Gambar Utama ---
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_rgb, caption='Citra Asli (RGB)', use_container_width=True)
    with col2:
        st.image(gray, caption='Citra Grayscale', use_container_width=True)
    
    # ======================================================
    # 6. PENGGUNAAN st.tabs UNTUK LAYOUT YANG BERSIH
    # ======================================================
    tab1, tab2, tab3 = st.tabs(["📊 Analisis Histogram", "🔳 Thresholding Biner", "✨ Equalisasi Histogram"])

    with tab1:
        st.subheader("Visualisasi Sebaran Intensitas Piksel")
        t1_col1, t1_col2 = st.columns(2)
        with t1_col1:
            # Plot RGB
            fig_rgb, ax_rgb = plt.subplots(figsize=(7,4))
            for i, color in enumerate(['R', 'G', 'B']):
                hist_rgb = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
                ax_rgb.plot(hist_rgb, color=color.lower(), label=f'Channel {color}')
            ax_rgb.set_title('Histogram RGB')
            ax_rgb.set_xlabel('Intensitas')
            ax_rgb.set_ylabel('Frekuensi')
            ax_rgb.legend()
            st.pyplot(fig_rgb, use_container_width=True)
        with t1_col2:
            # Plot Grayscale
            fig_gray, ax_gray = plt.subplots(figsize=(7,4))
            ax_gray.plot(hist_gray_values, color='gray')
            ax_gray.set_title('Histogram Grayscale')
            ax_gray.set_xlabel('Intensitas')
            ax_gray.set_ylabel('Frekuensi')
            st.pyplot(fig_gray, use_container_width=True)
            
        with st.expander("Lihat Tabel Data Histogram Grayscale"):
            df_hist = pd.DataFrame({"Intensitas": np.arange(256), "Jumlah Piksel": hist_gray_values})
            st.dataframe(df_hist)

    with tab2:
        st.subheader("Segmentasi Citra Menjadi Hitam & Putih")
        
        # --- Logika penentuan Threshold ---
        if thresh_method == "Rata-rata 2 Puncak":
            peak_indices = np.argsort(hist_gray_values)[-2:]
            peak1, peak2 = peak_indices[0], peak_indices[1]
            threshold_auto = int(np.round((peak1 + peak2) / 2))
            st.info(f"Metode: Rata-rata 2 Puncak. Ditemukan puncak di **{peak1}** dan **{peak2}**. Threshold otomatis: **{threshold_auto}**.")
        else: # Otsu Otomatis
            threshold_auto, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            st.info(f"Metode: Otsu Otomatis. Ditemukan threshold optimal di **{int(threshold_auto)}**.")

        # Tentukan threshold final
        threshold_val = threshold_manual_val if manual_thresh else int(threshold_auto)
        
        _, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
        
        t2_col1, t2_col2 = st.columns(2)
        with t2_col1:
            st.image(binary, caption=f'Citra Biner (Threshold={threshold_val})', use_container_width=True)
        with t2_col2:
            # Plot histogram dengan garis threshold
            fig_thresh, ax_thresh = plt.subplots(figsize=(7,4))
            ax_thresh.plot(hist_gray_values, color='gray')
            ax_thresh.axvline(threshold_val, color='r', linestyle='--', label=f'Threshold={threshold_val}')
            ax_thresh.set_title('Histogram dengan Garis Threshold')
            ax_thresh.set_xlabel('Intensitas')
            ax_thresh.set_ylabel('Frekuensi')
            ax_thresh.legend()
            st.pyplot(fig_thresh, use_container_width=True)
            
        st.download_button("Download Citra Biner (.png)", cv2.imencode('.png', binary)[1].tobytes(), 'binary.png', 'image/png')

    with tab3:
        st.subheader("Peningkatan Kontras Citra Otomatis")
        equalized = cv2.equalizeHist(gray)
        
        t3_col1, t3_col2 = st.columns(2)
        with t3_col1:
            st.image(gray, caption='Gambar Sebelum Equalization', use_container_width=True)
            fig_hist_before, ax_before = plt.subplots(figsize=(7,4))
            ax_before.hist(gray.ravel(), bins=256, range=[0,256], color='gray')
            ax_before.set_title('Histogram Sebelum')
            st.pyplot(fig_hist_before, use_container_width=True)
            
        with t3_col2:
            st.image(equalized, caption='Gambar Sesudah Equalization', use_container_width=True)
            fig_hist_after, ax_after = plt.subplots(figsize=(7,4))
            ax_after.hist(equalized.ravel(), bins=256, range=[0,256], color='blue')
            ax_after.set_title('Histogram Sesudah (Lebih Merata)')
            st.pyplot(fig_hist_after, use_container_width=True)
            
        st.download_button("Download Citra Equalized (.png)", cv2.imencode('.png', equalized)[1].tobytes(), 'equalized.png', 'image/png')