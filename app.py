import streamlit as st
import numpy as np
#import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="Color Picker dari Gambar", layout="centered")
st.title("Dominant Color Picker dari Gambar")

uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    # Resize agar proses lebih cepat
    img = image.resize((150, 150))
    img_np = np.array(img)

    # Ubah ke format RGB (kadang image bisa dalam mode RGBA)
    if img_np.shape[-1] == 4:
        img_np = img_np[:, :, :3]

    img_data = img_np.reshape((-1, 3))  # (jumlah_pixel, 3)

    # Jalankan KMeans
    k = 5
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(img_data)
    colors = kmeans.cluster_centers_.astype(int)

    # Buat palet warna
    def plot_colors(colors):
        height = 50
        palette = np.zeros((height, k * height, 3), dtype=np.uint8)
        for idx, color in enumerate(colors):
            palette[:, idx * height:(idx + 1) * height, :] = color
        return palette

    palette = plot_colors(colors)
    st.image(palette, caption="Palet Warna Dominan", use_column_width=False)

    # Tampilkan kode warna dalam bentuk HEX dan color picker
    st.markdown("### Kode Warna (HEX)")
    for i, color in enumerate(colors):
        hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
        st.color_picker(f"Warna #{i+1}", hex_color, key=i)
