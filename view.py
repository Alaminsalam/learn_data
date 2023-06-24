import streamlit as st

def view_home():
    st.title("Analisis Sentimen Review Pengguna Play Store")
    st.markdown("""<p class="description">
    Aplikasi berbasis website sebagai analisis sentimen menggunakan Metode K-Nearest Neighbor
    </p>""", unsafe_allow_html=True)

    img_url = "https://i.gadgets360cdn.com/large/google_play_1559395346287.jpg"
    st.image(img_url, caption='My Image', width=600)
