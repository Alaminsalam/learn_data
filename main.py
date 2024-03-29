import streamlit as st
from app import view_scraper
from view import view_home
from data import process

    
def main():
    st.set_page_config(page_title="AnalisisSentimen App", page_icon="👨‍🎓", layout="wide")
    st.sidebar.title("Dashboard")
    pages = {
    "Home": view_home,
    "Scraper": view_scraper,
    "Insert Data" : process
    }
    page = st.sidebar.radio("Fitur", pages.keys())
    pages[page]()
    
if __name__ == "__main__":
    main()

