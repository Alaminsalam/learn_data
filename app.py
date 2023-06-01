import pandas as pd
import pandas_profiling
import streamlit as st

from streamlit_pandas_profiling import st_profile_report

df = pd.read_csv("files.csv")
pr = df.profile_report()

st_profile_report(pr)