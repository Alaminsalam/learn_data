import streamlit as st
import pandas as pd
import string
import nltk
import re
import swifter
import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from textblob import TextBlob
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
# from nltk.probability import FreqDist
from nltk.corpus import stopwords
from google_play_scraper import Sort, reviews

df = None
def view_scraper():
    st.header('Web Scrapping Aplikasi di Play Store')
    st.markdown("contoh ID aplikasi mypertamina yang diambil pada URL diatas = <span style='color:red'>com.dafturn.mypertamina</span>", unsafe_allow_html=True)
    app_name = st.text_input("Masukkan ID aplikasi")
    jumlah = st.slider("Jumlah komentar yang ingin diambil:", min_value=1, max_value=10000, value=50)
    sorting_options = ["Relevant", "Terbaru"]
    sorting = st.selectbox('Filter Komentar ', sorting_options)
    col1,col2 = st.columns(2)
    combine = col1.checkbox("Gabungkan Semua Komentar")
    scraped = col2.checkbox('Scrape')
    global df
    if scraped:
        if sorting == "Terbaru":
            sort = Sort.NEWEST
        elif sorting == "Relevant":
            sort = Sort.MOST_RELEVANT
        result, continuation_token = reviews(
            app_name,
            lang='id', # bahasa yang diinginkan
            country='id', # negara yang diinginkan
            sort= sort, # sortir komentar
            count =jumlah, # jumlah komentar yang ingin diambil
            filter_score_with=None
        )
        result_newest, continuation_token = reviews(
            app_name,
            lang='id', # bahasa yang diinginkan
            country='id', # negara yang diinginkan
            sort= Sort.NEWEST, # sortir komentar
            count =jumlah, # jumlah komentar yang ingin diambil
            filter_score_with=None
        )
        result_relevant, continuation_token = reviews(
            app_name,
            lang='id', # bahasa yang diinginkan
            country='id', # negara yang diinginkan
            sort= Sort.MOST_RELEVANT, # sortir komentar
            count =jumlah, # jumlah komentar yang ingin diambil
            filter_score_with=None
        )
        cols_to_drop = None
        if not result_newest or not result_relevant:
            st.error("ID Aplikasi tidak ditemukan. Silahkan cek kembali ID yang Anda masukkan.")
        else:
        #membuat dataframe
            df = pd.DataFrame(result, columns=['userName', 'score','content','reviewCreatedVersion','at','thumbsUpCount'])
            df_newest = pd.DataFrame(result_newest, columns=['userName', 'score','content','reviewCreatedVersion','at','thumbsUpCount'])
            df_relevant = pd.DataFrame(result_relevant, columns=['userName', 'score','content','reviewCreatedVersion','at','thumbsUpCount'])
            if combine:
                df = pd.concat([df_newest, df_relevant])
                st.dataframe(df)
            else:
                st.dataframe(df)
            cols = df.columns.tolist()
            cols_to_drop = st.multiselect("Pilih kolom yang ingin dihapus", cols, key='cols')
            if cols_to_drop:
                # hapus kolom yang dipilih dari dataframe
                df = df.drop(columns=cols_to_drop)
            col = df.columns.tolist()
            col.insert(0, "None") # menambahkan opsi "None" sebagai pilihan pertama
            sort_by = st.selectbox("Pilih Kriteria Sort", col, index=0) # mengatur nilai default menjadi "None"
            if sort_by !="None" :
                df = df.sort_values(by=sort_by, ascending=False)
            if 'reviewCreatedVersion' in df.columns:
                vers = df['reviewCreatedVersion'].unique()
                vers_select = st.multiselect("Pilih Version aplikasi yang di inginkan", vers)
                if vers_select:
                    df = df[df['reviewCreatedVersion'].isin(vers_select)]
            if 'at' in df.columns:
                df['at'] = df['at'].dt.date
                start_date = st.date_input('Start date', value=df['at'].min())
                end_date = st.date_input('End date', value=df['at'].max())
                if start_date and end_date:
                    df = df[(df['at'] >= start_date) & (df['at'] <= end_date)]
                # columns_to_drop = st.selectbox("Pilih Tipe data Waktu yang ingin dihapus", ['None','at', 'at_date'])
                # if columns_to_drop != 'None':
                #     df = df.drop(columns=[columns_to_drop])
            st.write('Jumlah Baris pada DataFrame:', df.shape[0])



normalized_word = pd.read_csv("normalisasi.csv")

normalized_word_dict = {}

for index, row in normalized_word.iterrows():
    if row[0] not in normalized_word_dict:
        normalized_word_dict[row[0]] = row[1] 

def normalized_term(document):
    return [normalized_word_dict[term] if term in normalized_word_dict else term for term in document]

def preprocessing():
    global df
    view_scraper()
    if df is not None:
        st.write("Data Preprocessing")
        st.dataframe(df)
        # col = df.columns.tolist()
        # col_to_drop = st.multiselect("Pilih kolom yang ingin dihapus", col, key='col')#Gunakan key yang berbeda untuk 2 objek 
        # if col_to_drop:
        #     # hapus kolom yang dipilih dari dataframe
        #     df = df.drop(columns=col_to_drop)
        #     st.write(df)
        if st.checkbox('Case Folding'):
            df['content'] = df['content'].apply(lambda x: x.lower())
            st.write('Hasil Case Folding')
            df.reset_index(drop=True, inplace=True) # mengurutkan kembali nomor pada dataframe
            st.write(df)
            # st.write("Tipe data kolom content: ", df['content'].dtype)#untuk mengetahui tipe data pada datframe

        if st.checkbox('Cleaning'):
            df['content'] = df['content'].apply(lambda x: re.sub(r'[^\w\s]', ' ', x))
            df['content'] = df['content'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
            df['content'] = df['content'].apply(lambda x: re.sub(r"\d+","",x))#untuk menghilangkan angka
            df['content'] = df['content'].apply(lambda x: x.strip())#remove whitespace leading & trailing 
            df['content'] = df['content'].apply(lambda x: re.sub(r'\s+',' ',x))#remove multiple whitespace leading & trailing 
            df['content'] = df['content'].apply(lambda x: re.sub(r'[^\x00-\x7f]',r'',x))#untuk menghilangkan emoji
            df['content'] = df['content'].apply(lambda x: re.sub(r"\b[a-zA-Z]\b", "", x))#untuk menghilangkan satu karakter
            st.write("Hasil Cleaning")
            st.write(df)
        

        if st.checkbox('Labeling data'):
            analyzer = SentimentIntensityAnalyzer()
             # Menentukan label data menggunakan vaderSentiment menjadi 2 label saja
            labels = []
            for content in df['content'].tolist():
                score = analyzer.polarity_scores(content)
                if score['compound'] >= 0.05:
                    labels.append('positif')
                else:
                    labels.append('negatif')
            # Tambahkan label ke dalam DataFrame
            df['label'] = labels
            # analyzer = SentimentIntensityAnalyzer()
            # # Menentukan label data menggunakan vaderSentiment menjadi 3 label 
            # labels = []
            # for content in df['content'].tolist():
            #     score = analyzer.polarity_scores(content)
            #     if score['compound'] >= 0.05:
            #         labels.append('positif')
            #     elif score['compound'] > -0.05 and score['compound'] < 0.05:
            #         labels.append('netral')
            #     else:
            #         labels.append('negatif')
            # # Tambahkan label ke dalam DataFrame
            # df['label'] = labels
            # Tampilkan DataFrame dengan label baru
            st.write('DataFrame dengan Label vaderSentiment')
            st.dataframe(df)
        if st.checkbox('Tokenizing'):
            df['content'] = df['content'].apply(lambda x: word_tokenize(x))
            st.write("Hasil Tokenizing")
            st.write(df.head())
        if st.checkbox('Normalisasi'):
            df['content'] = df['content'].apply(normalized_term)
            st.write('Hasil Normalisasi')
            st.write(df)
        #stopword menggunakan library Sastrawi menggunakan list stopword
        if st.checkbox('Stopword'):
            txt_stopword = pd.read_csv("stopwords.txt", names= ["stopwords"], header = None)
            list_stopword = set(txt_stopword["stopwords"][0].split(' '))
            factory = StopWordRemoverFactory()
            stopword = factory.create_stop_word_remover()
            df['content'] = df['content'].apply(lambda x: [stopword.remove(word) for word in x if not word in list_stopword])
            df['content'] = df['content'].apply(lambda x: " ".join(x) if isinstance(x, list) else x)#mengubah tipe data list menjadi string
            df['content'] = df['content'].apply(lambda x: word_tokenize(x))#mengubah string menjadi list
            df = df.loc[df['content'].apply(lambda x: True if x else False)].reset_index(drop=True) # menghapus baris yang kosong


            # # stopword menggunakan sastrawi saja
            # factory = StopWordRemoverFactory()
            # stopword = factory.create_stop_word_remover()
            # df['content'] = df['content'].apply(lambda x: [stopword.remove(word) for word in x ])


            # # penggunaan stopword menggunakan nltk
            # stop_words = set(stopwords.words('indonesian'))
            # # Tambahkan kata-kata yang ingin dikecualikan dari stopwords
            # exceptions = ['kurang']
            # stop_words = stop_words.difference(exceptions)
            # df['content'] = df['content'].apply(lambda x: [word for word in x if not word.lower() in stop_words])
            # # Stopword menggunakan library nltk
            # stop_words = stopwords.words('indonesian')
            # txt_stopword = pd.read_csv("stopwords.txt", names= ["stopwords"], header = None)
            # list_stopwords = stop_words + txt_stopword["stopwords"][0].split(' ')
            # list_stopword = set(list_stopwords)
            # df['content'] = df['content'].apply(lambda x: [word for word in x if not word in  list_stopword])

            # Script dibawah ini  untuk mengumpulkan kata yang dianggap stopword
            # stopwords_collected = set(stop_words)
            # df['content'].apply(lambda x: stopwords_collected.update(x))
            # st.write("Stopwords collected:", stopwords_collected)
            # if st.button("Save as CSV"):
            #     import csv
            #     with open("stopwords_dataset.csv", "w") as file:
            #         writer = csv.writer(file)
            #         writer.writerow(["stopword"])
            #         for word in stopwords_collected:
            #             writer.writerow([word])
            #     st.success("Stopwords saved to stopwords_dataset.csv")
            st.write('Hasil Stopword Removal')
            st.write('Jumlah Baris pada DataFrame:', df.shape[0])
            st.write(df)
        
        # # Proses stemming menggunakan nltk
        # stemmer = PorterStemmer()
        # # aplikasikan proses stemming pada setiap elemen dalam kolom 'content' dalam dataframe
        # df['content'] = df['content'].apply(lambda x: [stemmer.stem(word) for word in x])
        # # tampilkan dataframe hasil stemming
        # st.write('Proses Stemming')
        # st.write(df)
        # df['content'] = df['content'].apply(lambda x: ' '.join(x))

        if st.checkbox('Stemming'):
            # Proses Stemming menggunakan sastrawi
            factory = StemmerFactory()
            stemmer = factory.create_stemmer()
            df['content'] = df['content'].swifter.apply(lambda x: [stemmer.stem(word) for word in x])
            st.write('Hasil Stemming')
            st.write(df)
            df['content'] = df['content'].apply(lambda x: ' '.join(x))

        if st.checkbox('Tf-Idf'):
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_matrix = tfidf_vectorizer.fit_transform(df['content'].tolist())
            st.write('Proses TF-IDF')
            st.dataframe(tfidf_matrix)

        if st.checkbox('Labeling Data'):
            # # Pelabelan dengan k-means
            kmeans = KMeans(n_clusters=3, random_state=0)
            # Latih KMeans dengan data hasil dari TF-IDF
            kmeans.fit(tfidf_matrix)
            # Menentukan label berdasarkan jarak data dengan centroid
            centroids = kmeans.cluster_centers_
            labels = []
            for i in range(len(tfidf_matrix.toarray())):
                distance = np.sum((tfidf_matrix.toarray()[i] - centroids)**2, axis=1)
                label = np.argmin(distance)
                labels.append(label)
            # Menentukan label positif, netral, dan negatif
            labels_mapping = {0: 'negatif', 1: 'netral', 2: 'positif'}
            labels = [labels_mapping[i] for i in labels]
            # Tambahkan label ke dalam DataFrame
            df['label'] = labels
            # Tampilkan DataFrame dengan label baru
            st.write('DataFrame dengan Label')
            st.dataframe(df)
        if st.checkbox('Klasifikasi Data'):
            # Menambahkan fitur klasifikasi menggunakan algoritma k-nearest neighbor
            knn = KNeighborsClassifier(n_neighbors=3)
            # Latih KNeighborsClassifier dengan data hasil dari TF-IDF
            knn.fit(tfidf_matrix, df['label'].tolist())
            # Prediksi label
            prediction = knn.predict(tfidf_matrix)
            # Tambahkan prediksi label ke dalam DataFrame
            df['prediction'] = prediction
            # Tampilkan DataFrame dengan prediksi label
            st.write('DataFrame dengan Prediksi Label')
            st.dataframe(df)
        if st.checkbox('Visualisasi Data'):
            st.set_option('deprecation.showPyplotGlobalUse', False)
            # Plot jumlah data per label asli dan prediksi
            sns.countplot(x='label', data=df)
            plt.title('Jumlah Data per Label Asli')
            st.pyplot()
            sns.countplot(x='prediction', data=df)
            plt.title('Jumlah Data per Label Prediksi')
            st.pyplot()

            # Buat matriks konfusi
            cm = confusion_matrix(df['label'].tolist(), df['prediction'].tolist())
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            # Plot matriks konfusi
            sns.heatmap(cm_normalized, annot=True, xticklabels=df['label'].unique(), yticklabels=df['label'].unique())
            plt.xlabel('Label Prediksi')
            plt.ylabel('Label Asli')
            plt.title('Confusion Matriks')
            st.pyplot()


        # Pelabelan menggunakan vadersentimen
        # Inisialisasi SentimentIntensityAnalyzer
        # analyzer = SentimentIntensityAnalyzer()
        # # Menentukan label data menggunakan vaderSentiment
        # labels = []
        # for content in df['content'].tolist():
        #     score = analyzer.polarity_scores(content)
        #     if score['compound'] >= 0.05:
        #         labels.append('positif')
        #     elif score['compound'] > -0.05 and score['compound'] < 0.05:
        #         labels.append('netral')
        #     else:
        #         labels.append('negatif')
        # # Tambahkan label ke dalam DataFrame
        # df['label'] = labels
        # # Tampilkan DataFrame dengan label baru
        # st.write('DataFrame dengan Label')
        # st.dataframe(df)



        # def convert_df(df):
        #     return df.to_csv(index=False).encode('utf-8')
        # csv = convert_df(df)
        # st.download_button(
        #     "Download sebagai CSV",
        #     csv,
        #     "file.csv",
        #     "text/csv",
        #     key='download-csv'
        # )

     



if __name__ == "__main__":
    preprocessing()