import streamlit as st
import pandas as pd
import mysql.connector
from sqlalchemy import create_engine



def process():
    # Membuat koneksi ke database MySQL
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="sentimen_app"
    )
    
    uploaded_files = st.file_uploader("Choose a CSV or Excel file", accept_multiple_files=True)
    
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error('File format not supported. Please upload a CSV or Excel file.')
            continue
        # Menampilkan informasi file yang diunggah dan DataFrame
        st.write("Filename:", uploaded_file.name)
        st.write(df)
        def create_table(df, table_name):
            # Mengambil struktur kolom dari DataFrame
            columns = []
            for column in df.columns:
                column_type = df[column].dtype
                if column_type == "int64":
                    column_type = "INT"
                elif column_type == "float64":
                    column_type = "FLOAT"
                else:
                    column_type = "VARCHAR(255)"
                columns.append(f"{column} {column_type}")
            
            # Membuat tabel baru dalam database
            cursor = connection.cursor()
            query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})"
            cursor.execute(query)
            connection.commit()
            cursor.close()

        def insert_data(df, table_name):
            # Menyimpan data ke database
            engine = create_engine("mysql+mysqlconnector://root:@localhost/sentimen_app")
            df.to_sql(table_name, engine, if_exists="replace", index=False)

        # Mengirim DataFrame ke database
        table_name = "mypertamina"  # Ganti dengan nama tabel yang diinginkan
        create_table(df, table_name)
        insert_data(df, table_name)

        st.success("Data berhasil disimpan ke database.")



        
    #     # Menyimpan data ke database
    #     cursor = connection.cursor()
    #     for index, row in df.iterrows():
    #         # Mengganti 'table_name' dengan nama tabel yang sesuai di database
    #         query = "INSERT INTO komentar (username, content, label) VALUES (%s, %s, %s)"
    #         values = (row['username'], row['content'], row['label'])  # Sesuaikan dengan nama kolom yang sesuai
    #         cursor.execute(query, values)

    #     # Melakukan commit ke database
    #     connection.commit()
    #     cursor.close()

    #     st.success("Data berhasil disimpan ke database.")
    # if st.checkbox("Database"):
    #     query = "SELECT * FROM komentar"  # Ganti dengan query yang sesuai
    #     cursor = connection.cursor()
    #     cursor.execute(query)
    #     results = cursor.fetchall()

    #     # Menampilkan data menggunakan Streamlit
    #     st.write("Data from Database:")
    #     df = pd.DataFrame(results, columns=[col[0] for col in cursor.description])
    #     df = df.drop("id", axis=1)
    #     df.index = df.index + 1
    #     st.dataframe(df)
    # Menutup koneksi
    connection.close()


