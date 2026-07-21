import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.set_page_config(page_title="Manufacturing Defect Clustering", layout="wide")
st.title("Manufacturing Product Defect Clustering")
st.write("Analisis Clustering Cacat Produk Industri Manufaktur menggunakan K-Means")

df = pd.read_csv("defects_data.csv")
st.subheader("Dataset")
st.dataframe(df)

data = df.copy()
categorical = ["defect_type","defect_location","severity","inspection_method"]
enc = LabelEncoder()
for c in categorical:
    if c in data.columns:
        data[c]=enc.fit_transform(data[c].astype(str))

if "defect_date" in data.columns:
    data["defect_date"]=pd.to_datetime(data["defect_date"])
    data["day"]=data["defect_date"].dt.day
    data["month"]=data["defect_date"].dt.month
    data["year"]=data["defect_date"].dt.year
    data=data.drop(columns=["defect_date"])

X=StandardScaler().fit_transform(data)
km=KMeans(n_clusters=3,random_state=42,n_init=10)
df["Cluster"]=km.fit_predict(X)

st.subheader("Hasil Clustering")
st.dataframe(df)

fig,ax=plt.subplots()
counts=df["Cluster"].value_counts().sort_index()
ax.bar(counts.index.astype(str),counts.values)
ax.set_xlabel("Cluster"); ax.set_ylabel("Jumlah")
st.pyplot(fig)

if "repair_cost" in df.columns:
    fig2,ax2=plt.subplots()
    rc=df.groupby("Cluster")["repair_cost"].mean()
    ax2.bar(rc.index.astype(str),rc.values)
    ax2.set_ylabel("Average Repair Cost")
    st.pyplot(fig2)

st.subheader("Interpretasi")
for c in sorted(df["Cluster"].unique()):
    st.write(f"Cluster {c}: {len(df[df['Cluster']==c])} data")

st.success("Clustering membantu mengelompokkan cacat produk sehingga perusahaan dapat menentukan prioritas perbaikan kualitas.")
