# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import plotly.express as px
import plotly.graph_objects as go

# 设置页面标题
st.set_page_config(page_title="酒店推荐系统", layout="wide")

# 标题
st.title("基于文本相似度的酒店推荐系统")

@st.cache_data
def load_data():
    # 使用你的实际文件路径
    file_path = r"/Users/zhuyanrun/Desktop/酒店推荐系统/酒店推荐系统/第三章：基于相似度的酒店推荐系统/Seattle_Hotels.csv"
    try:
        df = pd.read_csv(file_path, encoding="latin-1")
        return df
    except FileNotFoundError:
        st.error(f"找不到数据文件：{file_path}")
        st.error("请确保数据文件路径正确")
        return None

# 加载数据
df = load_data()

# 侧边栏
st.sidebar.header("功能选择")
page = st.sidebar.radio(
    "选择功能页面",
    ["数据概览", "文本分析", "酒店推荐"]
)

# 数据概览页面
if page == "数据概览":
    st.header("数据概览")
    
    # 显示基本统计信息
    st.subheader("数据基本信息")
    st.write(f"数据集包含 {df.shape[0]} 个酒店和 {df.shape[1]} 个特征")
    
    # 显示数据样例
    st.subheader("数据样例")
    st.dataframe(df.head())
    
    # 显示词数分布
    st.subheader("酒店描述词数分布")
    df['word_count'] = df['desc'].apply(lambda x: len(str(x).split()))
    
    fig = px.histogram(
        df,
        x='word_count',
        nbins=50,
        title='酒店描述词数分布',
        labels={'word_count': '词数', 'count': '频率'}
    )
    st.plotly_chart(fig)

# 文本分析页面
elif page == "文本分析":
    st.header("文本分析")
    
    # 文本清理和分析函数
    @st.cache_data
    def get_top_n_words(corpus, n=None):
        vec = TfidfVectorizer(stop_words='english').fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:n]
    
    # 获取常用词
    common_words = get_top_n_words(df['desc'], 20)
    words_df = pd.DataFrame(common_words, columns=['word', 'frequency'])
    
    # 显示词频图表
    st.subheader("最常见的20个词")
    fig = px.bar(
        words_df,
        x='frequency',
        y='word',
        orientation='h',
        title='Top 20 Words in Hotel Descriptions'
    )
    st.plotly_chart(fig)

# 酒店推荐页面
elif page == "酒店推荐":
    st.header("酒店推荐")
    
    # 准备推荐系统
    @st.cache_data
    def prepare_recommendation_system(df):
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['desc'])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        return cosine_sim
    
    # 推荐函数
    def get_recommendations(name, cosine_sim, df):
        try:
            idx = df[df['name'] == name].index[0]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:11]
            hotel_indices = [i[0] for i in sim_scores]
            return df['name'].iloc[hotel_indices]
        except IndexError:
            return None
    
    # 准备推荐系统
    cosine_sim = prepare_recommendation_system(df)
    
    # 选择酒店
    hotel_name = st.selectbox(
        "选择一个酒店获取推荐",
        df['name'].tolist()
    )
    
    if st.button("获取推荐"):
        recommendations = get_recommendations(hotel_name, cosine_sim, df)
        if recommendations is not None:
            st.subheader("为您推荐的相似酒店：")
            for i, hotel in enumerate(recommendations, 1):
                st.write(f"{i}. {hotel}")
        else:
            st.error("未找到该酒店，请重试。")

# 添加页脚
st.markdown("---")
st.markdown("酒店推荐系统 © 2024")