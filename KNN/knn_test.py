import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import json

# 读取CSV文件
fact_checks_df = pd.read_csv(r'data_preprocess\monolingual_fact_checks.csv')
posts_df = pd.read_csv(r'data_preprocess\monolingual_pairs_dev.csv')

# 提取fact checks和posts的文本内容，处理NaN值
fact_checks_texts = fact_checks_df['claim'].apply(lambda x: x[1] if isinstance(x, tuple) else x).fillna('').values
posts_texts = posts_df['text'].apply(lambda x: x[1] if isinstance(x, tuple) else x).fillna('').values

# 使用TF-IDF对文本进行向量化
vectorizer = TfidfVectorizer()
fact_checks_tfidf = vectorizer.fit_transform(fact_checks_texts)
posts_tfidf = vectorizer.transform(posts_texts)

# 使用KNN算法进行最近邻检索
knn = NearestNeighbors(n_neighbors=10, metric='cosine')
knn.fit(fact_checks_tfidf)

# 对每个帖子进行检索，找到前10个最相似的fact checks
distances, indices = knn.kneighbors(posts_tfidf)

# 将检索结果格式化为所需的JSON格式
retrieval_results = {}
for i, post_id in enumerate(posts_df['post_id']):
    retrieval_results[str(post_id)] = fact_checks_df['fact_check_id'].iloc[indices[i]].tolist()

# 将结果保存为JSON文件
with open('knn_retrieval_results.json', 'w') as json_file:
    json.dump(retrieval_results, json_file, indent=4)

print("KNN 检索完成，结果已保存为 knn_retrieval_results.json")
