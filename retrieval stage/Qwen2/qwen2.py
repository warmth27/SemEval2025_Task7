from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import json
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = SentenceTransformer("/opt/data/private/gte-Qwen2-7B-instruct", trust_remote_code=True, model_kwargs={"torch_dtype": torch.float16}).to(device)

# 如果你希望调整最大序列长度
model.max_seq_length = 1024

# 加载post和fact_check JSON文件
print("Loading post and fact_check data...")
with open("/opt/data/private/SemEval2025_Task7/json_dataset/fact_check/monolingual_fact_checks_tha.json", "r", encoding="utf-8") as f:
    fact_check_data = json.load(f)
with open("/opt/data/private/SemEval2025_Task7/json_dataset/post_dev/monolingual_posts_dev_tha.json", "r", encoding="utf-8") as f:
    posts_data = json.load(f)

# # 任务描述
# task = 'Given a social media post, retrieve relevant fact-checked claims for the given post' # task1
# # task = 'Given a social media post below, retrieve the 10 most relevant fact-checks for the given post.' # task2
# # task = 'Given a social media post, retrieve relevant fact-checks' # task3
# # task = '' # task4

# 提取查询（post文本：ocr_text, ocr_translation, text_original, text_translation）
print("Extracting queries from posts...")
queries = []
post_ids = []
for post in posts_data:
    query = f"{post['ocr_text']} {post['ocr_translation']} {post['text_original']} {post['text_translation']}"
#   queries.append(get_detailed_instruct(task, query))
    queries.append(query)
    post_ids.append(post['post_id'])  # Store post_id

# 提取文档（fact_check文本：claim_text, claim_translation, title_text, title_translation）
print("Extracting documents from fact_check data...")
documents = []
fact_check_ids = []
for fact in fact_check_data:
    document = f"{fact['claim_text']} {fact['claim_translation']} {fact['title_text']} {fact['title_translation']}"
    documents.append(document)
    fact_check_ids.append(fact['fact_check_id'])  # Store fact_check_id

# 设置批次大小
batch_size = 8  # 根据显存大小调整批次大小

# 生成查询和文档的嵌入
print("Generating embeddings for queries and documents...")
query_embeddings = []
document_embeddings = []

# 处理查询的批次
for i in range(0, len(queries), batch_size):
    batch_queries = queries[i:i+batch_size]
    query_embeddings_batch = model.encode(batch_queries, device=device, show_progress_bar=True)
    query_embeddings.append(query_embeddings_batch) 
    # 处理完一个批次后释放显存
    torch.cuda.empty_cache()

# 处理文档的批次
for i in range(0, len(documents), batch_size):
    batch_documents = documents[i:i+batch_size]
    document_embeddings_batch = model.encode(batch_documents, device=device, show_progress_bar=True)
    document_embeddings.append(document_embeddings_batch) 
    # 处理完一个批次后释放显存
    torch.cuda.empty_cache()

# 将所有批次的嵌入合并
query_embeddings = np.vstack(query_embeddings)
document_embeddings = np.vstack(document_embeddings)

# 将嵌入转换为PyTorch张量
query_embeddings_tensor = torch.tensor(query_embeddings, dtype=torch.float32, device=device)
document_embeddings_tensor = torch.tensor(document_embeddings, dtype=torch.float32, device=device)

# 计算查询与文档之间的相似度
similarities = torch.matmul(query_embeddings_tensor, document_embeddings_tensor.T)

# 准备结果
results = {}

# 对每个查询，找到最相关的10个文档
for idx, query_similarity in enumerate(similarities):
    # 排序并获取前10个最相关的文档
    sorted_indices = torch.argsort(query_similarity, descending=True)[:10]
    top_10_ids = [fact_check_ids[i] for i in sorted_indices]
    results[post_ids[idx]] = top_10_ids  # 保存结果与查询ID（post_id）

# 保存结果为JSON文件
output_file = "qwen_plan3_tha.json"
with open(output_file, "w") as json_file:
    json.dump(results, json_file, indent=4)

print(f"Results saved to {output_file}")