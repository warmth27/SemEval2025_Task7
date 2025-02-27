import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import json

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_path = "/opt/data/private/NV-Embed-v2"

# 加载 nvidia/NV-Embed-v2 模型，并使用 float16 精度
model = SentenceTransformer(model_path, trust_remote_code=True, model_kwargs={"torch_dtype": torch.float16}).to(device)
model.max_seq_length = 128  # 设置最大序列长度
model.tokenizer.padding_side = "right"  # 设置填充方式

# 函数：为输入的查询和文档添加 EOS 标记
def add_eos(input_examples):
    return [example + model.tokenizer.eos_token for example in input_examples]

# 加载post和fact_check JSON文件
print("Loading post and fact_check data...")
with open("/opt/data/private/SemEval2025_Task7/json_dataset/fact_check/monolingual_fact_checks_ara.json", "r", encoding="utf-8") as f:
    fact_check_data = json.load(f)
with open("/opt/data/private/SemEval2025_Task7/json_dataset/post_dev/monolingual_posts_dev_ara.json", "r", encoding="utf-8") as f:
    posts_data = json.load(f)

# Prepare the query instruction and queries
instruction = "Given a social media post, retrieve relevant fact-checks for the given post"

# 函数来格式化任务描述和查询
def get_detailed_instruct(instruction: str, query: str) -> str:
    return f"Instruct: {instruction}\nQuery: {query}"

# 提取查询（post文本：ocr_text, ocr_translation, text_original, text_translation）
print("Extracting queries from posts...")
queries = []
for post in posts_data:
    query = f"{post['ocr_text']} {post['ocr_translation']} {post['text_original']} {post['text_translation']}"
    queries.append(get_detailed_instruct(instruction, query))

# 提取文档（fact_check文本：claim_text, claim_translation, title_text, title_translation）
print("Extracting documents from fact_check data...")
passages = []
for fact in fact_check_data:
    passage = f"{fact['claim_text']} {fact['claim_translation']} {fact['title_text']} {fact['title_translation']}"
    passages.append(passage)

# 生成查询和文档的嵌入
batch_size = 1
queries_with_instruction = [get_detailed_instruct(instruction, query) for query in queries]
query_embeddings = model.encode(add_eos(queries_with_instruction), batch_size=batch_size, normalize_embeddings=True)
passage_embeddings = model.encode(add_eos(passages), batch_size=batch_size, normalize_embeddings=True)

# 计算查询和文档的相似度
similarities = torch.matmul(torch.tensor(query_embeddings), torch.tensor(passage_embeddings).T) * 100  # 将相似度放大100

# 准备结果数据
results = {}

# 对于每个查询（post），获取与之最相关的10个fact_check
print("Retrieving top 10 fact_check IDs for each post...")
for i, post in enumerate(posts_data):
    # 获取最相关的10个文档的索引
    top_10_indices = similarities[i].topk(10).indices.tolist()
    
    # 获取相应的fact_check_id
    top_10_fact_check_ids = [fact_check_data[idx]['fact_check_id'] for idx in top_10_indices]
    
    # 存储结果
    results[post['post_id']] = top_10_fact_check_ids

# 保存结果为JSON格式
with open("NV_results.json", "w", encoding="utf-8") as json_file:
    json.dump(results, json_file, ensure_ascii=False, indent=4)

print("Results saved to 'NV_results.json'")
