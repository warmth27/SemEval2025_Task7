import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import json

model_name = '/Users/ASUS/Desktop/NLP/Models/multilingual-e5-large-instruct'

# 加载CSV文件
posts_df = pd.read_csv(r'data_preprocess2/dev/monolingual_posts_dev_eng.csv')  # 假设你已经加载了 posts.csv 文件
fact_check_df = pd.read_csv(r'data_preprocess2/fact_check/monolingual_fact_checks_eng.csv')  # 假设你也加载了 fact_check.csv 文件

# 选择需要的字段
post_texts = (posts_df['ocr'] + ' ' + posts_df['text']).tolist()  # 拼接 'ocr' 和 'text' 作为查询
fact_check_titles = fact_check_df['title'].tolist()  # 假设事实核查的标题在 'title' 列
fact_check_claims = fact_check_df['claim'].tolist()  # 假设事实核查的声明在 'claim' 列
fact_check_ids = fact_check_df['fact_check_id'].tolist()  # 假设事实核查的 ID 在 'fact_check_id' 列

# 任务描述
# task = 'Given a social media post, retrieve relevant fact-checked claims for the given post'
task = 'Given a social media post below, retrieve the 10 most relevant fact-checks for the given post.'

# 初始化模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 将模型转移到 GPU（如果有的话）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义平均池化函数
def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# 定义生成任务描述和查询的函数
def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

# 将帖子文本与事实核查的标题+声明合并成查询-文档对
queries = [get_detailed_instruct(task, post_text) for post_text in post_texts]
documents = [str(claim) + ' ' + str(title) for claim, title in zip(fact_check_claims, fact_check_titles)]

# 合并查询和文档
input_texts = queries + documents

# 批量生成嵌入的函数
def generate_embeddings(texts, tokenizer, model, batch_size=4, device='cuda'):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_dict = tokenizer(batch_texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**batch_dict)
        embeddings_batch = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings_batch = F.normalize(embeddings_batch, p=2, dim=1)  # 归一化
        embeddings.append(embeddings_batch)
        torch.cuda.empty_cache()  # 清理GPU显存
    return torch.cat(embeddings, dim=0)

# 生成查询和文档的嵌入
embeddings = generate_embeddings(input_texts, tokenizer, model, batch_size=4, device=device)

# 计算查询与文档的相似度
scores = (embeddings[:len(queries)] @ embeddings[len(queries):].T) * 100

# 存储每个帖子的最相关10条事实核查的ID
result = {}

for i, post_id in enumerate(posts_df['post_id']):
    # 获取当前帖子的相似度分数
    post_scores = scores[i].tolist()
    
    # 找到最相关的10个事实核查的索引
    top_10_indices = sorted(range(len(post_scores)), key=lambda x: post_scores[x], reverse=True)[:10]
    
    # 获取对应的fact_check_id
    top_10_ids = [fact_check_ids[idx] for idx in top_10_indices]
    
    # 存储结果
    result[str(post_id)] = top_10_ids

# 将结果保存为JSON格式
with open('multi_instruct_results_eng.json', 'w') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

print("保存完成：multi_instruct_results_deu.json")
