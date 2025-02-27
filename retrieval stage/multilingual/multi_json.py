import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
import json

# 平均池化函数
def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

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
 
# 本地模型路径
local_model_path = '/Users/ASUS/Desktop/NLP/Models/multilingual-e5-large'

# 加载本地模型和分词器
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModel.from_pretrained(local_model_path).to('cuda')

# 加载数据（替换为JSON文件）
posts_file = r'json_dataset\cross\crosslingual_posts_dev.json'  # 替换为你的帖子文件路径
fact_checks_file = r'json_dataset\cross\crosslingual_fact_checks.json'  # 替换为你的fact_check文件路径

# 读取JSON文件
posts_df = pd.read_json(posts_file)
fact_checks_df = pd.read_json(fact_checks_file)

# 提取文本内容（post包含ocr和text，fact_check包含claim和title）
post_texts = (posts_df['ocr_translation'].fillna('') + " " + posts_df['text_translation'].fillna('')).tolist()
fact_check_texts = (fact_checks_df['claim_translation'].fillna('') + " " + fact_checks_df['title_translation'].fillna('')).tolist()

# 生成嵌入
print("Generating embeddings for posts...")
post_embeddings = generate_embeddings([f"query: {text}" for text in post_texts], tokenizer, model)
print("Generating embeddings for fact checks...")
fact_check_embeddings = generate_embeddings([f"passage: {text}" for text in fact_check_texts], tokenizer, model)

# 计算相似度
print("Calculating similarities...")
similarities = torch.matmul(post_embeddings, fact_check_embeddings.T) * 100

# 获取每个帖子最相关的10条fact_check
top_k = 10  # 选择最相关的10条
results = []
for i, similarity in enumerate(similarities):
    top_indices = similarity.topk(top_k).indices.tolist()
    top_fact_checks = [int(fact_checks_df.iloc[idx]['fact_check_id']) for idx in top_indices]
    results.append({
        'post_id': int(posts_df.iloc[i]['post_id']),
        'related_fact_check_ids': top_fact_checks
    })

# 保存结果到JSON文件
output_file = 'multi_cross_plan2.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)

print("Results saved to multi_results.json.")
