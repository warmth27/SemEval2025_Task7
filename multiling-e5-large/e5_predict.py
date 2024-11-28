import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型和分词器
local_model_path = '/opt/data/private/multilingual-e5-large'

tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModel.from_pretrained(local_model_path)
model = model.to(device)

# 计算平均池化
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

# 从JSON文件读取数据
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    

# 主程序
if __name__ == "__main__":
    # 设置JSON文件路径
    query_file_path = './datasets/monolingual_pairs_dev_tha.json'  # 查询文件路径
    passage_file_path = './datasets/monolingual_fact_checks_dev_tha.json'  # 段落文件路径
    output_file_path = './results/tha.json'  # 输出的文件路径

    # 读取查询和段落数据
    queries = load_data(query_file_path)
    passages = load_data(passage_file_path)

    # 提取文本内容
    query_texts = [f"query: {item['query']}" for item in queries]
    passage_texts = [f"passage: {item['passage']}" for item in passages]
    
    # 生成嵌入
    print("Generating embeddings for queries...")
    query_embeddings = generate_embeddings(query_texts, tokenizer, model)
    
    print("Generating embeddings for passages...")
    passage_embeddings = generate_embeddings(passage_texts, tokenizer, model)
    
    # 计算相似度矩阵
    print("Calculating similarities...")
    similarities = torch.matmul(query_embeddings, passage_embeddings.T) * 100
    
    # 获取每个查询最相关的10个段落
    top_k = 10
    query_passage_map = {}
    
    
    for i, similarity in tqdm(enumerate(similarities), total=len(similarities), desc="Processing Queries", unit="query"):
        query_id = queries[i]['post_id']  # 获取查询的ID
        top_indices = similarity.topk(top_k).indices.tolist()
        top_passage_ids = [passages[idx]['fact_check_id'] for idx in top_indices]  # 获取最相关的段落ID
        query_passage_map[str(query_id)] = top_passage_ids

    # 保存结果到JSON文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(query_passage_map, f, ensure_ascii=False, indent=4)

    print(f"Query-passage IDs have been saved to {output_file_path}")
