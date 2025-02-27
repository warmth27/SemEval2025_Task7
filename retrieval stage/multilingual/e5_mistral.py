import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import json
from torch import Tensor

# 提取最后一个token的hidden state用于生成嵌入
def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

# 格式化任务描述和查询
def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

# 批量生成嵌入的函数
def generate_embeddings_in_batches(input_texts, tokenizer, model, batch_size, max_length, device='cuda'):
    embeddings_list = []
    for i in range(0, len(input_texts), batch_size):
        batch_texts = input_texts[i:i+batch_size]
        
        # Tokenize当前批次的文本
        batch_dict = tokenizer(batch_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt').to(device)
        
        # 获取模型输出
        outputs = model(**batch_dict)
        
        # 提取嵌入并进行归一化
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        embeddings_list.append(embeddings)
        
        print(f"Processed batch {i // batch_size + 1}/{len(input_texts) // batch_size + 1}")
    
    # 合并所有批次的嵌入
    return torch.cat(embeddings_list, dim=0)

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化分词器和模型
tokenizer = AutoTokenizer.from_pretrained('/opt/data/private/e5-mistral-7b-instruct')
model = AutoModel.from_pretrained('/opt/data/private/e5-mistral-7b-instruct').to(device)

# 加载post和fact_check JSON文件
print("Loading post and fact_check data...")
with open("/opt/data/private/SemEval2025_Task7/json_dataset/fact_check/monolingual_fact_checks_ara.json", "r", encoding="utf-8") as f:
    fact_check_data = json.load(f)
with open("/opt/data/private/SemEval2025_Task7/json_dataset/post_dev/monolingual_posts_dev_ara.json", "r", encoding="utf-8") as f:
    posts_data = json.load(f)

# 任务描述
task = 'Given a social media post, retrieve relevant fact-checked claims for the given post'

# 提取查询（post文本：ocr_text, ocr_translation, text_original, text_translation）
print("Extracting queries from posts...")
queries = []
for post in posts_data:
    query = f"{post['ocr_text']} {post['ocr_translation']} {post['text_original']} {post['text_translation']}"
    queries.append(get_detailed_instruct(task, query))

# 提取文档（fact_check文本：claim_text, claim_translation, title_text, title_translation）
print("Extracting documents from fact_check data...")
documents = []
for fact in fact_check_data:
    document = f"{fact['claim_text']} {fact['claim_translation']} {fact['title_text']} {fact['title_translation']}"
    documents.append(document)

# 合并查询和文档生成嵌入
input_texts = queries + documents

# 批量生成嵌入
print("Generating embeddings in batches...")
embeddings = generate_embeddings_in_batches(input_texts, tokenizer, model, batch_size=8, max_length=4096, device=device)

# 将嵌入分为查询和文档
query_embeddings = embeddings[:len(queries)]
document_embeddings = embeddings[len(queries):]

# 计算相似度分数（点积）
print("Calculating similarity scores...")
similarity_scores = (query_embeddings @ document_embeddings.T) * 100

# 创建结果字典
result = {}

# 对于每个查询（post），获取与之最相关的10个fact_check
print("Retrieving top 10 fact_check IDs for each post...")
for i, post in enumerate(posts_data):
    # 获取最相关的10个文档的索引
    top_10_indices = similarity_scores[i].topk(10).indices.tolist()
    
    # 获取相应的fact_check_id
    top_10_fact_check_ids = [fact_check_data[idx]['fact_check_id'] for idx in top_10_indices]
    
    # 存储结果
    result[post['post_id']] = top_10_fact_check_ids

# 保存结果为JSON文件
print("Saving results to file...")
with open("e5_mistral_results_plan3.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=4, ensure_ascii=False)

print("Top 10 fact_check IDs retrieval completed.")
