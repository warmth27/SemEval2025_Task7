from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F

# 自定义数据集
class RetrievalDataset(Dataset):
    def __init__(self, queries, passages, labels, tokenizer, max_length=512):
        self.queries = queries
        self.passages = passages
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        query = self.queries[idx]
        passage = self.passages[idx]
        label = self.labels[idx]
        
        # Tokenize query and passage
        tokenized = self.tokenizer(
            query, passage,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.float),
        }

# 微调模型的设计
def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# 准备数据
queries = ["query: How much protein should a female eat?"]
passages = [
    "passage: The average requirement of protein for women is 46 grams per day.",
    "passage: Some random text about carbohydrates.",
    "passage: Women need varying protein intake based on their activity level."
]
labels = [1, 0, 1]  # 每个 passage 的相关性标签

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')

# 创建数据集
dataset = RetrievalDataset(queries, passages, labels, tokenizer)

# 微调参数设置
training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=10,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=10
)

# 定义训练过程
class RetrievalModel(torch.nn.Module):
    def __init__(self, model):
        super(RetrievalModel, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        embeddings = average_pool(outputs.last_hidden_state, attention_mask)
        
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(embeddings, labels.unsqueeze(1))
            return loss, embeddings
        return embeddings

# 初始化模型
retrieval_model = RetrievalModel(model)

# Trainer API
trainer = Trainer(
    model=retrieval_model,
    args=training_args,
    train_dataset=dataset
)

# 开始训练
trainer.train()

# 训练后评估
trainer.save_model()

# 预测
def get_query_passage_similarity(query, passages):
    query_embedding = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
    query_embedding = model(**query_embedding).last_hidden_state
    query_embedding = average_pool(query_embedding, query_embedding['attention_mask'])
    
    similarities = []
    for passage in passages:
        passage_embedding = tokenizer(passage, return_tensors="pt", padding=True, truncation=True, max_length=512)
        passage_embedding = model(**passage_embedding).last_hidden_state
        passage_embedding = average_pool(passage_embedding, passage_embedding['attention_mask'])
        
        similarity = torch.cosine_similarity(query_embedding, passage_embedding)
        similarities.append(similarity.item())
    
    return similarities

# 示例：预测查询与多个 passage 之间的相似度
query = "query: How much protein should a female eat?"
passages = [
    "passage: The average requirement of protein for women is 46 grams per day.",
    "passage: Some random text about carbohydrates.",
    "passage: Women need varying protein intake based on their activity level."
]
similarities = get_query_passage_similarity(query, passages)
print(similarities)
