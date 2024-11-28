from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, AutoModelForSequenceClassification
from torch.utils.data import Dataset
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型和分词器
local_model_path = '/opt/data/private/multilingual-e5-large'

# 自定义数据集
class CustomRetrievalDataset(Dataset):
    def __init__(self, queries, passages, labels, tokenizer, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for i, query in enumerate(queries):
            for j, passage in enumerate(passages):
                self.data.append((query, passage, labels[i][j]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query, passage, label = self.data[idx]
        
        tokenized = self.tokenizer(
            query, passage,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# 输入示例
queries = [
    "query: How much protein should a female eat?",
    "query: What is the role of carbohydrates?",
    "query: How much protein is needed for active women?",
    "query: How to increase protein intake?",
    "query: Effects of low protein diet?"
]

passages = [
    "passage: The average requirement of protein for women is 46 grams per day.",
    "passage: Some random text about carbohydrates.",
    "passage: Women need varying protein intake based on their activity level.",
    "passage: Increasing protein can be done by adding more beans and legumes.",
    "passage: Carbohydrates are the body's primary source of energy.",
    "passage: Too little protein can lead to muscle loss.",
    "passage: Protein requirements depend on body weight and activity.",
    "passage: Consuming more protein helps with muscle gain.",
    "passage: Protein deficiency can cause fatigue.",
    "passage: Excess protein intake is not stored but excreted."
]

# 对每个 query 分别提供一组 labels（5 组，每组 10 个）
labels = [
    [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],  # query1 对应的 labels
    [0, 1, 0, 0, 1, 0, 1, 0, 0, 0],  # query2 对应的 labels
    [1, 0, 1, 0, 0, 1, 0, 1, 1, 0],  # query3 对应的 labels
    [0, 0, 0, 1, 0, 0, 1, 0, 1, 1],  # query4 对应的 labels
    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0]   # query5 对应的 labels
]



tokenizer = AutoTokenizer.from_pretrained(local_model_path)

dataset = CustomRetrievalDataset(queries, passages, labels, tokenizer)

# 加载模型
model = AutoModelForSequenceClassification.from_pretrained(local_model_path, num_labels=1)
model = model.to(device)

# 微调参数
training_args = TrainingArguments(
    output_dir='./save_models',
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=10,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=10
)

# Trainer API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

print('start to train...............')

# 开始训练
# trainer.train()


#  开始预测
trained_model = AutoModelForSequenceClassification.from_pretrained('./save_models/checkpoint-150')

# 测试示例
new_queries = [
    "query: How much protein should a female eat?",
    "query: What are the benefits of a high-protein diet?",
    "query: How many grams of protein are in chicken breast?",
    "query: What is the recommended daily protein intake?",
    "query: How does protein affect weight loss?"
]

# new_passages = [f"passage: Passage content {i}" for i in range(1, 21)]
new_passages = [
    "passage: The average requirement of protein for women is 46 grams per day.",
    "passage: Some random text about carbohydrates.",
    "passage: After all, the kid is too young to go to school.",
    "passage: Increasing protein can be done by adding more beans and legumes.",
    "passage: Carbohydrates are the body's primary source of energy.",
    "passage: Too little protein can lead to muscle loss.",
    "passage: Protein requirements depend on body weight and activity.",
    "passage: Consuming more protein helps with muscle gain.",
    "passage: Protein deficiency can cause fatigue.",
    "passage: Excess protein intake is not stored but excreted."
    "passage: From now on, I not only study harder but also try my best to get better grades.",
    "passage: We should put out the fire as soon as we finish cooking.",
    "passage: I don’t like to show off myself.",
    "passage: I used to spend so much time on computer games that I lost interest in study.",
    "passage: I used to call on my friends.",
    "passage: My parents don’t go to bed until I come back every day.",
    "passage: I came across my old friend on my way home.",
    "passage: I congratulate you on your great progress.",
    "passage: I am afraid to get on badly withhim.",
    "passage: I have fun with my friends"
]

# 创建预测用数据集
class InferenceDataset(Dataset):
    def __init__(self, queries, passages, tokenizer, max_length=512):
        self.queries = queries
        self.passages = passages
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.queries) * len(self.passages)

    def __getitem__(self, idx):
        query_idx = idx // len(self.passages)
        passage_idx = idx % len(self.passages)
        query = self.queries[query_idx]
        passage = self.passages[passage_idx]
        
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
        }
    
inference_dataset = InferenceDataset(new_queries, new_passages, tokenizer)

print('start to predict...............')
predictions = trainer.predict(inference_dataset)

# 获取logits并计算概率
logits = predictions.predictions
probs = torch.sigmoid(torch.tensor(logits))

# 将预测转换为标签 (0 或 1)
threshold = 0.5
predicted_labels = (probs > threshold).int().numpy()

# 生成按 query 分组的 labels
num_queries = len(new_queries)
num_passages = len(new_passages)

# 输出每个 query 对应的 20 个 passage 的 labels
for q_idx in range(num_queries):
    start_idx = q_idx * num_passages
    end_idx = start_idx + num_passages
    labels_for_query = predicted_labels[start_idx:end_idx].flatten().tolist()
    
    print(f"Query {q_idx + 1}: {new_queries[q_idx]}")
    print(f"Labels: {labels_for_query}")
    print("-" * 60)


# predict results
# Query 1: query: How much protein should a female eat?
# Labels: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# ------------------------------------------------------------
# Query 2: query: What are the benefits of a high-protein diet?
# Labels: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# ------------------------------------------------------------
# Query 3: query: How many grams of protein are in chicken breast?
# Labels: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# ------------------------------------------------------------
# Query 4: query: What is the recommended daily protein intake?
# Labels: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# ------------------------------------------------------------
# Query 5: query: How does protein affect weight loss?
# Labels: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# ------------------------------------------------------------
