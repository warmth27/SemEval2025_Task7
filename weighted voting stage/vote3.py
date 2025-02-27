import json
from collections import defaultdict

# 使用原始字符串或双反斜杠来避免路径问题
file1 = r'por1.json'
file2 = r'por2.json'
file3 = r'translation\por5.json'

output_file = "fra-por.json"

# 假设权重是这样的
file_weights = {
    file1: 3,
    file2: 2,
    file3: 1
}

# 读取文件的函数
def load_json_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# 读取三个文件的数据
file1_data = load_json_data(file1)
file2_data = load_json_data(file2)
file3_data = load_json_data(file3)

# 统计 id 在每个文件中的出现次数
def count_ids_for_post(post_id, file_data, file_name, id_count):
    # 确保我们只处理每个 post_id 的 ids
    for id_ in file_data.get(post_id, []):
        id_count[id_][file_name] += 1

# 处理每个 post_id
def process_post(post_id):
    # 为每个 post_id 初始化一个新的 id_count
    id_count = defaultdict(lambda: defaultdict(int))
    
    # 统计在每个文件中出现的 id
    count_ids_for_post(post_id, file1_data, file1, id_count)
    count_ids_for_post(post_id, file2_data, file2, id_count)
    count_ids_for_post(post_id, file3_data, file3, id_count)

    # 存储该 post_id 对应的所有 id 和它们的权重
    id_with_counts = []
    
    # 统计每个 id 的出现次数和优先级分数
    for id_, counts in id_count.items():
        total_count = sum(counts.values())  # 统计该 id 在所有文件中的总出现次数
        priority_count = 0
        for file_name, count in counts.items():
            priority_count += count * file_weights[file_name]  # 权重加成
        
        id_with_counts.append((id_, total_count, priority_count))
    
    # 按照出现次数降序排序，出现次数相同的按优先级排序
    id_with_counts.sort(key=lambda x: (-x[1], -x[2]))  # 先按总出现次数排序，再按优先级排序
    
    # 返回排序后的前 10 个 id
    return [x[0] for x in id_with_counts[:10]]  # 只保留前 10 个 id

# 存储每个 post_id 的结果
result = {}

# 处理所有 post_id
for post_id in file1_data.keys():
    sorted_ids = process_post(post_id)
    result[post_id] = sorted_ids

# 保存结果为 JSON 文件
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=4, ensure_ascii=False)

print(f"Results saved to {output_file}")
