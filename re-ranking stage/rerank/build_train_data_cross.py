import json
import random

# 文件路径
pairs_file = 'json_dataset/train_pairs.json'  # post_id 和 fact_check_id 的映射关系
post_file = 'json_dataset\cross\crosslingual_posts_train.json'    # post.json 文件路径
fact_check_file = 'json_dataset/cross/crosslingual_fact_checks.json'  # fact_check.json 文件路径
output_file = 'rerank_train_data_cross.json'  # 输出的新文件

# 读取 pairs.json 文件，获取 post_id 到 fact_check_id 的映射关系
with open(pairs_file, 'r', encoding='utf-8') as f:
    pairs = json.load(f)

# 读取 post.json 文件，创建 post_id 到 ocr_text、ocr_translation、text_original 和 text_translation 的拼接
with open(post_file, 'r', encoding='utf-8') as f:
    posts = json.load(f)
    post_dict = {}
    for post in posts:
        post_id = post.get('post_id', '')
        # ocr_text = post.get('ocr_text', '')
        ocr_translation = post.get('ocr_translation', '')
        # text_original = post.get('text_original', '')
        text_translation = post.get('text_translation', '')
        post_dict[post_id] = {
            # 'ocr_text': ocr_text,
            'ocr_translation': ocr_translation,
            # 'text_original': text_original,
            'text_translation': text_translation
        }

    # 输出检查
    print(f"post_dict (部分内容): {list(post_dict.items())[:2]}")  # 输出前两个项，确保数据加载正确

# 读取 fact_check.json 文件，创建 fact_check_id 到 claim_text、claim_translation、title_text 和 title_translation 的拼接
with open(fact_check_file, 'r', encoding='utf-8') as f:
    fact_checks = json.load(f)
    fact_check_dict = {}
    for fact_check in fact_checks:
        fact_check_id = fact_check.get('fact_check_id', '')
        # claim_text = fact_check.get('claim_text', '')
        claim_translation = fact_check.get('claim_translation', '')
        # title_text = fact_check.get('title_text', '')
        title_translation = fact_check.get('title_translation', '')
        # fact_check_dict[fact_check_id] = ' '.join([claim_text, claim_translation, title_text, title_translation])
        fact_check_dict[fact_check_id] = ' '.join([claim_translation, title_translation])

    # 输出检查
    print(f"fact_check_dict (部分内容): {list(fact_check_dict.items())[:2]}")  # 输出前两个项，确保数据加载正确

# 准备存储最终数据的列表
final_data = []

# 遍历 post.json 中的每一项，构建新的数据格式
for post in posts:
    post_id = post.get('post_id', '')  # 去除可能的空格
    print(f"正在处理 post_id: {post_id}")  # 打印当前 post_id

    # 获取 query：该 post_id 对应的拼接后的文本
    post_fields = post_dict.get(post_id, {})
    query = ' '.join([post_fields.get('ocr_translation', ''),
                      post_fields.get('text_translation', '')])
    
    if not query:
        print(f"警告: post_id {post_id} 的 query 为空")
    
    # 获取 fact_check_id 列表
    fact_check_ids = pairs.get(str(post_id), [])
    print(f"对应的 fact_check_ids: {fact_check_ids}") # 打印获取到的 fact_check_ids

    # 获取 pos：该 post_id 对应的所有 fact_check 的拼接后的 claim_text
    pos = []
    for fact_check_id in fact_check_ids:
        # 遍历 fact_check_id，获取对应的 fact_check 内容
        fact_check_content = fact_check_dict.get(fact_check_id, '')
        if fact_check_content:
            pos.append(fact_check_content)
        else:
            print(f"警告: fact_check_id {fact_check_id} 的内容为空")
    
    if not pos:
        print(f"警告: post_id {post_id} 的 pos 为空")
    
    # 获取 neg：从 fact_check.json 中随机选择 100 个非该 post 相关的 fact_check
    neg_fact_check_ids = [fc_id for fc_id in fact_check_dict.keys() if fc_id not in fact_check_ids]
    random_neg_ids = random.sample(neg_fact_check_ids, 100)  # 随机选择 100 个 neg fact_check_id
    neg = [fact_check_dict.get(fact_check_id, '') for fact_check_id in random_neg_ids]
    
    # 构建新的数据结构
    prompt = "Given a social media post, retrieve relevant fact-checks for the given post."
    new_data = {
        "query": query,
        "pos": pos,
        "neg": neg,
        "prompt": prompt
    }
    
    # 将新数据添加到 final_data 列表中
    final_data.append(new_data)

print(type(pairs))
# 打印前几行数据，检查加载的字典是否正确
print(list(pairs.items())[:5])  # 只打印前5个项

# 将最终结果保存到输出文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(final_data, f, indent=4, ensure_ascii=False)

print(f"数据已成功保存为 {output_file}")
