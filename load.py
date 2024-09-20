import ast
import os
import pandas as pd

our_dataset_path = 'dataset'
output_json_path = 'dataset/'  # 设置输出JSON的路径

posts_path = os.path.join(our_dataset_path, 'posts.csv')
fact_checks_path = os.path.join(our_dataset_path, 'fact_checks.csv')
fact_check_post_mapping_path = os.path.join(our_dataset_path, 'pairs.csv')

# 确保每个文件都存在
for path in [posts_path, fact_checks_path, fact_check_post_mapping_path]:
    assert os.path.isfile(path), f"File not found: {path}"

# 预处理文本字段，处理换行问题
parse_col = lambda s: ast.literal_eval(s.replace('\n', '\\n')) if s else s

# 读取并处理事实核查数据
df_fact_checks = pd.read_csv(fact_checks_path).fillna('').set_index('fact_check_id')
for col in ['fact_check_id', 'claim', 'instances', 'title']:
    df_fact_checks[col] = df_fact_checks[col].apply(parse_col)

# 读取并处理帖子数据
df_posts = pd.read_csv(posts_path).fillna('').set_index('post_id')
for col in ['instances', 'ocr', 'verdicts', 'text']:
    df_posts[col] = df_posts[col].apply(parse_col)

# 读取映射数据
df_fact_check_post_mapping = pd.read_csv(fact_check_post_mapping_path)

# 确保输出路径存在
if not os.path.exists(output_json_path):
    os.makedirs(output_json_path)

# 保存数据到JSON文件
df_fact_checks.to_json(os.path.join(output_json_path, 'fact_checks.json'), orient='records', index)
df_posts.to_json(os.path.join(output_json_path, 'posts.json'), orient='records', lines=True)
df_fact_check_post_mapping.to_json(os.path.join(output_json_path, 'fact_check_post_mapping.json'), orient='records', lines=True)
