# 将task.json中，单语言和跨语言的train和dev的id合并，并按照id值大小排序写入新的json文件

import json

# 加载task.json文件
with open(r'dataset\tasks.json', 'r') as file:
    tasks = json.load(file)


# 初始化
combined_monolingual_train_ids = []
combined_monolingual_dev_ids = []
combined_crosslingual_train_ids = sorted(tasks['crosslingual']['posts_train'])  # Sort before using
combined_crosslingual_dev_ids = sorted(tasks['crosslingual']['posts_dev'])      # Sort before using

# 合并单语言的train和dev
for lang, details in tasks['monolingual'].items():
    combined_monolingual_train_ids.extend(details['posts_train'])
    combined_monolingual_dev_ids.extend(details['posts_dev'])

# 合并跨语言的train和dev
combined_monolingual_train_ids = sorted(combined_monolingual_train_ids)
combined_monolingual_dev_ids = sorted(combined_monolingual_dev_ids)


# 保存到新的json文件中，这里只保存了单语言
with open('monolingual_combined_tasks.json', 'w') as outfile:
    combined_tasks = {
        'monolingual': {
            'train': combined_monolingual_train_ids,
            'dev': combined_monolingual_dev_ids
        },
        # 'crosslingual': {
        #     'train': combined_crosslingual_train_ids,
        #     'dev': combined_crosslingual_dev_ids
        # }
    }
    json.dump(combined_tasks, outfile, indent=4)
