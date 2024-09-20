import json
import pandas as pd

# Load the tasks.json to extract the required splits
with open(r'dataset\tasks.json', 'r') as file:
    tasks = json.load(file)

# Load the datasets
df_fact_checks = pd.read_csv(r'dataset\fact_checks.csv')
df_pairs = pd.read_csv(r'dataset\posts.csv')


# Collect all monolingual and crosslingual fact_check_ids and post_ids
monolingual_fact_check_ids = []
monolingual_post_ids_train = []
monolingual_post_ids_dev = []

for lang, details in tasks['monolingual'].items():
    monolingual_fact_check_ids.extend(details['fact_checks'])
    monolingual_post_ids_train.extend(details['posts_train'])
    monolingual_post_ids_dev.extend(details['posts_dev'])

crosslingual_fact_check_ids = tasks['crosslingual']['fact_checks']
crosslingual_post_ids_train = tasks['crosslingual']['posts_train']
crosslingual_post_ids_dev = tasks['crosslingual']['posts_dev']

# Filter fact checks and pairs
monolingual_fact_checks = df_fact_checks[df_fact_checks['fact_check_id'].isin(monolingual_fact_check_ids)]
crosslingual_fact_checks = df_fact_checks[df_fact_checks['fact_check_id'].isin(crosslingual_fact_check_ids)]

monolingual_pairs_train = df_pairs[df_pairs['post_id'].isin(monolingual_post_ids_train)]
monolingual_pairs_dev = df_pairs[df_pairs['post_id'].isin(monolingual_post_ids_dev)]

crosslingual_pairs_train = df_pairs[df_pairs['post_id'].isin(crosslingual_post_ids_train)]
crosslingual_pairs_dev = df_pairs[df_pairs['post_id'].isin(crosslingual_post_ids_dev)]

# Optional: Save outputs to CSV
monolingual_pairs_train.to_csv('monolingual_pairs_train.csv', index=False)
crosslingual_fact_checks.to_csv('crosslingual_fact_checks.csv', index=False)

# monolingual_pairs_dev.to_csv('monolingual_pairs_dev.csv', index=False)
# crosslingual_pairs_train.to_csv('crosslingual_pairs_train.csv', index=False)
# crosslingual_pairs_dev.to_csv('crosslingual_pairs_dev.csv', index=False)
