"""
MIND News Recommendation System with BERT + DIN
Author: [Your Name]
Date: 2025-11-29
Description: 
    This project implements a CTR prediction model using DIN (Deep Interest Network).
    It utilizes BERT for semantic warm-up of news embeddings and adopts a 
    two-stage training strategy (Warm-up + Fine-tuning).
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

# Deep Learning Imports
# Ensure you have tensorflow or keras-preprocessing installed
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sentence_transformers import SentenceTransformer

# Recommender System Imports (Only Torch-Rechub)
from torch_rechub.models.ranking import DIN
from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator

# --- 1. Global Configuration ---
class Config:
    # Paths (Modify these to your actual paths)
    TRAIN_PATH = 'data/MINDlarge_train'
    VAL_PATH = 'data/MINDlarge_dev'
    
    # Model Hyperparams
    EMBEDDING_DIM = 32  # BERT PCA dimension
    SEQ_LEN = 20        # History sequence length
    SEED = 2025
    
    # Training
    BATCH_SIZE = 4096
    LR_WARMUP = 1e-3
    LR_FINETUNE = 5e-5
    EPOCH_WARMUP = 1
    EPOCH_FINETUNE = 8
    
    # Environment (Auto-detect)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Ensure reproducibility
def seed_everything(seed=Config.SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

# --- 2. Data Processing Utils ---
def load_raw_data():
    print("🚀 Loading raw data...")
    news_cols = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
    behaviors_cols = ['impression_id', 'user_id', 'time', 'history', 'impressions']

    try:
        train_news = pd.read_csv(f'{Config.TRAIN_PATH}/news.tsv', sep='\t', header=None, names=news_cols)
        train_behaviors = pd.read_csv(f'{Config.TRAIN_PATH}/behaviors.tsv', sep='\t', header=None, names=behaviors_cols)
        
        valid_news = pd.read_csv(f'{Config.VAL_PATH}/news.tsv', sep='\t', header=None, names=news_cols)
        valid_behaviors = pd.read_csv(f'{Config.VAL_PATH}/behaviors.tsv', sep='\t', header=None, names=behaviors_cols)
    except FileNotFoundError:
        print("❌ Error: Data files not found. Please check Config.TRAIN_PATH.")
        exit(1)
    
    # Concat news for full vocabulary
    all_news = pd.concat([train_news, valid_news]).drop_duplicates(subset=['news_id'])
    return train_behaviors, valid_behaviors, all_news

def process_behaviors(df, mode='train', neg_ratio=4):
    """ Process behavior logs into samples with negative sampling """
    samples = []
    # Check if impression_id exists, otherwise use index
    has_imp_id = 'impression_id' in df.columns
    
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing {mode}"):
        user_id = str(row['user_id'])
        # Use existing impression_id or fallback to row index
        imp_id = row['impression_id'] if has_imp_id else idx
        
        hist = str(row['history']).split() if pd.notna(row['history']) else ['N0']
        hist_str = " ".join(hist)
        
        impressions = str(row['impressions']).split()
        pos_list = []
        neg_list = []
        
        for imp in impressions:
            if '-' not in imp: continue
            news_id, label = imp.split('-')
            label = int(label)
            if label == 1: pos_list.append(news_id)
            else: neg_list.append(news_id)
            
        if mode == 'train':
            for pos_news in pos_list:
                samples.append([user_id, hist_str, pos_news, 1, imp_id])
                # Negative Sampling
                if len(neg_list) >= neg_ratio:
                    negs = random.sample(neg_list, neg_ratio)
                else:
                    # Circular padding if not enough negatives
                    negs = neg_list * (neg_ratio // len(neg_list)) + neg_list[:neg_ratio % len(neg_list)] if neg_list else []
                for neg_news in negs:
                    samples.append([user_id, hist_str, neg_news, 0, imp_id])
        else:
            # Validation: keep all samples
            for imp in impressions:
                if '-' not in imp: continue
                news_id, label = imp.split('-')
                samples.append([user_id, hist_str, news_id, int(label), imp_id])
                
    return pd.DataFrame(samples, columns=['user_id', 'history_str', 'news_id', 'label', 'impression_id'])

# --- 3. Feature Engineering ---
def build_features(train_df, val_df, all_news):
    print("🛠 Building features & encoders...")
    
    # 3.1 Label Encoding User & News
    lbe_user = LabelEncoder()
    all_users = pd.concat([train_df['user_id'], val_df['user_id']]).unique()
    lbe_user.fit(all_users)
    vocab_size_user = len(lbe_user.classes_)
    
    train_df['user_id_idx'] = lbe_user.transform(train_df['user_id'])
    val_df['user_id_idx'] = lbe_user.transform(val_df['user_id'])
    
    # News ID
    lbe_news = LabelEncoder()
    # Collect all possible news IDs from News info and History
    all_news_ids = set(all_news['news_id']) | {'N0'} 
    lbe_news.fit(list(all_news_ids))
    vocab_size_news = len(lbe_news.classes_) + 1
    news_map = dict(zip(lbe_news.classes_, range(1, len(lbe_news.classes_) + 1))) # 0 is padding
    
    # Category & Subcategory
    all_news['category'] = all_news['category'].fillna('UNK')
    all_news['subcategory'] = all_news['subcategory'].fillna('UNK')
    
    lbe_cat = LabelEncoder()
    all_news['cat_idx'] = lbe_cat.fit_transform(all_news['category']) + 1
    vocab_size_cat = len(lbe_cat.classes_) + 1
    
    lbe_subcat = LabelEncoder()
    all_news['subcat_idx'] = lbe_subcat.fit_transform(all_news['subcategory']) + 1
    vocab_size_subcat = len(lbe_subcat.classes_) + 1
    
    # Mapping dicts
    news2cat = dict(zip(all_news['news_id'], all_news['cat_idx'])); news2cat['N0'] = 0
    news2subcat = dict(zip(all_news['news_id'], all_news['subcat_idx'])); news2subcat['N0'] = 0
    
    # 3.2 Apply Mapping
    def process_df_features(df):
        # Target Item Features
        df['news_id_idx'] = df['news_id'].apply(lambda x: news_map.get(x, 0))
        df['cat_idx'] = df['news_id'].apply(lambda x: news2cat.get(x, 0))
        df['subcat_idx'] = df['news_id'].apply(lambda x: news2subcat.get(x, 0))
        
        # History Sequence Features
        def get_seq(s, mapping):
            return [mapping.get(x, 0) for x in s.split()]
        
        df['hist_idx'] = df['history_str'].apply(lambda x: get_seq(x, news_map))
        df['hist_cat'] = df['history_str'].apply(lambda x: get_seq(x, news2cat))
        df['hist_subcat'] = df['history_str'].apply(lambda x: get_seq(x, news2subcat))
        
        # Padding
        df['history_seq'] = list(pad_sequences(df['hist_idx'], maxlen=Config.SEQ_LEN, padding='post', value=0))
        df['hist_cat_seq'] = list(pad_sequences(df['hist_cat'], maxlen=Config.SEQ_LEN, padding='post', value=0))
        df['hist_subcat_seq'] = list(pad_sequences(df['hist_subcat'], maxlen=Config.SEQ_LEN, padding='post', value=0))
        return df

    train_df = process_df_features(train_df)
    val_df = process_df_features(val_df)
    
    return train_df, val_df, lbe_news, news_map, vocab_size_user, vocab_size_news, vocab_size_cat, vocab_size_subcat

# --- 4. BERT Warm-up ---
def get_bert_embeddings(all_news, lbe_news, vocab_size_news):
    print("🤖 Generating BERT embeddings...")
    # Initialize zero matrix (Row 0 is padding)
    pretrained_emb = np.zeros((vocab_size_news, Config.EMBEDDING_DIM))
    
    # Load BERT (Use CPU or GPU automatically)
    model_bert = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Reindex to match LabelEncoder order
    titles = all_news.set_index('news_id').reindex(lbe_news.classes_)['title'].fillna("").tolist()
    
    # Encode
    embeddings = model_bert.encode(titles, batch_size=256, show_progress_bar=True)
    
    # PCA Reduction
    print(f"📉 Reducing dimension to {Config.EMBEDDING_DIM}...")
    pca = PCA(n_components=Config.EMBEDDING_DIM)
    embeddings_reduced = pca.fit_transform(embeddings)
    
    # Fill matrix (Index 1 to N)
    pretrained_emb[1:] = embeddings_reduced
    
    return torch.FloatTensor(pretrained_emb)

# --- 5. Metrics Calculation ---
def calculate_metrics(grouped_df):
    aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []
    for _, group in tqdm(grouped_df, desc="Evaluating"):
        labels = group['label'].values
        preds = group['pred'].values
        if len(np.unique(labels)) == 1: continue
        
        try: aucs.append(roc_auc_score(labels, preds))
        except: continue
            
        sorted_indices = np.argsort(preds)[::-1]
        sorted_labels = labels[sorted_indices]
        
        first_pos = np.where(sorted_labels == 1)[0]
        mrrs.append(1.0 / (first_pos[0] + 1) if len(first_pos) > 0 else 0)
        
        # Numpy 2.0 compatible ndcg calculation
        def ndcg(r, k):
            r = np.asarray(r, dtype=float)[:k]
            dcg = np.sum(r / np.log2(np.arange(2, r.size + 2))) if r.size else 0
            idcg = np.sum(np.ones_like(r) / np.log2(np.arange(2, r.size + 2))) if r.size else 0
            return dcg / idcg if idcg else 0

        ndcg5s.append(ndcg(sorted_labels, 5))
        ndcg10s.append(ndcg(sorted_labels, 10))
        
    return np.mean(aucs), np.mean(mrrs), np.mean(ndcg5s), np.mean(ndcg10s)

# --- 6. Main Pipeline ---
if __name__ == "__main__":
    seed_everything()
    
    # 1. Load & Process Data
    train_behaviors, valid_behaviors, all_news = load_raw_data()
    train_df = process_behaviors(train_behaviors, mode='train', neg_ratio=4)
    val_df = process_behaviors(valid_behaviors, mode='valid')
    
    # 2. Features
    train_df, val_df, lbe_news, news_map, vocab_user, vocab_news, vocab_cat, vocab_subcat = \
        build_features(train_df, val_df, all_news)
        
    # 3. BERT Embeddings
    pretrained_emb = get_bert_embeddings(all_news, lbe_news, vocab_news)
    
    # 4. Model Definition
    print("🏗 Defining DIN Model...")
    # Fix: use safe vocab size
    user_cols = [SparseFeature("user_id_idx", vocab_size=vocab_user + 100, embed_dim=Config.EMBEDDING_DIM)]
    item_cols = [
        SparseFeature("news_id_idx", vocab_size=vocab_news, embed_dim=Config.EMBEDDING_DIM),
        SparseFeature("cat_idx", vocab_size=vocab_cat, embed_dim=Config.EMBEDDING_DIM),
        SparseFeature("subcat_idx", vocab_size=vocab_subcat, embed_dim=Config.EMBEDDING_DIM)
    ]
    history_cols = [
        SequenceFeature("history_seq", vocab_size=vocab_news, embed_dim=Config.EMBEDDING_DIM, pooling="concat", shared_with="news_id_idx"),
        SequenceFeature("hist_cat_seq", vocab_size=vocab_cat, embed_dim=Config.EMBEDDING_DIM, pooling="concat", shared_with="cat_idx"),
        SequenceFeature("hist_subcat_seq", vocab_size=vocab_subcat, embed_dim=Config.EMBEDDING_DIM, pooling="concat", shared_with="subcat_idx")
    ]
    features = user_cols + item_cols
    
    model = DIN(features=features, history_features=history_cols, target_features=item_cols, 
                mlp_params={"dims": [256, 128], "dropout": 0.3}, attention_mlp_params={"dims": [64, 32]})
    
    # 5. Inject Weights
    embedding_param = None
    for name, param in model.named_parameters():
        if "news_id_idx" in name and "weight" in name:
            try:
                param.data.copy_(pretrained_emb)
                embedding_param = param
                print("✅ BERT weights injected.")
            except Exception as e:
                print(f"❌ Weight Injection Failed: {e}")
            break
            
    # 6. DataLoader
    input_cols = ['user_id_idx', 'news_id_idx', 'cat_idx', 'subcat_idx', 'history_seq', 'hist_cat_seq', 'hist_subcat_seq']
    train_dg = DataGenerator(x=train_df[input_cols], y=train_df['label'])
    val_dg = DataGenerator(x=val_df[input_cols], y=val_df['label'])
    train_loader, val_loader, _ = train_dg.generate_dataloader(x_val=val_df[input_cols], y_val=val_df['label'], 
                                                               batch_size=Config.BATCH_SIZE, num_workers=0)
    
    # 7. Training Strategy
    print(f"🚀 Training on Device: {Config.DEVICE}")
    
    # Stage 1: Warm-up
    print("🔥 Stage 1: Warm-up...")
    if embedding_param is not None:
        embedding_param.requires_grad = False
        
    trainer = CTRTrainer(model, optimizer_params={"lr": Config.LR_WARMUP, "weight_decay": 1e-4}, 
                         n_epoch=Config.EPOCH_WARMUP, device=Config.DEVICE)
    trainer.fit(train_loader, val_loader)
    
    # Stage 2: Fine-tuning
    print("🚀 Stage 2: Fine-tuning...")
    if embedding_param is not None:
        embedding_param.requires_grad = True
        
    trainer = CTRTrainer(model, optimizer_params={"lr": Config.LR_FINETUNE, "weight_decay": 1e-4}, 
                         n_epoch=Config.EPOCH_FINETUNE, device=Config.DEVICE)
    trainer.fit(train_loader, val_loader)
    
    # 8. Final Evaluation
    print("📊 Evaluating...")
    model.eval()
    # Fix: pass model explicitly to predict
    y_pred = trainer.predict(model, val_loader)
    val_df['pred'] = np.array(y_pred)
    
    gauc, mrr, ndcg5, ndcg10 = calculate_metrics(val_df.groupby('impression_id'))
    
    print(f"Global AUC: {roc_auc_score(val_df['label'], y_pred):.4f}")
    print(f"GAUC: {gauc:.4f} | MRR: {mrr:.4f} | nDCG@10: {ndcg10:.4f}")
    
    # Save
    torch.save(model.state_dict(), 'din_best_model.pth')
    print("✅ Training Complete. Model saved.")
