# News Recommendation with DIN + BERT Warm-up 

This project implements a Deep Interest Network (DIN) for news recommendation on the **MIND Dataset**. 
It features a **Semantic Warm-up** strategy using BERT embeddings to solve the item cold-start problem.

## Performance (MIND-Large Validation)
| Metric | Score |
| :--- | :--- |
| **Global AUC** | **0.7133** |
| **GAUC** | **0.6932** |
| **MRR** | **0.3818** |
| **nDCG@10** | **0.3917** |

## Key Features
1. **Semantic Initialization**: Uses `all-MiniLM-L6-v2` (BERT) to encode news titles, reducing dimensions via PCA (32-dim).
2. **Two-Stage Training**:
   - **Warm-up**: Freeze BERT embeddings, train MLP.
   - **Fine-tuning**: Unfreeze all layers with low LR (`5e-5`).
3. **Multi-Modal Features**: Incorporates News ID, Category, and Subcategory.

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Download MIND Dataset into `data/` folder.
3. Run training: `python main.py`

##  Visualization
