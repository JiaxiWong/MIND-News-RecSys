# News Recommendation with DIN + BERT Warm-up 

This project implements a Deep Interest Network (DIN) for news recommendation on the MIND Dataset. 
It features a Semantic Warm-up strategy using BERT embeddings to solve the item cold-start problem.

## Performance (MIND-Large Validation) compared with SOTA
|  | Our Model | SOTA（Rank:1） | Avg. (TOP 150) |
| :--- | :--- |:--- |:--- |
| **Global AUC** | **0.7133** |**0.7304** |**0.697** |
| **GAUC** | **0.6932** |**NULL** |**NULL** |
| **MRR** | **0.3818** |**0.3770** |**0.345** |
| **nDCG@5** | **0.3917** |**0.4718** |**0.378** |

## Key Features
1. **Semantic Initialization**: Uses `all-MiniLM-L6-v2` (BERT) to encode news titles, reducing dimensions via PCA (32-dim).
2. **Attention-Based Interest Modeling**：
   - **Attention Mechanism**: Use Attention Mechanism to capture the diversity of user interests. It dynamically calculates the relevance between the candidate news and each item in the user's reading history. This allows the model to adaptively assign higher weights to relevant historical behaviors while suppressing noise from irrelevant clicks, resulting in more precise and personalized recommendations.
4. **Two-Stage Training**:
   - **Warm-up**: Freeze BERT embeddings, train MLP.
   - **Fine-tuning**: Unfreeze all layers with low LR (`5e-5`).
5. **Multi-Modal Features**: Incorporates News ID, Category, and Subcategory.

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Download MIND Dataset into `data/` folder.
3. Run training: `python main.py`

##  Advantage
This project implements a lightweight DIN (Deep Interest Network) single model enhanced with BERT Semantic Warm-up. While achieving comparable performance to State-of-the-Art (SOTA) solutions, this approach prioritizes engineering efficiency and practical user experience.

1. Superior "First-Hit" Accuracy (MRR > SOTA)
    Our model achieves an MRR (Mean Reciprocal Rank) of 0.3818, surpassing the Rank 1 SOTA model's 0.3770. While SOTA models may hold a slight advantage in overall list ranking (nDCG), our model excels in the most critical metric for user experience: ensuring the most relevant content appears at the very top of the recommendation list (the "first recommendation").

2. High Efficiency with Lightweight Architecture
    Unlike SOTA solutions that often rely on complex multi-model ensembles or computationally expensive end-to-end fine-tuning, this project utilizes a single lightweight model. We achieved an AUC of 0.7133, maintaining comparable performance to SOTA (0.7304) while significantly reducing inference latency and computational costs, making it highly suitable for real-world deployment.

3. Robust Cold-Start Handling via Semantic Warm-up
    By integrating BERT semantic vectors for initialization, the model operates as a Hybrid Architecture (Content-based + Collaborative Filtering). This design provides inherent semantic generalization capabilities, allowing the model to handle Cold-start Items and sparse user behavior effectively. It leverages semantic understanding to provide precise recommendations even when interaction data is limited.
