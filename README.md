# News Recommendation with DIN + BERT Warm-up 

This project implements a Deep Interest Network (DIN) for news recommendation on the MIND Dataset. 
It features a Semantic Warm-up strategy using BERT embeddings to solve the item cold-start problem.

## Performance (MIND-Large Validation) compared with SOTA
|  | Our Model | SOTA（Rank:1） | Avg. (TOP 150) |
| :--- | :--- |:--- |:--- |
| **Global AUC** | **0.7133** |**0.7304** |**0.697** |
| **Group AUC** | **0.6932** |**NONE** |**NONE** |
| **MRR** | **0.3818** |**0.3770** |**0.345** |
| **nDCG@5** | **0.3917** |**0.4718** |**0.378** |

## Key Features
### 1. Semantic Initialization
Utilizes the pre-trained **`all-MiniLM-L6-v2` (BERT)** model to encode news titles into high-quality semantic vectors. These vectors are compressed to **32 dimensions via PCA**, ensuring efficient computation while retaining rich semantic information to solve the item cold-start problem.

### 2. Attention-Based Interest Modeling
Unlike traditional pooling methods, this model leverages an **Attention Mechanism** to capture diverse user interests dynamically:
* **Dynamic Relevance:** Dynamically calculates the relevance score between the candidate news and each item in the user's reading history.
* **Noise Suppression:** Adaptively assigns higher weights to relevant historical behaviors while suppressing noise from accidental or irrelevant clicks, resulting in highly precise and personalized recommendations.

### 3. Robust Two-Stage Training Strategy
To effectively bridge the gap between general semantic knowledge (BERT) and specific user preference patterns (CTR task), we adopt a robust training pipeline:
* **Stage 1: Warm-up (Freezing Strategy):**
    * The pre-trained BERT embeddings are **frozen** (gradients not calculated).
    * Only the downstream DIN network (MLP) is trained with a higher learning rate (`1e-3`).
    * *Purpose:* Prevents noisy gradients from the randomly initialized MLP layers from distorting the high-quality semantic representations of the pre-trained language model.
* **Stage 2: Global Fine-tuning:**
    * All layers (including BERT embeddings) are **unfrozen** and trained jointly with a very low learning rate (`5e-5`).
    * *Purpose:* Performs **Domain Adaptation**, allowing generic semantic vectors to align with the specific content distribution and user click behaviors of the MIND dataset.

### 4. Multi-Modal Feature Fusion
To tackle data sparsity, the model constructs a dense, hierarchical representation for every news item instead of relying solely on sparse Item IDs:
* **Hierarchical Structure:** Integrates **News ID**, **Category** (e.g., Sports), and **Subcategory** (e.g., Basketball_NBA).
* **Cold-Start Robustness:** By incorporating category features, the model can leverage **generalized user interests** (e.g., recommending a new "Tech" article to a tech-savvy user) even if the specific News ID has no interaction history.
* **DIN Integration:** The attention mechanism is applied across both ID and category sequences, capturing user interests at multiple levels of granularity.

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
