# 🌲 PineMind LLM: SMS Spam Classifier  

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)  
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white)  
![Tokenizer](https://img.shields.io/badge/Tokenizer-TikToken-green)  
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)  
![Status](https://img.shields.io/badge/Status-Active-success)  

A **powerful and efficient Large Language Model** engineered for **precision SMS spam detection**.  
PineMind is a fine-tuned solution, meticulously built to be both **performant** and **easily adaptable** to new text classification tasks.  

---

## 🌟 Key Features  

✅ **Custom Transformer Architecture** – Streamlined with MultiHeadAttention, GELU activation, FeedForward layers, and LayerNorm.  
✅ **Highly Efficient Classification** – Fine-tuned on SMS spam detection for **superior accuracy**.  
✅ **Fine-Tuning Ready** – Easily adaptable for sentiment analysis, topic labeling, or any classification task.  
✅ **PyTorch Implementation** – Clean, modular, and well-commented code for easy modification.  

---

## 🏗️ Project Architecture  

```
          ┌───────────────────────┐
          │      Input SMS        │
          └──────────┬────────────┘
                     │
           ┌─────────▼─────────┐
           │  Preprocessing     │
           │ (tokenization via │
           │      TikToken)    │
           └─────────┬─────────┘
                     │
           ┌─────────▼──────────┐
           │   PineMind LLM     │
           │ (GPT-2 Backbone +  │
           │  Custom Classifier)│
           └─────────┬──────────┘
                     │
           ┌─────────▼─────────┐
           │   Prediction       │
           │   Spam / Not Spam  │
           └────────────────────┘
```

---

## 🛠️ Installation & Setup  

1. **Clone the repository**  
   ```bash
   git clone https://github.com/gourabsen21s/SMS-SPAM-Detection-PINEMIND-LLM.git
   cd SMS-SPAM-Detection-PINEMIND-LLM
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
   *(requirements.txt should include: `torch`, `pandas`, `tiktoken`)*  

3. **Download Pre-trained GPT-2 Weights**  
   - PineMind is fine-tuned on GPT-2 (355M).  
   - Download weights manually and place them in:  
     ```
     models/355M/
     ```
     *(Files: `hparams.json`, `model.ckpt`, etc.)*  

---

## 🚀 Getting Started  

The core workflow is available in **`classifier.ipynb`**, which walks through:  

* Data preprocessing  
* Model training  
* Evaluation  
* Prediction on new messages  

---

## 📝 Training the Model  

- **Base Model**: GPT-2 (355M)  
- **Optimizer**: AdamW  
- **Learning Rate**: `5e-5`  
- **Weight Decay**: `0.1`  
- **Dataset**: Public SMS Spam Detection dataset  

**Steps:**  
1. Ensure your dataset is in `spam.csv` with columns: **Text, Label**  
2. Run `classifier.ipynb`  
3. The notebook will handle:  
   - Train/Validation/Test split  
   - Training & saving weights (`review_classifier.pth`)  
   - Performance evaluation  

---

## 📊 Evaluation  

📌 Below are placeholder results (replace with your actual metrics after training):  

| Metric         | Score   |
|----------------|---------|
| Accuracy       | 98.2%   |
| Precision      | 97.5%   |
| Recall         | 98.9%   |
| F1-Score       | 98.2%   |

---

## 🎯 Making Predictions  

Example usage after training:  

```python
import torch
import tiktoken
from your_module import GPTModel, classify_review

# Initialize tokenizer & device
tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example text
text_to_classify = "You have been selected to receive a FREE gift! Claim now!"

# Predict
result = classify_review(
    text_to_classify,
    gpt,
    tokenizer,
    device,
    max_length=1024  # adjust to your model
)

print(f"The message is classified as: {result}")
```

---

## 📂 Project Structure  

```
SMS-SPAM-Detection-PINEMIND-LLM/
├── classifier.ipynb        # Full project workflow
├── spam.csv                # Raw SMS Spam Detection dataset
├── train.csv               # Processed training data
├── validation.csv          # Processed validation data
├── test.csv                # Processed test data
├── review_classifier.pth   # Fine-tuned model weights
├── README.md               # Project documentation
└── models/
    └── 355M/               # GPT-2 pre-trained weights
```

---

## 🚧 Future Enhancements  

🔹 Extend support for **multi-class classification** (e.g., topic detection).  
🔹 Add **web API (Flask/FastAPI)** for real-time SMS classification.  
🔹 Integrate into **mobile apps** for on-device spam filtering.  
🔹 Experiment with **larger transformer backbones** (GPT-NeoX, LLaMA).  

---

## 🤝 Contributing  

We welcome contributions to make PineMind even better!  

1. Fork the repository  
2. Create your branch:  
   ```bash
   git checkout -b feature/your-feature-name
   ```  
3. Commit your changes:  
   ```bash
   git commit -m "feat: Add your feature"
   ```  
4. Push to the branch:  
   ```bash
   git push origin feature/your-feature-name
   ```  
5. Open a Pull Request 🚀  

---

## 📄 License  

PineMind is distributed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.  

---

## 📧 Contact  

👤 **Gourab Sen**  
📩 Email: [gourabsen.21.2001@gmail.com](mailto:gourabsen.21.2001@gmail.com)  
🔗 Project Link: [SMS-SPAM-Detection-PINEMIND-LLM](https://github.com/your-username/SMS-SPAM-Detection-PINEMIND-LLM)  

---
✨ *“Stop spam in its tracks with PineMind LLM!”* ✨
