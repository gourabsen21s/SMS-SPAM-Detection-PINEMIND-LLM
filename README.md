🌲 PineMind LLM: SMS Spam ClassifierA powerful and efficient Large Language Model engineered for precision SMS spam detection. PineMind is a fine-tuned solution, meticulously built to be both performant and easily adaptable to new text classification tasks.🌟 Key FeaturesCustom Transformer Architecture: PineMind is built from the ground up on a streamlined transformer architecture. It incorporates core components like MultiHeadAttention, FeedForward layers with GELU activation, and LayerNorm to ensure robust and accurate understanding of text sequences.Highly Efficient Classification: By fine-tuning a pre-trained model on a specific task, PineMind achieves high accuracy in classifying messages as "spam" or "not spam." This approach provides superior performance compared to traditional methods.Fine-Tuning Ready: The modular design of the model and training pipeline allows for effortless adaptation. You can easily fine-tune PineMind on any new dataset for various text classification challenges, such as sentiment analysis or topic labeling.PyTorch Implementation: Built on the industry-standard PyTorch framework, PineMind offers a flexible and robust environment for deep learning. Its clean, well-commented code is easy to follow and modify.🛠️ Installation & SetupClone the repository:git clone https://github.com/your-username/SMS-SPAM-Detection-PINEMIND-LLM.git
cd SMS-SPAM-Detection-PINEMIND-LLM
Install dependencies:The project relies on a few key libraries. To install them, run the following command:pip install -r requirements.txt
(Note: A requirements.txt file is required. It should include torch, pandas, and tiktoken).Download Pre-trained GPT-2 Weights:PineMind is fine-tuned on the GPT-2 architecture. You must manually download the 355M pre-trained model checkpoint.Create a directory at the project root: models/355MPlace the downloaded model files (hparams.json, model.ckpt, etc.) into this directory.🚀 Getting StartedThe core of this project is the classifier.ipynb Jupyter notebook, which walks you through the entire workflow from start to finish.📝 Training the ModelThe model is a fine-tuned version of a GPT-2 355M model. The training was performed on a publicly available SMS Spam Detection dataset. The optimization was handled by the AdamW optimizer with a learning rate of 5times10−5 and a weight decay of 0.1.Steps:Prepare your data: Ensure your data is in a CSV file named spam.csv with Text and Label columns. The notebook will automatically handle the train/validation/test split.Run the notebook: Open and execute all the cells in classifier.ipynb to train the model, save the weights, and evaluate its performance.🎯 Making PredictionsAfter training, you can use the classify_review function to classify new text messages. The notebook demonstrates how to load the model and make predictions.import torch
import tiktoken
from your_module import GPTModel, classify_review

# Initialize tokenizer and model (based on your notebook)
tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ... (load gpt model and weights as in the notebook)

text_to_classify = "You have been selected to receive a FREE gift! Claim now!"
result = classify_review(
    text_to_classify,
    gpt,
    tokenizer,
    device,
    max_length=1024 # Or whatever max_length you used
)

print(f"The message is classified as: {result}")
📂 Project Structure.
├── SMS-SPAM-Detection-PINEMIND-LLM/
│   ├── classifier.ipynb        # Jupyter notebook containing the full project workflow.
│   ├── spam.csv                # The raw public SMS Spam Detection dataset.
│   ├── train.csv               # Processed training data.
│   ├── validation.csv          # Processed validation data.
│   ├── test.csv                # Processed test data.
│   ├── review_classifier.pth   # The fine-tuned model weights saved after training.
│   ├── README.md               # The project documentation (this file).
│   └── models/
│       └── 355M/               # Directory to store the GPT-2 pre-trained weights.
│           └── ...             # GPT-2 model files go here.
└── .gitignore
🤝 How to ContributeWe welcome contributions to make PineMind even better!Fork the repository.Create your feature branch: git checkout -b feature/your-feature-nameCommit your changes: git commit -m 'feat: Add your feature'Push to the branch: git push origin feature/your-feature-nameOpen a pull request and describe your changes.📄 LicensePineMind is distributed under the MIT License. See the LICENSE file for more details.📧 ContactGourab Sen - gourabsen.21.2001@gmail.comProject Link: https://github.com/your-username/SMS-SPAM-Detection-PINEMIND-LLM
