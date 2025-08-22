PineMind LLM
A powerful and efficient Large Language Model built from the ground up to classify messages as "spam" or "not spam." PineMind is designed for simplicity, performance, and fine-tuning on custom tasks.

🌟 Features
Custom Architecture: PineMind is built using a custom-designed transformer architecture, providing a deep understanding of natural language patterns.

Efficient Classification: The model is optimized for text classification, making it ideal for tasks like spam detection, sentiment analysis, and topic labeling.

Fine-tuning Ready: The codebase is structured to allow for easy fine-tuning on new datasets and for different classification tasks.

PyTorch Implementation: Built with PyTorch, providing a flexible and robust framework for deep learning.

🛠️ Installation
Clone the repository:

git clone https://github.com/your-username/PineMind.git
cd PineMind

Install dependencies:

pip install -r requirements.txt

(Note: You will need to create a requirements.txt file listing all necessary libraries like torch, pandas, tiktoken, etc.)

Download Pre-trained Weights:
This project uses pre-trained weights from GPT-2. You must download the 355M model and store the contents in a models/355M directory at the root of the project.

🚀 Usage
Training the Model
The core of the project is the classifier.ipynb Jupyter notebook. This notebook contains the full workflow, from data loading to model training and evaluation.

The model is a fine-tuned version of a GPT-2 355M model. The training was performed on a publicly available Spam Detection dataset, and the optimization was handled by the AdamW optimizer with a learning rate of 5
times10 
−5
  and a weight decay of 0.1.

Prepare your data: Ensure your data is in a CSV format with Text and Label columns.

Run the notebook: Open and run all the cells in classifier.ipynb to train the model.

Making Predictions
You can use the trained model to classify new text messages. The classify_review function in the notebook demonstrates this.

from your_module import classify_review, gpt, tokenizer, device, train_dataset

text_to_classify = "Hey, just wanted to check if we're still on for dinner tonight? Let me know!"
result = classify_review(
    text_to_classify,
    gpt,
    tokenizer,
    device,
    max_length=train_dataset.max_length
)

print(result)

📂 Project Structure
├── SMS-SPAM-Detection-PINEMIND-LLM/
│   ├── classifier.ipynb        # Main Jupyter notebook for training and usage
│   ├── spam.csv                # Sample dataset used for training (placeholder)
│   ├── train.csv               # Training data split
│   ├── validation.csv          # Validation data split
│   ├── test.csv                # Test data split
│   ├── review_classifier.pth   # Trained model weights (after running the notebook)
│   ├── README.md               # This file
│   └── models/
│       └── 355M/               # Directory for GPT-2 pre-trained weights
│           └── ... (model files)
└── .gitignore

🤝 Contributing
Contributions are welcome! Please feel free to open an issue or submit a pull request.

Fork the repository.

Create your feature branch (git checkout -b feature/AmazingFeature).

Commit your changes (git commit -m 'Add some AmazingFeature').

Push to the branch (git push origin feature/AmazingFeature).

Open a pull request.


📄 License
Distributed under the MIT License. See LICENSE for more information.

📧 Contact
Gourab Sen - gourabsen.21.2001@gmail.com
