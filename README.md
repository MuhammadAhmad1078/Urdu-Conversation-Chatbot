# 🤖 Urdu Chatbot Transformer

A sophisticated Urdu language chatbot built using Transformer architecture with PyTorch. This project implements a complete end-to-end solution for Urdu conversational AI, featuring a modern Streamlit web interface and comprehensive training pipeline.

## 🌟 Features

- **Transformer Architecture**: Custom implementation with multi-head attention, positional encoding, and encoder-decoder structure
- **Urdu Language Support**: Full RTL (Right-to-Left) text support with proper Urdu text normalization
- **Interactive Web Interface**: Beautiful Streamlit app with Urdu UI and real-time chat
- **Comprehensive Training**: Complete training pipeline with BLEU evaluation and early stopping
- **Model Evaluation**: Built-in evaluation metrics including BLEU and ROUGE scores
- **Data Processing**: Automated data preprocessing and vocabulary building

## 🏗️ Architecture

### Model Components
- **Positional Encoding**: Sinusoidal positional embeddings for sequence understanding
- **Multi-Head Attention**: 2-head attention mechanism for context understanding
- **Encoder-Decoder**: 2-layer encoder and decoder with residual connections
- **Feed Forward Networks**: 512-dimensional hidden layers with ReLU activation
- **Dropout Regularization**: 0.15 dropout rate for better generalization

### Model Specifications
- **Embedding Dimension**: 512
- **Number of Heads**: 2
- **Encoder Layers**: 2
- **Decoder Layers**: 2
- **Feed Forward Dimension**: 1024
- **Vocabulary Size**: Dynamic (based on dataset)
- **Maximum Sequence Length**: 40 tokens

## 📁 Project Structure

```
urdu_chatbot_transformer/
├── data/
│   ├── final_main_dataset.tsv          # Original dataset
│   ├── char_to_num_vocab.pkl          # Character-level vocabulary
│   └── processed/
│       ├── train.csv                  # Training data
│       ├── val.csv                    # Validation data
│       ├── test.csv                   # Test data
│       └── vocab.json                 # Word-level vocabulary
├── models/                            # Trained model checkpoints
├── src/
│   ├── app_streamlit.py               # Streamlit web interface
│   ├── model.py                       # Transformer model implementation
│   ├── train.py                       # Training script
│   ├── inference.py                   # Model inference and generation
│   ├── dataset.py                     # Data loading and preprocessing
│   ├── preprocess.py                   # Data preprocessing pipeline
│   ├── evaluate.py                    # Model evaluation
│   ├── utils.py                       # Utility functions
│   └── inspect_dataset.py             # Dataset inspection tools
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd urdu_chatbot_transformer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the dataset**
   ```bash
   python src/preprocess.py
   ```

4. **Train the model**
   ```bash
   python src/train.py
   ```

5. **Launch the web interface**
   ```bash
   streamlit run src/app_streamlit.py
   ```

## 📊 Training

### Hyperparameters
- **Batch Size**: 64
- **Learning Rate**: 3e-4
- **Epochs**: 25
- **Gradient Clipping**: 1.0
- **Label Smoothing**: 0.1
- **Early Stopping Patience**: 5 epochs

### Training Process
1. **Data Loading**: Automatic train/validation split
2. **Model Initialization**: Custom Transformer architecture
3. **Training Loop**: Teacher forcing with causal masking
4. **Validation**: BLEU score evaluation on validation set
5. **Model Saving**: Best model saved based on validation BLEU

### Monitoring Training
The training script provides real-time feedback:
- Training loss per batch
- Average training loss per epoch
- Validation BLEU score
- Best model checkpoint saving

## 🎯 Usage

### Web Interface
1. Run `streamlit run src/app_streamlit.py`
2. Open your browser to the provided URL
3. Type your Urdu question in the input box
4. Click "بھیجیں" (Send) to get a response

### Command Line Interface
```python
from src.inference import load_model, generate_reply

# Load the trained model
model, vocab, inv_vocab, device = load_model()

# Generate a response
user_input = "آپ کیسے ہیں؟"
response = generate_reply(user_input, model, vocab, inv_vocab, device)
print(response)
```

### Programmatic Usage
```python
import torch
from src.model import TransformerChatbot
from src.inference import load_model, generate_reply

# Load model
model, vocab, inv_vocab, device = load_model()

# Chat loop
while True:
    user_input = input("👤 User (Urdu): ")
    if user_input.lower() in ["exit", "quit"]:
        break
    reply = generate_reply(user_input, model, vocab, inv_vocab, device)
    print(f"🤖 Bot: {reply}\n")
```

## 📈 Evaluation

### Metrics
- **BLEU Score**: Primary evaluation metric for translation quality
- **ROUGE Score**: Additional evaluation for text generation quality
- **Perplexity**: Model confidence measurement

### Evaluation Script
```bash
python src/evaluate.py
```

## 🔧 Configuration

### Model Configuration
Edit `src/train.py` to modify:
- Model architecture (layers, heads, dimensions)
- Training hyperparameters
- Data processing parameters

### Data Configuration
Edit `src/preprocess.py` to modify:
- Text normalization rules
- Vocabulary building parameters
- Train/validation split ratios

## 📝 Data Format

### Input Data
The model expects TSV format with the following structure:
```
sentence
آپ کیسے ہیں؟
میں ٹھیک ہوں، شکریہ
```

### Processed Data
After preprocessing, the data is split into:
- **Training set**: 80% of the data
- **Validation set**: 10% of the data  
- **Test set**: 10% of the data

## 🌐 Web Interface Features

### Urdu Support
- **RTL Text**: Proper right-to-left text rendering
- **Urdu Fonts**: Native Urdu font support
- **Input Validation**: Urdu text input validation
- **Response Formatting**: Proper Urdu text formatting

### User Experience
- **Real-time Chat**: Instant response generation
- **Chat History**: Conversation history tracking
- **Loading States**: Visual feedback during processing
- **Error Handling**: Graceful error handling and user feedback

## 🛠️ Development

### Adding New Features
1. **Model Improvements**: Modify `src/model.py`
2. **Training Enhancements**: Update `src/train.py`
3. **UI Improvements**: Enhance `src/app_streamlit.py`
4. **Data Processing**: Extend `src/preprocess.py`

### Debugging
- Use `src/inspect_dataset.py` to examine data
- Check model outputs with `src/inference.py`
- Monitor training with TensorBoard (if implemented)

## 📚 Dependencies

### Core Dependencies
- `torch`: PyTorch deep learning framework
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `streamlit`: Web interface
- `scikit-learn`: Machine learning utilities

### Evaluation Dependencies
- `sacrebleu`: BLEU score calculation
- `rouge-score`: ROUGE score calculation
- `tqdm`: Progress bars

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is part of an NLP assignment and is for educational purposes.

## 🎓 Academic Context

This project was developed as part of an NLP course assignment focusing on:
- Transformer architecture implementation
- Urdu language processing
- Sequence-to-sequence modeling
- Chatbot development
- Model evaluation and optimization

## 🔮 Future Enhancements

- **Beam Search Decoding**: Improve response quality
- **Attention Visualization**: Show attention weights
- **Multi-turn Conversations**: Context-aware responses
- **Model Quantization**: Optimize for deployment
- **API Integration**: REST API for external use
- **Mobile App**: Flutter-based mobile application

## 📞 Support

For questions or issues:
1. Check the documentation
2. Review the code comments
3. Open an issue on GitHub
4. Contact the development team

---

**Note**: This is an educational project demonstrating Transformer architecture for Urdu language processing. The model performance depends on the quality and size of the training dataset.
