# Siamese Network for Image Matching

A Siamese Neural Network implementation using EfficientNet backbone for general image matching and similarity detection. This project demonstrates how to build and train a deep learning model for person re-identification and image similarity tasks.

## 🚀 Features

- **Siamese Network Architecture**: Implements Anchor-Positive-Negative (APN) triplet learning
- **EfficientNet Backbone**: Uses pre-trained EfficientNet-B0 for feature extraction
- **Triplet Loss**: Optimizes embeddings to minimize distance between similar images and maximize distance between dissimilar ones
- **Person Re-identification**: Specifically designed for person re-identification tasks
- **Image Similarity**: Can be adapted for general image similarity detection
- **PyTorch Implementation**: Built with PyTorch for flexibility and performance

## 📋 Requirements

- Python 3.8+
- PyTorch 1.8+
- CUDA-capable GPU (recommended)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SiameseNetowrkForImageMatching.git
cd SiameseNetowrkForImageMatching
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Dataset

This project uses the Person Re-identification dataset with the following structure:
- **Anchor**: Reference images
- **Positive**: Images of the same person as anchor
- **Negative**: Images of different people

The dataset should be organized as:
```
Person-Re-Id-Dataset/
├── train/
│   │── img1.jpg
│   │── img2.jpg
│   │── ...
│── train.csv
│── utils.py
```

## 🏃‍♂️ Quick Start

### Training the Model

1. **Prepare your dataset** following the structure above
2. **Update configuration** in the notebook:
   ```python
   # Dataset paths
   DATA_DIR = 'Person-Re-Id-Dataset/train/'
   CSV_FILE = 'Person-Re-Id-Dataset/train.csv'
   
   # Training hyperparameters
   BATCH_SIZE = 32
   LR = 0.001  # Learning rate
   EPOCHS = 15
   ```

3. **Run the training**:
   ```python
   # The notebook will automatically:
   # - Load and preprocess the data
   # - Create train/validation splits
   # - Initialize the Siamese network
   # - Train with triplet loss
   # - Save the best model weights
   ```

### Inference

```python
# Load trained model
model = APN_Model()
model.load_state_dict(torch.load('best_model.pt'))

# Get embeddings for similarity search
embeddings = get_encoding_csv(model, image_list)

# Find similar images
similar_images = find_similar_images(query_image, embeddings)
```

## 🏗️ Architecture

### Siamese Network Structure
```
Input Images (Anchor, Positive, Negative)
           ↓
    EfficientNet-B0 Backbone
           ↓
    Feature Embeddings (512D)
           ↓
    Triplet Loss Calculation
```

### Key Components

1. **EfficientNet-B0**: Pre-trained backbone for feature extraction
2. **Embedding Layer**: 512-dimensional feature vectors
3. **Triplet Loss**: Ensures similar images are close and dissimilar images are far apart
4. **Adam Optimizer**: Adaptive learning rate optimization

## 📈 Training Process

The model uses triplet loss with the following components:

- **Anchor**: Reference image
- **Positive**: Same person as anchor
- **Negative**: Different person from anchor

**Loss Function**:
```
Loss = max(d(A,P) - d(A,N) + margin, 0)
```
Where:
- `d(A,P)`: Distance between Anchor and Positive
- `d(A,N)`: Distance between Anchor and Negative
- `margin`: Minimum distance between positive and negative pairs

## 📊 Results

The model achieves:
- **Training Loss**: ~0.05 (after 15 epochs)
- **Validation Loss**: ~0.11 (best model)
- **Embedding Dimension**: 512 features
- **Input Size**: 128x64 pixels

## 🔧 Configuration

Key hyperparameters you can adjust:

```python
BATCH_SIZE = 32          # Batch size for training
LR = 0.001              # Learning rate
EPOCHS = 15             # Number of training epochs
EMBED_SIZE = 512        # Embedding dimension
MARGIN = 1.0            # Triplet loss margin
```

## 📁 Project Structure

```
SiameseNetowrkForImageMatching/
├── README.md
├── requirements.txt
├── LICENSE
├── DeepLearningwithPyTorch _ SiameseNetwork.ipynb
└── utils/
    └── plot_closest_imgs.py
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [EfficientNet](https://arxiv.org/abs/1905.11946) paper for the backbone architecture
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [timm](https://github.com/rwightman/pytorch-image-models) for pre-trained models
- Person Re-identification dataset contributors

## 📞 Contact

**Venkata Satya Murali Krishna Chittlu**
- Email: satyamuralikrishna13@gmail.com
- Contact: +1 (520) 283-5536
- GitHub: [@vsatyamuralikrishna](https://github.com/vsatyamuralikrishna)

---

⭐ If you found this project helpful, please give it a star!
