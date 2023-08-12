Of course! Here's the complete content in Markdown format with proper headers:

```markdown
# Image Captioning

## Overview

This repository contains my code for training and generating image captions using a deep learning model. The model utilizes a convolutional neural network (CNN) for image feature extraction and a recurrent neural network (RNN) for generating captions. This was my first attempt at implementing a deep learning model from scratch in which I applied my theortical knowledge of Bidirectional LSTMs and CNNs. I have used the flickr8K dataset for training and testing the model. The model was trained on a GPU-enabled machine with an NVIDIA GeForce GTX 1050 Ti GPU. The model was trained for 30 epochs and the best results were obtained with the following hyperparameters which can be found later in this document:

## Prerequisites

- Python 3.x
- PyTorch (>=1.7.0)
- torchvision
- NLTK (Natural Language Toolkit) for tokenization
- PIL (Python Imaging Library)
- Numpy

You can install the required Python packages using the following command:

```bash
pip install torch torchvision nltk PIL numpy
```

## Dataset

The code assumes you have a dataset of images and corresponding caption descriptions. Organize your dataset and create a data loader for training. Refer to your dataset structure for more information.

## Usage

1. **Clone this repository**:

    ```bash
    git clone https://github.com/Dhanvine321/AI-Image-Captioner-using-Bidirectional-LSTM-and-CNN.git
    cd image-captioning
    ```

2. **Prepare your dataset and adjust paths in the code**.

3. **Training**:

    Edit the hyperparameters, then run the respective cells to start training:
    I have experimented with different hyperparameters and the best results were obtained with the following parameters:

    ```python
    # Hyperparameters
    embed_size = 256
    hidden_size = 512
    vocab_size = len(vocab)
    num_epochs = 30
    batch_size = 64
    num_layers = 1
    ```

    I am also looking at the possibility of incorporating attention mechanism in the model to improve the results.

4. **Generating Captions**:

    Edit paths and parameters in `image_to_caption.ipynb`, then run the respective cells to generate captions for new images:


## Model Architecture

The model architecture consists of an Encoder-Decoder framework. The encoder is a pre-trained ResNet-50 CNN for feature extraction, and the decoder is an LSTM-based RNN for generating captions.

## Improvements

While the current image captioning model has achieved notable progress, there remain areas that could benefit from further refinement and exploration:

1. **Better Feature Identification**: Enhancing feature extraction techniques could lead to improved recognition and capture of image details that correspond accurately to specific captions.

2. **Attention Mechanism**: Integrating an attention mechanism into the model architecture may enhance its ability to focus on relevant parts of an image when generating captions, potentially leading to more contextually coherent descriptions.

3. **Fine-Tuning Pre-trained Model**: Exploring the possibility of fine-tuning the pre-trained ResNet-50 model for feature extraction could potentially result in more effective image feature representations.

4. **Diverse Caption Generation**: Investigating methods for generating diverse and varied captions for a single image could enhance the overall quality and uniqueness of the model's outputs.

5. **Hyperparameter Tuning**: Continuing to experiment with various hyperparameters can contribute to finding the optimal combination that improves caption quality and accelerates model convergence.

## Credits

The code in this repository was inspired by various tutorials and resources. Credits to the authors of those resources for providing valuable insights and knowledge and also for creators of the datasets used in this project.