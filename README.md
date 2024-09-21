# Batik-Classification - ViT Model

## Overview
This project is an experiment following my deep learning project assigment to classify batik motifs ([Github link repo](https://github.com/marvelm69/Batik-Classification)) using MobileNetV2 and scratch CNN based model.In my previous project, the trained models showed unstable performance that could be seen in confusion matrix and loss of both train and validation. Furthermore, the uses of vision transformer model is preferable, since it can capture motif's relationship globally using self-attention mechanism.

## ViT Model
The Vision Transformer (ViT) is an image classification model that applies a Transformer architecture to fixed-size patches of an image. Each patch is linearly embedded with added position embeddings, forming a sequence of vectors fed into a Transformer encoder. For classification, a learnable "classification token" is appended to the sequence.  ViT's self-attention mechanism allows it to look at the entire batik motif globally. It can capture how different parts of the motif influence each other across the image, potentially identifying larger motifs as a whole without being restricted to local features. Comparing to the CNN based model, it relies on convolutional layers to detect local patterns first, gradually building up to a full understanding of the motif through hierarchical feature extraction. This can be particularly effective for batik motifs that have strong, repeatable local structures like lines, dots, or specific small shapes.

## Workflow

### Data Preprocessing
The data are obtained from [indonesian batik motifs](https://www.kaggle.com/datasets/dionisiusdh/indonesian-batik-motifs) and there're 150 images used in total (each class are 50 images). The image data splitted into train (80%) , validation (10%) and split (10%). Also, the images were downsampled into 224 x 224 pixels using LANCZOS resampling method . 

### Data Augmentation
This data augmentation pipeline for training applies random transformations to the images, such as slight rotations, horizontal and vertical flips, and color adjustments (brightness, contrast, saturation, and hue), along with random resized cropping to 224x224 pixels. These augmentations help the model generalize by simulating variations in the dataset. After augmentation, the images are converted to tensors, normalized with standard mean and standard deviation values, and scaled to the proper data type. The validation and test sets are only resized, converted, and normalized without augmentations to maintain consistency.

### Modelling
This model is based on a Vision Transformer (ViT) with pre-trained weights from PyTorch Framework. All pre-trained parameters are frozen, except for a custom classification head. The custom head consists of:
1. A linear layer mapping the ViT output to 512 features,
2. A LeakyReLU activation function with a 0.2 negative slope,
3. A dropout layer with 50% probability,
4. Another linear layer mapping to 3 output classes,
5. A Softmax function applied across the output to generate class probabilities.

### Training Model 
The K-Fold Cross-Validation training method splits the dataset into 5 parts (folds), where for each fold, the model is trained on a portion of the data and validated on the remaining fold. This process repeats k times, ensuring every data point is used in validation once. During each fold, the model alternates between training and evaluation, with performance metrics (loss and accuracy) tracked for both phases. The best validation accuracy is recorded, and the learning rate is adjusted based on validation performance. This technique helps improve model generalization and avoids overfitting to a single validation set.

## Evaluation Results 

### K-fold result (Based on final epoch : 29)

| Fold |  Train loss | Validation Loss | Train accuracy | Validation accuracy |
|------|-------------|-----------------|----------------|---------------------|
| 1    | 0.8607      | 1.0120          | 0.8364         | 0.5714              |
| 2    | 0.9021      | 0.8466          | 0.7727         | 0.8571              |
| 3    | 0.9085      | 0.8470          | 0.7364         | 0.8929              |
| 4    | 0.9033      | 0.8787          | 0.7207         | 0.7407              |
| 5    | 0.9         | 0.8838          | 0.7748         | 0.8519              |

### Metrics (Tested on Test dataset)

| Motifs  | Precision | Recall | F1-score |
|---------|-----------|--------|----------|
| Betawi  | 1         | 0.80   | 0.89     |
| Bali    | 0.55      | 0.67   | 0.57     |
| Keraton | 0.75      | 0.75   | 0.75     |

- **Accuracy** : 0.75
- **Macro-average** :
  - precision : 0.75
  - recall : 0.74
  - f1-score : 0.74 



