# Signature Similarity and Verification Network

This repository contains a deep learning pipeline for analyzing and comparing handwriting signatures. It utilizes a Siamese-style convolutional neural network to determine the similarity between signature pairs and flag potential forgeries.

## Dataset
The model is trained on the Sig-DS dataset, which is structured into two main categories:
*   The dataset contains 1320 genuine signatures, consisting of 24 genuine samples for each of the 55 distinct writers.
*   It also contains 1320 forgery signatures, providing 24 forgeries for each of the 55 writers.

## Preprocessing Pipeline
To prepare the images for the neural network, the following preprocessing steps are applied:
*   Signatures are loaded and converted to grayscale.
*   Images are resized to a standardized dimension of 128x128 pixels.
*   Pixel values are normalized to a 0-1 range by dividing by 255.0.
*   The dataset is dynamically reorganized into positive pairs (two signatures from the same writer, labeled `1`) and negative pairs (two signatures from different writers, labeled `0`).
*   The generated pairs are split into training and testing sets using a 25% test split.
