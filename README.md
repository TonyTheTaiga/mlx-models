# Neural Network Models Collection

A collection of neural network architectures implemented in various frameworks, with initial focus on [MLX](https://ml-explore.github.io/mlx/build/html/index.html), Apple's machine learning framework for efficient computation on Mac hardware.

## Overview

This repository contains implementations of various neural network architectures for study and experimentation. The initial models use MLX because it provides efficient computation without requiring high-end GPUs, making these models accessible for users with Apple Silicon Macs. Future implementations may target other frameworks.

## Models

### ResNet

Location: `networks/resnet/`

A general-purpose convolutional neural network architecture for image classification with skip connections that help with training deeper networks.

- Implements both standard ResNet and ResNet50 with bottleneck blocks
- Includes modern design variants (ResNet-B, ResNet-C, ResNet-D) as shown in `variants.png`

### Autoencoders - Super Resolution

Location: `networks/autoencoders/super_resolution/`

Models that enhance low-resolution images by upscaling them to higher resolution using deep learning.

- Uses encoder-decoder architecture with residual blocks
- Implements pixel shuffle technique for efficient upscaling
- Supports perceptual loss using VGG features
- Trained on the DIV2K dataset
- Example results can be found in `result.png`

### Transformers - Autoregressive

Location: `networks/transformers/autoregressive/`

Text generation model using a decoder-only transformer architecture, similar to models like GPT.

- Includes causal attention mechanism
- Implements positional encoding, multi-head attention, and feed-forward networks
- Uses efficient key-value caching for faster inference
- Features RMSNorm instead of LayerNorm
- Supports temperature-based sampling for text generation

### VGG16

Location: `networks/vgg16/`

Classic CNN architecture for image classification, also used as a feature extractor for perceptual loss in other models.

- Standard implementation with 13 convolutional layers and 3 fully connected layers
- Includes dropout for regularization
- Provides feature extraction capabilities at different network depths
- Includes weight conversion utility

### DDPM (Denoising Diffusion Probabilistic Models)

Location: `networks/ddpm/`

Generative model based on denoising diffusion processes for high-quality image synthesis.

- U-Net architecture with time conditioning for noise prediction
- Sinusoidal time embeddings with learned transformations
- FiLM (Feature-wise Linear Modulation) for time-conditional generation
- ResNet-style blocks with RMSNorm for stable training
- Skip connections between encoder and decoder paths
- Supports variable timesteps for flexible sampling schedules

### Utilities

Location: `networks/utils/`

Helper functions and components used across different models.

- Perceptual loss implementation using pre-trained feature extractors
- Allows weighting of different feature layers

## Requirements

- Python 3.11+
- Framework-specific dependencies (currently MLX)

## Future Work

- Additional model architectures
- Conversion tools between frameworks
