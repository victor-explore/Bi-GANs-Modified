# Modified BiGANs

This project implements a modified version of Bidirectional Generative Adversarial Networks (BiGANs) with additional components for improved performance and analysis.

## Key Features

1. **Decoder Network**: Implements a decoder network that takes generated images as input and outputs the original random variable used by the generator in the GAN.

2. **Reconstruction Loss**: Adds a norm-based reconstruction loss between the input to the generator and the output of the decoder, trained simultaneously with regular GAN losses.

3. **Classification Task**: Utilizes the trained decoder to obtain latent representations of input images, which are then used to train an MLP for a classification task.

4. **Performance Metrics**: Reports classification accuracy and F1 score to evaluate the quality of the learned representations.

5. **t-SNE Visualization**: Generates a t-SNE plot of the decoded latents for all real images to visualize class separability.

## How It Differs from Standard BiGANs

While standard BiGANs learn an encoder alongside the generator and discriminator, our modified approach:

1. Replaces the encoder with a decoder that works on generated images.
2. Introduces a reconstruction loss to ensure the decoder accurately recovers the input noise.
3. Uses the decoder to extract features from real images for downstream tasks.

This modification allows for better understanding of the generator's latent space and provides a way to extract meaningful features from both generated and real images.

## Usage

The main implementation can be found in the `Modified_BiGANs.ipynb` notebook. This notebook contains:

- Data loading and preprocessing
- Implementation of the Generator, Discriminator, and Decoder networks
- Training loop with combined GAN and reconstruction losses
- MLP classifier training on decoded latents
- Evaluation metrics calculation
- t-SNE visualization of latent space

## Results

The project demonstrates:

1. The ability to reconstruct input noise from generated images.
2. The effectiveness of learned representations for classification tasks.
3. Visualization of the latent space structure using t-SNE.

Detailed results, including classification accuracy and F1 score, can be found in the notebook.

## Requirements

- PyTorch
- torchvision
- numpy
- matplotlib
- scikit-learn (for t-SNE)
- tqdm

## Future Work

Potential areas for improvement and exploration include:

1. Experimenting with different architectures for the decoder network.
2. Investigating the impact of the reconstruction loss weight on overall GAN performance.
3. Applying the modified BiGAN to different datasets and tasks.
