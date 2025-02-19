MSRGAN: Modified Super Resolution GAN for Consumer GPUs
Hey there! 👋 This is MSRGAN, a modified version of SRGAN that I developed for my thesis project. The main goal was to create a super-resolution model that can actually run on regular consumer GPUs (yes, even your GTX 1660!), while still giving good results.

What Makes This Special? 🌟
The cool thing about MSRGAN is that it combines the best parts of SRGAN and ESRGAN, but with some tweaks that make it more practical for everyday use:
Works on consumer GPUs (tested on GTX 1660, RTX 3060 Ti, and RTX 4090)
Uses a mix of LeakyReLU and PReLU activations for better results
Achieves better PSNR and SSIM scores than base SRGAN and ESRGAN
Handles 4x upscaling of images
Can process both single-category and diverse image datasets

Results 📊
When tested on a single-category furniture dataset:
PSNR: 19.17 (beating SRGAN's 17.84 and ESRGAN's 18.73)
SSIM: 0.6731 (compared to SRGAN's 0.6271 and ESRGAN's 0.6184)

Requirements 💻
Python 3.8+
PyTorch
NVIDIA GPU with at least 6GB VRAM
CUDA support

Model Architecture 🏗️
MSRGAN builds on the RRDB architecture with some key modifications:
Modified dense blocks with alternating LeakyReLU and PReLU
Optimized network depth for consumer GPU compatibility
Enhanced perceptual loss using a pre-trained ResNet-18

Future Improvements 🔮
I've got some ideas for making this even better:
Add attention mechanisms for focusing on important image regions
Implement multiscale super-resolution
Explore knowledge distillation from larger models

Contributing 🤝
Feel free to contribute! Whether it's bug fixes, improvements, or new features, I'd love to see what you can add to this project.

Acknowledgments 🙏
Big thanks to my thesis supervisor Dr. Muhammad Iqbal Hossain and my teammates at BRAC University for their support and guidance throughout this project.

Note: This is an academic research project. While it performs well, there's always room for improvement!
