# NeRF & Neural Radiance Fields

_Updated: 2026-01-21 06:54 UTC_

Total papers shown: **4**


---

- **One-Shot Refiner: Boosting Feed-forward Novel View Synthesis via One-Step Diffusion**  
  Yitong Dong, Qi Zhang, Minchao Jiang, Zhiqiang Wu, Qingnan Fan, Ying Feng, Huaqi Zhang, Hujun Bao, Guofeng Zhang  
  _2026-01-20_ · https://arxiv.org/abs/2601.14161v1  
  <details><summary>Abstract</summary>

  We present a novel framework for high-fidelity novel view synthesis (NVS) from sparse images, addressing key limitations in recent feed-forward 3D Gaussian Splatting (3DGS) methods built on Vision Transformer (ViT) backbones. While ViT-based pipelines offer strong geometric priors, they are often constrained by low-resolution inputs due to computational costs. Moreover, existing generative enhancement methods tend to be 3D-agnostic, resulting in inconsistent structures across views, especially in unseen regions. To overcome these challenges, we design a Dual-Domain Detail Perception Module, which enables handling high-resolution images without being limited by the ViT backbone, and endows Gaussians with additional features to store high-frequency details. We develop a feature-guided diffusion network, which can preserve high-frequency details during the restoration process. We introduce a unified training strategy that enables joint optimization of the ViT-based geometric backbone and the diffusion-based refinement module. Experiments demonstrate that our method can maintain superior generation quality across multiple datasets.

  </details>



- **VENI: Variational Encoder for Natural Illumination**  
  Paul Walker, James A. D. Gardner, Andreea Ardelean, William A. P. Smith, Bernhard Egger  
  _2026-01-20_ · https://arxiv.org/abs/2601.14079v1  
  <details><summary>Abstract</summary>

  Inverse rendering is an ill-posed problem, but priors like illumination priors, can simplify it. Existing work either disregards the spherical and rotation-equivariant nature of illumination environments or does not provide a well-behaved latent space. We propose a rotation-equivariant variational autoencoder that models natural illumination on the sphere without relying on 2D projections. To preserve the SO(2)-equivariance of environment maps, we use a novel Vector Neuron Vision Transformer (VN-ViT) as encoder and a rotation-equivariant conditional neural field as decoder. In the encoder, we reduce the equivariance from SO(3) to SO(2) using a novel SO(2)-equivariant fully connected layer, an extension of Vector Neurons. We show that our SO(2)-equivariant fully connected layer outperforms standard Vector Neurons when used in our SO(2)-equivariant model. Compared to previous methods, our variational autoencoder enables smoother interpolation in latent space and offers a more well-behaved latent space.

  </details>



- **CSGaussian: Progressive Rate-Distortion Compression and Segmentation for 3D Gaussian Splatting**  
  Yu-Jen Tseng, Chia-Hao Kao, Jing-Zhong Chen, Alessandro Gnutti, Shao-Yuan Lo, Yen-Yu Lin, Wen-Hsiao Peng  
  _2026-01-19_ · https://arxiv.org/abs/2601.12814v1  
  <details><summary>Abstract</summary>

  We present the first unified framework for rate-distortion-optimized compression and segmentation of 3D Gaussian Splatting (3DGS). While 3DGS has proven effective for both real-time rendering and semantic scene understanding, prior works have largely treated these tasks independently, leaving their joint consideration unexplored. Inspired by recent advances in rate-distortion-optimized 3DGS compression, this work integrates semantic learning into the compression pipeline to support decoder-side applications--such as scene editing and manipulation--that extend beyond traditional scene reconstruction and view synthesis. Our scheme features a lightweight implicit neural representation-based hyperprior, enabling efficient entropy coding of both color and semantic attributes while avoiding costly grid-based hyperprior as seen in many prior works. To facilitate compression and segmentation, we further develop compression-guided segmentation learning, consisting of quantization-aware training to enhance feature separability and a quality-aware weighting mechanism to suppress unreliable Gaussian primitives. Extensive experiments on the LERF and 3D-OVS datasets demonstrate that our approach significantly reduces transmission cost while preserving high rendering quality and strong segmentation performance.

  </details>



- **Near-Light Color Photometric Stereo for mono-Chromaticity non-lambertian surface**  
  Zonglin Li, Jieji Ren, Shuangfan Zhou, Heng Guo, Jinnuo Zhang, Jiang Zhou, Boxin Shi, Zhanyu Ma, Guoying Gu  
  _2026-01-19_ · https://arxiv.org/abs/2601.12666v1  
  <details><summary>Abstract</summary>

  Color photometric stereo enables single-shot surface reconstruction, extending conventional photometric stereo that requires multiple images of a static scene under varying illumination to dynamic scenarios. However, most existing approaches assume ideal distant lighting and Lambertian reflectance, leaving more practical near-light conditions and non-Lambertian surfaces underexplored. To overcome this limitation, we propose a framework that leverages neural implicit representations for depth and BRDF modeling under the assumption of mono-chromaticity (uniform chromaticity and homogeneous material), which alleviates the inherent ill-posedness of color photometric stereo and allows for detailed surface recovery from just one image. Furthermore, we design a compact optical tactile sensor to validate our approach. Experiments on both synthetic and real-world datasets demonstrate that our method achieves accurate and robust surface reconstruction.

  </details>


