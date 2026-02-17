# NeRF & Neural Radiance Fields

_Updated: 2026-02-17 07:13 UTC_

Total papers shown: **6**


---

- **Gaussian Mesh Renderer for Lightweight Differentiable Rendering**  
  Xinpeng Liu, Fumio Okura  
  _2026-02-16_ · https://arxiv.org/abs/2602.14493v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has enabled high-fidelity virtualization with fast rendering and optimization for novel view synthesis. On the other hand, triangle mesh models still remain a popular choice for surface reconstruction but suffer from slow or heavy optimization in traditional mesh-based differentiable renderers. To address this problem, we propose a new lightweight differentiable mesh renderer leveraging the efficient rasterization process of 3DGS, named Gaussian Mesh Renderer (GMR), which tightly integrates the Gaussian and mesh representations. Each Gaussian primitive is analytically derived from the corresponding mesh triangle, preserving structural fidelity and enabling the gradient flow. Compared to the traditional mesh renderers, our method achieves smoother gradients, which especially contributes to better optimization using smaller batch sizes with limited memory. Our implementation is available in the public GitHub repository at https://github.com/huntorochi/Gaussian-Mesh-Renderer.

  </details>



- **Learnable Multi-level Discrete Wavelet Transforms for 3D Gaussian Splatting Frequency Modulation**  
  Hung Nguyen, An Le, Truong Nguyen  
  _2026-02-15_ · https://arxiv.org/abs/2602.14199v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has emerged as a powerful approach for novel view synthesis. However, the number of Gaussian primitives often grows substantially during training as finer scene details are reconstructed, leading to increased memory and storage costs. Recent coarse-to-fine strategies regulate Gaussian growth by modulating the frequency content of the ground-truth images. In particular, AutoOpti3DGS employs the learnable Discrete Wavelet Transform (DWT) to enable data-adaptive frequency modulation. Nevertheless, its modulation depth is limited by the 1-level DWT, and jointly optimizing wavelet regularization with 3D reconstruction introduces gradient competition that promotes excessive Gaussian densification. In this paper, we propose a multi-level DWT-based frequency modulation framework for 3DGS. By recursively decomposing the low-frequency subband, we construct a deeper curriculum that provides progressively coarser supervision during early training, consistently reducing Gaussian counts. Furthermore, we show that the modulation can be performed using only a single scaling parameter, rather than learning the full 2-tap high-pass filter. Experimental results on standard benchmarks demonstrate that our method further reduces Gaussian counts while maintaining competitive rendering quality.

  </details>



- **SemanticFeels: Semantic Labeling during In-Hand Manipulation**  
  Anas Al Shikh Khalil, Haozhi Qi, Roberto Calandra  
  _2026-02-15_ · https://arxiv.org/abs/2602.14099v1  
  <details><summary>Abstract</summary>

  As robots become increasingly integrated into everyday tasks, their ability to perceive both the shape and properties of objects during in-hand manipulation becomes critical for adaptive and intelligent behavior. We present SemanticFeels, an extension of the NeuralFeels framework that integrates semantic labeling with neural implicit shape representation, from vision and touch. To illustrate its application, we focus on material classification: high-resolution Digit tactile readings are processed by a fine-tuned EfficientNet-B0 convolutional neural network (CNN) to generate local material predictions, which are then embedded into an augmented signed distance field (SDF) network that jointly predicts geometry and continuous material regions. Experimental results show that the system achieves a high correspondence between predicted and actual materials on both single- and multi-material objects, with an average matching accuracy of 79.87% across multiple manipulation trials on a multi-material object.

  </details>



- **High-fidelity 3D reconstruction for planetary exploration**  
  Alfonso Martínez-Petersen, Levin Gerdes, David Rodríguez-Martínez, C. J. Pérez-del-Pulgar  
  _2026-02-14_ · https://arxiv.org/abs/2602.13909v1  
  <details><summary>Abstract</summary>

  Planetary exploration increasingly relies on autonomous robotic systems capable of perceiving, interpreting, and reconstructing their surroundings in the absence of global positioning or real-time communication with Earth. Rovers operating on planetary surfaces must navigate under sever environmental constraints, limited visual redundancy, and communication delays, making onboard spatial awareness and visual localization key components for mission success. Traditional techniques based on Structure-from-Motion (SfM) and Simultaneous Localization and Mapping (SLAM) provide geometric consistency but struggle to capture radiometric detail or to scale efficiently in unstructured, low-texture terrains typical of extraterrestrial environments. This work explores the integration of radiance field-based methods - specifically Neural Radiance Fields (NeRF) and Gaussian Splatting - into a unified, automated environment reconstruction pipeline for planetary robotics. Our system combines the Nerfstudio and COLMAP frameworks with a ROS2-compatible workflow capable of processing raw rover data directly from rosbag recordings. This approach enables the generation of dense, photorealistic, and metrically consistent 3D representations from minimal visual input, supporting improved perception and planning for autonomous systems operating in planetary-like conditions. The resulting pipeline established a foundation for future research in radiance field-based mapping, bridging the gap between geometric and neural representations in planetary exploration.

  </details>



- **Gaussian Sequences with Multi-Scale Dynamics for 4D Reconstruction from Monocular Casual Videos**  
  Can Li, Jie Gu, Jingmin Chen, Fangzhou Qiu, Lei Sun  
  _2026-02-14_ · https://arxiv.org/abs/2602.13806v1  
  <details><summary>Abstract</summary>

  Understanding dynamic scenes from casual videos is critical for scalable robot learning, yet four-dimensional (4D) reconstruction under strictly monocular settings remains highly ill-posed. To address this challenge, our key insight is that real-world dynamics exhibits a multi-scale regularity from object to particle level. To this end, we design the multi-scale dynamics mechanism that factorizes complex motion fields. Within this formulation, we propose Gaussian sequences with multi-scale dynamics, a novel representation for dynamic 3D Gaussians derived through compositions of multi-level motion. This layered structure substantially alleviates ambiguity of reconstruction and promotes physically plausible dynamics. We further incorporate multi-modal priors from vision foundation models to establish complementary supervision, constraining the solution space and improving the reconstruction fidelity. Our approach enables accurate and globally consistent 4D reconstruction from monocular casual videos. Experiments of dynamic novel-view synthesis (NVS) on benchmark and real-world manipulation datasets demonstrate considerable improvements over existing methods.

  </details>



- **Joint Orientation and Weight Optimization for Robust Watertight Surface Reconstruction via Dirichlet-Regularized Winding Fields**  
  Jiaze Li, Daisheng Jin, Fei Hou, Junhui Hou, Zheng Liu, Shiqing Xin, Wenping Wang, Ying He  
  _2026-02-14_ · https://arxiv.org/abs/2602.13801v1  
  <details><summary>Abstract</summary>

  We propose Dirichlet Winding Reconstruction (DiWR), a robust method for reconstructing watertight surfaces from unoriented point clouds with non-uniform sampling, noise, and outliers. Our method uses the generalized winding number (GWN) field as the target implicit representation and jointly optimizes point orientations, per-point area weights, and confidence coefficients in a single pipeline. The optimization minimizes the Dirichlet energy of the induced winding field together with additional GWN-based constraints, allowing DiWR to compensate for non-uniform sampling, reduce the impact of noise, and downweight outliers during reconstruction, with no reliance on separate preprocessing. We evaluate DiWR on point clouds from 3D Gaussian Splatting, a computer-vision pipeline, and corrupted graphics benchmarks. Experiments show that DiWR produces plausible watertight surfaces on these challenging inputs and outperforms both traditional multi-stage pipelines and recent joint orientation-reconstruction methods.

  </details>


