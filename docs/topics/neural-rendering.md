# Neural Rendering & View Synthesis

_Updated: 2026-02-28 06:55 UTC_

Total papers shown: **6**


---

- **DMAligner: Enhancing Image Alignment via Diffusion Model Based View Synthesis**  
  Xinglong Luo, Ao Luo, Zhengning Wang, Yueqi Yang, Chaoyu Feng, Lei Lei, Bing Zeng, Shuaicheng Liu  
  _2026-02-26_ · https://arxiv.org/abs/2602.23022v1  
  <details><summary>Abstract</summary>

  Image alignment is a fundamental task in computer vision with broad applications. Existing methods predominantly employ optical flow-based image warping. However, this technique is susceptible to common challenges such as occlusions and illumination variations, leading to degraded alignment visual quality and compromised accuracy in downstream tasks. In this paper, we present DMAligner, a diffusion-based framework for image alignment through alignment-oriented view synthesis. DMAligner is crafted to tackle the challenges in image alignment from a new perspective, employing a generation-based solution that showcases strong capabilities and avoids the problems associated with flow-based image warping. Specifically, we propose a Dynamics-aware Diffusion Training approach for learning conditional image generation, synthesizing a novel view for image alignment. This incorporates a Dynamics-aware Mask Producing (DMP) module to adaptively distinguish dynamic foreground regions from static backgrounds, enabling the diffusion model to more effectively handle challenges that classical methods struggle to solve. Furthermore, we develop the Dynamic Scene Image Alignment (DSIA) dataset using Blender, which includes 1,033 indoor and outdoor scenes with over 30K image pairs tailored for image alignment. Extensive experimental results demonstrate the superiority of the proposed approach on DSIA benchmarks, as well as on a series of widely-used video datasets for qualitative comparisons. Our code is available at https://github.com/boomluo02/DMAligner.

  </details>



- **ArtPro: Self-Supervised Articulated Object Reconstruction with Adaptive Integration of Mobility Proposals**  
  Xuelu Li, Zhaonan Wang, Xiaogang Wang, Lei Wu, Manyi Li, Changhe Tu  
  _2026-02-26_ · https://arxiv.org/abs/2602.22666v1  
  <details><summary>Abstract</summary>

  Reconstructing articulated objects into high-fidelity digital twins is crucial for applications such as robotic manipulation and interactive simulation. Recent self-supervised methods using differentiable rendering frameworks like 3D Gaussian Splatting remain highly sensitive to the initial part segmentation. Their reliance on heuristic clustering or pre-trained models often causes optimization to converge to local minima, especially for complex multi-part objects. To address these limitations, we propose ArtPro, a novel self-supervised framework that introduces adaptive integration of mobility proposals. Our approach begins with an over-segmentation initialization guided by geometry features and motion priors, generating part proposals with plausible motion hypotheses. During optimization, we dynamically merge these proposals by analyzing motion consistency among spatial neighbors, while a collision-aware motion pruning mechanism prevents erroneous kinematic estimation. Extensive experiments on both synthetic and real-world objects demonstrate that ArtPro achieves robust reconstruction of complex multi-part objects, significantly outperforming existing methods in accuracy and stability.

  </details>



- **DiffBMP: Differentiable Rendering with Bitmap Primitives**  
  Seongmin Hong, Junghun James Kim, Daehyeop Kim, Insoo Chung, Se Young Chun  
  _2026-02-26_ · https://arxiv.org/abs/2602.22625v1  
  <details><summary>Abstract</summary>

  We introduce DiffBMP, a scalable and efficient differentiable rendering engine for a collection of bitmap images. Our work addresses a limitation that traditional differentiable renderers are constrained to vector graphics, given that most images in the world are bitmaps. Our core contribution is a highly parallelized rendering pipeline, featuring a custom CUDA implementation for calculating gradients. This system can, for example, optimize the position, rotation, scale, color, and opacity of thousands of bitmap primitives all in under 1 min using a consumer GPU. We employ and validate several techniques to facilitate the optimization: soft rasterization via Gaussian blur, structure-aware initialization, noisy canvas, and specialized losses/heuristics for videos or spatially constrained images. We demonstrate DiffBMP is not just an isolated tool, but a practical one designed to integrate into creative workflows. It supports exporting compositions to a native, layered file format, and the entire framework is publicly accessible via an easy-to-hack Python package.

  </details>



- **BetterScene: 3D Scene Synthesis with Representation-Aligned Generative Model**  
  Yuci Han, Charles Toth, John E. Anderson, William J. Shuart, Alper Yilmaz  
  _2026-02-26_ · https://arxiv.org/abs/2602.22596v1  
  <details><summary>Abstract</summary>

  We present BetterScene, an approach to enhance novel view synthesis (NVS) quality for diverse real-world scenes using extremely sparse, unconstrained photos. BetterScene leverages the production-ready Stable Video Diffusion (SVD) model pretrained on billions of frames as a strong backbone, aiming to mitigate artifacts and recover view-consistent details at inference time. Conventional methods have developed similar diffusion-based solutions to address these challenges of novel view synthesis. Despite significant improvements, these methods typically rely on off-the-shelf pretrained diffusion priors and fine-tune only the UNet module while keeping other components frozen, which still leads to inconsistent details and artifacts even when incorporating geometry-aware regularizations like depth or semantic conditions. To address this, we investigate the latent space of the diffusion model and introduce two components: (1) temporal equivariance regularization and (2) vision foundation model-aligned representation, both applied to the variational autoencoder (VAE) module within the SVD pipeline. BetterScene integrates a feed-forward 3D Gaussian Splatting (3DGS) model to render features as inputs for the SVD enhancer and generate continuous, artifact-free, consistent novel views. We evaluate on the challenging DL3DV-10K dataset and demonstrate superior performance compared to state-of-the-art methods.

  </details>



- **SwiftNDC: Fast Neural Depth Correction for High-Fidelity 3D Reconstruction**  
  Kang Han, Wei Xiang, Lu Yu, Mathew Wyatt, Gaowen Liu, Ramana Rao Kompella  
  _2026-02-26_ · https://arxiv.org/abs/2602.22565v1  
  <details><summary>Abstract</summary>

  Depth-guided 3D reconstruction has gained popularity as a fast alternative to optimization-heavy approaches, yet existing methods still suffer from scale drift, multi-view inconsistencies, and the need for substantial refinement to achieve high-fidelity geometry. Here, we propose SwiftNDC, a fast and general framework built around a Neural Depth Correction field that produces cross-view consistent depth maps. From these refined depths, we generate a dense point cloud through back-projection and robust reprojection-error filtering, obtaining a clean and uniformly distributed geometric initialization for downstream reconstruction. This reliable dense geometry substantially accelerates 3D Gaussian Splatting (3DGS) for mesh reconstruction, enabling high-quality surfaces with significantly fewer optimization iterations. For novel-view synthesis, SwiftNDC can also improve 3DGS rendering quality, highlighting the benefits of strong geometric initialization. We conduct a comprehensive study across five datasets, including two for mesh reconstruction, as well as three for novel-view synthesis. SwiftNDC consistently reduces running time for accurate mesh reconstruction and boosts rendering fidelity for view synthesis, demonstrating the effectiveness of combining neural depth refinement with robust geometric initialization for high-fidelity and efficient 3D reconstruction.

  </details>



- **Geometry-as-context: Modulating Explicit 3D in Scene-consistent Video Generation to Geometry Context**  
  JiaKui Hu, Jialun Liu, Liying Yang, Xinliang Zhang, Kaiwen Li, Shuang Zeng, Yuanwei Li, Haibin Huang, Chi Zhang, Yanye Lu  
  _2026-02-25_ · https://arxiv.org/abs/2602.21929v1  
  <details><summary>Abstract</summary>

  Scene-consistent video generation aims to create videos that explore 3D scenes based on a camera trajectory. Previous methods rely on video generation models with external memory for consistency, or iterative 3D reconstruction and inpainting, which accumulate errors during inference due to incorrect intermediary outputs, non-differentiable processes, and separate models. To overcome these limitations, we introduce ``geometry-as-context". It iteratively completes the following steps using an autoregressive camera-controlled video generation model: (1) estimates the geometry of the current view necessary for 3D reconstruction, and (2) simulates and restores novel view images rendered by the 3D scene. Under this multi-task framework, we develop the camera gated attention module to enhance the model's capability to effectively leverage camera poses. During the training phase, text contexts are utilized to ascertain whether geometric or RGB images should be generated. To ensure that the model can generate RGB-only outputs during inference, the geometry context is randomly dropped from the interleaved text-image-geometry training sequence. The method has been tested on scene video generation with one-direction and forth-and-back trajectories. The results show its superiority over previous approaches in maintaining scene consistency and camera control.

  </details>


