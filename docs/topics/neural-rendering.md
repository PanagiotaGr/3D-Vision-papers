# Neural Rendering & View Synthesis

_Updated: 2026-03-14 07:04 UTC_

Total papers shown: **3**


---

- **NBAvatar: Neural Billboards Avatars with Realistic Hand-Face Interaction**  
  David Svitov, Mahtab Dahaghin  
  _2026-03-12_ · https://arxiv.org/abs/2603.12063v1  
  <details><summary>Abstract</summary>

  We present NBAvatar - a method for realistic rendering of head avatars handling non-rigid deformations caused by hand-face interaction. We introduce a novel representation for animated avatars by combining the training of oriented planar primitives with neural rendering. Such a combination of explicit and implicit representations enables NBAvatar to handle temporally and pose-consistent geometry, along with fine-grained appearance details provided by the neural rendering technique. In our experiments, we demonstrate that NBAvatar implicitly learns color transformations caused by face-hand interactions and surpasses existing approaches in terms of novel-view and novel-pose rendering quality. Specifically, NBAvatar achieves up to 30% LPIPS reduction under high-resolution megapixel rendering compared to Gaussian-based avatar methods, while also improving PSNR and SSIM, and achieves higher structural similarity compared to the state-of-the-art hand-face interaction method InteractAvatar.

  </details>



- **InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction**  
  Dingqiang Ye, Jiacong Xu, Jianglu Ping, Yuxiang Guo, Chao Fan, Vishal M. Patel  
  _2026-03-11_ · https://arxiv.org/abs/2603.11298v1  
  <details><summary>Abstract</summary>

  High dynamic range (HDR) novel view synthesis (NVS) aims to reconstruct HDR scenes from multi-exposure low dynamic range (LDR) images. Existing HDR pipelines heavily rely on known camera poses, well-initialized dense point clouds, and time-consuming per-scene optimization. Current feed-forward alternatives overlook the HDR problem by assuming exposure-invariant appearance. To bridge this gap, we propose InstantHDR, a feed-forward network that reconstructs 3D HDR scenes from uncalibrated multi-exposure LDR collections in a single forward pass. Specifically, we design a geometry-guided appearance modeling for multi-exposure fusion, and a meta-network for generalizable scene-specific tone mapping. Due to the lack of HDR scene data, we build a pre-training dataset, called HDR-Pretrain, for generalizable feed-forward HDR models, featuring 168 Blender-rendered scenes, diverse lighting types, and multiple camera response functions. Comprehensive experiments show that our InstantHDR delivers comparable synthesis performance to the state-of-the-art optimization-based HDR methods while enjoying $\sim700\times$ and $\sim20\times$ reconstruction speed improvement with our single-forward and post-optimization settings. All code, models, and datasets will be released after the review process.

  </details>



- **S2D: Sparse to Dense Lifting for 3D Reconstruction with Minimal Inputs**  
  Yuzhou Ji, Qijian Tian, He Zhu, Xiaoqi Jiang, Guangzhi Cao, Lizhuang Ma, Yuan Xie, Xin Tan  
  _2026-03-11_ · https://arxiv.org/abs/2603.10893v1  
  <details><summary>Abstract</summary>

  Explicit 3D representations have already become an essential medium for 3D simulation and understanding. However, the most commonly used point cloud and 3D Gaussian Splatting (3DGS) each suffer from non-photorealistic rendering and significant degradation under sparse inputs. In this paper, we introduce Sparse to Dense lifting (S2D), a novel pipeline that bridges the two representations and achieves high-quality 3DGS reconstruction with minimal inputs. Specifically, the S2D lifting is two-fold. We first present an efficient one-step diffusion model that lifts sparse point cloud for high-fidelity image artifact fixing. Meanwhile, to reconstruct 3D consistent scenes, we also design a corresponding reconstruction strategy with random sample drop and weighted gradient for robust model fitting from sparse input views to dense novel views. Extensive experiments show that S2D achieves the best consistency in generating novel view guidance and first-tier sparse view reconstruction quality under different input sparsity. By reconstructing stable scenes with the least possible captures among existing methods, S2D enables minimal input requirements for 3DGS applications.

  </details>


