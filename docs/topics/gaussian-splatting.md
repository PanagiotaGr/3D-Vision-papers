# Gaussian Splatting & 3DGS

_Updated: 2026-02-02 07:16 UTC_

Total papers shown: **3**


---

- **EAG-PT: Emission-Aware Gaussians and Path Tracing for Indoor Scene Reconstruction and Editing**  
  Xijie Yang, Mulin Yu, Changjian Jiang, Kerui Ren, Tao Lu, Jiangmiao Pang, Dahua Lin, Bo Dai, Linning Xu  
  _2026-01-30_ · https://arxiv.org/abs/2601.23065v1  
  <details><summary>Abstract</summary>

  Recent reconstruction methods based on radiance field such as NeRF and 3DGS reproduce indoor scenes with high visual fidelity, but break down under scene editing due to baked illumination and the lack of explicit light transport. In contrast, physically based inverse rendering relies on mesh representations and path tracing, which enforce correct light transport but place strong requirements on geometric fidelity, becoming a practical bottleneck for real indoor scenes. In this work, we propose Emission-Aware Gaussians and Path Tracing (EAG-PT), aiming for physically based light transport with a unified 2D Gaussian representation. Our design is based on three cores: (1) using 2D Gaussians as a unified scene representation and transport-friendly geometry proxy that avoids reconstructed mesh, (2) explicitly separating emissive and non-emissive components during reconstruction for further scene editing, and (3) decoupling reconstruction from final rendering by using efficient single-bounce optimization and high-quality multi-bounce path tracing after scene editing. Experiments on synthetic and real indoor scenes show that EAG-PT produces more natural and physically consistent renders after editing than radiant scene reconstructions, while preserving finer geometric detail and avoiding mesh-induced artifacts compared to mesh-based inverse path tracing. These results suggest promising directions for future use in interior design, XR content creation, and embodied AI.

  </details>



- **Self-Supervised Slice-to-Volume Reconstruction with Gaussian Representations for Fetal MRI**  
  Yinsong Wang, Thomas Fletcher, Xinzhe Luo, Aine Travers Dineen, Rhodri Cusack, Chen Qin  
  _2026-01-30_ · https://arxiv.org/abs/2601.22990v1  
  <details><summary>Abstract</summary>

  Reconstructing 3D fetal MR volumes from motion-corrupted stacks of 2D slices is a crucial and challenging task. Conventional slice-to-volume reconstruction (SVR) methods are time-consuming and require multiple orthogonal stacks for reconstruction. While learning-based SVR approaches have significantly reduced the time required at the inference stage, they heavily rely on ground truth information for training, which is inaccessible in practice. To address these challenges, we propose GaussianSVR, a self-supervised framework for slice-to-volume reconstruction. GaussianSVR represents the target volume using 3D Gaussian representations to achieve high-fidelity reconstruction. It leverages a simulated forward slice acquisition model to enable self-supervised training, alleviating the need for ground-truth volumes. Furthermore, to enhance both accuracy and efficiency, we introduce a multi-resolution training strategy that jointly optimizes Gaussian parameters and spatial transformations across different resolution levels. Experiments show that GaussianSVR outperforms the baseline methods on fetal MR volumetric reconstruction. Code will be available upon acceptance.

  </details>



- **GaussianOcc3D: A Gaussian-Based Adaptive Multi-modal 3D Occupancy Prediction**  
  A. Enes Doruk, Hasan F. Ates  
  _2026-01-30_ · https://arxiv.org/abs/2601.22729v1  
  <details><summary>Abstract</summary>

  3D semantic occupancy prediction is a pivotal task in autonomous driving, providing a dense and fine-grained understanding of the surrounding environment, yet single-modality methods face trade-offs between camera semantics and LiDAR geometry. Existing multi-modal frameworks often struggle with modality heterogeneity, spatial misalignment, and the representation crisis--where voxels are computationally heavy and BEV alternatives are lossy. We present GaussianOcc3D, a multi-modal framework bridging camera and LiDAR through a memory-efficient, continuous 3D Gaussian representation. We introduce four modules: (1) LiDAR Depth Feature Aggregation (LDFA), using depth-wise deformable sampling to lift sparse signals onto Gaussian primitives; (2) Entropy-Based Feature Smoothing (EBFS) to mitigate domain noise; (3) Adaptive Camera-LiDAR Fusion (ACLF) with uncertainty-aware reweighting for sensor reliability; and (4) a Gauss-Mamba Head leveraging Selective State Space Models for global context with linear complexity. Evaluations on Occ3D, SurroundOcc, and SemanticKITTI benchmarks demonstrate state-of-the-art performance, achieving mIoU scores of 49.4%, 28.9%, and 25.2% respectively. GaussianOcc3D exhibits superior robustness across challenging rainy and nighttime conditions.

  </details>


