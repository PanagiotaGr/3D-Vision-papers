# Gaussian Splatting & 3DGS

_Updated: 2026-02-23 07:19 UTC_

Total papers shown: **2**


---

- **Unifying Color and Lightness Correction with View-Adaptive Curve Adjustment for Robust 3D Novel View Synthesis**  
  Ziteng Cui, Shuhong Liu, Xiaoyu Dong, Xuangeng Chu, Lin Gu, Ming-Hsuan Yang, Tatsuya Harada  
  _2026-02-20_ · https://arxiv.org/abs/2602.18322v1  
  <details><summary>Abstract</summary>

  High-quality image acquisition in real-world environments remains challenging due to complex illumination variations and inherent limitations of camera imaging pipelines. These issues are exacerbated in multi-view capture, where differences in lighting, sensor responses, and image signal processor (ISP) configurations introduce photometric and chromatic inconsistencies that violate the assumptions of photometric consistency underlying modern 3D novel view synthesis (NVS) methods, including Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS), leading to degraded reconstruction and rendering quality. We propose Luminance-GS++, a 3DGS-based framework for robust NVS under diverse illumination conditions. Our method combines a globally view-adaptive lightness adjustment with a local pixel-wise residual refinement for precise color correction. We further design unsupervised objectives that jointly enforce lightness correction and multi-view geometric and photometric consistency. Extensive experiments demonstrate state-of-the-art performance across challenging scenarios, including low-light, overexposure, and complex luminance and chromatic variations. Unlike prior approaches that modify the underlying representation, our method preserves the explicit 3DGS formulation, improving reconstruction fidelity while maintaining real-time rendering efficiency.

  </details>



- **Diff2DGS: Reliable Reconstruction of Occluded Surgical Scenes via 2D Gaussian Splatting**  
  Tianyi Song, Danail Stoyanov, Evangelos Mazomenos, Francisco Vasconcelos  
  _2026-02-20_ · https://arxiv.org/abs/2602.18314v1  
  <details><summary>Abstract</summary>

  Real-time reconstruction of deformable surgical scenes is vital for advancing robotic surgery, improving surgeon guidance, and enabling automation. Recent methods achieve dense reconstructions from da Vinci robotic surgery videos, with Gaussian Splatting (GS) offering real-time performance via graphics acceleration. However, reconstruction quality in occluded regions remains limited, and depth accuracy has not been fully assessed, as benchmarks like EndoNeRF and StereoMIS lack 3D ground truth. We propose Diff2DGS, a novel two-stage framework for reliable 3D reconstruction of occluded surgical scenes. In the first stage, a diffusion-based video module with temporal priors inpaints tissue occluded by instruments with high spatial-temporal consistency. In the second stage, we adapt 2D Gaussian Splatting (2DGS) with a Learnable Deformation Model (LDM) to capture dynamic tissue deformation and anatomical geometry. We also extend evaluation beyond prior image-quality metrics by performing quantitative depth accuracy analysis on the SCARED dataset. Diff2DGS outperforms state-of-the-art approaches in both appearance and geometry, reaching 38.02 dB PSNR on EndoNeRF and 34.40 dB on StereoMIS. Furthermore, our experiments demonstrate that optimizing for image quality alone does not necessarily translate into optimal 3D reconstruction accuracy. To address this, we further optimize the depth quality of the reconstructed 3D results, ensuring more faithful geometry in addition to high-fidelity appearance.

  </details>


