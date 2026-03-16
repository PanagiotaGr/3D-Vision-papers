# SLAM & Localization

_Updated: 2026-03-16 07:46 UTC_

Total papers shown: **2**


---

- **VIRD: View-Invariant Representation through Dual-Axis Transformation for Cross-View Pose Estimation**  
  Juhye Park, Wooju Lee, Dasol Hong, Changki Sung, Youngwoo Seo, Dongwan Kang, Hyun Myung  
  _2026-03-13_ · https://arxiv.org/abs/2603.12918v1  
  <details><summary>Abstract</summary>

  Accurate global localization is crucial for autonomous driving and robotics, but GNSS-based approaches often degrade due to occlusion and multipath effects. As an emerging alternative, cross-view pose estimation predicts the 3-DoF camera pose corresponding to a ground-view image with respect to a geo-referenced satellite image. However, existing methods struggle to bridge the significant viewpoint gap between the ground and satellite views mainly due to limited spatial correspondences. We propose a novel cross-view pose estimation method that constructs view-invariant representations through dual-axis transformation (VIRD). VIRD first applies a polar transformation to the satellite view to establish horizontal correspondence, then uses context-enhanced positional attention on the ground and polar-transformed satellite features to resolve vertical misalignment, explicitly mitigating the viewpoint gap. A view-reconstruction loss is introduced to strengthen the view invariance further, encouraging the derived representations to reconstruct the original and cross-view images. Experiments on the KITTI and VIGOR datasets demonstrate that VIRD outperforms the state-of-the-art methods without orientation priors, reducing median position and orientation errors by 50.7% and 76.5% on KITTI, and 18.0% and 46.8% on VIGOR, respectively.

  </details>



- **Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis**  
  Yinuo Jiang, Jun Cheng, Yiran Wang, Cheng Cheng  
  _2026-03-13_ · https://arxiv.org/abs/2603.12903v1  
  <details><summary>Abstract</summary>

  Neural Radiance Fields (NeRF) have shown remarkable success in image novel view synthesis (NVS), inspiring extensions to LiDAR NVS. However, most methods heavily rely on accurate camera poses for scene reconstruction. The sparsity and textureless nature of LiDAR data also present distinct challenges, leading to geometric holes and discontinuous surfaces. To address these issues, we propose SG-NLF, a pose-free LiDAR NeRF framework that integrates spectral information with geometric consistency. Specifically, we design a hybrid representation based on spectral priors to reconstruct smooth geometry. For pose optimization, we construct a confidence-aware graph based on feature compatibility to achieve global alignment. In addition, an adversarial learning strategy is introduced to enforce cross-frame consistency, thereby enhancing reconstruction quality. Comprehensive experiments demonstrate the effectiveness of our framework, especially in challenging low-frequency scenarios. Compared to previous state-of-the-art methods, SG-NLF improves reconstruction quality and pose accuracy by over 35.8% and 68.8%. Our work can provide a novel perspective for LiDAR view synthesis.

  </details>


