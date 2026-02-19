# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-02-19 07:14 UTC_

Total papers shown: **3**


---

- **Position-Aware Scene-Appearance Disentanglement for Bidirectional Photoacoustic Microscopy Registration**  
  Yiwen Wang, Jiahao Qin  
  _2026-02-17_ · https://arxiv.org/abs/2602.15959v1  
  <details><summary>Abstract</summary>

  High-speed optical-resolution photoacoustic microscopy (OR-PAM) with bidirectional raster scanning doubles imaging speed but introduces coupled domain shift and geometric misalignment between forward and backward scan lines. Existing registration methods, constrained by brightness constancy assumptions, achieve limited alignment quality, while recent generative approaches address domain shift through complex architectures that lack temporal awareness across frames. We propose GPEReg-Net, a scene-appearance disentanglement framework that separates domain-invariant scene features from domain-specific appearance codes via Adaptive Instance Normalization (AdaIN), enabling direct image-to-image registration without explicit deformation field estimation. To exploit temporal structure in sequential acquisitions, we introduce a Global Position Encoding (GPE) module that combines learnable position embeddings with sinusoidal encoding and cross-frame attention, allowing the network to leverage context from neighboring frames for improved temporal coherence. On the OR-PAM-Reg-4K benchmark (432 test samples), GPEReg-Net achieves NCC of 0.953, SSIM of 0.932, and PSNR of 34.49dB, surpassing the state-of-the-art by 3.8% in SSIM and 1.99dB in PSNR while maintaining competitive NCC. Code is available at https://github.com/JiahaoQin/GPEReg-Net.

  </details>



- **NeRFscopy: Neural Radiance Fields for in-vivo Time-Varying Tissues from Endoscopy**  
  Laura Salort-Benejam, Antonio Agudo  
  _2026-02-17_ · https://arxiv.org/abs/2602.15775v1  
  <details><summary>Abstract</summary>

  Endoscopy is essential in medical imaging, used for diagnosis, prognosis and treatment. Developing a robust dynamic 3D reconstruction pipeline for endoscopic videos could enhance visualization, improve diagnostic accuracy, aid in treatment planning, and guide surgery procedures. However, challenges arise due to the deformable nature of the tissues, the use of monocular cameras, illumination changes, occlusions and unknown camera trajectories. Inspired by neural rendering, we introduce NeRFscopy, a self-supervised pipeline for novel view synthesis and 3D reconstruction of deformable endoscopic tissues from a monocular video. NeRFscopy includes a deformable model with a canonical radiance field and a time-dependent deformation field parameterized by SE(3) transformations. In addition, the color images are efficiently exploited by introducing sophisticated terms to learn a 3D implicit model without assuming any template or pre-trained model, solely from data. NeRFscopy achieves accurate results in terms of novel view synthesis, outperforming competing methods across various challenging endoscopy scenes.

  </details>



- **Time-Archival Camera Virtualization for Sports and Visual Performances**  
  Yunxiao Zhang, William Stone, Suryansh Kumar  
  _2026-02-16_ · https://arxiv.org/abs/2602.15181v1  
  <details><summary>Abstract</summary>

  Camera virtualization -- an emerging solution to novel view synthesis -- holds transformative potential for visual entertainment, live performances, and sports broadcasting by enabling the generation of photorealistic images from novel viewpoints using images from a limited set of calibrated multiple static physical cameras. Despite recent advances, achieving spatially and temporally coherent and photorealistic rendering of dynamic scenes with efficient time-archival capabilities, particularly in fast-paced sports and stage performances, remains challenging for existing approaches. Recent methods based on 3D Gaussian Splatting (3DGS) for dynamic scenes could offer real-time view-synthesis results. Yet, they are hindered by their dependence on accurate 3D point clouds from the structure-from-motion method and their inability to handle large, non-rigid, rapid motions of different subjects (e.g., flips, jumps, articulations, sudden player-to-player transitions). Moreover, independent motions of multiple subjects can break the Gaussian-tracking assumptions commonly used in 4DGS, ST-GS, and other dynamic splatting variants. This paper advocates reconsidering a neural volume rendering formulation for camera virtualization and efficient time-archival capabilities, making it useful for sports broadcasting and related applications. By modeling a dynamic scene as rigid transformations across multiple synchronized camera views at a given time, our method performs neural representation learning, providing enhanced visual rendering quality at test time. A key contribution of our approach is its support for time-archival, i.e., users can revisit any past temporal instance of a dynamic scene and can perform novel view synthesis, enabling retrospective rendering for replay, analysis, and archival of live events, a functionality absent in existing neural rendering approaches and novel view synthesis...

  </details>


