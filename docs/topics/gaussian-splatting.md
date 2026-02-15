# Gaussian Splatting & 3DGS

_Updated: 2026-02-15 07:05 UTC_

Total papers shown: **4**


---

- **GSO-SLAM: Bidirectionally Coupled Gaussian Splatting and Direct Visual Odometry**  
  Jiung Yeon, Seongbo Ha, Hyeonwoo Yu  
  _2026-02-12_ · https://arxiv.org/abs/2602.11714v1  
  <details><summary>Abstract</summary>

  We propose GSO-SLAM, a real-time monocular dense SLAM system that leverages Gaussian scene representation. Unlike existing methods that couple tracking and mapping with a unified scene, incurring computational costs, or loosely integrate them with well-structured tracking frameworks, introducing redundancies, our method bidirectionally couples Visual Odometry (VO) and Gaussian Splatting (GS). Specifically, our approach formulates joint optimization within an Expectation-Maximization (EM) framework, enabling the simultaneous refinement of VO-derived semi-dense depth estimates and the GS representation without additional computational overhead. Moreover, we present Gaussian Splat Initialization, which utilizes image information, keyframe poses, and pixel associations from VO to produce close approximations to the final Gaussian scene, thereby eliminating the need for heuristic methods. Through extensive experiments, we validate the effectiveness of our method, showing that it not only operates in real time but also achieves state-of-the-art geometric/photometric fidelity of the reconstructed scene and tracking accuracy.

  </details>



- **TG-Field: Geometry-Aware Radiative Gaussian Fields for Tomographic Reconstruction**  
  Yuxiang Zhong, Jun Wei, Chaoqi Chen, Senyou An, Hui Huang  
  _2026-02-12_ · https://arxiv.org/abs/2602.11705v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has revolutionized 3D scene representation with superior efficiency and quality. While recent adaptations for computed tomography (CT) show promise, they struggle with severe artifacts under highly sparse-view projections and dynamic motions. To address these challenges, we propose Tomographic Geometry Field (TG-Field), a geometry-aware Gaussian deformation framework tailored for both static and dynamic CT reconstruction. A multi-resolution hash encoder is employed to capture local spatial priors, regularizing primitive parameters under ultra-sparse settings. We further extend the framework to dynamic reconstruction by introducing time-conditioned representations and a spatiotemporal attention block to adaptively aggregate features, thereby resolving spatiotemporal ambiguities and enforcing temporal coherence. In addition, a motion-flow network models fine-grained respiratory motion to track local anatomical deformations. Extensive experiments on synthetic and real-world datasets demonstrate that TG-Field consistently outperforms existing methods, achieving state-of-the-art reconstruction accuracy under highly sparse-view conditions.

  </details>



- **OMEGA-Avatar: One-shot Modeling of 360° Gaussian Avatars**  
  Zehao Xia, Yiqun Wang, Zhengda Lu, Kai Liu, Jun Xiao, Peter Wonka  
  _2026-02-12_ · https://arxiv.org/abs/2602.11693v1  
  <details><summary>Abstract</summary>

  Creating high-fidelity, animatable 3D avatars from a single image remains a formidable challenge. We identified three desirable attributes of avatar generation: 1) the method should be feed-forward, 2) model a 360° full-head, and 3) should be animation-ready. However, current work addresses only two of the three points simultaneously. To address these limitations, we propose OMEGA-Avatar, the first feed-forward framework that simultaneously generates a generalizable, 360°-complete, and animatable 3D Gaussian head from a single image. Starting from a feed-forward and animatable framework, we address the 360° full-head avatar generation problem with two novel components. First, to overcome poor hair modeling in full-head avatar generation, we introduce a semantic-aware mesh deformation module that integrates multi-view normals to optimize a FLAME head with hair while preserving its topology structure. Second, to enable effective feed-forward decoding of full-head features, we propose a multi-view feature splatting module that constructs a shared canonical UV representation from features across multiple views through differentiable bilinear splatting, hierarchical UV mapping, and visibility-aware fusion. This approach preserves both global structural coherence and local high-frequency details across all viewpoints, ensuring 360° consistency without per-instance optimization. Extensive experiments demonstrate that OMEGA-Avatar achieves state-of-the-art performance, significantly outperforming existing baselines in 360° full-head completeness while robustly preserving identity across different viewpoints.

  </details>



- **GR-Diffusion: 3D Gaussian Representation Meets Diffusion in Whole-Body PET Reconstruction**  
  Mengxiao Geng, Zijie Chen, Ran Hong, Bingxuan Li, Qiegen Liu  
  _2026-02-12_ · https://arxiv.org/abs/2602.11653v1  
  <details><summary>Abstract</summary>

  Positron emission tomography (PET) reconstruction is a critical challenge in molecular imaging, often hampered by noise amplification, structural blurring, and detail loss due to sparse sampling and the ill-posed nature of inverse problems. The three-dimensional discrete Gaussian representation (GR), which efficiently encodes 3D scenes using parameterized discrete Gaussian distributions, has shown promise in computer vision. In this work, we pro-pose a novel GR-Diffusion framework that synergistically integrates the geometric priors of GR with the generative power of diffusion models for 3D low-dose whole-body PET reconstruction. GR-Diffusion employs GR to generate a reference 3D PET image from projection data, establishing a physically grounded and structurally explicit benchmark that overcomes the low-pass limitations of conventional point-based or voxel-based methods. This reference image serves as a dual guide during the diffusion process, ensuring both global consistency and local accuracy. Specifically, we employ a hierarchical guidance mechanism based on the GR reference. Fine-grained guidance leverages differences to refine local details, while coarse-grained guidance uses multi-scale difference maps to correct deviations. This strategy allows the diffusion model to sequentially integrate the strong geometric prior from GR and recover sub-voxel information. Experimental results on the UDPET and Clinical datasets with varying dose levels show that GR-Diffusion outperforms state-of-the-art methods in enhancing 3D whole-body PET image quality and preserving physiological details.

  </details>


