# NeRF & Neural Radiance Fields

_Updated: 2026-02-19 07:14 UTC_

Total papers shown: **4**


---

- **Subtractive Modulative Network with Learnable Periodic Activations**  
  Tiou Wang, Zhuoqian Yang, Markus Flierl, Mathieu Salzmann, Sabine Süsstrunk  
  _2026-02-18_ · https://arxiv.org/abs/2602.16337v1  
  <details><summary>Abstract</summary>

  We propose the Subtractive Modulative Network (SMN), a novel, parameter-efficient Implicit Neural Representation (INR) architecture inspired by classical subtractive synthesis. The SMN is designed as a principled signal processing pipeline, featuring a learnable periodic activation layer (Oscillator) that generates a multi-frequency basis, and a series of modulative mask modules (Filters) that actively generate high-order harmonics. We provide both theoretical analysis and empirical validation for our design. Our SMN achieves a PSNR of $40+$ dB on two image datasets, comparing favorably against state-of-the-art methods in terms of both reconstruction accuracy and parameter efficiency. Furthermore, consistent advantage is observed on the challenging 3D NeRF novel view synthesis task. Supplementary materials are available at https://inrainbws.github.io/smn/.

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



- **Refine Now, Query Fast: A Decoupled Refinement Paradigm for Implicit Neural Fields**  
  Tianyu Xiong, Skylar Wurster, Han-Wei Shen  
  _2026-02-16_ · https://arxiv.org/abs/2602.15155v1  
  <details><summary>Abstract</summary>

  Implicit Neural Representations (INRs) have emerged as promising surrogates for large 3D scientific simulations due to their ability to continuously model spatial and conditional fields, yet they face a critical fidelity-speed dilemma: deep MLPs suffer from high inference cost, while efficient embedding-based models lack sufficient expressiveness. To resolve this, we propose the Decoupled Representation Refinement (DRR) architectural paradigm. DRR leverages a deep refiner network, alongside non-parametric transformations, in a one-time offline process to encode rich representations into a compact and efficient embedding structure. This approach decouples slow neural networks with high representational capacity from the fast inference path. We introduce DRR-Net, a simple network that validates this paradigm, and a novel data augmentation strategy, Variational Pairs (VP) for improving INRs under complex tasks like high-dimensional surrogate modeling. Experiments on several ensemble simulation datasets demonstrate that our approach achieves state-of-the-art fidelity, while being up to 27$\times$ faster at inference than high-fidelity baselines and remaining competitive with the fastest models. The DRR paradigm offers an effective strategy for building powerful and practical neural field surrogates and \rev{INRs in broader applications}, with a minimal compromise between speed and quality.

  </details>


