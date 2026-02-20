# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-02-20 07:12 UTC_

Total papers shown: **6**


---

- **4D Monocular Surgical Reconstruction under Arbitrary Camera Motions**  
  Jiwei Shan, Zeyu Cai, Cheng-Tai Hsieh, Yirui Li, Hao Liu, Lijun Han, Hesheng Wang, Shing Shin Cheng  
  _2026-02-19_ · https://arxiv.org/abs/2602.17473v1  
  <details><summary>Abstract</summary>

  Reconstructing deformable surgical scenes from endoscopic videos is challenging and clinically important. Recent state-of-the-art methods based on implicit neural representations or 3D Gaussian splatting have made notable progress. However, most are designed for deformable scenes with fixed endoscope viewpoints and rely on stereo depth priors or accurate structure-from-motion for initialization and optimization, limiting their ability to handle monocular sequences with large camera motion in real clinical settings. To address this, we propose Local-EndoGS, a high-quality 4D reconstruction framework for monocular endoscopic sequences with arbitrary camera motion. Local-EndoGS introduces a progressive, window-based global representation that allocates local deformable scene models to each observed window, enabling scalability to long sequences with substantial motion. To overcome unreliable initialization without stereo depth or accurate structure-from-motion, we design a coarse-to-fine strategy integrating multi-view geometry, cross-window information, and monocular depth priors, providing a robust foundation for optimization. We further incorporate long-range 2D pixel trajectory constraints and physical motion priors to improve deformation plausibility. Experiments on three public endoscopic datasets with deformable scenes and varying camera motions show that Local-EndoGS consistently outperforms state-of-the-art methods in appearance quality and geometry. Ablation studies validate the effectiveness of our key designs. Code will be released upon acceptance at: https://github.com/IRMVLab/Local-EndoGS.

  </details>



- **Tree crop mapping of South America reveals links to deforestation and conservation**  
  Yuchang Jiang, Anton Raichuk, Xiaoye Tong, Vivien Sainte Fare Garnot, Daniel Ortiz-Gonzalo, Dan Morris, Konrad Schindler, Jan Dirk Wegner, Maxim Neumann  
  _2026-02-19_ · https://arxiv.org/abs/2602.17372v1  
  <details><summary>Abstract</summary>

  Monitoring tree crop expansion is vital for zero-deforestation policies like the European Union's Regulation on Deforestation-free Products (EUDR). However, these efforts are hindered by a lack of highresolution data distinguishing diverse agricultural systems from forests. Here, we present the first 10m-resolution tree crop map for South America, generated using a multi-modal, spatio-temporal deep learning model trained on Sentinel-1 and Sentinel-2 satellite imagery time series. The map identifies approximately 11 million hectares of tree crops, 23% of which is linked to 2000-2020 forest cover loss. Critically, our analysis reveals that existing regulatory maps supporting the EUDR often classify established agriculture, particularly smallholder agroforestry, as "forest". This discrepancy risks false deforestation alerts and unfair penalties for small-scale farmers. Our work mitigates this risk by providing a high-resolution baseline, supporting conservation policies that are effective, inclusive, and equitable.

  </details>



- **A Multi-modal Detection System for Infrastructure-based Freight Signal Priority**  
  Ziyan Zhang, Chuheng Wei, Xuanpeng Zhao, Siyan Li, Will Snyder, Mike Stas, Peng Hao, Kanok Boriboonsomsin, Guoyuan Wu  
  _2026-02-19_ · https://arxiv.org/abs/2602.17252v1  
  <details><summary>Abstract</summary>

  Freight vehicles approaching signalized intersections require reliable detection and motion estimation to support infrastructure-based Freight Signal Priority (FSP). Accurate and timely perception of vehicle type, position, and speed is essential for enabling effective priority control strategies. This paper presents the design, deployment, and evaluation of an infrastructure-based multi-modal freight vehicle detection system integrating LiDAR and camera sensors. A hybrid sensing architecture is adopted, consisting of an intersection-mounted subsystem and a midblock subsystem, connected via wireless communication for synchronized data transmission. The perception pipeline incorporates both clustering-based and deep learning-based detection methods with Kalman filter tracking to achieve stable real-time performance. LiDAR measurements are registered into geodetic reference frames to support lane-level localization and consistent vehicle tracking. Field evaluations demonstrate that the system can reliably monitor freight vehicle movements at high spatio-temporal resolution. The design and deployment provide practical insights for developing infrastructure-based sensing systems to support FSP applications.

  </details>



- **PartRAG: Retrieval-Augmented Part-Level 3D Generation and Editing**  
  Peize Li, Zeyu Zhang, Hao Tang  
  _2026-02-19_ · https://arxiv.org/abs/2602.17033v1  
  <details><summary>Abstract</summary>

  Single-image 3D generation with part-level structure remains challenging: learned priors struggle to cover the long tail of part geometries and maintain multi-view consistency, and existing systems provide limited support for precise, localized edits. We present PartRAG, a retrieval-augmented framework that integrates an external part database with a diffusion transformer to couple generation with an editable representation. To overcome the first challenge, we introduce a Hierarchical Contrastive Retrieval module that aligns dense image patches with 3D part latents at both part and object granularity, retrieving from a curated bank of 1,236 part-annotated assets to inject diverse, physically plausible exemplars into denoising. To overcome the second challenge, we add a masked, part-level editor that operates in a shared canonical space, enabling swaps, attribute refinements, and compositional updates without regenerating the whole object while preserving non-target parts and multi-view consistency. PartRAG achieves competitive results on Objaverse, ShapeNet, and ABO-reducing Chamfer Distance from 0.1726 to 0.1528 and raising F-Score from 0.7472 to 0.844 on Objaverse-with inference of 38s and interactive edits in 5-8s. Qualitatively, PartRAG produces sharper part boundaries, better thin-structure fidelity, and robust behavior on articulated objects. Code: https://github.com/AIGeeksGroup/PartRAG. Website: https://aigeeksgroup.github.io/PartRAG.

  </details>



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


