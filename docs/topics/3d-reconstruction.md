# 3D Reconstruction

_Updated: 2026-02-19 07:14 UTC_

Total papers shown: **7**


---

- **Breaking the Sub-Millimeter Barrier: Eyeframe Acquisition from Color Images**  
  Manel Guzmán, Antonio Agudo  
  _2026-02-18_ · https://arxiv.org/abs/2602.16281v1  
  <details><summary>Abstract</summary>

  Eyeframe lens tracing is an important process in the optical industry that requires sub-millimeter precision to ensure proper lens fitting and optimal vision correction. Traditional frame tracers rely on mechanical tools that need precise positioning and calibration, which are time-consuming and require additional equipment, creating an inefficient workflow for opticians. This work presents a novel approach based on artificial vision that utilizes multi-view information. The proposed algorithm operates on images captured from an InVision system. The full pipeline includes image acquisition, frame segmentation to isolate the eyeframe from background, depth estimation to obtain 3D spatial information, and multi-view processing that integrates segmented RGB images with depth data for precise frame contour measurement. To this end, different configurations and variants are proposed and analyzed on real data, providing competitive measurements from still color images with respect to other solutions, while eliminating the need for specialized tracing equipment and reducing workflow complexity for optical technicians.

  </details>



- **NeRFscopy: Neural Radiance Fields for in-vivo Time-Varying Tissues from Endoscopy**  
  Laura Salort-Benejam, Antonio Agudo  
  _2026-02-17_ · https://arxiv.org/abs/2602.15775v1  
  <details><summary>Abstract</summary>

  Endoscopy is essential in medical imaging, used for diagnosis, prognosis and treatment. Developing a robust dynamic 3D reconstruction pipeline for endoscopic videos could enhance visualization, improve diagnostic accuracy, aid in treatment planning, and guide surgery procedures. However, challenges arise due to the deformable nature of the tissues, the use of monocular cameras, illumination changes, occlusions and unknown camera trajectories. Inspired by neural rendering, we introduce NeRFscopy, a self-supervised pipeline for novel view synthesis and 3D reconstruction of deformable endoscopic tissues from a monocular video. NeRFscopy includes a deformable model with a canonical radiance field and a time-dependent deformation field parameterized by SE(3) transformations. In addition, the color images are efficiently exploited by introducing sophisticated terms to learn a 3D implicit model without assuming any template or pre-trained model, solely from data. NeRFscopy achieves accurate results in terms of novel view synthesis, outperforming competing methods across various challenging endoscopy scenes.

  </details>



- **An Industrial Dataset for Scene Acquisitions and Functional Schematics Alignment**  
  Flavien Armangeon, Thibaud Ehret, Enric Meinhardt-Llopis, Rafael Grompone von Gioi, Guillaume Thibault, Marc Petit, Gabriele Facciolo  
  _2026-02-17_ · https://arxiv.org/abs/2602.15584v1  
  <details><summary>Abstract</summary>

  Aligning functional schematics with 2D and 3D scene acquisitions is crucial for building digital twins, especially for old industrial facilities that lack native digital models. Current manual alignment using images and LiDAR data does not scale due to tediousness and complexity of industrial sites. Inconsistencies between schematics and reality, and the scarcity of public industrial datasets, make the problem both challenging and underexplored. This paper introduces IRIS-v2, a comprehensive dataset to support further research. It includes images, point clouds, 2D annotated boxes and segmentation masks, a CAD model, 3D pipe routing information, and the P&ID (Piping and Instrumentation Diagram). The alignment is experimented on a practical case study, aiming at reducing the time required for this task by combining segmentation and graph matching.

  </details>



- **Time-Archival Camera Virtualization for Sports and Visual Performances**  
  Yunxiao Zhang, William Stone, Suryansh Kumar  
  _2026-02-16_ · https://arxiv.org/abs/2602.15181v1  
  <details><summary>Abstract</summary>

  Camera virtualization -- an emerging solution to novel view synthesis -- holds transformative potential for visual entertainment, live performances, and sports broadcasting by enabling the generation of photorealistic images from novel viewpoints using images from a limited set of calibrated multiple static physical cameras. Despite recent advances, achieving spatially and temporally coherent and photorealistic rendering of dynamic scenes with efficient time-archival capabilities, particularly in fast-paced sports and stage performances, remains challenging for existing approaches. Recent methods based on 3D Gaussian Splatting (3DGS) for dynamic scenes could offer real-time view-synthesis results. Yet, they are hindered by their dependence on accurate 3D point clouds from the structure-from-motion method and their inability to handle large, non-rigid, rapid motions of different subjects (e.g., flips, jumps, articulations, sudden player-to-player transitions). Moreover, independent motions of multiple subjects can break the Gaussian-tracking assumptions commonly used in 4DGS, ST-GS, and other dynamic splatting variants. This paper advocates reconsidering a neural volume rendering formulation for camera virtualization and efficient time-archival capabilities, making it useful for sports broadcasting and related applications. By modeling a dynamic scene as rigid transformations across multiple synchronized camera views at a given time, our method performs neural representation learning, providing enhanced visual rendering quality at test time. A key contribution of our approach is its support for time-archival, i.e., users can revisit any past temporal instance of a dynamic scene and can perform novel view synthesis, enabling retrospective rendering for replay, analysis, and archival of live events, a functionality absent in existing neural rendering approaches and novel view synthesis...

  </details>



- **AnchorWeave: World-Consistent Video Generation with Retrieved Local Spatial Memories**  
  Zun Wang, Han Lin, Jaehong Yoon, Jaemin Cho, Yue Zhang, Mohit Bansal  
  _2026-02-16_ · https://arxiv.org/abs/2602.14941v1  
  <details><summary>Abstract</summary>

  Maintaining spatial world consistency over long horizons remains a central challenge for camera-controllable video generation. Existing memory-based approaches often condition generation on globally reconstructed 3D scenes by rendering anchor videos from the reconstructed geometry in the history. However, reconstructing a global 3D scene from multiple views inevitably introduces cross-view misalignment, as pose and depth estimation errors cause the same surfaces to be reconstructed at slightly different 3D locations across views. When fused, these inconsistencies accumulate into noisy geometry that contaminates the conditioning signals and degrades generation quality. We introduce AnchorWeave, a memory-augmented video generation framework that replaces a single misaligned global memory with multiple clean local geometric memories and learns to reconcile their cross-view inconsistencies. To this end, AnchorWeave performs coverage-driven local memory retrieval aligned with the target trajectory and integrates the selected local memories through a multi-anchor weaving controller during generation. Extensive experiments demonstrate that AnchorWeave significantly improves long-term scene consistency while maintaining strong visual quality, with ablation and analysis studies further validating the effectiveness of local geometric conditioning, multi-anchor control, and coverage-driven retrieval.

  </details>



- **Wrivinder: Towards Spatial Intelligence for Geo-locating Ground Images onto Satellite Imagery**  
  Chandrakanth Gudavalli, Tajuddin Manhar Mohammed, Abhay Yadav, Ananth Vishnu Bhaskar, Hardik Prajapati, Cheng Peng, Rama Chellappa, Shivkumar Chandrasekaran, B. S. Manjunath  
  _2026-02-16_ · https://arxiv.org/abs/2602.14929v1  
  <details><summary>Abstract</summary>

  Aligning ground-level imagery with geo-registered satellite maps is crucial for mapping, navigation, and situational awareness, yet remains challenging under large viewpoint gaps or when GPS is unreliable. We introduce Wrivinder, a zero-shot, geometry-driven framework that aggregates multiple ground photographs to reconstruct a consistent 3D scene and align it with overhead satellite imagery. Wrivinder combines SfM reconstruction, 3D Gaussian Splatting, semantic grounding, and monocular depth--based metric cues to produce a stable zenith-view rendering that can be directly matched to satellite context for metrically accurate camera geo-localization. To support systematic evaluation of this task, which lacks suitable benchmarks, we also release MC-Sat, a curated dataset linking multi-view ground imagery with geo-registered satellite tiles across diverse outdoor environments. Together, Wrivinder and MC-Sat provide a first comprehensive baseline and testbed for studying geometry-centered cross-view alignment without paired supervision. In zero-shot experiments, Wrivinder achieves sub-30\,m geolocation accuracy across both dense and large-area scenes, highlighting the promise of geometry-based aggregation for robust ground-to-satellite localization.

  </details>



- **Cross-view Domain Generalization via Geometric Consistency for LiDAR Semantic Segmentation**  
  Jindong Zhao, Yuan Gao, Yang Xia, Sheng Nie, Jun Yue, Weiwei Sun, Shaobo Xia  
  _2026-02-16_ · https://arxiv.org/abs/2602.14525v1  
  <details><summary>Abstract</summary>

  Domain-generalized LiDAR semantic segmentation (LSS) seeks to train models on source-domain point clouds that generalize reliably to multiple unseen target domains, which is essential for real-world LiDAR applications. However, existing approaches assume similar acquisition views (e.g., vehicle-mounted) and struggle in cross-view scenarios, where observations differ substantially due to viewpoint-dependent structural incompleteness and non-uniform point density. Accordingly, we formulate cross-view domain generalization for LiDAR semantic segmentation and propose a novel framework, termed CVGC (Cross-View Geometric Consistency). Specifically, we introduce a cross-view geometric augmentation module that models viewpoint-induced variations in visibility and sampling density, generating multiple cross-view observations of the same scene. Subsequently, a geometric consistency module enforces consistent semantic and occupancy predictions across geometrically augmented point clouds of the same scene. Extensive experiments on six public LiDAR datasets establish the first systematic evaluation of cross-view domain generalization for LiDAR semantic segmentation, demonstrating that CVGC consistently outperforms state-of-the-art methods when generalizing from a single source domain to multiple target domains with heterogeneous acquisition viewpoints. The source code will be publicly available at https://github.com/KintomZi/CVGC-DG

  </details>


