# Gaussian Splatting & 3DGS

_Updated: 2026-02-19 07:14 UTC_

Total papers shown: **4**


---

- **Semantic-Guided 3D Gaussian Splatting for Transient Object Removal**  
  Aditi Prabakaran, Priyesh Shukla  
  _2026-02-17_ · https://arxiv.org/abs/2602.15516v1  
  <details><summary>Abstract</summary>

  Transient objects in casual multi-view captures cause ghosting artifacts in 3D Gaussian Splatting (3DGS) reconstruction. Existing solutions relied on scene decomposition at significant memory cost or on motion-based heuristics that were vulnerable to parallax ambiguity. A semantic filtering framework was proposed for category-aware transient removal using vision-language models. CLIP similarity scores between rendered views and distractor text prompts were accumulated per-Gaussian across training iterations. Gaussians exceeding a calibrated threshold underwent opacity regularization and periodic pruning. Unlike motion-based approaches, semantic classification resolved parallax ambiguity by identifying object categories independently of motion patterns. Experiments on the RobustNeRF benchmark demonstrated consistent improvement in reconstruction quality over vanilla 3DGS across four sequences, while maintaining minimal memory overhead and real-time rendering performance. Threshold calibration and comparisons with baselines validated semantic guidance as a practical strategy for transient removal in scenarios with predictable distractor categories.

  </details>



- **DAV-GSWT: Diffusion-Active-View Sampling for Data-Efficient Gaussian Splatting Wang Tiles**  
  Rong Fu, Jiekai Wu, Haiyun Wei, Yee Tan Jia, Wenxin Zhang, Yang Li, Xiaowen Ma, Wangyu Wu, Simon Fong  
  _2026-02-17_ · https://arxiv.org/abs/2602.15355v1  
  <details><summary>Abstract</summary>

  The emergence of 3D Gaussian Splatting has fundamentally redefined the capabilities of photorealistic neural rendering by enabling high-throughput synthesis of complex environments. While procedural methods like Wang Tiles have recently been integrated to facilitate the generation of expansive landscapes, these systems typically remain constrained by a reliance on densely sampled exemplar reconstructions. We present DAV-GSWT, a data-efficient framework that leverages diffusion priors and active view sampling to synthesize high-fidelity Gaussian Splatting Wang Tiles from minimal input observations. By integrating a hierarchical uncertainty quantification mechanism with generative diffusion models, our approach autonomously identifies the most informative viewpoints while hallucinating missing structural details to ensure seamless tile transitions. Experimental results indicate that our system significantly reduces the required data volume while maintaining the visual integrity and interactive performance necessary for large-scale virtual environments.

  </details>



- **Time-Archival Camera Virtualization for Sports and Visual Performances**  
  Yunxiao Zhang, William Stone, Suryansh Kumar  
  _2026-02-16_ · https://arxiv.org/abs/2602.15181v1  
  <details><summary>Abstract</summary>

  Camera virtualization -- an emerging solution to novel view synthesis -- holds transformative potential for visual entertainment, live performances, and sports broadcasting by enabling the generation of photorealistic images from novel viewpoints using images from a limited set of calibrated multiple static physical cameras. Despite recent advances, achieving spatially and temporally coherent and photorealistic rendering of dynamic scenes with efficient time-archival capabilities, particularly in fast-paced sports and stage performances, remains challenging for existing approaches. Recent methods based on 3D Gaussian Splatting (3DGS) for dynamic scenes could offer real-time view-synthesis results. Yet, they are hindered by their dependence on accurate 3D point clouds from the structure-from-motion method and their inability to handle large, non-rigid, rapid motions of different subjects (e.g., flips, jumps, articulations, sudden player-to-player transitions). Moreover, independent motions of multiple subjects can break the Gaussian-tracking assumptions commonly used in 4DGS, ST-GS, and other dynamic splatting variants. This paper advocates reconsidering a neural volume rendering formulation for camera virtualization and efficient time-archival capabilities, making it useful for sports broadcasting and related applications. By modeling a dynamic scene as rigid transformations across multiple synchronized camera views at a given time, our method performs neural representation learning, providing enhanced visual rendering quality at test time. A key contribution of our approach is its support for time-archival, i.e., users can revisit any past temporal instance of a dynamic scene and can perform novel view synthesis, enabling retrospective rendering for replay, analysis, and archival of live events, a functionality absent in existing neural rendering approaches and novel view synthesis...

  </details>



- **Wrivinder: Towards Spatial Intelligence for Geo-locating Ground Images onto Satellite Imagery**  
  Chandrakanth Gudavalli, Tajuddin Manhar Mohammed, Abhay Yadav, Ananth Vishnu Bhaskar, Hardik Prajapati, Cheng Peng, Rama Chellappa, Shivkumar Chandrasekaran, B. S. Manjunath  
  _2026-02-16_ · https://arxiv.org/abs/2602.14929v1  
  <details><summary>Abstract</summary>

  Aligning ground-level imagery with geo-registered satellite maps is crucial for mapping, navigation, and situational awareness, yet remains challenging under large viewpoint gaps or when GPS is unreliable. We introduce Wrivinder, a zero-shot, geometry-driven framework that aggregates multiple ground photographs to reconstruct a consistent 3D scene and align it with overhead satellite imagery. Wrivinder combines SfM reconstruction, 3D Gaussian Splatting, semantic grounding, and monocular depth--based metric cues to produce a stable zenith-view rendering that can be directly matched to satellite context for metrically accurate camera geo-localization. To support systematic evaluation of this task, which lacks suitable benchmarks, we also release MC-Sat, a curated dataset linking multi-view ground imagery with geo-registered satellite tiles across diverse outdoor environments. Together, Wrivinder and MC-Sat provide a first comprehensive baseline and testbed for studying geometry-centered cross-view alignment without paired supervision. In zero-shot experiments, Wrivinder achieves sub-30\,m geolocation accuracy across both dense and large-area scenes, highlighting the promise of geometry-based aggregation for robust ground-to-satellite localization.

  </details>


