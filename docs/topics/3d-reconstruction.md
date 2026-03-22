# 3D Reconstruction

_Updated: 2026-03-22 07:06 UTC_

Total papers shown: **8**


---

- **MonoArt: Progressive Structural Reasoning for Monocular Articulated 3D Reconstruction**  
  Haitian Li, Haozhe Xie, Junxiang Xu, Beichen Wen, Fangzhou Hong, Ziwei Liu  
  _2026-03-19_ · https://arxiv.org/abs/2603.19231v1  
  <details><summary>Abstract</summary>

  Reconstructing articulated 3D objects from a single image requires jointly inferring object geometry, part structure, and motion parameters from limited visual evidence. A key difficulty lies in the entanglement between motion cues and object structure, which makes direct articulation regression unstable. Existing methods address this challenge through multi-view supervision, retrieval-based assembly, or auxiliary video generation, often sacrificing scalability or efficiency. We present MonoArt, a unified framework grounded in progressive structural reasoning. Rather than predicting articulation directly from image features, MonoArt progressively transforms visual observations into canonical geometry, structured part representations, and motion-aware embeddings within a single architecture. This structured reasoning process enables stable and interpretable articulation inference without external motion templates or multi-stage pipelines. Extensive experiments on PartNet-Mobility demonstrate that OM achieves state-of-the-art performance in both reconstruction accuracy and inference speed. The framework further generalizes to robotic manipulation and articulated scene reconstruction.

  </details>



- **Reconstruction Matters: Learning Geometry-Aligned BEV Representation through 3D Gaussian Splatting**  
  Yiren Lu, Xin Ye, Burhaneddin Yaman, Jingru Luo, Zhexiao Xiong, Liu Ren, Yu Yin  
  _2026-03-19_ · https://arxiv.org/abs/2603.19193v1  
  <details><summary>Abstract</summary>

  Bird's-Eye-View (BEV) perception serves as a cornerstone for autonomous driving, offering a unified spatial representation that fuses surrounding-view images to enable reasoning for various downstream tasks, such as semantic segmentation, 3D object detection, and motion prediction. However, most existing BEV perception frameworks adopt an end-to-end training paradigm, where image features are directly transformed into the BEV space and optimized solely through downstream task supervision. This formulation treats the entire perception process as a black box, often lacking explicit 3D geometric understanding and interpretability, leading to suboptimal performance. In this paper, we claim that an explicit 3D representation matters for accurate BEV perception, and we propose Splat2BEV, a Gaussian Splatting-assisted framework for BEV tasks. Splat2BEV aims to learn BEV feature representations that are both semantically rich and geometrically precise. We first pre-train a Gaussian generator that explicitly reconstructs 3D scenes from multi-view inputs, enabling the generation of geometry-aligned feature representations. These representations are then projected into the BEV space to serve as inputs for downstream tasks. Extensive experiments on nuScenes and argoverse dataset demonstrate that Splat2BEV achieves state-of-the-art performance and validate the effectiveness of incorporating explicit 3D reconstruction into BEV perception.

  </details>



- **Generalized Hand-Object Pose Estimation with Occlusion Awareness**  
  Hui Yang, Wei Sun, Jian Liu, Jian Xiao Tao Xie, Hossein Rahmani, Ajmal Saeed mian, Nicu Sebe, Gim Hee Lee  
  _2026-03-19_ · https://arxiv.org/abs/2603.19013v1  
  <details><summary>Abstract</summary>

  Generalized 3D hand-object pose estimation from a single RGB image remains challenging due to the large variations in object appearances and interaction patterns, especially under heavy occlusion. We propose GenHOI, a framework for generalized hand-object pose estimation with occlusion awareness. GenHOI integrates hierarchical semantic knowledge with hand priors to enhance model generalization under challenging occlusion conditions. Specifically, we introduce a hierarchical semantic prompt that encodes object states, hand configurations, and interaction patterns via textual descriptions. This enables the model to learn abstract high-level representations of hand-object interactions for generalization to unseen objects and novel interactions while compensating for missing or ambiguous visual cues. To enable robust occlusion reasoning, we adopt a multi-modal masked modeling strategy over RGB images, predicted point clouds, and textual descriptions. Moreover, we leverage hand priors as stable spatial references to extract implicit interaction constraints. This allows reliable pose inference even under significant variations in object shapes and interaction patterns. Extensive experiments on the challenging DexYCB and HO3Dv2 benchmarks demonstrate that our method achieves state-of-the-art performance in hand-object pose estimation.

  </details>



- **VGGT-360: Geometry-Consistent Zero-Shot Panoramic Depth Estimation**  
  Jiayi Yuan, Haobo Jiang, De Wen Soh, Na Zhao  
  _2026-03-19_ · https://arxiv.org/abs/2603.18943v1  
  <details><summary>Abstract</summary>

  This paper presents VGGT-360, a novel training-free framework for zero-shot, geometry-consistent panoramic depth estimation. Unlike prior view-independent training-free approaches, VGGT-360 reformulates the task as panoramic reprojection over multi-view reconstructed 3D models by leveraging the intrinsic 3D consistency of VGGT-like foundation models, thereby unifying fragmented per-view reasoning into a coherent panoramic understanding. To achieve robust and accurate estimation, VGGT-360 integrates three plug-and-play modules that form a unified panorama-to-3D-to-depth framework: (i) Uncertainty-guided adaptive projection slices panoramas into perspective views to bridge the domain gap between panoramic inputs and VGGT's perspective prior. It estimates gradient-based uncertainty to allocate denser views to geometry-poor regions, yielding geometry-informative inputs for VGGT. (ii) Structure-saliency enhanced attention strengthens VGGT's robustness during 3D reconstruction by injecting structure-aware confidence into its attention layers, guiding focus toward geometrically reliable regions and enhancing cross-view coherence. (iii) Correlation-weighted 3D model correction refines the reconstructed 3D model by reweighting overlapping points using attention-inferred correlation scores, providing a consistent geometric basis for accurate panoramic reprojection. Extensive experiments show that VGGT-360 outperforms both trained and training-free state-of-the-art methods across multiple resolutions and diverse indoor and outdoor datasets.

  </details>



- **GHOST: Fast Category-agnostic Hand-Object Interaction Reconstruction from RGB Videos using Gaussian Splatting**  
  Ahmed Tawfik Aboukhadra, Marcel Rogge, Nadia Robertini, Abdalla Arafa, Jameel Malik, Ahmed Elhayek, Didier Stricker  
  _2026-03-19_ · https://arxiv.org/abs/2603.18912v1  
  <details><summary>Abstract</summary>

  Understanding realistic hand-object interactions from monocular RGB videos is essential for AR/VR, robotics, and embodied AI. Existing methods rely on category-specific templates or heavy computation, yet still produce physically inconsistent hand-object alignment in 3D. We introduce GHOST (Gaussian Hand-Object Splatting), a fast, category-agnostic framework for reconstructing dynamic hand-object interactions using 2D Gaussian Splatting. GHOST represents both hands and objects as dense, view-consistent Gaussian discs and introduces three key innovations: (1) a geometric-prior retrieval and consistency loss that completes occluded object regions, (2) a grasp-aware alignment that refines hand translations and object scale to ensure realistic contact, and (3) a hand-aware background loss that prevents penalizing hand-occluded object regions. GHOST achieves complete, physically consistent, and animatable reconstructions from a single RGB video while running an order of magnitude faster than prior category-agnostic methods. Extensive experiments on ARCTIC, HO3D, and in-the-wild datasets demonstrate state-of-the-art accuracy in 3D reconstruction and 2D rendering quality, establishing GHOST as an efficient and robust solution for realistic hand-object interaction modeling. Code is available at https://github.com/ATAboukhadra/GHOST.

  </details>



- **Points-to-3D: Structure-Aware 3D Generation with Point Cloud Priors**  
  Jiatong Xia, Zicheng Duan, Anton van den Hengel, Lingqiao Liu  
  _2026-03-19_ · https://arxiv.org/abs/2603.18782v1  
  <details><summary>Abstract</summary>

  Recent progress in 3D generation has been driven largely by models conditioned on images or text, while readily available 3D priors are still underused. In many real-world scenarios, the visible-region point cloud are easy to obtain from active sensors such as LiDAR or from feed-forward predictors like VGGT, offering explicit geometric constraints that current methods fail to exploit. In this work, we introduce Points-to-3D, a diffusion-based framework that leverages point cloud priors for geometry-controllable 3D asset and scene generation. Built on a latent 3D diffusion model TRELLIS, Points-to-3D first replaces pure-noise sparse structure latent initialization with a point cloud priors tailored input formulation.A structure inpainting network, trained within the TRELLIS framework on task-specific data designed to learn global structural inpainting, is then used for inference with a staged sampling strategy (structural inpainting followed by boundary refinement), completing the global geometry while preserving the visible regions of the input priors.In practice, Points-to-3D can take either accurate point-cloud priors or VGGT-estimated point clouds from single images as input. Experiments on both objects and scene scenarios consistently demonstrate superior performance over state-of-the-art baselines in terms of rendering quality and geometric fidelity, highlighting the effectiveness of explicitly embedding point-cloud priors for achieving more accurate and structurally controllable 3D generation.

  </details>



- **SEAR: Simple and Efficient Adaptation of Visual Geometric Transformers for RGB+Thermal 3D Reconstruction**  
  Vsevolod Skorokhodov, Chenghao Xu, Shuo Sun, Olga Fink, Malcolm Mielle  
  _2026-03-19_ · https://arxiv.org/abs/2603.18774v1  
  <details><summary>Abstract</summary>

  Foundational feed-forward visual geometry models enable accurate and efficient camera pose estimation and scene reconstruction by learning strong scene priors from massive RGB datasets. However, their effectiveness drops when applied to mixed sensing modalities, such as RGB-thermal (RGB-T) images. We observe that while a visual geometry grounded transformer pretrained on RGB data generalizes well to thermal-only reconstruction, it struggles to align RGB and thermal modalities when processed jointly. To address this, we propose SEAR, a simple yet efficient fine-tuning strategy that adapts a pretrained geometry transformer to multimodal RGB-T inputs. Despite being trained on a relatively small RGB-T dataset, our approach significantly outperforms state-of-the-art methods for 3D reconstruction and camera pose estimation, achieving significant improvements over all metrics (e.g., over 29\% in AUC@30) and delivering higher detail and consistency between modalities with negligible overhead in inference time compared to the original pretrained model. Notably, SEAR enables reliable multimodal pose estimation and reconstruction even under challenging conditions, such as low lighting and dense smoke. We validate our architecture through extensive ablation studies, demonstrating how the model aligns both modalities. Additionally, we introduce a new dataset featuring RGB and thermal sequences captured at different times, viewpoints, and illumination conditions, providing a robust benchmark for future work in multimodal 3D scene reconstruction. Code and models are publicly available at https://www.github.com/Schindler-EPFL-Lab/SEAR.

  </details>



- **SwiftGS: Episodic Priors for Immediate Satellite Surface Recovery**  
  Rong Fu, Jiekai Wu, Haiyun Wei, Xiaowen Ma, Shiyin Lin, Kangan Qian, Chuang Liu, Jianyuan Ni, Simon James Fong  
  _2026-03-19_ · https://arxiv.org/abs/2603.18634v1  
  <details><summary>Abstract</summary>

  Rapid, large-scale 3D reconstruction from multi-date satellite imagery is vital for environmental monitoring, urban planning, and disaster response, yet remains difficult due to illumination changes, sensor heterogeneity, and the cost of per-scene optimization. We introduce SwiftGS, a meta-learned system that reconstructs 3D surfaces in a single forward pass by predicting geometry-radiation-decoupled Gaussian primitives together with a lightweight SDF, replacing expensive per-scene fitting with episodic training that captures transferable priors. The model couples a differentiable physics graph for projection, illumination, and sensor response with spatial gating that blends sparse Gaussian detail and global SDF structure, and incorporates semantic-geometric fusion, conditional lightweight task heads, and multi-view supervision from a frozen geometric teacher under an uncertainty-aware multi-task loss. At inference, SwiftGS operates zero-shot with optional compact calibration and achieves accurate DSM reconstruction and view-consistent rendering at significantly reduced computational cost, with ablations highlighting the benefits of the hybrid representation, physics-aware rendering, and episodic meta-training.

  </details>


