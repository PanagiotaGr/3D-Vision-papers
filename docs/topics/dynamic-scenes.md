# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-04-12 07:42 UTC_

Total papers shown: **7**


---

- **Self-Improving 4D Perception via Self-Distillation**  
  Nan Huang, Pengcheng Yu, Weijia Zeng, James M. Rehg, Angjoo Kanazawa, Haiwen Feng, Qianqian Wang  
  _2026-04-09_ · https://arxiv.org/abs/2604.08532v1  
  <details><summary>Abstract</summary>

  Large-scale multi-view reconstruction models have made remarkable progress, but most existing approaches still rely on fully supervised training with ground-truth 3D/4D annotations. Such annotations are expensive and particularly scarce for dynamic scenes, limiting scalability. We propose SelfEvo, a self-improving framework that continually improves pretrained multi-view reconstruction models using unlabeled videos. SelfEvo introduces a self-distillation scheme using spatiotemporal context asymmetry, enabling self-improvement for learning-based 4D perception without external annotations. We systematically study design choices that make self-improvement effective, including loss signals, forms of asymmetry, and other training strategies. Across eight benchmarks spanning diverse datasets and domains, SelfEvo consistently improves pretrained baselines and generalizes across base models (e.g. VGGT and $π^3$), with significant gains on dynamic scenes. Overall, SelfEvo achieves up to 36.5% relative improvement in video depth estimation and 20.1% in camera estimation, without using any labeled data. Project Page: https://self-evo.github.io/.

  </details>



- **AdaSpark: Adaptive Sparsity for Efficient Long-Video Understanding**  
  Handong Li, Zikang Liu, Longteng Guo, Tongtian Yue, Yepeng Tang, Xinxin Zhu, Chuanyang Zheng, Ziming Wang, Zhibin Wang, Jun Song, et al.  
  _2026-04-09_ · https://arxiv.org/abs/2604.08077v1  
  <details><summary>Abstract</summary>

  Processing long-form videos with Video Large Language Models (Video-LLMs) is computationally prohibitive. Current efficiency methods often compromise fine-grained perception through irreversible information disposal or inhibit long-range temporal modeling via rigid, predefined sparse patterns. This paper introduces AdaSpark, an adaptive sparsity framework designed to address these limitations. AdaSpark first partitions video inputs into 3D spatio-temporal cubes. It then employs two co-designed, context-aware components: (1) Adaptive Cube-Selective Attention (AdaS-Attn), which adaptively selects a subset of relevant video cubes to attend for each query token, and (2) Adaptive Token-Selective FFN (AdaS-FFN), which selectively processes only the most salient tokens within each cube. An entropy-based (Top-p) selection mechanism adaptively allocates computational resources based on input complexity. Experiments demonstrate that AdaSpark significantly reduces computational load by up to 57% FLOPs while maintaining comparable performance to dense models and preserving fine-grained, long-range dependencies, as validated on challenging hour-scale video benchmarks.

  </details>



- **Bridging Time and Space: Decoupled Spatio-Temporal Alignment for Video Grounding**  
  Xuezhen Tu, Jingyu Wu, Fangyu Kang, Qingpeng Nong, Kaijin Zhang, Chaoyue Niu, Fan Wu  
  _2026-04-09_ · https://arxiv.org/abs/2604.08014v1  
  <details><summary>Abstract</summary>

  Spatio-Temporal Video Grounding requires jointly localizing target objects across both temporal and spatial dimensions based on natural language queries, posing fundamental challenges for existing Multimodal Large Language Models (MLLMs). We identify two core challenges: \textit{entangled spatio-temporal alignment}, arising from coupling two heterogeneous sub-tasks within the same autoregressive output space, and \textit{dual-domain visual token redundancy}, where target objects exhibit simultaneous temporal and spatial sparsity, rendering the overwhelming majority of visual tokens irrelevant to the grounding query. To address these, we propose \textbf{Bridge-STG}, an end-to-end framework that decouples temporal and spatial localization while maintaining semantic coherence. While decoupling is the natural solution to this entanglement, it risks creating a semantic gap between the temporal MLLM and the spatial decoder. Bridge-STG resolves this through two pivotal designs: the \textbf{Spatio-Temporal Semantic Bridging (STSB)} mechanism with Explicit Temporal Alignment (ETA) distills the MLLM's temporal reasoning context into enriched bridging queries as a robust semantic interface; and the \textbf{Query-Guided Spatial Localization (QGSL)} module leverages these queries to drive a purpose-built spatial decoder with multi-layer interactive queries and positive/negative frame sampling, jointly eliminating dual-domain visual token redundancy. Extensive experiments across multiple benchmarks demonstrate that Bridge-STG achieves state-of-the-art performance among MLLM-based methods. Bridge-STG improves average m\_vIoU from $26.4$ to $34.3$ on VidSTG and demonstrates strong cross-task transfer across various fine-grained video understanding tasks under a unified multi-task training regime.

  </details>



- **SceneScribe-1M: A Large-Scale Video Dataset with Comprehensive Geometric and Semantic Annotations**  
  Yunnan Wang, Kecheng Zheng, Jianyuan Wang, Minghao Chen, David Novotny, Christian Rupprecht, Yinghao Xu, Xing Zhu, Wenjun Zeng, Xin Jin, et al.  
  _2026-04-09_ · https://arxiv.org/abs/2604.07990v1  
  <details><summary>Abstract</summary>

  The convergence of 3D geometric perception and video synthesis has created an unprecedented demand for large-scale video data that is rich in both semantic and spatio-temporal information. While existing datasets have advanced either 3D understanding or video generation, a significant gap remains in providing a unified resource that supports both domains at scale. To bridge this chasm, we introduce SceneScribe-1M, a new large-scale, multi-modal video dataset. It comprises one million in-the-wild videos, each meticulously annotated with detailed textual descriptions, precise camera parameters, dense depth maps, and consistent 3D point tracks. We demonstrate the versatility and value of SceneScribe-1M by establishing benchmarks across a wide array of downstream tasks, including monocular depth estimation, scene reconstruction, and dynamic point tracking, as well as generative tasks such as text-to-video synthesis, with or without camera control. By open-sourcing SceneScribe-1M, we aim to provide a comprehensive benchmark and a catalyst for research, fostering the development of models that can both perceive the dynamic 3D world and generate controllable, realistic video content.

  </details>



- **DP-DeGauss: Dynamic Probabilistic Gaussian Decomposition for Egocentric 4D Scene Reconstruction**  
  Tingxi Chen, Zhengxue Cheng, Houqiang Zhong, Su Wang, Rong Xie, Li Song  
  _2026-04-09_ · https://arxiv.org/abs/2604.07986v1  
  <details><summary>Abstract</summary>

  Egocentric video is crucial for next-generation 4D scene reconstruction, with applications in AR/VR and embodied AI. However, reconstructing dynamic first-person scenes is challenging due to complex ego-motion, occlusions, and hand-object interactions. Existing decomposition methods are ill-suited, assuming fixed viewpoints or merging dynamics into a single foreground. To address these limitations, we introduce DP-DeGauss, a dynamic probabilistic Gaussian decomposition framework for egocentric 4D reconstruction. Our method initializes a unified 3D Gaussian set from COLMAP priors, augments each with a learnable category probability, and dynamically routes them into specialized deformation branches for background, hands, or object modeling. We employ category-specific masks for better disentanglement and introduce brightness and motion-flow control to improve static rendering and dynamic reconstruction. Extensive experiments show that DP-DeGauss outperforms baselines by +1.70dB in PSNR on average with SSIM and LPIPS gains. More importantly, our framework achieves the first and state-of-the-art disentanglement of background, hand, and object components, enabling explicit, fine-grained separation, paving the way for more intuitive ego scene understanding and editing.

  </details>



- **Seeing enough: non-reference perceptual resolution selection for power-efficient client-side rendering**  
  Yaru Liu, Dayllon Vinícius Xavier Lemos, Ali Bozorgian, Chengxi Zeng, Alexander Hepburn, Arnau Raventos  
  _2026-04-09_ · https://arxiv.org/abs/2604.07959v1  
  <details><summary>Abstract</summary>

  Many client-side applications, especially games, render video at high resolution and frame rate on power-constrained devices, even when users perceive little or no benefit from all those extra pixels. Existing perceptual video quality metrics can indicate when a lower resolution is "good enough", but they are full-reference and computationally expensive, making them impractical for real-world applications and deployment on-device. In this work, we leverage the spatio-temporal limits of the human visual system and propose a non-reference method that predicts, from the rendered video alone, the lowest resolution that remains perceptually indistinguishable from the best available option, enabling power-efficient client-side rendering. Our approach is codec-agnostic and requires only minimal modifications to existing infrastructure. The network is trained on a large dataset of rendered content labeled with a full-reference perceptual video quality metric. The prediction significantly enhances perceptual quality while substantially reducing computational costs, suggesting a practical path toward perception-guided, power-efficient client-side rendering.

  </details>



- **Stitch4D: Sparse Multi-Location 4D Urban Reconstruction via Spatio-Temporal Interpolation**  
  Hina Kogure, Kei Katsumata, Taiki Miyanishi, Komei Sugiura  
  _2026-04-09_ · https://arxiv.org/abs/2604.07923v1  
  <details><summary>Abstract</summary>

  Dynamic urban environments are often captured by cameras placed at spatially separated locations with little or no view overlap. However, most existing 4D reconstruction methods assume densely overlapping views. When applied to such sparse observations, these methods fail to reconstruct intermediate regions and often introduce temporal artifacts. To address this practical yet underexplored sparse multi-location setting, we propose Stitch4D, a unified 4D reconstruction framework that explicitly compensates for missing spatial coverage in sparse observations. Stitch4D (i) synthesizes intermediate bridge views to densify spatial constraints and improve spatial coverage, and (ii) jointly optimizes real and synthesized observations within a unified coordinate frame under explicit inter-location consistency constraints. By restoring intermediate coverage before optimization, Stitch4D prevents geometric collapse and reconstructs coherent geometry and smooth scene dynamics even in sparsely observed environments. To evaluate this setting, we introduce Urban Sparse 4D (U-S4D), a CARLA-based benchmark designed to assess spatiotemporal alignment under sparse multi-location configurations. Experimental results on U-S4D show that Stitch4D surpasses representative 4D reconstruction baselines and achieves superior visual quality. These results indicate that recovering intermediate spatial coverage is essential for stable 4D reconstruction in sparse urban environments.

  </details>


