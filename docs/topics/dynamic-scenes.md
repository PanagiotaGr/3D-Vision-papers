# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-04-10 07:57 UTC_

Total papers shown: **13**


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



- **Fast Spatial Memory with Elastic Test-Time Training**  
  Ziqiao Ma, Xueyang Yu, Haoyu Zhen, Yuncong Yang, Joyce Chai, Chuang Gan  
  _2026-04-08_ · https://arxiv.org/abs/2604.07350v1  
  <details><summary>Abstract</summary>

  Large Chunk Test-Time Training (LaCT) has shown strong performance on long-context 3D reconstruction, but its fully plastic inference-time updates remain vulnerable to catastrophic forgetting and overfitting. As a result, LaCT is typically instantiated with a single large chunk spanning the full input sequence, falling short of the broader goal of handling arbitrarily long sequences in a single pass. We propose Elastic Test-Time Training inspired by elastic weight consolidation, that stabilizes LaCT fast-weight updates with a Fisher-weighted elastic prior around a maintained anchor state. The anchor evolves as an exponential moving average of past fast weights to balance stability and plasticity. Based on this updated architecture, we introduce Fast Spatial Memory (FSM), an efficient and scalable model for 4D reconstruction that learns spatiotemporal representations from long observation sequences and renders novel view-time combinations. We pre-trained FSM on large-scale curated 3D/4D data to capture the dynamics and semantics of complex spatial environments. Extensive experiments show that FSM supports fast adaptation over long sequences and delivers high-quality 3D/4D reconstruction with smaller chunks and mitigating the camera-interpolation shortcut. Overall, we hope to advance LaCT beyond the bounded single-chunk setting toward robust multi-chunk adaptation, a necessary step for generalization to genuinely longer sequences, while substantially alleviating the activation-memory bottleneck.

  </details>



- **Q-Zoom: Query-Aware Adaptive Perception for Efficient Multimodal Large Language Models**  
  Yuheng Shi, Xiaohuan Pei, Linfeng Wen, Minjing Dong, Chang Xu  
  _2026-04-08_ · https://arxiv.org/abs/2604.06912v1  
  <details><summary>Abstract</summary>

  MLLMs require high-resolution visual inputs for fine-grained tasks like document understanding and dense scene perception. However, current global resolution scaling paradigms indiscriminately flood the quadratic self-attention mechanism with visually redundant tokens, severely bottlenecking inference throughput while ignoring spatial sparsity and query intent. To overcome this, we propose Q-Zoom, a query-aware adaptive high-resolution perception framework that operates in an efficient coarse-to-fine manner. First, a lightweight Dynamic Gating Network safely bypasses high-resolution processing when coarse global features suffice. Second, for queries demanding fine-grained perception, a Self-Distilled Region Proposal Network (SD-RPN) precisely localizes the task-relevant Region-of-Interest (RoI) directly from intermediate feature spaces. To optimize these modules efficiently, the gating network uses a consistency-aware generation strategy to derive deterministic routing labels, while the SD-RPN employs a fully self-supervised distillation paradigm. A continuous spatio-temporal alignment scheme and targeted fine-tuning then seamlessly fuse the dense local RoI with the coarse global layout. Extensive experiments demonstrate that Q-Zoom establishes a dominant Pareto frontier. Using Qwen2.5-VL-7B as a primary testbed, Q-Zoom accelerates inference by 2.52 times on Document & OCR benchmarks and 4.39 times in High-Resolution scenarios while matching the baseline's peak accuracy. Furthermore, when configured for maximum perceptual fidelity, Q-Zoom surpasses the baseline's peak performance by 1.1% and 8.1% on these respective benchmarks. These robust improvements transfer seamlessly to Qwen3-VL, LLaVA, and emerging RL-based thinking-with-image models. Project page is available at https://yuhengsss.github.io/Q-Zoom/.

  </details>



- **SCT-MOT: Enhancing Air-to-Air Multiple UAVs Tracking with Swarm-Coupled Motion and Trajectory Guidance**  
  Zhaochen Chu, Tao Song, Ren Jin, Shaoming He, Defu Lin, Siqing Cheng  
  _2026-04-08_ · https://arxiv.org/abs/2604.06883v1  
  <details><summary>Abstract</summary>

  Air-to-air tracking of swarm UAVs presents significant challenges due to the complex nonlinear group motion and weak visual cues for small objects, which often cause detection failures, trajectory fragmentation, and identity switches. Although existing methods have attempted to improve performance by incorporating trajectory prediction, they model each object independently, neglecting the swarm-level motion dependencies. Their limited integration between motion prediction and appearance representation also weakens the spatio-temporal consistency required for tracking in visually ambiguous and cluttered environments, making it difficult to maintain coherent trajectories and reliable associations. To address these challenges, we propose SCT-MOT, a tracking framework that integrates Swarm-Coupled motion modeling and Trajectory-guided feature fusion. First, we develop a Swarm Motion-Aware Trajectory Prediction (SMTP) module jointly models historical trajectories and posture-aware appearance features from a swarm-level perspective, enabling more accurate forecasting of the nonlinear, coupled group trajectories. Second, we design a Trajectory-Guided Spatio-Temporal Feature Fusion (TG-STFF) module aligns predicted positions with historical visual cues and deeply integrates them with current frame features, enhancing temporal consistency and spatial discriminability for weak objects. Extensive experiments on three public air-to-air swarm UAV tracking datasets, including AIRMOT, MOT-FLY, and UAVSwarm, demonstrate that SMTP achieves more accurate trajectory forecasts and yields a 1.21\% IDF1 improvement over the state-of-the-art trajectory prediction module EqMotion when integrated into the same MOT framework. Overall, our SCT-MOT consistently achieves superior accuracy and robustness compared to state-of-the-art trackers across multiple metrics under complex swarm scenarios.

  </details>



- **LiveStre4m: Feed-Forward Live Streaming of Novel Views from Unposed Multi-View Video**  
  Pedro Quesado, Erkut Akdag, Yasaman Kashefbahrami, Willem Menu, Egor Bondarev  
  _2026-04-08_ · https://arxiv.org/abs/2604.06740v1  
  <details><summary>Abstract</summary>

  Live-streaming Novel View Synthesis (NVS) from unposed multi-view video remains an open challenge in a wide range of applications. Existing methods for dynamic scene representation typically require ground-truth camera parameters and involve lengthy optimizations ($\approx 2.67$s), which makes them unsuitable for live streaming scenarios. To address this issue, we propose a novel viewpoint video live-streaming method (LiveStre4m), a feed-forward model for real-time NVS from unposed sparse multi-view inputs. LiveStre4m introduces a multi-view vision transformer for keyframe 3D scene reconstruction coupled with a diffusion-transformer interpolation module that ensures temporal consistency and stable streaming. In addition, a Camera Pose Predictor module is proposed to efficiently estimate both poses and intrinsics directly from RGB images, removing the reliance on known camera calibration information. Our approach enables temporally consistent novel-view video streaming in real-time using as few as two synchronized unposed input streams. LiveStre4m attains an average reconstruction time of $ 0.07$s per-frame at $ 1024 \times 768$ resolution, outperforming the optimization-based dynamic scene representation methods by orders of magnitude in runtime. These results demonstrate that LiveStre4m makes real-time NVS streaming feasible in practical settings, marking a substantial step toward deployable live novel-view synthesis systems. Code available at: https://github.com/pedro-quesado/LiveStre4m

  </details>



- **DiffHDR: Re-Exposing LDR Videos with Video Diffusion Models**  
  Zhengming Yu, Li Ma, Mingming He, Leo Isikdogan, Yuancheng Xu, Dmitriy Smirnov, Pablo Salamanca, Dao Mi, Pablo Delgado, Ning Yu, et al.  
  _2026-04-07_ · https://arxiv.org/abs/2604.06161v1  
  <details><summary>Abstract</summary>

  Most digital videos are stored in 8-bit low dynamic range (LDR) formats, where much of the original high dynamic range (HDR) scene radiance is lost due to saturation and quantization. This loss of highlight and shadow detail precludes mapping accurate luminance to HDR displays and limits meaningful re-exposure in post-production workflows. Although techniques have been proposed to convert LDR images to HDR through dynamic range expansion, they struggle to restore realistic detail in the over- and underexposed regions. To address this, we present DiffHDR, a framework that formulates LDR-to-HDR conversion as a generative radiance inpainting task within the latent space of a video diffusion model. By operating in Log-Gamma color space, DiffHDR leverages spatio-temporal generative priors from a pretrained video diffusion model to synthesize plausible HDR radiance in over- and underexposed regions while recovering the continuous scene radiance of the quantized pixels. Our framework further enables controllable LDR-to-HDR video conversion guided by text prompts or reference images. To address the scarcity of paired HDR video data, we develop a pipeline that synthesizes high-quality HDR video training data from static HDRI maps. Extensive experiments demonstrate that DiffHDR significantly outperforms state-of-the-art approaches in radiance fidelity and temporal stability, producing realistic HDR videos with considerable latitude for re-exposure.

  </details>



- **Mixture-of-Modality-Experts with Holistic Token Learning for Fine-Grained Multimodal Visual Analytics in Driver Action Recognition**  
  Tianyi Liu, Yiming Li, Wenqian Wang, Jiaojiao Wang, Chen Cai, Yi Wang, Kim-Hui Yap  
  _2026-04-07_ · https://arxiv.org/abs/2604.05947v1  
  <details><summary>Abstract</summary>

  Robust multimodal visual analytics remains challenging when heterogeneous modalities provide complementary but input-dependent evidence for decision-making.Existing multimodal learning methods mainly rely on fixed fusion modules or predefined cross-modal interactions, which are often insufficient to adapt to changing modality reliability and to capture fine-grained action cues. To address this issue, we propose a Mixture-of-Modality-Experts (MoME) framework with a Holistic Token Learning (HTL) strategy. MoME enables adaptive collaboration among modality-specific experts, while HTL improves both intra-expert refinement and inter-expert knowledge transfer through class tokens and spatio-temporal tokens. In this way, our method forms a knowledge-centric multimodal learning framework that improves expert specialization while reducing ambiguity in multimodal fusion.We validate the proposed framework on driver action recognition as a representative multimodal understanding taskThe experimental results on the public benchmark show that the proposed MoME framework and the HTL strategy jointly outperform representative single-modal and multimodal baselines. Additional ablation, validation, and visualization results further verify that the proposed HTL strategy improves subtle multimodal understanding and offers better interpretability.

  </details>


