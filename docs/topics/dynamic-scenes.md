# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-03-01 07:02 UTC_

Total papers shown: **7**


---

- **Towards Long-Form Spatio-Temporal Video Grounding**  
  Xin Gu, Bing Fan, Jiali Yao, Zhipeng Zhang, Yan Huang, Cheng Han, Heng Fan, Libo Zhang  
  _2026-02-26_ · https://arxiv.org/abs/2602.23294v1  
  <details><summary>Abstract</summary>

  In real scenarios, videos can span several minutes or even hours. However, existing research on spatio-temporal video grounding (STVG), given a textual query, mainly focuses on localizing targets in short videos of tens of seconds, typically less than one minute, which limits real-world applications. In this paper, we explore Long-Form STVG (LF-STVG), which aims to locate targets in long-term videos. Compared with short videos, long-term videos contain much longer temporal spans and more irrelevant information, making it difficult for existing STVG methods that process all frames at once. To address this challenge, we propose an AutoRegressive Transformer architecture for LF-STVG, termed ART-STVG. Unlike conventional STVG methods that require the entire video sequence to make predictions at once, ART-STVG treats the video as streaming input and processes frames sequentially, enabling efficient handling of long videos. To model spatio-temporal context, we design spatial and temporal memory banks and apply them to the decoders. Since memories from different moments are not always relevant to the current frame, we introduce simple yet effective memory selection strategies to provide more relevant information to the decoders, significantly improving performance. Furthermore, instead of parallel spatial and temporal localization, we propose a cascaded spatio-temporal design that connects the spatial decoder to the temporal decoder, allowing fine-grained spatial cues to assist complex temporal localization in long videos. Experiments on newly extended LF-STVG datasets show that ART-STVG significantly outperforms state-of-the-art methods, while achieving competitive performance on conventional short-form STVG.

  </details>



- **Spatio-Temporal Token Pruning for Efficient High-Resolution GUI Agents**  
  Zhou Xu, Bowen Zhou, Qi Wang, Shuwen Feng, Jingyu Xiao  
  _2026-02-26_ · https://arxiv.org/abs/2602.23235v1  
  <details><summary>Abstract</summary>

  Pure-vision GUI agents provide universal interaction capabilities but suffer from severe efficiency bottlenecks due to the massive spatiotemporal redundancy inherent in high-resolution screenshots and historical trajectories. We identify two critical misalignments in existing compression paradigms: the temporal mismatch, where uniform history encoding diverges from the agent's "fading memory" attention pattern, and the spatial topology conflict, where unstructured pruning compromises the grid integrity required for precise coordinate grounding, inducing spatial hallucinations. To address these challenges, we introduce GUIPruner, a training-free framework tailored for high-resolution GUI navigation. It synergizes Temporal-Adaptive Resolution (TAR), which eliminates historical redundancy via decay-based resizing, and Stratified Structure-aware Pruning (SSP), which prioritizes interactive foregrounds and semantic anchors while safeguarding global layout. Extensive evaluations across diverse benchmarks demonstrate that GUIPruner consistently achieves state-of-the-art performance, effectively preventing the collapse observed in large-scale models under high compression. Notably, on Qwen2-VL-2B, our method delivers a 3.4x reduction in FLOPs and a 3.3x speedup in vision encoding latency while retaining over 94% of the original performance, enabling real-time, high-precision navigation with minimal resource consumption.

  </details>



- **Motion-aware Event Suppression for Event Cameras**  
  Roberto Pellerito, Nico Messikommer, Giovanni Cioffi, Marco Cannici, Davide Scaramuzza  
  _2026-02-26_ · https://arxiv.org/abs/2602.23204v1  
  <details><summary>Abstract</summary>

  In this work, we introduce the first framework for Motion-aware Event Suppression, which learns to filter events triggered by IMOs and ego-motion in real time. Our model jointly segments IMOs in the current event stream while predicting their future motion, enabling anticipatory suppression of dynamic events before they occur. Our lightweight architecture achieves 173 Hz inference on consumer-grade GPUs with less than 1 GB of memory usage, outperforming previous state-of-the-art methods on the challenging EVIMO benchmark by 67\% in segmentation accuracy while operating at a 53\% higher inference rate. Moreover, we demonstrate significant benefits for downstream applications: our method accelerates Vision Transformer inference by 83\% via token pruning and improves event-based visual odometry accuracy, reducing Absolute Trajectory Error (ATE) by 13\%.

  </details>



- **Uni-Animator: Towards Unified Visual Colorization**  
  Xinyuan Chen, Yao Xu, Shaowen Wang, Pengjie Song, Bowen Deng  
  _2026-02-26_ · https://arxiv.org/abs/2602.23191v1  
  <details><summary>Abstract</summary>

  We propose Uni-Animator, a novel Diffusion Transformer (DiT)-based framework for unified image and video sketch colorization. Existing sketch colorization methods struggle to unify image and video tasks, suffering from imprecise color transfer with single or multiple references, inadequate preservation of high-frequency physical details, and compromised temporal coherence with motion artifacts in large-motion scenes. To tackle imprecise color transfer, we introduce visual reference enhancement via instance patch embedding, enabling precise alignment and fusion of reference color information. To resolve insufficient physical detail preservation, we design physical detail reinforcement using physical features that effectively capture and retain high-frequency textures. To mitigate motion-induced temporal inconsistency, we propose sketch-based dynamic RoPE encoding that adaptively models motion-aware spatial-temporal dependencies. Extensive experimental results demonstrate that Uni-Animator achieves competitive performance on both image and video sketch colorization, matching that of task-specific methods while unlocking unified cross-domain capabilities with high detail fidelity and robust temporal consistency.

  </details>



- **DMAligner: Enhancing Image Alignment via Diffusion Model Based View Synthesis**  
  Xinglong Luo, Ao Luo, Zhengning Wang, Yueqi Yang, Chaoyu Feng, Lei Lei, Bing Zeng, Shuaicheng Liu  
  _2026-02-26_ · https://arxiv.org/abs/2602.23022v1  
  <details><summary>Abstract</summary>

  Image alignment is a fundamental task in computer vision with broad applications. Existing methods predominantly employ optical flow-based image warping. However, this technique is susceptible to common challenges such as occlusions and illumination variations, leading to degraded alignment visual quality and compromised accuracy in downstream tasks. In this paper, we present DMAligner, a diffusion-based framework for image alignment through alignment-oriented view synthesis. DMAligner is crafted to tackle the challenges in image alignment from a new perspective, employing a generation-based solution that showcases strong capabilities and avoids the problems associated with flow-based image warping. Specifically, we propose a Dynamics-aware Diffusion Training approach for learning conditional image generation, synthesizing a novel view for image alignment. This incorporates a Dynamics-aware Mask Producing (DMP) module to adaptively distinguish dynamic foreground regions from static backgrounds, enabling the diffusion model to more effectively handle challenges that classical methods struggle to solve. Furthermore, we develop the Dynamic Scene Image Alignment (DSIA) dataset using Blender, which includes 1,033 indoor and outdoor scenes with over 30K image pairs tailored for image alignment. Extensive experimental results demonstrate the superiority of the proposed approach on DSIA benchmarks, as well as on a series of widely-used video datasets for qualitative comparisons. Our code is available at https://github.com/boomluo02/DMAligner.

  </details>



- **OSDaR-AR: Enhancing Railway Perception Datasets via Multi-modal Augmented Reality**  
  Federico Nesti, Gianluca D'Amico, Mauro Marinoni, Giorgio Buttazzo  
  _2026-02-26_ · https://arxiv.org/abs/2602.22920v1  
  <details><summary>Abstract</summary>

  Although deep learning has significantly advanced the perception capabilities of intelligent transportation systems, railway applications continue to suffer from a scarcity of high-quality, annotated data for safety-critical tasks like obstacle detection. While photorealistic simulators offer a solution, they often struggle with the ``sim-to-real" gap; conversely, simple image-masking techniques lack the spatio-temporal coherence required to obtain augmented single- and multi-frame scenes with the correct appearance and dimensions. This paper introduces a multi-modal augmented reality framework designed to bridge this gap by integrating photorealistic virtual objects into real-world railway sequences from the OSDaR23 dataset. Utilizing Unreal Engine 5 features, our pipeline leverages LiDAR point-clouds and INS/GNSS data to ensure accurate object placement and temporal stability across RGB frames. This paper also proposes a segmentation-based refinement strategy for INS/GNSS data to significantly improve the realism of the augmented sequences, as confirmed by the comparative study presented in the paper. Carefully designed augmented sequences are collected to produce OSDaR-AR, a public dataset designed to support the development of next-generation railway perception systems. The dataset is available at the following page: https://syndra.retis.santannapisa.it/osdarar.html

  </details>



- **ProjFlow: Projection Sampling with Flow Matching for Zero-Shot Exact Spatial Motion Control**  
  Akihisa Watanabe, Qing Yu, Edgar Simo-Serra, Kent Fujiwara  
  _2026-02-26_ · https://arxiv.org/abs/2602.22742v1  
  <details><summary>Abstract</summary>

  Generating human motion with precise spatial control is a challenging problem. Existing approaches often require task-specific training or slow optimization, and enforcing hard constraints frequently disrupts motion naturalness. Building on the observation that many animation tasks can be formulated as a linear inverse problem, we introduce ProjFlow, a training-free sampler that achieves zero-shot, exact satisfaction of linear spatial constraints while preserving motion realism. Our key advance is a novel kinematics-aware metric that encodes skeletal topology. This metric allows the sampler to enforce hard constraints by distributing corrections coherently across the entire skeleton, avoiding the unnatural artifacts of naive projection. Furthermore, for sparse inputs, such as filling in long gaps between a few keyframes, we introduce a time-varying formulation using pseudo-observations that fade during sampling. Extensive experiments on representative applications, motion inpainting, and 2D-to-3D lifting, demonstrate that ProjFlow achieves exact constraint satisfaction and matches or improves realism over zero-shot baselines, while remaining competitive with training-based controllers.

  </details>


