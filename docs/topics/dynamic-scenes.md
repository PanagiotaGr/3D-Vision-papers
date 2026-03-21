# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-03-21 07:01 UTC_

Total papers shown: **7**


---

- **MonoArt: Progressive Structural Reasoning for Monocular Articulated 3D Reconstruction**  
  Haitian Li, Haozhe Xie, Junxiang Xu, Beichen Wen, Fangzhou Hong, Ziwei Liu  
  _2026-03-19_ · https://arxiv.org/abs/2603.19231v1  
  <details><summary>Abstract</summary>

  Reconstructing articulated 3D objects from a single image requires jointly inferring object geometry, part structure, and motion parameters from limited visual evidence. A key difficulty lies in the entanglement between motion cues and object structure, which makes direct articulation regression unstable. Existing methods address this challenge through multi-view supervision, retrieval-based assembly, or auxiliary video generation, often sacrificing scalability or efficiency. We present MonoArt, a unified framework grounded in progressive structural reasoning. Rather than predicting articulation directly from image features, MonoArt progressively transforms visual observations into canonical geometry, structured part representations, and motion-aware embeddings within a single architecture. This structured reasoning process enables stable and interpretable articulation inference without external motion templates or multi-stage pipelines. Extensive experiments on PartNet-Mobility demonstrate that OM achieves state-of-the-art performance in both reconstruction accuracy and inference speed. The framework further generalizes to robotic manipulation and articulated scene reconstruction.

  </details>



- **Motion-o: Trajectory-Grounded Video Reasoning**  
  Bishoy Galoaa, Shayda Moezzi, Xiangyu Bai, Sarah Ostadabbas  
  _2026-03-19_ · https://arxiv.org/abs/2603.18856v1  
  <details><summary>Abstract</summary>

  Recent research has made substantial progress on video reasoning, with many models leveraging spatio-temporal evidence chains to strengthen their inference capabilities. At the same time, a growing set of datasets and benchmarks now provides structured annotations designed to support and evaluate such reasoning. However, little attention has been paid to reasoning about \emph{how} objects move between observations: no prior work has articulated the motion patterns by connecting successive observations, leaving trajectory understanding implicit and difficult to verify. We formalize this missing capability as Spatial-Temporal-Trajectory (STT) reasoning and introduce \textbf{Motion-o}, a motion-centric video understanding extension to visual language models that makes trajectories explicit and verifiable. To enable motion reasoning, we also introduce a trajectory-grounding dataset artifact that expands sparse keyframe supervision via augmentation to yield denser bounding box tracks and a stronger trajectory-level training signal. Finally, we introduce Motion Chain of Thought (MCoT), a structured reasoning pathway that makes object trajectories through discrete \texttt{<motion/>} tag summarizing per-object direction, speed, and scale (of velocity) change to explicitly connect grounded observations into trajectories. To train Motion-o, we design a reward function that compels the model to reason directly over visual evidence, all while requiring no architectural modifications. Empirical results demonstrate that Motion-o improves spatial-temporal grounding and trajectory prediction while remaining fully compatible with existing frameworks, establishing motion reasoning as a critical extension for evidence-based video understanding. Code is available at https://github.com/ostadabbas/Motion-o.

  </details>



- **PhysVideo: Physically Plausible Video Generation with Cross-View Geometry Guidance**  
  Cong Wang, Hanxin Zhu, Xiao Tang, Jiayi Luo, Xin Jin, Long Chen, Fei-Yue Wang, Zhibo Chen  
  _2026-03-19_ · https://arxiv.org/abs/2603.18639v1  
  <details><summary>Abstract</summary>

  Recent progress in video generation has led to substantial improvements in visual fidelity, yet ensuring physically consistent motion remains a fundamental challenge. Intuitively, this limitation can be attributed to the fact that real-world object motion unfolds in three-dimensional space, while video observations provide only partial, view-dependent projections of such dynamics. To address these issues, we propose PhysVideo, a two-stage framework that first generates physics-aware orthogonal foreground videos and then synthesizes full videos with background. In the first stage, Phys4View leverages physics-aware attention to capture the influence of physical attributes on motion dynamics, and enhances spatio-temporal consistency by incorporating geometry-enhanced cross-view attention and temporal attention. In the second stage, VideoSyn uses the generated foreground videos as guidance and learns the interactions between foreground dynamics and background context for controllable video synthesis. To support training, we construct PhysMV, a dataset containing 40K scenes, each consisting of four orthogonal viewpoints, resulting in a total of 160K video sequences. Extensive experiments demonstrate that PhysVideo significantly improves physical realism and spatial-temporal coherence over existing video generation methods. Home page: https://anonymous.4open.science/w/Phys4D/.

  </details>



- **Sparse3DTrack: Monocular 3D Object Tracking Using Sparse Supervision**  
  Nikhil Gosala, B. Ravi Kiran, Senthil Yogamani, Abhinav Valada  
  _2026-03-18_ · https://arxiv.org/abs/2603.18298v1  
  <details><summary>Abstract</summary>

  Monocular 3D object tracking aims to estimate temporally consistent 3D object poses across video frames, enabling autonomous agents to reason about scene dynamics. However, existing state-of-the-art approaches are fully supervised and rely on dense 3D annotations over long video sequences, which are expensive to obtain and difficult to scale. In this work, we address this fundamental limitation by proposing the first sparsely supervised framework for monocular 3D object tracking. Our approach decomposes the task into two sequential sub-problems: 2D query matching and 3D geometry estimation. Both components leverage the spatio-temporal consistency of image sequences to augment a sparse set of labeled samples and learn rich 2D and 3D representations of the scene. Leveraging these learned cues, our model automatically generates high-quality 3D pseudolabels across entire videos, effectively transforming sparse supervision into dense 3D track annotations. This enables existing fully-supervised trackers to effectively operate under extreme label sparsity. Extensive experiments on the KITTI and nuScenes datasets demonstrate that our method significantly improves tracking performance, achieving an improvement of up to 15.50 p.p. while using at most four ground truth annotations per track.

  </details>



- **Unified Spatio-Temporal Token Scoring for Efficient Video VLMs**  
  Jianrui Zhang, Yue Yang, Rohun Tripathi, Winson Han, Ranjay Krishna, Christopher Clark, Yong Jae Lee, Sangho Lee  
  _2026-03-18_ · https://arxiv.org/abs/2603.18004v1  
  <details><summary>Abstract</summary>

  Token pruning is essential for enhancing the computational efficiency of vision-language models (VLMs), particularly for video-based tasks where temporal redundancy is prevalent. Prior approaches typically prune tokens either (1) within the vision transformer (ViT) exclusively for unimodal perception tasks such as action recognition and object segmentation, without adapting to downstream vision-language tasks; or (2) only within the LLM while leaving the ViT output intact, often requiring complex text-conditioned token selection mechanisms. In this paper, we introduce Spatio-Temporal Token Scoring (STTS), a simple and lightweight module that prunes vision tokens across both the ViT and the LLM without text conditioning or token merging, and is fully compatible with end-to-end training. By learning how to score temporally via an auxiliary loss and spatially via LLM downstream gradients, aided by our efficient packing algorithm, STTS prunes 50% of vision tokens throughout the entire architecture, resulting in a 62% improvement in efficiency during both training and inference with only a 0.7% drop in average performance across 13 short and long video QA tasks. Efficiency gains increase with more sampled frames per video. Applying test-time scaling for long-video QA further yields performance gains of 0.5-1% compared to the baseline. Overall, STTS represents a novel, simple yet effective technique for unified, architecture-wide vision token pruning.

  </details>



- **Feeling the Space: Egomotion-Aware Video Representation for Efficient and Accurate 3D Scene Understanding**  
  Shuyao Shi, Kang G. Shin  
  _2026-03-18_ · https://arxiv.org/abs/2603.17980v1  
  <details><summary>Abstract</summary>

  Recent Multimodal Large Language Models (MLLMs) have shown high potential for spatial reasoning within 3D scenes. However, they typically rely on computationally expensive 3D representations like point clouds or reconstructed Bird's-Eye View (BEV) maps, or lack physical grounding to resolve ambiguities in scale and size. This paper significantly enhances MLLMs with egomotion modality data, captured by Inertial Measurement Units (IMUs) concurrently with the video. In particular, we propose a novel framework, called Motion-MLLM, introducing two key components: (1) a cascaded motion-visual keyframe filtering module that leverages both IMU data and visual features to efficiently select a sparse yet representative set of keyframes, and (2) an asymmetric cross-modal fusion module where motion tokens serve as intermediaries that channel egomotion cues and cross-frame visual context into the visual representation. By grounding visual content in physical egomotion trajectories, Motion-MLLM can reason about absolute scale and spatial relationships across the scene. Our extensive evaluation shows that Motion-MLLM makes significant improvements in various tasks related to 3D scene understanding and spatial reasoning. Compared to state-of-the-art (SOTA) methods based on video frames and explicit 3D data, Motion-MLLM exhibits similar or even higher accuracy with significantly less overhead (i.e., 1.40$\times$ and 1.63$\times$ higher cost-effectiveness, respectively).

  </details>



- **VectorWorld: Efficient Streaming World Model via Diffusion Flow on Vector Graphs**  
  Chaokang Jiang, Desen Zhou, Jiuming Liu, Kevin Li Sun  
  _2026-03-18_ · https://arxiv.org/abs/2603.17652v1  
  <details><summary>Abstract</summary>

  Closed-loop evaluation of autonomous-driving policies requires interactive simulation beyond log replay. However, existing generative world models often degrade in closed loop due to (i) history-free initialization that mismatches policy inputs, (ii) multi-step sampling latency that violates real-time budgets, and (iii) compounding kinematic infeasibility over long horizons. We propose VectorWorld, a streaming world model that incrementally generates ego-centric $64 \mathrm{m}\times 64\mathrm{m}$ lane--agent vector-graph tiles during rollout. VectorWorld aligns initialization with history-conditioned policies by producing a policy-compatible interaction state via a motion-aware gated VAE. It enables real-time outpainting via solver-free one-step masked completion with an edge-gated relational DiT trained with interval-conditioned MeanFlow and JVP-based large-step supervision. To stabilize long-horizon rollouts, we introduce $Δ$Sim, a physics-aligned non-ego (NPC) policy with hybrid discrete--continuous actions and differentiable kinematic logit shaping. On Waymo open motion and nuPlan, VectorWorld improves map-structure fidelity and initialization validity, and supports stable, real-time $1\mathrm{km}+$ closed-loop rollouts (\href{https://github.com/jiangchaokang/VectorWorld}{code}).

  </details>


