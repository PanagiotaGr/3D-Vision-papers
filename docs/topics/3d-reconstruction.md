# 3D Reconstruction

_Updated: 2026-02-16 07:19 UTC_

Total papers shown: **5**


---

- **FlexAM: Flexible Appearance-Motion Decomposition for Versatile Video Generation Control**  
  Mingzhi Sheng, Zekai Gu, Peng Li, Cheng Lin, Hao-Xiang Guo, Ying-Cong Chen, Yuan Liu  
  _2026-02-13_ · https://arxiv.org/abs/2602.13185v1  
  <details><summary>Abstract</summary>

  Effective and generalizable control in video generation remains a significant challenge. While many methods rely on ambiguous or task-specific signals, we argue that a fundamental disentanglement of "appearance" and "motion" provides a more robust and scalable pathway. We propose FlexAM, a unified framework built upon a novel 3D control signal. This signal represents video dynamics as a point cloud, introducing three key enhancements: multi-frequency positional encoding to distinguish fine-grained motion, depth-aware positional encoding, and a flexible control signal for balancing precision and generative quality. This representation allows FlexAM to effectively disentangle appearance and motion, enabling a wide range of tasks including I2V/V2V editing, camera control, and spatial object editing. Extensive experiments demonstrate that FlexAM achieves superior performance across all evaluated tasks.

  </details>



- **LongStream: Long-Sequence Streaming Autoregressive Visual Geometry**  
  Chong Cheng, Xianda Chen, Tao Xie, Wei Yin, Weiqiang Ren, Qian Zhang, Xiaoyuang Guo, Hao Wang  
  _2026-02-13_ · https://arxiv.org/abs/2602.13172v1  
  <details><summary>Abstract</summary>

  Long-sequence streaming 3D reconstruction remains a significant open challenge. Existing autoregressive models often fail when processing long sequences. They typically anchor poses to the first frame, which leads to attention decay, scale drift, and extrapolation errors. We introduce LongStream, a novel gauge-decoupled streaming visual geometry model for metric-scale scene reconstruction across thousands of frames. Our approach is threefold. First, we discard the first-frame anchor and predict keyframe-relative poses. This reformulates long-range extrapolation into a constant-difficulty local task. Second, we introduce orthogonal scale learning. This method fully disentangles geometry from scale estimation to suppress drift. Finally, we solve Transformer cache issues such as attention-sink reliance and long-term KV-cache contamination. We propose cache-consistent training combined with periodic cache refresh. This approach suppresses attention degradation over ultra-long sequences and reduces the gap between training and inference. Experiments show LongStream achieves state-of-the-art performance. It delivers stable, metric-scale reconstruction over kilometer-scale sequences at 18 FPS. Project Page: https://3dagentworld.github.io/longstream/

  </details>



- **Implicit-Scale 3D Reconstruction for Multi-Food Volume Estimation from Monocular Images**  
  Yuhao Chen, Gautham Vinod, Siddeshwar Raghavan, Talha Ibn Mahmud, Bruce Coburn, Jinge Ma, Fengqing Zhu, Jiangpeng He  
  _2026-02-13_ · https://arxiv.org/abs/2602.13041v1  
  <details><summary>Abstract</summary>

  We present Implicit-Scale 3D Reconstruction from Monocular Multi-Food Images, a benchmark dataset designed to advance geometry-based food portion estimation in realistic dining scenarios. Existing dietary assessment methods largely rely on single-image analysis or appearance-based inference, including recent vision-language models, which lack explicit geometric reasoning and are sensitive to scale ambiguity. This benchmark reframes food portion estimation as an implicit-scale 3D reconstruction problem under monocular observations. To reflect real-world conditions, explicit physical references and metric annotations are removed; instead, contextual objects such as plates and utensils are provided, requiring algorithms to infer scale from implicit cues and prior knowledge. The dataset emphasizes multi-food scenes with diverse object geometries, frequent occlusions, and complex spatial arrangements. The benchmark was adopted as a challenge at the MetaFood 2025 Workshop, where multiple teams proposed reconstruction-based solutions. Experimental results show that while strong vision--language baselines achieve competitive performance, geometry-based reconstruction methods provide both improved accuracy and greater robustness, with the top-performing approach achieving 0.21 MAPE in volume estimation and 5.7 L1 Chamfer Distance in geometric accuracy.

  </details>



- **X-VORTEX: Spatio-Temporal Contrastive Learning for Wake Vortex Trajectory Forecasting**  
  Zhan Qu, Michael Färber  
  _2026-02-13_ · https://arxiv.org/abs/2602.12869v1  
  <details><summary>Abstract</summary>

  Wake vortices are strong, coherent air turbulences created by aircraft, and they pose a major safety and capacity challenge for air traffic management. Tracking how vortices move, weaken, and dissipate over time from LiDAR measurements is still difficult because scans are sparse, vortex signatures fade as the flow breaks down under atmospheric turbulence and instabilities, and point-wise annotation is prohibitively expensive. Existing approaches largely treat each scan as an independent, fully supervised segmentation problem, which overlooks temporal structure and does not scale to the vast unlabeled archives collected in practice. We present X-VORTEX, a spatio-temporal contrastive learning framework grounded in Augmentation Overlap Theory that learns physics-aware representations from unlabeled LiDAR point cloud sequences. X-VORTEX addresses two core challenges: sensor sparsity and time-varying vortex dynamics. It constructs paired inputs from the same underlying flight event by combining a weakly perturbed sequence with a strongly augmented counterpart produced via temporal subsampling and spatial masking, encouraging the model to align representations across missing frames and partial observations. Architecturally, a time-distributed geometric encoder extracts per-scan features and a sequential aggregator models the evolving vortex state across variable-length sequences. We evaluate on a real-world dataset of over one million LiDAR scans. X-VORTEX achieves superior vortex center localization while using only 1% of the labeled data required by supervised baselines, and the learned representations support accurate trajectory forecasting.

  </details>



- **GSM-GS: Geometry-Constrained Single and Multi-view Gaussian Splatting for Surface Reconstruction**  
  Xiao Ren, Yu Liu, Ning An, Jian Cheng, Xin Qiao, He Kong  
  _2026-02-13_ · https://arxiv.org/abs/2602.12796v1  
  <details><summary>Abstract</summary>

  Recently, 3D Gaussian Splatting has emerged as a prominent research direction owing to its ultrarapid training speed and high-fidelity rendering capabilities. However, the unstructured and irregular nature of Gaussian point clouds poses challenges to reconstruction accuracy. This limitation frequently causes high-frequency detail loss in complex surface microstructures when relying solely on routine strategies. To address this limitation, we propose GSM-GS: a synergistic optimization framework integrating single-view adaptive sub-region weighting constraints and multi-view spatial structure refinement. For single-view optimization, we leverage image gradient features to partition scenes into texture-rich and texture-less sub-regions. The reconstruction quality is enhanced through adaptive filtering mechanisms guided by depth discrepancy features. This preserves high-weight regions while implementing a dual-branch constraint strategy tailored to regional texture variations, thereby improving geometric detail characterization. For multi-view optimization, we introduce a geometry-guided cross-view point cloud association method combined with a dynamic weight sampling strategy. This constructs 3D structural normal constraints across adjacent point cloud frames, effectively reinforcing multi-view consistency and reconstruction fidelity. Extensive experiments on public datasets demonstrate that our method achieves both competitive rendering quality and geometric reconstruction. See our interactive project page

  </details>


