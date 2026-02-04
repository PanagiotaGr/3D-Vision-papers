# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-02-04 07:08 UTC_

Total papers shown: **8**


---

- **Occlusion-Free Conformal Lensing for Spatiotemporal Visualization in 3D Urban Analytics**  
  Roberta Mota, Julio D. Silva, Fabio Miranda, Usman Alim, Ehud Sharlin, Nivan Ferreira  
  _2026-02-03_ · https://arxiv.org/abs/2602.03743v1  
  <details><summary>Abstract</summary>

  The visualization of temporal data on urban buildings, such as shadows, noise, and solar potential, plays a critical role in the analysis of dynamic urban phenomena. However, in dense and geographically constrained 3D urban environments, visual representations of time-varying building data often suffer from occlusion and visual clutter. To address these two challenges, we introduce an immersive lens visualization that integrates a view-dependent cutaway de-occlusion technique and a temporal display derived from a conformal mapping algorithm. The mapping process first partitions irregular building footprints into smaller, sufficiently regular subregions that serve as structural primitives. These subregions are then seamlessly recombined to form a conformal, layered layout for our temporal lens visualization. The view-responsive cutaway is inspired by traditional architectural illustrations, preserving the overall layout of the building and its surroundings to maintain users' sense of spatial orientation. This lens design enables the occlusion-free embedding of shape-adaptive temporal displays across building facades on demand, supporting rapid time-space association for the discovery, access and interpretation of spatiotemporal urban patterns. Guided by domain and design goals, we outline the rationale behind the lens visual and interaction design choices, such as the encoding of time progression and temporal values in the conforming lens image. A user study compares our approach against conventional juxtaposition and x-ray spatiotemporal designs. Results validate the usage and utility of our lens, showing that it improves task accuracy and completion time, reduces navigation effort, and increases user confidence. From these findings, we distill design recommendations and promising directions for future research on spatially-embedded lenses in 3D visualization and urban analytics.

  </details>



- **Constrained Dynamic Gaussian Splatting**  
  Zihan Zheng, Zhenglong Wu, Xuanxuan Wang, Houqiang Zhong, Xiaoyun Zhang, Qiang Hu, Guangtao Zhai, Wenjun Zhang  
  _2026-02-03_ · https://arxiv.org/abs/2602.03538v1  
  <details><summary>Abstract</summary>

  While Dynamic Gaussian Splatting enables high-fidelity 4D reconstruction, its deployment is severely hindered by a fundamental dilemma: unconstrained densification leads to excessive memory consumption incompatible with edge devices, whereas heuristic pruning fails to achieve optimal rendering quality under preset Gaussian budgets. In this work, we propose Constrained Dynamic Gaussian Splatting (CDGS), a novel framework that formulates dynamic scene reconstruction as a budget-constrained optimization problem to enforce a strict, user-defined Gaussian budget during training. Our key insight is to introduce a differentiable budget controller as the core optimization driver. Guided by a multi-modal unified importance score, this controller fuses geometric, motion, and perceptual cues for precise capacity regulation. To maximize the utility of this fixed budget, we further decouple the optimization of static and dynamic elements, employing an adaptive allocation mechanism that dynamically distributes capacity based on motion complexity. Furthermore, we implement a three-phase training strategy to seamlessly integrate these constraints, ensuring precise adherence to the target count. Coupled with a dual-mode hybrid compression scheme, CDGS not only strictly adheres to hardware constraints (error < 2%}) but also pushes the Pareto frontier of rate-distortion performance. Extensive experiments demonstrate that CDGS delivers optimal rendering quality under varying capacity limits, achieving over 3x compression compared to state-of-the-art methods.

  </details>



- **From Single Scan to Sequential Consistency: A New Paradigm for LIDAR Relocalization**  
  Minghang Zhu, Zhijing Wang, Yuxin Guo, Wen Li, Sheng Ao, Cheng Wang  
  _2026-02-03_ · https://arxiv.org/abs/2602.03198v1  
  <details><summary>Abstract</summary>

  LiDAR relocalization aims to estimate the global 6-DoF pose of a sensor in the environment. However, existing regression-based approaches are prone to dynamic or ambiguous scenarios, as they either solely rely on single-frame inference or neglect the spatio-temporal consistency across scans. In this paper, we propose TempLoc, a new LiDAR relocalization framework that enhances the robustness of localization by effectively modeling sequential consistency. Specifically, a Global Coordinate Estimation module is first introduced to predict point-wise global coordinates and associated uncertainties for each LiDAR scan. A Prior Coordinate Generation module is then presented to estimate inter-frame point correspondences by the attention mechanism. Lastly, an Uncertainty-Guided Coordinate Fusion module is deployed to integrate both predictions of point correspondence in an end-to-end fashion, yielding a more temporally consistent and accurate global 6-DoF pose. Experimental results on the NCLT and Oxford Robot-Car benchmarks show that our TempLoc outperforms stateof-the-art methods by a large margin, demonstrating the effectiveness of temporal-aware correspondence modeling in LiDAR relocalization. Our code will be released soon.

  </details>



- **JRDB-Pose3D: A Multi-person 3D Human Pose and Shape Estimation Dataset for Robotics**  
  Sandika Biswas, Kian Izadpanah, Hamid Rezatofighi  
  _2026-02-03_ · https://arxiv.org/abs/2602.03064v1  
  <details><summary>Abstract</summary>

  Real-world scenes are inherently crowded. Hence, estimating 3D poses of all nearby humans, tracking their movements over time, and understanding their activities within social and environmental contexts are essential for many applications, such as autonomous driving, robot perception, robot navigation, and human-robot interaction. However, most existing 3D human pose estimation datasets primarily focus on single-person scenes or are collected in controlled laboratory environments, which restricts their relevance to real-world applications. To bridge this gap, we introduce JRDB-Pose3D, which captures multi-human indoor and outdoor environments from a mobile robotic platform. JRDB-Pose3D provides rich 3D human pose annotations for such complex and dynamic scenes, including SMPL-based pose annotations with consistent body-shape parameters and track IDs for each individual over time. JRDB-Pose3D contains, on average, 5-10 human poses per frame, with some scenes featuring up to 35 individuals simultaneously. The proposed dataset presents unique challenges, including frequent occlusions, truncated bodies, and out-of-frame body parts, which closely reflect real-world environments. Moreover, JRDB-Pose3D inherits all available annotations from the JRDB dataset, such as 2D pose, information about social grouping, activities, and interactions, full-scene semantic masks with consistent human- and object-level tracking, and detailed annotations for each individual, such as age, gender, and race, making it a holistic dataset for a wide range of downstream perception and human-centric understanding tasks.

  </details>



- **SharpTimeGS: Sharp and Stable Dynamic Gaussian Splatting via Lifespan Modulation**  
  Zhanfeng Liao, Jiajun Zhang, Hanzhang Tu, Zhixi Wang, Yunqi Gao, Hongwen Zhang, Yebin Liu  
  _2026-02-03_ · https://arxiv.org/abs/2602.02989v1  
  <details><summary>Abstract</summary>

  Novel view synthesis of dynamic scenes is fundamental to achieving photorealistic 4D reconstruction and immersive visual experiences. Recent progress in Gaussian-based representations has significantly improved real-time rendering quality, yet existing methods still struggle to maintain a balance between long-term static and short-term dynamic regions in both representation and optimization. To address this, we present SharpTimeGS, a lifespan-aware 4D Gaussian framework that achieves temporally adaptive modeling of both static and dynamic regions under a unified representation. Specifically, we introduce a learnable lifespan parameter that reformulates temporal visibility from a Gaussian-shaped decay into a flat-top profile, allowing primitives to remain consistently active over their intended duration and avoiding redundant densification. In addition, the learned lifespan modulates each primitives' motion, reducing drift in long-lived static points while retaining unrestricted motion for short-lived dynamic ones. This effectively decouples motion magnitude from temporal duration, improving long-term stability without compromising dynamic fidelity. Moreover, we design a lifespan-velocity-aware densification strategy that mitigates optimization imbalance between static and dynamic regions by allocating more capacity to regions with pronounced motion while keeping static areas compact and stable. Extensive experiments on multiple benchmarks demonstrate that our method achieves state-of-the-art performance while supporting real-time rendering up to 4K resolution at 100 FPS on one RTX 4090.

  </details>



- **Spatio-Temporal Transformers for Long-Term NDVI Forecasting**  
  Ido Faran, Nathan S. Netanyahu, Maxim Shoshany  
  _2026-02-02_ · https://arxiv.org/abs/2602.01799v1  
  <details><summary>Abstract</summary>

  Long-term satellite image time series (SITS) analysis in heterogeneous landscapes faces significant challenges, particularly in Mediterranean regions where complex spatial patterns, seasonal variations, and multi-decade environmental changes interact across different scales. This paper presents the Spatio-Temporal Transformer for Long Term Forecasting (STT-LTF ), an extended framework that advances beyond purely temporal analysis to integrate spatial context modeling with temporal sequence prediction. STT-LTF processes multi-scale spatial patches alongside temporal sequences (up to 20 years) through a unified transformer architecture, capturing both local neighborhood relationships and regional climate influences. The framework employs comprehensive self-supervised learning with spatial masking, temporal masking, and horizon sampling strategies, enabling robust model training from 40 years of unlabeled Landsat imagery. Unlike autoregressive approaches, STT-LTF directly predicts arbitrary future time points without error accumulation, incorporating spatial patch embeddings, cyclical temporal encoding, and geographic coordinates to learn complex dependencies across heterogeneous Mediterranean ecosystems. Experimental evaluation on Landsat data (1984-2024) demonstrates that STT-LTF achieves a Mean Absolute Error (MAE) of 0.0328 and R^2 of 0.8412 for next-year predictions, outperforming traditional statistical methods, CNN-based approaches, LSTM networks, and standard transformers. The framework's ability to handle irregular temporal sampling and variable prediction horizons makes it particularly suitable for analysis of heterogeneous landscapes experiencing rapid ecological transitions.

  </details>



- **PISCES: Annotation-free Text-to-Video Post-Training via Optimal Transport-Aligned Rewards**  
  Minh-Quan Le, Gaurav Mittal, Cheng Zhao, David Gu, Dimitris Samaras, Mei Chen  
  _2026-02-02_ · https://arxiv.org/abs/2602.01624v1  
  <details><summary>Abstract</summary>

  Text-to-video (T2V) generation aims to synthesize videos with high visual quality and temporal consistency that are semantically aligned with input text. Reward-based post-training has emerged as a promising direction to improve the quality and semantic alignment of generated videos. However, recent methods either rely on large-scale human preference annotations or operate on misaligned embeddings from pre-trained vision-language models, leading to limited scalability or suboptimal supervision. We present $\texttt{PISCES}$, an annotation-free post-training algorithm that addresses these limitations via a novel Dual Optimal Transport (OT)-aligned Rewards module. To align reward signals with human judgment, $\texttt{PISCES}$ uses OT to bridge text and video embeddings at both distributional and discrete token levels, enabling reward supervision to fulfill two objectives: (i) a Distributional OT-aligned Quality Reward that captures overall visual quality and temporal coherence; and (ii) a Discrete Token-level OT-aligned Semantic Reward that enforces semantic, spatio-temporal correspondence between text and video tokens. To our knowledge, $\texttt{PISCES}$ is the first to improve annotation-free reward supervision in generative post-training through the lens of OT. Experiments on both short- and long-video generation show that $\texttt{PISCES}$ outperforms both annotation-based and annotation-free methods on VBench across Quality and Semantic scores, with human preference studies further validating its effectiveness. We show that the Dual OT-aligned Rewards module is compatible with multiple optimization paradigms, including direct backpropagation and reinforcement learning fine-tuning.

  </details>



- **UniDWM: Towards a Unified Driving World Model via Multifaceted Representation Learning**  
  Shuai Liu, Siheng Ren, Xiaoyao Zhu, Quanmin Liang, Zefeng Li, Qiang Li, Xin Hu, Kai Huang  
  _2026-02-02_ · https://arxiv.org/abs/2602.01536v1  
  <details><summary>Abstract</summary>

  Achieving reliable and efficient planning in complex driving environments requires a model that can reason over the scene's geometry, appearance, and dynamics. We present UniDWM, a unified driving world model that advances autonomous driving through multifaceted representation learning. UniDWM constructs a structure- and dynamic-aware latent world representation that serves as a physically grounded state space, enabling consistent reasoning across perception, prediction, and planning. Specifically, a joint reconstruction pathway learns to recover the scene's structure, including geometry and visual texture, while a collaborative generation framework leverages a conditional diffusion transformer to forecast future world evolution within the latent space. Furthermore, we show that our UniDWM can be deemed as a variation of VAE, which provides theoretical guidance for the multifaceted representation learning. Extensive experiments demonstrate the effectiveness of UniDWM in trajectory planning, 4D reconstruction and generation, highlighting the potential of multifaceted world representations as a foundation for unified driving intelligence. The code will be publicly available at https://github.com/Say2L/UniDWM.

  </details>


