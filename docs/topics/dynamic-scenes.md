# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-02-03 07:08 UTC_

Total papers shown: **6**


---

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



- **Q-DiT4SR: Exploration of Detail-Preserving Diffusion Transformer Quantization for Real-World Image Super-Resolution**  
  Xun Zhang, Kaicheng Yang, Hongliang Lu, Haotong Qin, Yong Guo, Yulun Zhang  
  _2026-02-01_ · https://arxiv.org/abs/2602.01273v1  
  <details><summary>Abstract</summary>

  Recently, Diffusion Transformers (DiTs) have emerged in Real-World Image Super-Resolution (Real-ISR) to generate high-quality textures, yet their heavy inference burden hinders real-world deployment. While Post-Training Quantization (PTQ) is a promising solution for acceleration, existing methods in super-resolution mostly focus on U-Net architectures, whereas generic DiT quantization is typically designed for text-to-image tasks. Directly applying these methods to DiT-based super-resolution models leads to severe degradation of local textures. Therefore, we propose Q-DiT4SR, the first PTQ framework specifically tailored for DiT-based Real-ISR. We propose H-SVD, a hierarchical SVD that integrates a global low-rank branch with a local block-wise rank-1 branch under a matched parameter budget. We further propose Variance-aware Spatio-Temporal Mixed Precision: VaSMP allocates cross-layer weight bit-widths in a data-free manner based on rate-distortion theory, while VaTMP schedules intra-layer activation precision across diffusion timesteps via dynamic programming (DP) with minimal calibration. Experiments on multiple real-world datasets demonstrate that our Q-DiT4SR achieves SOTA performance under both W4A6 and W4A4 settings. Notably, the W4A4 quantization configuration reduces model size by 5.8$\times$ and computational operations by over 60$\times$. Our code and models will be available at https://github.com/xunzhang1128/Q-DiT4SR.

  </details>



- **FUSE-Flow: Scalable Real-Time Multi-View Point Cloud Reconstruction Using Confidence**  
  Chentian Sun  
  _2026-02-01_ · https://arxiv.org/abs/2602.01035v1  
  <details><summary>Abstract</summary>

  Real-time multi-view point cloud reconstruction is a core problem in 3D vision and immersive perception, with wide applications in VR, AR, robotic navigation, digital twins, and computer interaction. Despite advances in multi-camera systems and high-resolution depth sensors, fusing large-scale multi-view depth observations into high-quality point clouds under strict real-time constraints remains challenging. Existing methods relying on voxel-based fusion, temporal accumulation, or global optimization suffer from high computational complexity, excessive memory usage, and limited scalability, failing to simultaneously achieve real-time performance, reconstruction quality, and multi-camera extensibility. We propose FUSE-Flow, a frame-wise, stateless, and linearly scalable point cloud streaming reconstruction framework. Each frame independently generates point cloud fragments, fused via two weights, measurement confidence and 3D distance consistency to suppress noise while preserving geometric details. For large-scale multi-camera efficiency, we introduce an adaptive spatial hashing-based weighted aggregation method: 3D space is adaptively partitioned by local point cloud density, representative points are selected per cell, and weighted fusion is performed to handle both sparse and dense regions. With GPU parallelization, FUSE-Flow achieves high-throughput, low-latency point cloud generation and fusion with linear complexity. Experiments demonstrate that the framework improves reconstruction stability and geometric fidelity in overlapping, depth-discontinuous, and dynamic scenes, while maintaining real-time frame rates on modern GPUs, verifying its effectiveness, robustness, and scalability.

  </details>



- **Schrödinger-Inspired Time-Evolution for 4D Deformation Forecasting**  
  Ahsan Raza Siyal, Markus Haltmeier, Ruth Steiger, Elke Ruth Gizewski, Astrid Ellen Grams  
  _2026-01-31_ · https://arxiv.org/abs/2602.00661v1  
  <details><summary>Abstract</summary>

  Spatiotemporal forecasting of complex three-dimensional phenomena (4D: 3D + time) is fundamental to applications in medical imaging, fluid and material dynamics, and geophysics. In contrast to unconstrained neural forecasting models, we propose a Schrödinger-inspired, physics-guided neural architecture that embeds an explicit time-evolution operator within a deep convolutional framework for 4D prediction. From observed volumetric sequences, the model learns voxelwise amplitude, phase, and potential fields that define a complex-valued wavefunction $ψ= A e^{iφ}$, which is evolved forward in time using a differentiable, unrolled Schrödinger time stepper. This physics-guided formulation yields several key advantages: (i) temporal stability arising from the structured evolution operator, which mitigates drift and error accumulation in long-horizon forecasting; (ii) an interpretable latent representation, where phase encodes transport dynamics, amplitude captures structural intensity, and the learned potential governs spatiotemporal interactions; and (iii) natural compatibility with deformation-based synthesis, which is critical for preserving anatomical fidelity in medical imaging applications. By integrating physical priors directly into the learning process, the proposed approach combines the expressivity of deep networks with the robustness and interpretability of physics-based modeling. We demonstrate accurate and stable prediction of future 4D states, including volumetric intensities and deformation fields, on synthetic benchmarks that emulate realistic shape deformations and topological changes. To our knowledge, this is the first end-to-end 4D neural forecasting framework to incorporate a Schrödinger-type evolution operator, offering a principled pathway toward interpretable, stable, and anatomically consistent spatiotemporal prediction.

  </details>


