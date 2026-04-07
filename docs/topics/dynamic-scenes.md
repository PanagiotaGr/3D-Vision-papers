# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-04-07 07:47 UTC_

Total papers shown: **8**


---

- **PointTPA: Dynamic Network Parameter Adaptation for 3D Scene Understanding**  
  Siyuan Liu, Chaoqun Zheng, Xin Zhou, Tianrui Feng, Dingkang Liang, Xiang Bai  
  _2026-04-06_ · https://arxiv.org/abs/2604.04933v1  
  <details><summary>Abstract</summary>

  Scene-level point cloud understanding remains challenging due to diverse geometries, imbalanced category distributions, and highly varied spatial layouts. Existing methods improve object-level performance but rely on static network parameters during inference, limiting their adaptability to dynamic scene data. We propose PointTPA, a Test-time Parameter Adaptation framework that generates input-aware network parameters for scene-level point clouds. PointTPA adopts a Serialization-based Neighborhood Grouping (SNG) to form locally coherent patches and a Dynamic Parameter Projector (DPP) to produce patch-wise adaptive weights, enabling the backbone to adjust its behavior according to scene-specific variations while maintaining a low parameter overhead. Integrated into the PTv3 structure, PointTPA demonstrates strong parameter efficiency by introducing two lightweight modules of less than 2% of the backbone's parameters. Despite this minimal parameter overhead, PointTPA achieves 78.4% mIoU on ScanNet validation, surpassing existing parameter-efficient fine-tuning (PEFT) methods across multiple benchmarks, highlighting the efficacy of our test-time dynamic network parameter adaptation mechanism in enhancing 3D scene understanding. The code is available at https://github.com/H-EmbodVis/PointTPA.

  </details>



- **A Frame is Worth One Token: Efficient Generative World Modeling with Delta Tokens**  
  Tommie Kerssies, Gabriele Berton, Ju He, Qihang Yu, Wufei Ma, Daan de Geus, Gijs Dubbelman, Liang-Chieh Chen  
  _2026-04-06_ · https://arxiv.org/abs/2604.04913v1  
  <details><summary>Abstract</summary>

  Anticipating diverse future states is a central challenge in video world modeling. Discriminative world models produce a deterministic prediction that implicitly averages over possible futures, while existing generative world models remain computationally expensive. Recent work demonstrates that predicting the future in the feature space of a vision foundation model (VFM), rather than a latent space optimized for pixel reconstruction, requires significantly fewer world model parameters. However, most such approaches remain discriminative. In this work, we introduce DeltaTok, a tokenizer that encodes the VFM feature difference between consecutive frames into a single continuous "delta" token, and DeltaWorld, a generative world model operating on these tokens to efficiently generate diverse plausible futures. Delta tokens reduce video from a three-dimensional spatio-temporal representation to a one-dimensional temporal sequence, for example yielding a 1,024x token reduction with 512x512 frames. This compact representation enables tractable multi-hypothesis training, where many futures are generated in parallel and only the best is supervised. At inference, this leads to diverse predictions in a single forward pass. Experiments on dense forecasting tasks demonstrate that DeltaWorld forecasts futures that more closely align with real-world outcomes, while having over 35x fewer parameters and using 2,000x fewer FLOPs than existing generative world models. Code and weights: https://deltatok.github.io.

  </details>



- **GA-GS: Generation-Assisted Gaussian Splatting for Static Scene Reconstruction**  
  Yedong Shen, Shiqi Zhang, Sha Zhang, Yifan Duan, Xinran Zhang, Wenhao Yu, Lu Zhang, Jiajun Deng, Yanyong Zhang  
  _2026-04-06_ · https://arxiv.org/abs/2604.04331v1  
  <details><summary>Abstract</summary>

  Reconstructing static 3D scene from monocular video with dynamic objects is important for numerous applications such as virtual reality and autonomous driving. Current approaches typically rely on background for static scene reconstruction, limiting the ability to recover regions occluded by dynamic objects. In this paper, we propose GA-GS, a Generation-Assisted Gaussian Splatting method for Static Scene Reconstruction. The key innovation of our work lies in leveraging generation to assist in reconstructing occluded regions. We employ a motion-aware module to segment and remove dynamic regions, and thenuse a diffusion model to inpaint the occluded areas, providing pseudo-ground-truth supervision. To balance contributions from real background and generated region, we introduce a learnable authenticity scalar for each Gaussian primitive, which dynamically modulates opacity during splatting for authenticity-aware rendering and supervision. Since no existing dataset provides ground-truth static scene of video with dynamic objects, we construct a dataset named Trajectory-Match, using a fixed-path robot to record each scene with/without dynamic objects, enabling quantitative evaluation in reconstruction of occluded regions. Extensive experiments on both the DAVIS and our dataset show that GA-GS achieves state-of-the-art performance in static scene reconstruction, especially in challenging scenarios with large-scale, persistent occlusions.

  </details>



- **Uncertainty-Aware Test-Time Adaptation for Cross-Region Spatio-Temporal Fusion of Land Surface Temperature**  
  Sofiane Bouaziz, Adel Hafiane, Raphael Canals, Rachid Nedjai  
  _2026-04-05_ · https://arxiv.org/abs/2604.04153v1  
  <details><summary>Abstract</summary>

  Deep learning models have shown great promise in diverse remote sensing applications. However, they often struggle to generalize across geographic regions unseen during training due to domain shifts. Domain shifts occur when data distributions differ between the training region and new target regions, due to variations in land cover, climate, and environmental conditions. Test-time adaptation (TTA) has emerged as a solution to such shifts, but existing methods are primarily designed for classification and are not directly applicable to regression tasks. In this work, we address the regression task of spatio-temporal fusion (STF) for land surface temperature estimation. We propose an uncertainty-aware TTA framework that updates only the fusion module of a pre-trained STF model, guided by epistemic uncertainty, land use and land cover consistency, and bias correction, without requiring source data or labeled target samples. Experiments on four target regions with diverse climates, namely Rome in Italy, Cairo in Egypt, Madrid in Spain, and Montpellier in France, show consistent improvements in RMSE and MAE for a pre-trained model in Orléans, France. The average gains are 24.2% and 27.9%, respectively, even with limited unlabeled target data and only 10 TTA epochs.

  </details>



- **4C4D: 4 Camera 4D Gaussian Splatting**  
  Junsheng Zhou, Zhifan Yang, Liang Han, Wenyuan Zhang, Kanle Shi, Shenkun Xu, Yu-Shen Liu  
  _2026-04-05_ · https://arxiv.org/abs/2604.04063v1  
  <details><summary>Abstract</summary>

  This paper tackles the challenge of recovering 4D dynamic scenes from videos captured by as few as four portable cameras. Learning to model scene dynamics for temporally consistent novel-view rendering is a foundational task in computer graphics, where previous works often require dense multi-view captures using camera arrays of dozens or even hundreds of views. We propose \textbf{4C4D}, a novel framework that enables high-fidelity 4D Gaussian Splatting from video captures of extremely sparse cameras. Our key insight lies that the geometric learning under sparse settings is substantially more difficult than modeling appearance. Driven by this observation, we introduce a Neural Decaying Function on Gaussian opacities for enhancing the geometric modeling capability of 4D Gaussians. This design mitigates the inherent imbalance between geometry and appearance modeling in 4DGS by encouraging the 4DGS gradients to focus more on geometric learning. Extensive experiments across sparse-view datasets with varying camera overlaps show that 4C4D achieves superior performance over prior art. Project page at: https://junshengzhou.github.io/4C4D.

  </details>



- **HOIGS: Human-Object Interaction Gaussian Splatting**  
  Taewoo Kim, Suwoong Yeom, Jaehyun Pyun, Geonho Cha, Dongyoon Wee, Joonsik Nam, Yun-Seong Jeong, Kyeongbo Kong, Suk-Ju Kang  
  _2026-04-05_ · https://arxiv.org/abs/2604.04016v1  
  <details><summary>Abstract</summary>

  Reconstructing dynamic scenes with complex human-object interactions is a fundamental challenge in computer vision and graphics. Existing Gaussian Splatting methods either rely on human pose priors while neglecting dynamic objects, or approximate all motions within a single field, limiting their ability to capture interaction-rich dynamics. To address this gap, we propose Human-Object Interaction Gaussian Splatting (HOIGS), which explicitly models interaction-induced deformation between humans and objects through a cross-attention-based HOI module. Distinct deformation baselines are employed to extract features: HexPlane for humans and Cubic Hermite Spline (CHS) for objects. By integrating these heterogeneous features, HOIGS effectively captures interdependent motions and improves deformation estimation in scenarios involving occlusion, contact, and object manipulation. Comprehensive experiments on multiple datasets demonstrate that our method consistently outperforms state-of-the-art human-centric and 4D Gaussian approaches, highlighting the importance of explicitly modeling human-object interactions for high-fidelity reconstruction.

  </details>



- **Interpreting Video Representations with Spatio-Temporal Sparse Autoencoders**  
  Atahan Dokme, Sriram Vishwanath  
  _2026-04-05_ · https://arxiv.org/abs/2604.03919v1  
  <details><summary>Abstract</summary>

  We present the first systematic study of Sparse Autoencoders (SAEs) on video representations. Standard SAEs decompose video into interpretable, monosemantic features but destroy temporal coherence: hard TopK selection produces unstable feature assignments across frames, reducing autocorrelation by 36%. We propose spatio-temporal contrastive objectives and Matryoshka hierarchical grouping that recover and even exceed raw temporal coherence. The contrastive loss weight controls a tunable trade-off between reconstruction and temporal coherence. A systematic ablation on two backbones and two datasets shows that different configurations excel at different goals: reconstruction fidelity, temporal coherence, action discrimination, or interpretability. Contrastive SAE features improve action classification by +3.9% over raw features and text-video retrieval by up to 2.8xR@1. A cross-backbone analysis reveals that standard monosemanticity metrics contain a backbone-alignment artifact: both DINOv2 and VideoMAE produce equally monosemantic features under neutral (CLIP) similarity. Causal ablation confirms that contrastive training concentrates predictive signal into a small number of identifiable features.

  </details>



- **ART: Adaptive Relational Transformer for Pedestrian Trajectory Prediction with Temporal-Aware Relations**  
  Ruochen Li, Ziyi Chang, Junyan Hu, Jiannan Li, Amir Atapour-Abarghouei, Hubert P. H. Shum  
  _2026-04-04_ · https://arxiv.org/abs/2604.03649v1  
  <details><summary>Abstract</summary>

  Accurate prediction of real-world pedestrian trajectories is crucial for a wide range of robot-related applications. Recent approaches typically adopt graph-based or transformer-based frameworks to model interactions. Despite their effectiveness, these methods either introduce unnecessary computational overhead or struggle to represent the diverse and time-varying characteristics of human interactions. In this work, we present an Adaptive Relational Transformer (ART), which introduces a Temporal-Aware Relation Graph (TARG) to explicitly capture the evolution of pairwise interactions and an Adaptive Interaction Pruning (AIP) mechanism to reduce redundant computations efficiently. Extensive evaluations on ETH/UCY and NBA benchmarks show that ART delivers state-of-the-art accuracy with high computational efficiency.

  </details>


