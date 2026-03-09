# 3D Reconstruction

_Updated: 2026-03-09 07:16 UTC_

Total papers shown: **9**


---

- **SCOPE: Scene-Contextualized Incremental Few-Shot 3D Segmentation**  
  Vishal Thengane, Zhaochong An, Tianjin Huang, Son Lam Phung, Abdesselam Bouzerdoum, Lu Yin, Na Zhao, Xiatian Zhu  
  _2026-03-06_ · https://arxiv.org/abs/2603.06572v1  
  <details><summary>Abstract</summary>

  Incremental Few-Shot (IFS) segmentation aims to learn new categories over time from only a few annotations. Although widely studied in 2D, it remains underexplored for 3D point clouds. Existing methods suffer from catastrophic forgetting or fail to learn discriminative prototypes under sparse supervision, and often overlook a key cue: novel categories frequently appear as unlabelled background in base-training scenes. We introduce SCOPE (Scene-COntextualised Prototype Enrichment), a plug-and-play background-guided prototype enrichment framework that integrates with any prototype-based 3D segmentation method. After base training, a class-agnostic segmentation model extracts high-confidence pseudo-instances from background regions to build a prototype pool. When novel classes arrive with few labelled samples, relevant background prototypes are retrieved and fused with few-shot prototypes to form enriched representations without retraining the backbone or adding parameters. Experiments on ScanNet and S3DIS show that SCOPE achieves SOTA performance, improving novel-class IoU by up to 6.98% and 3.61%, and mean IoU by 2.25% and 1.70%, respectively, while maintaining low forgetting. Code is available https://github.com/Surrey-UP-Lab/SCOPE.

  </details>



- **SG-DOR: Learning Scene Graphs with Direction-Conditioned Occlusion Reasoning for Pepper Plants**  
  Rohit Menon, Niklas Mueller-Goldingen, Sicong Pan, Gokul Krishna Chenchani, Maren Bennewitz  
  _2026-03-06_ · https://arxiv.org/abs/2603.06512v1  
  <details><summary>Abstract</summary>

  Robotic harvesting in dense crop canopies requires effective interventions that depend not only on geometry, but also on explicit, direction-conditioned relations identifying which organs obstruct a target fruit. We present SG-DOR (Scene Graphs with Direction-Conditioned Occlusion Reasoning), a relational framework that, given instance-segmented organ point clouds, infers a scene graph encoding physical attachments and direction-conditioned occlusion. We introduce an occlusion ranking task for retrieving and ranking candidate leaves for a target fruit and approach direction, and propose a direction-aware graph neural architecture with per-fruit leaf-set attention and union-level aggregation. Experiments on a multi-plant synthetic pepper dataset show improved occlusion prediction (F1=0.73, NDCG@3=0.85) and attachment inference (edge F1=0.83) over strong ablations, yielding a structured relational signal for downstream intervention planning.

  </details>



- **Rewis3d: Reconstruction Improves Weakly-Supervised Semantic Segmentation**  
  Jonas Ernst, Wolfgang Boettcher, Lukas Hoyer, Jan Eric Lenssen, Bernt Schiele  
  _2026-03-06_ · https://arxiv.org/abs/2603.06374v1  
  <details><summary>Abstract</summary>

  We present Rewis3d, a framework that leverages recent advances in feed-forward 3D reconstruction to significantly improve weakly supervised semantic segmentation on 2D images. Obtaining dense, pixel-level annotations remains a costly bottleneck for training segmentation models. Alleviating this issue, sparse annotations offer an efficient weakly-supervised alternative. However, they still incur a performance gap. To address this, we introduce a novel approach that leverages 3D scene reconstruction as an auxiliary supervisory signal. Our key insight is that 3D geometric structure recovered from 2D videos provides strong cues that can propagate sparse annotations across entire scenes. Specifically, a dual student-teacher architecture enforces semantic consistency between 2D images and reconstructed 3D point clouds, using state-of-the-art feed-forward reconstruction to generate reliable geometric supervision. Extensive experiments demonstrate that Rewis3d achieves state-of-the-art performance in sparse supervision, outperforming existing approaches by 2-7% without requiring additional labels or inference overhead.

  </details>



- **P-SLCR: Unsupervised Point Cloud Semantic Segmentation via Prototypes Structure Learning and Consistent Reasoning**  
  Lixin Zhan, Jie Jiang, Tianjian Zhou, Yukun Du, Yan Zheng, Xuehu Duan  
  _2026-03-06_ · https://arxiv.org/abs/2603.06321v1  
  <details><summary>Abstract</summary>

  Current semantic segmentation approaches for point cloud scenes heavily rely on manual labeling, while research on unsupervised semantic segmentation methods specifically for raw point clouds is still in its early stages. Unsupervised point cloud learning poses significant challenges due to the absence of annotation information and the lack of pre-training. The development of effective strategies is crucial in this context. In this paper, we propose a novel prototype library-driven unsupervised point cloud semantic segmentation strategy that utilizes Structure Learning and Consistent Reasoning (P-SLCR). First, we propose a Consistent Structure Learning to establish structural feature learning between consistent points and the library of consistent prototypes by selecting high-quality features. Second, we propose a Semantic Relation Consistent Reasoning that constructs a prototype inter-relation matrix between consistent and ambiguous prototype libraries separately. This process ensures the preservation of semantic consistency by imposing constraints on consistent and ambiguous prototype libraries through the prototype inter-relation matrix. Finally, our method was extensively evaluated on the S3DIS, SemanticKITTI, and Scannet datasets, achieving the best performance compared to unsupervised methods. Specifically, the mIoU of 47.1% is achieved for Area-5 of the S3DIS dataset, surpassing the classical fully supervised method PointNet by 2.5%.

  </details>



- **Hierarchical Collaborative Fusion for 3D Instance-aware Referring Expression Segmentation**  
  Keshen Zhou, Runnan Chen, Mingming Gong, Tongliang Liu  
  _2026-03-06_ · https://arxiv.org/abs/2603.06250v1  
  <details><summary>Abstract</summary>

  Generalised 3D Referring Expression Segmentation (3D-GRES) localizes objects in 3D scenes based on natural language, even when descriptions match multiple or zero targets. Existing methods rely solely on sparse point clouds, lacking rich visual semantics for fine-grained descriptions. We propose HCF-RES, a multi-modal framework with two key innovations. First, Hierarchical Visual Semantic Decomposition leverages SAM instance masks to guide CLIP encoding at dual granularities -- pixel-level and instance-level features -- preserving object boundaries during 2D-to-3D projection. Second, Progressive Multi-level Fusion integrates representations through intra-modal collaboration, cross-modal adaptive weighting between 2D semantic and 3D geometric features, and language-guided refinement. HCF-RES achieves state-of-the-art results on both ScanRefer and Multi3DRefer.

  </details>



- **EntON: Eigenentropy-Optimized Neighborhood Densification in 3D Gaussian Splatting**  
  Miriam Jäger, Boris Jutzi  
  _2026-03-06_ · https://arxiv.org/abs/2603.06216v1  
  <details><summary>Abstract</summary>

  We present a novel Eigenentropy-optimized neighboorhood densification strategy EntON in 3D Gaussian Splatting (3DGS) for geometrically accurate and high-quality rendered 3D reconstruction. While standard 3DGS produces Gaussians whose centers and surfaces are poorly aligned with the underlying object geometry, surface-focused reconstruction methods frequently sacrifice photometric accuracy. In contrast to the conventional densification strategy, which relies on the magnitude of the view-space position gradient, our approach introduces a geometry-aware strategy to guide adaptive splitting and pruning. Specifically, we compute the 3D shape feature Eigenentropy from the eigenvalues of the covariance matrix in the k-nearest neighborhood of each Gaussian center, which quantifies the local structural order. These Eigenentropy values are integrated into an alternating optimization framework: During the optimization process, the algorithm alternates between (i) standard gradient-based densification, which refines regions via view-space gradients, and (ii) Eigenentropy-aware densification, which preferentially densifies Gaussians in low-Eigenentropy (ordered, flat) neighborhoods to better capture fine geometric details on the object surface, and prunes those in high-Eigenentropy (disordered, spherical) regions. We provide quantitative and qualitative evaluations on two benchmark datasets: small-scale DTU dataset and large-scale TUM2TWIN dataset, covering man-made objects and urban scenes. Experiments demonstrate that our Eigenentropy-aware alternating densification strategy improves geometric accuracy by up to 33% and rendering quality by up to 7%, while reducing the number of Gaussians by up to 50% and training time by up to 23%. Overall, EnTON achieves a favorable balance between geometric accuracy, rendering quality and efficiency by avoiding unnecessary scene expansion.

  </details>



- **JOPP-3D: Joint Open Vocabulary Semantic Segmentation on Point Clouds and Panoramas**  
  Sandeep Inuganti, Hideaki Kanayama, Kanta Shimizu, Mahdi Chamseddine, Soichiro Yokota, Didier Stricker, Jason Rambach  
  _2026-03-06_ · https://arxiv.org/abs/2603.06168v1  
  <details><summary>Abstract</summary>

  Semantic segmentation across visual modalities such as 3D point clouds and panoramic images remains a challenging task, primarily due to the scarcity of annotated data and the limited adaptability of fixed-label models. In this paper, we present JOPP-3D, an open-vocabulary semantic segmentation framework that jointly leverages panoramic and point cloud data to enable language-driven scene understanding. We convert RGB-D panoramic images into their corresponding tangential perspective images and 3D point clouds, then use these modalities to extract and align foundational vision-language features. This allows natural language querying to generate semantic masks on both input modalities. Experimental evaluation on the Stanford-2D-3D-s and ToF-360 datasets demonstrates the capability of JOPP-3D to produce coherent and semantically meaningful segmentations across panoramic and 3D domains. Our proposed method achieves a significant improvement compared to the SOTA in open and closed vocabulary 2D and 3D semantic segmentation.

  </details>



- **Transforming Omnidirectional RGB-LiDAR data into 3D Gaussian Splatting**  
  Semin Bae, Hansol Lim, Jongseong Brad Choi  
  _2026-03-06_ · https://arxiv.org/abs/2603.06061v1  
  <details><summary>Abstract</summary>

  The demand for large-scale digital twins is rapidly growing in robotics and autonomous driving. However, constructing these environments with 3D Gaussian Splatting (3DGS) usually requires expensive, purpose-built data collection. Meanwhile, deployed platforms routinely collect extensive omnidirectional RGB and LiDAR logs, but a significant portion of these sensor data is directly discarded or strictly underutilized due to transmission constraints and the lack of scalable reuse pipeline. In this paper, we present an omnidirectional RGB-LiDAR reuse pipeline that transforms these archived logs into robust initialization assets for 3DGS. Direct conversion of such raw logs introduces practical bottlenecks: inherent non-linear distortion leads to unreliable Structure-from-Motion (SfM) tracking, and dense, unorganized LiDAR clouds cause computational overhead during 3DGS optimization. To overcome these challenges, our pipeline strategically integrates an ERP-to-cubemap conversion module for deterministic spatial anchoring, alongside PRISM-a color stratified downsampling strategy. By bridging these multi-modal inputs via Fast Point Feature Histograms (FPFH) based global registration and Iterative Closest Point (ICP), our pipeline successfully repurposes a considerable fraction of discarded data into usable SfM geometry. Furthermore, our LiDAR-reinforced initialization consistently enhances the final 3DGS rendering fidelity in structurally complex scenes compared to vision-only baselines. Ultimately, this work provides a deterministic workflow for creating simulation-grade digital twins from standard archived sensor logs.

  </details>



- **RePer-360: Releasing Perspective Priors for 360$^\circ$ Depth Estimation via Self-Modulation**  
  Cheng Guan, Chunyu Lin, Zhijie Shen, Junsong Zhang, Jiyuan Wang  
  _2026-03-06_ · https://arxiv.org/abs/2603.05999v1  
  <details><summary>Abstract</summary>

  Recent depth foundation models trained on perspective imagery achieve strong performance, yet generalize poorly to 360$^\circ$ images due to the substantial geometric discrepancy between perspective and panoramic domains. Moreover, fully fine-tuning these models typically requires large amounts of panoramic data. To address this issue, we propose RePer-360, a distortion-aware self-modulation framework for monocular panoramic depth estimation that adapts depth foundation models while preserving powerful pretrained perspective priors. Specifically, we design a lightweight geometry-aligned guidance module to derive a modulation signal from two complementary projections (i.e., ERP and CP) and use it to guide the model toward the panoramic domain without overwriting its pretrained perspective knowledge. We further introduce a Self-Conditioned AdaLN-Zero mechanism that produces pixel-wise scaling factors to reduce the feature distribution gap between the perspective and panoramic domains. In addition, a cubemap-domain consistency loss further improves training stability and cross-projection alignment. By shifting the focus from complementary-projection fusion to panoramic domain adaptation under preserved pretrained perspective priors, RePer-360 surpasses standard fine-tuning methods while using only 1\% of the training data. Under the same in-domain training setting, it further achieves an approximately 20\% improvement in RMSE. Code will be released upon acceptance.

  </details>


