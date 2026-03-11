# 3D Reconstruction

_Updated: 2026-03-11 07:09 UTC_

Total papers shown: **18**


---

- **On the Structural Failure of Chamfer Distance in 3D Shape Optimization**  
  Chang-Yong Song, David Hyde  
  _2026-03-10_ · https://arxiv.org/abs/2603.09925v1  
  <details><summary>Abstract</summary>

  Chamfer distance is the standard training loss for point cloud reconstruction, completion, and generation, yet directly optimizing it can produce worse Chamfer values than not optimizing it at all. We show that this paradoxical failure is gradient-structural. The per-point Chamfer gradient creates a many-to-one collapse that is the unique attractor of the forward term and cannot be resolved by any local regularizer, including repulsion, smoothness, and density-aware re-weighting. We derive a necessary condition for collapse suppression: coupling must propagate beyond local neighborhoods. In a controlled 2D setting, shared-basis deformation suppresses collapse by providing global coupling; in 3D shape morphing, a differentiable MPM prior instantiates the same principle, consistently reducing the Chamfer gap across 20 directed pairs with a 2.5$\times$ improvement on the topologically complex dragon. The presence or absence of non-local coupling determines whether Chamfer optimization succeeds or collapses. This provides a practical design criterion for any pipeline that optimizes point-level distance metrics.

  </details>



- **VLM-Loc: Localization in Point Cloud Maps via Vision-Language Models**  
  Shuhao Kang, Youqi Liao, Peijie Wang, Wenlong Liao, Qilin Zhang, Benjamin Busam, Xieyuanli Chen, Yun Liu  
  _2026-03-10_ · https://arxiv.org/abs/2603.09826v1  
  <details><summary>Abstract</summary>

  Text-to-point-cloud (T2P) localization aims to infer precise spatial positions within 3D point cloud maps from natural language descriptions, reflecting how humans perceive and communicate spatial layouts through language. However, existing methods largely rely on shallow text-point cloud correspondence without effective spatial reasoning, limiting their accuracy in complex environments. To address this limitation, we propose VLM-Loc, a framework that leverages the spatial reasoning capability of large vision-language models (VLMs) for T2P localization. Specifically, we transform point clouds into bird's-eye-view (BEV) images and scene graphs that jointly encode geometric and semantic context, providing structured inputs for the VLM to learn cross-modal representations bridging linguistic and spatial semantics. On top of these representations, we introduce a partial node assignment mechanism that explicitly associates textual cues with scene graph nodes, enabling interpretable spatial reasoning for accurate localization. To facilitate systematic evaluation across diverse scenes, we present CityLoc, a benchmark built from multi-source point clouds for fine-grained T2P localization. Experiments on CityLoc demonstrate VLM-Loc achieves superior accuracy and robustness compared to state-of-the-art methods. Our code, model, and dataset are available at \href{https://github.com/MCG-NKU/nku-3d-vision}{repository}.

  </details>



- **ConfCtrl: Enabling Precise Camera Control in Video Diffusion via Confidence-Aware Interpolation**  
  Liudi Yang, George Eskandar, Fengyi Shen, Mohammad Altillawi, Yang Bai, Chi Zhang, Ziyuan Liu, Abhinav Valada  
  _2026-03-10_ · https://arxiv.org/abs/2603.09819v1  
  <details><summary>Abstract</summary>

  We address the challenge of novel view synthesis from only two input images under large viewpoint changes. Existing regression-based methods lack the capacity to reconstruct unseen regions, while camera-guided diffusion models often deviate from intended trajectories due to noisy point cloud projections or insufficient conditioning from camera poses. To address these issues, we propose ConfCtrl, a confidence-aware video interpolation framework that enables diffusion models to follow prescribed camera poses while completing unseen regions. ConfCtrl initializes the diffusion process by combining a confidence-weighted projected point cloud latent with noise as the conditioning input. It then applies a Kalman-inspired predict-update mechanism, treating the projected point cloud as a noisy measurement and using learned residual corrections to balance pose-driven predictions with noisy geometric observations. This allows the model to rely on reliable projections while down-weighting uncertain regions, yielding stable, geometry-aware generation. Experiments on multiple datasets show that ConfCtrl produces geometrically consistent and visually plausible novel views, effectively reconstructing occluded regions under large viewpoint changes.

  </details>



- **DRIFT: Dual-Representation Inter-Fusion Transformer for Automated Driving Perception with 4D Radar Point Clouds**  
  Siqi Pei, Andras Palffy, Dariu M. Gavrila  
  _2026-03-10_ · https://arxiv.org/abs/2603.09695v1  
  <details><summary>Abstract</summary>

  4D radars, which provide 3D point cloud data along with Doppler velocity, are attractive components of modern automated driving systems due to their low cost and robustness under adverse weather conditions. However, they provide a significantly lower point cloud density than LiDAR sensors. This makes it important to exploit not only local but also global contextual scene information. This paper proposes DRIFT, a model that effectively captures and fuses both local and global contexts through a dual-path architecture. The model incorporates a point path to aggregate fine-grained local features and a pillar path to encode coarse-grained global features. These two parallel paths are intertwined via novel feature-sharing layers at multiple stages, enabling full utilization of both representations. DRIFT is evaluated on the widely used View-of-Delft (VoD) dataset and a proprietary internal dataset. It outperforms the baselines on the tasks of object detection and/or free road estimation. For example, DRIFT achieves a mean average precision (mAP) of 52.6\% (compared to, say, 45.4\% of CenterPoint) on the VoD dataset.

  </details>



- **SurgFed: Language-guided Multi-Task Federated Learning for Surgical Video Understanding**  
  Zheng Fang, Ziwei Niu, Ziyue Wang, Zhu Zhuo, Haofeng Liu, Shuyang Qian, Jun Xia, Yueming Jin  
  _2026-03-10_ · https://arxiv.org/abs/2603.09496v1  
  <details><summary>Abstract</summary>

  Surgical scene Multi-Task Federated Learning (MTFL) is essential for robot-assisted minimally invasive surgery (RAS) but remains underexplored in surgical video understanding due to two key challenges: (1) Tissue Diversity: Local models struggle to adapt to site-specific tissue features, limiting their effectiveness in heterogeneous clinical environments and leading to poor local predictions. (2) Task Diversity: Server-side aggregation, relying solely on gradient-based clustering, often produces suboptimal or incorrect parameter updates due to inter-site task heterogeneity, resulting in inaccurate localization. In light of these two issues, we propose SurgFed, a multi-task federated learning framework, enabling federated learning for surgical scene segmentation and depth estimation across diverse surgical types. SurgFed is powered by two appealing designs, i.e., Language-guided Channel Selection (LCS) and Language-guided Hyper Aggregation (LHA), to address the challenge of fully exploration on corss-site and cross-task. Technically, the LCS is first designed a lightweight personalized channel selection network that enhances site-specific adaptation using pre-defined text inputs, which optimally the local model learn the specific embeddings. We further introduce the LHA that employs a layer-wise cross-attention mechanism with pre-defined text inputs to model task interactions across sites and guide a hypernetwork for personalized parameter updates. Extensive empirical evidence shows that SurgFed yields improvements over the state-of-the-art methods in five public datasets across four surgical types. The code is available at https://anonymous.4open.science/r/SurgFed-070E/.

  </details>



- **EventVGGT: Exploring Cross-Modal Distillation for Consistent Event-based Depth Estimation**  
  Yinrui Ren, Jinjing Zhu, Kanghao Chen, Zhuoxiao Li, Jing Ou, Zidong Cao, Tongyan Hua, Peilun Shi, Yingchun Fu, Wufan Zhao, et al.  
  _2026-03-10_ · https://arxiv.org/abs/2603.09385v1  
  <details><summary>Abstract</summary>

  Event cameras offer superior sensitivity to high-speed motion and extreme lighting, making event-based monocular depth estimation a promising approach for robust 3D perception in challenging conditions. However, progress is severely hindered by the scarcity of dense depth annotations. While recent annotation-free approaches mitigate this by distilling knowledge from Vision Foundation Models (VFMs), a critical limitation persists: they process event streams as independent frames. By neglecting the inherent temporal continuity of event data, these methods fail to leverage the rich temporal priors encoded in VFMs, ultimately yielding temporally inconsistent and less accurate depth predictions. To address this, we introduce EventVGGT, a novel framework that explicitly models the event stream as a coherent video sequence. To the best of our knowledge, we are the first to distill spatio-temporal and multi-view geometric priors from the Visual Geometry Grounded Transformer (VGGT) into the event domain. We achieve this via a comprehensive tri-level distillation strategy: (i) Cross-Modal Feature Mixture (CMFM) bridges the modality gap at the output level by fusing RGB and event features to generate auxiliary depth predictions; (ii) Spatio-Temporal Feature Distillation (STFD) distills VGGT's powerful spatio-temporal representations at the feature level; and (iii) Temporal Consistency Distillation (TCD) enforces cross-frame coherence at the temporal level by aligning inter-frame depth changes. Extensive experiments demonstrate that EventVGGT consistently outperforms existing methods -- reducing the absolute mean depth error at 30m by over 53\% on EventScape (from 2.30 to 1.06) -- while exhibiting robust zero-shot generalization on the unseen DENSE and MVSEC datasets.

  </details>



- **SpaceSense-Bench: A Large-Scale Multi-Modal Benchmark for Spacecraft Perception and Pose Estimation**  
  Aodi Wu, Jianhong Zuo, Zeyuan Zhao, Xubo Luo, Ruisuo Wang, Xue Wan  
  _2026-03-10_ · https://arxiv.org/abs/2603.09320v1  
  <details><summary>Abstract</summary>

  Autonomous space operations such as on-orbit servicing and active debris removal demand robust part-level semantic understanding and precise relative navigation of target spacecraft, yet collecting large-scale real data in orbit remains impractical due to cost and access constraints. Existing synthetic datasets, moreover, suffer from limited target diversity, single-modality sensing, and incomplete ground-truth annotations. We present \textbf{SpaceSense-Bench}, a large-scale multi-modal benchmark for spacecraft perception encompassing 136~satellite models with approximately 70~GB of data. Each frame provides time-synchronized 1024$\times$1024 RGB images, millimeter-precision depth maps, and 256-beam LiDAR point clouds, together with dense 7-class part-level semantic labels at both the pixel and point level as well as accurate 6-DoF pose ground truth. The dataset is generated through a high-fidelity space simulation built in Unreal Engine~5 and a fully automated pipeline covering data acquisition, multi-stage quality control, and conversion to mainstream formats. We benchmark five representative tasks (object detection, 2D semantic segmentation, RGB--LiDAR fusion-based 3D point cloud segmentation, monocular depth estimation, and orientation estimation) and identify two key findings: (i)~perceiving small-scale components (\emph{e.g.}, thrusters and omni-antennas) and generalizing to entirely unseen spacecraft in a zero-shot setting remain critical bottlenecks for current methods, and (ii)~scaling up the number of training satellites yields substantial performance gains on novel targets, underscoring the value of large-scale, diverse datasets for space perception research. The dataset, code, and toolkit are publicly available at https://github.com/wuaodi/SpaceSense-Bench.

  </details>



- **NLiPsCalib: An Efficient Calibration Framework for High-Fidelity 3D Reconstruction of Curved Visuotactile Sensors**  
  Xuhao Qin, Feiyu Zhao, Yatao Leng, Runze Hu, Chenxi Xiao  
  _2026-03-10_ · https://arxiv.org/abs/2603.09319v1  
  <details><summary>Abstract</summary>

  Recent advances in visuotactile sensors increasingly employ biomimetic curved surfaces to enhance sensorimotor capabilities. Although such curved visuotactile sensors enable more conformal object contact, their perceptual quality is often degraded by non-uniform illumination, which reduces reconstruction accuracy and typically necessitates calibration. Existing calibration methods commonly rely on customized indenters and specialized devices to collect large-scale photometric data, but these processes are expensive and labor-intensive. To overcome these calibration challenges, we present NLiPsCalib, a physics-consistent and efficient calibration framework for curved visuotactile sensors. NLiPsCalib integrates controllable near-field light sources and leverages Near-Light Photometric Stereo (NLiPs) to estimate contact geometry, simplifying calibration to just a few simple contacts with everyday objects. We further introduce NLiPsTac, a controllable-light-source tactile sensor developed to validate our framework. Experimental results demonstrate that our approach enables high-fidelity 3D reconstruction across diverse curved form factors with a simple calibration procedure. We emphasize that our approach lowers the barrier to developing customized visuotactile sensors of diverse geometries, thereby making visuotactile sensing more accessible to the broader community.

  </details>



- **Implicit Geometry Representations for Vision-and-Language Navigation from Web Videos**  
  Mingfei Han, Haihong Hao, Liang Ma, Kamila Zhumakhanova, Ekaterina Radionova, Jingyi Zhang, Xiaojun Chang, Xiaodan Liang, Ivan Laptev  
  _2026-03-10_ · https://arxiv.org/abs/2603.09259v1  
  <details><summary>Abstract</summary>

  Vision-and-Language Navigation (VLN) has long been constrained by the limited diversity and scalability of simulator-curated datasets, which fail to capture the complexity of real-world environments. To overcome this limitation, we introduce a large-scale video-instruction framework derived from web-based room tour videos, enabling agents to learn from natural human walking demonstrations in diverse, realistic indoor settings. Unlike existing datasets, our framework integrates both open-ended description-enriched trajectories and action-enriched trajectories reconstructed in 3D, providing richer spatial and semantic supervision. A key extension in this work is the incorporation of implicit geometry representations, which extract spatial cues directly from RGB frames without requiring fragile 3D reconstruction. This approach substantially improves data utilization, alleviates reconstruction failures, and unlocks large portions of previously unusable video data. Comprehensive experiments across multiple VLN benchmarks (CVDN, SOON, R2R, and REVERIE) demonstrate that our method not only sets new state-of-the-art performance but also enables the development of robust zero-shot navigation agents. By bridging large-scale web videos with implicit spatial reasoning, this work advances embodied navigation towards more scalable, generalizable, and real-world applicable solutions.

  </details>



- **Point Cloud as a Foreign Language for Multi-modal Large Language Model**  
  Sneha Paul, Zachary Patterson, Nizar Bouguila  
  _2026-03-10_ · https://arxiv.org/abs/2603.09173v1  
  <details><summary>Abstract</summary>

  Multi-modal large language models (MLLMs) have shown remarkable progress in integrating visual and linguistic understanding. Recent efforts have extended these capabilities to 3D understanding through encoder-based architectures that rely on pre-trained 3D encoders to extract geometric features. However, such approaches suffer from semantic misalignment between geometric and linguistic spaces, resolution sensitivity, and substantial computational overhead. In this work, we present SAGE, the first end-to-end 3D MLLM that directly processes raw point clouds without relying on a pre-trained 3D encoder. Our approach introduces a lightweight 3D tokenizer that combines geometric sampling and neighbourhood aggregation with vector quantization to convert point clouds into discrete tokens--treating 3D data as a foreign language that naturally extends the LLM's vocabulary. Furthermore, to enhance the model's reasoning capability on complex 3D tasks, we propose a preference optimization training strategy with a semantic alignment-based reward, specifically designed for open-ended 3D question answering where responses are descriptive. Extensive experiments across diverse 3D understanding benchmarks demonstrate that our end-to-end approach outperforms existing encoder-based methods while offering significant advantages in computational efficiency, generalization across LLM backbones, and robustness to input resolution variations. Code is available at: github.com/snehaputul/SAGE3D.

  </details>



- **mmGAT: Pose Estimation by Graph Attention with Mutual Features from mmWave Radar Point Cloud**  
  Abdullah Al Masud, Shi Xintong, Mondher Bouazizi, Ohtsuki Tomoaki  
  _2026-03-09_ · https://arxiv.org/abs/2603.08551v1  
  <details><summary>Abstract</summary>

  Pose estimation and human action recognition (HAR) are pivotal technologies spanning various domains. While the image-based pose estimation and HAR are widely admired for their superior performance, they lack in privacy protection and suboptimal performance in low-light and dark environments. This paper exploits the capabilities of millimeter-wave (mmWave) radar technology for human pose estimation by processing radar data with Graph Neural Network (GNN) architecture, coupled with the attention mechanism. Our goal is to capture the finer details of the radar point cloud to improve the pose estimation performance. To this end, we present a unique feature extraction technique that exploits the full potential of the GNN processing method for pose estimation. Our model mmGAT demonstrates remarkable performance on two publicly available benchmark mmWave datasets and establishes new state of the art results in most scenarios in terms of human pose estimation. Our approach achieves a noteworthy reduction of pose estimation mean per joint position error (MPJPE) by 35.6% and PA-MPJPE by 14.1% from the current state of the art benchmark within this domain.

  </details>



- **PCFEx: Point Cloud Feature Extraction for Graph Neural Networks**  
  Abdullah Al Masud, Shi Xintong, Mondher Bouazizi, Ohtsuki Tomoaki  
  _2026-03-09_ · https://arxiv.org/abs/2603.08540v1  
  <details><summary>Abstract</summary>

  Graph neural networks (GNNs) have gained significant attention for their effectiveness across various domains. This study focuses on applying GNN to process 3D point cloud data for human pose estimation (HPE) and human activity recognition (HAR). We propose novel point cloud feature extraction (PCFEx) techniques to capture meaningful information at the point, edge, and graph levels of the point cloud by considering point cloud as a graph. Moreover, we introduce a GNN architecture designed to efficiently process these features. Our approach is evaluated on four most popular publicly available millimeter wave radar datasets, three for HPE and one for HAR. The results show substantial improvements, with significantly reduced errors in all three HPE benchmarks, and an overall accuracy of 98.8% in mmWave-based HAR, outperforming the existing state of the art models. This work demonstrates the great potential of feature extraction incorporated with GNN modeling approach to enhance the precision of point cloud processing.

  </details>



- **OSCAR: Occupancy-based Shape Completion via Acoustic Neural Implicit Representations**  
  Magdalena Wysocki, Kadir Burak Buldu, Miruna-Alexandra Gafencu, Mohammad Farid Azampour, Nassir Navab  
  _2026-03-09_ · https://arxiv.org/abs/2603.08279v1  
  <details><summary>Abstract</summary>

  Accurate 3D reconstruction of vertebral anatomy from ultrasound is important for guiding minimally invasive spine interventions, but it remains challenging due to acoustic shadowing and view-dependent signal variations. We propose an occupancy-based shape completion method that reconstructs complete 3D anatomical geometry from partial ultrasound observations. Crucially for intra-operative applications, our approach extracts the anatomical surface directly from the image, avoiding the need for anatomical labels during inference. This label-free completion relies on a coupled latent space representing both the image appearance and the underlying anatomical shape. By leveraging a Neural Implicit Representation (NIR) that jointly models both spatial occupancy and acoustic interactions, the method uses acoustic parameters to become implicitly aware of the unseen regions without explicit shadowing labels through tracking acoustic signal transmission. We show that this method outperforms state-of-the-art shape completion for B-mode ultrasound by 80% in HD95 score. We validate our approach both in-silico and on phantom US images with registered mesh models from CT labels, demonstrating accurate reconstruction of occluded anatomy and robust generalization across diverse imaging conditions. Code and data will be released on publication.

  </details>



- **Topologically Stable Hough Transform**  
  Stefan Huber, Kristóf Huszár, Michael Kerber, Martin Uray  
  _2026-03-09_ · https://arxiv.org/abs/2603.08245v1  
  <details><summary>Abstract</summary>

  We propose an alternative formulation of the well-known Hough transform to detect lines in point clouds. Replacing the discretized voting scheme of the classical Hough transform by a continuous score function, its persistent features in the sense of persistent homology give a set of candidate lines. We also devise and implement an algorithm to efficiently compute these candidate lines.

  </details>



- **MV-Fashion: Towards Enabling Virtual Try-On and Size Estimation with Multi-View Paired Data**  
  Hunor Laczkó, Libang Jia, Loc-Phat Truong, Diego Hernández, Sergio Escalera, Jordi Gonzalez, Meysam Madadi  
  _2026-03-09_ · https://arxiv.org/abs/2603.08147v1  
  <details><summary>Abstract</summary>

  Existing 4D human datasets fall short for fashion-specific research, lacking either realistic garment dynamics or task-specific annotations. Synthetic datasets suffer from a realism gap, whereas real-world captures lack the detailed annotations and paired data required for virtual try-on (VTON) and size estimation tasks. To bridge this gap, we introduce MV-Fashion, a large-scale, multi-view video dataset engineered for domain-specific fashion analysis. MV-Fashion features 3,273 sequences (72.5 million frames) from 80 diverse subjects wearing 3-10 outfits each. It is designed to capture complex, real-world garment dynamics, including multiple layers and varied styling (e.g. rolled sleeves, tucked shirt). A core contribution is a rich data representation that includes pixel-level semantic annotations, ground-truth material properties like elasticity, and 3D point clouds. Crucially for VTON applications, MV-Fashion provides paired data: multi-view synchronized captures of worn garments alongside their corresponding flat, catalogue images. We leverage this dataset to establish baselines for fashion-centric tasks, including virtual try-on, clothing size estimation, and novel view synthesis. The dataset is available at https://hunorlaczko.github.io/MV-Fashion .

  </details>



- **Speed3R: Sparse Feed-forward 3D Reconstruction Models**  
  Weining Ren, Xiao Tan, Kai Han  
  _2026-03-09_ · https://arxiv.org/abs/2603.08055v1  
  <details><summary>Abstract</summary>

  While recent feed-forward 3D reconstruction models accelerate 3D reconstruction by jointly inferring dense geometry and camera poses in a single pass, their reliance on dense attention imposes a quadratic complexity, creating a prohibitive computational bottleneck that severely limits inference speed. To resolve this, we introduce Speed3R, an end-to-end trainable model inspired by the core principle of Structure-from-Motion: that a sparse set of keypoints is sufficient for robust pose estimation. Speed3R features a dual-branch attention mechanism where a compression branch creates a coarse contextual prior to guide a selection branch, which performs fine-grained attention only on the most informative image tokens. This strategy mimics the efficiency of traditional keypoint matching, achieving a remarkable 12.4x inference speedup on 1000-view sequences, while introducing a minimal, controlled trade-off in geometric accuracy. Validated on standard benchmarks with both VGGT and $π^3$ backbones, our method delivers high-quality reconstructions at a fraction of computational cost, paving the way for efficient large-scale scene modeling.

  </details>



- **$L^3$:Scene-agnostic Visual Localization in the Wild**  
  Yu Zhang, Muhua Zhu, Yifei Xue, Tie Ji, Yizhen Lao  
  _2026-03-09_ · https://arxiv.org/abs/2603.07937v1  
  <details><summary>Abstract</summary>

  Standard visual localization methods typically require offline pre-processing of scenes to obtain 3D structural information for better performance. This inevitably introduces additional computational and time costs, as well as the overhead of storing scene representations. Can we visually localize in a wild scene without any off-line preprocessing step? In this paper, we leverage the online inference capabilities of feed-forward 3D reconstruction networks to propose a novel map-free visual localization framework $L^3$. Specifically, by performing direct online 3D reconstruction on RGB images, followed by two-stage metric scale recovery and pose refinement based on 2D-3D correspondences, $L^3$ achieves high accuracy without the need to pre-build or store any offline scene representations. Extensive experiments demonstrate $L^3$ not only that the performance is comparable to state-of-the-art solutions on various benchmarks, but also that it exhibits significantly superior robustness in sparse scenes (fewer reference images per scene).

  </details>



- **Toward Unified Multimodal Representation Learning for Autonomous Driving**  
  Ximeng Tao, Dimitar Filev, Gaurav Pandey  
  _2026-03-09_ · https://arxiv.org/abs/2603.07874v1  
  <details><summary>Abstract</summary>

  Contrastive Language-Image Pre-training (CLIP) has shown impressive performance in aligning visual and textual representations. Recent studies have extended this paradigm to 3D vision to improve scene understanding for autonomous driving. A common strategy is to employ pairwise cosine similarity between modalities to guide the training of a 3D encoder. However, considering the similarity between individual modality pairs rather than all modalities jointly fails to ensure consistent and unified alignment across the entire multimodal space. In this paper, we propose a Contrastive Tensor Pre-training (CTP) framework that simultaneously aligns multiple modalities in a unified embedding space to enhance end-to-end autonomous driving. Compared with pairwise cosine similarity alignment, our method extends the 2D similarity matrix into a multimodal similarity tensor. Furthermore, we introduce a tensor loss to enable joint contrastive learning across all modalities. For experimental validation of our framework, we construct a text-image-point cloud triplet dataset derived from existing autonomous driving datasets. The results show that our proposed unified multimodal alignment framework achieves favorable performance for both scenarios: (i) aligning a 3D encoder with pretrained CLIP encoders, and (ii) pretraining all encoders from scratch.

  </details>


