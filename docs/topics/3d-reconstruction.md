# 3D Reconstruction

_Updated: 2026-03-06 07:06 UTC_

Total papers shown: **24**


---

- **FaceCam: Portrait Video Camera Control via Scale-Aware Conditioning**  
  Weijie Lyu, Ming-Hsuan Yang, Zhixin Shu  
  _2026-03-05_ · https://arxiv.org/abs/2603.05506v1  
  <details><summary>Abstract</summary>

  We introduce FaceCam, a system that generates video under customizable camera trajectories for monocular human portrait video input. Recent camera control approaches based on large video-generation models have shown promising progress but often exhibit geometric distortions and visual artifacts on portrait videos due to scale-ambiguous camera representations or 3D reconstruction errors. To overcome these limitations, we propose a face-tailored scale-aware representation for camera transformations that provides deterministic conditioning without relying on 3D priors. We train a video generation model on both multi-view studio captures and in-the-wild monocular videos, and introduce two camera-control data generation strategies: synthetic camera motion and multi-shot stitching, to exploit stationary training cameras while generalizing to dynamic, continuous camera trajectories at inference time. Experiments on Ava-256 dataset and diverse in-the-wild videos demonstrate that FaceCam achieves superior performance in camera controllability, visual quality, identity and motion preservation.

  </details>



- **RealWonder: Real-Time Physical Action-Conditioned Video Generation**  
  Wei Liu, Ziyu Chen, Zizhang Li, Yue Wang, Hong-Xing Yu, Jiajun Wu  
  _2026-03-05_ · https://arxiv.org/abs/2603.05449v1  
  <details><summary>Abstract</summary>

  Current video generation models cannot simulate physical consequences of 3D actions like forces and robotic manipulations, as they lack structural understanding of how actions affect 3D scenes. We present RealWonder, the first real-time system for action-conditioned video generation from a single image. Our key insight is using physics simulation as an intermediate bridge: instead of directly encoding continuous actions, we translate them through physics simulation into visual representations (optical flow and RGB) that video models can process. RealWonder integrates three components: 3D reconstruction from single images, physics simulation, and a distilled video generator requiring only 4 diffusion steps. Our system achieves 13.2 FPS at 480x832 resolution, enabling interactive exploration of forces, robot actions, and camera controls on rigid objects, deformable bodies, fluids, and granular materials. We envision RealWonder opens new opportunities to apply video models in immersive experiences, AR/VR, and robot learning. Our code and model weights are publicly available in our project website: https://liuwei283.github.io/RealWonder/

  </details>



- **OpenFrontier: General Navigation with Visual-Language Grounded Frontiers**  
  Esteban Padilla, Boyang Sun, Marc Pollefeys, Hermann Blum  
  _2026-03-05_ · https://arxiv.org/abs/2603.05377v1  
  <details><summary>Abstract</summary>

  Open-world navigation requires robots to make decisions in complex everyday environments while adapting to flexible task requirements. Conventional navigation approaches often rely on dense 3D reconstruction and hand-crafted goal metrics, which limits their generalization across tasks and environments. Recent advances in vision--language navigation (VLN) and vision--language--action (VLA) models enable end-to-end policies conditioned on natural language, but typically require interactive training, large-scale data collection, or task-specific fine-tuning with a mobile agent. We formulate navigation as a sparse subgoal identification and reaching problem and observe that providing visual anchoring targets for high-level semantic priors enables highly efficient goal-conditioned navigation. Based on this insight, we select navigation frontiers as semantic anchors and propose OpenFrontier, a training-free navigation framework that seamlessly integrates diverse vision--language prior models. OpenFrontier enables efficient navigation with a lightweight system design, without dense 3D mapping, policy training, or model fine-tuning. We evaluate OpenFrontier across multiple navigation benchmarks and demonstrate strong zero-shot performance, as well as effective real-world deployment on a mobile robot.

  </details>



- **Dark3R: Learning Structure from Motion in the Dark**  
  Andrew Y Guo, Anagh Malik, SaiKiran Tedla, Yutong Dai, Yiqian Qin, Zach Salehe, Benjamin Attal, Sotiris Nousias, Kyros Kutulakos, David B. Lindell  
  _2026-03-05_ · https://arxiv.org/abs/2603.05330v1  
  <details><summary>Abstract</summary>

  We introduce Dark3R, a framework for structure from motion in the dark that operates directly on raw images with signal-to-noise ratios (SNRs) below $-4$ dB -- a regime where conventional feature- and learning-based methods break down. Our key insight is to adapt large-scale 3D foundation models to extreme low-light conditions through a teacher--student distillation process, enabling robust feature matching and camera pose estimation in low light. Dark3R requires no 3D supervision; it is trained solely on noisy--clean raw image pairs, which can be either captured directly or synthesized using a simple Poisson--Gaussian noise model applied to well-exposed raw images. To train and evaluate our approach, we introduce a new, exposure-bracketed dataset that includes $\sim$42,000 multi-view raw images with ground-truth 3D annotations, and we demonstrate that Dark3R achieves state-of-the-art structure from motion in the low-SNR regime. Further, we demonstrate state-of-the-art novel view synthesis in the dark using Dark3R's predicted poses and a coarse-to-fine radiance field optimization procedure.

  </details>



- **Digital Twin Driven Textile Classification and Foreign Object Recognition in Automated Sorting Systems**  
  Serkan Ergun, Tobias Mitterer, Hubert Zangl  
  _2026-03-05_ · https://arxiv.org/abs/2603.05230v1  
  <details><summary>Abstract</summary>

  The increasing demand for sustainable textile recycling requires robust automation solutions capable of handling deformable garments and detecting foreign objects in cluttered environments. This work presents a digital twin driven robotic sorting system that integrates grasp prediction, multi modal perception, and semantic reasoning for real world textile classification. A dual arm robotic cell equipped with RGBD sensing, capacitive tactile feedback, and collision-aware motion planning autonomously separates garments from an unsorted basket, transfers them to an inspection zone, and classifies them using state of the art Visual Language Models (VLMs). We benchmark nine VLM s from five model families on a dataset of 223 inspection scenarios comprising shirts, socks, trousers, underwear, foreign objects (including garments outside of the aforementioned classes), and empty scenes. The evaluation assesses per class accuracy, hallucination behavior, and computational performance under practical hardware constraints. Results show that the Qwen model family achieves the highest overall accuracy (up to 87.9 %), with strong foreign object detection performance, while lighter models such as Gemma3 offer competitive speed accuracy trade offs for edge deployment. A digital twin combined with MoveIt enables collision aware path planning and integrates segmented 3D point clouds of inspected garments into the virtual environment for improved manipulation reliability. The presented system demonstrates the feasibility of combining semantic VLM reasoning with conventional grasp detection and digital twin technology for scalable, autonomous textile sorting in realistic industrial settings.

  </details>



- **SSR-GS: Separating Specular Reflection in Gaussian Splatting for Glossy Surface Reconstruction**  
  Ningjing Fan, Yiqun Wang  
  _2026-03-05_ · https://arxiv.org/abs/2603.05152v1  
  <details><summary>Abstract</summary>

  In recent years, 3D Gaussian splatting (3DGS) has achieved remarkable progress in novel view synthesis. However, accurately reconstructing glossy surfaces under complex illumination remains challenging, particularly in scenes with strong specular reflections and multi-surface interreflections. To address this issue, we propose SSR-GS, a specular reflection modeling framework for glossy surface reconstruction. Specifically, we introduce a prefiltered Mip-Cubemap to model direct specular reflections efficiently, and propose an IndiASG module to capture indirect specular reflections. Furthermore, we design Visual Geometry Priors (VGP) that couple a reflection-aware visual prior via a reflection score (RS) to downweight the photometric loss contribution of reflection-dominated regions, with geometry priors derived from VGGT, including progressively decayed depth supervision and transformed normal constraints. Extensive experiments on both synthetic and real-world datasets demonstrate that SSR-GS achieves state-of-the-art performance in glossy surface reconstruction.

  </details>



- **CoIn3D: Revisiting Configuration-Invariant Multi-Camera 3D Object Detection**  
  Zhaonian Kuang, Rui Ding, Haotian Wang, Xinhu Zheng, Meng Yang, Gang Hua  
  _2026-03-05_ · https://arxiv.org/abs/2603.05042v1  
  <details><summary>Abstract</summary>

  Multi-camera 3D object detection (MC3D) has attracted increasing attention with the growing deployment of multi-sensor physical agents, such as robots and autonomous vehicles. However, MC3D models still struggle to generalize to unseen platforms with new multi-camera configurations. Current solutions simply employ a meta-camera for unified representation but lack comprehensive consideration. In this paper, we revisit this issue and identify that the devil lies in spatial prior discrepancies across source and target configurations, including different intrinsics, extrinsics, and array layouts. To address this, we propose CoIn3D, a generalizable MC3D framework that enables strong transferability from source configurations to unseen target ones. CoIn3D explicitly incorporates all identified spatial priors into both feature embedding and image observation through spatial-aware feature modulation (SFM) and camera-aware data augmentation (CDA), respectively. SFM enriches feature space by integrating four spatial representations, such as focal length, ground depth, ground gradient, and Plücker coordinate. CDA improves observation diversity under various configurations via a training-free dynamic novel-view image synthesis scheme. Extensive experiments demonstrate that CoIn3D achieves strong cross-configuration performance on landmark datasets such as NuScenes, Waymo, and Lyft, under three dominant MC3D paradigms represented by BEVDepth, BEVFormer, and PETR.

  </details>



- **GloSplat: Joint Pose-Appearance Optimization for Faster and More Accurate 3D Reconstruction**  
  Tianyu Xiong, Rui Li, Linjie Li, Jiaqi Yang  
  _2026-03-05_ · https://arxiv.org/abs/2603.04847v1  
  <details><summary>Abstract</summary>

  Feature extraction, matching, structure from motion (SfM), and novel view synthesis (NVS) have traditionally been treated as separate problems with independent optimization objectives. We present GloSplat, a framework that performs \emph{joint pose-appearance optimization} during 3D Gaussian Splatting training. Unlike prior joint optimization methods (BARF, NeRF--, 3RGS) that rely purely on photometric gradients for pose refinement, GloSplat preserves \emph{explicit SfM feature tracks} as first-class entities throughout training: track 3D points are maintained as separate optimizable parameters from Gaussian primitives, providing persistent geometric anchors via a reprojection loss that operates alongside photometric supervision. This architectural choice prevents early-stage pose drift while enabling fine-grained refinement -- a capability absent in photometric-only approaches. We introduce two pipeline variants: (1) \textbf{GloSplat-F}, a COLMAP-free variant using retrieval-based pair selection for efficient reconstruction, and (2) \textbf{GloSplat-A}, an exhaustive matching variant for maximum quality. Both employ global SfM initialization followed by joint photometric-geometric optimization during 3DGS training. Experiments demonstrate that GloSplat-F achieves state-of-the-art among COLMAP-free methods while GloSplat-A surpasses all COLMAP-based baselines.

  </details>



- **MADCrowner: Margin Aware Dental Crown Design with Template Deformation and Refinement**  
  Linda Wei, Chang Liu, Wenran Zhang, Yuxuan Hu, Ruiyang Li, Feng Qi, Changyao Tian, Ke Wang, Yuanyuan Wang, Shaoting Zhang, et al.  
  _2026-03-05_ · https://arxiv.org/abs/2603.04771v1  
  <details><summary>Abstract</summary>

  Dental crown restoration is one of the most common treatment modalities for tooth defect, where personalized dental crown design is critical. While computer-aided design (CAD) systems have notably enhanced the efficiency of dental crown design, extensive manual adjustments are still required in the clinic workflow. Recent studies have explored the application of learning-based methods for the automated generation of restorative dental crowns. Nevertheless, these approaches were challenged by inadequate spatial resolution, noisy outputs, and overextension of surface reconstruction. To address these limitations, we propose \totalframework, a margin-aware mesh generation framework comprising CrownDeformR and CrownSegger. Inspired by the clinic manual workflow of dental crown design, we designed CrownDeformR to deform an initial template to the target crown based on anatomical context, which is extracted by a multi-scale intraoral scan encoder. Additionally, we introduced \marginseg, a novel margin segmentation network, to extract the cervical margin of the target tooth. The performance of CrownDeformR improved with the cervical margin as an extra constraint. And it was also utilized as the boundary condition for the tailored postprocessing method, which removed the overextended area of the reconstructed surface. We constructed a large-scale intraoral scan dataset and performed extensive experiments. The proposed method significantly outperformed existing approaches in both geometric accuracy and clinical feasibility.

  </details>



- **SGR3 Model: Scene Graph Retrieval-Reasoning Model in 3D**  
  Zirui Wang, Ruiping Liu, Yufan Chen, Junwei Zheng, Weijia Fan, Kunyu Peng, Di Wen, Jiale Wei, Jiaming Zhang, Rainer Stiefelhagen  
  _2026-03-04_ · https://arxiv.org/abs/2603.04614v1  
  <details><summary>Abstract</summary>

  3D scene graphs provide a structured representation of object entities and their relationships, enabling high-level interpretation and reasoning for robots while remaining intuitively understandable to humans. Existing approaches for 3D scene graph generation typically combine scene reconstruction with graph neural networks (GNNs). However, such pipelines require multi-modal data that may not always be available, and their reliance on heuristic graph construction can constrain the prediction of relationship triplets. In this work, we introduce a Scene Graph Retrieval-Reasoning Model in 3D (SGR3 Model), a training-free framework that leverages multi-modal large language models (MLLMs) with retrieval-augmented generation (RAG) for semantic scene graph generation. SGR3 Model bypasses the need for explicit 3D reconstruction. Instead, it enhances relational reasoning by incorporating semantically aligned scene graphs retrieved via a ColPali-style cross-modal framework. To improve retrieval robustness, we further introduce a weighted patch-level similarity selection mechanism that mitigates the negative impact of blurry or semantically uninformative regions. Experiments demonstrate that SGR3 Model achieves competitive performance compared to training-free baselines and on par with GNN-based expert models. Moreover, an ablation study on the retrieval module and knowledge base scale reveals that retrieved external information is explicitly integrated into the token generation process, rather than being implicitly internalized through abstraction.

  </details>



- **ZipMap: Linear-Time Stateful 3D Reconstruction with Test-Time Training**  
  Haian Jin, Rundi Wu, Tianyuan Zhang, Ruiqi Gao, Jonathan T. Barron, Noah Snavely, Aleksander Holynski  
  _2026-03-04_ · https://arxiv.org/abs/2603.04385v1  
  <details><summary>Abstract</summary>

  Feed-forward transformer models have driven rapid progress in 3D vision, but state-of-the-art methods such as VGGT and $π^3$ have a computational cost that scales quadratically with the number of input images, making them inefficient when applied to large image collections. Sequential-reconstruction approaches reduce this cost but sacrifice reconstruction quality. We introduce ZipMap, a stateful feed-forward model that achieves linear-time, bidirectional 3D reconstruction while matching or surpassing the accuracy of quadratic-time methods. ZipMap employs test-time training layers to zip an entire image collection into a compact hidden scene state in a single forward pass, enabling reconstruction of over 700 frames in under 10 seconds on a single H100 GPU, more than $20\times$ faster than state-of-the-art methods such as VGGT. Moreover, we demonstrate the benefits of having a stateful representation in real-time scene-state querying and its extension to sequential streaming reconstruction.

  </details>



- **EmbodiedSplat: Online Feed-Forward Semantic 3DGS for Open-Vocabulary 3D Scene Understanding**  
  Seungjun Lee, Zihan Wang, Yunsong Wang, Gim Hee Lee  
  _2026-03-04_ · https://arxiv.org/abs/2603.04254v1  
  <details><summary>Abstract</summary>

  Understanding a 3D scene immediately with its exploration is essential for embodied tasks, where an agent must construct and comprehend the 3D scene in an online and nearly real-time manner. In this study, we propose EmbodiedSplat, an online feed-forward 3DGS for open-vocabulary scene understanding that enables simultaneous online 3D reconstruction and 3D semantic understanding from the streaming images. Unlike existing open-vocabulary 3DGS methods which are typically restricted to either offline or per-scene optimization setting, our objectives are two-fold: 1) Reconstructs the semantic-embedded 3DGS of the entire scene from over 300 streaming images in an online manner. 2) Highly generalizable to novel scenes with feed-forward design and supports nearly real-time 3D semantic reconstruction when combined with real-time 2D models. To achieve these objectives, we propose an Online Sparse Coefficients Field with a CLIP Global Codebook where it binds the 2D CLIP embeddings to each 3D Gaussian while minimizing memory consumption and preserving the full semantic generalizability of CLIP. Furthermore, we generate 3D geometric-aware CLIP features by aggregating the partial point cloud of 3DGS through 3D U-Net to compensate the 3D geometric prior to 2D-oriented language embeddings. Extensive experiments on diverse indoor datasets, including ScanNet, ScanNet++, and Replica, demonstrate both the effectiveness and efficiency of our method. Check out our project page in https://0nandon.github.io/EmbodiedSplat/.

  </details>



- **NOVA3R: Non-pixel-aligned Visual Transformer for Amodal 3D Reconstruction**  
  Weirong Chen, Chuanxia Zheng, Ganlin Zhang, Andrea Vedaldi, Daniel Cremers  
  _2026-03-04_ · https://arxiv.org/abs/2603.04179v2  
  <details><summary>Abstract</summary>

  We present NOVA3R, an effective approach for non-pixel-aligned 3D reconstruction from a set of unposed images in a feed-forward manner. Unlike pixel-aligned methods that tie geometry to per-ray predictions, our formulation learns a global, view-agnostic scene representation that decouples reconstruction from pixel alignment. This addresses two key limitations in pixel-aligned 3D: (1) it recovers both visible and invisible points with a complete scene representation, and (2) it produces physically plausible geometry with fewer duplicated structures in overlapping regions. To achieve this, we introduce a scene-token mechanism that aggregates information across unposed images and a diffusion-based 3D decoder that reconstructs complete, non-pixel-aligned point clouds. Extensive experiments on both scene-level and object-level datasets demonstrate that NOVA3R outperforms state-of-the-art methods in terms of reconstruction accuracy and completeness.

  </details>



- **Efficient Point Cloud Processing with High-Dimensional Positional Encoding and Non-Local MLPs**  
  Yanmei Zou, Hongshan Yu, Yaonan Wang, Zhengeng Yang, Xieyuanli Chen, Kailun Yang, Naveed Akhtar  
  _2026-03-04_ · https://arxiv.org/abs/2603.04099v1  
  <details><summary>Abstract</summary>

  Multi-Layer Perceptron (MLP) models are the foundation of contemporary point cloud processing. However, their complex network architectures obscure the source of their strength and limit the application of these models. In this article, we develop a two-stage abstraction and refinement (ABS-REF) view for modular feature extraction in point cloud processing. This view elucidates that whereas the early models focused on ABS stages, the more recent techniques devise sophisticated REF stages to attain performance advantages. Then, we propose a High-dimensional Positional Encoding (HPE) module to explicitly utilize intrinsic positional information, extending the ``positional encoding'' concept from Transformer literature. HPE can be readily deployed in MLP-based architectures and is compatible with transformer-based methods. Within our ABS-REF view, we rethink local aggregation in MLP-based methods and propose replacing time-consuming local MLP operations, which are used to capture local relationships among neighbors. Instead, we use non-local MLPs for efficient non-local information updates, combined with the proposed HPE for effective local information representation. We leverage our modules to develop HPENets, a suite of MLP networks that follow the ABS-REF paradigm, incorporating a scalable HPE-based REF stage. Extensive experiments on seven public datasets across four different tasks show that HPENets deliver a strong balance between efficiency and effectiveness. Notably, HPENet surpasses PointNeXt, a strong MLP-based counterpart, by 1.1% mAcc, 4.0% mIoU, 1.8% mIoU, and 0.2% Cls. mIoU, with only 50.0%, 21.5%, 23.1%, 44.4% of FLOPs on ScanObjectNN, S3DIS, ScanNet, and ShapeNetPart, respectively. Source code is available at https://github.com/zouyanmei/HPENet_v2.git.

  </details>



- **Structural Action Transformer for 3D Dexterous Manipulation**  
  Xiaohan Lei, Min Wang, Bohong Weng, Wengang Zhou, Houqiang Li  
  _2026-03-04_ · https://arxiv.org/abs/2603.03960v1  
  <details><summary>Abstract</summary>

  Achieving human-level dexterity in robots via imitation learning from heterogeneous datasets is hindered by the challenge of cross-embodiment skill transfer, particularly for high-DoF robotic hands. Existing methods, often relying on 2D observations and temporal-centric action representation, struggle to capture 3D spatial relations and fail to handle embodiment heterogeneity. This paper proposes the Structural Action Transformer (SAT), a new 3D dexterous manipulation policy that challenges this paradigm by introducing a structural-centric perspective. We reframe each action chunk not as a temporal sequence, but as a variable-length, unordered sequence of joint-wise trajectories. This structural formulation allows a Transformer to natively handle heterogeneous embodiments, treating the joint count as a variable sequence length. To encode structural priors and resolve ambiguity, we introduce an Embodied Joint Codebook that embeds each joint's functional role and kinematic properties. Our model learns to generate these trajectories from 3D point clouds via a continuous-time flow matching objective. We validate our approach by pre-training on large-scale heterogeneous datasets and fine-tuning on simulation and real-world dexterous manipulation tasks. Our method consistently outperforms all baselines, demonstrating superior sample efficiency and effective cross-embodiment skill transfer. This structural-centric representation offers a new path toward scaling policies for high-DoF, heterogeneous manipulators.

  </details>



- **A novel network for classification of cuneiform tablet metadata**  
  Frederik Hagelskjær  
  _2026-03-04_ · https://arxiv.org/abs/2603.03892v1  
  <details><summary>Abstract</summary>

  In this paper, we present a network structure for classifying metadata of cuneiform tablets. The problem is of practical importance, as the size of the existing corpus far exceeds the number of experts available to analyze it. But the task is made difficult by the combination of limited annotated datasets and the high-resolution point-cloud representation of each tablet. To address this, we develop a convolution-inspired architecture that gradually down-scales the point cloud while integrating local neighbor information. The final down-scaled point cloud is then processed by computing neighbors in the feature space to include global information. Our method is compared with the state-of-the-art transformer-based network Point-BERT, and consistently obtains the best performance. Source code and datasets will be released at publication.

  </details>



- **LiDAR Prompted Spatio-Temporal Multi-View Stereo for Autonomous Driving**  
  Qihao Sun, Jiarun Liu, Ziqian Ni, Jianyun Xu, Tao Xie, Lijun Zhao, Ruifeng Li, Sheng Yang  
  _2026-03-04_ · https://arxiv.org/abs/2603.03765v1  
  <details><summary>Abstract</summary>

  Accurate metric depth is critical for autonomous driving perception and simulation, yet current approaches struggle to achieve high metric accuracy, multi-view and temporal consistency, and cross-domain generalization. To address these challenges, we present DriveMVS, a novel multi-view stereo framework that reconciles these competing objectives through two key insights: (1) Sparse but metrically accurate LiDAR observations can serve as geometric prompts to anchor depth estimation in absolute scale, and (2) deep fusion of diverse cues is essential for resolving ambiguities and enhancing robustness, while a spatio-temporal decoder ensures consistency across frames. Built upon these principles, DriveMVS embeds the LiDAR prompt in two ways: as a hard geometric prior that anchors the cost volume, and as soft feature-wise guidance fused by a triple-cue combiner. Regarding temporal consistency, DriveMVS employs a spatio-temporal decoder that jointly leverages geometric cues from the MVS cost volume and temporal context from neighboring frames. Experiments show that DriveMVS achieves state-of-the-art performance on multiple benchmarks, excelling in metric accuracy, temporal stability, and zero-shot cross-domain transfer, demonstrating its practical value for scalable, reliable autonomous driving systems.

  </details>



- **QD-PCQA: Quality-Aware Domain Adaptation for Point Cloud Quality Assessment**  
  Guohua Zhang, Jian Jin, Meiqin Liu, Chao Yao, Weisi Lin  
  _2026-03-04_ · https://arxiv.org/abs/2603.03726v1  
  <details><summary>Abstract</summary>

  No-Reference Point Cloud Quality Assessment (NR-PCQA) still struggles with generalization, primarily due to the scarcity of annotated point cloud datasets. Since the Human Visual System (HVS) drives perceptual quality assessment independently of media types, prior knowledge on quality learned from images can be repurposed for point clouds. This insight motivates adopting Unsupervised Domain Adaptation (UDA) to transfer quality-relevant priors from labeled images to unlabeled point clouds. However, existing UDA-based PCQA methods often overlook key characteristics of perceptual quality, such as sensitivity to quality ranking and quality-aware feature alignment, thereby limiting their effectiveness. To address these issues, we propose a novel Quality-aware Domain adaptation framework for PCQA, termed QD-PCQA. The framework comprises two main components: i) a Rank-weighted Conditional Alignment (RCA) strategy that aligns features under consistent quality levels and adaptively emphasizes misranked samples to reinforce perceptual quality ranking awareness; and ii) a Quality-guided Feature Augmentation (QFA) strategy, which includes quality-guided style mixup, multi-layer extension, and dual-domain augmentation modules to augment perceptual feature alignment. Extensive cross-domain experiments demonstrate that QD-PCQA significantly improves generalization in NR-PCQA tasks. The code is available at https://github.com/huhu-code/QD-PCQA.

  </details>



- **Field imaging framework for morphological characterization of aggregates with computer vision: Algorithms and applications**  
  Haohang Huang  
  _2026-03-04_ · https://arxiv.org/abs/2603.03654v1  
  <details><summary>Abstract</summary>

  Construction aggregates, including sand and gravel, crushed stone and riprap, are the core building blocks of the construction industry. State-of-the-practice characterization methods mainly relies on visual inspection and manual measurement. State-of-the-art aggregate imaging methods have limitations that are only applicable to regular-sized aggregates under well-controlled conditions. This dissertation addresses these major challenges by developing a field imaging framework for the morphological characterization of aggregates as a multi-scenario solution. For individual and non-overlapping aggregates, a field imaging system was designed and the associated segmentation and volume estimation algorithms were developed. For 2D image analyses of aggregates in stockpiles, an automated 2D instance segmentation and morphological analysis approach was established. For 3D point cloud analyses of aggregate stockpiles, an integrated 3D Reconstruction-Segmentation-Completion (RSC-3D) approach was established: 3D reconstruction procedures from multi-view images, 3D stockpile instance segmentation, and 3D shape completion to predict the unseen sides. First, a 3D reconstruction procedure was developed to obtain high-fidelity 3D models of collected aggregate samples, based on which a 3D aggregate particle library was constructed. Next, two datasets were derived from the 3D particle library for 3D learning: a synthetic dataset of aggregate stockpiles with ground-truth instance labels, and a dataset of partial-complete shape pairs, developed with varying-view raycasting schemes. A state-of-the-art 3D instance segmentation network and a 3D shape completion network were trained on the datasets, respectively. The application of the integrated approach was demonstrated on real stockpiles and validated with ground-truth, showing good performance in capturing and predicting the unseen sides of aggregates.

  </details>



- **Confidence-aware Monocular Depth Estimation for Minimally Invasive Surgery**  
  Muhammad Asad, Emanuele Colleoni, Pritesh Mehta, Nicolas Toussaint, Ricardo Sanchez-Matilla, Maria Robu, Faisal Bashir, Rahim Mohammadi, Imanol Luengo, Danail Stoyanov  
  _2026-03-03_ · https://arxiv.org/abs/2603.03571v1  
  <details><summary>Abstract</summary>

  Purpose: Monocular depth estimation (MDE) is vital for scene understanding in minimally invasive surgery (MIS). However, endoscopic video sequences are often contaminated by smoke, specular reflections, blur, and occlusions, limiting the accuracy of MDE models. In addition, current MDE models do not output depth confidence, which could be a valuable tool for improving their clinical reliability. Methods: We propose a novel confidence-aware MDE framework featuring three significant contributions: (i) Calibrated confidence targets: an ensemble of fine-tuned stereo matching models is used to capture disparity variance into pixel-wise confidence probabilities; (ii) Confidence-aware loss: Baseline MDE models are optimized with confidence-aware loss functions, utilizing pixel-wise confidence probabilities such that reliable pixels dominate training; and (iii) Inference-time confidence: a confidence estimation head is proposed with two convolution layers to predict per-pixel confidence at inference, enabling assessment of depth reliability. Results: Comprehensive experimental validation across internal and public datasets demonstrates that our framework improves depth estimation accuracy and can robustly quantify the prediction's confidence. On the internal clinical endoscopic dataset (StereoKP), we improve dense depth estimation accuracy by ~8% as compared to the baseline model. Conclusion: Our confidence-aware framework enables improved accuracy of MDE models in MIS, addressing challenges posed by noise and artifacts in pre-clinical and clinical data, and allows MDE models to provide confidence maps that may be used to improve their reliability for clinical applications.

  </details>



- **Utonia: Toward One Encoder for All Point Clouds**  
  Yujia Zhang, Xiaoyang Wu, Yunhan Yang, Xianzhe Fan, Han Li, Yuechen Zhang, Zehao Huang, Naiyan Wang, Hengshuang Zhao  
  _2026-03-03_ · https://arxiv.org/abs/2603.03283v1  
  <details><summary>Abstract</summary>

  We dream of a future where point clouds from all domains can come together to shape a single model that benefits them all. Toward this goal, we present Utonia, a first step toward training a single self-supervised point transformer encoder across diverse domains, spanning remote sensing, outdoor LiDAR, indoor RGB-D sequences, object-centric CAD models, and point clouds lifted from RGB-only videos. Despite their distinct sensing geometries, densities, and priors, Utonia learns a consistent representation space that transfers across domains. This unification improves perception capability while revealing intriguing emergent behaviors that arise only when domains are trained jointly. Beyond perception, we observe that Utonia representations can also benefit embodied and multimodal reasoning: conditioning vision-language-action policies on Utonia features improves robotic manipulation, and integrating them into vision-language models yields gains on spatial reasoning. We hope Utonia can serve as a step toward foundation models for sparse 3D data, and support downstream applications in AR/VR, robotics, and autonomous driving.

  </details>



- **LoGeR: Long-Context Geometric Reconstruction with Hybrid Memory**  
  Junyi Zhang, Charles Herrmann, Junhwa Hur, Chen Sun, Ming-Hsuan Yang, Forrester Cole, Trevor Darrell, Deqing Sun  
  _2026-03-03_ · https://arxiv.org/abs/2603.03269v1  
  <details><summary>Abstract</summary>

  Feedforward geometric foundation models achieve strong short-window reconstruction, yet scaling them to minutes-long videos is bottlenecked by quadratic attention complexity or limited effective memory in recurrent designs. We present LoGeR (Long-context Geometric Reconstruction), a novel architecture that scales dense 3D reconstruction to extremely long sequences without post-optimization. LoGeR processes video streams in chunks, leveraging strong bidirectional priors for high-fidelity intra-chunk reasoning. To manage the critical challenge of coherence across chunk boundaries, we propose a learning-based hybrid memory module. This dual-component system combines a parametric Test-Time Training (TTT) memory to anchor the global coordinate frame and prevent scale drift, alongside a non-parametric Sliding Window Attention (SWA) mechanism to preserve uncompressed context for high-precision adjacent alignment. Remarkably, this memory architecture enables LoGeR to be trained on sequences of 128 frames, and generalize up to thousands of frames during inference. Evaluated across standard benchmarks and a newly repurposed VBR dataset with sequences of up to 19k frames, LoGeR substantially outperforms prior state-of-the-art feedforward methods--reducing ATE on KITTI by over 74%--and achieves robust, globally consistent reconstruction over unprecedented horizons.

  </details>



- **VIRGi: View-dependent Instant Recoloring of 3D Gaussians Splats**  
  Alessio Mazzucchelli, Ivan Ojeda-Martin, Fernando Rivas-Manzaneque, Elena Garces, Adrian Penate-Sanchez, Francesc Moreno-Noguer  
  _2026-03-03_ · https://arxiv.org/abs/2603.02986v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has recently transformed the fields of novel view synthesis and 3D reconstruction due to its ability to accurately model complex 3D scenes and its unprecedented rendering performance. However, a significant challenge persists: the absence of an efficient and photorealistic method for editing the appearance of the scene's content. In this paper we introduce VIRGi, a novel approach for rapidly editing the color of scenes modeled by 3DGS while preserving view-dependent effects such as specular highlights. Key to our method are a novel architecture that separates color into diffuse and view-dependent components, and a multi-view training strategy that integrates image patches from multiple viewpoints. Improving over the conventional single-view batch training, our 3DGS representation provides more accurate reconstruction and serves as a solid representation for the recoloring task. For 3DGS recoloring, we then introduce a rapid scheme requiring only one manually edited image of the scene from the end-user. By fine-tuning the weights of a single MLP, alongside a module for single-shot segmentation of the editable area, the color edits are seamlessly propagated to the entire scene in just two seconds, facilitating real-time interaction and providing control over the strength of the view-dependent effects. An exhaustive validation on diverse datasets demonstrates significant quantitative and qualitative advancements over competitors based on Neural Radiance Fields representations.

  </details>



- **The Dresden Dataset for 4D Reconstruction of Non-Rigid Abdominal Surgical Scenes**  
  Reuben Docea, Rayan Younis, Yonghao Long, Maxime Fleury, Jinjing Xu, Chenyang Li, André Schulze, Ann Wierick, Johannes Bender, Micha Pfeiffer, et al.  
  _2026-03-03_ · https://arxiv.org/abs/2603.02985v1  
  <details><summary>Abstract</summary>

  The D4D Dataset provides paired endoscopic video and high-quality structured-light geometry for evaluating 3D reconstruction of deforming abdominal soft tissue in realistic surgical conditions. Data were acquired from six porcine cadaver sessions using a da Vinci Xi stereo endoscope and a Zivid structured-light camera, registered via optical tracking and manually curated iterative alignment methods. Three sequence types - whole deformations, incremental deformations, and moved-camera clips - probe algorithm robustness to non-rigid motion, deformation magnitude, and out-of-view updates. Each clip provides rectified stereo images, per-frame instrument masks, stereo depth, start/end structured-light point clouds, curated camera poses and camera intrinsics. In postprocessing, ICP and semi-automatic registration techniques are used to register data, and instrument masks are created. The dataset enables quantitative geometric evaluation in both visible and occluded regions, alongside photometric view-synthesis baselines. Comprising over 300,000 frames and 369 point clouds across 98 curated recordings, this resource can serve as a comprehensive benchmark for developing and evaluating non-rigid SLAM, 4D reconstruction, and depth estimation methods.

  </details>


