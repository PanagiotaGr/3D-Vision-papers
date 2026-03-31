# 3D Reconstruction

_Updated: 2026-03-31 07:42 UTC_

Total papers shown: **15**


---

- **Industrial3D: A Terrestrial LiDAR Point Cloud Dataset and CrossParadigm Benchmark for Industrial Infrastructure**  
  Chao Yin, Hongzhe Yue, Qing Han, Difeng Hu, Zhenyu Liang, Fangzhou Lin, Bing Sun, Boyu Wang, Mingkai Li, Wei Yao, et al.  
  _2026-03-30_ · https://arxiv.org/abs/2603.28660v1  
  <details><summary>Abstract</summary>

  Automated semantic understanding of dense point clouds is a prerequisite for Scan-to-BIM pipelines, digital twin construction, and as-built verification--core tasks in the digital transformation of the construction industry. Yet for industrial mechanical, electrical, and plumbing (MEP) facilities, this challenge remains largely unsolved: TLS acquisitions of water treatment plants, chiller halls, and pumping stations exhibit extreme geometric ambiguity, severe occlusion, and extreme class imbalance that architectural benchmarks (e.g., S3DIS or ScanNet) cannot adequately represent. We present Industrial3D, a terrestrial LiDAR dataset comprising 612 million expertly labelled points at 6 mm resolution from 13 water treatment facilities. At 6.6x the scale of the closest comparable MEP dataset, Industrial3D provides the largest and most demanding testbed for industrial 3D scene understanding to date. We further establish the first industrial cross-paradigm benchmark, evaluating nine representative methods across fully supervised, weakly supervised, unsupervised, and foundation model settings under a unified benchmark protocol. The best supervised method achieves 55.74% mIoU, whereas zero-shot Point-SAM reaches only 15.79%--a 39.95 percentage-point gap that quantifies the unresolved domain-transfer challenge for industrial TLS data. Systematic analysis reveals that this gap originates from a dual crisis: statistical rarity (215:1 imbalance, 3.5x more severe than S3DIS) and geometric ambiguity (tail-class points share cylindrical primitives with head-class pipes) that frequency-based re-weighting alone cannot resolve. Industrial3D, along with benchmark code and pre-trained models, will be publicly available at https://github.com/pointcloudyc/Industrial3D.

  </details>



- **Seen2Scene: Completing Realistic 3D Scenes with Visibility-Guided Flow**  
  Quan Meng, Yujin Chen, Lei Li, Matthias Nießner, Angela Dai  
  _2026-03-30_ · https://arxiv.org/abs/2603.28548v1  
  <details><summary>Abstract</summary>

  We present Seen2Scene, the first flow matching-based approach that trains directly on incomplete, real-world 3D scans for scene completion and generation. Unlike prior methods that rely on complete and hence synthetic 3D data, our approach introduces visibility-guided flow matching, which explicitly masks out unknown regions in real scans, enabling effective learning from real-world, partial observations. We represent 3D scenes using truncated signed distance field (TSDF) volumes encoded in sparse grids and employ a sparse transformer to efficiently model complex scene structures while masking unknown regions. We employ 3D layout boxes as an input conditioning signal, and our approach is flexibly adapted to various other inputs such as text or partial scans. By learning directly from real-world, incomplete 3D scans, Seen2Scene enables realistic 3D scene completion for complex, cluttered real environments. Experiments demonstrate that our model produces coherent, complete, and realistic 3D scenes, outperforming baselines in completion accuracy and generation quality.

  </details>



- **Tele-Catch: Adaptive Teleoperation for Dexterous Dynamic 3D Object Catching**  
  Weiguang Zhao, Junting Dong, Rui Zhang, Kailin Li, Qin Zhao, Kaizhu Huang  
  _2026-03-30_ · https://arxiv.org/abs/2603.28427v1  
  <details><summary>Abstract</summary>

  Teleoperation is a key paradigm for transferring human dexterity to robots, yet most prior work targets objects that are initially static, such as grasping or manipulation. Dynamic object catch, where objects move before contact, remains underexplored. Pure teleoperation in this task often fails due to timing, pose, and force errors, highlighting the need for shared autonomy that combines human input with autonomous policies. To this end, we present Tele-Catch, a systematic framework for dexterous hand teleoperation in dynamic object catching. At its core, we design DAIM, a dynamics-aware adaptive integration mechanism that realizes shared autonomy by fusing glove-based teleoperation signals into the diffusion policy denoising process. It adaptively modulates control based on the interaction object state. To improve policy robustness, we introduce DP-U3R, which integrates unsupervised geometric representations from point cloud observations into diffusion policy learning, enabling geometry-aware decision making. Extensive experiments demonstrate that Tele-Catch significantly improves accuracy and robustness in dynamic catching tasks, while also exhibiting consistent gains across distinct dexterous hand embodiments and previously unseen object categories.

  </details>



- **TerraSky3D: Multi-View Reconstructions of European Landmarks in 4K**  
  Mattia D'Urso, Yuxi Hu, Christian Sormann, Mattia Rossi, Friedrich Fraundorfer  
  _2026-03-30_ · https://arxiv.org/abs/2603.28287v1  
  <details><summary>Abstract</summary>

  Despite the growing need for data of more and more sophisticated 3D reconstruction pipelines, we can still observe a scarcity of suitable public datasets. Existing 3D datasets are either low resolution, limited to a small amount of scenes, based on images of varying quality because retrieved from the internet, or limited to specific capturing scenarios. Motivated by this lack of suitable 3D datasets, we captured TerraSky3D, a high-resolution large-scale 3D reconstruction dataset comprising 50,000 images divided into 150 ground, aerial, and mixed scenes. The dataset focuses on European landmarks and comes with curated calibration data, camera poses, and depth maps. TerraSky3D tries to answer the need for challenging dataset that can be used to train and evaluate 3D reconstruction-related pipelines.

  </details>



- **Ghost-FWL: A Large-Scale Full-Waveform LiDAR Dataset for Ghost Detection and Removal**  
  Kazuma Ikeda, Ryosei Hara, Rokuto Nagata, Ozora Sako. Zihao Ding, Takahiro Kado, Ibuki Fujioka, Taro Beppu, Mariko Isogawa, Kentaro Yoshioka  
  _2026-03-30_ · https://arxiv.org/abs/2603.28224v1  
  <details><summary>Abstract</summary>

  LiDAR has become an essential sensing modality in autonomous driving, robotics, and smart-city applications. However, ghost points (or ghosts), which are false reflections caused by multi-path laser returns from glass and reflective surfaces, severely degrade 3D mapping and localization accuracy. Prior ghost removal relies on geometric consistency in dense point clouds, failing on mobile LiDAR's sparse, dynamic data. We address this by exploiting full-waveform LiDAR (FWL), which captures complete temporal intensity profiles rather than just peak distances, providing crucial cues for distinguishing ghosts from genuine reflections in mobile scenarios. As this is a new task, we present Ghost-FWL, the first and largest annotated mobile FWL dataset for ghost detection and removal. Ghost-FWL comprises 24K frames across 10 diverse scenes with 7.5 billion peak-level annotations, which is 100x larger than existing annotated FWL datasets. Benefiting from this large-scale dataset, we establish a FWL-based baseline model for ghost detection and propose FWL-MAE, a masked autoencoder for efficient self-supervised representation learning on FWL data. Experiments show that our baseline outperforms existing methods in ghost removal accuracy, and our ghost removal further enhances downstream tasks such as LiDAR-based SLAM (66% trajectory error reduction) and 3D object detection (50x false positive reduction). The dataset and code is publicly available and can be accessed via the project page: https://keio-csg.github.io/Ghost-FWL

  </details>



- **Octree-based Learned Point Cloud Geometry Compression: A Lossy Perspective**  
  Kaiyu Zheng, Wei Gao, Huiming Zheng  
  _2026-03-30_ · https://arxiv.org/abs/2603.28095v1  
  <details><summary>Abstract</summary>

  Octree-based context learning has recently become a leading method in point cloud compression. However, its potential on lossy compression remains undiscovered. The traditional lossy compression paradigm using lossless octree representation with quantization step adjustment may result in severe distortions due to massive missing points in quantization. Therefore, we analyze data characteristics of different point clouds and propose lossy approaches specifically. For object point clouds that suffer from quantization step adjustment, we propose a new leaf nodes lossy compression method, which achieves lossy compression by performing bit-wise coding and binary prediction on leaf nodes. For LiDAR point clouds, we explore variable rate approaches and propose a simple but effective rate control method. Experimental results demonstrate that the proposed leaf nodes lossy compression method significantly outperforms the previous octree-based method on object point clouds, and the proposed rate control method achieves about 1% bit error without finetuning on LiDAR point clouds.

  </details>



- **\textit{4DSurf}: High-Fidelity Dynamic Scene Surface Reconstruction**  
  Renjie Wu, Hongdong Li, Jose M. Alvarez, Miaomiao Liu  
  _2026-03-30_ · https://arxiv.org/abs/2603.28064v1  
  <details><summary>Abstract</summary>

  This paper addresses the problem of dynamic scene surface reconstruction using Gaussian Splatting (GS), aiming to recover temporally consistent geometry. While existing GS-based dynamic surface reconstruction methods can yield superior reconstruction, they are typically limited to either a single object or objects with only small deformations, struggling to maintain temporally consistent surface reconstruction of large deformations over time. We propose ``\textit{4DSurf}'', a novel and unified framework for generic dynamic surface reconstruction that does not require specifying the number or types of objects in the scene, can handle large surface deformations and temporal inconsistency in reconstruction. The key innovation of our framework is the introduction of Gaussian deformations induced Signed Distance Function Flow Regularization that constrains the motion of Gaussians to align with the evolving surface. To handle large deformations, we introduce an Overlapping Segment Partitioning strategy that divides the sequence into overlapping segments with small deformations and incrementally passes geometric information across segments through the shared overlapping timestep. Experiments on two challenging dynamic scene datasets, Hi4D and CMU Panoptic, demonstrate that our method outperforms state-of-the-art surface reconstruction methods by 49\% and 19\% in Chamfer distance, respectively, and achieves superior temporal consistency under sparse-view settings.

  </details>



- **AffordMatcher: Affordance Learning in 3D Scenes from Visual Signifiers**  
  Nghia Vu, Tuong Do, Khang Nguyen, Baoru Huang, Nhat Le, Binh Xuan Nguyen, Erman Tjiputra, Quang D. Tran, Ravi Prakash, Te-Chuan Chiu, et al.  
  _2026-03-30_ · https://arxiv.org/abs/2603.27970v1  
  <details><summary>Abstract</summary>

  Affordance learning is a complex challenge in many applications, where existing approaches primarily focus on the geometric structures, visual knowledge, and affordance labels of objects to determine interactable regions. However, extending this learning capability to a scene is significantly more complicated, as incorporating object- and scene-level semantics is not straightforward. In this work, we introduce AffordBridge, a large-scale dataset with 291,637 functional interaction annotations across 685 high-resolution indoor scenes in the form of point clouds. Our affordance annotations are complemented by RGB images that are linked to the same instances within the scenes. Building upon our dataset, we propose AffordMatcher, an affordance learning method that establishes coherent semantic correspondences between image-based and point cloud-based instances for keypoint matching, enabling a more precise identification of affordance regions based on cues, so-called visual signifiers. Experimental results on our dataset demonstrate the effectiveness of our approach compared to other methods.

  </details>



- **Hg-I2P: Bridging Modalities for Generalizable Image-to-Point-Cloud Registration via Heterogeneous Graphs**  
  Pei An, Junfeng Ding, Jiaqi Yang, Yulong Wang, Jie Ma, Liangliang Nan  
  _2026-03-30_ · https://arxiv.org/abs/2603.27969v1  
  <details><summary>Abstract</summary>

  Image-to-point-cloud (I2P) registration aims to align 2D images with 3D point clouds by establishing reliable 2D-3D correspondences. The drastic modality gap between images and point clouds makes it challenging to learn features that are both discriminative and generalizable, leading to severe performance drops in unseen scenarios. We address this challenge by introducing a heterogeneous graph that enables refining both cross-modal features and correspondences within a unified architecture. The proposed graph represents a mapping between segmented 2D and 3D regions, which enhances cross-modal feature interaction and thus improves feature discriminability. In addition, modeling the consistency among vertices and edges within the graph enables pruning of unreliable correspondences. Building on these insights, we propose a heterogeneous graph embedded I2P registration method, termed Hg-I2P. It learns a heterogeneous graph by mining multi-path feature relationships, adapts features under the guidance of heterogeneous edges, and prunes correspondences using graph-based projection consistency. Experiments on six indoor and outdoor benchmarks under cross-domain setups demonstrate that Hg-I2P significantly outperforms existing methods in both generalization and accuracy. Code is released on https://github.com/anpei96/hg-i2p-demo.

  </details>



- **LiDAR for Crowd Management: Applications, Benefits, and Future Directions**  
  Abdullah Khanfor, Chaima Zaghouani, Hakim Ghazzai, Ahmad Alsharoa, Gianluca Setti  
  _2026-03-29_ · https://arxiv.org/abs/2603.27663v1  
  <details><summary>Abstract</summary>

  Light Detection and Ranging (LiDAR) technology offers significant advantages for effective crowd management. This article presents LiDAR technology and highlights its primary advantages over other monitoring technologies, including enhanced privacy, performance in various weather conditions, and precise 3D mapping. We present a general taxonomy of four key tasks in crowd management: crowd detection, counting, tracking, and behavior classification, with illustrative examples of LiDAR applications for each task. We identify challenges and open research directions, including the scarcity of dedicated datasets, sensor fusion requirements, artificial intelligence integration, and processing needs for LiDAR point clouds. This article offers actionable insights for developing crowd management solutions tailored to public safety applications.

  </details>



- **SPREAD: Spatial-Physical REasoning via geometry Aware Diffusion**  
  Minzhang Li, Kuixiang Shao, Xuebing Li, Yuyang Jiao, Yinuo Bai, Hengan Zhou, Sixian Shen, Jiayuan Gu, Jingyi Yu  
  _2026-03-29_ · https://arxiv.org/abs/2603.27573v1  
  <details><summary>Abstract</summary>

  Automated 3D scene generation is pivotal for applications spanning virtual reality, digital content creation, and Embodied AI. While computer graphics prioritizes aesthetic layouts, vision and robotics demand scenes that mirror real-world complexity which current data-driven methods struggle to achieve due to limited unstructured training data and insufficient spatial and physical modeling. We propose SPREAD, a diffusion-based framework that jointly learns spatial and physical relationships through a graph transformer, explicitly conditioning on posed scene point clouds for geometric awareness. Moreover, our model integrates differentiable guidance for collision avoidance, relational constraint, and gravity, ensuring physically coherent scenes without sacrificing relational context. Our experiments on 3D-FRONT and ProcTHOR datasets demonstrate state-of-the-art performance in spatial-relational reasoning and physical metrics. Moreover, \ours{} outperforms baselines in scene consistency and stability during pre- and post-physics simulation, proving its capability to generate simulation-ready environments for embodied AI agents.

  </details>



- **Annotation-Free Detection of Drivable Areas and Curbs Leveraging LiDAR Point Cloud Maps**  
  Fulong Ma, Daojie Peng, Jun Ma  
  _2026-03-29_ · https://arxiv.org/abs/2603.27553v1  
  <details><summary>Abstract</summary>

  Drivable areas and curbs are critical traffic elements for autonomous driving, forming essential components of the vehicle visual perception system and ensuring driving safety. Deep neural networks (DNNs) have significantly improved perception performance for drivable area and curb detection, but most DNN-based methods rely on large manually labeled datasets, which are costly, time-consuming, and expert-dependent, limiting their real-world application. Thus, we developed an automated training data generation module. Our previous work generated training labels using single-frame LiDAR and RGB data, suffering from occlusion and distant point cloud sparsity. In this paper, we propose a novel map-based automatic data labeler (MADL) module, combining LiDAR mapping/localization with curb detection to automatically generate training data for both tasks. MADL avoids occlusion and point cloud sparsity issues via LiDAR mapping, creating accurate large-scale datasets for DNN training. In addition, we construct a data review agent to filter the data generated by the MADL module, eliminating low-quality samples. Experiments on the KITTI, KITTI-CARLA and 3D-Curb datasets show that MADL achieves impressive performance compared to manual labeling, and outperforms traditional and state-of-the-art self-supervised methods in robustness and accuracy.

  </details>



- **MV-RoMa: From Pairwise Matching into Multi-View Track Reconstruction**  
  Jongmin Lee, Seungyeop Kang, Sungjoo Yoo  
  _2026-03-29_ · https://arxiv.org/abs/2603.27542v1  
  <details><summary>Abstract</summary>

  Establishing consistent correspondences across images is essential for 3D vision tasks such as structure-from-motion (SfM), yet most existing matchers operate in a pairwise manner, often producing fragmented and geometrically inconsistent tracks when their predictions are chained across views. We propose MV-RoMa, a multi-view dense matching model that jointly estimates dense correspondences from a source image to multiple co-visible targets. Specifically, we design an efficient model architecture which avoids high computational cost of full cross-attention for multi-view feature interaction: (i) multi-view encoder that leverages pair-wise matching results as a geometric prior, and (ii) multi-view matching refiner that refines correspondences using pixel-wise attention. Additionally, we propose a post-processing strategy that integrates our model's consistent multi-view correspondences as high-quality tracks for SfM. Across diverse and challenging benchmarks, MV-RoMa produces more reliable correspondences and substantially denser, more accurate 3D reconstructions than existing sparse and dense matching methods. Project page: https://icetea-cv.github.io/mv-roma/.

  </details>



- **From None to All: Self-Supervised 3D Reconstruction via Novel View Synthesis**  
  Ranran Huang, Weixun Luo, Ye Mao, Krystian Mikolajczyk  
  _2026-03-29_ · https://arxiv.org/abs/2603.27455v1  
  <details><summary>Abstract</summary>

  In this paper, we introduce NAS3R, a self-supervised feed-forward framework that jointly learns explicit 3D geometry and camera parameters with no ground-truth annotations and no pretrained priors. During training, NAS3R reconstructs 3D Gaussians from uncalibrated and unposed context views and renders target views using its self-predicted camera parameters, enabling self-supervised training from 2D photometric supervision. To ensure stable convergence, NAS3R integrates reconstruction and camera prediction within a shared transformer backbone regulated by masked attention, and adopts a depth-based Gaussian formulation that facilitates well-conditioned optimization. The framework is compatible with state-of-the-art supervised 3D reconstruction architectures and can incorporate pretrained priors or intrinsic information when available. Extensive experiments show that NAS3R achieves superior results to other self-supervised methods, establishing a scalable and geometry-aware paradigm for 3D reconstruction from unconstrained data. Code and models are publicly available at https://ranrhuang.github.io/nas3r/.

  </details>



- **HD-VGGT: High-Resolution Visual Geometry Transformer**  
  Tianrun Chen, Yuanqi Hu, Yidong Han, Hanjie Xu, Deyi Ji, Qi Zhu, Chunan Yu, Xin Zhang, Cheng Chen, Chaotao Ding, et al.  
  _2026-03-28_ · https://arxiv.org/abs/2603.27222v1  
  <details><summary>Abstract</summary>

  High-resolution imagery is essential for accurate 3D reconstruction, as many geometric details only emerge at fine spatial scales. Recent feed-forward approaches, such as the Visual Geometry Grounded Transformer (VGGT), have demonstrated the ability to infer scene geometry from large collections of images in a single forward pass. However, scaling these models to high-resolution inputs remains challenging: the number of tokens in transformer architectures grows rapidly with both image resolution and the number of views, leading to prohibitive computational and memory costs. Moreover, we observe that visually ambiguous regions, such as repetitive patterns, weak textures, or specular surfaces, often produce unstable feature tokens that degrade geometric inference, especially at higher resolutions. We introduce HD-VGGT, a dual-branch architecture for efficient and robust high-resolution 3D reconstruction. A low-resolution branch predicts a coarse, globally consistent geometry, while a high-resolution branch refines details via a learned feature upsampling module. To handle unstable tokens, we propose Feature Modulation, which suppresses unreliable features early in the transformer. HD-VGGT leverages high-resolution images and supervision without full-resolution transformer costs, achieving state-of-the-art reconstruction quality.

  </details>


