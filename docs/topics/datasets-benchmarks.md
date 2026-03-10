# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-03-10 07:06 UTC_

Total papers shown: **50**


---

- **Talking Together: Synthesizing Co-Located 3D Conversations from Audio**  
  Mengyi Shan, Shouchieh Chang, Ziqian Bai, Shichen Liu, Yinda Zhang, Luchuan Song, Rohit Pandey, Sean Fanello, Zeng Huang  
  _2026-03-09_ · https://arxiv.org/abs/2603.08674v1  
  <details><summary>Abstract</summary>

  We tackle the challenging task of generating complete 3D facial animations for two interacting, co-located participants from a mixed audio stream. While existing methods often produce disembodied "talking heads" akin to a video conference call, our work is the first to explicitly model the dynamic 3D spatial relationship -- including relative position, orientation, and mutual gaze -- that is crucial for realistic in-person dialogues. Our system synthesizes the full performance of both individuals, including precise lip-sync, and uniquely allows their relative head poses to be controlled via textual descriptions. To achieve this, we propose a dual-stream architecture where each stream is responsible for one participant's output. We employ speaker's role embeddings and inter-speaker cross-attention mechanisms designed to disentangle the mixed audio and model the interaction. Furthermore, we introduce a novel eye gaze loss to promote natural, mutual eye contact. To power our data-hungry approach, we introduce a novel pipeline to curate a large-scale conversational dataset consisting of over 2 million dyadic pairs from in-the-wild videos. Our method generates fluid, controllable, and spatially aware dyadic animations suitable for immersive applications in VR and telepresence, significantly outperforming existing baselines in perceived realism and interaction coherence.

  </details>



- **ImprovedGS+: A High-Performance C++/CUDA Re-Implementation Strategy for 3D Gaussian Splatting**  
  Jordi Muñoz Vicente  
  _2026-03-09_ · https://arxiv.org/abs/2603.08661v1  
  <details><summary>Abstract</summary>

  Recent advancements in 3D Gaussian Splatting (3DGS) have shifted the focus toward balancing reconstruction fidelity with computational efficiency. In this work, we propose ImprovedGS+, a high-performance, low-level reinvention of the ImprovedGS strategy, implemented natively within the LichtFeld-Studio framework. By transitioning from high-level Python logic to hardware-optimized C++/CUDA kernels, we achieve a significant reduction in host-device synchronization and training latency. Our implementation introduces a Long-Axis-Split (LAS) CUDA kernel, custom Laplacian-based importance kernels with Non-Maximum Suppression (NMS) for edge scores, and an adaptive Exponential Scale Scheduler. Experimental results on the Mip-NeRF360 dataset demonstrate that ImprovedGS+ establishes a new Pareto-optimal front for scene reconstruction. Our 1M-budget variant outperforms the state-of-the-art MCMC baseline by achieving a 26.8% reduction in training time (saving 17 minutes per session) and utilizing 13.3% fewer Gaussians while maintaining superior visual quality. Furthermore, our full variant demonstrates a 1.28 dB PSNR increase over the ADC baseline with a 38.4% reduction in parametric complexity. These results validate ImprovedGS+ as a scalable, high-speed solution that upholds the core pillars of Speed, Quality, and Usability within the LichtFeld-Studio ecosystem.

  </details>



- **CAST: Modeling Visual State Transitions for Consistent Video Retrieval**  
  Yanqing Liu, Yingcheng Liu, Fanghong Dong, Budianto Budianto, Cihang Xie, Yan Jiao  
  _2026-03-09_ · https://arxiv.org/abs/2603.08648v1  
  <details><summary>Abstract</summary>

  As video content creation shifts toward long-form narratives, composing short clips into coherent storylines becomes increasingly important. However, prevailing retrieval formulations remain context-agnostic at inference time, prioritizing local semantic alignment while neglecting state and identity consistency. To address this structural limitation, we formalize the task of Consistent Video Retrieval (CVR) and introduce a diagnostic benchmark spanning YouCook2, COIN, and CrossTask. We propose CAST (Context-Aware State Transition), a lightweight, plug-and-play adapter compatible with diverse frozen vision-language embedding spaces. By predicting a state-conditioned residual update ($Δ$) from visual history, CAST introduces an explicit inductive bias for latent state evolution. Extensive experiments show that CAST improves performance on YouCook2 and CrossTask, remains competitive on COIN, and consistently outperforms zero-shot baselines across diverse foundation backbones. Furthermore, CAST provides a useful reranking signal for black-box video generation candidates (e.g., from Veo), promoting more temporally coherent continuations.

  </details>



- **Retrieval-Augmented Gaussian Avatars: Improving Expression Generalization**  
  Matan Levy, Gavriel Habib, Issar Tzachor, Dvir Samuel, Rami Ben-Ari, Nir Darshan, Or Litany, Dani Lischinski  
  _2026-03-09_ · https://arxiv.org/abs/2603.08645v1  
  <details><summary>Abstract</summary>

  Template-free animatable head avatars can achieve high visual fidelity by learning expression-dependent facial deformation directly from a subject's capture, avoiding parametric face templates and hand-designed blendshape spaces. However, since learned deformation is supervised only by the expressions observed for a single identity, these models suffer from limited expression coverage and often struggle when driven by motions that deviate from the training distribution. We introduce RAF (Retrieval-Augmented Faces), a simple training-time augmentation designed for template-free head avatars that learn deformation from data. RAF constructs a large unlabeled expression bank and, during training, replaces a subset of the subject's expression features with nearest-neighbor expressions retrieved from this bank while still reconstructing the subject's original frames. This exposes the deformation field to a broader range of expression conditions, encouraging stronger identity-expression decoupling and improving robustness to expression distribution shift without requiring paired cross-identity data, additional annotations, or architectural changes. We further analyze how retrieval augmentation increases expression diversity and validate retrieval quality with a user study showing that retrieved neighbors are perceptually closer in expression and pose. Experiments on the NeRSemble benchmark demonstrate that RAF consistently improves expression fidelity over the baseline, in both self-driving and cross-driving scenarios.

  </details>



- **StreamReady: Learning What to Answer and When in Long Streaming Videos**  
  Shehreen Azad, Vibhav Vineet, Yogesh Singh Rawat  
  _2026-03-09_ · https://arxiv.org/abs/2603.08620v1  
  <details><summary>Abstract</summary>

  Streaming video understanding often involves time-sensitive scenarios where models need to answer exactly when the supporting visual evidence appears: answering before the evidence reflects speculation, answering after it has passed reduces real-time utility. To capture this behavior, we introduce a readiness-aware formulation of streaming video understanding with the Answer Readiness Score (ARS), a timing-aware objective with asymmetric early and late penalties. When combined with correctness, ARS defines an effective accuracy that measures not just whether a model is right, but whether it answers at the appropriate moment. Building on this formulation, we introduce StreamReady, a framework to unify temporal reasoning with on-time answering through a lightweight readiness mechanism that decides if sufficient evidence has been observed before responding. To evaluate this capability, we further introduce ProReady-QA, a benchmark with annotated answer evidence windows and proactive multi-turn questions across local and global contexts. StreamReady achieves superior performance on ProReady-QA, and consistently outperforms prior methods across eight additional streaming and offline long-video benchmarks, demonstrating robust and broadly generalizable video understanding capability.

  </details>



- **Weakly Supervised Teacher-Student Framework with Progressive Pseudo-mask Refinement for Gland Segmentation**  
  Hikmat Khan, Wei Chen, Muhammad Khalid Khan Niazi  
  _2026-03-09_ · https://arxiv.org/abs/2603.08605v1  
  <details><summary>Abstract</summary>

  Background and objectives: Colorectal cancer histopathological grading depends on accurate segmentation of glandular structures. Current deep learning approaches rely on large scale pixel level annotations that are labor intensive and difficult to obtain in routine clinical practice. Weakly supervised semantic segmentation offers a promising alternative. However, class activation map based methods often produce incomplete pseudo masks that emphasize highly discriminative regions and fail to supervise unannotated glandular structures. We propose a weakly supervised teacher student framework that leverages sparse pathologist annotations and an Exponential Moving Average stabilized teacher network to generate refined pseudo masks. Methods: The framework integrates confidence based filtering, adaptive fusion of teacher predictions with limited ground truth, and curriculum guided refinement to progressively segment unannotated glandular regions. The method was evaluated on an institutional colorectal cancer cohort from The Ohio State University Wexner Medical Center consisting of 60 hematoxylin and eosin stained whole slide images and on public datasets including the Gland Segmentation dataset, TCGA COAD, TCGA READ, and SPIDER. Results: On the Gland Segmentation dataset the framework achieved a mean Intersection over Union of 80.10 and a mean Dice coefficient of 89.10. Cross cohort evaluation demonstrated robust generalization on TCGA COAD and TCGA READ without additional annotations, while reduced performance on SPIDER reflected domain shift. Conclusions: The proposed framework provides an annotation efficient and generalizable approach for gland segmentation in colorectal histopathology.

  </details>



- **BioGait-VLM: A Tri-Modal Vision-Language-Biomechanics Framework for Interpretable Clinical Gait Assessment**  
  Erdong Chen, Yuyang Ji, Jacob K. Greenberg, Benjamin Steel, Faraz Arkam, Abigail Lewis, Pranay Singh, Feng Liu  
  _2026-03-09_ · https://arxiv.org/abs/2603.08564v1  
  <details><summary>Abstract</summary>

  Video-based Clinical Gait Analysis often suffers from poor generalization as models overfit environmental biases instead of capturing pathological motion. To address this, we propose BioGait-VLM, a tri-modal Vision-Language-Biomechanics framework for interpretable clinical gait assessment. Unlike standard video encoders, our architecture incorporates a Temporal Evidence Distillation branch to capture rhythmic dynamics and a Biomechanical Tokenization branch that projects 3D skeleton sequences into language-aligned semantic tokens. This enables the model to explicitly reason about joint mechanics independent of visual shortcuts. To ensure rigorous benchmarking, we augment the public GAVD dataset with a high-fidelity Degenerative Cervical Myelopathy (DCM) cohort to form a unified 8-class taxonomy, establishing a strict subject-disjoint protocol to prevent data leakage. Under this setting, BioGait-VLM achieves state-of-the-art recognition accuracy. Furthermore, a blinded expert study confirms that biomechanical tokens significantly improve clinical plausibility and evidence grounding, offering a path toward transparent, privacy-enhanced gait assessment.

  </details>



- **mmGAT: Pose Estimation by Graph Attention with Mutual Features from mmWave Radar Point Cloud**  
  Abdullah Al Masud, Shi Xintong, Mondher Bouazizi, Ohtsuki Tomoaki  
  _2026-03-09_ · https://arxiv.org/abs/2603.08551v1  
  <details><summary>Abstract</summary>

  Pose estimation and human action recognition (HAR) are pivotal technologies spanning various domains. While the image-based pose estimation and HAR are widely admired for their superior performance, they lack in privacy protection and suboptimal performance in low-light and dark environments. This paper exploits the capabilities of millimeter-wave (mmWave) radar technology for human pose estimation by processing radar data with Graph Neural Network (GNN) architecture, coupled with the attention mechanism. Our goal is to capture the finer details of the radar point cloud to improve the pose estimation performance. To this end, we present a unique feature extraction technique that exploits the full potential of the GNN processing method for pose estimation. Our model mmGAT demonstrates remarkable performance on two publicly available benchmark mmWave datasets and establishes new state of the art results in most scenarios in terms of human pose estimation. Our approach achieves a noteworthy reduction of pose estimation mean per joint position error (MPJPE) by 35.6% and PA-MPJPE by 14.1% from the current state of the art benchmark within this domain.

  </details>



- **Interactive World Simulator for Robot Policy Training and Evaluation**  
  Yixuan Wang, Rhythm Syed, Fangyu Wu, Mengchao Zhang, Aykut Onol, Jose Barreiros, Hooshang Nayyeri, Tony Dear, Huan Zhang, Yunzhu Li  
  _2026-03-09_ · https://arxiv.org/abs/2603.08546v1  
  <details><summary>Abstract</summary>

  Action-conditioned video prediction models (often referred to as world models) have shown strong potential for robotics applications, but existing approaches are often slow and struggle to capture physically consistent interactions over long horizons, limiting their usefulness for scalable robot policy training and evaluation. We present Interactive World Simulator, a framework for building interactive world models from a moderate-sized robot interaction dataset. Our approach leverages consistency models for both image decoding and latent-space dynamics prediction, enabling fast and stable simulation of physical interactions. In our experiments, the learned world models produce interaction-consistent pixel-level predictions and support stable long-horizon interactions for more than 10 minutes at 15 FPS on a single RTX 4090 GPU. Our framework enables scalable demonstration collection solely within the world models to train state-of-the-art imitation policies. Through extensive real-world evaluation across diverse tasks involving rigid objects, deformable objects, object piles, and their interactions, we find that policies trained on world-model-generated data perform comparably to those trained on the same amount of real-world data. Additionally, we evaluate policies both within the world models and in the real world across diverse tasks, and observe a strong correlation between simulated and real-world performance. Together, these results establish the Interactive World Simulator as a stable and physically consistent surrogate for scalable robotic data generation and faithful, reproducible policy evaluation.

  </details>



- **SecAgent: Efficient Mobile GUI Agent with Semantic Context**  
  Yiping Xie, Song Chen, Jingxuan Xing, Wei Jiang, Zekun Zhu, Yingyao Wang, Pi Bu, Jun Song, Yuning Jiang, Bo Zheng  
  _2026-03-09_ · https://arxiv.org/abs/2603.08533v1  
  <details><summary>Abstract</summary>

  Mobile Graphical User Interface (GUI) agents powered by multimodal large language models have demonstrated promising capabilities in automating complex smartphone tasks. However, existing approaches face two critical limitations: the scarcity of high-quality multilingual datasets, particularly for non-English ecosystems, and inefficient history representation methods. To address these challenges, we present SecAgent, an efficient mobile GUI agent at 3B scale. We first construct a human-verified Chinese mobile GUI dataset with 18k grounding samples and 121k navigation steps across 44 applications, along with a Chinese navigation benchmark featuring multi-choice action annotations. Building upon this dataset, we propose a semantic context mechanism that distills history screenshots and actions into concise, natural language summaries, significantly reducing computational costs while preserving task-relevant information. Through supervised and reinforcement fine-tuning, SecAgent outperforms similar-scale baselines and achieves performance comparable to 7B-8B models on our and public navigation benchmarks. We will open-source the training dataset, benchmark, model, and code to advance research in multilingual mobile GUI automation.

  </details>



- **BuildMamba: A Visual State-Space Based Model for Multi-Task Building Segmentation and Height Estimation from Satellite Images**  
  Sinan U. Ulu, A. Enes Doruk, I. Can Yagmur, Bahadir K. Gunturk, Oguz Hanoglu, Hasan F. Ates  
  _2026-03-09_ · https://arxiv.org/abs/2603.08523v1  
  <details><summary>Abstract</summary>

  Accurate building segmentation and height estimation from single-view RGB satellite imagery are fundamental for urban analytics, yet remain ill-posed due to structural variability and the high computational cost of global context modeling. While current approaches typically adapt monocular depth architectures, they often suffer from boundary bleeding and systematic underestimation of high-rise structures. To address these limitations, we propose BuildMamba, a unified multi-task framework designed to exploit the linear-time global modeling of visual state-space models. Motivated by the need for stronger structural coupling and computational efficiency, we introduce three modules: a Mamba Attention Module for dynamic spatial recalibration, a Spatial-Aware Mamba-FPN for multi-scale feature aggregation via gated state-space scans, and a Mask-Aware Height Refinement module using semantic priors to suppress height artifacts. Extensive experiments demonstrate that BuildMamba establishes a new performance upper bound across three benchmarks. Specifically, it achieves an IoU of 0.93 and RMSE of 1.77~m on DFC23 benchmark, surpassing state-of-the-art by 0.82~m in height estimation. Simulation results confirm the model's superior robustness and scalability for large-scale 3D urban reconstruction.

  </details>



- **OccTrack360: 4D Panoptic Occupancy Tracking from Surround-View Fisheye Cameras**  
  Yongzhi Lin, Kai Luo, Yuanfan Zheng, Hao Shi, Mengfei Duan, Yang Liu, Kailun Yang  
  _2026-03-09_ · https://arxiv.org/abs/2603.08521v1  
  <details><summary>Abstract</summary>

  Understanding dynamic 3D environments in a spatially continuous and temporally consistent manner is fundamental for robotics and autonomous driving. While recent advances in occupancy prediction provide a unified representation of scene geometry and semantics, progress in 4D panoptic occupancy tracking remains limited by the lack of benchmarks that support surround-view fisheye sensing, long temporal sequences, and instance-level voxel tracking. To address this gap, we present OccTrack360, a new benchmark for 4D panoptic occupancy tracking from surround-view fisheye cameras. OccTrack360 provides substantially longer and more diverse sequences (174~2234 frames) than prior benchmarks, together with principled voxel visibility annotations, including an all-direction occlusion mask and an MEI-based fisheye field-of-view mask. To establish a strong fisheye-oriented baseline, we further propose Focus on Sphere Occ (FoSOcc), a framework that addresses two core challenges in fisheye occupancy tracking: distorted spherical projection and inaccurate voxel-space localization. FoSOcc includes a Center Focusing Module (CFM) to enhance instance-aware spatial localization through supervised focus guidance, and a Spherical Lift Module (SLM) that extends perspective lifting to fisheye imaging under the Unified Projection Model. Extensive experiments on Occ3D-Waymo and OccTrack360 show that our method improves occupancy tracking quality with notable gains on geometrically regular categories, and establishes a strong baseline for future research on surround-view fisheye 4D occupancy tracking. The benchmark and source code will be made publicly available at https://github.com/YouthZest-Lin/OccTrack360.

  </details>



- **AtomVLA: Scalable Post-Training for Robotic Manipulation via Predictive Latent World Models**  
  Xiaoquan Sun, Zetian Xu, Chen Cao, Zonghe Liu, Yihan Sun, Jingrui Pang, Ruijian Zhang, Zhen Yang, Kang Pang, Dingxin He, et al.  
  _2026-03-09_ · https://arxiv.org/abs/2603.08519v1  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models demonstrate remarkable potential for generalizable robotic manipulation. The execution of complex multi-step behaviors in VLA models can be improved by robust instruction grounding, a critical component for effective control. However, current paradigms predominantly rely on coarse, high-level task instructions during supervised fine-tuning. This instruction grounding gap leaves models without explicit intermediate guidance, leading to severe compounding errors in long-horizon tasks. Therefore, bridging this instruction gap and providing scalable post-training for VLA models is urgent. To tackle this problem, we propose \method, the first subtask-aware VLA framework integrated with a scalable offline post-training pipeline. Our framework leverages a large language model to decompose high-level demonstrations into fine-grained atomic subtasks. This approach utilizes a pretrained predictive world model to score candidate action chunks against subtask goals in the latent space, mitigating error accumulation while significantly improving long-horizon robustness. Furthermore, this approach enables highly efficient Group Relative Policy Optimization without the prohibitive expenses associated with online rollouts on physical robots. Extensive simulations validate that our AtomVLA maintains strong robustness under perturbations. When evaluated against fundamental baseline models, it achieves an average success rate of 97.0\% on the LIBERO benchmark and 48.0\% on the LIBERO-PRO benchmark. Finally, experiments conducted in the real world using the Galaxea R1 Lite platform confirm its broad applicability across diverse tasks, especially long-horizon tasks. All datasets, checkpoints, and code will be released to the public domain following the acceptance of this work for future research.

  </details>



- **Spherical-GOF: Geometry-Aware Panoramic Gaussian Opacity Fields for 3D Scene Reconstruction**  
  Zhe Yang, Guoqiang Zhao, Sheng Wu, Kai Luo, Kailun Yang  
  _2026-03-09_ · https://arxiv.org/abs/2603.08503v1  
  <details><summary>Abstract</summary>

  Omnidirectional images are increasingly used in robotics and vision due to their wide field of view. However, extending 3D Gaussian Splatting (3DGS) to panoramic camera models remains challenging, as existing formulations are designed for perspective projections and naive adaptations often introduce distortion and geometric inconsistencies. We present Spherical-GOF, an omnidirectional Gaussian rendering framework built upon Gaussian Opacity Fields (GOF). Unlike projection-based rasterization, Spherical-GOF performs GOF ray sampling directly on the unit sphere in spherical ray space, enabling consistent ray-Gaussian interactions for panoramic rendering. To make the spherical ray casting efficient and robust, we derive a conservative spherical bounding rule for fast ray-Gaussian culling and introduce a spherical filtering scheme that adapts Gaussian footprints to distortion-varying panoramic pixel sampling. Extensive experiments on standard panoramic benchmarks (OmniBlender and OmniPhotos) demonstrate competitive photometric quality and substantially improved geometric consistency. Compared with the strongest baseline, Spherical-GOF reduces depth reprojection error by 57% and improves cycle inlier ratio by 21%. Qualitative results show cleaner depth and more coherent normal maps, with strong robustness to global panorama rotations. We further validate generalization on OmniRob, a real-world robotic omnidirectional dataset introduced in this work, featuring UAV and quadruped platforms. The source code and the OmniRob dataset will be released at https://github.com/1170632760/Spherical-GOF.

  </details>



- **Global Cross-Modal Geo-Localization: A Million-Scale Dataset and a Physical Consistency Learning Framework**  
  Yutong Hu, Jinhui Chen, Chaoqiang Xu, Yuan Kou, Sili Zhou, Shaocheng Yan, Pengcheng Shi, Qingwu Hu, Jiayuan Li  
  _2026-03-09_ · https://arxiv.org/abs/2603.08491v1  
  <details><summary>Abstract</summary>

  Cross-modal Geo-localization (CMGL) matches ground-level text descriptions with geo-tagged aerial imagery, which is crucial for pedestrian navigation and emergency response. However, existing researches are constrained by narrow geographic coverage and simplistic scene diversity, failing to reflect the immense spatial heterogeneity of global architectural styles and topographic features. To bridge this gap and facilitate universal positioning, we introduce CORE, the first million-scale dataset dedicated to global CMGL. CORE comprises 1,034,786 cross-view images sampled from 225 distinct geographic regions across all continents, offering an unprecedented variety of perspectives in varying environmental conditions and urban layouts. We leverage the zero-shot reasoning of Large Vision-Language Models (LVLMs) to synthesize high-quality scene descriptions rich in discriminative cues. Furthermore, we propose a physical-law-aware network (PLANET) for cross-modal geo-localization. PLANET introduces a novel contrastive learning paradigm to guide textual representations in capturing the intrinsic physical signatures of satellite imagery. Extensive experiments across varied geographic regions demonstrate that PLANet significantly outperforms state-of-the-art methods, establishing a new benchmark for robust, global-scale geo-localization. The dataset and source code will be released at https://github.com/YtH0823/CORE.

  </details>



- **An Open-Source Robotics Research Platform for Autonomous Laparoscopic Surgery**  
  Ariel Rodriguez, Lorenzo Mazza, Martin Lelis, Rayan Younis, Sebastian Bodenstedt, Martin Wagner, Stefanie Speidel  
  _2026-03-09_ · https://arxiv.org/abs/2603.08490v1  
  <details><summary>Abstract</summary>

  Autonomous robot-assisted surgery demands reliable, high-precision platforms that strictly adhere to the safety and kinematic constraints of minimally invasive procedures. Existing research platforms, primarily based on the da Vinci Research Kit, suffer from cable-driven mechanical limitations that degrade state-space consistency and hinder the downstream training of reliable autonomous policies. We present an open-source, robot-agnostic Remote Center of Motion (RCM) controller based on a closed-form analytical velocity solver that enforces the trocar constraint deterministically without iterative optimization. The controller operates in Cartesian space, enabling any industrial manipulator to function as a surgical robot. We provide implementations for the UR5e and Franka Emika Panda manipulators, and integrate stereoscopic 3D perception. We integrate the robot control into a full-stack ROS-based surgical robotics platform supporting teleoperation, demonstration recording, and deployment of learned policies via a decoupled server-client architecture. We validate the system on a bowel grasping and retraction task across phantom, ex vivo, and in vivo porcine laparoscopic procedures. RCM deviations remain sub-millimeter across all conditions, and trajectory smoothness metrics (SPARC, LDLJ) are comparable to expert demonstrations from the JIGSAWS benchmark recorded on the da Vinci system. These results demonstrate that the platform provides the precision and robustness required for teleoperation, data collection and autonomous policy deployment in realistic surgical scenarios.

  </details>



- **X-AVDT: Audio-Visual Cross-Attention for Robust Deepfake Detection**  
  Youngseo Kim, Kwan Yun, Seokhyeon Hong, Sihun Cha, Colette Suhjung Koo, Junyong Noh  
  _2026-03-09_ · https://arxiv.org/abs/2603.08483v1  
  <details><summary>Abstract</summary>

  The surge of highly realistic synthetic videos produced by contemporary generative systems has significantly increased the risk of malicious use, challenging both humans and existing detectors. Against this backdrop, we take a generator-side view and observe that internal cross-attention mechanisms in these models encode fine-grained speech-motion alignment, offering useful correspondence cues for forgery detection. Building on this insight, we propose X-AVDT, a robust and generalizable deepfake detector that probes generator-internal audio-visual signals accessed via DDIM inversion to expose these cues. X-AVDT extracts two complementary signals: (i) a video composite capturing inversion-induced discrepancies, and (ii) an audio-visual cross-attention feature reflecting modality alignment enforced during generation. To enable faithful cross-generator evaluation, we further introduce MMDF, a new multimodal deepfake dataset spanning diverse manipulation types and rapidly evolving synthesis paradigms, including GANs, diffusion, and flow-matching. Extensive experiments demonstrate that X-AVDT achieves leading performance on MMDF and generalizes strongly to external benchmarks and unseen generators, outperforming existing methods with accuracy improved by 13.1%. Our findings highlight the importance of leveraging internal audio-visual consistency cues for robustness to future generators in deepfake detection.

  </details>



- **LAR-MoE: Latent-Aligned Routing for Mixture of Experts in Robotic Imitation Learning**  
  Ariel Rodriguez, Chenpan Li, Lorenzo Mazza, Rayan Younis, Ortrun Hellig, Sebastian Bodenstedt, Martin Wagner, Stefanie Speidel  
  _2026-03-09_ · https://arxiv.org/abs/2603.08476v1  
  <details><summary>Abstract</summary>

  Imitation learning enables robots to acquire manipulation skills from demonstrations, yet deploying a policy across tasks with heterogeneous dynamics remains challenging, as models tend to average over distinct behavioral modes present in the demonstrations. Mixture-of-Experts (MoE) architectures address this by activating specialized subnetworks, but requires meaningful skill decompositions for expert routing. We introduce Latent-Aligned Routing for Mixture of Experts (LAR-MoE), a two-stage framework that decouples unsupervised skill discovery from policy learning. In pre-training, we learn a joint latent representation between observations and future actions through student-teacher co-training. In a post-training stage, the expert routing is regularized to follow the structure of the learned latent space, preventing expert collapse while maintaining parameter efficiency. We evaluate LAR-MoE in simulation and on hardware. On the LIBERO benchmark, our method achieves a 95.2% average success rate with 150M parameters. On a surgical bowel grasping and retraction task, LAR-MoE matches a supervised MoE baseline without requiring any phase annotations, and transfers zero-shot to ex vivo porcine tissue. Our findings suggest that latent-aligned routing provides a principled alternative to supervised skill decomposition, enabling structured expert specialization from unlabeled demonstrations.

  </details>



- **Alfa: Attentive Low-Rank Filter Adaptation for Structure-Aware Cross-Domain Personalized Gaze Estimation**  
  He-Yen Hsieh, Wei-Te Mark Ting, H. T. Kung  
  _2026-03-09_ · https://arxiv.org/abs/2603.08445v1  
  <details><summary>Abstract</summary>

  Pre-trained gaze models learn to identify useful patterns commonly found across users, but subtle user-specific variations (i.e., eyelid shape or facial structure) can degrade model performance. Test-time personalization (TTP) adapts pre-trained models to these user-specific domain shifts using only a few unlabeled samples. Efficient fine-tuning is critical in performing this domain adaptation: data and computation resources can be limited-especially for on-device customization. While popular parameter-efficient fine-tuning (PEFT) methods address adaptation costs by updating only a small set of weights, they may not be taking full advantage of structures encoded in pre-trained filters. To more effectively leverage existing structures learned during pre-training, we reframe personalization as a process to reweight existing features rather than learning entirely new ones. We present Attentive Low-Rank Filter Adaptation (Alfa) to adapt gaze models by reweighting semantic patterns in pre-trained filters. With Alfa, singular value decomposition (SVD) extracts dominant spatial components that capture eye and facial characteristics across users. Via an attention mechanism, we need only a few unlabeled samples to adjust and reweight pre-trained structures, selectively amplifying those relevant to a target user. Alfa achieves the lowest average gaze errors across four cross-dataset gaze benchmarks, outperforming existing TTP methods and low-rank adaptation (LoRA)-based variants. We also show that Alfa's attentive low-rank methods can be applied to applications beyond vision, such as diffusion-based language models.

  </details>



- **FoMo: A Multi-Season Dataset for Robot Navigation in Forêt Montmorency**  
  Matěj Boxan, Gabriel Jeanson, Alexander Krawciw, Effie Daum, Xinyuan Qiao, Sven Lilge, Timothy D. Barfoot, François Pomerleau  
  _2026-03-09_ · https://arxiv.org/abs/2603.08433v1  
  <details><summary>Abstract</summary>

  The Forêt Montmorency (FoMo) dataset is a comprehensive multi-season data collection, recorded over the span of one year in a boreal forest. Featuring a unique combination of on- and off-pavement environments with significant environmental changes, the dataset challenges established odometry and SLAM pipelines. Some highlights of the data include the accumulation of snow exceeding 1 m, significant vegetation growth in front of sensors, and operations at the traction limits of the platform. In total, the FoMo dataset includes over 64 km of six diverse trajectories, repeated during 12 deployments throughout the year. The dataset features data from one rotating and one hybrid solid-state lidar, a Frequency Modulated Continuous Wave (FMCW) radar, full-HD images from a stereo camera and a wide lens monocular camera, as well as data from two IMUs. Ground Truth is calculated by post-processing three GNSS receivers mounted on the Uncrewed Ground Vehicle (UGV) and a static GNSS base station. Additional metadata, such as one measurement per minute from an on-site weather station, camera calibration intrinsics, and vehicle power consumption, is available for all sequences. To highlight the relevance of the dataset, we performed a preliminary evaluation of the robustness of a lidar-inertial, radar-gyro, and a visual-inertial localization and mapping techniques to seasonal changes. We show that seasonal changes have serious effects on the re-localization capabilities of the state-of-the-art methods. The dataset and development kit are available at https://fomo.norlab.ulaval.ca.

  </details>



- **Tactile Recognition of Both Shapes and Materials with Automatic Feature Optimization-Enabled Meta Learning**  
  Hongliang Zhao, Wenhui Yang, Yang Chen, Zhuorui Wang, Baiheng Liu, Longhui Qin  
  _2026-03-09_ · https://arxiv.org/abs/2603.08423v1  
  <details><summary>Abstract</summary>

  Tactile perception is indispensable for robots to implement various manipulations dexterously, especially in contact-rich scenarios. However, alongside the development of deep learning techniques, it meanwhile suffers from training data scarcity and a time-consuming learning process in practical applications since the collection of a large amount of tactile data is costly and sometimes even impossible. Hence, we propose an automatic feature optimization-enabled prototypical network to realize meta-learning, i.e., AFOP-ML framework. As a ``learn to learn" network, it not only adapts to new unseen classes rapidly with few-shot, but also learns how to determine the optimal feature space automatically. Based on the four-channel signals acquired from a tactile finger, both shapes and materials are recognized. On a 36-category benchmark, it outperforms several existing approaches by attaining an accuracy of 96.08% in 5-way-1-shot scenario, where only 1 example is available for training. It still remains 88.7% in the extreme 36-way-1-shot case. The generalization ability is further validated through three groups of experiment involving unseen shapes, materials and force/speed perturbations. More insights are additionally provided by this work for the interpretation of recognition tasks and improved design of tactile sensors.

  </details>



- **SPIRAL: A Closed-Loop Framework for Self-Improving Action World Models via Reflective Planning Agents**  
  Yu Yang, Yue Liao, Jianbiao Mei, Baisen Wang, Xuemeng Yang, Licheng Wen, Jiangning Zhang, Xiangtai Li, Hanlin Chen, Botian Shi, et al.  
  _2026-03-09_ · https://arxiv.org/abs/2603.08403v1  
  <details><summary>Abstract</summary>

  We introduce SPIRAL, a self-improving planning and iterative reflective action world modeling closed-loop framework that enables controllable long-horizon video generation conditioned on high-level semantic actions. Existing one-shot video generation models operate in open-loop, often resulting in incomplete action execution, weak semantic grounding, and temporal drift. SPIRAL formulates ActWM as a closed-loop think-act-reflect process, where generation proceeds step by step under explicit planning and feedback. A PlanAgent decomposes abstract actions into object-centric sub-actions, while a CriticAgent evaluates intermediate results and guides iterative refinement with long-horizon memory. This closed-loop design naturally supports RL evolving optimization, improving semantic alignment and temporal consistency over extended horizons. We further introduce the ActWM-Dataset and ActWM-Bench for training and evaluation. Experiments across multiple TI2V backbones demonstrate consistent gains on ActWM-Bench and mainstream video generation benchmarks, validating SPIRAL's effectiveness.

  </details>



- **Rectified flow-based prediction of post-treatment brain MRI from pre-radiotherapy priors for patients with glioma**  
  Selena Huisman, Nordin Belkacemi, Vera Keil, Joost Verhoeff, Szabolcs David  
  _2026-03-09_ · https://arxiv.org/abs/2603.08385v1  
  <details><summary>Abstract</summary>

  Purpose/Objective: Brain tumors result in 20 years of lost life on average. Standard therapies induce complex structural changes in the brain that are monitored through MRI. Recent developments in artificial intelligence (AI) enable conditional multimodal image generation from clinical data. In this study, we investigate AI-driven generation of follow-up MRI in patients with in- tracranial tumors through conditional image generation. This approach enables realistic modeling of post-radiotherapy changes, allowing for treatment optimization. Material/Methods: The public SAILOR dataset of 25 patients was used to create a 2D rectified flow model conditioned on axial slices of pre-treatment MRI and RT dose maps. Cross-attention conditioning was used to incorporate temporal and chemotherapy data. The resulting images were validated with structural similarity index measure (SSIM), peak signal-to-noise ratio (PSNR), Dice scores and Jacobian determinants. Results: The resulting model generates realistic follow-up MRI for any time point, while integrating treatment information. Comparing real versus predicted images, SSIM is 0.88, and PSNR is 22.82. Tissue segmentations from real versus predicted MRI result in a mean Dice-Sørensen coefficient (DSC) of 0.91. The rectified flow (RF) model enables up to 250x faster inference than Denoising Diffusion Probabilistic Models (DDPM). Conclusion: The proposed model generates realistic follow-up MRI in real-time, preserving both semantic and visual fidelity as confirmed by image quality metrics and tissue segmentations. Conditional generation allows counterfactual simulations by varying treatment parameters, producing predicted morphological changes. This capability has potential to support adaptive treatment dose planning and personalized outcome prediction for patients with intracranial tumors.

  </details>



- **Diffusion-Based Data Augmentation for Image Recognition: A Systematic Analysis and Evaluation**  
  Zekun Li, Yinghuan Shi, Yang Gao, Dong Xu  
  _2026-03-09_ · https://arxiv.org/abs/2603.08364v1  
  <details><summary>Abstract</summary>

  Diffusion-based data augmentation (DiffDA) has emerged as a promising approach to improving classification performance under data scarcity. However, existing works vary significantly in task configurations, model choices, and experimental pipelines, making it difficult to fairly compare methods or assess their effectiveness across different scenarios. Moreover, there remains a lack of systematic understanding of the full DiffDA workflow. In this work, we introduce UniDiffDA, a unified analytical framework that decomposes DiffDA methods into three core components: model fine-tuning, sample generation, and sample utilization. This perspective enables us to identify key differences among existing methods and clarify the overall design space. Building on this framework, we develop a comprehensive and fair evaluation protocol, benchmarking representative DiffDA methods across diverse low-data classification tasks. Extensive experiments reveal the relative strengths and limitations of different DiffDA strategies and offer practical insights into method design and deployment. All methods are re-implemented within a unified codebase, with full release of code and configurations to ensure reproducibility and to facilitate future research.

  </details>



- **Local-Global Prompt Learning via Sparse Optimal Transport**  
  Deniz Kizaroğlu, Ülku Tuncer Küçüktas, Emre Çakmakyurdu, Alptekin Temizel  
  _2026-03-09_ · https://arxiv.org/abs/2603.08347v1  
  <details><summary>Abstract</summary>

  Few-shot adaptation of vision-language models (VLMs) like CLIP typically relies on learning textual prompts matched to global image embeddings. Recent works extend this paradigm by incorporating local image-text alignment to capture fine-grained visual cues, yet these approaches often select local regions independently for each prompt, leading to redundant local feature usage and prompt overlap. We propose SOT-GLP, which introduces a shared sparse patch support and balanced optimal transport allocation to explicitly partition salient visual regions among class-specific local prompts while preserving global alignment. Our method learns shared global prompts and class-specific local prompts. The global branch maintains standard image-text matching for robust category-level alignment. The local branch constructs a class-conditioned sparse patch set using V-V attention and aligns it to multiple class-specific prompts via balanced entropic optimal transport, yielding a soft partition of patches that prevents prompt overlap and collapse. We evaluate our method on two complementary objectives: (i) few-shot classification accuracy on 11 standard benchmarks and (ii) out-of-distribution (OOD) detection. On the standard 11-dataset benchmark with 16-shot ViT-B/16, SOT-GLP achieves 85.1% average accuracy, outperforming prior prompt-learning methods. We identify a distinct accuracy-robustness trade-off in prompt learning: while learnable projections optimize in-distribution fit, they alter the foundational feature space. We demonstrate that a projection-free local alignment preserves the native geometry of the CLIP manifold, yielding state-of-the-art OOD detection performance (94.2% AUC) that surpasses fully adapted models. Implementation available at: https://github.com/Deniz2304988/SOT-GLP

  </details>



- **Beyond Attention Heatmaps: How to Get Better Explanations for Multiple Instance Learning Models in Histopathology**  
  Mina Jamshidi Idaji, Julius Hense, Tom Neuhäuser, Augustin Krause, Yanqing Luo, Oliver Eberle, Thomas Schnake, Laure Ciernik, Farnoush Rezaei Jafari, Reza Vahidimajd, et al.  
  _2026-03-09_ · https://arxiv.org/abs/2603.08328v1  
  <details><summary>Abstract</summary>

  Multiple instance learning (MIL) has enabled substantial progress in computational histopathology, where a large amount of patches from gigapixel whole slide images are aggregated into slide-level predictions. Heatmaps are widely used to validate MIL models and to discover tissue biomarkers. Yet, the validity of these heatmaps has barely been investigated. In this work, we introduce a general framework for evaluating the quality of MIL heatmaps without requiring additional labels. We conduct a large-scale benchmark experiment to assess six explanation methods across histopathology task types (classification, regression, survival), MIL model architectures (Attention-, Transformer-, Mamba-based), and patch encoder backbones (UNI2, Virchow2). Our results show that explanation quality mostly depends on MIL model architecture and task type, with perturbation ("Single"), layer-wise relevance propagation (LRP), and integrated gradients (IG) consistently outperforming attention-based and gradient-based saliency heatmaps, which often fail to reflect model decision mechanisms. We further demonstrate the advanced capabilities of the best-performing explanation methods: (i) We provide a proof-of-concept that MIL heatmaps of a bulk gene expression prediction model can be correlated with spatial transcriptomics for biological validation, and (ii) showcase the discovery of distinct model strategies for predicting human papillomavirus (HPV) infection from head and neck cancer slides. Our work highlights the importance of validating MIL heatmaps and establishes that improved explainability can enable more reliable model validation and yield biological insights, making a case for a broader adoption of explainable AI in digital pathology. Our code is provided in a public GitHub repository: https://github.com/bifold-pathomics/xMIL/tree/xmil-journal

  </details>



- **Human-AI Divergence in Ego-centric Action Recognition under Spatial and Spatiotemporal Manipulations**  
  Sadegh Rahmaniboldaji, Filip Rybansky, Quoc C. Vuong, Anya C. Hurlbert, Frank Guerin, Andrew Gilbert  
  _2026-03-09_ · https://arxiv.org/abs/2603.08317v1  
  <details><summary>Abstract</summary>

  Humans consistently outperform state-of-the-art AI models in action recognition, particularly in challenging real-world conditions involving low resolution, occlusion, and visual clutter. Understanding the sources of this performance gap is essential for developing more robust and human-aligned models. In this paper, we present a large-scale human-AI comparative study of egocentric action recognition using Minimal Identifiable Recognition Crops (MIRCs), defined as the smallest spatial or spatiotemporal regions sufficient for reliable human recognition. We used our previously introduced, Epic ReduAct, a systematically spatially reduced and temporally scrambled dataset derived from 36 EPIC KITCHENS videos, spanning multiple spatial reduction levels and temporal conditions. Recognition performance is evaluated using over 3,000 human participants and the Side4Video model. Our analysis combines quantitative metrics, Average Reduction Rate and Recognition Gap, with qualitative analyses of spatial (high-, mid-, and low-level visual features) and spatiotemporal factors, including a categorisation of actions into Low Temporal Actions (LTA) and High Temporal Actions (HTA). Results show that human performance exhibits sharp declines when transitioning from MIRCs to subMIRCs, reflecting a strong reliance on sparse, semantically critical cues such as hand-object interactions. In contrast, the model degrades more gradually and often relies on contextual and mid- to low-level features, sometimes even exhibiting increased confidence under spatial reduction. Temporally, humans remain robust to scrambling when key spatial cues are preserved, whereas the model often shows insensitivity to temporal disruption, revealing class-dependent temporal sensitivities.

  </details>



- **HDR-NSFF: High Dynamic Range Neural Scene Flow Fields**  
  Shin Dong-Yeon, Kim Jun-Seong, Kwon Byung-Ki, Tae-Hyun Oh  
  _2026-03-09_ · https://arxiv.org/abs/2603.08313v1  
  <details><summary>Abstract</summary>

  Radiance of real-world scenes typically spans a much wider dynamic range than what standard cameras can capture. While conventional HDR methods merge alternating-exposure frames, these approaches are inherently constrained to 2D pixel-level alignment, often leading to ghosting artifacts and temporal inconsistency in dynamic scenes. To address these limitations, we present HDR-NSFF, a paradigm shift from 2D-based merging to 4D spatio-temporal modeling. Our framework reconstructs dynamic HDR radiance fields from alternating-exposure monocular videos by representing the scene as a continuous function of space and time, and is compatible with both neural radiance field and 4D Gaussian Splatting (4DGS) based dynamic representations. This unified end-to-end pipeline explicitly models HDR radiance, 3D scene flow, geometry, and tone-mapping, ensuring physical plausibility and global coherence. We further enhance robustness by (i) extending semantic-based optical flow with DINO features to achieve exposure-invariant motion estimation, and (ii) incorporating a generative prior as a regularizer to compensate for limited observation in monocular captures and saturation-induced information loss. To evaluate HDR space-time view synthesis, we present the first real-world HDR-GoPro dataset specifically designed for dynamic HDR scenes. Experiments demonstrate that HDR-NSFF recovers fine radiance details and coherent dynamics even under challenging exposure variations, thereby achieving state-of-the-art performance in novel space-time view synthesis. Project page: https://shin-dong-yeon.github.io/HDR-NSFF/

  </details>



- **Concept-Guided Fine-Tuning: Steering ViTs away from Spurious Correlations to Improve Robustness**  
  Yehonatan Elisha, Oren Barkan, Noam Koenigstein  
  _2026-03-09_ · https://arxiv.org/abs/2603.08309v1  
  <details><summary>Abstract</summary>

  Vision Transformers (ViTs) often degrade under distribution shifts because they rely on spurious correlations, such as background cues, rather than semantically meaningful features. Existing regularization methods, typically relying on simple foreground-background masks, which fail to capture the fine-grained semantic concepts that define an object (e.g., ``long beak'' and ``wings'' for a ``bird''). As a result, these methods provide limited robustness to distribution shifts. To address this limitation, we introduce a novel finetuning framework that steers model reasoning toward concept-level semantics. Our approach optimizes the model's internal relevance maps to align with spatially grounded concept masks. These masks are generated automatically, without manual annotation: class-relevant concepts are first proposed using an LLM-based, label-free method, and then segmented using a VLM. The finetuning objective aligns relevance with these concept regions while simultaneously suppressing focus on spurious background areas. Notably, this process requires only a minimal set of images and uses half of the dataset classes. Extensive experiments on five out-of-distribution benchmarks demonstrate that our method improves robustness across multiple ViT-based models. Furthermore, we show that the resulting relevance maps exhibit stronger alignment with semantic object parts, offering a scalable path toward more robust and interpretable vision models. Finally, we confirm that concept-guided masks provide more effective supervision for model robustness than conventional segmentation maps, supporting our central hypothesis.

  </details>



- **Retrieval-Augmented Anatomical Guidance for Text-to-CT Generation**  
  Daniele Molino, Camillo Maria Caruso, Paolo Soda, Valerio Guarrasi  
  _2026-03-09_ · https://arxiv.org/abs/2603.08305v1  
  <details><summary>Abstract</summary>

  Text-conditioned generative models for volumetric medical imaging provide semantic control but lack explicit anatomical guidance, often resulting in outputs that are spatially ambiguous or anatomically inconsistent. In contrast, structure-driven methods ensure strong anatomical consistency but typically assume access to ground-truth annotations, which are unavailable when the target image is to be synthesized. We propose a retrieval-augmented approach for Text-to-CT generation that integrates semantic and anatomical information under a realistic inference setting. Given a radiology report, our method retrieves a semantically related clinical case using a 3D vision-language encoder and leverages its associated anatomical annotation as a structural proxy. This proxy is injected into a text-conditioned latent diffusion model via a ControlNet branch, providing coarse anatomical guidance while maintaining semantic flexibility. Experiments on the CT-RATE dataset show that retrieval-augmented generation improves image fidelity and clinical consistency compared to text-only baselines, while additionally enabling explicit spatial controllability, a capability inherently absent in such approaches. Further analysis highlights the importance of retrieval quality, with semantically aligned proxies yielding consistent gains across all evaluation axes. This work introduces a principled and scalable mechanism to bridge semantic conditioning and anatomical plausibility in volumetric medical image synthesis. Code will be released.

  </details>



- **Exploring Deep Learning and Ultra-Widefield Imaging for Diabetic Retinopathy and Macular Edema**  
  Pablo Jimenez-Lizcano, Sergio Romero-Tapiador, Ruben Tolosana, Aythami Morales, Guillermo González de Rivera, Ruben Vera-Rodriguez, Julian Fierrez  
  _2026-03-09_ · https://arxiv.org/abs/2603.08235v1  
  <details><summary>Abstract</summary>

  Diabetic retinopathy (DR) and diabetic macular edema (DME) are leading causes of preventable blindness among working-age adults. Traditional approaches in the literature focus on standard color fundus photography (CFP) for the detection of these conditions. Nevertheless, recent ultra-widefield imaging (UWF) offers a significantly wider field of view in comparison to CFP. Motivated by this, the present study explores state-of-the-art deep learning (DL) methods and UWF imaging on three clinically relevant tasks: i) image quality assessment for UWF, ii) identification of referable diabetic retinopathy (RDR), and iii) identification of DME. Using the publicly available UWF4DR Challenge dataset, released as part of the MICCAI 2024 conference, we benchmark DL models in the spatial (RGB) and frequency domains, including popular convolutional neural networks (CNNs) as well as recent vision transformers (ViTs) and foundation models. In addition, we explore a final feature-level fusion to increase robustness. Finally, we also analyze the decisions of the DL models using Grad-CAM, increasing the explainability. Our proposal achieves consistently strong performance across all architectures, underscoring the competitiveness of emerging ViTs and foundation models and the promise of feature-level fusion and frequency-domain representations for UWF analysis.

  </details>



- **Alignment-Aware and Reliability-Gated Multimodal Fusion for Unmanned Aerial Vehicle Detection Across Heterogeneous Thermal-Visual Sensors**  
  Ishrat Jahan, Molla E Majid, M Murugappan, Muhammad E. H. Chowdhury, N. B. Prakash, Saad Bin Abul Kashem, Balamurugan Balusamy, Amith Khandakar  
  _2026-03-09_ · https://arxiv.org/abs/2603.08208v1  
  <details><summary>Abstract</summary>

  Reliable unmanned aerial vehicle (UAV) detection is critical for autonomous airspace monitoring but remains challenging when integrating sensor streams that differ substantially in resolution, perspective, and field of view. Conventional fusion methods-such as wavelet-, Laplacian-, and decision-level approaches-often fail to preserve spatial correspondence across modalities and suffer from annotation of inconsistencies, limiting their robustness in real-world settings. This study introduces two fusion strategies, Registration-aware Guided Image Fusion (RGIF) and Reliability-Gated Modality-Attention Fusion (RGMAF), designed to overcome these limitations. RGIF employs Enhanced Correlation Coefficient (ECC)-based affine registration combined with guided filtering to maintain thermal saliency while enhancing structural detail. RGMAF integrates affine and optical-flow registration with a reliability-weighted attention mechanism that adaptively balances thermal contrast and visual sharpness. Experiments were conducted on the Multi-Sensor and Multi-View Fixed-Wing (MMFW)-UAV dataset comprising 147,417 annotated air-to-air frames collected from infrared, wide-angle, and zoom sensors. Among single-modality detectors, YOLOv10x demonstrated the most stable cross-domain performance and was selected as the detection backbone for evaluating fused imagery. RGIF improved the visual baseline by 2.13% mAP@50 (achieving 97.65%), while RGMAF attained the highest recall of 98.64%. These findings show that registration-aware and reliability-adaptive fusion provides a robust framework for integrating heterogeneous modalities, substantially enhancing UAV detection performance in multimodal environments.

  </details>



- **ALOOD: Exploiting Language Representations for LiDAR-based Out-of-Distribution Object Detection**  
  Michael Kösel, Marcel Schreiber, Michael Ulrich, Claudius Gläser, Klaus Dietmayer  
  _2026-03-09_ · https://arxiv.org/abs/2603.08180v1  
  <details><summary>Abstract</summary>

  LiDAR-based 3D object detection plays a critical role for reliable and safe autonomous driving systems. However, existing detectors often produce overly confident predictions for objects not belonging to known categories, posing significant safety risks. This is caused by so-called out-of-distribution (OOD) objects, which were not part of the training data, resulting in incorrect predictions. To address this challenge, we propose ALOOD (Aligned LiDAR representations for Out-Of-Distribution Detection), a novel approach that incorporates language representations from a vision-language model (VLM). By aligning the object features from the object detector to the feature space of the VLM, we can treat the detection of OOD objects as a zero-shot classification task. We demonstrate competitive performance on the nuScenes OOD benchmark, establishing a novel approach to OOD object detection in LiDAR using language representations. The source code is available at https://github.com/uulm-mrm/mmood3d.

  </details>



- **MERLIN: Building Low-SNR Robust Multimodal LLMs for Electromagnetic Signals**  
  Junyu Shen, Zhendong She, Chenghanyu Zhang, Yuchuang Sun, Luqing Luo, Dingwei Tan, Zonghao Guo, Bo Guo, Zehua Han, Wupeng Xie, et al.  
  _2026-03-09_ · https://arxiv.org/abs/2603.08174v1  
  <details><summary>Abstract</summary>

  The paradigm of Multimodal Large Language Models (MLLMs) offers a promising blueprint for advancing the electromagnetic (EM) domain. However, prevailing approaches often deviate from the native MLLM paradigm, instead using task-specific or pipelined architectures that lead to fundamental limitations in model performance and generalization. Fully realizing the MLLM potential in EM domain requires overcoming three main challenges: (1) Data. The scarcity of high-quality datasets with paired EM signals and descriptive text annotations used for MLLMs pre-training; (2) Benchmark. The absence of comprehensive benchmarks to systematically evaluate and compare the performance of models on EM signal-to-text tasks; (3) Model. A critical fragility in low Signal-to-Noise Ratio (SNR) environments, where critical signal features can be obscured, leading to significant performance degradation. To address these challenges, we introduce a tripartite contribution to establish a foundation for MLLMs in the EM domain. First, to overcome data scarcity, we construct and release EM-100k, a large-scale dataset comprising over 100,000 EM signal-text pairs. Second, to enable rigorous and standardized evaluation, we propose EM-Bench, the most comprehensive benchmark featuring diverse downstream tasks spanning from perception to reasoning. Finally, to tackle the core modeling challenge, we present MERLIN, a novel training framework designed not only to align low-level signal representations with high-level semantic text, but also to explicitly enhance model robustness and performance in challenging low-SNR environments. Comprehensive experiments validate our method, showing that MERLIN is state-of-the-art in the EM-Bench and exhibits remarkable robustness in low-SNR settings.

  </details>



- **MV-Fashion: Towards Enabling Virtual Try-On and Size Estimation with Multi-View Paired Data**  
  Hunor Laczkó, Libang Jia, Loc-Phat Truong, Diego Hernández, Sergio Escalera, Jordi Gonzalez, Meysam Madadi  
  _2026-03-09_ · https://arxiv.org/abs/2603.08147v1  
  <details><summary>Abstract</summary>

  Existing 4D human datasets fall short for fashion-specific research, lacking either realistic garment dynamics or task-specific annotations. Synthetic datasets suffer from a realism gap, whereas real-world captures lack the detailed annotations and paired data required for virtual try-on (VTON) and size estimation tasks. To bridge this gap, we introduce MV-Fashion, a large-scale, multi-view video dataset engineered for domain-specific fashion analysis. MV-Fashion features 3,273 sequences (72.5 million frames) from 80 diverse subjects wearing 3-10 outfits each. It is designed to capture complex, real-world garment dynamics, including multiple layers and varied styling (e.g. rolled sleeves, tucked shirt). A core contribution is a rich data representation that includes pixel-level semantic annotations, ground-truth material properties like elasticity, and 3D point clouds. Crucially for VTON applications, MV-Fashion provides paired data: multi-view synchronized captures of worn garments alongside their corresponding flat, catalogue images. We leverage this dataset to establish baselines for fashion-centric tasks, including virtual try-on, clothing size estimation, and novel view synthesis. The dataset is available at https://hunorlaczko.github.io/MV-Fashion .

  </details>



- **Multifingered force-aware control for humanoid robots**  
  Pasquale Marra, Gabriele M. Caddeo, Ugo Pattacini, Lorenzo Natale  
  _2026-03-09_ · https://arxiv.org/abs/2603.08142v1  
  <details><summary>Abstract</summary>

  In this paper, we address force-aware control and force distribution in robotic platforms with multi-fingered hands. Given a target goal and force estimates from tactile sensors, we design a controller that adapts the motion of the torso, arm, wrist, and fingers, redistributing forces to maintain stable contact with objects of varying mass distribution or unstable contacts. To estimate forces, we collect a dataset of tactile signals and ground-truth force measurements using five Xela magnetic sensors interacting with indenters, and train force estimators. We then introduce a model-based control scheme that minimizes the distance between the Center of Pressure (CoP) and the centroid of the fingertips contact polygon. Since our method relies on estimated forces rather than raw tactile signals, it has the potential to be applied to any sensor capable of force estimation. We validate our framework on a balancing task with five objects, achieving a $82.7\%$ success rate, and further evaluate it in multi-object scenarios, achieving $80\%$ accuracy. Code and data can be found here https://github.com/hsp-iit/multifingered-force-aware-control.

  </details>



- **VesselFusion: Diffusion Models for Vessel Centerline Extraction from 3D CT Images**  
  Soichi Mita, Shumpei Takezaki, Ryoma Bise  
  _2026-03-09_ · https://arxiv.org/abs/2603.08135v1  
  <details><summary>Abstract</summary>

  Vessel centerline extraction from 3D CT images is an important task because it reduces annotation effort to build a model that estimates a vessel structure. It is challenging to estimate natural vessel structures since conventional approaches are deterministic models, which cannot capture a complex human structure. In this study, we propose VesselFusion, which is a diffusion model to extract the vessel centerline from 3D CT image. The proposed method uses a coarse-to-fine representation of the centerline and a voting-based aggregation for a natural and stable extraction. VesselFusion was evaluated on a publicly available CT image dataset and achieved higher extraction accuracy and a more natural result than conventional approaches.

  </details>



- **SAMoE-VLA: A Scene Adaptive Mixture-of-Experts Vision-Language-Action Model for Autonomous Driving**  
  Zihan You, Hongwei Liu, Chenxu Dang, Zhe Wang, Sining Ang, Aoqi Wang, Yan Wang  
  _2026-03-09_ · https://arxiv.org/abs/2603.08113v1  
  <details><summary>Abstract</summary>

  Recent advances in Vision-Language-Action (VLA) models have shown promising capabilities in autonomous driving by leveraging the understanding and reasoning strengths of Large Language Models(LLMs).However, our empirical analysis reveals that directly applying existing token-level MoE mechanisms--which are inherited from LLM architectures--to VLA models results in unstable performance and safety degradation in autonomous driving, highlighting a misalignment between token-based expert specialization and scene-level decision-making.To address this, we propose SAMoE-VLA, a scene-adaptive Vision-Language-Action framework that conditions expert selection on structured scene representations instead of token embeddings. Our key idea is to derive the MoE routing signal from bird's-eye-view (BEV) features that encapsulates traffic scene context, enabling scenario-dependent expert weighting and merging tailored to distinct driving conditions. Furthermore, to support temporally consistent reasoning across world-knowledge, perception, language, and action, we introduce a Conditional Cross-Modal Causal Attention mechanism that integrates world state, linguistic intent, and action history into a unified causal reasoning process. Extensive experiments on the nuScenes open loop planning dataset and LangAuto closed-loop benchmark demonstrate that SAMoE-VLA achieves state-of-the-art performance, outperforming prior VLA-based and world-model-based approaches with fewer parameters.Our code will be released soon.

  </details>



- **DSH-Bench: A Difficulty- and Scenario-Aware Benchmark with Hierarchical Subject Taxonomy for Subject-Driven Text-to-Image Generation**  
  Zhenyu Hu, Qing Wang, Te Cao, Luo Liao, Longfei Lu, Liqun Liu, Shuang Li, Hang Chen, Mengge Xue, Yuan Chen, et al.  
  _2026-03-09_ · https://arxiv.org/abs/2603.08090v1  
  <details><summary>Abstract</summary>

  Significant progress has been achieved in subject-driven text-to-image (T2I) generation, which aims to synthesize new images depicting target subjects according to user instructions. However, evaluating these models remains a significant challenge. Existing benchmarks exhibit critical limitations: 1) insufficient diversity and comprehensiveness in subject images, 2) inadequate granularity in assessing model performance across different subject difficulty levels and prompt scenarios, and 3) a profound lack of actionable insights and diagnostic guidance for subsequent model refinement. To address these limitations, we propose DSH-Bench, a comprehensive benchmark that enables systematic multi-perspective analysis of subject-driven T2I models through four principal innovations: 1) a hierarchical taxonomy sampling mechanism ensuring comprehensive subject representation across 58 fine-grained categories, 2) an innovative classification scheme categorizing both subject difficulty level and prompt scenario for granular capability assessment, 3) a novel Subject Identity Consistency Score (SICS) metric demonstrating a 9.4\% higher correlation with human evaluation compared to existing measures in quantifying subject preservation, and 4) a comprehensive set of diagnostic insights derived from the benchmark, offering critical guidance for optimizing future model training paradigms and data construction strategies. Through an extensive empirical evaluation of 19 leading models, DSH-Bench uncovers previously obscured limitations in current approaches, establishing concrete directions for future research and development.

  </details>



- **Synthetic Defect Image Generation for Power Line Insulator Inspection Using Multimodal Large Language Models**  
  Xuesong Wang, Caisheng Wang  
  _2026-03-09_ · https://arxiv.org/abs/2603.08069v1  
  <details><summary>Abstract</summary>

  Utility companies increasingly rely on drone imagery for post-event and routine inspection, but training accurate defect-type classifiers remains difficult because defect examples are rare and inspection datasets are often limited or proprietary. We address this data-scarcity setting by using an off-the-shelf multimodal large language model (MLLM) as a training-free image generator to synthesize defect images from visual references and text prompts. Our pipeline increases diversity via dual-reference conditioning, improves label fidelity with lightweight human verification and prompt refinement, and filters the resulting synthetic pool using an embedding-based selection rule based on distances to class centroids computed from the real training split. We evaluate on ceramic insulator defect-type classification (shell vs. glaze) using a public dataset with a realistic low training-data regime (104 real training images; 152 validation; 308 test). Augmenting the 10% real training set with embedding-selected synthetic images improves test F1 score (harmonic mean of precision and recall) from 0.615 to 0.739 (20% relative), corresponding to an estimated 4--5x data-efficiency gain, and the gains persist with stronger backbone models and frozen-feature linear-probe baselines. These results suggest a practical, low-barrier path for improving defect recognition when collecting additional real defects is slow or infeasible.

  </details>



- **Evaluating Generative Models via One-Dimensional Code Distributions**  
  Zexi Jia, Pengcheng Luo, Yijia Zhong, Jinchao Zhang, Jie Zhou  
  _2026-03-09_ · https://arxiv.org/abs/2603.08064v1  
  <details><summary>Abstract</summary>

  Most evaluations of generative models rely on feature-distribution metrics such as FID, which operate on continuous recognition features that are explicitly trained to be invariant to appearance variations, and thus discard cues critical for perceptual quality. We instead evaluate models in the space of \emph{discrete} visual tokens, where modern 1D image tokenizers compactly encode both semantic and perceptual information and quality manifests as predictable token statistics. We introduce \emph{Codebook Histogram Distance} (CHD), a training-free distribution metric in token space, and \emph{Code Mixture Model Score} (CMMS), a no-reference quality metric learned from synthetic degradations of token sequences. To stress-test metrics under broad distribution shifts, we further propose \emph{VisForm}, a benchmark of 210K images spanning 62 visual forms and 12 generative models with expert annotations. Across AGIQA, HPDv2/3, and VisForm, our token-based metrics achieve state-of-the-art correlation with human judgments, and we will release all code and datasets to facilitate future research.

  </details>



- **Solution to the 10th ABAW Expression Recognition Challenge: A Robust Multimodal Framework with Safe Cross-Attention and Modality Dropout**  
  Jun Yu, Naixiang Zheng, Guoyuan Wang, Yunxiang Zhang, Lingsi Zhu, Jiaen Liang, Wei Huang, Shengping Liu  
  _2026-03-09_ · https://arxiv.org/abs/2603.08034v1  
  <details><summary>Abstract</summary>

  Emotion recognition in real-world environments is hindered by partial occlusions, missing modalities, and severe class imbalance. To address these issues, particularly for the Affective Behavior Analysis in-the-wild (ABAW) Expression challenge, we propose a multimodal framework that dynamically fuses visual and audio representations. Our approach uses a dual-branch Transformer architecture featuring a safe cross-attention mechanism and a modality dropout strategy. This design allows the network to rely on audio-based predictions when visual cues are absent. To mitigate the long-tail distribution of the Aff-Wild2 dataset, we apply focal loss optimization, combined with a sliding-window soft voting strategy to capture dynamic emotional transitions and reduce frame-level classification jitter. Experiments demonstrate that our framework effectively handles missing modalities and complex spatiotemporal dependencies, achieving an accuracy of 60.79% and an F1-score of 0.5029 on the Aff-Wild2 validation set.

  </details>



- **Controllable Complex Human Motion Video Generation via Text-to-Skeleton Cascades**  
  Ashkan Taghipour, Morteza Ghahremani, Zinuo Li, Hamid Laga, Farid Boussaid, Mohammed Bennamoun  
  _2026-03-09_ · https://arxiv.org/abs/2603.08028v1  
  <details><summary>Abstract</summary>

  Generating videos of complex human motions such as flips, cartwheels, and martial arts remains challenging for current video diffusion models. Text-only conditioning is temporally ambiguous for fine-grained motion control, while explicit pose-based controls, though effective, require users to provide complete skeleton sequences that are costly to produce for long and dynamic actions. We propose a two-stage cascaded framework that addresses both limitations. First, an autoregressive text-to-skeleton model generates 2D pose sequences from natural language descriptions by predicting each joint conditioned on previously generated poses. This design captures long-range temporal dependencies and inter-joint coordination required for complex motions. Second, a pose-conditioned video diffusion model synthesizes videos from a reference image and the generated skeleton sequence. It employs DINO-ALF (Adaptive Layer Fusion), a multi-level reference encoder that preserves appearance and clothing details under large pose changes and self-occlusions. To address the lack of publicly available datasets for complex human motion video generation, we introduce a Blender-based synthetic dataset containing 2,000 videos with diverse characters performing acrobatic and stunt-like motions. The dataset provides full control over appearance, motion, and environment. It fills an important gap because existing benchmarks significantly under-represent acrobatic motions while web-collected datasets raise copyright and privacy concerns. Experiments on our synthetic dataset and the Motion-X Fitness benchmark show that our text-to-skeleton model outperforms prior methods on FID, R-precision, and motion diversity. Our pose-to-video model also achieves the best results among all compared methods on VBench metrics for temporal consistency, motion smoothness, and subject preservation.

  </details>



- **AffordGrasp: Cross-Modal Diffusion for Affordance-Aware Grasp Synthesis**  
  Xiaofei Wu, Yi Zhang, Yumeng Liu, Yuexin Ma, Yujiao Shi, Xuming He  
  _2026-03-09_ · https://arxiv.org/abs/2603.08021v1  
  <details><summary>Abstract</summary>

  Generating human grasping poses that accurately reflect both object geometry and user-specified interaction semantics is essential for natural hand-object interactions in AR/VR and embodied AI. However, existing semantic grasping approaches struggle with the large modality gap between 3D object representations and textual instructions, and often lack explicit spatial or semantic constraints, leading to physically invalid or semantically inconsistent grasps. In this work, we present AffordGrasp, a diffusion-based framework that produces physically stable and semantically faithful human grasps with high precision. We first introduce a scalable annotation pipeline that automatically enriches hand-object interaction datasets with fine-grained structured language labels capturing interaction intent. Building upon these annotations, AffordGrasp integrates an affordance-aware latent representation of hand poses with a dual-conditioning diffusion process, enabling the model to jointly reason over object geometry, spatial affordances, and instruction semantics. A distribution adjustment module further enforces physical contact consistency and semantic alignment. We evaluate AffordGrasp across four instruction-augmented benchmarks derived from HO-3D, OakInk, GRAB, and AffordPose, and observe substantial improvements over state-of-the-art methods in grasp quality, semantic accuracy, and diversity.

  </details>



- **VSDiffusion: Taming Ill-Posed Shadow Generation via Visibility-Constrained Diffusion**  
  Jing Li, Jing Zhang  
  _2026-03-09_ · https://arxiv.org/abs/2603.08020v1  
  <details><summary>Abstract</summary>

  Generating realistic cast shadows for inserted foreground objects is a crucial yet challenging problem in image composition, where maintaining geometric consistency of shadow and object in complex scenes remains difficult due to the ill-posed nature of shadow formation. To address this issue, we propose VSDiffusion, a visibility-constrained two-stage framework designed to narrow the solution space by incorporating visibility priors. In Stage I, we predict a coarse shadow mask to localize plausible shadow generated regions. And in Stage II, conditional diffusion is performed guided by lighting and depth cues estimated from the composite to generate accurate shadows. In VSDiffusion, we inject visibility priors through two complementary pathways. First, a visibility control branch with shadow-gated cross attention that provides multi-scale structural guidance. Then, a learned soft prior map that reweights training loss in error-prone regions to enhance geometric correction. Additionally, we also introduce high-frequency guided enhancement module to sharpen boundaries and improve texture interaction with the background. Experiments on widely used public DESOBAv2 dataset demonstrated that our proposed VSDiffusion can generate accurate shadow, and establishes new SOTA results across most evaluation metrics.

  </details>



- **It's Time to Get It Right: Improving Analog Clock Reading and Clock-Hand Spatial Reasoning in Vision-Language Models**  
  Jaeha Choi, Jin Won Lee, Siwoo You, Jangho Lee  
  _2026-03-09_ · https://arxiv.org/abs/2603.08011v1  
  <details><summary>Abstract</summary>

  Advances in vision-language models (VLMs) have achieved remarkable success on complex multimodal reasoning tasks, leading to the assumption that they should also excel at reading analog clocks. However, contrary to this expectation, our study reveals that reading analog clocks in real-world environments remains a significant challenge for state-of-the-art VLMs. Existing analog clock datasets are largely synthetic or planar with limited stylistic diversity and minimal background context, failing to capture the visual variability of real-world scenes. As a result, VLMs trained on such data exhibit weak spatial-temporal reasoning, frequently confusing the hour and minute hands and struggling under common visual conditions such as occlusion, lighting variation, and cluttered backgrounds. To address this issue, we introduce TickTockVQA, a human-annotated dataset containing analog clocks in diverse real-world scenarios. TickTockVQA provides explicit hour and minute annotations, and includes an AM/PM tag when it is inferable from the visual context. Furthermore, we propose Swap-DPO, a direct preference optimization based fine-tuning framework to align model reasoning toward accurate time interpretation. Experimental results demonstrate that our approach substantially enhances clock reading accuracy and robustness under real-world conditions, establishing a foundation for future research on spatial-temporal reasoning and visual understanding in VLMs.

  </details>



- **ViSA-Enhanced Aerial VLN: A Visual-Spatial Reasoning Enhanced Framework for Aerial Vision-Language Navigation**  
  Haoyu Tong, Xiangyu Dong, Xiaoguang Ma, Haoran Zhao, Yaoming Zhou, Chenghao Lin  
  _2026-03-09_ · https://arxiv.org/abs/2603.08007v1  
  <details><summary>Abstract</summary>

  Existing aerial Vision-Language Navigation (VLN) methods predominantly adopt a detection-and-planning pipeline, which converts open-vocabulary detections into discrete textual scene graphs. These approaches are plagued by inadequate spatial reasoning capabilities and inherent linguistic ambiguities. To address these bottlenecks, we propose a Visual-Spatial Reasoning (ViSA) enhanced framework for aerial VLN. Specifically, a triple-phase collaborative architecture is designed to leverage structured visual prompting, enabling Vision-Language Models (VLMs) to perform direct reasoning on image planes without the need for additional training or complex intermediate representations. Comprehensive evaluations on the CityNav benchmark demonstrate that the ViSA-enhanced VLN achieves a 70.3\% improvement in success rate compared to the fully trained state-of-the-art (SOTA) method, elucidating its great potential as a backbone for aerial VLN systems.

  </details>



- **AutoTraces: Autoregressive Trajectory Forecasting via Multimodal Large Language Models**  
  Teng Wang, Yanting Lu, Ruize Wang  
  _2026-03-09_ · https://arxiv.org/abs/2603.07989v1  
  <details><summary>Abstract</summary>

  We present AutoTraces, an autoregressive vision-language-trajectory model for robot trajectory forecasting in humam-populated environments, which harnesses the inherent reasoning capabilities of large language models (LLMs) to model complex human behaviors. In contrast to prior works that rely solely on textual representations, our key innovation lies in a novel trajectory tokenization scheme, which represents waypoints with point tokens as categorical and positional markers while encoding waypoint numerical values as corresponding point embeddings, seamlessly integrated into the LLM's space through a lightweight encoder-decoder architecture. This design preserves the LLM's native autoregressive generation mechanism while extending it to physical coordinate spaces, facilitates modeling of long-term interactions in trajectory data. We further introduce an automated chain-of-thought (CoT) generation mechanism that leverages a multimodal LLM to infer spatio-temporal relationships from visual observations and trajectory data, eliminating reliance on manual annotation. Through a two-stage training strategy, our AutoTraces achieves SOTA forecasting accuracy, particularly in long-horizon prediction, while exhibiting strong cross-scene generalization and supporting flexible-length forecasting.

  </details>



- **Listening with the Eyes: Benchmarking Egocentric Co-Speech Grounding across Space and Time**  
  Weijie Zhou, Xuantang Xiong, Zhenlin Hu, Xiaomeng Zhu, Chaoyang Zhao, Honghui Dong, Zhengyou Zhang, Ming Tang, Jinqiao Wang  
  _2026-03-09_ · https://arxiv.org/abs/2603.07966v1  
  <details><summary>Abstract</summary>

  In situated collaboration, speakers often use intentionally underspecified deictic commands (e.g., ``pass me \textit{that}''), whose referent becomes identifiable only by aligning speech with a brief co-speech pointing \emph{stroke}. However, many embodied benchmarks admit language-only shortcuts, allowing MLLMs to perform well without learning the \emph{audio--visual alignment} required by deictic interaction. To bridge this gap, we introduce \textbf{Egocentric Co-Speech Grounding (EcoG)}, where grounding is executable only if an agent jointly predicts \textit{What}, \textit{Where}, and \textit{When}. To operationalize this, we present \textbf{EcoG-Bench}, an evaluation-only bilingual (EN/ZH) diagnostic benchmark of \textbf{811} egocentric clips with dense spatial annotations and millisecond-level stroke supervision. It is organized under a \textbf{Progressive Cognitive Evaluation} protocol. Benchmarking state-of-the-art MLLMs reveals a severe executability gap: while human subjects achieve near-ceiling performance on EcoG-Bench (\textbf{96.9\%} strict Eco-Accuracy), the best native video-audio setting remains low (Gemini-3-Pro: \textbf{17.0\%}). Moreover, in a diagnostic ablation, replacing the native video--audio interface with timestamped frame samples and externally verified ASR (with word-level timing) substantially improves the same model (\textbf{17.0\%}$\to$\textbf{42.9\%}). Overall, EcoG-Bench provides a strict, executable testbed for event-level speech--gesture binding, and suggests that multimodal interfaces may bottleneck the observability of temporal alignment cues, independently of model reasoning.

  </details>



- **A Hybrid Vision Transformer Approach for Mathematical Expression Recognition**  
  Anh Duy Le, Van Linh Pham, Vinh Loi Ly, Nam Quan Nguyen, Huu Thang Nguyen, Tuan Anh Tran  
  _2026-03-09_ · https://arxiv.org/abs/2603.07929v1  
  <details><summary>Abstract</summary>

  One of the crucial challenges taken in document analysis is mathematical expression recognition. Unlike text recognition which only focuses on one-dimensional structure images, mathematical expression recognition is a much more complicated problem because of its two-dimensional structure and different symbol size. In this paper, we propose using a Hybrid Vision Transformer (HVT) with 2D positional encoding as the encoder to extract the complex relationship between symbols from the image. A coverage attention decoder is used to better track attention's history to handle the under-parsing and over-parsing problems. We also showed the benefit of using the [CLS] token of ViT as the initial embedding of the decoder. Experiments performed on the IM2LATEX-100K dataset have shown the effectiveness of our method by achieving a BLEU score of 89.94 and outperforming current state-of-the-art methods.

  </details>


