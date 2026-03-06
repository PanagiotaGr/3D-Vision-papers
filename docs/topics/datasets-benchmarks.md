# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-03-06 07:06 UTC_

Total papers shown: **50**


---

- **FaceCam: Portrait Video Camera Control via Scale-Aware Conditioning**  
  Weijie Lyu, Ming-Hsuan Yang, Zhixin Shu  
  _2026-03-05_ · https://arxiv.org/abs/2603.05506v1  
  <details><summary>Abstract</summary>

  We introduce FaceCam, a system that generates video under customizable camera trajectories for monocular human portrait video input. Recent camera control approaches based on large video-generation models have shown promising progress but often exhibit geometric distortions and visual artifacts on portrait videos due to scale-ambiguous camera representations or 3D reconstruction errors. To overcome these limitations, we propose a face-tailored scale-aware representation for camera transformations that provides deterministic conditioning without relying on 3D priors. We train a video generation model on both multi-view studio captures and in-the-wild monocular videos, and introduce two camera-control data generation strategies: synthetic camera motion and multi-shot stitching, to exploit stationary training cameras while generalizing to dynamic, continuous camera trajectories at inference time. Experiments on Ava-256 dataset and diverse in-the-wild videos demonstrate that FaceCam achieves superior performance in camera controllability, visual quality, identity and motion preservation.

  </details>



- **Towards Multimodal Lifelong Understanding: A Dataset and Agentic Baseline**  
  Guo Chen, Lidong Lu, Yicheng Liu, Liangrui Dong, Lidong Zou, Jixin Lv, Zhenquan Li, Xinyi Mao, Baoqi Pei, Shihao Wang, et al.  
  _2026-03-05_ · https://arxiv.org/abs/2603.05484v1  
  <details><summary>Abstract</summary>

  While datasets for video understanding have scaled to hour-long durations, they typically consist of densely concatenated clips that differ from natural, unscripted daily life. To bridge this gap, we introduce MM-Lifelong, a dataset designed for Multimodal Lifelong Understanding. Comprising 181.1 hours of footage, it is structured across Day, Week, and Month scales to capture varying temporal densities. Extensive evaluations reveal two critical failure modes in current paradigms: end-to-end MLLMs suffer from a Working Memory Bottleneck due to context saturation, while representative agentic baselines experience Global Localization Collapse when navigating sparse, month-long timelines. To address this, we propose the Recursive Multimodal Agent (ReMA), which employs dynamic memory management to iteratively update a recursive belief state, significantly outperforming existing methods. Finally, we establish dataset splits designed to isolate temporal and domain biases, providing a rigorous foundation for future research in supervised learning and out-of-distribution generalization.

  </details>



- **Towards 3D Scene Understanding of Gas Plumes in LWIR Hyperspectral Images Using Neural Radiance Fields**  
  Scout Jarman, Zigfried Hampel-Arias, Adra Carr, Kevin R. Moon  
  _2026-03-05_ · https://arxiv.org/abs/2603.05473v1  
  <details><summary>Abstract</summary>

  Hyperspectral images (HSI) have many applications, ranging from environmental monitoring to national security, and can be used for material detection and identification. Longwave infrared (LWIR) HSI can be used for gas plume detection and analysis. Oftentimes, only a few images of a scene of interest are available and are analyzed individually. The ability to combine information from multiple images into a single, cohesive representation could enhance analysis by providing more context on the scene's geometry and spectral properties. Neural radiance fields (NeRFs) create a latent neural representation of volumetric scene properties that enable novel-view rendering and geometry reconstruction, offering a promising avenue for hyperspectral 3D scene reconstruction. We explore the possibility of using NeRFs to create 3D scene reconstructions from LWIR HSI and demonstrate that the model can be used for the basic downstream analysis task of gas plume detection. The physics-based DIRSIG software suite was used to generate a synthetic multi-view LWIR HSI dataset of a simple facility with a strong sulfur hexafluoride gas plume. Our method, built on the standard Mip-NeRF architecture, combines state-of-the-art methods for hyperspectral NeRFs and sparse-view NeRFs, along with a novel adaptive weighted MSE loss. Our final NeRF method requires around 50% fewer training images than the standard Mip-NeRF and achieves an average PSNR of 39.8 dB with as few as 30 training images. Gas plume detection applied to NeRF-rendered test images using the adaptive coherence estimator achieves an average AUC of 0.821 when compared with detection masks generated from ground-truth test images.

  </details>



- **EdgeDAM: Real-time Object Tracking for Mobile Devices**  
  Syed Muhammad Raza, Syed Murtaza Hussain Abidi, Khawar Islam, Muhammad Ibrahim, Ajmal Saeed Mian  
  _2026-03-05_ · https://arxiv.org/abs/2603.05463v1  
  <details><summary>Abstract</summary>

  Single-object tracking (SOT) on edge devices is a critical computer vision task, requiring accurate and continuous target localization across video frames under occlusion, distractor interference, and fast motion. However, recent state-of-the-art distractor-aware memory mechanisms are largely built on segmentation-based trackers and rely on mask prediction and attention-driven memory updates, which introduce substantial computational overhead and limit real-time deployment on resource-constrained hardware; meanwhile, lightweight trackers sustain high throughput but are prone to drift when visually similar distractors appear. To address these challenges, we propose EdgeDAM, a lightweight detection-guided tracking framework that reformulates distractor-aware memory for bounding-box tracking under strict edge constraints. EdgeDAM introduces two key strategies: (1) Dual-Buffer Distractor-Aware Memory (DAM), which integrates a Recent-Aware Memory to preserve temporally consistent target hypotheses and a Distractor-Resolving Memory to explicitly store hard negative candidates and penalize their re-selection during recovery; and (2) Confidence-Driven Switching with Held-Box Stabilization, where tracker reliability and temporal consistency criteria adaptively activate detection and memory-guided re-identification during occlusion, while a held-box mechanism temporarily freezes and expands the estimate to suppress distractor contamination. Extensive experiments on five benchmarks, including the distractor-focused DiDi dataset, demonstrate improved robustness under occlusion and fast motion while maintaining real-time performance on mobile devices, achieving 88.2% accuracy on DiDi and 25 FPS on an iPhone 15. Code will be released.

  </details>



- **NaiLIA: Multimodal Nail Design Retrieval Based on Dense Intent Descriptions and Palette Queries**  
  Kanon Amemiya, Daichi Yashima, Kei Katsumata, Takumi Komatsu, Ryosuke Korekata, Seitaro Otsuki, Komei Sugiura  
  _2026-03-05_ · https://arxiv.org/abs/2603.05446v1  
  <details><summary>Abstract</summary>

  We focus on the task of retrieving nail design images based on dense intent descriptions, which represent multi-layered user intent for nail designs. This is challenging because such descriptions specify unconstrained painted elements and pre-manufactured embellishments as well as visual characteristics, themes, and overall impressions. In addition to these descriptions, we assume that users provide palette queries by specifying zero or more colors via a color picker, enabling the expression of subtle and continuous color nuances. Existing vision-language foundation models often struggle to incorporate such descriptions and palettes. To address this, we propose NaiLIA, a multimodal retrieval method for nail design images, which comprehensively aligns with dense intent descriptions and palette queries during retrieval. Our approach introduces a relaxed loss based on confidence scores for unlabeled images that can align with the descriptions. To evaluate NaiLIA, we constructed a benchmark consisting of 10,625 images collected from people with diverse cultural backgrounds. The images were annotated with long and dense intent descriptions given by over 200 annotators. Experimental results demonstrate that NaiLIA outperforms standard methods.

  </details>



- **SAIL: Similarity-Aware Guidance and Inter-Caption Augmentation-based Learning for Weakly-Supervised Dense Video Captioning**  
  Ye-Chan Kim, SeungJu Cha, Si-Woo Kim, Minju Jeon, Hyungee Kim, Dong-Jin Kim  
  _2026-03-05_ · https://arxiv.org/abs/2603.05437v1  
  <details><summary>Abstract</summary>

  Weakly-Supervised Dense Video Captioning aims to localize and describe events in videos trained only on caption annotations, without temporal boundaries. Prior work introduced an implicit supervision paradigm based on Gaussian masking and complementary captioning. However, existing method focuses merely on generating non-overlapping masks without considering their semantic relationship to corresponding events, resulting in simplistic, uniformly distributed masks that fail to capture semantically meaningful regions. Moreover, relying solely on ground-truth captions leads to sub-optimal performance due to the inherent sparsity of existing datasets. In this work, we propose SAIL, which constructs semantically-aware masks through cross-modal alignment. Our similarity aware training objective guides masks to emphasize video regions with high similarity to their corresponding event captions. Furthermore, to guide more accurate mask generation under sparse annotation settings, we introduce an LLM-based augmentation strategy that generates synthetic captions to provide additional alignment signals. These synthetic captions are incorporated through an inter-mask mechanism, providing auxiliary guidance for precise temporal localization without degrading the main objective. Experiments on ActivityNet Captions and YouCook2 demonstrate state-of-the-art performance on both captioning and localization metrics.

  </details>



- **Video-based Locomotion Analysis for Fish Health Monitoring**  
  Timon Palm, Clemens Seibold, Anna Hilsmann, Peter Eisert  
  _2026-03-05_ · https://arxiv.org/abs/2603.05407v1  
  <details><summary>Abstract</summary>

  Monitoring the health conditions of fish is essential, as it enables the early detection of disease, safeguards animal welfare, and contributes to sustainable aquaculture practices. Physiological and pathological conditions of cultivated fish can be inferred by analyzing locomotion activities. In this paper, we present a system that estimates the locomotion activities from videos using multi object tracking. The core of our approach is a YOLOv11 detector embedded in a tracking-by-detection framework. We investigate various configurations of the YOLOv11-architecture as well as extensions that incorporate multiple frames to improve detection accuracy. Our system is evaluated on a manually annotated dataset of Sulawesi ricefish recorded in a home-aquarium-like setup, demonstrating its ability to reliably measure swimming direction and speed for fish health monitoring. The dataset will be made publicly available upon publication.

  </details>



- **Loop Closure via Maximal Cliques in 3D LiDAR-Based SLAM**  
  Javier Laserna, Saurabh Gupta, Oscar Martinez Mozos, Cyrill Stachniss, Pablo San Segundo  
  _2026-03-05_ · https://arxiv.org/abs/2603.05397v1  
  <details><summary>Abstract</summary>

  Reliable loop closure detection remains a critical challenge in 3D LiDAR-based SLAM, especially under sensor noise, environmental ambiguity, and viewpoint variation conditions. RANSAC is often used in the context of loop closures for geometric model fitting in the presence of outliers. However, this approach may fail, leading to map inconsistency. We introduce a novel deterministic algorithm, CliReg, for loop closure validation that replaces RANSAC verification with a maximal clique search over a compatibility graph of feature correspondences. This formulation avoids random sampling and increases robustness in the presence of noise and outliers. We integrated our approach into a real- time pipeline employing binary 3D descriptors and a Hamming distance embedding binary search tree-based matching. We evaluated it on multiple real-world datasets featuring diverse LiDAR sensors. The results demonstrate that our proposed technique consistently achieves a lower pose error and more reliable loop closures than RANSAC, especially in sparse or ambiguous conditions. Additional experiments on 2D projection-based maps confirm its generality across spatial domains, making our approach a robust and efficient alternative for loop closure detection.

  </details>



- **ORMOT: A Dataset and Framework for Omnidirectional Referring Multi-Object Tracking**  
  Sijia Chen, Zihan Zhou, Yanqiu Yu, En Yu, Wenbing Tao  
  _2026-03-05_ · https://arxiv.org/abs/2603.05384v1  
  <details><summary>Abstract</summary>

  Multi-Object Tracking (MOT) is a fundamental task in computer vision, aiming to track targets across video frames. Existing MOT methods perform well in general visual scenes, but face significant challenges and limitations when extended to visual-language settings. To bridge this gap, the task of Referring Multi-Object Tracking (RMOT) has recently been proposed, which aims to track objects that correspond to language descriptions. However, current RMOT methods are primarily developed on datasets captured by conventional cameras, which suffer from limited field of view. This constraint often causes targets to move out of the frame, leading to fragmented tracking and loss of contextual information. In this work, we propose a novel task, called Omnidirectional Referring Multi-Object Tracking (ORMOT), which extends RMOT to omnidirectional imagery, aiming to overcome the field-of-view (FoV) limitation of conventional datasets and improve the model's ability to understand long-horizon language descriptions. To advance the ORMOT task, we construct ORSet, an Omnidirectional Referring Multi-Object Tracking dataset, which contains 27 diverse omnidirectional scenes, 848 language descriptions, and 3,401 annotated objects, providing rich visual, temporal, and language information. Furthermore, we propose ORTrack, a Large Vision-Language Model (LVLM)-driven framework tailored for Omnidirectional Referring Multi-Object Tracking. Extensive experiments on the ORSet dataset demonstrate the effectiveness of our ORTrack framework. The dataset and code will be open-sourced at https://github.com/chen-si-jia/ORMOT.

  </details>



- **Dark3R: Learning Structure from Motion in the Dark**  
  Andrew Y Guo, Anagh Malik, SaiKiran Tedla, Yutong Dai, Yiqian Qin, Zach Salehe, Benjamin Attal, Sotiris Nousias, Kyros Kutulakos, David B. Lindell  
  _2026-03-05_ · https://arxiv.org/abs/2603.05330v1  
  <details><summary>Abstract</summary>

  We introduce Dark3R, a framework for structure from motion in the dark that operates directly on raw images with signal-to-noise ratios (SNRs) below $-4$ dB -- a regime where conventional feature- and learning-based methods break down. Our key insight is to adapt large-scale 3D foundation models to extreme low-light conditions through a teacher--student distillation process, enabling robust feature matching and camera pose estimation in low light. Dark3R requires no 3D supervision; it is trained solely on noisy--clean raw image pairs, which can be either captured directly or synthesized using a simple Poisson--Gaussian noise model applied to well-exposed raw images. To train and evaluate our approach, we introduce a new, exposure-bracketed dataset that includes $\sim$42,000 multi-view raw images with ground-truth 3D annotations, and we demonstrate that Dark3R achieves state-of-the-art structure from motion in the low-SNR regime. Further, we demonstrate state-of-the-art novel view synthesis in the dark using Dark3R's predicted poses and a coarse-to-fine radiance field optimization procedure.

  </details>



- **UltraDexGrasp: Learning Universal Dexterous Grasping for Bimanual Robots with Synthetic Data**  
  Sizhe Yang, Yiman Xie, Zhixuan Liang, Yang Tian, Jia Zeng, Dahua Lin, Jiangmiao Pang  
  _2026-03-05_ · https://arxiv.org/abs/2603.05312v1  
  <details><summary>Abstract</summary>

  Grasping is a fundamental capability for robots to interact with the physical world. Humans, equipped with two hands, autonomously select appropriate grasp strategies based on the shape, size, and weight of objects, enabling robust grasping and subsequent manipulation. In contrast, current robotic grasping remains limited, particularly in multi-strategy settings. Although substantial efforts have targeted parallel-gripper and single-hand grasping, dexterous grasping for bimanual robots remains underexplored, with data being a primary bottleneck. Achieving physically plausible and geometrically conforming grasps that can withstand external wrenches poses significant challenges. To address these issues, we introduce UltraDexGrasp, a framework for universal dexterous grasping with bimanual robots. The proposed data-generation pipeline integrates optimization-based grasp synthesis with planning-based demonstration generation, yielding high-quality and diverse trajectories across multiple grasp strategies. With this framework, we curate UltraDexGrasp-20M, a large-scale, multi-strategy grasp dataset comprising 20 million frames across 1,000 objects. Based on UltraDexGrasp-20M, we further develop a simple yet effective grasp policy that takes point clouds as input, aggregates scene features via unidirectional attention, and predicts control commands. Trained exclusively on synthetic data, the policy achieves robust zero-shot sim-to-real transfer and consistently succeeds on novel objects with varied shapes, sizes, and weights, attaining an average success rate of 81.2% in real-world universal dexterous grasping. To facilitate future research on grasping with bimanual robots, we open-source the data generation pipeline at https://github.com/InternRobotics/UltraDexGrasp.

  </details>



- **Fusion4CA: Boosting 3D Object Detection via Comprehensive Image Exploitation**  
  Kang Luo, Xin Chen, Yangyi Xiao, Hesheng Wang  
  _2026-03-05_ · https://arxiv.org/abs/2603.05305v1  
  <details><summary>Abstract</summary>

  Nowadays, an increasing number of works fuse LiDAR and RGB data in the bird's-eye view (BEV) space for 3D object detection in autonomous driving systems. However, existing methods suffer from over-reliance on the LiDAR branch, with insufficient exploration of RGB information. To tackle this issue, we propose Fusion4CA, which is built upon the classic BEVFusion framework and dedicated to fully exploiting visual input with plug-and-play components. Specifically, a contrastive alignment module is designed to calibrate image features with 3D geometry, and a camera auxiliary branch is introduced to mine RGB information sufficiently during training. For further performance enhancement, we leverage an off-the-shelf cognitive adapter to make the most of pretrained image weights, and integrate a standard coordinate attention module into the fusion stage as a supplementary boost. Experiments on the nuScenes dataset demonstrate that our method achieves 69.7% mAP with only 6 training epochs and a mere 3.48% increase in inference parameters, yielding a 1.2% improvement over the baseline which is fully trained for 20 epochs. Extensive experiments in a simulated lunar environment further validate the effectiveness and generalization of our method. Our code will be released through Fusion4CA.

  </details>



- **Latent Policy Steering through One-Step Flow Policies**  
  Hokyun Im, Andrey Kolobov, Jianlong Fu, Youngwoon Lee  
  _2026-03-05_ · https://arxiv.org/abs/2603.05296v1  
  <details><summary>Abstract</summary>

  Offline reinforcement learning (RL) allows robots to learn from offline datasets without risky exploration. Yet, offline RL's performance often hinges on a brittle trade-off between (1) return maximization, which can push policies outside the dataset support, and (2) behavioral constraints, which typically require sensitive hyperparameter tuning. Latent steering offers a structural way to stay within the dataset support during RL, but existing offline adaptations commonly approximate action values using latent-space critics learned via indirect distillation, which can lose information and hinder convergence. We propose Latent Policy Steering (LPS), which enables high-fidelity latent policy improvement by backpropagating original-action-space Q-gradients through a differentiable one-step MeanFlow policy to update a latent-action-space actor. By eliminating proxy latent critics, LPS allows an original-action-space critic to guide end-to-end latent-space optimization, while the one-step MeanFlow policy serves as a behavior-constrained generative prior. This decoupling yields a robust method that works out-of-the-box with minimal tuning. Across OGBench and real-world robotic tasks, LPS achieves state-of-the-art performance and consistently outperforms behavioral cloning and strong latent steering baselines.

  </details>



- **WebChain: A Large-Scale Human-Annotated Dataset of Real-World Web Interaction Traces**  
  Sicheng Fan, Rui Wan, Yifei Leng, Gaoning Liang, Li Ling, Yanyi Shang, Dehan Kong  
  _2026-03-05_ · https://arxiv.org/abs/2603.05295v1  
  <details><summary>Abstract</summary>

  We introduce WebChain, the largest open-source dataset of human-annotated trajectories on real-world websites, designed to accelerate reproducible research in web agents. It contains 31,725 trajectories and 318k steps, featuring a core Triple Alignment of visual, structural, and action data to provide rich, multi-modal supervision. The data is collected via a scalable pipeline that ensures coverage of complex, high-value tasks often missed by synthetic methods. Leveraging this dataset, we propose a Dual Mid-Training recipe that decouples spatial grounding from planning, achieving state-of-the-art performance on our proposed WebChainBench and other public GUI benchmarks. Our work provides the data and insights necessary to build and rigorously evaluate the next generation of scalable web agents.

  </details>



- **Iterative On-Policy Refinement of Hierarchical Diffusion Policies for Language-Conditioned Manipulation**  
  Clemence Grislain, Olivier Sigaud, Mohamed Chetouani  
  _2026-03-05_ · https://arxiv.org/abs/2603.05291v1  
  <details><summary>Abstract</summary>

  Hierarchical policies for language-conditioned manipulation decompose tasks into subgoals, where a high-level planner guides a low-level controller. However, these hierarchical agents often fail because the planner generates subgoals without considering the actual limitations of the controller. Existing solutions attempt to bridge this gap via intermediate modules or shared representations, but they remain limited by their reliance on fixed offline datasets. We propose HD-ExpIt, a framework for iterative fine-tuning of hierarchical diffusion policies via environment feedback. HD-ExpIt organizes training into a self-reinforcing cycle: it utilizes diffusion-based planning to autonomously discover successful behaviors, which are then distilled back into the hierarchical policy. This loop enables both components to improve while implicitly grounding the planner in the controller's actual capabilities without requiring explicit proxy models. Empirically, HD-ExpIt significantly improves hierarchical policies trained solely on offline data, achieving state-of-the-art performance on the long-horizon CALVIN benchmark among methods trained from scratch.

  </details>



- **Curve-Induced Dynamical Systems on Riemannian Manifolds and Lie Groups**  
  Saray Bakker, Martin Schonger, Tobias Löw, Javier Alonso-Mora, Sylvain Calinon  
  _2026-03-05_ · https://arxiv.org/abs/2603.05268v1  
  <details><summary>Abstract</summary>

  Deploying robots in household environments requires safe, adaptable, and interpretable behaviors that respect the geometric structure of tasks. Often represented on Lie groups and Riemannian manifolds, this includes poses on SE(3) or symmetric positive definite matrices encoding stiffness or damping matrices. In this context, dynamical system-based approaches offer a natural framework for generating such behavior, providing stability and convergence while remaining responsive to changes in the environment. We introduce Curve-induced Dynamical systems on Smooth Manifolds (CDSM), a real-time framework for constructing dynamical systems directly on Riemannian manifolds and Lie groups. The proposed approach constructs a nominal curve on the manifold, and generates a dynamical system which combines a tangential component that drives motion along the curve and a normal component that attracts the state toward the curve. We provide a stability analysis of the resulting dynamical system and validate the method quantitatively. On an S2 benchmark, CDSM demonstrates improved trajectory accuracy, reduced path deviation, and faster generation and query times compared to state-of-the-art methods. Finally, we demonstrate the practical applicability of the framework on both a robotic manipulator, where poses on SE(3) and damping matrices on SPD(n) are adapted online, and a mobile manipulator.

  </details>



- **Digital Twin Driven Textile Classification and Foreign Object Recognition in Automated Sorting Systems**  
  Serkan Ergun, Tobias Mitterer, Hubert Zangl  
  _2026-03-05_ · https://arxiv.org/abs/2603.05230v1  
  <details><summary>Abstract</summary>

  The increasing demand for sustainable textile recycling requires robust automation solutions capable of handling deformable garments and detecting foreign objects in cluttered environments. This work presents a digital twin driven robotic sorting system that integrates grasp prediction, multi modal perception, and semantic reasoning for real world textile classification. A dual arm robotic cell equipped with RGBD sensing, capacitive tactile feedback, and collision-aware motion planning autonomously separates garments from an unsorted basket, transfers them to an inspection zone, and classifies them using state of the art Visual Language Models (VLMs). We benchmark nine VLM s from five model families on a dataset of 223 inspection scenarios comprising shirts, socks, trousers, underwear, foreign objects (including garments outside of the aforementioned classes), and empty scenes. The evaluation assesses per class accuracy, hallucination behavior, and computational performance under practical hardware constraints. Results show that the Qwen model family achieves the highest overall accuracy (up to 87.9 %), with strong foreign object detection performance, while lighter models such as Gemma3 offer competitive speed accuracy trade offs for edge deployment. A digital twin combined with MoveIt enables collision aware path planning and integrates segmented 3D point clouds of inspected garments into the virtual environment for improved manipulation reliability. The presented system demonstrates the feasibility of combining semantic VLM reasoning with conventional grasp detection and digital twin technology for scalable, autonomous textile sorting in realistic industrial settings.

  </details>



- **SPyCer: Semi-Supervised Physics-Guided Contextual Attention for Near-Surface Air Temperature Estimation from Satellite Imagery**  
  Sofiane Bouaziz, Adel Hafiane, Raphael Canals, Rachid Nedjai  
  _2026-03-05_ · https://arxiv.org/abs/2603.05219v1  
  <details><summary>Abstract</summary>

  Modern Earth observation relies on satellites to capture detailed surface properties. Yet, many phenomena that affect humans and ecosystems unfold in the atmosphere close to the surface. Near-ground sensors provide accurate measurements of certain environmental characteristics, such as near-surface air temperature (NSAT). However, they remain sparse and unevenly distributed, limiting their ability to provide continuous spatial measurements. To bridge this gap, we introduce SPyCer, a semi-supervised physics-guided network that can leverage pixel information and physical modeling to guide the learning process through meaningful physical properties. It is designed for continuous estimation of NSAT by proxy using satellite imagery. SPyCer frames NSAT prediction as a pixel-wise vision problem, where each near-ground sensor is projected onto satellite image coordinates and positioned at the center of a local image patch. The corresponding sensor pixel is supervised using both observed NSAT and physics-based constraints, while surrounding pixels contribute through physics-guided regularization derived from the surface energy balance and advection-diffusion-reaction partial differential equations. To capture the physical influence of neighboring pixels, SPyCer employs a multi-head attention guided by land cover characteristics and modulated with Gaussian distance weighting. Experiments on real-world datasets demonstrate that SPyCer produces spatially coherent and physically consistent NSAT estimates, outperforming existing baselines in terms of accuracy, generalization, and alignment with underlying physical processes.

  </details>



- **Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation**  
  Yingxue Su, Yiheng Zhong, Keying Zhu, Zimu Zhang, Zhuoru Zhang, Yifang Wang, Yuxin Zhang, Jingxin Liu  
  _2026-03-05_ · https://arxiv.org/abs/2603.05202v1  
  <details><summary>Abstract</summary>

  Medical image segmentation is critical for computer-aided diagnosis. However, dense pixel-level annotation is time-consuming and expensive, and medical datasets often exhibit severe class imbalance. Such imbalance causes minority structures to be overwhelmed by dominant classes in feature representations, hindering the learning of discriminative features and making reliable segmentation particularly challenging. To address this, we propose the Semantic Class Distribution Learning (SCDL) framework, a plug-and-play module that mitigates supervision and representation biases by learning structured class-conditional feature distributions. SCDL integrates Class Distribution Bidirectional Alignment (CDBA) to align embeddings with learnable class proxies and leverages Semantic Anchor Constraints (SAC) to guide proxies using labeled data. Experiments on the Synapse and AMOS datasets demonstrate that SCDL significantly improves segmentation performance across both overall and class-level metrics, with particularly strong gains on minority classes, achieving state-of-the-art results. Our code is released at https://github.com/Zyh55555/SCDL.

  </details>



- **SSR-GS: Separating Specular Reflection in Gaussian Splatting for Glossy Surface Reconstruction**  
  Ningjing Fan, Yiqun Wang  
  _2026-03-05_ · https://arxiv.org/abs/2603.05152v1  
  <details><summary>Abstract</summary>

  In recent years, 3D Gaussian splatting (3DGS) has achieved remarkable progress in novel view synthesis. However, accurately reconstructing glossy surfaces under complex illumination remains challenging, particularly in scenes with strong specular reflections and multi-surface interreflections. To address this issue, we propose SSR-GS, a specular reflection modeling framework for glossy surface reconstruction. Specifically, we introduce a prefiltered Mip-Cubemap to model direct specular reflections efficiently, and propose an IndiASG module to capture indirect specular reflections. Furthermore, we design Visual Geometry Priors (VGP) that couple a reflection-aware visual prior via a reflection score (RS) to downweight the photometric loss contribution of reflection-dominated regions, with geometry priors derived from VGGT, including progressively decayed depth supervision and transformed normal constraints. Extensive experiments on both synthetic and real-world datasets demonstrate that SSR-GS achieves state-of-the-art performance in glossy surface reconstruction.

  </details>



- **SRasP: Self-Reorientation Adversarial Style Perturbation for Cross-Domain Few-Shot Learning**  
  Wenqian Li, Pengfei Fang, Hui Xue  
  _2026-03-05_ · https://arxiv.org/abs/2603.05135v1  
  <details><summary>Abstract</summary>

  Cross-Domain Few-Shot Learning (CD-FSL) aims to transfer knowledge from a seen source domain to unseen target domains, serving as a key benchmark for evaluating the robustness and transferability of models. Existing style-based perturbation methods mitigate domain shift but often suffer from gradient instability and convergence to sharp minima.To address these limitations, we propose a novel crop-global style perturbation network, termed Self-Reorientation Adversarial \underline{S}tyle \underline{P}erturbation (SRasP). Specifically, SRasP leverages global semantic guidance to identify incoherent crops, followed by reorienting and aggregating the style gradients of these crops with the global style gradients within one image. Furthermore, we propose a novel multi-objective optimization function to maximize visual discrepancy while enforcing semantic consistency among global, crop, and adversarial features. Applying the stabilized perturbations during training encourages convergence toward flatter and more transferable solutions, improving generalization to unseen domains. Extensive experiments are conducted on multiple CD-FSL benchmarks, demonstrating consistent improvements over state-of-the-art methods.

  </details>



- **SeedPolicy: Horizon Scaling via Self-Evolving Diffusion Policy for Robot Manipulation**  
  Youqiang Gui, Yuxuan Zhou, Shen Cheng, Xinyang Yuan, Haoqiang Fan, Peng Cheng, Shuaicheng Liu  
  _2026-03-05_ · https://arxiv.org/abs/2603.05117v1  
  <details><summary>Abstract</summary>

  Imitation Learning (IL) enables robots to acquire manipulation skills from expert demonstrations. Diffusion Policy (DP) models multi-modal expert behaviors but suffers performance degradation as observation horizons increase, limiting long-horizon manipulation. We propose Self-Evolving Gated Attention (SEGA), a temporal module that maintains a time-evolving latent state via gated attention, enabling efficient recurrent updates that compress long-horizon observations into a fixed-size representation while filtering irrelevant temporal information. Integrating SEGA into DP yields Self-Evolving Diffusion Policy (SeedPolicy), which resolves the temporal modeling bottleneck and enables scalable horizon extension with moderate overhead. On the RoboTwin 2.0 benchmark with 50 manipulation tasks, SeedPolicy outperforms DP and other IL baselines. Averaged across both CNN and Transformer backbones, SeedPolicy achieves 36.8% relative improvement in clean settings and 169% relative improvement in randomized challenging settings over the DP. Compared to vision-language-action models such as RDT with 1.2B parameters, SeedPolicy achieves competitive performance with one to two orders of magnitude fewer parameters, demonstrating strong efficiency and scalability. These results establish SeedPolicy as a state-of-the-art imitation learning method for long-horizon robotic manipulation. Code is available at: https://github.com/Youqiang-Gui/SeedPolicy.

  </details>



- **UniPAR: A Unified Framework for Pedestrian Attribute Recognition**  
  Minghe Xu, Rouying Wu, Jiarui Xu, Minhao Sun, Zikang Yan, Xiao Wang, ChiaWei Chu, Yu Li  
  _2026-03-05_ · https://arxiv.org/abs/2603.05114v1  
  <details><summary>Abstract</summary>

  Pedestrian Attribute Recognition is a foundational computer vision task that provides essential support for downstream applications, including person retrieval in video surveillance and intelligent retail analytics. However, existing research is frequently constrained by the ``one-model-per-dataset" paradigm and struggles to handle significant discrepancies across domains in terms of modalities, attribute definitions, and environmental scenarios. To address these challenges, we propose UniPAR, a unified Transformer-based framework for PAR. By incorporating a unified data scheduling strategy and a dynamic classification head, UniPAR enables a single model to simultaneously process diverse datasets from heterogeneous modalities, including RGB images, video sequences, and event streams. We also introduce an innovative phased fusion encoder that explicitly aligns visual features with textual attribute queries through a late deep fusion strategy. Experimental results on the widely used benchmark datasets, including MSP60K, DukeMTMC, and EventPAR, demonstrate that UniPAR achieves performance comparable to specialized SOTA methods. Furthermore, multi-dataset joint training significantly enhances the model's cross-domain generalization and recognition robustness in extreme environments characterized by low light and motion blur. The source code of this paper will be released on https://github.com/Event-AHU/OpenPAR

  </details>



- **AIM-SLAM: Dense Monocular SLAM via Adaptive and Informative Multi-View Keyframe Prioritization with Foundation Model**  
  Jinwoo Jeon, Dong-Uk Seo, Eungchang Mason Lee, Hyun Myung  
  _2026-03-05_ · https://arxiv.org/abs/2603.05097v1  
  <details><summary>Abstract</summary>

  Recent advances in geometric foundation models have emerged as a promising alternative for addressing the challenge of dense reconstruction in monocular visual simultaneous localization and mapping (SLAM). Although geometric foundation models enable SLAM to leverage variable input views, the previous methods remain confined to two-view pairs or fixed-length inputs without sufficient deliberation of geometric context for view selection. To tackle this problem, we propose AIM-SLAM, a dense monocular SLAM framework that exploits an adaptive and informative multi-view keyframe prioritization with dense pointmap predictions from visual geometry grounded transformer (VGGT). Specifically, we introduce the selective information- and geometric-aware multi-view adaptation (SIGMA) module, which employs voxel overlap and information gain to retrieve a candidate set of keyframes and adaptively determine its size. Furthermore, we formulate a joint multi-view Sim(3) optimization that enforces consistent alignment across selected views, substantially improving pose estimation accuracy. The effectiveness of AIM-SLAM is demonstrated on real-world datasets, where it achieves state-of-the-art performance in both pose estimation and dense reconstruction. Our system supports ROS integration, with code is available at https://aimslam.github.io/.

  </details>



- **GEM-TFL: Bridging Weak and Full Supervision for Forgery Localization through EM-Guided Decomposition and Temporal Refinement**  
  Xiaodong Zhu, Yuanming Zheng, Suting Wang, Junqi Yang, Yuhong Yang, Weiping Tu, Zhongyuan Wang  
  _2026-03-05_ · https://arxiv.org/abs/2603.05095v1  
  <details><summary>Abstract</summary>

  Temporal Forgery Localization (TFL) aims to precisely identify manipulated segments within videos or audio streams, providing interpretable evidence for multimedia forensics and security. While most existing TFL methods rely on dense frame-level labels in a fully supervised manner, Weakly Supervised TFL (WS-TFL) reduces labeling cost by learning only from binary video-level labels. However, current WS-TFL approaches suffer from mismatched training and inference objectives, limited supervision from binary labels, gradient blockage caused by non-differentiable top-k aggregation, and the absence of explicit modeling of inter-proposal relationships. To address these issues, we propose GEM-TFL (Graph-based EM-powered Temporal Forgery Localization), a two-phase classification-regression framework that effectively bridges the supervision gap between training and inference. Built upon this foundation, (1) we enhance weak supervision by reformulating binary labels into multi-dimensional latent attributes through an EM-based optimization process; (2) we introduce a training-free temporal consistency refinement that realigns frame-level predictions for smoother temporal dynamics; and (3) we design a graph-based proposal refinement module that models temporal-semantic relationships among proposals for globally consistent confidence estimation. Extensive experiments on benchmark datasets demonstrate that GEM-TFL achieves more accurate and robust temporal forgery localization, substantially narrowing the gap with fully supervised methods.

  </details>



- **UniM: A Unified Any-to-Any Interleaved Multimodal Benchmark**  
  Yanlin Li, Minghui Guo, Kaiwen Zhang, Shize Zhang, Yiran Zhao, Haodong Li, Congyue Zhou, Weijie Zheng, Yushen Yan, Shengqiong Wu, et al.  
  _2026-03-05_ · https://arxiv.org/abs/2603.05075v1  
  <details><summary>Abstract</summary>

  In real-world multimodal applications, systems usually need to comprehend arbitrarily combined and interleaved multimodal inputs from users, while also generating outputs in any interleaved multimedia form. This capability defines the goal of any-to-any interleaved multimodal learning under a unified paradigm of understanding and generation, posing new challenges and opportunities for advancing Multimodal Large Language Models (MLLMs). To foster and benchmark this capability, this paper introduces the UniM benchmark, the first Unified Any-to-Any Interleaved Multimodal dataset. UniM contains 31K high-quality instances across 30 domains and 7 representative modalities: text, image, audio, video, document, code, and 3D, each requiring multiple intertwined reasoning and generation capabilities. We further introduce the UniM Evaluation Suite, which assesses models along three dimensions: Semantic Correctness & Generation Quality, Response Structure Integrity, and Interleaved Coherence. In addition, we propose UniMA, an agentic baseline model equipped with traceable reasoning for structured interleaved generation. Comprehensive experiments demonstrate the difficulty of UniM and highlight key challenges and directions for advancing unified any-to-any multimodal intelligence. The project page is https://any2any-mllm.github.io/unim.

  </details>



- **VinePT-Map: Pole-Trunk Semantic Mapping for Resilient Autonomous Robotics in Vineyards**  
  Giorgio Audrito, Mauro Martini, Alessandro Navone, Giorgia Galluzzo, Marcello Chiaberge  
  _2026-03-05_ · https://arxiv.org/abs/2603.05070v1  
  <details><summary>Abstract</summary>

  Reliable long-term deployment of autonomous robots in agricultural environments remains challenging due to perceptual aliasing, seasonal variability, and the dynamic nature of crop canopies. Vineyards, characterized by repetitive row structures and significant visual changes across phenological stages, represent a pivotal field challenge, limiting the robustness of conventional feature-based localization and mapping approaches. This paper introduces VinePT-Map, a semantic mapping framework that leverages vine trunks and support poles as persistent structural landmarks to enable season-agnostic and resilient robot localization. The proposed method formulates the mapping problem as a factor graph, integrating GPS, IMU, and RGB-D observations through robust geometrical constraints that exploit vineyard structure. An efficient perception pipeline based on instance segmentation and tracking, combined with a clustering filter for outlier rejection and pose refinement, enables accurate landmark detection using low-cost sensors and onboard computation. To validate the pipeline, we present a multi-season dataset for trunk and pole segmentation and tracking. Extensive field experiments conducted across diverse seasons demonstrate the robustness and accuracy of the proposed approach, highlighting its suitability for long-term autonomous operation in agricultural environments.

  </details>



- **A 360-degree Multi-camera System for Blue Emergency Light Detection Using Color Attention RT-DETR and the ABLDataset**  
  Francisco Vacalebri-Lloret, Lucas Banchero, Jose J. Lopez, Jose M. Mossi  
  _2026-03-05_ · https://arxiv.org/abs/2603.05058v1  
  <details><summary>Abstract</summary>

  This study presents an advanced system for detecting blue lights on emergency vehicles, developed using ABLDataset, a curated dataset that includes images of European emergency vehicles under various climatic and geographic conditions. The system employs a configuration of four fisheye cameras, each with a 180-degree horizontal field of view, mounted on the sides of the vehicle. A calibration process enables the azimuthal localization of the detections. Additionally, a comparative analysis of major deep neural network algorithms was conducted, including YOLO (v5, v8, and v10), RetinaNet, Faster R-CNN, and RT-DETR. RT-DETR was selected as the base model and enhanced through the incorporation of a color attention block, achieving an accuracy of 94.7 percent and a recall of 94.1 percent on the test set, with field test detections reaching up to 70 meters. Furthermore, the system estimates the approach angle of the emergency vehicle relative to the center of the car using geometric transformations. Designed for integration into a multimodal system that combines visual and acoustic data, this system has demonstrated high efficiency, offering a promising approach to enhancing Advanced Driver Assistance Systems (ADAS) and road safety.

  </details>



- **Generalizable Multiscale Segmentation of Heterogeneous Map Collections**  
  Remi Petitpierre  
  _2026-03-05_ · https://arxiv.org/abs/2603.05037v1  
  <details><summary>Abstract</summary>

  Historical map collections are highly diverse in style, scale, and geographic focus, often consisting of many single-sheet documents. Yet most work in map recognition focuses on specialist models tailored to homogeneous map series. In contrast, this article aims to develop generalizable semantic segmentation models and ontology. First, we introduce Semap, a new open benchmark dataset comprising 1,439 manually annotated patches designed to reflect the variety of historical map documents. Second, we present a segmentation framework that combines procedural data synthesis with multiscale integration to improve robustness and transferability. This framework achieves state-of-the-art performance on both the HCMSSD and Semap datasets, showing that a diversity-driven approach to map recognition is not only viable but also beneficial. The results indicate that segmentation performance remains largely stable across map collections, scales, geographic regions, and publication contexts. By proposing benchmark datasets and methods for the generic segmentation of historical maps, this work opens the way to integrating the long tail of cartographic archives to historical geographic studies.

  </details>



- **How far have we gone in Generative Image Restoration? A study on its capability, limitations and evaluation practices**  
  Xiang Yin, Jinfan Hu, Zhiyuan You, Kainan Yan, Yu Tang, Chao Dong, Jinjin Gu  
  _2026-03-05_ · https://arxiv.org/abs/2603.05010v1  
  <details><summary>Abstract</summary>

  Generative Image Restoration (GIR) has achieved impressive perceptual realism, but how far have its practical capabilities truly advanced compared with previous methods? To answer this, we present a large-scale study grounded in a new multi-dimensional evaluation pipeline that assesses models on detail, sharpness, semantic correctness, and overall quality. Our analysis covers diverse architectures, including diffusion-based, GAN-based, PSNR-oriented, and general-purpose generation models, revealing critical performance disparities. Furthermore, our analysis uncovers a key evolution in failure modes that signifies a paradigm shift for the perception-oriented low-level vision field. The central challenge is evolving from the previous problem of detail scarcity (under-generation) to the new frontier of detail quality and semantic control (preventing over-generation). We also leverage our benchmark to train a new IQA model that better aligns with human perceptual judgments. Ultimately, this work provides a systematic study of modern generative image restoration models, offering crucial insights that redefine our understanding of their true state and chart a course for future development.

  </details>



- **TAPFormer: Robust Arbitrary Point Tracking via Transient Asynchronous Fusion of Frames and Events**  
  Jiaxiong Liu, Zhen Tan, Jinpu Zhang, Yi Zhou, Hui Shen, Xieyuanli Chen, Dewen Hu  
  _2026-03-05_ · https://arxiv.org/abs/2603.04989v1  
  <details><summary>Abstract</summary>

  Tracking any point (TAP) is a fundamental yet challenging task in computer vision, requiring high precision and long-term motion reasoning. Recent attempts to combine RGB frames and event streams have shown promise, yet they typically rely on synchronous or non-adaptive fusion, leading to temporal misalignment and severe degradation when one modality fails. We introduce TAPFormer, a transformer-based framework that performs asynchronous temporal-consistent fusion of frames and events for robust and high-frequency arbitrary point tracking. Our key innovation is a Transient Asynchronous Fusion (TAF) mechanism, which explicitly models the temporal evolution between discrete frames through continuous event updates, bridging the gap between low-rate frames and high-rate events. In addition, a Cross-modal Locally Weighted Fusion (CLWF) module adaptively adjusts spatial attention according to modality reliability, yielding stable and discriminative features even under blur or low light. To evaluate our approach under realistic conditions, we construct a novel real-world frame-event TAP dataset under diverse illumination and motion conditions. Our method outperforms existing point trackers, achieving a 28.2% improvement in average pixel error within threshold. Moreover, on standard point tracking benchmarks, our tracker consistently achieves the best performance. Project website: tapformer.github.io

  </details>



- **BiEvLight: Bi-level Learning of Task-Aware Event Refinement for Low-Light Image Enhancement**  
  Zishu Yao, Xiang-Xiang Su, Shengning Zhou, Guang-Yong Chen, Guodong Fan, Xing Chen  
  _2026-03-05_ · https://arxiv.org/abs/2603.04975v1  
  <details><summary>Abstract</summary>

  Event cameras, with their high dynamic range, show great promise for Low-light Image Enhancement (LLIE). Existing works primarily focus on designing effective modal fusion strategies. However, a key challenge is the dual degradation from intrinsic background activity (BA) noise in events and low signal-to-noise ratio (SNR) in images, which causes severe noise coupling during modal fusion, creating a critical performance bottleneck. We therefore posit that precise event denoising is the prerequisite to unlocking the full potential of event-based fusion. To this end, we propose BiEvLight, a hierarchical and task-aware framework that collaboratively optimizes enhancement and denoising by exploiting their intrinsic interdependence. Specifically, BiEvLight exploits the strong gradient correlation between images and events to build a gradient-guided event denoising prior that alleviates insufficient denoising in heavily noisy regions. Moreover, instead of treating event denoising as a static pre-processing stage-which inevitably incurs a trade-off between over- and under-denoising and cannot adapt to the requirements of a specific enhancement objective-we recast it as a bilevel optimization problem constrained by the enhancement task. Through cross-task interaction, the upper-level denoising problem learns event representations tailored to the lower-level enhancement objective, thereby substantially improving overall enhancement quality. Extensive experiments on the Real-world noise Dataset SDE demonstrate that our method significantly outperforms state-of-the-art (SOTA) approaches, with average improvements of 1.30dB in PSNR, 2.03dB in PSNR* and 0.047 in SSIM, respectively. The code will be publicly available at https://github.com/iijjlk/BiEvlight.

  </details>



- **Revisiting an Old Perspective Projection for Monocular 3D Morphable Models Regression**  
  Toby Chong, Ryota Nakajima  
  _2026-03-05_ · https://arxiv.org/abs/2603.04958v1  
  <details><summary>Abstract</summary>

  We introduce a novel camera model for monocular 3D Morphable Model (3DMM) regression methods that effectively captures the perspective distortion effect commonly seen in close-up facial images. Fitting 3D morphable models to video is a key technique in content creation. In particular, regression-based approaches have produced fast and accurate results by matching the rendered output of the morphable model to the target image. These methods typically achieve stable performance with orthographic projection, which eliminates the ambiguity between focal length and object distance. However, this simplification makes them unsuitable for close-up footage, such as that captured with head-mounted cameras. We extend orthographic projection with a new shrinkage parameter, incorporating a pseudo-perspective effect while preserving the stability of the original projection. We present several techniques that allow finetuning of existing models, and demonstrate the effectiveness of our modification through both quantitative and qualitative comparisons using a custom dataset recorded with head-mounted cameras.

  </details>



- **VisionPangu: A Compact and Fine-Grained Multimodal Assistant with 1.7B Parameters**  
  Jiaxin Fan, Wenpo Song  
  _2026-03-05_ · https://arxiv.org/abs/2603.04957v1  
  <details><summary>Abstract</summary>

  Large Multimodal Models (LMMs) have achieved strong performance in vision-language understanding, yet many existing approaches rely on large-scale architectures and coarse supervision, which limits their ability to generate detailed image captions. In this work, we present VisionPangu, a compact 1.7B-parameter multimodal model designed to improve detailed image captioning through efficient multimodal alignment and high-quality supervision. Our model combines an InternVL-derived vision encoder with the OpenPangu-Embedded language backbone via a lightweight MLP projector and adopts an instruction-tuning pipeline inspired by LLaVA. By incorporating dense human-authored descriptions from the DOCCI dataset, VisionPangu improves semantic coherence and descriptive richness without relying on aggressive model scaling. Experimental results demonstrate that compact multimodal models can achieve competitive performance while producing more structured and detailed captions. The code and model weights will be publicly available at https://www.modelscope.cn/models/asdfgh007/visionpangu.

  </details>



- **TimeWarp: Evaluating Web Agents by Revisiting the Past**  
  Md Farhan Ishmam, Kenneth Marino  
  _2026-03-05_ · https://arxiv.org/abs/2603.04949v1  
  <details><summary>Abstract</summary>

  The improvement of web agents on current benchmarks raises the question: Do today's agents perform just as well when the web changes? We introduce TimeWarp, a benchmark that emulates the evolving web using containerized environments that vary in UI, design, and layout. TimeWarp consists of three web environments, each with six UI versions spanning different eras of the internet, paired with a set of complex, realistic tasks requiring different forms of web navigation. Our experiments reveal web agents' vulnerability to changes and the limitations of behavior cloning (BC) on single-version trajectories. To address this, we propose TimeTraj, a simple yet effective algorithm that uses plan distillation to collect trajectories across multiple versions. By training agents on teacher rollouts using our BC-variant, we achieve substantial performance gains: $20.4\%\rightarrow37.7\%$ for Qwen-3 4B and $0\%\rightarrow27.0\%$ for Llama-3.1 8B models. We hope our work helps researchers study generalization across web designs and unlock a new paradigm for collecting plans rather than trajectories, thereby improving the robustness of web agents.

  </details>



- **Adaptive Prototype-based Interpretable Grading of Prostate Cancer**  
  Riddhasree Bhattacharyya, Pallabi Dutta, Sushmita Mitra  
  _2026-03-05_ · https://arxiv.org/abs/2603.04947v1  
  <details><summary>Abstract</summary>

  Prostate cancer being one of the frequently diagnosed malignancy in men, the rising demand for biopsies places a severe workload on pathologists. The grading procedure is tedious and subjective, motivating the development of automated systems. Although deep learning has made inroads in terms of performance, its limited interpretability poses challenges for widespread adoption in high-stake applications like medicine. Existing interpretability techniques for prostate cancer classifiers provide a coarse explanation but do not reveal why the highlighted regions matter. In this scenario, we propose a novel prototype-based weakly-supervised framework for an interpretable grading of prostate cancer from histopathology images. These networks can prove to be more trustworthy since their explicit reasoning procedure mirrors the workflow of a pathologist in comparing suspicious regions with clinically validated examples. The network is initially pre-trained at patch-level to learn robust prototypical features associated with each grade. In order to adapt it to a weakly-supervised setup for prostate cancer grading, the network is fine-tuned with a new prototype-aware loss function. Finally, a new attention-based dynamic pruning mechanism is introduced to handle inter-sample heterogeneity, while selectively emphasizing relevant prototypes for optimal performance. Extensive validation on the benchmark PANDA and SICAP datasets confirms that the framework can serve as a reliable assistive tool for pathologists in their routine diagnostic workflows.

  </details>



- **Person Detection and Tracking from an Overhead Crane LiDAR**  
  Nilusha Jayawickrama, Henrik Toikka, Risto Ojala  
  _2026-03-05_ · https://arxiv.org/abs/2603.04938v1  
  <details><summary>Abstract</summary>

  This paper investigates person detection and tracking in an industrial indoor workspace using a LiDAR mounted on an overhead crane. The overhead viewpoint introduces a strong domain shift from common vehicle-centric LiDAR benchmarks, and limited availability of suitable public training data. Henceforth, we curate a site-specific overhead LiDAR dataset with 3D human bounding-box annotations and adapt selected candidate 3D detectors under a unified training and evaluation protocol. We further integrate lightweight tracking-by-detection using AB3DMOT and SimpleTrack to maintain person identities over time. Detection performance is reported with distance-sliced evaluation to quantify the practical operating envelope of the sensing setup. The best adapted detector configurations achieve average precision (AP) up to 0.84 within a 5.0 m horizontal radius, increasing to 0.97 at 1.0 m, with VoxelNeXt and SECOND emerging as the most reliable backbones across this range. The acquired results contribute in bridging the domain gap between standard driving datasets and overhead sensing for person detection and tracking. We also report latency measurements, highlighting practical real-time feasibility. Finally, we release our dataset and implementations in GitHub to support further research

  </details>



- **VPWEM: Non-Markovian Visuomotor Policy with Working and Episodic Memory**  
  Yuheng Lei, Zhixuan Liang, Hongyuan Zhang, Ping Luo  
  _2026-03-05_ · https://arxiv.org/abs/2603.04910v1  
  <details><summary>Abstract</summary>

  Imitation learning from human demonstrations has achieved significant success in robotic control, yet most visuomotor policies still condition on single-step observations or short-context histories, making them struggle with non-Markovian tasks that require long-term memory. Simply enlarging the context window incurs substantial computational and memory costs and encourages overfitting to spurious correlations, leading to catastrophic failures under distribution shift and violating real-time constraints in robotic systems. By contrast, humans can compress important past experiences into long-term memories and exploit them to solve tasks throughout their lifetime. In this paper, we propose VPWEM, a non-Markovian visuomotor policy equipped with working and episodic memories. VPWEM retains a sliding window of recent observation tokens as short-term working memory, and introduces a Transformer-based contextual memory compressor that recursively converts out-of-window observations into a fixed number of episodic memory tokens. The compressor uses self-attention over a cache of past summary tokens and cross-attention over a cache of historical observations, and is trained jointly with the policy. We instantiate VPWEM on diffusion policies to exploit both short-term and episode-wide information for action generation with nearly constant memory and computation per step. Experiments demonstrate that VPWEM outperforms state-of-the-art baselines including diffusion policies and vision-language-action (VLA) models by more than 20% on the memory-intensive manipulation tasks in MIKASA and achieves an average 5% improvement on the mobile manipulation benchmark MoMaRT. Code is available at https://github.com/HarryLui98/code_vpwem.

  </details>



- **Interpretable Pre-Release Baseball Pitch Type Anticipation from Broadcast 3D Kinematics**  
  Jerrin Bright, Michelle Lu, John Zelek  
  _2026-03-05_ · https://arxiv.org/abs/2603.04874v1  
  <details><summary>Abstract</summary>

  How much can a pitcher's body reveal about the upcoming pitch? We study this question at scale by classifying eight pitch types from monocular 3D pose sequences, without access to ball-flight data. Our pipeline chains a diffusion-based 3D pose backbone with automatic pitching-event detection, groundtruth-validated biomechanical feature extraction, and gradient-boosted classification over 229 kinematic features. Evaluated on 119,561 professional pitches, the largest such benchmark to date, we achieve 80.4\% accuracy using body kinematics alone. A systematic importance analysis reveals that upper-body mechanics contribute 64.9\% of the predictive signal versus 35.1\% for the lower body, with wrist position (14.8\%) and trunk lateral tilt emerging as the most informative joint group and biomechanical feature, respectively. We further show that grip-defined variants (four-seam vs.\ two-seam fastball) are not separable from pose, establishing an empirical ceiling near 80\% and delineating where kinematic information ends and ball-flight information begins.

  </details>



- **Diffusion-Based sRGB Real Noise Generation via Prompt-Driven Noise Representation Learning**  
  Jaekyun Ko, Dongjin Kim, Soomin Lee, Guanghui Wang, Tae Hyun Kim  
  _2026-03-05_ · https://arxiv.org/abs/2603.04870v1  
  <details><summary>Abstract</summary>

  Denoising in the sRGB image space is challenging due to noise variability. Although end-to-end methods perform well, their effectiveness in real-world scenarios is limited by the scarcity of real noisy-clean image pairs, which are expensive and difficult to collect. To address this limitation, several generative methods have been developed to synthesize realistic noisy images from limited data. These generative approaches often rely on camera metadata during both training and testing to synthesize real-world noise. However, the lack of metadata or inconsistencies between devices restricts their usability. Therefore, we propose a novel framework called Prompt-Driven Noise Generation (PNG). This model is capable of acquiring high-dimensional prompt features that capture the characteristics of real-world input noise and creating a variety of realistic noisy images consistent with the distribution of the input noise. By eliminating the dependency on explicit camera metadata, our approach significantly enhances the generalizability and applicability of noise synthesis. Comprehensive experiments reveal that our model effectively produces realistic noisy images and show the successful application of these generated images in removing real-world noise across various benchmark datasets.

  </details>



- **On Multi-Step Theorem Prediction via Non-Parametric Structural Priors**  
  Junbo Zhao, Ting Zhang, Can Li, Wei He, Jingdong Wang, Hua Huang  
  _2026-03-05_ · https://arxiv.org/abs/2603.04852v1  
  <details><summary>Abstract</summary>

  Multi-step theorem prediction is a central challenge in automated reasoning. Existing neural-symbolic approaches rely heavily on supervised parametric models, which exhibit limited generalization to evolving theorem libraries. In this work, we explore training-free theorem prediction through the lens of in-context learning (ICL). We identify a critical scalability bottleneck, termed Structural Drift: as reasoning depth increases, the performance of vanilla ICL degrades sharply, often collapsing to near zero. We attribute this failure to the LLM's inability to recover latent topological dependencies, leading to unstructured exploration. To address this issue, we propose Theorem Precedence Graphs, which encode temporal dependencies from historical solution traces as directed graphs, and impose explicit topological constraints that effectively prune the search space during inference. Coupled with retrieval-augmented graph construction and a stepwise symbolic executor, our approach enables LLMs to act as structured planners without any gradient-based optimization. Experiments on the FormalGeo7k benchmark show that our method achieves 89.29% accuracy, substantially outperforming ICL baselines and matching state-of-the-art supervised models. These results indicate that explicit structural priors offer a promising direction for scaling LLM-based symbolic reasoning.

  </details>



- **Hyperbolic Multiview Pretraining for Robotic Manipulation**  
  Jin Yang, Ping Wei, Yixin Chen  
  _2026-03-05_ · https://arxiv.org/abs/2603.04848v1  
  <details><summary>Abstract</summary>

  3D-aware visual pretraining has proven effective in improving the performance of downstream robotic manipulation tasks. However, existing methods are constrained to Euclidean embedding spaces, whose flat geometry limits their ability to model structural relations among embeddings. As a result, they struggle to learn structured embeddings that are essential for robust spatial perception in robotic applications. To this end, we propose HyperMVP, a self-supervised framework for \underline{Hyper}bolic \underline{M}ulti\underline{V}iew \underline{P}retraining. Hyperbolic space offers geometric properties well suited for capturing structural relations. Methodologically, we extend the masked autoencoder paradigm and design a GeoLink encoder to learn multiview hyperbolic representations. The pretrained encoder is then finetuned with visuomotor policies on manipulation tasks. In addition, we introduce 3D-MOV, a large-scale dataset comprising multiple types of 3D point clouds to support pretraining. We evaluate HyperMVP on COLOSSEUM, RLBench, and real-world scenarios, where it consistently outperforms strong baselines across diverse tasks and perturbation settings. Our results highlight the potential of 3D-aware pretraining in a non-Euclidean space for learning robust and generalizable robotic manipulation policies.

  </details>



- **Revisiting Shape from Polarization in the Era of Vision Foundation Models**  
  Chenhao Li, Taishi Ono, Takeshi Uemori, Yusuke Moriuchi  
  _2026-03-05_ · https://arxiv.org/abs/2603.04817v1  
  <details><summary>Abstract</summary>

  We show that, with polarization cues, a lightweight model trained on a small dataset can outperform RGB-only vision foundation models (VFMs) in single-shot object-level surface normal estimation. Shape from polarization (SfP) has long been studied due to the strong physical relationship between polarization and surface geometry. Meanwhile, driven by scaling laws, RGB-only VFMs trained on large datasets have recently achieved impressive performance and surpassed existing SfP methods. This situation raises questions about the necessity of polarization cues, which require specialized hardware and have limited training data. We argue that the weaker performance of prior SfP methods does not come from the polarization modality itself, but from domain gaps. These domain gaps mainly arise from two sources. First, existing synthetic datasets use limited and unrealistic 3D objects, with simple geometry and random texture maps that do not match the underlying shapes. Second, real-world polarization signals are often affected by sensor noise, which is not well modeled during training. To address the first issue, we render a high-quality polarization dataset using 1,954 3D-scanned real-world objects. We further incorporate pretrained DINOv3 priors to improve generalization to unseen objects. To address the second issue, we introduce polarization sensor-aware data augmentation that better reflects real-world conditions. With only 40K training scenes, our method significantly outperforms both state-of-the-art SfP approaches and RGB-only VFMs. Extensive experiments show that polarization cues enable a 33x reduction in training data or an 8x reduction in model parameters, while still achieving better performance than RGB-only counterparts.

  </details>



- **Diffusion Policy through Conditional Proximal Policy Optimization**  
  Ben Liu, Shunpeng Yang, Hua Chen  
  _2026-03-05_ · https://arxiv.org/abs/2603.04790v1  
  <details><summary>Abstract</summary>

  Reinforcement learning (RL) has been extensively employed in a wide range of decision-making problems, such as games and robotics. Recently, diffusion policies have shown strong potential in modeling multi-modal behaviors, enabling more diverse and flexible action generation compared to the conventional Gaussian policy. Despite various attempts to combine RL with diffusion, a key challenge is the difficulty of computing action log-likelihood under the diffusion model. This greatly hinders the direct application of diffusion policies in on-policy reinforcement learning. Most existing methods calculate or approximate the log-likelihood through the entire denoising process in the diffusion model, which can be memory- and computationally inefficient. To overcome this challenge, we propose a novel and efficient method to train a diffusion policy in an on-policy setting that requires only evaluating a simple Gaussian probability. This is achieved by aligning the policy iteration with the diffusion process, which is a distinct paradigm compared to previous work. Moreover, our formulation can naturally handle entropy regularization, which is often difficult to incorporate into diffusion policies. Experiments demonstrate that the proposed method produces multimodal policy behaviors and achieves superior performance on a variety of benchmark tasks in both IsaacLab and MuJoCo Playground.

  </details>



- **MADCrowner: Margin Aware Dental Crown Design with Template Deformation and Refinement**  
  Linda Wei, Chang Liu, Wenran Zhang, Yuxuan Hu, Ruiyang Li, Feng Qi, Changyao Tian, Ke Wang, Yuanyuan Wang, Shaoting Zhang, et al.  
  _2026-03-05_ · https://arxiv.org/abs/2603.04771v1  
  <details><summary>Abstract</summary>

  Dental crown restoration is one of the most common treatment modalities for tooth defect, where personalized dental crown design is critical. While computer-aided design (CAD) systems have notably enhanced the efficiency of dental crown design, extensive manual adjustments are still required in the clinic workflow. Recent studies have explored the application of learning-based methods for the automated generation of restorative dental crowns. Nevertheless, these approaches were challenged by inadequate spatial resolution, noisy outputs, and overextension of surface reconstruction. To address these limitations, we propose \totalframework, a margin-aware mesh generation framework comprising CrownDeformR and CrownSegger. Inspired by the clinic manual workflow of dental crown design, we designed CrownDeformR to deform an initial template to the target crown based on anatomical context, which is extracted by a multi-scale intraoral scan encoder. Additionally, we introduced \marginseg, a novel margin segmentation network, to extract the cervical margin of the target tooth. The performance of CrownDeformR improved with the cervical margin as an extra constraint. And it was also utilized as the boundary condition for the tailored postprocessing method, which removed the overextended area of the reconstructed surface. We constructed a large-scale intraoral scan dataset and performed extensive experiments. The proposed method significantly outperformed existing approaches in both geometric accuracy and clinical feasibility.

  </details>



- **Evaluating and Correcting Human Annotation Bias in Dynamic Micro-Expression Recognition**  
  Feng Liu, Bingyu Nan, Xuezhong Qian, Xiaolan Fu  
  _2026-03-05_ · https://arxiv.org/abs/2603.04766v1  
  <details><summary>Abstract</summary>

  Existing manual labeling of micro-expressions is subject to errors in accuracy, especially in cross-cultural scenarios where deviation in labeling of key frames is more prominent. To address this issue, this paper presents a novel Global Anti-Monotonic Differential Selection Strategy (GAMDSS) architecture for enhancing the effectiveness of spatio-temporal modeling of micro-expressions through keyframe re-selection. Specifically, the method identifies Onset and Apex frames, which are characterized by significant micro-expression variation, from complete micro-expression action sequences via a dynamic frame reselection mechanism. It then uses these to determine Offset frames and construct a rich spatio-temporal dynamic representation. A two-branch structure with shared parameters is then used to efficiently extract spatio-temporal features. Extensive experiments are conducted on seven widely recognized micro-expression datasets. The results demonstrate that GAMDSS effectively reduces subjective errors caused by human factors in multicultural datasets such as SAMM and 4DME. Furthermore, quantitative analyses confirm that offset-frame annotations in multicultural datasets are more uncertain, providing theoretical justification for standardizing micro-expression annotations. These findings directly support our argument for reconsidering the validity and generalizability of dataset annotation paradigms. Notably, this design can be integrated into existing models without increasing the number of parameters, offering a new approach to enhancing micro-expression recognition performance. The source code is available on GitHub[https://github.com/Cross-Innovation-Lab/GAMDSS].

  </details>



- **Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset**  
  Yang Zou, Jun Ma, Zhidong Jiao, Xingyuan Li, Zhiying Jiang, Jinyuan Liu  
  _2026-03-05_ · https://arxiv.org/abs/2603.04745v1  
  <details><summary>Abstract</summary>

  Infrared image super-resolution (IISR) under real-world conditions is a practically significant yet rarely addressed task. Pioneering works are often trained and evaluated on simulated datasets or neglect the intrinsic differences between infrared and visible imaging. In practice, however, real infrared images are affected by coupled optical and sensing degradations that jointly deteriorate both structural sharpness and thermal fidelity. To address these challenges, we propose Real-IISR, a unified autoregressive framework for real-world IISR that progressively reconstructs fine-grained thermal structures and clear backgrounds in a scale-by-scale manner via thermal-structural guided visual autoregression. Specifically, a Thermal-Structural Guidance module encodes thermal priors to mitigate the mismatch between thermal radiation and structural edges. Since non-uniform degradations typically induce quantization bias, Real-IISR adopts a Condition-Adaptive Codebook that dynamically modulates discrete representations based on degradation-aware thermal priors. Also, a Thermal Order Consistency Loss enforces a monotonic relation between temperature and pixel intensity, ensuring relative brightness order rather than absolute values to maintain physical consistency under spatial misalignment and thermal drift. We build FLIR-IISR, a real-world IISR dataset with paired LR-HR infrared images acquired via automated focus variation and motion-induced blur. Extensive experiments demonstrate the promising performance of Real-IISR, providing a unified foundation for real-world IISR and benchmarking. The dataset and code are available at: https://github.com/JZD151/Real-IISR.

  </details>



- **A Benchmark Study of Neural Network Compression Methods for Hyperspectral Image Classification**  
  Sai Shi  
  _2026-03-05_ · https://arxiv.org/abs/2603.04720v1  
  <details><summary>Abstract</summary>

  Deep neural networks have achieved strong performance in image classification tasks due to their ability to learn complex patterns from high-dimensional data. However, their large computational and memory requirements often limit deployment on resource-constrained platforms such as remote sensing devices and edge systems. Network compression techniques have therefore been proposed to reduce model size and computational cost while maintaining predictive performance. In this study, we conduct a systematic evaluation of neural network compression methods for a remote sensing application, namely hyperspectral land cover classification. Specifically, we examine three widely used compression strategies for convolutional neural networks: pruning, quantization, and knowledge distillation. Experiments are conducted on two benchmark hyperspectral datasets, considering classification accuracy, memory consumption, and inference efficiency. Our results demonstrate that compressed models can significantly reduce model size and computational cost while maintaining competitive classification performance. These findings provide insights into the trade-offs between compression ratio, efficiency, and accuracy, and highlight the potential of compression techniques for enabling efficient deep learning deployment in remote sensing applications.

  </details>



- **Decoding the Pulse of Reasoning VLMs in Multi-Image Understanding Tasks**  
  Chenjun Li  
  _2026-03-04_ · https://arxiv.org/abs/2603.04676v1  
  <details><summary>Abstract</summary>

  Multi-image reasoning remains a significant challenge for vision-language models (VLMs). We investigate a previously overlooked phenomenon: during chain-of-thought (CoT) generation, the text-to-image (T2I) attention of reasoning VLMs exhibits diffuse "pulses": sporadic and unfocused attention patterns that fail to concentrate on task-relevant images. We further reveal a systematic positional bias in attention allocation across images. Motivated by these observations, we propose PulseFocus, a training-free, inference-time method that structures CoT reasoning into interleaved plan/focus blocks with soft attention gating. By forcing the model to explicitly plan which image to examine and then gating decode-time attention to the referenced image, PulseFocus sharpens attention focus and yields consistent improvements on multi-image benchmarks like BLINK benchmark (+3.7%) and MuirBench (+1.07%).

  </details>



- **RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies**  
  Yinpei Dai, Hongze Fu, Jayjun Lee, Yuejiang Liu, Haoran Zhang, Jianing Yang, Chelsea Finn, Nima Fazeli, Joyce Chai  
  _2026-03-04_ · https://arxiv.org/abs/2603.04639v1  
  <details><summary>Abstract</summary>

  Memory is critical for long-horizon and history-dependent robotic manipulation. Such tasks often involve counting repeated actions or manipulating objects that become temporarily occluded. Recent vision-language-action (VLA) models have begun to incorporate memory mechanisms; however, their evaluations remain confined to narrow, non-standardized settings. This limits their systematic understanding, comparison, and progress measurement. To address these challenges, we introduce RoboMME: a large-scale standardized benchmark for evaluating and advancing VLA models in long-horizon, history-dependent scenarios. Our benchmark comprises 16 manipulation tasks constructed under a carefully designed taxonomy that evaluates temporal, spatial, object, and procedural memory. We further develop a suite of 14 memory-augmented VLA variants built on the π0.5 backbone to systematically explore different memory representations across multiple integration strategies. Experimental results show that the effectiveness of memory representations is highly task-dependent, with each design offering distinct advantages and limitations across different tasks. Videos and code can be found at our website https://robomme.github.io.

  </details>


