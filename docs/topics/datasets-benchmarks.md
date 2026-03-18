# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-03-18 07:16 UTC_

Total papers shown: **50**


---

- **WorldCam: Interactive Autoregressive 3D Gaming Worlds with Camera Pose as a Unifying Geometric Representation**  
  Jisu Nam, Yicong Hong, Chun-Hao Paul Huang, Feng Liu, JoungBin Lee, Jiyoung Kim, Siyoon Jin, Yunsung Lee, Jaeyoon Jung, Suhwan Choi, et al.  
  _2026-03-17_ · https://arxiv.org/abs/2603.16871v1  
  <details><summary>Abstract</summary>

  Recent advances in video diffusion transformers have enabled interactive gaming world models that allow users to explore generated environments over extended horizons. However, existing approaches struggle with precise action control and long-horizon 3D consistency. Most prior works treat user actions as abstract conditioning signals, overlooking the fundamental geometric coupling between actions and the 3D world, whereby actions induce relative camera motions that accumulate into a global camera pose within a 3D world. In this paper, we establish camera pose as a unifying geometric representation to jointly ground immediate action control and long-term 3D consistency. First, we define a physics-based continuous action space and represent user inputs in the Lie algebra to derive precise 6-DoF camera poses, which are injected into the generative model via a camera embedder to ensure accurate action alignment. Second, we use global camera poses as spatial indices to retrieve relevant past observations, enabling geometrically consistent revisiting of locations during long-horizon navigation. To support this research, we introduce a large-scale dataset comprising 3,000 minutes of authentic human gameplay annotated with camera trajectories and textual descriptions. Extensive experiments show that our approach substantially outperforms state-of-the-art interactive gaming world models in action controllability, long-horizon visual quality, and 3D spatial consistency.

  </details>



- **MessyKitchens: Contact-rich object-level 3D scene reconstruction**  
  Junaid Ahmed Ansari, Ran Ding, Fabio Pizzati, Ivan Laptev  
  _2026-03-17_ · https://arxiv.org/abs/2603.16868v1  
  <details><summary>Abstract</summary>

  Monocular 3D scene reconstruction has recently seen significant progress. Powered by the modern neural architectures and large-scale data, recent methods achieve high performance in depth estimation from a single image. Meanwhile, reconstructing and decomposing common scenes into individual 3D objects remains a hard challenge due to the large variety of objects, frequent occlusions and complex object relations. Notably, beyond shape and pose estimation of individual objects, applications in robotics and animation require physically-plausible scene reconstruction where objects obey physical principles of non-penetration and realistic contacts. In this work we advance object-level scene reconstruction along two directions. First, we introduceMessyKitchens, a new dataset with real-world scenes featuring cluttered environments and providing high-fidelity object-level ground truth in terms of 3D object shapes, poses and accurate object contacts. Second, we build on the recent SAM 3D approach for single-object reconstruction and extend it with Multi-Object Decoder (MOD) for joint object-level scene reconstruction. To validate our contributions, we demonstrate MessyKitchens to significantly improve previous datasets in registration accuracy and inter-object penetration. We also compare our multi-object reconstruction approach on three datasets and demonstrate consistent and significant improvements of MOD over the state of the art. Our new benchmark, code and pre-trained models will become publicly available on our project website: https://messykitchens.github.io/.

  </details>



- **ManiTwin: Scaling Data-Generation-Ready Digital Object Dataset to 100K**  
  Kaixuan Wang, Tianxing Chen, Jiawei Liu, Honghao Su, Shaolong Zhu, Minxuan Wang, Zixuan Li, Yue Chen, Huan-ang Gao, Yusen Qin, et al.  
  _2026-03-17_ · https://arxiv.org/abs/2603.16866v1  
  <details><summary>Abstract</summary>

  Learning in simulation provides a useful foundation for scaling robotic manipulation capabilities. However, this paradigm often suffers from a lack of data-generation-ready digital assets, in both scale and diversity. In this work, we present ManiTwin, an automated and efficient pipeline for generating data-generation-ready digital object twins. Our pipeline transforms a single image into simulation-ready and semantically annotated 3D asset, enabling large-scale robotic manipulation data generation. Using this pipeline, we construct ManiTwin-100K, a dataset containing 100K high-quality annotated 3D assets. Each asset is equipped with physical properties, language descriptions, functional annotations, and verified manipulation proposals. Experiments demonstrate that ManiTwin provides an efficient asset synthesis and annotation workflow, and that ManiTwin-100K offers high-quality and diverse assets for manipulation data generation, random scene synthesis, and VQA data generation, establishing a strong foundation for scalable simulation data synthesis and policy learning. Our webpage is available at https://manitwin.github.io/.

  </details>



- **MolmoB0T: Large-Scale Simulation Enables Zero-Shot Manipulation**  
  Abhay Deshpande, Maya Guru, Rose Hendrix, Snehal Jauhri, Ainaz Eftekhar, Rohun Tripathi, Max Argus, Jordi Salvador, Haoquan Fang, Matthew Wallingford, et al.  
  _2026-03-17_ · https://arxiv.org/abs/2603.16861v1  
  <details><summary>Abstract</summary>

  A prevailing view in robot learning is that simulation alone is not enough; effective sim-to-real transfer is widely believed to require at least some real-world data collection or task-specific fine-tuning to bridge the gap between simulated and physical environments. We challenge that assumption. With sufficiently large-scale and diverse simulated synthetic training data, we show that zero-shot transfer to the real world is not only possible, but effective for both static and mobile manipulation. We introduce MolmoBot-Engine, a fully open-source pipeline for procedural data generation across robots, tasks, and diverse simulated environments in MolmoSpaces. With it, we release MolmoBot-Data, a dataset of 1.8 million expert trajectories for articulated object manipulation and pick-and-place tasks. We train three policy classes: MolmoBot, a Molmo2-based multi-frame vision-language model with a flow-matching action head; MolmoBot-Pi0, which replicates the $π_0$ architecture to enable direct comparison; and MolmoBot-SPOC, a lightweight policy suitable for edge deployment and amenable to RL fine-tuning. We evaluate on two robotic platforms: the Franka FR3 for tabletop manipulation tasks and the Rainbow Robotics RB-Y1 mobile manipulator for door opening, drawer manipulation, cabinet interaction, and mobile pick-and-place. Without any real-world fine-tuning, our policies achieve zero-shot transfer to unseen objects and environments. On tabletop pick-and-place, MolmoBot achieves a success rate of 79.2% in real world evaluations across 4 settings, outperforming $π_{0.5}$ at 39.2%. Our results demonstrate that procedural environment generation combined with diverse articulated assets can produce robust manipulation policies that generalize broadly to the real world. Technical Blog: https://allenai.org/blog/molmobot-robot-manipulation

  </details>



- **M^3: Dense Matching Meets Multi-View Foundation Models for Monocular Gaussian Splatting SLAM**  
  Kerui Ren, Guanghao Li, Changjian Jiang, Yingxiang Xu, Tao Lu, Linning Xu, Junting Dong, Jiangmiao Pang, Mulin Yu, Bo Dai  
  _2026-03-17_ · https://arxiv.org/abs/2603.16844v1  
  <details><summary>Abstract</summary>

  Streaming reconstruction from uncalibrated monocular video remains challenging, as it requires both high-precision pose estimation and computationally efficient online refinement in dynamic environments. While coupling 3D foundation models with SLAM frameworks is a promising paradigm, a critical bottleneck persists: most multi-view foundation models estimate poses in a feed-forward manner, yielding pixel-level correspondences that lack the requisite precision for rigorous geometric optimization. To address this, we present M^3, which augments the Multi-view foundation model with a dedicated Matching head to facilitate fine-grained dense correspondences and integrates it into a robust Monocular Gaussian Splatting SLAM. M^3 further enhances tracking stability by incorporating dynamic area suppression and cross-inference intrinsic alignment. Extensive experiments on diverse indoor and outdoor benchmarks demonstrate state-of-the-art accuracy in both pose estimation and scene reconstruction. Notably, M^3 reduces ATE RMSE by 64.3% compared to VGGT-SLAM 2.0 and outperforms ARTDECO by 2.11 dB in PSNR on the ScanNet++ dataset.

  </details>



- **An assessment of data-centric methods for label noise identification in remote sensing data sets**  
  Felix Kröber, Genc Hoxha, Ribana Roscher  
  _2026-03-17_ · https://arxiv.org/abs/2603.16835v1  
  <details><summary>Abstract</summary>

  Label noise in the sense of incorrect labels is present in many real-world data sets and is known to severely limit the generalizability of deep learning models. In the field of remote sensing, however, automated treatment of label noise in data sets has received little attention to date. In particular, there is a lack of systematic analysis of the performance of data-centric methods that not only cope with label noise but also explicitly identify and isolate noisy labels. In this paper, we examine three such methods and evaluate their behavior under different label noise assumptions. To do this, we inject different types of label noise with noise levels ranging from 10 to 70% into two benchmark data sets, followed by an analysis of how well the selected methods filter the label noise and how this affects task performances. With our analyses, we clearly prove the value of data-centric methods for both parts - label noise identification and task performance improvements. Our analyses provide insights into which method is the best choice depending on the setting and objective. Finally, we show in which areas there is still a need for research in the transfer of data-centric label noise methods to remote sensing data. As such, our work is a step forward in bridging the methodological establishment of data-centric label noise methods and their usage in practical settings in the remote sensing domain.

  </details>



- **WildDepth: A Multimodal Dataset for 3D Wildlife Perception and Depth Estimation**  
  Muhammad Aamir, Naoya Muramatsu, Sangyun Shin, Matthew Wijers, Jiaxing Jhong, Xinyu Hou, Amir Patel, Andrew Markham  
  _2026-03-17_ · https://arxiv.org/abs/2603.16816v1  
  <details><summary>Abstract</summary>

  Depth estimation and 3D reconstruction have been extensively studied as core topics in computer vision. Starting from rigid objects with relatively simple geometric shapes, such as vehicles, the research has expanded to address general objects, including challenging deformable objects, such as humans and animals. However, for the animal, in particular, the majority of existing models are trained based on datasets without metric scale, which can help validate image-only models. To address this limitation, we present WildDepth, a multimodal dataset and benchmark suite for depth estimation, behavior detection, and 3D reconstruction from diverse categories of animals ranging from domestic to wild environments with synchronized RGB and LiDAR. Experimental results show that the use of multi-modal data improves depth reliability by up to 10% RMSE, while RGB-LiDAR fusion enhances 3D reconstruction fidelity by 12% in Chamfer distance. By releasing WildDepth and its benchmarks, we aim to foster robust multimodal perception systems that generalize across domains.

  </details>



- **DexGrasp-Zero: A Morphology-Aligned Policy for Zero-Shot Cross-Embodiment Dexterous Grasping**  
  Yuliang Wu, Yanhan Lin, WengKit Lao, Yuhao Lin, Yi-Lin Wei, Wei-Shi Zheng, Ancong Wu  
  _2026-03-17_ · https://arxiv.org/abs/2603.16806v1  
  <details><summary>Abstract</summary>

  To meet the demands of increasingly diverse dexterous hand hardware, it is crucial to develop a policy that enables zero-shot cross-embodiment grasping without redundant re-learning. Cross-embodiment alignment is challenging due to heterogeneous hand kinematics and physical constraints. Existing approaches typically predict intermediate motion targets and retarget them to each embodiment, which may introduce errors and violate embodiment-specific limits, hindering transfer across diverse hands. To overcome these limitations, we propose \textit{DexGrasp-Zero}, a policy that learns universal grasping skills from diverse embodiments, enabling zero-shot transfer to unseen hands. We first introduce a morphology-aligned graph representation that maps each hand's kinematic keypoints to anatomically grounded nodes and equips each node with tri-axial orthogonal motion primitives, enabling structural and semantic alignment across different morphologies. Relying on this graph-based representation, we design a \textit{Morphology-Aligned Graph Convolutional Network} (MAGCN) to encode the graph for policy learning. MAGCN incorporates a \textit{Physical Property Injection} mechanism that fuses hand-specific physical constraints into the graph features, enabling adaptive compensation for varying link lengths and actuation limits for precise and stable grasping. Our extensive simulation evaluations on the YCB dataset demonstrate that our policy, jointly trained on four heterogeneous hands (Allegro, Shadow, Schunk, Ability), achieves an 85\% zero-shot success rate on unseen hardware (LEAP, Inspire), outperforming the state-of-the-art method by 59.5\%. Real-world experiments further evaluate our policy on three robot platforms (LEAP, Inspire, Revo2), achieving an 82\% average success rate on unseen objects.

  </details>



- **IOSVLM: A 3D Vision-Language Model for Unified Dental Diagnosis from Intraoral Scans**  
  Huimin Xiong, Zijie Meng, Tianxiang Hu, Chenyi Zhou, Yang Feng, Zuozhu Liu  
  _2026-03-17_ · https://arxiv.org/abs/2603.16781v1  
  <details><summary>Abstract</summary>

  3D intraoral scans (IOS) are increasingly adopted in routine dentistry due to abundant geometric evidence, and unified multi-disease diagnosis is desirable for clinical documentation and communication. While recent works introduce dental vision-language models (VLMs) to enable unified diagnosis and report generation on 2D images or multi-view images rendered from IOS, they do not fully leverage native 3D geometry. Such work is necessary and also challenging, due to: (i) heterogeneous scan forms and the complex IOS topology, (ii) multi-disease co-occurrence with class imbalance and fine-grained morphological ambiguity, (iii) limited paired 3D IOS-text data. Thus, we present IOSVLM, an end-to-end 3D VLM that represents scans as point clouds and follows a 3D encoder-projector-LLM design for unified diagnosis and generative visual question-answering (VQA), together with IOSVQA, a large-scale multi-source IOS diagnosis VQA dataset comprising 19,002 cases and 249,055 VQA pairs over 23 oral diseases and heterogeneous scan types. To address the distribution gap between color-free IOS data and color-dependent 3D pre-training, we propose a geometry-to-chromatic proxy that stabilizes fine-grained geometric perception and cross-modal alignment. A two-stage curriculum training strategy further enhances robustness. IOSVLM consistently outperforms strong baselines, achieving gains of at least +9.58% macro accuracy and +1.46% macro F1, indicating the effectiveness of direct 3D geometry modeling for IOS-based diagnosis.

  </details>



- **SuCor: Susceptibility Distortion Correction via Parameter-Free and Self-Regularized Optimal Transport**  
  Sreekar Chigurupati, Eleftherios Garyfallidis  
  _2026-03-17_ · https://arxiv.org/abs/2603.16758v1  
  <details><summary>Abstract</summary>

  We present SuCor, a method for correcting susceptibility induced geometric distortions in echo planar imaging (EPI) using optimal transport (OT) along the phase encoding direction. Given a pair of reversed phase encoding EPI volumes, we model each column of the distortion field as a Wasserstein-2 barycentric displacement between the opposing-polarity intensity profiles. Regularization is performed in the spectral domain using a bending-energy penalty whose strength is selected automatically via the Morozov discrepancy principle, requiring no manual tuning. On a human connectome project (HCP) dataset with left-right/right-left b0 EPI pairs and a co-registered T1 structural reference, SuCor achieves a mean volumetric mutual information of 0.341 with the T1 image, compared to 0.317 for FSL TOPUP, while running in approximately 12 seconds on a single CPU core.

  </details>



- **Semi-supervised Latent Disentangled Diffusion Model for Textile Pattern Generation**  
  Chenggong Hu, Yi Wang, Mengqi Xue, Haofei Zhang, Jie Song, Li Sun  
  _2026-03-17_ · https://arxiv.org/abs/2603.16747v1  
  <details><summary>Abstract</summary>

  Textile pattern generation (TPG) aims to synthesize fine-grained textile pattern images based on given clothing images. Although previous studies have not explicitly investigated TPG, existing image-to-image models appear to be natural candidates for this task. However, when applied directly, these methods often produce unfaithful results, failing to preserve fine-grained details due to feature confusion between complex textile patterns and the inherent non-rigid texture distortions in clothing images. In this paper, we propose a novel method, SLDDM-TPG, for faithful and high-fidelity TPG. Our method consists of two stages: (1) a latent disentangled network (LDN) that resolves feature confusion in clothing representations and constructs a multi-dimensional, independent clothing feature space; and (2) a semi-supervised latent diffusion model (S-LDM), which receives guidance signals from LDN and generates faithful results through semi-supervised diffusion training, combined with our designed fine-grained alignment strategy. Extensive evaluations show that SLDDM-TPG reduces FID by 4.1 and improves SSIM by up to 0.116 on our CTP-HD dataset, and also demonstrate good generalization on the VITON-HD dataset.

  </details>



- **When the City Teaches the Car: Label-Free 3D Perception from Infrastructure**  
  Zhen Xu, Jinsu Yoo, Cristian Bautista, Zanming Huang, Tai-Yu Pan, Zhenzhen Liu, Katie Z Luo, Mark Campbell, Bharath Hariharan, Wei-Lun Chao  
  _2026-03-17_ · https://arxiv.org/abs/2603.16742v1  
  <details><summary>Abstract</summary>

  Building robust 3D perception for self-driving still relies heavily on large-scale data collection and manual annotation, yet this paradigm becomes impractical as deployment expands across diverse cities and regions. Meanwhile, modern cities are increasingly instrumented with roadside units (RSUs), static sensors deployed along roads and at intersections to monitor traffic. This raises a natural question: can the city itself help train the vehicle? We propose infrastructure-taught, label-free 3D perception, a paradigm in which RSUs act as stationary, unsupervised teachers for ego vehicles. Leveraging their fixed viewpoints and repeated observations, RSUs learn local 3D detectors from unlabeled data and broadcast predictions to passing vehicles, which are aggregated as pseudo-label supervision for training a standalone ego detector. The resulting model requires no infrastructure or communication at test time. We instantiate this idea as a fully label-free three-stage pipeline and conduct a concept-and-feasibility study in a CARLA-based multi-agent environment. With CenterPoint, our pipeline achieves 82.3% AP for detecting vehicles, compared to a fully supervised ego upper bound of 94.4%. We further systematically analyze each stage, evaluate its scalability, and demonstrate complementarity with existing ego-centric label-free methods. Together, these results suggest that city infrastructure itself can potentially provide a scalable supervisory signal for autonomous vehicles, positioning infrastructure-taught learning as a promising orthogonal paradigm for reducing annotation cost in 3D perception.

  </details>



- **Emotion-Aware Classroom Quality Assessment Leveraging IoT-Based Real-Time Student Monitoring**  
  Hai Nguyen, Hieu Dao, Hung Nguyen, Nam Vu, Cong Tran  
  _2026-03-17_ · https://arxiv.org/abs/2603.16719v1  
  <details><summary>Abstract</summary>

  This study presents high-throughput, real-time multi-agent affective computing framework designed to enhance classroom learning through emotional state monitoring. As large classroom sizes and limited teacher student interaction increasingly challenge educators, there is a growing need for scalable, data-driven tools capable of capturing students' emotional and engagement patterns in real time. The system was evaluated using the Classroom Emotion Dataset, consisting of 1,500 labeled images and 300 classroom detection videos. Tailored for IoT devices, the system addresses load balancing and latency challenges through efficient real-time processing. Field testing was conducted across three educational institutions in a large metropolitan area: a primary school (hereafter school A), a secondary school (school B), and a high school (school C). The system demonstrated robust performance, detecting up to 50 faces at 25 FPS and achieving 88% overall accuracy in classifying classroom engagement states. Implementation results showed positive outcomes, with favorable feedback from students, teachers, and parents regarding improved classroom interaction and teaching adaptation. Key contributions of this research include establishing a practical, IoT-based framework for emotion-aware learning environments and introducing the 'Classroom Emotion Dataset' to facilitate further validation and research.

  </details>



- **HMAR: Hierarchical Modality-Aware Expert and Dynamic Routing Medical Image Retrieval Architecture**  
  Aojie Yuan  
  _2026-03-17_ · https://arxiv.org/abs/2603.16679v1  
  <details><summary>Abstract</summary>

  Medical image retrieval (MIR) is a critical component of computer-aided diagnosis, yet existing systems suffer from three persistent limitations: uniform feature encoding that fails to account for the varying clinical importance of anatomical structures, ambiguous similarity metrics based on coarse classification labels, and an exclusive focus on global image similarity that cannot meet the clinical demand for fine-grained region-specific retrieval. We propose HMAR (Hierarchical Modality-Aware Expert and Dynamic Routing), an adaptive retrieval framework built on a Mixture-of-Experts (MoE) architecture. HMAR employs a dual-expert mechanism: Expert0 extracts global features for holistic similarity matching, while Expert1 learns position-invariant local representations for precise lesion-region retrieval. A two-stage contrastive learning strategy eliminates the need for expensive bounding-box annotations, and a sliding-window matching algorithm enables dense local comparison at inference time. Hash codes are generated via Kolmogorov-Arnold Network (KAN) layers for efficient Hamming-distance search. Experiments on the RadioImageNet-CT dataset (16 clinical patterns, 29,903 images) show that HMAR achieves mean Average Precision (mAP) of 0.711 and 0.724 for 64-bit and 128-bit hash codes, improving over the state-of-the-art ACIR method by 0.7% and 1.1%, respectively.

  </details>



- **When Should a Robot Think? Resource-Aware Reasoning via Reinforcement Learning for Embodied Robotic Decision-Making**  
  Jun Liu, Pu Zhao, Zhenglun Kong, Xuan Shen, Peiyan Dong, Fan Yang, Lin Cui, Hao Tang, Geng Yuan, Wei Niu, et al.  
  _2026-03-17_ · https://arxiv.org/abs/2603.16673v1  
  <details><summary>Abstract</summary>

  Embodied robotic systems increasingly rely on large language model (LLM)-based agents to support high-level reasoning, planning, and decision-making during interactions with the environment. However, invoking LLM reasoning introduces substantial computational latency and resource overhead, which can interrupt action execution and reduce system reliability. Excessive reasoning may delay actions, while insufficient reasoning often leads to incorrect decisions and task failures. This raises a fundamental question for embodied agents: when should the agent reason, and when should it act? In this work, we propose RARRL (Resource-Aware Reasoning via Reinforcement Learning), a hierarchical framework for resource-aware orchestration of embodied agents. Rather than learning low-level control policies, RARRL learns a high-level orchestration policy that operates at the agent's decision-making layer. This policy enables the agent to adaptively determine whether to invoke reasoning, which reasoning role to employ, and how much computational budget to allocate based on current observations, execution history, and remaining resources. Extensive experiments, including evaluations with empirical latency profiles derived from the ALFRED benchmark, show that RARRL consistently improves task success rates while reducing execution latency and enhancing robustness compared with fixed or heuristic reasoning strategies. These results demonstrate that adaptive reasoning control is essential for building reliable and efficient embodied robotic agents.

  </details>



- **Kinema4D: Kinematic 4D World Modeling for Spatiotemporal Embodied Simulation**  
  Mutian Xu, Tianbao Zhang, Tianqi Liu, Zhaoxi Chen, Xiaoguang Han, Ziwei Liu  
  _2026-03-17_ · https://arxiv.org/abs/2603.16669v1  
  <details><summary>Abstract</summary>

  Simulating robot-world interactions is a cornerstone of Embodied AI. Recently, a few works have shown promise in leveraging video generations to transcend the rigid visual/physical constraints of traditional simulators. However, they primarily operate in 2D space or are guided by static environmental cues, ignoring the fundamental reality that robot-world interactions are inherently 4D spatiotemporal events that require precise interactive modeling. To restore this 4D essence while ensuring the precise robot control, we introduce Kinema4D, a new action-conditioned 4D generative robotic simulator that disentangles the robot-world interaction into: i) Precise 4D representation of robot controls: we drive a URDF-based 3D robot via kinematics, producing a precise 4D robot control trajectory. ii) Generative 4D modeling of environmental reactions: we project the 4D robot trajectory into a pointmap as a spatiotemporal visual signal, controlling the generative model to synthesize complex environments' reactive dynamics into synchronized RGB/pointmap sequences. To facilitate training, we curated a large-scale dataset called Robo4D-200k, comprising 201,426 robot interaction episodes with high-quality 4D annotations. Extensive experiments demonstrate that our method effectively simulates physically-plausible, geometry-consistent, and embodiment-agnostic interactions that faithfully mirror diverse real-world dynamics. For the first time, it shows potential zero-shot transfer capability, providing a high-fidelity foundation for advancing next-generation embodied simulation.

  </details>



- **Efficient Brood Cell Detection in Layer Trap Nests for Bees and Wasps: Balancing Labeling Effort and Species Coverage**  
  Chenchang Liu, Felix Fornoff, Annika Grasreiner, Patrick Maeder, Henri Greil, Marco Seeland  
  _2026-03-17_ · https://arxiv.org/abs/2603.16652v1  
  <details><summary>Abstract</summary>

  Monitoring cavity-nesting wild bees and wasps is vital for biodiversity research and conservation. Layer trap nests (LTNs) are emerging as a valuable tool to study the abundance and species richness of these insects, offering insights into their nesting activities and ecological needs. However, manually evaluating LTNs to detect and classify brood cells is labor-intensive and time-consuming. To address this, we propose a deep learning based approach for efficient brood cell detection and classification in LTNs. LTNs present additional challenges due to densely packed brood cells, leading to a high labeling effort per image. Moreover, we observe a significant imbalance in class distribution, with common species having notably more occurrences than rare species. Comprehensive labeling of common species is time-consuming and exacerbates data imbalance, while partial labeling introduces data incompleteness which degrades model performance. To reduce labeling effort and mitigate the impact of unlabeled data, we introduce a novel Constrained False Positive Loss (CFPL) strategy. CFPL dynamically masks predictions from unlabeled data, preventing them from interfering with the classification loss during training. We evaluate our approach on a dataset of 712 LTN images collected over one season, covering 28 fine-grained classes describing the taxonomy and status of brood cells. To minimize labeling effort, we limit the training set to a maximum of 300 labels per class. Experimental results demonstrate that deep learning can be effectively used to detect brood cells in LTNs. Our CFPL method further improves performance and balances model accuracy and labeling effort while also mitigating class imbalance.

  </details>



- **Mixture of Style Experts for Diverse Image Stylization**  
  Shihao Zhu, Ziheng Ouyang, Yijia Kang, Qilong Wang, Mi Zhou, Bo Li, Ming-Ming Cheng, Qibin Hou  
  _2026-03-17_ · https://arxiv.org/abs/2603.16649v1  
  <details><summary>Abstract</summary>

  Diffusion-based stylization has advanced significantly, yet existing methods are limited to color-driven transformations, neglecting complex semantics and material details.We introduce StyleExpert, a semantic-aware framework based on the Mixture of Experts (MoE). Our framework employs a unified style encoder, trained on our large-scale dataset of content-style-stylized triplets, to embed diverse styles into a consistent latent space. This embedding is then used to condition a similarity-aware gating mechanism, which dynamically routes styles to specialized experts within the MoE architecture. Leveraging this MoE architecture, our method adeptly handles diverse styles spanning multiple semantic levels, from shallow textures to deep semantics. Extensive experiments show that StyleExpert outperforms existing approaches in preserving semantics and material details, while generalizing to unseen styles. Our code and collected images are available at the project page: https://hh-lg.github.io/StyleExpert-Page/.

  </details>



- **BUSSARD: Normalizing Flows for Bijective Universal Scene-Specific Anomalous Relationship Detection**  
  Melissa Schween, Mathis Kruse, Bodo Rosenhahn  
  _2026-03-17_ · https://arxiv.org/abs/2603.16645v1  
  <details><summary>Abstract</summary>

  We propose Bijective Universal Scene-Specific Anomalous Relationship Detection (BUSSARD), a normalizing flow-based model for detecting anomalous relations in scene graphs, generated from images. Our work follows a multimodal approach, embedding object and relationship tokens from scene graphs with a language model to leverage semantic knowledge from the real world. A normalizing flow model is used to learn bijective transformations that map object-relation-object triplets from scene graphs to a simple base distribution (typically Gaussian), allowing anomaly detection through likelihood estimation. We evaluate our approach on the SARD dataset containing office and dining room scenes. Our method achieves around 10% better AUROC results compared to the current state-of-the-art model, while simultaneously being five times faster. Through ablation studies, we demonstrate superior robustness and universality, particularly regarding the use of synonyms, with our model maintaining stable performance while the baseline shows 17.5% deviation. This work demonstrates the strong potential of learning-based methods for relationship anomaly detection in scene graphs. Our code is available at https://github.com/mschween/BUSSARD .

  </details>



- **MLLM-based Textual Explanations for Face Comparison**  
  Redwan Sony, Anil K Jain, Ross Arun  
  _2026-03-17_ · https://arxiv.org/abs/2603.16629v1  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLMs) have recently been proposed as a means to generate natural-language explanations for face recognition decisions. While such explanations facilitate human interpretability, their reliability on unconstrained face images remains underexplored. In this work, we systematically analyze MLLM-generated explanations for the unconstrained face verification task on the challenging IJB-S dataset, with a particular focus on extreme pose variation and surveillance imagery. Our results show that even when MLLMs produce correct verification decisions, the accompanying explanations frequently rely on non-verifiable or hallucinated facial attributes that are not supported by visual evidence. We further study the effect of incorporating information from traditional face recognition systems, viz., scores and decisions, alongside the input images. Although such information improves categorical verification performance, it does not consistently lead to faithful explanations. To evaluate the explanations beyond decision accuracy, we introduce a likelihood-ratio-based framework that measures the evidential strength of textual explanations. Our findings highlight fundamental limitations of current MLLMs for explainable face recognition and underscore the need for a principled evaluation of reliable and trustworthy explanations in biometric applications. Code is available at https://github.com/redwankarimsony/LR-MLLMFR-Explainability.

  </details>



- **TCATSeg: A Tooth Center-Wise Attention Network for 3D Dental Model Semantic Segmentation**  
  Qiang He, Wentian Qu, Jiajia Dai, Changsong Lei, Shaofeng Wang, Feifei Zuo, Yajie Wang, Yaqian Liang, Xiaoming Deng, Cuixia Ma, et al.  
  _2026-03-17_ · https://arxiv.org/abs/2603.16620v1  
  <details><summary>Abstract</summary>

  Accurate semantic segmentation of 3D dental models is essential for digital dentistry applications such as orthodontics and dental implants. However, due to complex tooth arrangements and similarities in shape among adjacent teeth, existing methods struggle with accurate segmentation, because they often focus on local geometry while neglecting global contextual information. To address this, we propose TCATSeg, a novel framework that combines local geometric features with global semantic context. We introduce a set of sparse yet physically meaningful superpoints to capture global semantic relationships and enhance segmentation accuracy. Additionally, we present a new dataset of 400 dental models, including pre-orthodontic samples, to evaluate the generalization of our method. Extensive experiments demonstrate that TCATSeg outperforms state-of-the-art approaches.

  </details>



- **ACPV-Net: All-Class Polygonal Vectorization for Seamless Vector Map Generation from Aerial Imagery**  
  Weiqin Jiao, Hao Cheng, George Vosselman, Claudio Persello  
  _2026-03-17_ · https://arxiv.org/abs/2603.16616v1  
  <details><summary>Abstract</summary>

  We tackle the problem of generating a complete vector map representation from aerial imagery in a single run: producing polygons for all land-cover classes with shared boundaries and without gaps or overlaps. Existing polygonization methods are typically class-specific; extending them to multiple classes via per-class runs commonly leads to topological inconsistencies, such as duplicated edges, gaps, and overlaps. We formalize this new task as All-Class Polygonal Vectorization (ACPV) and release the first public benchmark, Deventer-512, with standardized metrics jointly evaluating semantic fidelity, geometric accuracy, vertex efficiency, per-class topological fidelity and global topological consistency. To realize ACPV, we propose ACPV-Net, a unified framework introducing a novel Semantically Supervised Conditioning (SSC) mechanism coupling semantic perception with geometric primitive generation, along with a topological reconstruction that enforces shared-edge consistency by design. While enforcing such strict topological constraints, ACPV-Net surpasses all class-specific baselines in polygon quality across classes on Deventer-512. It also applies to single-class polygonal vectorization without any architectural modification, achieving the best-reported results on WHU-Building. Data, code, and models will be released at: https://github.com/HeinzJiao/ACPV-Net.

  </details>



- **FSMC-Pose: Frequency and Spatial Fusion with Multiscale Self-calibration for Cattle Mounting Pose Estimation**  
  Fangjing Li, Zhihai Wang, Xinxin Ding, Haiyang Liu, Ronghua Gao, Rong Wang, Yao Zhu, Ming Jin  
  _2026-03-17_ · https://arxiv.org/abs/2603.16596v1  
  <details><summary>Abstract</summary>

  Mounting posture is an important visual indicator of estrus in dairy cattle. However, achieving reliable mounting pose estimation in real-world environments remains challenging due to cluttered backgrounds and frequent inter-animal occlusion. We present FSMC-Pose, a top-down framework that integrates a lightweight frequency-spatial fusion backbone, CattleMountNet, and a multiscale self-calibration head, SC2Head. Specifically, we design two algorithmic components for CattleMountNet: the Spatial Frequency Enhancement Block (SFEBlock) and the Receptive Aggregation Block (RABlock). SFEBlock separates cattle from cluttered backgrounds, while RABlock captures multiscale contextual information. The Spatial-Channel Self-Calibration Head (SC2Head) attends to spatial and channel dependencies and introduces a self-calibration branch to mitigate structural misalignment under inter-animal overlap. We construct a mounting dataset, MOUNT-Cattle, covering 1176 mounting instances, which follows the COCO format and supports drop-in training across pose estimation models. Using a comprehensive dataset that combines MOUNT-Cattle with the public NWAFU-Cattle dataset, FSMC-Pose achieves higher accuracy than strong baselines, with markedly lower computational and parameter costs, while maintaining real-time inference on commodity GPUs. Extensive experiments and qualitative analyses show that FSMC-Pose effectively captures and estimates cattle mounting pose in complex and cluttered environments. Dataset and code are available at https://github.com/elianafang/FSMC-Pose.

  </details>



- **SAMSEM -- A Generic and Scalable Approach for IC Metal Line Segmentation**  
  Christian Gehrmann, Jonas Ricker, Simon Damm, Deruo Cheng, Julian Speith, Yiqiong Shi, Asja Fischer, Christof Paar  
  _2026-03-17_ · https://arxiv.org/abs/2603.16548v1  
  <details><summary>Abstract</summary>

  In light of globalized hardware supply chains, the assurance of hardware components has gained significant interest, particularly in cryptographic applications and high-stakes scenarios. Identifying metal lines on scanning electron microscope (SEM) images of integrated circuits (ICs) is one essential step in verifying the absence of malicious circuitry in chips manufactured in untrusted environments. Due to varying manufacturing processes and technologies, such verification usually requires tuning parameters and algorithms for each target IC. Often, a machine learning model trained on images of one IC fails to accurately detect metal lines on other ICs. To address this challenge, we create SAMSEM by adapting Meta's Segment Anything Model 2 (SAM2) to the domain of IC metal line segmentation. Specifically, we develop a multi-scale segmentation approach that can handle SEM images of varying sizes, resolutions, and magnifications. Furthermore, we deploy a topology-based loss alongside pixel-based losses to focus our segmentation on electrical connectivity rather than pixel-level accuracy. Based on a hyperparameter optimization, we then fine-tune the SAM2 model to obtain a model that generalizes across different technology nodes, manufacturing materials, sample preparation methods, and SEM imaging technologies. To this end, we leverage an unprecedented dataset of SEM images obtained from 48 metal layers across 14 different ICs. When fine-tuned on seven ICs, SAMSEM achieves an error rate as low as 0.72% when evaluated on other images from the same ICs. For the remaining seven unseen ICs, it still achieves error rates as low as 5.53%. Finally, when fine-tuned on all 14 ICs, we observe an error rate of 0.62%. Hence, SAMSEM proves to be a reliable tool that significantly advances the frontier in metal line segmentation, a key challenge in post-manufacturing IC verification.

  </details>



- **Conservative Offline Robot Policy Learning via Posterior-Transition Reweighting**  
  Wanpeng Zhang, Hao Luo, Sipeng Zheng, Yicheng Feng, Haiweng Xu, Ziheng Xi, Chaoyi Xu, Haoqi Yuan, Zongqing Lu  
  _2026-03-17_ · https://arxiv.org/abs/2603.16542v1  
  <details><summary>Abstract</summary>

  Offline post-training adapts a pretrained robot policy to a target dataset by supervised regression on recorded actions. In practice, robot datasets are heterogeneous: they mix embodiments, camera setups, and demonstrations of varying quality, so many trajectories reflect recovery behavior, inconsistent operator skill, or weakly informative supervision. Uniform post-training gives equal credit to all samples and can therefore average over conflicting or low-attribution data. We propose Posterior-Transition Reweighting (PTR), a reward-free and conservative post-training method that decides how much each training sample should influence the supervised update. For each sample, PTR encodes the observed post-action consequence as a latent target, inserts it into a candidate pool of mismatched targets, and uses a separate transition scorer to estimate a softmax identification posterior over target indices. The posterior-to-uniform ratio defines the PTR score, which is converted into a clipped-and-mixed weight and applied to the original action objective through self-normalized weighted regression. This construction requires no tractable policy likelihood and is compatible with both diffusion and flow-matching action heads. Rather than uniformly trusting all recorded supervision, PTR reallocates credit according to how attributable each sample's post-action consequence is under the current representation, improving conservative offline adaptation to heterogeneous robot data.

  </details>



- **VIEW2SPACE: Studying Multi-View Visual Reasoning from Sparse Observations**  
  Fucai Ke, Zhixi Cai, Boying Li, Long Chen, Beibei Lin, Weiqing Wang, Pari Delir Haghighi, Gholamreza Haffari, Hamid Rezatofighi  
  _2026-03-17_ · https://arxiv.org/abs/2603.16506v1  
  <details><summary>Abstract</summary>

  Multi-view visual reasoning is essential for intelligent systems that must understand complex environments from sparse and discrete viewpoints, yet existing research has largely focused on single-image or temporally dense video settings. In real-world scenarios, reasoning across views requires integrating partial observations without explicit guidance, while collecting large-scale multi-view data with accurate geometric and semantic annotations remains challenging. To address this gap, we leverage physically grounded simulation to construct diverse, high-fidelity 3D scenes with precise per-view metadata, enabling scalable data generation that remains transferable to real-world settings. Based on this engine, we introduce VIEW2SPACE, a multi-dimensional benchmark for sparse multi-view reasoning, together with a scalable, disjoint training split supporting millions of grounded question-answer pairs. Using this benchmark, a comprehensive evaluation of state-of-the-art vision-language and spatial models reveals that multi-view reasoning remains largely unsolved, with most models performing only marginally above random guessing. We further investigate whether training can bridge this gap. Our proposed Grounded Chain-of-Thought with Visual Evidence substantially improves performance under moderate difficulty, and generalizes to real-world data, outperforming existing approaches in cross-dataset evaluation. We further conduct difficulty-aware scaling analyses across model size, data scale, reasoning depth, and visibility constraints, indicating that while geometric perception can benefit from scaling under sufficient visibility, deep compositional reasoning across sparse views remains a fundamental challenge.

  </details>



- **DST-Net: A Dual-Stream Transformer with Illumination-Independent Feature Guidance and Multi-Scale Spatial Convolution for Low-Light Image Enhancement**  
  Yicui Shi, Yuhan Chen, Xiangfei Huang, Zhenguo Wang, Wenxuan Yu, Ying Fang  
  _2026-03-17_ · https://arxiv.org/abs/2603.16482v1  
  <details><summary>Abstract</summary>

  Low-light image enhancement aims to restore the visibility of images captured by visual sensors in dim environments by addressing their inherent signal degradations, such as luminance attenuation and structural corruption. Although numerous algorithms attempt to improve image quality, existing methods often cause a severe loss of intrinsic signal priors. To overcome these challenges, we propose a Dual-Stream Transformer Network (DST-Net) based on illumination-agnostic signal prior guidance and multi-scale spatial convolutions. First, to address the loss of critical signal features under low-light conditions, we design a feature extraction module. This module integrates Difference of Gaussians (DoG), LAB color space transformations, and VGG-16 for texture extraction, utilizing decoupled illumination-agnostic features as signal priors to continuously guide the enhancement process. Second, we construct a dual-stream interaction architecture. By employing a cross-modal attention mechanism, the network leverages the extracted priors to dynamically rectify the deteriorated signal representation of the enhanced image, ultimately achieving iterative enhancement through differentiable curve estimation. Furthermore, to overcome the inability of existing methods to preserve fine structures and textures, we propose a Multi-Scale Spatial Fusion Block (MSFB) featuring pseudo-3D and 3D gradient operator convolutions. This module integrates explicit gradient operators to recover high-frequency edges while capturing inter-channel spatial correlations via multi-scale spatial convolutions. Extensive evaluations and ablation studies demonstrate that DST-Net achieves superior performance in subjective visual quality and objective metrics. Specifically, our method achieves a PSNR of 25.64 dB on the LOL dataset. Subsequent validation on the LSRW dataset further confirms its robust cross-scene generalization.

  </details>



- **TinyGLASS: Real-Time Self-Supervised In-Sensor Anomaly Detection**  
  Pietro Bonazzi, Rafael Sutter, Luigi Capogrosso, Mischa Buob, Michele Magno  
  _2026-03-17_ · https://arxiv.org/abs/2603.16451v1  
  <details><summary>Abstract</summary>

  Anomaly detection plays a key role in industrial quality control, where defects must be identified despite the scarcity of labeled faulty samples. Recent self-supervised approaches, such as GLASS, learn normal visual patterns using only defect-free data and have shown strong performance on industrial benchmarks. However, their computational requirements limit deployment on resource-constrained edge platforms. This work introduces TinyGLASS, a lightweight adaptation of the GLASS framework designed for real-time in-sensor anomaly detection on the Sony IMX500 intelligent vision sensor. The proposed architecture replaces the original WideResNet-50 backbone with a compact ResNet-18 and introduces deployment-oriented modifications that enable static graph tracing and INT8 quantization using Sony's Model Compression Toolkit. In addition to evaluating performance on the MVTec-AD benchmark, we investigate robustness to contaminated training data and introduce a custom industrial dataset, named MMS Dataset, for cross-device evaluation. Experimental results show that TinyGLASS achieves 8.7x parameter compression while maintaining competitive detection performance, reaching 94.2% image-level AUROC on MVTec-AD and operating at 20 FPS within the 8 MB memory constraints of the IMX500 platform. System profiling demonstrates low power consumption (4.0 mJ per inference), real-time end-to-end latency (20 FPS), and high energy efficiency (470 GMAC/J). Furthermore, the model maintains stable performance under moderate levels of training data contamination.

  </details>



- **Unified Removal of Raindrops and Reflections: A New Benchmark and A Novel Pipeline**  
  Xingyu Liu, Zewei He, Yu Chen, Chunyu Zhu, Zixuan Chen, Xing Luo, Zhe-Ming Lu  
  _2026-03-17_ · https://arxiv.org/abs/2603.16446v1  
  <details><summary>Abstract</summary>

  When capturing images through glass surfaces or windshields on rainy days, raindrops and reflections frequently co-occur to significantly reduce the visibility of captured images. This practical problem lacks attention and needs to be resolved urgently. Prior de-raindrop, de-reflection, and all-in-one models have failed to address this composite degradation. To this end, we first formally define the unified removal of raindrops and reflections (UR$^3$) task for the first time and construct a real-shot dataset, namely RainDrop and ReFlection (RDRF), which provides a new benchmark with substantial, high-quality, diverse image pairs. Then, we propose a novel diffusion-based framework (i.e., DiffUR$^3$) with several target designs to address this challenging task. By leveraging the powerful generative prior, DiffUR$^3$ successfully removes both types of degradations. Extensive experiments demonstrate that our method achieves state-of-the-art performance on our benchmark and on challenging in-the-wild images. The RDRF dataset and the codes will be made public upon acceptance.

  </details>



- **IRIS: A Real-World Benchmark for Inverse Recovery and Identification of Physical Dynamic Systems from Monocular Video**  
  Rasul Khanbayov, Mohamed Rayan Barhdadi, Erchin Serpedin, Hasan Kurban  
  _2026-03-17_ · https://arxiv.org/abs/2603.16432v1  
  <details><summary>Abstract</summary>

  Unsupervised physical parameter estimation from video lacks a common benchmark: existing methods evaluate on non-overlapping synthetic data, the sole real-world dataset is restricted to single-body systems, and no established protocol addresses governing-equation identification. This work introduces IRIS, a high-fidelity benchmark comprising 220 real-world videos captured at 4K resolution and 60\,fps, spanning both single- and multi-body dynamics with independently measured ground-truth parameters and uncertainty estimates. Each dynamical system is recorded under controlled laboratory conditions and paired with its governing equations, enabling principled evaluation. A standardized evaluation protocol is defined encompassing parameter accuracy, identifiability, extrapolation, robustness, and governing-equation selection. Multiple baselines are evaluated, including a multi-step physics loss formulation and four complementary equation-identification strategies (VLM temporal reasoning, describe-then-classify prompting, CNN-based classification, and path-based labelling), establishing reference performance across all IRIS scenarios and exposing systematic failure modes that motivate future research. The dataset, annotations, evaluation toolkit, and all baseline implementations are publicly released.

  </details>



- **LenghuSky-8: An 8-Year All-Sky Cloud Dataset with Star-Aware Masks and Alt-Az Calibration for Segmentation and Nowcasting**  
  Yicheng Rui, Xiao-Wei Duan, Licai Deng, Fan Yang, Zhengming Dang, Zhengjun Du, Junhao Peng, Wenhao Chu, Umut Mahmut, Kexin Li, et al.  
  _2026-03-17_ · https://arxiv.org/abs/2603.16429v1  
  <details><summary>Abstract</summary>

  Ground-based time-domain observatories require minute-by-minute, site-scale awareness of cloud cover, yet existing all-sky datasets are short, daylight-biased, or lack astrometric calibration. We present LenghuSky-8, an eight-year (2018-2025) all-sky imaging dataset from a premier astronomical site, comprising 429,620 $512 \times 512$ frames with 81.2% night-time coverage, star-aware cloud masks, background masks, and per-pixel altitude-azimuth (Alt-Az) calibration. For robust cloud segmentation across day, night, and lunar phases, we train a linear probe on DINOv3 local features and obtain 93.3% $\pm$ 1.1% overall accuracy on a balanced, manually labeled set of 1,111 images. Using stellar astrometry, we map each pixel to local alt-az coordinates and measure calibration uncertainties of approximately 0.37 deg at zenith and approximately 1.34 deg at 30 deg altitude, sufficient for integration with telescope schedulers. Beyond segmentation, we introduce a short-horizon nowcasting benchmark over per-pixel three-class logits (sky/cloud/contamination) with four baselines: persistence (copying the last frame), optical flow, ConvLSTM, and VideoGPT. ConvLSTM performs best but yields only limited gains over persistence, underscoring the difficulty of near-term cloud evolution. We release the dataset, calibrations, and an open-source toolkit for loading, evaluation, and scheduler-ready alt-az maps to boost research in segmentation, nowcasting, and autonomous observatory operations.

  </details>



- **Early-Terminable Energy-Safe Iterative Coupling for Parallel Simulation of Port-Hamiltonian Systems**  
  Qi Wei, Jianfeng Tao, Hongyu Nie, Wangtao Tan  
  _2026-03-17_ · https://arxiv.org/abs/2603.16424v1  
  <details><summary>Abstract</summary>

  Parallel simulation and control of large-scale robotic systems often rely on partitioned time stepping, yet finite-iteration coupling can inject spurious energy by violating power consistency--even when each subsystem is passive. This letter proposes a novel energy-safe, early-terminable iterative coupling for port-Hamiltonian subsystems by embedding a Douglas--Rachford (DR) splitting scheme in scattering (wave) coordinates. The lossless interconnection is enforced as an orthogonal constraint in the wave domain, while each subsystem contributes a discrete-time scattering port map induced by its one-step integrator. Under a discrete passivity condition on the subsystem time steps and a mild impedance-tuning condition, we prove an augmented-storage inequality certifying discrete passivity of the coupled macro-step for any finite inner-iteration budget, with the remaining mismatch captured by an explicit residual. As the inner budget increases, the partitioned update converges to the monolithic discrete-time update induced by the same integrators, yielding a principled, adaptive accuracy--compute trade-off, supporting energy-consistent real-time parallel simulation under varying computational budgets. Experiments on a coupled-oscillator benchmark validate the passivity certificates at numerical roundoff (on the order of 10e-14 in double precision) and show that the reported RMS state error decays monotonically with increasing inner-iteration budgets, consistent with the hard-coupling limit.

  </details>



- **DermaFlux: Synthetic Skin Lesion Generation with Rectified Flows for Enhanced Image Classification**  
  Stathis Galanakis, Alexandros Koliousis, Stefanos Zafeiriou  
  _2026-03-17_ · https://arxiv.org/abs/2603.16392v1  
  <details><summary>Abstract</summary>

  Despite recent advances in deep generative modeling, skin lesion classification systems remain constrained by the limited availability of large, diverse, and well-annotated clinical datasets, resulting in class imbalance between benign and malignant lesions and consequently reduced generalization performance. We introduce DermaFlux, a rectified flow-based text-to-image generative framework that synthesizes clinically grounded skin lesion images from natural language descriptions of dermatological attributes. Built upon Flux.1, DermaFlux is fine-tuned using parameter-efficient Low-Rank Adaptation (LoRA) on a large curated collection of publicly available clinical image datasets. We construct image-text pairs using synthetic textual captions generated by Llama 3.2, following established dermatological criteria including lesion asymmetry, border irregularity, and color variation. Extensive experiments demonstrate that DermaFlux generates diverse and clinically meaningful dermatology images that improve binary classification performance by up to 6% when augmenting small real-world datasets, and by up to 9% when classifiers are trained on DermaFlux-generated synthetic images rather than diffusion-based synthetic images. Our ImageNet-pretrained ViT fine-tuned with only 2,500 real images and 4,375 DermaFlux-generated samples achieves 78.04% binary classification accuracy and an AUC of 0.859, surpassing the next best dermatology model by 8%.

  </details>



- **InViC: Intent-aware Visual Cues for Medical Visual Question Answering**  
  Zhisong Wang, Ziyang Chen, Zanting Ye, Hongze Zhu, Yefeng Zheng, Yong Xia  
  _2026-03-17_ · https://arxiv.org/abs/2603.16372v1  
  <details><summary>Abstract</summary>

  Medical visual question answering (Med-VQA) aims to answer clinically relevant questions grounded in medical images. However, existing multimodal large language models (MLLMs) often exhibit shortcut answering, producing plausible responses by exploiting language priors or dataset biases while insufficiently attending to visual evidence. This behavior undermines clinical reliability, especially when subtle imaging findings are decisive. We propose a lightweight plug-in framework, termed Intent-aware Visual Cues (InViC), to explicitly enhance image-based answer generation in medical VQA. InViC introduces a Cue Tokens Extraction (CTE) module that distills dense visual tokens into a compact set of K question-conditioned cue tokens, which serve as structured visual intermediaries injected into the LLM decoder to promote intent-aligned visual evidence. To discourage bypassing of visual information, we further design a two-stage fine-tuning strategy with a cue-bottleneck attention mask. In Stage I, we employ an attention mask to block the LLM's direct view of raw visual features, thereby funneling all visual evidence through the cue pathway. In Stage II, standard causal attention is restored to train the LLM to jointly exploit the visual and cue tokens. We evaluate InViC on three public Med-VQA benchmarks (VQA-RAD, SLAKE, and ImageCLEF VQA-Med 2019) across multiple representative MLLMs. InViC consistently improves over zero-shot inference and standard LoRA fine-tuning, demonstrating that intent-aware visual cues with bottlenecked training is a practical and effective strategy for improving trustworthy Med-VQA.

  </details>



- **Automated identification of Ichneumonoidea wasps via YOLO-based deep learning: Integrating HiresCam for Explainable AI**  
  Joao Manoel Herrera Pinheiro, Gabriela Do Nascimento Herrera, Alvaro Doria Dos Santos, Luciana Bueno Dos Reis Fernandes, Ricardo V. Godoy, Eduardo A. B. Almeida, Helena Carolina Onody, Marcelo Andrade Da Costa Vieira, Angelica Maria Penteado-Dias, Marcelo Becker  
  _2026-03-17_ · https://arxiv.org/abs/2603.16351v1  
  <details><summary>Abstract</summary>

  Accurate taxonomic identification of parasitoid wasps within the superfamily Ichneumonoidea is essential for biodiversity assessment, ecological monitoring, and biological control programs. However, morphological similarity, small body size, and fine-grained interspecific variation make manual identification labor-intensive and expertise-dependent. This study proposes a deep learning-based framework for the automated identification of Ichneumonoidea wasps using a YOLO-based architecture integrated with High-Resolution Class Activation Mapping (HiResCAM) to enhance interpretability. The proposed system simultaneously identifies wasp families from high-resolution images. The dataset comprises 3556 high-resolution images of Hymenoptera specimens. The taxonomic distribution is primarily concentrated among the families Ichneumonidae (n = 786), Braconidae (n = 648), Apidae (n = 466), and Vespidae (n = 460). Extensive experiments were conducted using a curated dataset, with model performance evaluated through precision, recall, F1 score, and accuracy. The results demonstrate high accuracy of over 96 % and robust generalization across morphological variations. HiResCAM visualizations confirm that the model focuses on taxonomically relevant anatomical regions, such as wing venation, antennae segmentation, and metasomal structures, thereby validating the biological plausibility of the learned features. The integration of explainable AI techniques improves transparency and trustworthiness, making the system suitable for entomological research to accelerate biodiversity characterization in an under-described parasitoid superfamily.

  </details>



- **Toward Deep Representation Learning for Event-Enhanced Visual Autonomous Perception: the eAP Dataset**  
  Jinghang Li, Shichao Li, Qing Lian, Peiliang Li, Xiaozhi Chen, Yi Zhou  
  _2026-03-17_ · https://arxiv.org/abs/2603.16303v1  
  <details><summary>Abstract</summary>

  Recent visual autonomous perception systems achieve remarkable performances with deep representation learning. However, they fail in scenarios with challenging illumination.While event cameras can mitigate this problem, there is a lack of a large-scale dataset to develop event-enhanced deep visual perception models in autonomous driving scenes. To address the gap, we present the eAP (event-enhanced Autonomous Perception) dataset, the largest dataset with event cameras for autonomous perception. We demonstrate how eAP can facilitate the study of different autonomous perception tasks, including 3D vehicle detection and object time-to-contact (TTC) estimation, through deep representation learning. Based on eAP, we demonstrate the ffrst successful use of events to improve a popular 3D vehicle detection network in challenging illumination scenarios. eAP also enables a devoted study of the representation learning problem of object TTC estimation. We show how a geometryaware representation learning framework leads to the best eventbased object TTC estimation network that operates at 200 FPS. The dataset, code, and pre-trained models will be made publicly available for future research.

  </details>



- **VisBrowse-Bench: Benchmarking Visual-Native Search for Multimodal Browsing Agents**  
  Zhengbo Zhang, Jinbo Su, Zhaowen Zhou, Changtao Miao, Yuhan Hong, Qimeng Wu, Yumeng Liu, Feier Wu, Yihe Tian, Yuhao Liang, et al.  
  _2026-03-17_ · https://arxiv.org/abs/2603.16289v1  
  <details><summary>Abstract</summary>

  The rapid advancement of Multimodal Large Language Models (MLLMs) has enabled browsing agents to acquire and reason over multimodal information in the real world. But existing benchmarks suffer from two limitations: insufficient evaluation of visual reasoning ability and the neglect of native visual information of web pages in the reasoning chains. To address these challenges, we introduce a new benchmark for visual-native search, VisBrowse-Bench. It contains 169 VQA instances covering multiple domains and evaluates the models' visual reasoning capabilities during the search process through multimodal evidence cross-validation via text-image retrieval and joint reasoning. These data were constructed by human experts using a multi-stage pipeline and underwent rigorous manual verification. We additionally propose an agent workflow that can effectively drive the browsing agent to actively collect and reason over visual information during the search process. We comprehensively evaluated both open-source and closed-source models in this workflow. Experimental results show that even the best-performing model, Claude-4.6-Opus only achieves an accuracy of 47.6%, while the proprietary Deep Research model, o3-deep-research only achieves an accuracy of 41.1%. The code and data can be accessed at: https://github.com/ZhengboZhang/VisBrowse-Bench

  </details>



- **Locate-then-Sparsify: Attribution Guided Sparse Strategy for Visual Hallucination Mitigation**  
  TianTian Dang, Chao Bi, Shufan Shen, Jinzhe Liu, Qingming Huang, Shuhui Wang  
  _2026-03-17_ · https://arxiv.org/abs/2603.16284v1  
  <details><summary>Abstract</summary>

  Despite the significant advancements in Large Vision-Language Models (LVLMs), their tendency to generate hallucinations undermines reliability and restricts broader practical deployment. Among the hallucination mitigation methods, feature steering emerges as a promising approach that reduces erroneous outputs in LVLMs without increasing inference costs. However, current methods apply uniform feature steering across all layers. This heuristic strategy ignores inter-layer differences, potentially disrupting layers unrelated to hallucinations and ultimately leading to performance degradation on general tasks. In this paper, we propose a plug-and-play framework called Locate-Then-Sparsify for Feature Steering (LTS-FS), which controls the steering intensity according to the hallucination relevance of each layer. We first construct a synthetic dataset comprising token-level and sentence-level hallucination cases. Based on this dataset, we introduce an attribution method based on causal interventions to quantify the hallucination relevance of each layer. With the attribution scores across layers, we propose a layerwise strategy that converts these scores into feature steering intensities for individual layers, enabling more precise adjustments specifically on hallucination-relevant layers. Extensive experiments across multiple LVLMs and benchmarks demonstrate that our LTS-FS framework effectively mitigates hallucination while preserving strong performance.

  </details>



- **MG-Grasp: Metric-Scale Geometric 6-DoF Grasping Framework with Sparse RGB Observations**  
  Kangxu Wang, Siang Chen, Chenxing Jiang, Shaojie Shen, Yixiang Dai, Guijin Wang  
  _2026-03-17_ · https://arxiv.org/abs/2603.16270v1  
  <details><summary>Abstract</summary>

  Single-view RGB-D grasp detection remains a com- mon choice in 6-DoF robotic grasping systems, which typically requires a depth sensor. While RGB-only 6-DoF grasp methods has been studied recently, their inaccurate geometric repre- sentation is not directly suitable for physically reliable robotic manipulation, thereby hindering reliable grasp generation. To address these limitations, we propose MG-Grasp, a novel depth- free 6-DoF grasping framework that achieves high-quality object grasping. Leveraging two-view 3D foundation model with camera intrinsic/extrinsic, our method reconstructs metric- scale and multi-view consistent dense point clouds from sparse RGB images and generates stable 6-DoF grasp. Experiments on GraspNet-1Billion dataset and real world demonstrate that MG-Grasp achieves state-of-the-art (SOTA) grasp performance among RGB-based 6-DoF grasping methods.

  </details>



- **FG-SGL: Fine-Grained Semantic Guidance Learning via Motion Process Decomposition for Micro-Gesture Recognition**  
  Jinsheng Wei, Zhaodi Xu, Guanming Lu, Haoyu Chen, Jingjie Yan  
  _2026-03-17_ · https://arxiv.org/abs/2603.16269v1  
  <details><summary>Abstract</summary>

  Micro-gesture recognition (MGR) is challenging due to subtle inter-class variations. Existing methods rely on category-level supervision, which is insufficient for capturing subtle and localized motion differences. Thus, this paper proposes a Fine-Grained Semantic Guidance Learning (FG-SGL) framework that jointly integrates fine-grained and category-level semantics to guide vision--language models in perceiving local MG motions. FG-SA adopts fine-grained semantic cues to guide the learning of local motion features, while CP-A enhances the separability of MG features through category-level semantic guidance. To support fine-grained semantic guidance, this work constructs a fine-grained textual dataset with human annotations that describes the dynamic process of MGs in four refined semantic dimensions. Furthermore, a Multi-Level Contrastive Optimization strategy is designed to jointly optimize both modules in a coarse-to-fine pattern. Experiments show that FG-SGL achieves competitive performance, validating the effectiveness of fine-grained semantic guidance for MGR.

  </details>



- **AW-MoE: All-Weather Mixture of Experts for Robust Multi-Modal 3D Object Detection**  
  Hongwei Lin, Xun Huang, Chenglu Wen, Cheng Wang  
  _2026-03-17_ · https://arxiv.org/abs/2603.16261v1  
  <details><summary>Abstract</summary>

  Robust 3D object detection under adverse weather conditions is crucial for autonomous driving. However, most existing methods simply combine all weather samples for training while overlooking data distribution discrepancies across different weather scenarios, leading to performance conflicts. To address this issue, we introduce AW-MoE, the framework that innovatively integrates Mixture of Experts (MoE) into weather-robust multi-modal 3D object detection approaches. AW-MoE incorporates Image-guided Weather-aware Routing (IWR), which leverages the superior discriminability of image features across weather conditions and their invariance to scene variations for precise weather classification. Based on this accurate classification, IWR selects the top-K most relevant Weather-Specific Experts (WSE) that handle data discrepancies, ensuring optimal detection under all weather conditions. Additionally, we propose a Unified Dual-Modal Augmentation (UDMA) for synchronous LiDAR and 4D Radar dual-modal data augmentation while preserving the realism of scenes. Extensive experiments on the real-world dataset demonstrate that AW-MoE achieves ~ 15% improvement in adverse-weather performance over state-of-the-art methods, while incurring negligible inference overhead. Moreover, integrating AW-MoE into established baseline detectors yields performance improvements surpassing current state-of-the-art methods. These results show the effectiveness and strong scalability of our AW-MoE. We will release the code publicly at https://github.com/windlinsherlock/AW-MoE.

  </details>



- **Point-to-Mask: From Arbitrary Point Annotations to Mask-Level Infrared Small Target Detection**  
  Weihua Gao, Wenlong Niu, Jie Tang, Man Yang, Jiafeng Zhang, Xiaodong Peng  
  _2026-03-17_ · https://arxiv.org/abs/2603.16257v1  
  <details><summary>Abstract</summary>

  Infrared small target detection (IRSTD) methods predominantly formulate the task as pixel-level segmentation, which requires costly dense annotations and is not well suited to tiny targets with weak texture and ambiguous boundaries. To address this issue, we propose Point-to-Mask, a framework that bridges low-cost point supervision and mask-level detection through two components: a Physics-driven Adaptive Mask Generation (PAMG) module that converts point annotations into compact target masks and geometric cues, and a lightweight Radius-aware Point Regression Network (RPR-Net) that reformulates IRSTD as target center localization and effective radius regression using spatiotemporal motion cues. The two modules form a closed loop: PAMG generates pseudo masks and geometric supervision during training, while the geometric predictions of RPR-Net are fed back to PAMG for pixel-level mask recovery during inference. To facilitate systematic evaluation, we further construct SIRSTD-Pixel, a sequential dataset with refined pixel-level annotations. Experiments show that the proposed framework achieves strong pseudo-label quality, high detection accuracy, and efficient inference, approaching full-supervision performance under point-supervised settings with substantially lower annotation cost. Code and datasets will be available at: https://github.com/GaoScience/point-to-mask.

  </details>



- **Industrial cuVSLAM Benchmark & Integration**  
  Charbel Abi Hana, Kameel Amareen, Mohamad Mostafa, Dmitry Slepichev, Hesam Rabeti, Zheng Wang, Mihir Acharya, Anthony Rizk  
  _2026-03-17_ · https://arxiv.org/abs/2603.16240v1  
  <details><summary>Abstract</summary>

  This work presents a comprehensive benchmark evaluation of visual odometry (VO) and visual SLAM (VSLAM) systems for mobile robot navigation in real-world logistical environments. We compare multiple visual odometry approaches across controlled trajectories covering translational, rotational, and mixed motion patterns, as well as a large-scale production facility dataset spanning approximately 1.7 km. Performance is evaluated using Absolute Pose Error (APE) against ground truth from a Vicon motion capture system and a LiDAR-based SLAM reference. Our results show that a hybrid stack combining the cuVSLAM front-end with a custom SLAM back-end achieves the strongest mapping accuracy, motivating a deeper integration of cuVSLAM as the core VO component in our robotics stack. We further validate this integration by deploying and testing the cuVSLAM-based VO stack on an NVIDIA Jetson platform.

  </details>



- **Ground Reaction Inertial Poser: Physics-based Human Motion Capture from Sparse IMUs and Insole Pressure Sensors**  
  Ryosuke Hori, Jyun-Ting Song, Zhengyi Luo, Jinkun Cao, Soyong Shin, Hideo Saito, Kris Kitani  
  _2026-03-17_ · https://arxiv.org/abs/2603.16233v1  
  <details><summary>Abstract</summary>

  We propose Ground Reaction Inertial Poser (GRIP), a method that reconstructs physically plausible human motion using four wearable devices. Unlike conventional IMU-only approaches, GRIP combines IMU signals with foot pressure data to capture both body dynamics and ground interactions. Furthermore, rather than relying solely on kinematic estimation, GRIP uses a digital twin of a person, in the form of a synthetic humanoid in a physics simulator, to reconstruct realistic and physically plausible motion. At its core, GRIP consists of two modules: KinematicsNet, which estimates body poses and velocities from sensor data, and DynamicsNet, which controls the humanoid in the simulator using the residual between the KinematicsNet prediction and the simulated humanoid state. To enable robust training and fair evaluation, we introduce a large-scale dataset, Pressure and Inertial Sensing for Human Motion and Interaction (PRISM), that captures diverse human motions with synchronized IMUs and insole pressure sensors. Experimental results show that GRIP outperforms existing IMU-only and IMU-pressure fusion methods across all evaluated datasets, achieving higher global pose accuracy and improved physical consistency.

  </details>



- **Reliable Reasoning in SVG-LLMs via Multi-Task Multi-Reward Reinforcement Learning**  
  Haomin Wang, Qi Wei, Qianli Ma, Shengyuan Ding, Jinhui Yin, Kai Chen, Hongjie Zhang  
  _2026-03-17_ · https://arxiv.org/abs/2603.16189v1  
  <details><summary>Abstract</summary>

  With the rapid advancement of vision-language models, an increasing number of studies have explored their potential for SVG generation tasks. Although existing approaches improve performance by constructing large-scale SVG datasets and introducing SVG-specific tokens, they still suffer from limited generalization, redundant paths in code outputs, and a lack of explicit reasoning. In this work, we present CTRL-S (Chain-of-Thought Reinforcement Learning for SVG), a unified framework that introduces a chain-of-thought mechanism to explicitly expose the model's reasoning process during SVG generation. To support this structured reasoning, we construct SVG-Sophia, a high-quality dataset containing 145K samples across SVG code refinement, Text-to-SVG, and Image-to-SVG tasks. By training the model to generate group-level structured SVG code, CTRL-S significantly improves structural coherence and visual fidelity. Furthermore, we adopt the GRPO algorithm and design a multi-reward optimization framework, incorporating DINO, image-text similarity, format, and code efficiency rewards. Through joint multi-reward optimization and multi-task training, our approach systematically enhances overall generation capabilities. Extensive experiments show that CTRL-S outperforms existing methods, achieving higher task success rates, superior SVG code quality, and exceptional visual fidelity.

  </details>



- **ECHO: Edge-Cloud Humanoid Orchestration for Language-to-Motion Control**  
  Haozhe Jia, Jianfei Song, Yuan Zhang, Honglei Jin, Youcheng Fan, Wenshuo Chen, Wei Zhang, Yutao Yue  
  _2026-03-17_ · https://arxiv.org/abs/2603.16188v1  
  <details><summary>Abstract</summary>

  We present ECHO, an edge--cloud framework for language-driven whole-body control of humanoid robots. A cloud-hosted diffusion-based text-to-motion generator synthesizes motion references from natural language instructions, while an edge-deployed reinforcement-learning tracker executes them in closed loop on the robot. The two modules are bridged by a compact, robot-native 38-dimensional motion representation that encodes joint angles, root planar velocity, root height, and a continuous 6D root orientation per frame, eliminating inference-time retargeting from human body models and remaining directly compatible with low-level PD control. The generator adopts a 1D convolutional UNet with cross-attention conditioned on CLIP-encoded text features; at inference, DDIM sampling with 10 denoising steps and classifier-free guidance produces motion sequences in approximately one second on a cloud GPU. The tracker follows a Teacher--Student paradigm: a privileged teacher policy is distilled into a lightweight student equipped with an evidential adaptation module for sim-to-real transfer, further strengthened by morphological symmetry constraints and domain randomization. An autonomous fall recovery mechanism detects falls via onboard IMU readings and retrieves recovery trajectories from a pre-built motion library. We evaluate ECHO on a retargeted HumanML3D benchmark, where it achieves strong generation quality (FID 0.029, R-Precision Top-1 0.686) under a unified robot-domain evaluator, while maintaining high motion safety and trajectory consistency. Real-world experiments on a Unitree G1 humanoid demonstrate stable execution of diverse text commands with zero hardware fine-tuning.

  </details>



- **360° Image Perception with MLLMs: A Comprehensive Benchmark and a Training-Free Method**  
  Huyen T. T. Tran, Van-Quang Nguyen, Farros Alferro, Kang-Jun Liu, Takayuki Okatani  
  _2026-03-17_ · https://arxiv.org/abs/2603.16179v1  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLMs) have shown impressive abilities in understanding and reasoning over conventional images. However, their perception of 360° images remains largely underexplored. Unlike conventional images, 360° images capture the entire surrounding environment, enabling holistic spatial reasoning but introducing challenges such as geometric distortion and complex spatial relations. To comprehensively assess MLLMs' capabilities to perceive 360° images, we introduce 360Bench, a Visual Question Answering (VQA) benchmark featuring 7K-resolution 360° images, seven representative (sub)tasks with annotations carefully curated by human annotators. Using 360Bench, we systematically evaluate seven MLLMs and six enhancement methods, revealing their shortcomings in 360° image perception. To address these challenges, we propose Free360, a training-free scene-graph-based framework for high-resolution 360° VQA. Free360 decomposes the reasoning process into modular steps, applies adaptive spherical image transformations to 360° images tailored to each step, and seamlessly integrates the resulting information into a unified graph representation for answer generation. Experiments show that Free360 consistently improves its base MLLM and provides a strong training-free solution for 360° VQA tasks. The source code and dataset will be publicly released upon acceptance.

  </details>



- **SignNav: Leveraging Signage for Semantic Visual Navigation in Large-Scale Indoor Environments**  
  Jian Sun, Yuming Huang, He Li, Shuqi Xiao, Shenyan Guo, Maani Ghaffari, Qingbiao Li, Chengzhong Xu, Hui Kong  
  _2026-03-17_ · https://arxiv.org/abs/2603.16166v1  
  <details><summary>Abstract</summary>

  Humans routinely leverage semantic hints provided by signage to navigate to destinations within novel Large-Scale Indoor (LSI) environments, such as hospitals and airport terminals. However, this capability remains underexplored within the field of embodied navigation. This paper introduces a novel embodied navigation task, SignNav, which requires the agent to interpret semantic hint from signage and reason about the subsequent action based on current observation. To facilitate research in this domain, we construct the LSI-Dataset for the training and evaluation of various SignNav agents. Dynamically changing semantic hints and sparse placement of signage in LSI environments present significant challenges to the SignNav task. To address these challenges, we propose the Spatial-Temporal Aware Transformer (START) model for end-to-end decision-making. The spatial-aware module grounds the semantic hint of signage into physical world, while the temporal-aware module captures long-range dependencies between historical states and current observation. Leveraging a two-stage training strategy with Dataset Aggregation (DAgger), our approach achieves state-of-the-art performance, recording an 80% Success Rate (SR) and 0.74 NDTW on val-unseen split. Real-world deployment further demonstrates the practicality of our method in physical environment without pre-built map.

  </details>



- **STARK: Spatio-Temporal Attention for Representation of Keypoints for Continuous Sign Language Recognition**  
  Suvajit Patra, Soumitra Samanta  
  _2026-03-17_ · https://arxiv.org/abs/2603.16163v1  
  <details><summary>Abstract</summary>

  Continuous Sign Language Recognition (CSLR) is a crucial task for understanding the languages of deaf communities. Contemporary keypoint-based approaches typically rely on spatio-temporal encoding, where spatial interactions among keypoints are modeled using Graph Convolutional Networks or attention mechanisms, while temporal dynamics are captured using 1D convolutional networks. However, such designs often introduce a large number of parameters in both the encoder and the decoder. This paper introduces a unified spatio-temporal attention network that computes attention scores both spatially (across keypoints) and temporally (within local windows), and aggregates features to produce a local context-aware spatio-temporal representation. The proposed encoder contains approximately $70-80\%$ fewer parameters than existing state-of-the-art models while achieving comparable performance to keypoint-based methods on the Phoenix-14T dataset.

  </details>



- **EFF-Grasp: Energy-Field Flow Matching for Physics-Aware Dexterous Grasp Generation**  
  Yukun Zhao, Zichen Zhong, Yongshun Gong, Yilong Yin, Haoliang Sun  
  _2026-03-17_ · https://arxiv.org/abs/2603.16151v1  
  <details><summary>Abstract</summary>

  Denoising generative models have recently become the dominant paradigm for dexterous grasp generation, owing to their ability to model complex grasp distributions from large-scale data. However, existing diffusion-based methods typically formulate generation as a stochastic differential equation (SDE), which often requires many sequential denoising steps and introduces trajectory instability that can lead to physically infeasible grasps. In this paper, we propose EFF-Grasp, a novel Flow-Matching-based framework for physics-aware dexterous grasp generation. Specifically, we reformulate grasp synthesis as a deterministic ordinary differential equation (ODE) process, which enables efficient and stable generation through smooth probability flows. To further enforce physical feasibility, we introduce a training-free physics-aware energy guidance strategy. Our method defines an energy-guided target distribution using adapted explicit physical energy functions that capture key grasp constraints, and estimates the corresponding guidance term via a local Monte Carlo approximation during inference. In this way, EFF-Grasp dynamically steers the generation trajectory toward physically feasible regions without requiring additional physics-based training or simulation feedback. Extensive experiments on five benchmark datasets show that EFF-Grasp achieves superior performance in grasp quality and physical feasibility, while requiring substantially fewer sampling steps than diffusion-based baselines.

  </details>


