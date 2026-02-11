# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-02-11 07:17 UTC_

Total papers shown: **50**


---

- **SAGE: Scalable Agentic 3D Scene Generation for Embodied AI**  
  Hongchi Xia, Xuan Li, Zhaoshuo Li, Qianli Ma, Jiashu Xu, Ming-Yu Liu, Yin Cui, Tsung-Yi Lin, Wei-Chiu Ma, Shenlong Wang, et al.  
  _2026-02-10_ · https://arxiv.org/abs/2602.10116v1  
  <details><summary>Abstract</summary>

  Real-world data collection for embodied agents remains costly and unsafe, calling for scalable, realistic, and simulator-ready 3D environments. However, existing scene-generation systems often rely on rule-based or task-specific pipelines, yielding artifacts and physically invalid scenes. We present SAGE, an agentic framework that, given a user-specified embodied task (e.g., "pick up a bowl and place it on the table"), understands the intent and automatically generates simulation-ready environments at scale. The agent couples multiple generators for layout and object composition with critics that evaluate semantic plausibility, visual realism, and physical stability. Through iterative reasoning and adaptive tool selection, it self-refines the scenes until meeting user intent and physical validity. The resulting environments are realistic, diverse, and directly deployable in modern simulators for policy training. Policies trained purely on this data exhibit clear scaling trends and generalize to unseen objects and layouts, demonstrating the promise of simulation-driven scaling for embodied AI. Code, demos, and the SAGE-10k dataset can be found on the project page here: https://nvlabs.github.io/sage.

  </details>



- **Quantum Multiple Rotation Averaging**  
  Shuteng Wang, Natacha Kuete Meli, Michael Möller, Vladislav Golyanik  
  _2026-02-10_ · https://arxiv.org/abs/2602.10115v1  
  <details><summary>Abstract</summary>

  Multiple rotation averaging (MRA) is a fundamental optimization problem in 3D vision and robotics that aims to recover globally consistent absolute rotations from noisy relative measurements. Established classical methods, such as L1-IRLS and Shonan, face limitations including local minima susceptibility and reliance on convex relaxations that fail to preserve the exact manifold geometry, leading to reduced accuracy in high-noise scenarios. We introduce IQARS (Iterative Quantum Annealing for Rotation Synchronization), the first algorithm that reformulates MRA as a sequence of local quadratic non-convex sub-problems executable on quantum annealers after binarization, to leverage inherent hardware advantages. IQARS removes convex relaxation dependence and better preserves non-Euclidean rotation manifold geometry while leveraging quantum tunneling and parallelism for efficient solution space exploration. We evaluate IQARS's performance on synthetic and real-world datasets. While current annealers remain in their nascent phase and only support solving problems of limited scale with constrained performance, we observed that IQARS on D-Wave annealers can already achieve ca. 12% higher accuracy than Shonan, i.e., the best-performing classical method evaluated empirically.

  </details>



- **ConsID-Gen: View-Consistent and Identity-Preserving Image-to-Video Generation**  
  Mingyang Wu, Ashirbad Mishra, Soumik Dey, Shuo Xing, Naveen Ravipati, Hansi Wu, Binbin Li, Zhengzhong Tu  
  _2026-02-10_ · https://arxiv.org/abs/2602.10113v1  
  <details><summary>Abstract</summary>

  Image-to-Video generation (I2V) animates a static image into a temporally coherent video sequence following textual instructions, yet preserving fine-grained object identity under changing viewpoints remains a persistent challenge. Unlike text-to-video models, existing I2V pipelines often suffer from appearance drift and geometric distortion, artifacts we attribute to the sparsity of single-view 2D observations and weak cross-modal alignment. Here we address this problem from both data and model perspectives. First, we curate ConsIDVid, a large-scale object-centric dataset built with a scalable pipeline for high-quality, temporally aligned videos, and establish ConsIDVid-Bench, where we present a novel benchmarking and evaluation framework for multi-view consistency using metrics sensitive to subtle geometric and appearance deviations. We further propose ConsID-Gen, a view-assisted I2V generation framework that augments the first frame with unposed auxiliary views and fuses semantic and structural cues via a dual-stream visual-geometric encoder as well as a text-visual connector, yielding unified conditioning for a Diffusion Transformer backbone. Experiments across ConsIDVid-Bench demonstrate that ConsID-Gen consistently outperforms in multiple metrics, with the best overall performance surpassing leading video generation models like Wan2.1 and HunyuanVideo, delivering superior identity fidelity and temporal coherence under challenging real-world scenarios. We will release our model and dataset at https://myangwu.github.io/ConsID-Gen.

  </details>



- **VideoWorld 2: Learning Transferable Knowledge from Real-world Videos**  
  Zhongwei Ren, Yunchao Wei, Xiao Yu, Guixun Luo, Yao Zhao, Bingyi Kang, Jiashi Feng, Xiaojie Jin  
  _2026-02-10_ · https://arxiv.org/abs/2602.10102v1  
  <details><summary>Abstract</summary>

  Learning transferable knowledge from unlabeled video data and applying it in new environments is a fundamental capability of intelligent agents. This work presents VideoWorld 2, which extends VideoWorld and offers the first investigation into learning transferable knowledge directly from raw real-world videos. At its core, VideoWorld 2 introduces a dynamic-enhanced Latent Dynamics Model (dLDM) that decouples action dynamics from visual appearance: a pretrained video diffusion model handles visual appearance modeling, enabling the dLDM to learn latent codes that focus on compact and meaningful task-related dynamics. These latent codes are then modeled autoregressively to learn task policies and support long-horizon reasoning. We evaluate VideoWorld 2 on challenging real-world handcraft making tasks, where prior video generation and latent-dynamics models struggle to operate reliably. Remarkably, VideoWorld 2 achieves up to 70% improvement in task success rate and produces coherent long execution videos. In robotics, we show that VideoWorld 2 can acquire effective manipulation knowledge from the Open-X dataset, which substantially improves task performance on CALVIN. This study reveals the potential of learning transferable world knowledge directly from raw videos, with all code, data, and models to be open-sourced for further research.

  </details>



- **Robo3R: Enhancing Robotic Manipulation with Accurate Feed-Forward 3D Reconstruction**  
  Sizhe Yang, Linning Xu, Hao Li, Juncheng Mu, Jia Zeng, Dahua Lin, Jiangmiao Pang  
  _2026-02-10_ · https://arxiv.org/abs/2602.10101v1  
  <details><summary>Abstract</summary>

  3D spatial perception is fundamental to generalizable robotic manipulation, yet obtaining reliable, high-quality 3D geometry remains challenging. Depth sensors suffer from noise and material sensitivity, while existing reconstruction models lack the precision and metric consistency required for physical interaction. We introduce Robo3R, a feed-forward, manipulation-ready 3D reconstruction model that predicts accurate, metric-scale scene geometry directly from RGB images and robot states in real time. Robo3R jointly infers scale-invariant local geometry and relative camera poses, which are unified into the scene representation in the canonical robot frame via a learned global similarity transformation. To meet the precision demands of manipulation, Robo3R employs a masked point head for sharp, fine-grained point clouds, and a keypoint-based Perspective-n-Point (PnP) formulation to refine camera extrinsics and global alignment. Trained on Robo3R-4M, a curated large-scale synthetic dataset with four million high-fidelity annotated frames, Robo3R consistently outperforms state-of-the-art reconstruction methods and depth sensors. Across downstream tasks including imitation learning, sim-to-real transfer, grasp synthesis, and collision-free motion planning, we observe consistent gains in performance, suggesting the promise of this alternative 3D sensing module for robotic manipulation.

  </details>



- **UniVTAC: A Unified Simulation Platform for Visuo-Tactile Manipulation Data Generation, Learning, and Benchmarking**  
  Baijun Chen, Weijie Wan, Tianxing Chen, Xianda Guo, Congsheng Xu, Yuanyang Qi, Haojie Zhang, Longyan Wu, Tianling Xu, Zixuan Li, et al.  
  _2026-02-10_ · https://arxiv.org/abs/2602.10093v1  
  <details><summary>Abstract</summary>

  Robotic manipulation has seen rapid progress with vision-language-action (VLA) policies. However, visuo-tactile perception is critical for contact-rich manipulation, as tasks such as insertion are difficult to complete robustly using vision alone. At the same time, acquiring large-scale and reliable tactile data in the physical world remains costly and challenging, and the lack of a unified evaluation platform further limits policy learning and systematic analysis. To address these challenges, we propose UniVTAC, a simulation-based visuo-tactile data synthesis platform that supports three commonly used visuo-tactile sensors and enables scalable and controllable generation of informative contact interactions. Based on this platform, we introduce the UniVTAC Encoder, a visuo-tactile encoder trained on large-scale simulation-synthesized data with designed supervisory signals, providing tactile-centric visuo-tactile representations for downstream manipulation tasks. In addition, we present the UniVTAC Benchmark, which consists of eight representative visuo-tactile manipulation tasks for evaluating tactile-driven policies. Experimental results show that integrating the UniVTAC Encoder improves average success rates by 17.1% on the UniVTAC Benchmark, while real-world robotic experiments further demonstrate a 25% improvement in task success. Our webpage is available at https://univtac.github.io/.

  </details>



- **Can Image Splicing and Copy-Move Forgery Be Detected by the Same Model? Forensim: An Attention-Based State-Space Approach**  
  Soumyaroop Nandi, Prem Natarajan  
  _2026-02-10_ · https://arxiv.org/abs/2602.10079v1  
  <details><summary>Abstract</summary>

  We introduce Forensim, an attention-based state-space framework for image forgery detection that jointly localizes both manipulated (target) and source regions. Unlike traditional approaches that rely solely on artifact cues to detect spliced or forged areas, Forensim is designed to capture duplication patterns crucial for understanding context. In scenarios such as protest imagery, detecting only the forged region, for example a duplicated act of violence inserted into a peaceful crowd, can mislead interpretation, highlighting the need for joint source-target localization. Forensim outputs three-class masks (pristine, source, target) and supports detection of both splicing and copy-move forgeries within a unified architecture. We propose a visual state-space model that leverages normalized attention maps to identify internal similarities, paired with a region-based block attention module to distinguish manipulated regions. This design enables end-to-end training and precise localization. Forensim achieves state-of-the-art performance on standard benchmarks. We also release CMFD-Anything, a new dataset addressing limitations of existing copy-move forgery datasets.

  </details>



- **Vendi Novelty Scores for Out-of-Distribution Detection**  
  Amey P. Pasarkar, Adji Bousso Dieng  
  _2026-02-10_ · https://arxiv.org/abs/2602.10062v1  
  <details><summary>Abstract</summary>

  Out-of-distribution (OOD) detection is critical for the safe deployment of machine learning systems. Existing post-hoc detectors typically rely on model confidence scores or likelihood estimates in feature space, often under restrictive distributional assumptions. In this work, we introduce a third paradigm and formulate OOD detection from a diversity perspective. We propose the Vendi Novelty Score (VNS), an OOD detector based on the Vendi Scores (VS), a family of similarity-based diversity metrics. VNS quantifies how much a test sample increases the VS of the in-distribution feature set, providing a principled notion of novelty that does not require density modeling. VNS is linear-time, non-parametric, and naturally combines class-conditional (local) and dataset-level (global) novelty signals. Across multiple image classification benchmarks and network architectures, VNS achieves state-of-the-art OOD detection performance. Remarkably, VNS retains this performance when computed using only 1% of the training data, enabling deployment in memory- or access-constrained settings.

  </details>



- **RoboSubtaskNet: Temporal Sub-task Segmentation for Human-to-Robot Skill Transfer in Real-World Environments**  
  Dharmendra Sharma, Archit Sharma, John Reberio, Vaibhav Kesharwani, Peeyush Thakur, Narendra Kumar Dhar, Laxmidhar Behera  
  _2026-02-10_ · https://arxiv.org/abs/2602.10015v1  
  <details><summary>Abstract</summary>

  Temporally locating and classifying fine-grained sub-task segments in long, untrimmed videos is crucial to safe human-robot collaboration. Unlike generic activity recognition, collaborative manipulation requires sub-task labels that are directly robot-executable. We present RoboSubtaskNet, a multi-stage human-to-robot sub-task segmentation framework that couples attention-enhanced I3D features (RGB plus optical flow) with a modified MS-TCN employing a Fibonacci dilation schedule to capture better short-horizon transitions such as reach-pick-place. The network is trained with a composite objective comprising cross-entropy and temporal regularizers (truncated MSE and a transition-aware term) to reduce over-segmentation and to encourage valid sub-task progressions. To close the gap between vision benchmarks and control, we introduce RoboSubtask, a dataset of healthcare and industrial demonstrations annotated at the sub-task level and designed for deterministic mapping to manipulator primitives. Empirically, RoboSubtaskNet outperforms MS-TCN and MS-TCN++ on GTEA and our RoboSubtask benchmark (boundary-sensitive and sequence metrics), while remaining competitive on the long-horizon Breakfast benchmark. Specifically, RoboSubtaskNet attains F1 @ 50 = 79.5%, Edit = 88.6%, Acc = 78.9% on GTEA; F1 @ 50 = 30.4%, Edit = 52.0%, Acc = 53.5% on Breakfast; and F1 @ 50 = 94.2%, Edit = 95.6%, Acc = 92.2% on RoboSubtask. We further validate the full perception-to-execution pipeline on a 7-DoF Kinova Gen3 manipulator, achieving reliable end-to-end behavior in physical trials (overall task success approx 91.25%). These results demonstrate a practical path from sub-task level video understanding to deployed robotic manipulation in real-world settings.

  </details>



- **Online Monitoring Framework for Automotive Time Series Data using JEPA Embeddings**  
  Alexander Fertig, Karthikeyan Chandra Sekaran, Lakshman Balasubramanian, Michael Botsch  
  _2026-02-10_ · https://arxiv.org/abs/2602.09985v1  
  <details><summary>Abstract</summary>

  As autonomous vehicles are rolled out, measures must be taken to ensure their safe operation. In order to supervise a system that is already in operation, monitoring frameworks are frequently employed. These run continuously online in the background, supervising the system status and recording anomalies. This work proposes an online monitoring framework to detect anomalies in object state representations. Thereby, a key challenge is creating a framework for anomaly detection without anomaly labels, which are usually unavailable for unknown anomalies. To address this issue, this work applies a self-supervised embedding method to translate object data into a latent representation space. For this, a JEPA-based self-supervised prediction task is constructed, allowing training without anomaly labels and the creation of rich object embeddings. The resulting expressive JEPA embeddings serve as input for established anomaly detection methods, in order to identify anomalies within object state representations. This framework is particularly useful for applications in real-world environments, where new or unknown anomalies may occur during operation for which there are no labels available. Experiments performed on the publicly available, real-world nuScenes dataset illustrate the framework's capabilities.

  </details>



- **Learning to Detect Baked Goods with Limited Supervision**  
  Thomas H. Schmitt, Maximilian Bundscherer, Tobias Bocklet  
  _2026-02-10_ · https://arxiv.org/abs/2602.09979v1  
  <details><summary>Abstract</summary>

  Monitoring leftover products provides valuable insights that can be used to optimize future production. This is especially important for German bakeries because freshly baked goods have a very short shelf life. Automating this process can reduce labor costs, improve accuracy, and streamline operations. We propose automating this process using an object detection model to identify baked goods from images. However, the large diversity of German baked goods makes fully supervised training prohibitively expensive and limits scalability. Although open-vocabulary detectors (e.g., OWLv2, Grounding DINO) offer lexibility, we demonstrate that they are insufficient for our task. While motivated by bakeries, our work addresses the broader challenges of deploying computer vision in industries, where tasks are specialized and annotated datasets are scarce. We compile dataset splits with varying supervision levels, covering 19 classes of baked goods. We propose two training workflows to train an object detection model with limited supervision. First, we combine OWLv2 and Grounding DINO localization with image-level supervision to train the model in a weakly supervised manner. Second, we improve viewpoint robustness by fine-tuning on video frames annotated using Segment Anything 2 as a pseudo-label propagation model. Using these workflows, we train YOLOv11 for our detection task due to its favorable speed accuracy tradeoff. Relying solely on image-level supervision, the model achieves a mean Average Precision (mAP) of 0.91. Finetuning with pseudo-labels raises model performance by 19.3% under non-ideal deployment conditions. Combining these workflows trains a model that surpasses our fully-supervised baseline model under non-ideal deployment conditions, despite relying only on image-level supervision.

  </details>



- **RoboInter: A Holistic Intermediate Representation Suite Towards Robotic Manipulation**  
  Hao Li, Ziqin Wang, Zi-han Ding, Shuai Yang, Yilun Chen, Yang Tian, Xiaolin Hu, Tai Wang, Dahua Lin, Feng Zhao, et al.  
  _2026-02-10_ · https://arxiv.org/abs/2602.09973v1  
  <details><summary>Abstract</summary>

  Advances in large vision-language models (VLMs) have stimulated growing interest in vision-language-action (VLA) systems for robot manipulation. However, existing manipulation datasets remain costly to curate, highly embodiment-specific, and insufficient in coverage and diversity, thereby hindering the generalization of VLA models. Recent approaches attempt to mitigate these limitations via a plan-then-execute paradigm, where high-level plans (e.g., subtasks, trace) are first generated and subsequently translated into low-level actions, but they critically rely on extra intermediate supervision, which is largely absent from existing datasets. To bridge this gap, we introduce the RoboInter Manipulation Suite, a unified resource including data, benchmarks, and models of intermediate representations for manipulation. It comprises RoboInter-Tool, a lightweight GUI that enables semi-automatic annotation of diverse representations, and RoboInter-Data, a large-scale dataset containing over 230k episodes across 571 diverse scenes, which provides dense per-frame annotations over more than 10 categories of intermediate representations, substantially exceeding prior work in scale and annotation quality. Building upon this foundation, RoboInter-VQA introduces 9 spatial and 20 temporal embodied VQA categories to systematically benchmark and enhance the embodied reasoning capabilities of VLMs. Meanwhile, RoboInter-VLA offers an integrated plan-then-execute framework, supporting modular and end-to-end VLA variants that bridge high-level planning with low-level execution via intermediate supervision. In total, RoboInter establishes a practical foundation for advancing robust and generalizable robotic learning via fine-grained and diverse intermediate representations.

  </details>



- **Bladder Vessel Segmentation using a Hybrid Attention-Convolution Framework**  
  Franziska Krauß, Matthias Ege, Zoltan Lovasz, Albrecht Bartz-Schmidt, Igor Tsaur, Oliver Sawodny, Carina Veil  
  _2026-02-10_ · https://arxiv.org/abs/2602.09949v1  
  <details><summary>Abstract</summary>

  Urinary bladder cancer surveillance requires tracking tumor sites across repeated interventions, yet the deformable and hollow bladder lacks stable landmarks for orientation. While blood vessels visible during endoscopy offer a patient-specific "vascular fingerprint" for navigation, automated segmentation is challenged by imperfect endoscopic data, including sparse labels, artifacts like bubbles or variable lighting, continuous deformation, and mucosal folds that mimic vessels. State-of-the-art vessel segmentation methods often fail to address these domain-specific complexities. We introduce a Hybrid Attention-Convolution (HAC) architecture that combines Transformers to capture global vessel topology prior with a CNN that learns a residual refinement map to precisely recover thin-vessel details. To prioritize structural connectivity, the Transformer is trained on optimized ground truth data that exclude short and terminal branches. Furthermore, to address data scarcity, we employ a physics-aware pretraining, that is a self-supervised strategy using clinically grounded augmentations on unlabeled data. Evaluated on the BlaVeS dataset, consisting of endoscopic video frames, our approach achieves high accuracy (0.94) and superior precision (0.61) and clDice (0.66) compared to state-of-the-art medical segmentation models. Crucially, our method successfully suppresses false positives from mucosal folds that dynamically appear and vanish as the bladder fills and empties during surgery. Hence, HAC provides the reliable structural stability required for clinical navigation.

  </details>



- **Instruct2Act: From Human Instruction to Actions Sequencing and Execution via Robot Action Network for Robotic Manipulation**  
  Archit Sharma, Dharmendra Sharma, John Rebeiro, Peeyush Thakur, Narendra Dhar, Laxmidhar Behera  
  _2026-02-10_ · https://arxiv.org/abs/2602.09940v1  
  <details><summary>Abstract</summary>

  Robots often struggle to follow free-form human instructions in real-world settings due to computational and sensing limitations. We address this gap with a lightweight, fully on-device pipeline that converts natural-language commands into reliable manipulation. Our approach has two stages: (i) the instruction to actions module (Instruct2Act), a compact BiLSTM with a multi-head-attention autoencoder that parses an instruction into an ordered sequence of atomic actions (e.g., reach, grasp, move, place); and (ii) the robot action network (RAN), which uses the dynamic adaptive trajectory radial network (DATRN) together with a vision-based environment analyzer (YOLOv8) to generate precise control trajectories for each sub-action. The entire system runs on a modest system with no cloud services. On our custom proprietary dataset, Instruct2Act attains 91.5% sub-actions prediction accuracy while retaining a small footprint. Real-robot evaluations across four tasks (pick-place, pick-pour, wipe, and pick-give) yield an overall 90% success; sub-action inference completes in < 3.8s, with end-to-end executions in 30-60s depending on task complexity. These results demonstrate that fine-grained instruction-to-action parsing, coupled with DATRN-based trajectory generation and vision-guided grounding, provides a practical path to deterministic, real-time manipulation in resource-constrained, single-camera settings.

  </details>



- **Monocular Normal Estimation via Shading Sequence Estimation**  
  Zongrui Li, Xinhua Ma, Minghui Hu, Yunqing Zhao, Yingchen Yu, Qian Zheng, Chang Liu, Xudong Jiang, Song Bai  
  _2026-02-10_ · https://arxiv.org/abs/2602.09929v1  
  <details><summary>Abstract</summary>

  Monocular normal estimation aims to estimate the normal map from a single RGB image of an object under arbitrary lights. Existing methods rely on deep models to directly predict normal maps. However, they often suffer from 3D misalignment: while the estimated normal maps may appear to have a correct appearance, the reconstructed surfaces often fail to align with the geometric details. We argue that this misalignment stems from the current paradigm: the model struggles to distinguish and reconstruct varying geometry represented in normal maps, as the differences in underlying geometry are reflected only through relatively subtle color variations. To address this issue, we propose a new paradigm that reformulates normal estimation as shading sequence estimation, where shading sequences are more sensitive to various geometric information. Building on this paradigm, we present RoSE, a method that leverages image-to-video generative models to predict shading sequences. The predicted shading sequences are then converted into normal maps by solving a simple ordinary least-squares problem. To enhance robustness and better handle complex objects, RoSE is trained on a synthetic dataset, MultiShade, with diverse shapes, materials, and light conditions. Experiments demonstrate that RoSE achieves state-of-the-art performance on real-world benchmark datasets for object-based monocular normal estimation.

  </details>



- **A benchmark for video-based laparoscopic skill analysis and assessment**  
  Isabel Funke, Sebastian Bodenstedt, Felix von Bechtolsheim, Florian Oehme, Michael Maruschke, Stefanie Herrlich, Jürgen Weitz, Marius Distler, Sören Torge Mees, Stefanie Speidel  
  _2026-02-10_ · https://arxiv.org/abs/2602.09927v1  
  <details><summary>Abstract</summary>

  Laparoscopic surgery is a complex surgical technique that requires extensive training. Recent advances in deep learning have shown promise in supporting this training by enabling automatic video-based assessment of surgical skills. However, the development and evaluation of deep learning models is currently hindered by the limited size of available annotated datasets. To address this gap, we introduce the Laparoscopic Skill Analysis and Assessment (LASANA) dataset, comprising 1270 stereo video recordings of four basic laparoscopic training tasks. Each recording is annotated with a structured skill rating, aggregated from three independent raters, as well as binary labels indicating the presence or absence of task-specific errors. The majority of recordings originate from a laparoscopic training course, thereby reflecting a natural variation in the skill of participants. To facilitate benchmarking of both existing and novel approaches for video-based skill assessment and error recognition, we provide predefined data splits for each task. Furthermore, we present baseline results from a deep learning model as a reference point for future comparisons.

  </details>



- **TaCo: A Benchmark for Lossless and Lossy Codecs of Heterogeneous Tactile Data**  
  Zhengxue Cheng, Yan Zhao, Keyu Wang, Hengdi Zhang, Li Song  
  _2026-02-10_ · https://arxiv.org/abs/2602.09893v1  
  <details><summary>Abstract</summary>

  Tactile sensing is crucial for embodied intelligence, providing fine-grained perception and control in complex environments. However, efficient tactile data compression, which is essential for real-time robotic applications under strict bandwidth constraints, remains underexplored. The inherent heterogeneity and spatiotemporal complexity of tactile data further complicate this challenge. To bridge this gap, we introduce TaCo, the first comprehensive benchmark for Tactile data Codecs. TaCo evaluates 30 compression methods, including off-the-shelf compression algorithms and neural codecs, across five diverse datasets from various sensor types. We systematically assess both lossless and lossy compression schemes on four key tasks: lossless storage, human visualization, material and object classification, and dexterous robotic grasping. Notably, we pioneer the development of data-driven codecs explicitly trained on tactile data, TaCo-LL (lossless) and TaCo-L (lossy). Results have validated the superior performance of our TaCo-LL and TaCo-L. This benchmark provides a foundational framework for understanding the critical trade-offs between compression efficiency and task performance, paving the way for future advances in tactile perception.

  </details>



- **ARK: A Dual-Axis Multimodal Retrieval Benchmark along Reasoning and Knowledge**  
  Yijie Lin, Guofeng Ding, Haochen Zhou, Haobin Li, Mouxing Yang, Xi Peng  
  _2026-02-10_ · https://arxiv.org/abs/2602.09839v1  
  <details><summary>Abstract</summary>

  Existing multimodal retrieval benchmarks largely emphasize semantic matching on daily-life images and offer limited diagnostics of professional knowledge and complex reasoning. To address this gap, we introduce ARK, a benchmark designed to analyze multimodal retrieval from two complementary perspectives: (i) knowledge domains (five domains with 17 subtypes), which characterize the content and expertise retrieval relies on, and (ii) reasoning skills (six categories), which characterize the type of inference over multimodal evidence required to identify the correct candidate. Specifically, ARK evaluates retrieval with both unimodal and multimodal queries and candidates, covering 16 heterogeneous visual data types. To avoid shortcut matching during evaluation, most queries are paired with targeted hard negatives that require multi-step reasoning. We evaluate 23 representative text-based and multimodal retrievers on ARK and observe a pronounced gap between knowledge-intensive and reasoning-intensive retrieval, with fine-grained visual and spatial reasoning emerging as persistent bottlenecks. We further show that simple enhancements such as re-ranking and rewriting yield consistent improvements, but substantial headroom remains.

  </details>



- **SciFlow-Bench: Evaluating Structure-Aware Scientific Diagram Generation via Inverse Parsing**  
  Tong Zhang, Honglin Lin, Zhou Liu, Chong Chen, Wentao Zhang  
  _2026-02-10_ · https://arxiv.org/abs/2602.09809v1  
  <details><summary>Abstract</summary>

  Scientific diagrams convey explicit structural information, yet modern text-to-image models often produce visually plausible but structurally incorrect results. Existing benchmarks either rely on image-centric or subjective metrics insensitive to structure, or evaluate intermediate symbolic representations rather than final rendered images, leaving pixel-based diagram generation underexplored. We introduce SciFlow-Bench, a structure-first benchmark for evaluating scientific diagram generation directly from pixel-level outputs. Built from real scientific PDFs, SciFlow-Bench pairs each source framework figure with a canonical ground-truth graph and evaluates models as black-box image generators under a closed-loop, round-trip protocol that inverse-parses generated diagram images back into structured graphs for comparison. This design enforces evaluation by structural recoverability rather than visual similarity alone, and is enabled by a hierarchical multi-agent system that coordinates planning, perception, and structural reasoning. Experiments show that preserving structural correctness remains a fundamental challenge, particularly for diagrams with complex topology, underscoring the need for structure-aware evaluation.

  </details>



- **Where Do Images Come From? Analyzing Captions to Geographically Profile Datasets**  
  Abhipsa Basu, Yugam Bahl, Kirti Bhagat, Preethi Seshadri, R. Venkatesh Babu, Danish Pruthi  
  _2026-02-10_ · https://arxiv.org/abs/2602.09775v1  
  <details><summary>Abstract</summary>

  Recent studies show that text-to-image models often fail to generate geographically representative images, raising concerns about the representativeness of their training data and motivating the question: which parts of the world do these training examples come from? We geographically profile large-scale multimodal datasets by mapping image-caption pairs to countries based on location information extracted from captions using LLMs. Studying English captions from three widely used datasets (Re-LAION, DataComp1B, and Conceptual Captions) across $20$ common entities (e.g., house, flag), we find that the United States, the United Kingdom, and Canada account for $48.0\%$ of samples, while South American and African countries are severely under-represented with only $1.8\%$ and $3.8\%$ of images, respectively. We observe a strong correlation between a country's GDP and its representation in the data ($ρ= 0.82$). Examining non-English subsets for $4$ languages from the Re-LAION dataset, we find that representation skews heavily toward countries where these languages are predominantly spoken. Additionally, we find that higher representation does not necessarily translate to greater visual or semantic diversity. Finally, analyzing country-specific images generated by Stable Diffusion v1.3 trained on Re-LAION, we show that while generations appear realistic, they are severely limited in their coverage compared to real-world images.

  </details>



- **NavDreamer: Video Models as Zero-Shot 3D Navigators**  
  Xijie Huang, Weiqi Gai, Tianyue Wu, Congyu Wang, Zhiyang Liu, Xin Zhou, Yuze Wu, Fei Gao  
  _2026-02-10_ · https://arxiv.org/abs/2602.09765v1  
  <details><summary>Abstract</summary>

  Previous Vision-Language-Action models face critical limitations in navigation: scarce, diverse data from labor-intensive collection and static representations that fail to capture temporal dynamics and physical laws. We propose NavDreamer, a video-based framework for 3D navigation that leverages generative video models as a universal interface between language instructions and navigation trajectories. Our main hypothesis is that video's ability to encode spatiotemporal information and physical dynamics, combined with internet-scale availability, enables strong zero-shot generalization in navigation. To mitigate the stochasticity of generative predictions, we introduce a sampling-based optimization method that utilizes a VLM for trajectory scoring and selection. An inverse dynamics model is employed to decode executable waypoints from generated video plans for navigation. To systematically evaluate this paradigm in several video model backbones, we introduce a comprehensive benchmark covering object navigation, precise navigation, spatial grounding, language control, and scene reasoning. Extensive experiments demonstrate robust generalization across novel objects and unseen environments, with ablation studies revealing that navigation's high-level decision-making nature makes it particularly suited for video-based planning.

  </details>



- **From Lightweight CNNs to SpikeNets: Benchmarking Accuracy-Energy Tradeoffs with Pruned Spiking SqueezeNet**  
  Radib Bin Kabir, Tawsif Tashwar Dipto, Mehedi Ahamed, Sabbir Ahmed, Md Hasanul Kabir  
  _2026-02-10_ · https://arxiv.org/abs/2602.09717v1  
  <details><summary>Abstract</summary>

  Spiking Neural Networks (SNNs) are increasingly studied as energy-efficient alternatives to Convolutional Neural Networks (CNNs), particularly for edge intelligence. However, prior work has largely emphasized large-scale models, leaving the design and evaluation of lightweight CNN-to-SNN pipelines underexplored. In this paper, we present the first systematic benchmark of lightweight SNNs obtained by converting compact CNN architectures into spiking networks, where activations are modeled with Leaky-Integrate-and-Fire (LIF) neurons and trained using surrogate gradient descent under a unified setup. We construct spiking variants of ShuffleNet, SqueezeNet, MnasNet, and MixNet, and evaluate them on CIFAR-10, CIFAR-100, and TinyImageNet, measuring accuracy, F1-score, parameter count, computational complexity, and energy consumption. Our results show that SNNs can achieve up to 15.7x higher energy efficiency than their CNN counterparts while retaining competitive accuracy. Among these, the SNN variant of SqueezeNet consistently outperforms other lightweight SNNs. To further optimize this model, we apply a structured pruning strategy that removes entire redundant modules, yielding a pruned architecture, SNN-SqueezeNet-P. This pruned model improves CIFAR-10 accuracy by 6% and reduces parameters by 19% compared to the original SNN-SqueezeNet. Crucially, it narrows the gap with CNN-SqueezeNet, achieving nearly the same accuracy (only 1% lower) but with an 88.1% reduction in energy consumption due to sparse spike-driven computations. Together, these findings establish lightweight SNNs as practical, low-power alternatives for edge deployment, highlighting a viable path toward deploying high-performance, low-power intelligence on the edge.

  </details>



- **Stroke3D: Lifting 2D strokes into rigged 3D model via latent diffusion models**  
  Ruisi Zhao, Haoren Zheng, Zongxin Yang, Hehe Fan, Yi Yang  
  _2026-02-10_ · https://arxiv.org/abs/2602.09713v1  
  <details><summary>Abstract</summary>

  Rigged 3D assets are fundamental to 3D deformation and animation. However, existing 3D generation methods face challenges in generating animatable geometry, while rigging techniques lack fine-grained structural control over skeleton creation. To address these limitations, we introduce Stroke3D, a novel framework that directly generates rigged meshes from user inputs: 2D drawn strokes and a descriptive text prompt. Our approach pioneers a two-stage pipeline that separates the generation into: 1) Controllable Skeleton Generation, we employ the Skeletal Graph VAE (Sk-VAE) to encode the skeleton's graph structure into a latent space, where the Skeletal Graph DiT (Sk-DiT) generates a skeletal embedding. The generation process is conditioned on both the text for semantics and the 2D strokes for explicit structural control, with the VAE's decoder reconstructing the final high-quality 3D skeleton; and 2) Enhanced Mesh Synthesis via TextuRig and SKA-DPO, where we then synthesize a textured mesh conditioned on the generated skeleton. For this stage, we first enhance an existing skeleton-to-mesh model by augmenting its training data with TextuRig: a dataset of textured and rigged meshes with captions, curated from Objaverse-XL. Additionally, we employ a preference optimization strategy, SKA-DPO, guided by a skeleton-mesh alignment score, to further improve geometric fidelity. Together, our framework enables a more intuitive workflow for creating ready to animate 3D content. To the best of our knowledge, our work is the first to generate rigged 3D meshes conditioned on user-drawn 2D strokes. Extensive experiments demonstrate that Stroke3D produces plausible skeletons and high-quality meshes.

  </details>



- **AutoFly: Vision-Language-Action Model for UAV Autonomous Navigation in the Wild**  
  Xiaolou Sun, Wufei Si, Wenhui Ni, Yuntian Li, Dongming Wu, Fei Xie, Runwei Guan, He-Yang Xu, Henghui Ding, Yuan Wu, et al.  
  _2026-02-10_ · https://arxiv.org/abs/2602.09657v1  
  <details><summary>Abstract</summary>

  Vision-language navigation (VLN) requires intelligent agents to navigate environments by interpreting linguistic instructions alongside visual observations, serving as a cornerstone task in Embodied AI. Current VLN research for unmanned aerial vehicles (UAVs) relies on detailed, pre-specified instructions to guide the UAV along predetermined routes. However, real-world outdoor exploration typically occurs in unknown environments where detailed navigation instructions are unavailable. Instead, only coarse-grained positional or directional guidance can be provided, requiring UAVs to autonomously navigate through continuous planning and obstacle avoidance. To bridge this gap, we propose AutoFly, an end-to-end Vision-Language-Action (VLA) model for autonomous UAV navigation. AutoFly incorporates a pseudo-depth encoder that derives depth-aware features from RGB inputs to enhance spatial reasoning, coupled with a progressive two-stage training strategy that effectively aligns visual, depth, and linguistic representations with action policies. Moreover, existing VLN datasets have fundamental limitations for real-world autonomous navigation, stemming from their heavy reliance on explicit instruction-following over autonomous decision-making and insufficient real-world data. To address these issues, we construct a novel autonomous navigation dataset that shifts the paradigm from instruction-following to autonomous behavior modeling through: (1) trajectory collection emphasizing continuous obstacle avoidance, autonomous planning, and recognition workflows; (2) comprehensive real-world data integration. Experimental results demonstrate that AutoFly achieves a 3.9% higher success rate compared to state-of-the-art VLA baselines, with consistent performance across simulated and real environments.

  </details>



- **VideoAfford: Grounding 3D Affordance from Human-Object-Interaction Videos via Multimodal Large Language Model**  
  Hanqing Wang, Mingyu Liu, Xiaoyu Chen, Chengwei MA, Yiming Zhong, Wenti Yin, Yuhao Liu, Zhiqing Cui, Jiahao Yuan, Lu Dai, et al.  
  _2026-02-10_ · https://arxiv.org/abs/2602.09638v1  
  <details><summary>Abstract</summary>

  3D affordance grounding aims to highlight the actionable regions on 3D objects, which is crucial for robotic manipulation. Previous research primarily focused on learning affordance knowledge from static cues such as language and images, which struggle to provide sufficient dynamic interaction context that can reveal temporal and causal cues. To alleviate this predicament, we collect a comprehensive video-based 3D affordance dataset, \textit{VIDA}, which contains 38K human-object-interaction videos covering 16 affordance types, 38 object categories, and 22K point clouds. Based on \textit{VIDA}, we propose a strong baseline: VideoAfford, which activates multimodal large language models with additional affordance segmentation capabilities, enabling both world knowledge reasoning and fine-grained affordance grounding within a unified framework. To enhance action understanding capability, we leverage a latent action encoder to extract dynamic interaction priors from HOI videos. Moreover, we introduce a \textit{spatial-aware} loss function to enable VideoAfford to obtain comprehensive 3D spatial knowledge. Extensive experimental evaluations demonstrate that our model significantly outperforms well-established methods and exhibits strong open-world generalization with affordance reasoning abilities. All datasets and code will be publicly released to advance research in this area.

  </details>



- **AnyTouch 2: General Optical Tactile Representation Learning For Dynamic Tactile Perception**  
  Ruoxuan Feng, Yuxuan Zhou, Siyu Mei, Dongzhan Zhou, Pengwei Wang, Shaowei Cui, Bin Fang, Guocai Yao, Di Hu  
  _2026-02-10_ · https://arxiv.org/abs/2602.09617v1  
  <details><summary>Abstract</summary>

  Real-world contact-rich manipulation demands robots to perceive temporal tactile feedback, capture subtle surface deformations, and reason about object properties as well as force dynamics. Although optical tactile sensors are uniquely capable of providing such rich information, existing tactile datasets and models remain limited. These resources primarily focus on object-level attributes (e.g., material) while largely overlooking fine-grained tactile temporal dynamics during physical interactions. We consider that advancing dynamic tactile perception requires a systematic hierarchy of dynamic perception capabilities to guide both data collection and model design. To address the lack of tactile data with rich dynamic information, we present ToucHD, a large-scale hierarchical tactile dataset spanning tactile atomic actions, real-world manipulations, and touch-force paired data. Beyond scale, ToucHD establishes a comprehensive tactile dynamic data ecosystem that explicitly supports hierarchical perception capabilities from the data perspective. Building on it, we propose AnyTouch 2, a general tactile representation learning framework for diverse optical tactile sensors that unifies object-level understanding with fine-grained, force-aware dynamic perception. The framework captures both pixel-level and action-specific deformations across frames, while explicitly modeling physical force dynamics, thereby learning multi-level dynamic perception capabilities from the model perspective. We evaluate our model on benchmarks that covers static object properties and dynamic physical attributes, as well as real-world manipulation tasks spanning multiple tiers of dynamic perception capabilities-from basic object-level understanding to force-aware dexterous manipulation. Experimental results demonstrate consistent and strong performance across sensors and tasks.

  </details>



- **Hand2World: Autoregressive Egocentric Interaction Generation via Free-Space Hand Gestures**  
  Yuxi Wang, Wenqi Ouyang, Tianyi Wei, Yi Dong, Zhiqi Shen, Xingang Pan  
  _2026-02-10_ · https://arxiv.org/abs/2602.09600v1  
  <details><summary>Abstract</summary>

  Egocentric interactive world models are essential for augmented reality and embodied AI, where visual generation must respond to user input with low latency, geometric consistency, and long-term stability. We study egocentric interaction generation from a single scene image under free-space hand gestures, aiming to synthesize photorealistic videos in which hands enter the scene, interact with objects, and induce plausible world dynamics under head motion. This setting introduces fundamental challenges, including distribution shift between free-space gestures and contact-heavy training data, ambiguity between hand motion and camera motion in monocular views, and the need for arbitrary-length video generation. We present Hand2World, a unified autoregressive framework that addresses these challenges through occlusion-invariant hand conditioning based on projected 3D hand meshes, allowing visibility and occlusion to be inferred from scene context rather than encoded in the control signal. To stabilize egocentric viewpoint changes, we inject explicit camera geometry via per-pixel Plücker-ray embeddings, disentangling camera motion from hand motion and preventing background drift. We further develop a fully automated monocular annotation pipeline and distill a bidirectional diffusion model into a causal generator, enabling arbitrary-length synthesis. Experiments on three egocentric interaction benchmarks show substantial improvements in perceptual quality and 3D consistency while supporting camera control and long-horizon interactive generation.

  </details>



- **MieDB-100k: A Comprehensive Dataset for Medical Image Editing**  
  Yongfan Lai, Wen Qian, Bo Liu, Hongyan Li, Hao Luo, Fan Wang, Bohan Zhuang, Shenda Hong  
  _2026-02-10_ · https://arxiv.org/abs/2602.09587v1  
  <details><summary>Abstract</summary>

  The scarcity of high-quality data remains a primary bottleneck in adapting multimodal generative models for medical image editing. Existing medical image editing datasets often suffer from limited diversity, neglect of medical image understanding and inability to balance quality with scalability. To address these gaps, we propose MieDB-100k, a large-scale, high-quality and diverse dataset for text-guided medical image editing. It categorizes editing tasks into perspectives of Perception, Modification and Transformation, considering both understanding and generation abilities. We construct MieDB-100k via a data curation pipeline leveraging both modality-specific expert models and rule-based data synthetic methods, followed by rigorous manual inspection to ensure clinical fidelity. Extensive experiments demonstrate that model trained with MieDB-100k consistently outperform both open-source and proprietary models while exhibiting strong generalization ability. We anticipate that this dataset will serve as a cornerstone for future advancements in specialized medical image editing.

  </details>



- **ECG-IMN: Interpretable Mesomorphic Neural Networks for 12-Lead Electrocardiogram Interpretation**  
  Vajira Thambawita, Jonas L. Isaksen, Jørgen K. Kanters, Hugo L. Hammer, Pål Halvorsen  
  _2026-02-10_ · https://arxiv.org/abs/2602.09566v1  
  <details><summary>Abstract</summary>

  Deep learning has achieved expert-level performance in automated electrocardiogram (ECG) diagnosis, yet the "black-box" nature of these models hinders their clinical deployment. Trust in medical AI requires not just high accuracy but also transparency regarding the specific physiological features driving predictions. Existing explainability methods for ECGs typically rely on post-hoc approximations (e.g., Grad-CAM and SHAP), which can be unstable, computationally expensive, and unfaithful to the model's actual decision-making process. In this work, we propose the ECG-IMN, an Interpretable Mesomorphic Neural Network tailored for high-resolution 12-lead ECG classification. Unlike standard classifiers, the ECG-IMN functions as a hypernetwork: a deep convolutional backbone generates the parameters of a strictly linear model specific to each input sample. This architecture enforces intrinsic interpretability, as the decision logic is mathematically transparent and the generated weights (W) serve as exact, high-resolution feature attribution maps. We introduce a transition decoder that effectively maps latent features to sample-wise weights, enabling precise localization of pathological evidence (e.g., ST-elevation, T-wave inversion) in both time and lead dimensions. We evaluate our approach on the PTB-XL dataset for classification tasks, demonstrating that the ECG-IMN achieves competitive predictive performance (AUROC comparable to black-box baselines) while providing faithful, instance-specific explanations. By explicitly decoupling parameter generation from prediction execution, our framework bridges the gap between deep learning capability and clinical trustworthiness, offering a principled path toward "white-box" cardiac diagnostics.

  </details>



- **AUHead: Realistic Emotional Talking Head Generation via Action Units Control**  
  Jiayi Lyu, Leigang Qu, Wenjing Zhang, Hanyu Jiang, Kai Liu, Zhenglin Zhou, Xiaobo Xia, Jian Xue, Tat-Seng Chua  
  _2026-02-10_ · https://arxiv.org/abs/2602.09534v1  
  <details><summary>Abstract</summary>

  Realistic talking-head video generation is critical for virtual avatars, film production, and interactive systems. Current methods struggle with nuanced emotional expressions due to the lack of fine-grained emotion control. To address this issue, we introduce a novel two-stage method (AUHead) to disentangle fine-grained emotion control, i.e. , Action Units (AUs), from audio and achieve controllable generation. In the first stage, we explore the AU generation abilities of large audio-language models (ALMs), by spatial-temporal AU tokenization and an "emotion-then-AU" chain-of-thought mechanism. It aims to disentangle AUs from raw speech, effectively capturing subtle emotional cues. In the second stage, we propose an AU-driven controllable diffusion model that synthesizes realistic talking-head videos conditioned on AU sequences. Specifically, we first map the AU sequences into the structured 2D facial representation to enhance spatial fidelity, and then model the AU-vision interaction within cross-attention modules. To achieve flexible AU-quality trade-off control, we introduce an AU disentanglement guidance strategy during inference, further refining the emotional expressiveness and identity consistency of the generated videos. Results on benchmark datasets demonstrate that our approach achieves competitive performance in emotional realism, accurate lip synchronization, and visual coherence, significantly surpassing existing techniques. Our implementation is available at https://github.com/laura990501/AUHead_ICLR

  </details>



- **HLGFA: High-Low Resolution Guided Feature Alignment for Unsupervised Anomaly Detection**  
  Han Zhou, Yuxuan Gao, Yinchao Du, Xuezhe Zheng  
  _2026-02-10_ · https://arxiv.org/abs/2602.09524v1  
  <details><summary>Abstract</summary>

  Unsupervised industrial anomaly detection (UAD) is essential for modern manufacturing inspection, where defect samples are scarce and reliable detection is required. In this paper, we propose HLGFA, a high-low resolution guided feature alignment framework that learns normality by modeling cross-resolution feature consistency between high-resolution and low-resolution representations of normal samples, instead of relying on pixel-level reconstruction. Dual-resolution inputs are processed by a shared frozen backbone to extract multi-level features, and high-resolution representations are decomposed into structure and detail priors to guide the refinement of low-resolution features through conditional modulation and gated residual correction. During inference, anomalies are naturally identified as regions where cross-resolution alignment breaks down. In addition, a noise-aware data augmentation strategy is introduced to suppress nuisance-induced responses commonly observed in industrial environments. Extensive experiments on standard benchmarks demonstrate the effectiveness of HLGFA, achieving 97.9% pixel-level AUROC and 97.5% image-level AUROC on the MVTec AD dataset, outperforming representative reconstruction-based and feature-based methods.

  </details>



- **Singpath-VL Technical Report**  
  Zhen Qiu, Kaiwen Xiao, Zhengwei Lu, Xiangyu Liu, Lei Zhao, Hao Zhang  
  _2026-02-10_ · https://arxiv.org/abs/2602.09523v1  
  <details><summary>Abstract</summary>

  We present Singpath-VL, a vision-language large model, to fill the vacancy of AI assistant in cervical cytology. Recent advances in multi-modal large language models (MLLMs) have significantly propelled the field of computational pathology. However, their application in cytopathology, particularly cervical cytology, remains underexplored, primarily due to the scarcity of large-scale, high-quality annotated datasets. To bridge this gap, we first develop a novel three-stage pipeline to synthesize a million-scale image-description dataset. The pipeline leverages multiple general-purpose MLLMs as weak annotators, refines their outputs through consensus fusion and expert knowledge injection, and produces high-fidelity descriptions of cell morphology. Using this dataset, we then fine-tune the Qwen3-VL-4B model via a multi-stage strategy to create a specialized cytopathology MLLM. The resulting model, named Singpath-VL, demonstrates superior performance in fine-grained morphological perception and cell-level diagnostic classification. To advance the field, we will open-source a portion of the synthetic dataset and benchmark.

  </details>



- **Equilibrium contrastive learning for imbalanced image classification**  
  Sumin Roh, Harim Kim, Ho Yun Lee, Il Yong Chun  
  _2026-02-10_ · https://arxiv.org/abs/2602.09506v1  
  <details><summary>Abstract</summary>

  Contrastive learning (CL) is a predominant technique in image classification, but they showed limited performance with an imbalanced dataset. Recently, several supervised CL methods have been proposed to promote an ideal regular simplex geometric configuration in the representation space-characterized by intra-class feature collapse and uniform inter-class mean spacing, especially for imbalanced datasets. In particular, existing prototype-based methods include class prototypes, as additional samples to consider all classes. However, the existing CL methods suffer from two limitations. First, they do not consider the alignment between the class means/prototypes and classifiers, which could lead to poor generalization. Second, existing prototype-based methods treat prototypes as only one additional sample per class, making their influence depend on the number of class instances in a batch and causing unbalanced contributions across classes. To address these limitations, we propose Equilibrium Contrastive Learning (ECL), a supervised CL framework designed to promote geometric equilibrium, where class features, means, and classifiers are harmoniously balanced under data imbalance. The proposed ECL framework uses two main components. First, ECL promotes the representation geometric equilibrium (i.e., a regular simplex geometry characterized by collapsed class samples and uniformly distributed class means), while balancing the contributions of class-average features and class prototypes. Second, ECL establishes a classifier-class center geometric equilibrium by aligning classifier weights and class prototypes. We ran experiments with three long-tailed datasets, the CIFAR-10(0)-LT, ImageNet-LT, and the two imbalanced medical datasets, the ISIC 2019 and our constructed LCCT dataset. Results show that ECL outperforms existing SOTA supervised CL methods designed for imbalanced classification.

  </details>



- **FD-DB: Frequency-Decoupled Dual-Branch Network for Unpaired Synthetic-to-Real Domain Translation**  
  Chuanhai Zang, Jiabao Hu, XW Song  
  _2026-02-10_ · https://arxiv.org/abs/2602.09476v1  
  <details><summary>Abstract</summary>

  Synthetic data provide low-cost, accurately annotated samples for geometry-sensitive vision tasks, but appearance and imaging differences between synthetic and real domains cause severe domain shift and degrade downstream performance. Unpaired synthetic-to-real translation can reduce this gap without paired supervision, yet existing methods often face a trade-off between photorealism and structural stability: unconstrained generation may introduce deformation or spurious textures, while overly rigid constraints limit adaptation to real-domain statistics. We propose FD-DB, a frequency-decoupled dual-branch model that separates appearance transfer into low-frequency interpretable editing and high-frequency residual compensation. The interpretable branch predicts physically meaningful editing parameters (white balance, exposure, contrast, saturation, blur, and grain) to build a stable low-frequency appearance base with strong content preservation. The free branch complements fine details through residual generation, and a gated fusion mechanism combines the two branches under explicit frequency constraints to limit low-frequency drift. We further adopt a two-stage training schedule that first stabilizes the editing branch and then releases the residual branch to improve optimization stability. Experiments on the YCB-V dataset show that FD-DB improves real-domain appearance consistency and significantly boosts downstream semantic segmentation performance while preserving geometric and semantic structures.

  </details>



- **ArtifactLens: Hundreds of Labels Are Enough for Artifact Detection with VLMs**  
  James Burgess, Rameen Abdal, Dan Stoddart, Sergey Tulyakov, Serena Yeung-Levy, Kuan-Chieh Jackson Wang  
  _2026-02-10_ · https://arxiv.org/abs/2602.09475v1  
  <details><summary>Abstract</summary>

  Modern image generators produce strikingly realistic images, where only artifacts like distorted hands or warped objects reveal their synthetic origin. Detecting these artifacts is essential: without detection, we cannot benchmark generators or train reward models to improve them. Current detectors fine-tune VLMs on tens of thousands of labeled images, but this is expensive to repeat whenever generators evolve or new artifact types emerge. We show that pretrained VLMs already encode the knowledge needed to detect artifacts - with the right scaffolding, this capability can be unlocked using only a few hundred labeled examples per artifact category. Our system, ArtifactLens, achieves state-of-the-art on five human artifact benchmarks (the first evaluation across multiple datasets) while requiring orders of magnitude less labeled data. The scaffolding consists of a multi-component architecture with in-context learning and text instruction optimization, with novel improvements to each. Our methods generalize to other artifact types - object morphology, animal anatomy, and entity interactions - and to the distinct task of AIGC detection.

  </details>



- **A Scoping Review of Deep Learning for Urban Visual Pollution and Proposal of a Real-Time Monitoring Framework with a Visual Pollution Index**  
  Mohammad Masudur Rahman, Md. Rashedur Rahman, Ashraful Islam, Saadia B Alam, M Ashraful Amin  
  _2026-02-10_ · https://arxiv.org/abs/2602.09446v1  
  <details><summary>Abstract</summary>

  Urban Visual Pollution (UVP) has emerged as a critical concern, yet research on automatic detection and application remains fragmented. This scoping review maps the existing deep learning-based approaches for detecting, classifying, and designing a comprehensive application framework for visual pollution management. Following the PRISMA-ScR guidelines, seven academic databases (Scopus, Web of Science, IEEE Xplore, ACM DL, ScienceDirect, SpringerNatureLink, and Wiley) were systematically searched and reviewed, and 26 articles were found. Most research focuses on specific pollutant categories and employs variations of YOLO, Faster R-CNN, and EfficientDet architectures. Although several datasets exist, they are limited to specific areas and lack standardized taxonomies. Few studies integrate detection into real-time application systems, yet they tend to be geographically skewed. We proposed a framework for monitoring visual pollution that integrates a visual pollution index to assess the severity of visual pollution for a certain area. This review highlights the need for a unified UVP management system that incorporates pollutant taxonomy, a cross-city benchmark dataset, a generalized deep learning model, and an assessment index that supports sustainable urban aesthetics and enhances the well-being of urban dwellers.

  </details>



- **Fine-T2I: An Open, Large-Scale, and Diverse Dataset for High-Quality T2I Fine-Tuning**  
  Xu Ma, Yitian Zhang, Qihua Dong, Yun Fu  
  _2026-02-10_ · https://arxiv.org/abs/2602.09439v1  
  <details><summary>Abstract</summary>

  High-quality and open datasets remain a major bottleneck for text-to-image (T2I) fine-tuning. Despite rapid progress in model architectures and training pipelines, most publicly available fine-tuning datasets suffer from low resolution, poor text-image alignment, or limited diversity, resulting in a clear performance gap between open research models and enterprise-grade models. In this work, we present Fine-T2I, a large-scale, high-quality, and fully open dataset for T2I fine-tuning. Fine-T2I spans 10 task combinations, 32 prompt categories, 11 visual styles, and 5 prompt templates, and combines synthetic images generated by strong modern models with carefully curated real images from professional photographers. All samples are rigorously filtered for text-image alignment, visual fidelity, and prompt quality, with over 95% of initial candidates removed. The final dataset contains over 6 million text-image pairs, around 2 TB on disk, approaching the scale of pretraining datasets while maintaining fine-tuning-level quality. Across a diverse set of pretrained diffusion and autoregressive models, fine-tuning on Fine-T2I consistently improves both generation quality and instruction adherence, as validated by human evaluation, visual comparison, and automatic metrics. We release Fine-T2I under an open license to help close the data gap in T2I fine-tuning in the open community.

  </details>



- **SceneReVis: A Self-Reflective Vision-Grounded Framework for 3D Indoor Scene Synthesis via Multi-turn RL**  
  Yang Zhao, Shizhao Sun, Meisheng Zhang, Yingdong Shi, Xubo Yang, Jiang Bian  
  _2026-02-10_ · https://arxiv.org/abs/2602.09432v1  
  <details><summary>Abstract</summary>

  Current one-pass 3D scene synthesis methods often suffer from spatial hallucinations, such as collisions, due to a lack of deliberative reasoning. To bridge this gap, we introduce SceneReVis, a vision-grounded self-reflection framework that employs an iterative ``diagnose-and-act'' loop to explicitly intercept and resolve spatial conflicts using multi-modal feedback. To support this step-wise paradigm, we construct SceneChain-12k, a large-scale dataset of causal construction trajectories derived through a novel reverse engineering pipeline. We further propose a two-stage training recipe that transitions from Supervised Fine-Tuning to Agentic Reinforcement Learning, evolving the model into an active spatial planner. Extensive experiments demonstrate that SceneReVis achieves state-of-the-art performance in high-fidelity generation and goal-oriented optimization, with robust generalization to long-tail domains.

  </details>



- **Bridging the Modality Gap in Roadside LiDAR: A Training-Free Vision-Language Model Framework for Vehicle Classification**  
  Yiqiao Li, Bo Shang, Jie Wei  
  _2026-02-10_ · https://arxiv.org/abs/2602.09425v1  
  <details><summary>Abstract</summary>

  Fine-grained truck classification is critical for intelligent transportation systems (ITS), yet current LiDAR-based methods face scalability challenges due to their reliance on supervised deep learning and labor-intensive manual annotation. Vision-Language Models (VLMs) offer promising few-shot generalization, but their application to roadside LiDAR is limited by a modality gap between sparse 3D point clouds and dense 2D imagery. We propose a framework that bridges this gap by adapting off-the-shelf VLMs for fine-grained truck classification without parameter fine-tuning. Our new depth-aware image generation pipeline applies noise removal, spatial and temporal registration, orientation rectification, morphological operations, and anisotropic smoothing to transform sparse, occluded LiDAR scans into depth-encoded 2D visual proxies. Validated on a real-world dataset of 20 vehicle classes, our approach achieves competitive classification accuracy with as few as 16-30 examples per class, offering a scalable alternative to data-intensive supervised baselines. We further observe a "Semantic Anchor" effect: text-based guidance regularizes performance in ultra-low-shot regimes $k < 4$, but degrades accuracy in more-shot settings due to semantic mismatch. Furthermore, we demonstrate the efficacy of this framework as a Cold Start strategy, using VLM-generated labels to bootstrap lightweight supervised models. Notably, the few-shot VLM-based model achieves over correct classification rate of 75 percent for specific drayage categories (20ft, 40ft, and 53ft containers) entirely without the costly training or fine-tuning, significantly reducing the intensive demands of initial manual labeling, thus achieving a method of practical use in ITS applications.

  </details>



- **K-Sort Eval: Efficient Preference Evaluation for Visual Generation via Corrected VLM-as-a-Judge**  
  Zhikai Li, Jiatong Li, Xuewen Liu, Wangbo Zhao, Pan Du, Kaicheng Zhou, Qingyi Gu, Yang You, Zhen Dong, Kurt Keutzer  
  _2026-02-10_ · https://arxiv.org/abs/2602.09411v1  
  <details><summary>Abstract</summary>

  The rapid development of visual generative models raises the need for more scalable and human-aligned evaluation methods. While the crowdsourced Arena platforms offer human preference assessments by collecting human votes, they are costly and time-consuming, inherently limiting their scalability. Leveraging vision-language model (VLMs) as substitutes for manual judgments presents a promising solution. However, the inherent hallucinations and biases of VLMs hinder alignment with human preferences, thus compromising evaluation reliability. Additionally, the static evaluation approach lead to low efficiency. In this paper, we propose K-Sort Eval, a reliable and efficient VLM-based evaluation framework that integrates posterior correction and dynamic matching. Specifically, we curate a high-quality dataset from thousands of human votes in K-Sort Arena, with each instance containing the outputs and rankings of K models. When evaluating a new model, it undergoes (K+1)-wise free-for-all comparisons with existing models, and the VLM provide the rankings. To enhance alignment and reliability, we propose a posterior correction method, which adaptively corrects the posterior probability in Bayesian updating based on the consistency between the VLM prediction and human supervision. Moreover, we propose a dynamic matching strategy, which balances uncertainty and diversity to maximize the expected benefit of each comparison, thus ensuring more efficient evaluation. Extensive experiments show that K-Sort Eval delivers evaluation results consistent with K-Sort Arena, typically requiring fewer than 90 model runs, demonstrating both its efficiency and reliability.

  </details>



- **Single-Slice-to-3D Reconstruction in Medical Imaging and Natural Objects: A Comparative Benchmark with SAM 3D**  
  Yan Luo, Advaith Ravishankar, Serena Liu, Yutong Yang, Mengyu Wang  
  _2026-02-10_ · https://arxiv.org/abs/2602.09407v1  
  <details><summary>Abstract</summary>

  A 3D understanding of anatomy is central to diagnosis and treatment planning, yet volumetric imaging remains costly with long wait times. Image-to-3D foundations models can solve this issue by reconstructing 3D data from 2D modalites. Current foundation models are trained on natural image distributions to reconstruct naturalistic objects from a single image by leveraging geometric priors across pixels. However, it is unclear whether these learned geometric priors transfer to medical data. In this study, we present a controlled zero-shot benchmark of single slice medical image-to-3D reconstruction across five state-of-the-art image-to-3D models: SAM3D, Hunyuan3D-2.1, Direct3D, Hi3DGen, and TripoSG. These are evaluated across six medical datasets spanning anatomical and pathological structures and two natrual datasets, using voxel based metrics and point cloud distance metrics. Across medical datasets, voxel based overlap remains moderate for all models, consistent with a depth reconstruction failure mode when inferring volume from a single slice. In contrast, global distance metrics show more separation between methods: SAM3D achieves the strongest overall topological similarity to ground truth medical 3D data, while alternative models are more prone to over-simplication of reconstruction. Our results quantify the limits of single-slice medical reconstruction and highlight depth ambiguity caused by the planar nature of 2D medical data, motivating multi-view image-to-3D reconstruction to enable reliable medical 3D inference.

  </details>



- **Fully Differentiable Bidirectional Dual-Task Synergistic Learning for Semi-Supervised 3D Medical Image Segmentation**  
  Jun Li  
  _2026-02-10_ · https://arxiv.org/abs/2602.09378v1  
  <details><summary>Abstract</summary>

  Semi-supervised learning relaxes the need of large pixel-wise labeled datasets for image segmentation by leveraging unlabeled data. The scarcity of high-quality labeled data remains a major challenge in medical image analysis due to the high annotation costs and the need for specialized clinical expertise. Semi-supervised learning has demonstrated significant potential in addressing this bottleneck, with pseudo-labeling and consistency regularization emerging as two predominant paradigms. Dual-task collaborative learning, an emerging consistency-aware paradigm, seeks to derive supplementary supervision by establishing prediction consistency between related tasks. However, current methodologies are limited to unidirectional interaction mechanisms (typically regression-to-segmentation), as segmentation results can only be transformed into regression outputs in an offline manner, thereby failing to fully exploit the potential benefits of online bidirectional cross-task collaboration. Thus, we propose a fully Differentiable Bidirectional Synergistic Learning (DBiSL) framework, which seamlessly integrates and enhances four critical SSL components: supervised learning, consistency regularization, pseudo-supervised learning, and uncertainty estimation. Experiments on two benchmark datasets demonstrate our method's state-of-the-art performance. Beyond technical contributions, this work provides new insights into unified SSL framework design and establishes a new architectural foundation for dual-task-driven SSL, while offering a generic multitask learning framework applicable to broader computer vision applications. The code will be released on github upon acceptance.

  </details>



- **CAPER: Constrained and Procedural Reasoning for Robotic Scientific Experiments**  
  Jinghan Yang, Jingyi Hou, Xinbo Yu, Wei He, Yifan Wu  
  _2026-02-10_ · https://arxiv.org/abs/2602.09367v1  
  <details><summary>Abstract</summary>

  Robotic assistance in scientific laboratories requires procedurally correct long-horizon manipulation, reliable execution under limited supervision, and robustness in low-demonstration regimes. Such conditions greatly challenge end-to-end vision-language-action (VLA) models, whose assumptions of recoverable errors and data-driven policy learning often break down in protocol-sensitive experiments. We propose CAPER, a framework for Constrained And ProcEdural Reasoning for robotic scientific experiments, which explicitly restricts where learning and reasoning occur in the planning and control pipeline. Rather than strengthening end-to-end policies, CAPER enforces a responsibility-separated structure: task-level reasoning generates procedurally valid action sequences under explicit constraints, mid-level multimodal grounding realizes subtasks without delegating spatial decision-making to large language models, and low-level control adapts to physical uncertainty via reinforcement learning with minimal demonstrations. By encoding procedural commitments through interpretable intermediate representations, CAPER prevents execution-time violations of experimental logic, improving controllability, robustness, and data efficiency. Experiments on a scientific workflow benchmark and a public long-horizon manipulation dataset demonstrate consistent improvements in success rate and procedural correctness, particularly in low-data and long-horizon settings.

  </details>



- **Impact of domain adaptation in deep learning for medical image classifications**  
  Yihang Wu, Ahmad Chaddad  
  _2026-02-10_ · https://arxiv.org/abs/2602.09355v1  
  <details><summary>Abstract</summary>

  Domain adaptation (DA) is a quickly expanding area in machine learning that involves adjusting a model trained in one domain to perform well in another domain. While there have been notable progressions, the fundamental concept of numerous DA methodologies has persisted: aligning the data from various domains into a shared feature space. In this space, knowledge acquired from labeled source data can improve the model training on target data that lacks sufficient labels. In this study, we demonstrate the use of 10 deep learning models to simulate common DA techniques and explore their application in four medical image datasets. We have considered various situations such as multi-modality, noisy data, federated learning (FL), interpretability analysis, and classifier calibration. The experimental results indicate that using DA with ResNet34 in a brain tumor (BT) data set results in an enhancement of 4.7\% in model performance. Similarly, the use of DA can reduce the impact of Gaussian noise, as it provides $\sim 3\%$ accuracy increase using ResNet34 on a BT dataset. Furthermore, simply introducing DA into FL framework shows limited potential (e.g., $\sim 0.3\%$ increase in performance) for skin cancer classification. In addition, the DA method can improve the interpretability of the models using the gradcam++ technique, which offers clinical values. Calibration analysis also demonstrates that using DA provides a lower expected calibration error (ECE) value $\sim 2\%$ compared to CNN alone on a multi-modality dataset.

  </details>



- **Deep Modeling and Interpretation for Bladder Cancer Classification**  
  Ahmad Chaddad, Yihang Wu, Xianrui Chen  
  _2026-02-10_ · https://arxiv.org/abs/2602.09324v1  
  <details><summary>Abstract</summary>

  Deep models based on vision transformer (ViT) and convolutional neural network (CNN) have demonstrated remarkable performance on natural datasets. However, these models may not be similar in medical imaging, where abnormal regions cover only a small portion of the image. This challenge motivates this study to investigate the latest deep models for bladder cancer classification tasks. We propose the following to evaluate these deep models: 1) standard classification using 13 models (four CNNs and eight transormer-based models), 2) calibration analysis to examine if these models are well calibrated for bladder cancer classification, and 3) we use GradCAM++ to evaluate the interpretability of these models for clinical diagnosis. We simulate $\sim 300$ experiments on a publicly multicenter bladder cancer dataset, and the experimental results demonstrate that the ConvNext series indicate limited generalization ability to classify bladder cancer images (e.g., $\sim 60\%$ accuracy). In addition, ViTs show better calibration effects compared to ConvNext and swin transformer series. We also involve test time augmentation to improve the models interpretability. Finally, no model provides a one-size-fits-all solution for a feasible interpretable model. ConvNext series are suitable for in-distribution samples, while ViT and its variants are suitable for interpreting out-of-distribution samples.

  </details>



- **GAFR-Net: A Graph Attention and Fuzzy-Rule Network for Interpretable Breast Cancer Image Classification**  
  Lin-Guo Gao, Suxing Liu  
  _2026-02-10_ · https://arxiv.org/abs/2602.09318v1  
  <details><summary>Abstract</summary>

  Accurate classification of breast cancer histopathology images is pivotal for early oncological diagnosis and therapeutic intervention.However, conventional deep learning architectures often encounter performance degradation under limited annotations and suffer from a "blackbox" nature, hindering their clinical integration. To mitigate these limitations, we propose GAFRNet, a robust and interpretable Graph Attention and FuzzyRule Network specifically engineered for histopathology image classification with scarce supervision. GAFRNet constructs a similarity-driven graph representation to model intersample relationships and employs a multihead graph attention mechanism to capture complex relational features across heterogeneous tissue structures.Concurrently, a differentiable fuzzy-rule module encodes intrinsic topological descriptorsincluding node degree, clustering coefficient, and label consistencyinto explicit, human-understandable diagnostic logic. This design establishes transparent "IF-THEN" mappings that mimic the heuristic deduction process of medical experts, providing clear reasoning behind each prediction without relying on post-hoc attribution methods. Extensive evaluations on three benchmark datasets (BreakHis, Mini-DDSM, and ICIAR2018) demonstrate that GAFR-Net consistently outperforms various state-of-the-art methods across multiple magnifications and classification tasks. These results validate the superior generalization and practical utility of GAFR-Net as a reliable decision-support tool for weakly supervised medical image analysis.

  </details>



- **X-Mark: Saliency-Guided Robust Dataset Ownership Verification for Medical Imaging**  
  Pranav Kulkarni, Junfeng Guo, Heng Huang  
  _2026-02-10_ · https://arxiv.org/abs/2602.09284v1  
  <details><summary>Abstract</summary>

  High-quality medical imaging datasets are essential for training deep learning models, but their unauthorized use raises serious copyright and ethical concerns. Medical imaging presents a unique challenge for existing dataset ownership verification methods designed for natural images, as static watermark patterns generated in fixed-scale images scale poorly dynamic and high-resolution scans with limited visual diversity and subtle anatomical structures, while preserving diagnostic quality. In this paper, we propose X-Mark, a sample-specific clean-label watermarking method for chest x-ray copyright protection. Specifically, X-Mark uses a conditional U-Net to generate unique perturbations within salient regions of each sample. We design a multi-component training objective to ensure watermark efficacy, robustness against dynamic scaling processes while preserving diagnostic quality and visual-distinguishability. We incorporate Laplacian regularization into our training objective to penalize high-frequency perturbations and achieve watermark scale-invariance. Ownership verification is performed in a black-box setting to detect characteristic behaviors in suspicious models. Extensive experiments on CheXpert verify the effectiveness of X-Mark, achieving WSR of 100% and reducing probability of false positives in Ind-M scenario by 12%, while demonstrating resistance to potential adaptive attacks.

  </details>



- **Data-centric Design of Learning-based Surgical Gaze Perception Models in Multi-Task Simulation**  
  Yizhou Li, Shuyuan Yang, Jiaji Su, Zonghe Chua  
  _2026-02-09_ · https://arxiv.org/abs/2602.09259v1  
  <details><summary>Abstract</summary>

  In robot-assisted minimally invasive surgery (RMIS), reduced haptic feedback and depth cues increase reliance on expert visual perception, motivating gaze-guided training and learning-based surgical perception models. However, operative expert gaze is costly to collect, and it remains unclear how the source of gaze supervision, both expertise level (intermediate vs. novice) and perceptual modality (active execution vs. passive viewing), shapes what attention models learn. We introduce a paired active-passive, multi-task surgical gaze dataset collected on the da Vinci SimNow simulator across four drills. Active gaze was recorded during task execution using a VR headset with eye tracking, and the corresponding videos were reused as stimuli to collect passive gaze from observers, enabling controlled same-video comparisons. We quantify skill- and modality-dependent differences in gaze organization and evaluate the substitutability of passive gaze for operative supervision using fixation density overlap analyses and single-frame saliency modeling. Across settings, MSI-Net produced stable, interpretable predictions, whereas SalGAN was unstable and often poorly aligned with human fixations. Models trained on passive gaze recovered a substantial portion of intermediate active attention, but with predictable degradation, and transfer was asymmetric between active and passive targets. Notably, novice passive labels approximated intermediate-passive targets with limited loss on higher-quality demonstrations, suggesting a practical path for scalable, crowd-sourced gaze supervision in surgical coaching and perception modeling.

  </details>



- **STaR: Scalable Task-Conditioned Retrieval for Long-Horizon Multimodal Robot Memory**  
  Mingfeng Yuan, Hao Zhang, Mahan Mohammadi, Runhao Li, Jinjun Shan, Steven L. Waslander  
  _2026-02-09_ · https://arxiv.org/abs/2602.09255v1  
  <details><summary>Abstract</summary>

  Mobile robots are often deployed over long durations in diverse open, dynamic scenes, including indoor setting such as warehouses and manufacturing facilities, and outdoor settings such as agricultural and roadway operations. A core challenge is to build a scalable long-horizon memory that supports an agentic workflow for planning, retrieval, and reasoning over open-ended instructions at variable granularity, while producing precise, actionable answers for navigation. We present STaR, an agentic reasoning framework that (i) constructs a task-agnostic, multimodal long-term memory that generalizes to unseen queries while preserving fine-grained environmental semantics (object attributes, spatial relations, and dynamic events), and (ii) introduces a Scalable TaskConditioned Retrieval algorithm based on the Information Bottleneck principle to extract from long-term memory a compact, non-redundant, information-rich set of candidate memories for contextual reasoning. We evaluate STaR on NaVQA (mixed indoor/outdoor campus scenes) and WH-VQA, a customized warehouse benchmark with many visually similar objects built with Isaac Sim, emphasizing contextual reasoning. Across the two datasets, STaR consistently outperforms strong baselines, achieving higher success rates and markedly lower spatial error. We further deploy STaR on a real Husky wheeled robot in both indoor and outdoor environments, demonstrating robust longhorizon reasoning, scalability, and practical utility.

  </details>



- **VLM-Guided Iterative Refinement for Surgical Image Segmentation with Foundation Models**  
  Ange Lou, Yamin Li, Qi Chang, Nan Xi, Luyuan Xie, Zichao Li, Tianyu Luan  
  _2026-02-09_ · https://arxiv.org/abs/2602.09252v1  
  <details><summary>Abstract</summary>

  Surgical image segmentation is essential for robot-assisted surgery and intraoperative guidance. However, existing methods are constrained to predefined categories, produce one-shot predictions without adaptive refinement, and lack mechanisms for clinician interaction. We propose IR-SIS, an iterative refinement system for surgical image segmentation that accepts natural language descriptions. IR-SIS leverages a fine-tuned SAM3 for initial segmentation, employs a Vision-Language Model to detect instruments and assess segmentation quality, and applies an agentic workflow that adaptively selects refinement strategies. The system supports clinician-in-the-loop interaction through natural language feedback. We also construct a multi-granularity language-annotated dataset from EndoVis2017 and EndoVis2018 benchmarks. Experiments demonstrate state-of-the-art performance on both in-domain and out-of-distribution data, with clinician interaction providing additional improvements. Our work establishes the first language-based surgical segmentation framework with adaptive self-refinement capabilities.

  </details>


