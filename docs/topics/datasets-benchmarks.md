# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-03-11 07:09 UTC_

Total papers shown: **50**


---

- **BEACON: Language-Conditioned Navigation Affordance Prediction under Occlusion**  
  Xinyu Gao, Gang Chen, Javier Alonso-Mora  
  _2026-03-10_ · https://arxiv.org/abs/2603.09961v1  
  <details><summary>Abstract</summary>

  Language-conditioned local navigation requires a robot to infer a nearby traversable target location from its current observation and an open-vocabulary, relational instruction. Existing vision-language spatial grounding methods usually rely on vision-language models (VLMs) to reason in image space, producing 2D predictions tied to visible pixels. As a result, they struggle to infer target locations in occluded regions, typically caused by furniture or moving humans. To address this issue, we propose BEACON, which predicts an ego-centric Bird's-Eye View (BEV) affordance heatmap over a bounded local region including occluded areas. Given an instruction and surround-view RGB-D observations from four directions around the robot, BEACON predicts the BEV heatmap by injecting spatial cues into a VLM and fusing the VLM's output with depth-derived BEV features. Using an occlusion-aware dataset built in the Habitat simulator, we conduct detailed experimental analysis to validate both our BEV space formulation and the design choices of each module. Our method improves the accuracy averaged across geodesic thresholds by 22.74 percentage points over the state-of-the-art image-space baseline on the validation subset with occluded target locations. Our project page is: https://xin-yu-gao.github.io/beacon.

  </details>



- **From Semantics to Pixels: Coarse-to-Fine Masked Autoencoders for Hierarchical Visual Understanding**  
  Wenzhao Xiang, Yue Wu, Hongyang Yu, Feng Gao, Fan Yang, Xilin Chen  
  _2026-03-10_ · https://arxiv.org/abs/2603.09955v1  
  <details><summary>Abstract</summary>

  Self-supervised visual pre-training methods face an inherent tension: contrastive learning (CL) captures global semantics but loses fine-grained detail, while masked image modeling (MIM) preserves local textures but suffers from "attention drift" due to semantically-agnostic random masking. We propose C2FMAE, a coarse-to-fine masked autoencoder that resolves this tension by explicitly learning hierarchical visual representations across three data granularities: semantic masks (scene-level), instance masks (object-level), and RGB images (pixel-level). Two synergistic innovations enforce a strict top-down learning principle. First, a cascaded decoder sequentially reconstructs from scene semantics to object instances to pixel details, establishing explicit cross-granularity dependencies that parallel decoders cannot capture. Second, a progressive masking curriculum dynamically shifts the training focus from semantic-guided to instance-guided and finally to random masking, creating a structured learning path from global context to local features. To support this framework, we construct a large-scale multi-granular dataset with high-quality pseudo-labels for all 1.28M ImageNet-1K images. Extensive experiments show that C2FMAE achieves significant performance gains on image classification, object detection, and semantic segmentation, validating the effectiveness of our hierarchical design in learning more robust and generalizable representations.

  </details>



- **NanoBench: A Multi-Task Benchmark Dataset for Nano-Quadrotor System Identification, Control, and State Estimation**  
  Syed Izzat Ullah, Jose Baca  
  _2026-03-10_ · https://arxiv.org/abs/2603.09908v1  
  <details><summary>Abstract</summary>

  Existing aerial-robotics benchmarks target vehicles from hundreds of grams to several kilograms and typically expose only high-level state data. They omit the actuator-level signals required to study nano-scale quadrotors, where low-Reynolds number aerodynamics, coreless DC motor nonlinearities, and severe computational constraints invalidate models and controllers developed for larger vehicles. We introduce NanoBench, an open-source multi-task benchmark collected on the commercially available Crazyflie 2.1 nano-quadrotor (takeoff weight 27 g) in a Vicon motion capture arena. The dataset contains over 170 flight trajectories spanning hover, multi-frequency excitation, standard tracking, and aggressive maneuvers across multiple speed regimes. Each trajectory provides synchronized Vicon ground truth, raw IMU data, onboard extended Kalman filter estimates, PID controller internals, and motor PWM commands at 100 Hz, alongside battery telemetry at 10 Hz, aligned with sub-0.5 ms consistency. NanoBench defines standardized evaluation protocols, train/test splits, and open-source baselines for three tasks: nonlinear system identification, closed-loop controller benchmarking, and onboard state estimation assessment. To our knowledge, it is the first public dataset to jointly provide actuator commands, controller internals, and estimator outputs with millimeter-accurate ground truth on a commercially available nano-scale aerial platform.

  </details>



- **Stepping VLMs onto the Court: Benchmarking Spatial Intelligence in Sports**  
  Yuchen Yang, Yuqing Shao, Duxiu Huang, Linfeng Dong, Yifei Liu, Suixin Tang, Xiang Zhou, Yuanyuan Gao, Wei Wang, Yue Zhou, et al.  
  _2026-03-10_ · https://arxiv.org/abs/2603.09896v1  
  <details><summary>Abstract</summary>

  Sports have long attracted broad attention as they push the limits of human physical and cognitive capabilities. Amid growing interest in spatial intelligence for vision-language models (VLMs), sports provide a natural testbed for understanding high-intensity human motion and dynamic object interactions. To this end, we present CourtSI, the first large-scale spatial intelligence dataset tailored to sports scenarios. CourtSI contains over 1M QA pairs, organized under a holistic taxonomy that systematically covers spatial counting, distance measurement, localization, and relational reasoning, across representative net sports including badminton, tennis, and table tennis. Leveraging well-defined court geometry as metric anchors, we develop a semi-automatic data engine to reconstruct sports scenes, enabling scalable curation of CourtSI. In addition, we introduce CourtSI-Bench, a high-quality evaluation benchmark comprising 3,686 QA pairs with rigorous human verification. We evaluate 25 proprietary and open-source VLMs on CourtSI-Bench, revealing a remaining human-AI performance gap and limited generalization from existing spatial intelligence benchmarks. These findings indicate that sports scenarios expose limitations in spatial intelligence capabilities captured by existing benchmarks. Further, fine-tuning Qwen3-VL-8B on CourtSI improves accuracy on CourtSI-Bench by 23.5 percentage points. The adapted model also generalizes effectively to CourtSI-Ext, an evaluation set built on a similar but unseen sport, and demonstrates enhanced spatial-aware commentary generation. Together, these findings demonstrate that CourtSI provides a scalable pathway toward advancing spatial intelligence of VLMs in sports.

  </details>



- **MissBench: Benchmarking Multimodal Affective Analysis under Imbalanced Missing Modalities**  
  Tien Anh Pham, Phuong-Anh Nguyen, Duc-Trong Le, Cam-Van Thi Nguyen  
  _2026-03-10_ · https://arxiv.org/abs/2603.09874v1  
  <details><summary>Abstract</summary>

  Multimodal affective computing underpins key tasks such as sentiment analysis and emotion recognition. Standard evaluations, however, often assume that textual, acoustic, and visual modalities are equally available. In real applications, some modalities are systematically more fragile or expensive, creating imbalanced missing rates and training biases that task-level metrics alone do not reveal. We introduce MissBench, a benchmark and framework for multimodal affective tasks that standardizes both shared and imbalanced missing-rate protocols on four widely used sentiment and emotion datasets. MissBench also defines two diagnostic metrics. The Modality Equity Index (MEI) measures how fairly different modalities contribute across missing-modality configurations. The Modality Learning Index (MLI) quantifies optimization imbalance by comparing modality-specific gradient norms during training, aggregated across modality-related modules. Experiments on representative method families show that models that appear robust under shared missing rates can still exhibit marked modality inequity and optimization imbalance under imbalanced conditions. These findings position MissBench, together with MEI and MLI, as practical tools for stress-testing and analyzing multimodal affective models in realistic incomplete-modality settings.For reproducibility, we release our code at: https://anonymous.4open.science/r/MissBench-4098/

  </details>



- **MA-EgoQA: Question Answering over Egocentric Videos from Multiple Embodied Agents**  
  Kangsan Kim, Yanlai Yang, Suji Kim, Woongyeong Yeo, Youngwan Lee, Mengye Ren, Sung Ju Hwang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09827v1  
  <details><summary>Abstract</summary>

  As embodied models become powerful, humans will collaborate with multiple embodied AI agents at their workplace or home in the future. To ensure better communication between human users and the multi-agent system, it is crucial to interpret incoming information from agents in parallel and refer to the appropriate context for each query. Existing challenges include effectively compressing and communicating high volumes of individual sensory inputs in the form of video and correctly aggregating multiple egocentric videos to construct system-level memory. In this work, we first formally define a novel problem of understanding multiple long-horizon egocentric videos simultaneously collected from embodied agents. To facilitate research in this direction, we introduce MultiAgent-EgoQA (MA-EgoQA), a benchmark designed to systemically evaluate existing models in our scenario. MA-EgoQA provides 1.7k questions unique to multiple egocentric streams, spanning five categories: social interaction, task coordination, theory-of-mind, temporal reasoning, and environmental interaction. We further propose a simple baseline model for MA-EgoQA named EgoMAS, which leverages shared memory across embodied agents and agent-wise dynamic retrieval. Through comprehensive evaluation across diverse baselines and EgoMAS on MA-EgoQA, we find that current approaches are unable to effectively handle multiple egocentric streams, highlighting the need for future advances in system-level understanding across the agents. The code and benchmark are available at https://ma-egoqa.github.io.

  </details>



- **VLM-Loc: Localization in Point Cloud Maps via Vision-Language Models**  
  Shuhao Kang, Youqi Liao, Peijie Wang, Wenlong Liao, Qilin Zhang, Benjamin Busam, Xieyuanli Chen, Yun Liu  
  _2026-03-10_ · https://arxiv.org/abs/2603.09826v1  
  <details><summary>Abstract</summary>

  Text-to-point-cloud (T2P) localization aims to infer precise spatial positions within 3D point cloud maps from natural language descriptions, reflecting how humans perceive and communicate spatial layouts through language. However, existing methods largely rely on shallow text-point cloud correspondence without effective spatial reasoning, limiting their accuracy in complex environments. To address this limitation, we propose VLM-Loc, a framework that leverages the spatial reasoning capability of large vision-language models (VLMs) for T2P localization. Specifically, we transform point clouds into bird's-eye-view (BEV) images and scene graphs that jointly encode geometric and semantic context, providing structured inputs for the VLM to learn cross-modal representations bridging linguistic and spatial semantics. On top of these representations, we introduce a partial node assignment mechanism that explicitly associates textual cues with scene graph nodes, enabling interpretable spatial reasoning for accurate localization. To facilitate systematic evaluation across diverse scenes, we present CityLoc, a benchmark built from multi-source point clouds for fine-grained T2P localization. Experiments on CityLoc demonstrate VLM-Loc achieves superior accuracy and robustness compared to state-of-the-art methods. Our code, model, and dataset are available at \href{https://github.com/MCG-NKU/nku-3d-vision}{repository}.

  </details>



- **RA-SSU: Towards Fine-Grained Audio-Visual Learning with Region-Aware Sound Source Understanding**  
  Muyi Sun, Yixuan Wang, Hong Wang, Chen Su, Man Zhang, Xingqun Qi, Qi Li, Zhenan Sun  
  _2026-03-10_ · https://arxiv.org/abs/2603.09809v1  
  <details><summary>Abstract</summary>

  Audio-Visual Learning (AVL) is one fundamental task of multi-modality learning and embodied intelligence, displaying the vital role in scene understanding and interaction. However, previous researchers mostly focus on exploring downstream tasks from a coarse-grained perspective (e.g., audio-visual correspondence, sound source localization, and audio-visual event localization). Considering providing more specific scene perception details, we newly define a fine-grained Audio-Visual Learning task, termed Region-Aware Sound Source Understanding (RA-SSU), which aims to achieve region-aware, frame-level, and high-quality sound source understanding. To support this goal, we innovatively construct two corresponding datasets, i.e. fine-grained Music (f-Music) and fine-grained Lifescene (f-Lifescene), each containing annotated sound source masks and frame-by-frame textual descriptions. The f-Music dataset includes 3,976 samples across 22 scene types related to specific application scenarios, focusing on music scenes with complex instrument mixing. The f-Lifescene dataset contains 6,156 samples across 61 types representing diverse sounding objects in life scenarios. Moreover, we propose SSUFormer, a Sound-Source Understanding TransFormer benchmark that facilitates both the sound source segmentation and sound region description with a multi-modal input and multi-modal output architecture. Specifically, we design two modules for this framework, Mask Collaboration Module (MCM) and Mixture of Hierarchical-prompted Experts (MoHE), to respectively enhance the accuracy and enrich the elaboration of the sound source description. Extensive experiments are conducted on our two datasets to verify the feasibility of the task, evaluate the availability of the datasets, and demonstrate the superiority of the SSUFormer, which achieves SOTA performance on the Sound Source Understanding benchmark.

  </details>



- **TIMID: Time-Dependent Mistake Detection in Videos of Robot Executions**  
  Nerea Gallego, Fernando Salanova, Claudio Mannarano, Cristian Mahulea, Eduardo Montijano  
  _2026-03-10_ · https://arxiv.org/abs/2603.09782v1  
  <details><summary>Abstract</summary>

  As robotic systems execute increasingly difficult task sequences, so does the number of ways in which they can fail. Video Anomaly Detection (VAD) frameworks typically focus on singular, low-level kinematic or action failures, struggling to identify more complex temporal or spatial task violations, because they do not necessarily manifest as low-level execution errors. To address this problem, the main contribution of this paper is a new VAD-inspired architecture, TIMID, which is able to detect robot time-dependent mistakes when executing high-level tasks. Our architecture receives as inputs a video and prompts of the task and the potential mistake, and returns a frame-level prediction in the video of whether the mistake is present or not. By adopting a VAD formulation, the model can be trained with weak supervision, requiring only a single label per video. Additionally, to alleviate the problem of data scarcity of incorrect executions, we introduce a multi-robot simulation dataset with controlled temporal errors and real executions for zero-shot sim-to-real evaluation. Our experiments demonstrate that out-of-the-box VLMs lack the explicit temporal reasoning required for this task, whereas our framework successfully detects different types of temporal errors. Project: https://ropertunizar.github.io/TIMID/

  </details>



- **PanoAffordanceNet: Towards Holistic Affordance Grounding in 360° Indoor Environments**  
  Guoliang Zhu, Wanjun Jia, Caoyang Shao, Yuheng Zhang, Zhiyong Li, Kailun Yang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09760v1  
  <details><summary>Abstract</summary>

  Global perception is essential for embodied agents in 360° spaces, yet current affordance grounding remains largely object-centric and restricted to perspective views. To bridge this gap, we introduce a novel task: Holistic Affordance Grounding in 360° Indoor Environments. This task faces unique challenges, including severe geometric distortions from Equirectangular Projection (ERP), semantic dispersion, and cross-scale alignment difficulties. We propose PanoAffordanceNet, an end-to-end framework featuring a Distortion-Aware Spectral Modulator (DASM) for latitude-dependent calibration and an Omni-Spherical Densification Head (OSDH) to restore topological continuity from sparse activations. By integrating multi-level constraints comprising pixel-wise, distributional, and region-text contrastive objectives, our framework effectively suppresses semantic drift under low supervision. Furthermore, we construct 360-AGD, the first high-quality panoramic affordance grounding dataset. Extensive experiments demonstrate that PanoAffordanceNet significantly outperforms existing methods, establishing a solid baseline for scene-level perception in embodied intelligence. The source code and benchmark dataset will be made publicly available at https://github.com/GL-ZHU925/PanoAffordanceNet.

  </details>



- **ENIGMA-360: An Ego-Exo Dataset for Human Behavior Understanding in Industrial Scenarios**  
  Francesco Ragusa, Rosario Leonardi, Michele Mazzamuto, Daniele Di Mauro, Camillo Quattrocchi, Alessandro Passanisi, Irene D'Ambra, Antonino Furnari, Giovanni Maria Farinella  
  _2026-03-10_ · https://arxiv.org/abs/2603.09741v1  
  <details><summary>Abstract</summary>

  Understanding human behavior from complementary egocentric (ego) and exocentric (exo) points of view enables the development of systems that can support workers in industrial environments and enhance their safety. However, progress in this area is hindered by the lack of datasets capturing both views in realistic industrial scenarios. To address this gap, we propose ENIGMA-360, a new ego-exo dataset acquired in a real industrial scenario. The dataset is composed of 180 egocentric and 180 exocentric procedural videos temporally synchronized offering complementary information of the same scene. The 360 videos have been labeled with temporal and spatial annotations, enabling the study of different aspects of human behavior in industrial domain. We provide baseline experiments for 3 foundational tasks for human behavior understanding: 1) Temporal Action Segmentation, 2) Keystep Recognition and 3) Egocentric Human-Object Interaction Detection, showing the limits of state-of-the-art approaches on this challenging scenario. These results highlight the need for new models capable of robust ego-exo understanding in real-world environments. We publicly release the dataset and its annotations at https://iplab.dmi.unict.it/ENIGMA-360.

  </details>



- **$M^2$-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs**  
  Kaixin Lin, Kunyu Peng, Di Wen, Yufan Chen, Ruiping Liu, Kailun Yang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09737v1  
  <details><summary>Abstract</summary>

  Semantic occupancy prediction enables dense 3D geometric and semantic understanding for autonomous driving. However, existing camera-based approaches implicitly assume complete surround-view observations, an assumption that rarely holds in real-world deployment due to occlusion, hardware malfunction, or communication failures. We study semantic occupancy prediction under incomplete multi-camera inputs and introduce $M^2$-Occ, a framework designed to preserve geometric structure and semantic coherence when views are missing. $M^2$-Occ addresses two complementary challenges. First, a Multi-view Masked Reconstruction (MMR) module leverages the spatial overlap among neighboring cameras to recover missing-view representations directly in the feature space. Second, a Feature Memory Module (FMM) introduces a learnable memory bank that stores class-level semantic prototypes. By retrieving and integrating these global priors, the FMM refines ambiguous voxel features, ensuring semantic consistency even when observational evidence is incomplete. We introduce a systematic missing-view evaluation protocol on the nuScenes-based SurroundOcc benchmark, encompassing both deterministic single-view failures and stochastic multi-view dropout scenarios. Under the safety-critical missing back-view setting, $M^2$-Occ improves the IoU by 4.93%. As the number of missing cameras increases, the robustness gap further widens; for instance, under the setting with five missing views, our method boosts the IoU by 5.01%. These gains are achieved without compromising full-view performance. The source code will be publicly released at https://github.com/qixi7up/M2-Occ.

  </details>



- **EXPLORE-Bench: Egocentric Scene Prediction with Long-Horizon Reasoning**  
  Chengjun Yu, Xuhan Zhu, Chaoqun Du, Pengfei Yu, Wei Zhai, Yang Cao, Zheng-Jun Zha  
  _2026-03-10_ · https://arxiv.org/abs/2603.09731v1  
  <details><summary>Abstract</summary>

  Multimodal large language models (MLLMs) are increasingly considered as a foundation for embodied agents, yet it remains unclear whether they can reliably reason about the long-term physical consequences of actions from an egocentric viewpoint. We study this gap through a new task, Egocentric Scene Prediction with LOng-horizon REasoning: given an initial-scene image and a sequence of atomic action descriptions, a model is asked to predict the final scene after all actions are executed. To enable systematic evaluation, we introduce EXPLORE-Bench, a benchmark curated from real first-person videos spanning diverse scenarios. Each instance pairs long action sequences with structured final-scene annotations, including object categories, visual attributes, and inter-object relations, which supports fine-grained, quantitative assessment. Experiments on a range of proprietary and open-source MLLMs reveal a significant performance gap to humans, indicating that long-horizon egocentric reasoning remains a major challenge. We further analyze test-time scaling via stepwise reasoning and show that decomposing long action sequences can improve performance to some extent, while incurring non-trivial computational overhead. Overall, EXPLORE-Bench provides a principled testbed for measuring and advancing long-horizon reasoning for egocentric embodied perception.

  </details>



- **GSStream: 3D Gaussian Splatting based Volumetric Scene Streaming System**  
  Zhiye Tang, Qiudan Zhang, Lei Zhang, Junhui Hou, You Yang, Xu Wang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09718v1  
  <details><summary>Abstract</summary>

  Recently, the 3D Gaussian splatting (3DGS) technique for real-time radiance field rendering has revolutionized the field of volumetric scene representation, providing users with an immersive experience. But in return, it also poses a large amount of data volume, which is extremely bandwidth-intensive. Cutting-edge researchers have tried to introduce different approaches and construct multiple variants for 3DGS to obtain a more compact scene representation, but it is still challenging for real-time distribution. In this paper, we propose GSStream, a novel volumetric scene streaming system to support 3DGS data format. Specifically, GSStream integrates a collaborative viewport prediction module to better predict users' future behaviors by learning collaborative priors and historical priors from multiple users and users' viewport sequences and a deep reinforcement learning (DRL)-based bitrate adaptation module to tackle the state and action space variability challenge of the bitrate adaptation problem, achieving efficient volumetric scene delivery. Besides, we first build a user viewport trajectory dataset for volumetric scenes to support the training and streaming simulation. Extensive experiments prove that our proposed GSStream system outperforms existing representative volumetric scene streaming systems in visual quality and network usage. Demo video: https://youtu.be/3WEe8PN8yvA.

  </details>



- **TemporalDoRA: Temporal PEFT for Robust Surgical Video Question Answering**  
  Luca Carlini, Chiara Lena, Cesare Hassan, Danail Stoyanov, Elena De Momi, Sophia Bano, Mobarak I. Hoque  
  _2026-03-10_ · https://arxiv.org/abs/2603.09696v1  
  <details><summary>Abstract</summary>

  Surgical Video Question Answering (VideoQA) requires accurate temporal grounding while remaining robust to natural variation in how clinicians phrase questions, where linguistic bias can arise. Standard Parameter Efficient Fine Tuning (PEFT) methods adapt pretrained projections without explicitly modeling frame-to-frame interactions within the adaptation pathway, limiting their ability to exploit sparse temporal evidence. We introduce TemporalDoRA, a video-specific PEFT formulation that extends Weight-Decomposed Low-Rank Adaptation by (i) inserting lightweight temporal Multi-Head Attention (MHA) inside the low-rank bottleneck of the vision encoder and (ii) selectively applying weight decomposition only to the trainable low-rank branch rather than the full adapted weight. This design enables temporally-aware updates while preserving a frozen backbone and stable scaling. By mixing information across frames within the adaptation subspace, TemporalDoRA steers updates toward temporally consistent visual cues and improves robustness with minimal parameter overhead. To benchmark this setting, we present REAL-Colon-VQA, a colonoscopy VideoQA dataset with 6,424 clip--question pairs, including paired rephrased Out-of-Template questions to evaluate sensitivity to linguistic variation. TemporalDoRA improves Out-of-Template performance, and ablation studies confirm that temporal mixing inside the low-rank branch is the primary driver of these gains. We also validate on EndoVis18-VQA adapted to short clips and observe consistent improvements on the Out-of-Template split. Code and dataset available at~\href{https://anonymous.4open.science/r/TemporalDoRA-BFC8/}{Anonymous GitHub}.

  </details>



- **DRIFT: Dual-Representation Inter-Fusion Transformer for Automated Driving Perception with 4D Radar Point Clouds**  
  Siqi Pei, Andras Palffy, Dariu M. Gavrila  
  _2026-03-10_ · https://arxiv.org/abs/2603.09695v1  
  <details><summary>Abstract</summary>

  4D radars, which provide 3D point cloud data along with Doppler velocity, are attractive components of modern automated driving systems due to their low cost and robustness under adverse weather conditions. However, they provide a significantly lower point cloud density than LiDAR sensors. This makes it important to exploit not only local but also global contextual scene information. This paper proposes DRIFT, a model that effectively captures and fuses both local and global contexts through a dual-path architecture. The model incorporates a point path to aggregate fine-grained local features and a pillar path to encode coarse-grained global features. These two parallel paths are intertwined via novel feature-sharing layers at multiple stages, enabling full utilization of both representations. DRIFT is evaluated on the widely used View-of-Delft (VoD) dataset and a proprietary internal dataset. It outperforms the baselines on the tasks of object detection and/or free road estimation. For example, DRIFT achieves a mean average precision (mAP) of 52.6\% (compared to, say, 45.4\% of CenterPoint) on the VoD dataset.

  </details>



- **AutoViVQA: A Large-Scale Automatically Constructed Dataset for Vietnamese Visual Question Answering**  
  Nguyen Anh Tuong, Phan Ba Duc, Nguyen Trung Quoc, Tran Dac Thinh, Dang Duy Lan, Nguyen Quoc Thinh, Tung Le  
  _2026-03-10_ · https://arxiv.org/abs/2603.09689v1  
  <details><summary>Abstract</summary>

  Visual Question Answering (VQA) is a fundamental multimodal task that requires models to jointly understand visual and textual information. Early VQA systems relied heavily on language biases, motivating subsequent work to emphasize visual grounding and balanced datasets. With the success of large-scale pre-trained transformers for both text and vision domains -- such as PhoBERT for Vietnamese language understanding and Vision Transformers (ViT) for image representation learning -- multimodal fusion has achieved remarkable progress. For Vietnamese VQA, several datasets have been introduced to promote research in low-resource multimodal learning, including ViVQA, OpenViVQA, and the recently proposed ViTextVQA. These resources enable benchmarking of models that integrate linguistic and visual features in the Vietnamese context. Evaluation of VQA systems often employs automatic metrics originally designed for image captioning or machine translation, such as BLEU, METEOR, CIDEr, Recall, Precision, and F1-score. However, recent research suggests that large language models can further improve the alignment between automatic evaluation and human judgment in VQA tasks. In this work, we explore Vietnamese Visual Question Answering using transformer-based architectures, leveraging both textual and visual pre-training while systematically comparing automatic evaluation metrics under multilingual settings.

  </details>



- **Improving 3D Foot Motion Reconstruction in Markerless Monocular Human Motion Capture**  
  Tom Wehrbein, Bodo Rosenhahn  
  _2026-03-10_ · https://arxiv.org/abs/2603.09681v1  
  <details><summary>Abstract</summary>

  State-of-the-art methods can recover accurate overall 3D human body motion from in-the-wild videos. However, they often fail to capture fine-grained articulations, especially in the feet, which are critical for applications such as gait analysis and animation. This limitation results from training datasets with inaccurate foot annotations and limited foot motion diversity. We address this gap with FootMR, a Foot Motion Refinement method that refines foot motion estimated by an existing human recovery model through lifting 2D foot keypoint sequences to 3D. By avoiding direct image input, FootMR circumvents inaccurate image-3D annotation pairs and can instead leverage large-scale motion capture data. To resolve ambiguities of 2D-to-3D lifting, FootMR incorporates knee and foot motion as context and predicts only residual foot motion. Generalization to extreme foot poses is further improved by representing joints in global rather than parent-relative rotations and applying extensive data augmentation. To support evaluation of foot motion reconstruction, we introduce MOOF, a 2D dataset of complex foot movements. Experiments on MOOF, MOYO, and RICH show that FootMR outperforms state-of-the-art methods, reducing ankle joint angle error on MOYO by up to 30% over the best video-based approach.

  </details>



- **DiffWind: Physics-Informed Differentiable Modeling of Wind-Driven Object Dynamics**  
  Yuanhang Lei, Boming Zhao, Zesong Yang, Xingxuan Li, Tao Cheng, Haocheng Peng, Ru Zhang, Yang Yang, Siyuan Huang, Yujun Shen, et al.  
  _2026-03-10_ · https://arxiv.org/abs/2603.09668v1  
  <details><summary>Abstract</summary>

  Modeling wind-driven object dynamics from video observations is highly challenging due to the invisibility and spatio-temporal variability of wind, as well as the complex deformations of objects. We present DiffWind, a physics-informed differentiable framework that unifies wind-object interaction modeling, video-based reconstruction, and forward simulation. Specifically, we represent wind as a grid-based physical field and objects as particle systems derived from 3D Gaussian Splatting, with their interaction modeled by the Material Point Method (MPM). To recover wind-driven object dynamics, we introduce a reconstruction framework that jointly optimizes the spatio-temporal wind force field and object motion through differentiable rendering and simulation. To ensure physical validity, we incorporate the Lattice Boltzmann Method (LBM) as a physics-informed constraint, enforcing compliance with fluid dynamics laws. Beyond reconstruction, our method naturally supports forward simulation under novel wind conditions and enables new applications such as wind retargeting. We further introduce WD-Objects, a dataset of synthetic and real-world wind-driven scenes. Extensive experiments demonstrate that our method significantly outperforms prior dynamic scene modeling approaches in both reconstruction accuracy and simulation fidelity, opening a new avenue for video-based wind-object interaction modeling.

  </details>



- **X-GS: An Extensible Open Framework Unifying 3DGS Architectures with Downstream Multimodal Models**  
  Yueen Ma, Irwin King  
  _2026-03-10_ · https://arxiv.org/abs/2603.09632v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has emerged as a powerful technique for novel view synthesis, subsequently extending into numerous spatial AI applications. However, most existing 3DGS methods are isolated, focusing on specific domains such as online SLAM, semantic enrichment, or 3DGS for unposed images. In this paper, we introduce X-GS, an extensible open framework that unifies a broad range of techniques to enable real-time 3DGS-based online SLAM enriched with semantics, bridging the gap to downstream multimodal models. At the core of X-GS is a highly efficient pipeline called X-GS-Perceiver, capable of taking unposed RGB (or optionally RGB-D) video streams as input to co-optimize geometry and poses, and distill high-dimensional semantic features from vision foundation models into the 3D Gaussians. We achieve real-time performance through a novel online Vector Quantization (VQ) module, a GPU-accelerated grid-sampling scheme, and a highly parallelized pipeline design. The semantic 3D Gaussians can then be utilized by vision-language models within the X-GS-Thinker component, enabling downstream tasks such as object detection, zero-shot caption generation, and potentially embodied tasks. Experimental results on real-world datasets showcase the efficacy, efficiency, and newly unlocked multimodal capabilities of the X-GS framework.

  </details>



- **Grounding Synthetic Data Generation With Vision and Language Models**  
  Ümit Mert Çağlar, Alptekin Temizel  
  _2026-03-10_ · https://arxiv.org/abs/2603.09625v1  
  <details><summary>Abstract</summary>

  Deep learning models benefit from increasing data diversity and volume, motivating synthetic data augmentation to improve existing datasets. However, existing evaluation metrics for synthetic data typically calculate latent feature similarity, which is difficult to interpret and does not always correlate with the contribution to downstream tasks. We propose a vision-language grounded framework for interpretable synthetic data augmentation and evaluation in remote sensing. Our approach combines generative models, semantic segmentation and image captioning with vision and language models. Based on this framework, we introduce ARAS400k: A large-scale Remote sensing dataset Augmented with Synthetic data for segmentation and captioning, containing 100k real images and 300k synthetic images, each paired with segmentation maps and descriptions. ARAS400k enables the automated evaluation of synthetic data by analyzing semantic composition, minimizing caption redundancy, and verifying cross-modal consistency between visual structures and language descriptions. Experimental results indicate that while models trained exclusively on synthetic data reach competitive performance levels, those trained with augmented data (a combination of real and synthetic images) consistently outperform real-data baselines. Consequently, this work establishes a scalable benchmark for remote sensing tasks, specifically in semantic segmentation and image captioning. The dataset is available at zenodo.org/records/18890661 and the code base at github.com/caglarmert/ARAS400k.

  </details>



- **A saccade-inspired approach to image classification using visiontransformer attention maps**  
  Matthis Dallain, Laurent Rodriguez, Laurent Udo Perrinet, Benoît Miramond  
  _2026-03-10_ · https://arxiv.org/abs/2603.09613v1  
  <details><summary>Abstract</summary>

  Human vision achieves remarkable perceptual performance while operating under strict metabolic constraints. A key ingredient is the selective attention mechanism, driven by rapid saccadic eye movements that constantly reposition the high-resolution fovea onto task-relevant locations, unlike conventional AI systems that process entire images with equal emphasis. Our work aims to draw inspiration from the human visual system to create smarter, more efficient image processing models. Using DINO, a self-supervised Vision Transformer that produces attention maps strikingly similar to human gaze patterns, we explore a saccade inspired method to focus the processing of information on key regions in visual space. To do so, we use the ImageNet dataset in a standard classification task and measure how each successive saccade affects the model's class scores. This selective-processing strategy preserves most of the full-image classification performance and can even outperform it in certain cases. By benchmarking against established saliency models built for human gaze prediction, we demonstrate that DINO provides superior fixation guidance for selecting informative regions. These findings highlight Vision Transformer attention as a promising basis for biologically inspired active vision and open new directions for efficient, neuromorphic visual processing.

  </details>



- **More than the Sum: Panorama-Language Models for Adverse Omni-Scenes**  
  Weijia Fan, Ruiping Liu, Jiale Wei, Yufan Chen, Junwei Zheng, Zichao Zeng, Jiaming Zhang, Qiufu Li, Linlin Shen, Rainer Stiefelhagen  
  _2026-03-10_ · https://arxiv.org/abs/2603.09573v1  
  <details><summary>Abstract</summary>

  Existing vision-language models (VLMs) are tailored for pinhole imagery, stitching multiple narrow field-of-view inputs to piece together a complete omni-scene understanding. Yet, such multi-view perception overlooks the holistic spatial and contextual relationships that a single panorama inherently preserves. In this work, we introduce the Panorama-Language Modeling (PLM)paradigm, a unified $360^\circ$ vision-language reasoning that is more than the sum of its pinhole counterparts. Besides, we present PanoVQA, a large-scale panoramic VQA dataset that involves adverse omni-scenes, enabling comprehensive reasoning under object occlusions and driving accidents. To establish a foundation for PLM, we develop a plug-and-play panoramic sparse attention module that allows existing pinhole-based VLMs to process equirectangular panoramas without retraining. Extensive experiments demonstrate that our PLM achieves superior robustness and holistic reasoning under challenging omni-scenes, yielding understanding greater than the sum of its narrow parts. Project page: https://github.com/InSAI-Lab/PanoVQA.

  </details>



- **GeoAlignCLIP: Enhancing Fine-Grained Vision-Language Alignment in Remote Sensing via Multi-Granular Consistency Learning**  
  Xiao Yang, Ronghao Fu, Zhuoran Duan, Zhiwen Lin, Xueyan Liu, Bo Yang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09566v1  
  <details><summary>Abstract</summary>

  Vision-language pretraining models have made significant progress in bridging remote sensing imagery with natural language. However, existing approaches often fail to effectively integrate multi-granular visual and textual information, relying primarily on global image-text alignment. This limitation hinders the model's ability to accurately capture fine-grained details in images, thus restricting its performance in complex, fine-grained tasks. To address this, we propose GeoAlignCLIP, a unified framework that achieves fine-grained alignment in remote sensing tasks by learning multi-granular semantic alignments and incorporating intra-modal consistency, enabling more precise visual-semantic alignment between image regions and text concepts. Additionally, we construct RSFG-100k, a fine-granular remote sensing dataset containing scene descriptions, region-level annotations, and challenging hard-negative samples, providing hierarchical supervision for model training. Extensive experiments conducted on multiple public remote-sensing benchmarks demonstrate that GeoAlignCLIP consistently outperforms existing RS-specific methods across diverse tasks, exhibiting more robust and accurate fine-grained vision-language alignment.

  </details>



- **ReTac-ACT: A State-Gated Vision-Tactile Fusion Transformer for Precision Assembly**  
  Minchi Ruan, LiangQing Zhou, Hongtong Li, Zongtao Wang, ZhaoMing Lu, Jianwei Zhang, Bin Fang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09565v1  
  <details><summary>Abstract</summary>

  Precision assembly requires sub-millimeter corrections in contact-rich "last-millimeter" regions where visual feedback fails due to occlusion from the end-effector and workpiece. We present ReTac-ACT (Reconstruction-enhanced Tactile ACT), a vision-tactile imitation learning policy that addresses this challenge through three synergistic mechanisms: (i) bidirectional cross-attention enabling reciprocal visuo-tactile feature enhancement before fusion, (ii) a proprioception-conditioned gating network that dynamically elevates tactile reliance when visual occlusion occurs, and (iii) a tactile reconstruction objective enforcing learning of manipulation-relevant contact information rather than generic visual textures. Evaluated on the standardized NIST Assembly Task Board M1 benchmark, ReTac-ACT achieves 90% peg-in-hole success, substantially outperforming vision-only and generalist baseline methods, and maintains 80% success at industrial-grade 0.1mm clearance. Ablation studies validate that each architectural component is indispensable. The ReTac-ACT codebase and a vision-tactile demonstration dataset covering various clearance levels with both visual and tactile features will be released to support reproducible research.

  </details>



- **GeoSolver: Scaling Test-Time Reasoning in Remote Sensing with Fine-Grained Process Supervision**  
  Lang Sun, Ronghao Fu, Zhuoran Duan, Haoran Liu, Xueyan Liu, Bo Yang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09551v1  
  <details><summary>Abstract</summary>

  While Vision-Language Models (VLMs) have significantly advanced remote sensing interpretation, enabling them to perform complex, step-by-step reasoning remains highly challenging. Recent efforts to introduce Chain-of-Thought (CoT) reasoning to this domain have shown promise, yet ensuring the visual faithfulness of these intermediate steps remains a critical bottleneck. To address this, we introduce GeoSolver, a novel framework that transitions remote sensing reasoning toward verifiable, process-supervised reinforcement learning. We first construct Geo-PRM-2M, a large-scale, token-level process supervision dataset synthesized via entropy-guided Monte Carlo Tree Search (MCTS) and targeted visual hallucination injection. Building upon this dataset, we train GeoPRM, a token-level process reward model (PRM) that provides granular faithfulness feedback. To effectively leverage these verification signals, we propose Process-Aware Tree-GRPO, a reinforcement learning algorithm that integrates tree-structured exploration with a faithfulness-weighted reward mechanism to precisely assign credit to intermediate steps. Extensive experiments demonstrate that our resulting model, GeoSolver-9B, achieves state-of-the-art performance across diverse remote sensing benchmarks. Crucially, GeoPRM unlocks robust Test-Time Scaling (TTS). Serving as a universal geospatial verifier, it seamlessly scales the performance of GeoSolver-9B and directly enhances general-purpose VLMs, highlighting its remarkable cross-model generalization.

  </details>



- **Memory-Guided View Refinement for Dynamic Human-in-the-loop EQA**  
  Xin Lu, Rui Li, Xun Huang, Weixin Li, Chuanqing Zhuang, Jiayuan Li, Zhengda Lu, Jun Xiao, Yunhong Wang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09541v1  
  <details><summary>Abstract</summary>

  Embodied Question Answering (EQA) has traditionally been evaluated in temporally stable environments where visual evidence can be accumulated reliably. However, in dynamic, human-populated scenes, human activities and occlusions introduce significant perceptual non-stationarity: task-relevant cues are transient and view-dependent, while a store-then-retrieve strategy over-accumulates redundant evidence and increases inference cost. This setting exposes two practical challenges for EQA agents: resolving ambiguity caused by viewpoint-dependent occlusions, and maintaining compact yet up-to-date evidence for efficient inference. To enable systematic study of this setting, we introduce DynHiL-EQA, a human-in-the-loop EQA dataset with two subsets: a Dynamic subset featuring human activities and temporal changes, and a Static subset with temporally stable observations. To address the above challenges, we present DIVRR (Dynamic-Informed View Refinement and Relevance-guided Adaptive Memory Selection), a training-free framework that couples relevance-guided view refinement with selective memory admission. By verifying ambiguous observations before committing them and retaining only informative evidence, DIVRR improves robustness under occlusions while preserving fast inference with compact memory. Extensive experiments on DynHiL-EQA and the established HM-EQA dataset demonstrate that DIVRR consistently improves over existing baselines in both dynamic and static settings while maintaining high inference efficiency.

  </details>



- **Towards Unified Multimodal Interleaved Generation via Group Relative Policy Optimization**  
  Ming Nie, Chunwei Wang, Jianhua Han, Hang Xu, Li Zhang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09538v1  
  <details><summary>Abstract</summary>

  Unified vision-language models have made significant progress in multimodal understanding and generation, yet they largely fall short in producing multimodal interleaved outputs, which is a crucial capability for tasks like visual storytelling and step-by-step visual reasoning. In this work, we propose a reinforcement learning-based post-training strategy to unlock this capability in existing unified models, without relying on large-scale multimodal interleaved datasets. We begin with a warm-up stage using a hybrid dataset comprising curated interleaved sequences and limited data for multimodal understanding and text-to-image generation, which exposes the model to interleaved generation patterns while preserving its pretrained capabilities. To further refine interleaved generation, we propose a unified policy optimization framework that extends Group Relative Policy Optimization (GRPO) to the multimodal setting. Our approach jointly models text and image generation within a single decoding trajectory and optimizes it with our novel hybrid rewards covering textual relevance, visual-text alignment, and structural fidelity. Additionally, we incorporate process-level rewards to provide step-wise guidance, enhancing training efficiency in complex multimodal tasks. Experiments on MMIE and InterleavedBench demonstrate that our approach significantly enhances the quality and coherence of multimodal interleaved generation.

  </details>



- **RESBev: Making BEV Perception More Robust**  
  Lifeng Zhuo, Kefan Jin, Zhe Liu, Hesheng Wang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09529v1  
  <details><summary>Abstract</summary>

  Bird's-eye-view (BEV) perception has emerged as a cornerstone of autonomous driving systems, providing a structured, ego-centric representation critical for downstream planning and control. However, real-world deployment faces challenges from sensor degradation and adversarial attacks, which can cause severe perceptual anomalies and ultimately compromise the safety of autonomous driving systems. To address this, we propose a resilient and plug-and-play BEV perception method, RESBev, which can be easily applied to existing BEV perception methods to enhance their robustness to diverse disturbances. Specifically, we reframe perception robustness as a latent semantic prediction problem. A latent world model is constructed to extract spatiotemporal correlations across sequential BEV observations, thereby learning the underlying BEV state transitions to predict clean BEV features for reconstructing corrupted observations. The proposed framework operates at the semantic feature level of the Lift-Splat-Shoot pipeline, enabling recovery that generalizes across both natural disturbances and adversarial attacks without modifying the underlying backbone. Extensive experiments on the nuScenes dataset demonstrate that, with few-shot fine-tuning, RESBev significantly improves the robustness of existing BEV perception models against various external disturbances and adversarial attacks.

  </details>



- **Beyond Short-Horizon: VQ-Memory for Robust Long-Horizon Manipulation in Non-Markovian Simulation Benchmarks**  
  Wang Honghui, Jing Zhi, Ao Jicong, Song Shiji, Li Xuelong, Huang Gao, Bai Chenjia  
  _2026-03-10_ · https://arxiv.org/abs/2603.09513v1  
  <details><summary>Abstract</summary>

  The high cost of collecting real-robot data has made robotic simulation a scalable platform for both evaluation and data generation. Yet most existing benchmarks concentrate on simple manipulation tasks such as pick-and-place, failing to capture the non-Markovian characteristics of real-world tasks and the complexity of articulated object interactions. To address this limitation, we present RuleSafe, a new articulated manipulation benchmark built upon a scalable LLM-aided simulation framework. RuleSafe features safes with diverse unlocking mechanisms, such as key locks, password locks, and logic locks, which require different multi-stage reasoning and manipulation strategies. These LLM-generated rules produce non-Markovian and long-horizon tasks that require temporal modeling and memory-based reasoning. We further propose VQ-Memory, a compact and structured temporal representation that uses vector-quantized variational autoencoders (VQ-VAEs) to encode past proprioceptive states into discrete latent tokens. This representation filters low-level noise while preserving high-level task-phase context, providing lightweight yet robust temporal cues that are compatible with existing Vision-Language-Action models (VLA). Extensive experiments on state-of-the-art VLA models and diffusion policies show that VQ-Memory consistently improves long-horizon planning, enhances generalization to unseen configurations, and enables more efficient manipulation with reduced computational cost. Project page: vqmemory.github.io

  </details>



- **Probing the Reliability of Driving VLMs: From Inconsistent Responses to Grounded Temporal Reasoning**  
  Chun-Peng Chang, Chen-Yu Wang, Holger Caesar, Alain Pagani  
  _2026-03-10_ · https://arxiv.org/abs/2603.09512v1  
  <details><summary>Abstract</summary>

  A reliable driving assistant should provide consistent responses based on temporally grounded reasoning derived from observed information. In this work, we investigate whether Vision-Language Models (VLMs), when applied as driving assistants, can response consistantly and understand how present observations shape future outcomes, or whether their outputs merely reflect patterns memorized during training without temporally grounded reasoning. While recent efforts have integrated VLMs into autonomous driving, prior studies typically emphasize scene understanding and instruction generation, implicitly assuming that strong visual interpretation naturally enables consistant future reasoning and thus ensures reliable decision-making, a claim we critically examine. We focus on two major challenges limiting VLM reliability in this setting: response inconsistency, where minor input perturbations yield different answers or, in some cases, responses degenerate toward near-random guessing, and limited temporal reasoning, in which models fail to reason and align sequential events from current observations, often resulting in incorrect or even contradictory responses. Moreover, we find that models with strong visual understanding do not necessarily perform best on tasks requiring temporal reasoning, indicating a tendency to over-rely on pretrained patterns rather than modeling temporal dynamics. To address these issues, we adopt existing evaluation methods and introduce FutureVQA, a human-annotated benchmark dataset specifically designed to assess future scene reasoning. In addition, we propose a simple yet effective self-supervised tuning approach with chain-of-thought reasoning that improves both consistency and temporal reasoning without requiring temporal labels.

  </details>



- **StyleVLA: Driving Style-Aware Vision Language Action Model for Autonomous Driving**  
  Yuan Gao, Dengyuan Hua, Mattia Piccinini, Finn Rasmus Schäfer, Korbinian Moller, Lin Li, Johannes Betz  
  _2026-03-10_ · https://arxiv.org/abs/2603.09482v1  
  <details><summary>Abstract</summary>

  Vision Language Models (VLMs) bridge visual perception and linguistic reasoning. In Autonomous Driving (AD), this synergy has enabled Vision Language Action (VLA) models, which translate high-level multimodal understanding into driving behaviors, typically represented as future trajectories. However, existing VLA models mainly generate generic collision-free trajectories. Beyond collision avoidance, adapting to diverse driving styles (e.g., sporty, comfortable) is essential for personalized driving. Moreover, many methods treat trajectory generation as naive token prediction, which can produce kinematically infeasible actions. To address these limitations, we present StyleVLA, a physics-informed VLA framework for generating diverse and physically plausible driving behaviors. We introduce a hybrid loss that combines a kinematic consistency constraint with a continuous regression head to improve trajectory feasibility. To train StyleVLA, built on Qwen3-VL-4B, we construct a large-scale instruction dataset with over 1.2k scenarios, 76k Bird's Eye View (BEV) samples, and 42k First Person View (FPV) samples, with ground-truth trajectories for five driving styles and natural-language instructions. Experiments show that our 4B-parameter StyleVLA significantly outperforms proprietary models (e.g., Gemini-3-Pro) and state-of-the-art VLA models. Using a composite driving score measuring success rate, physical feasibility, and style adherence, StyleVLA achieves 0.55 on BEV and 0.51 on FPV, versus 0.32 and 0.35 for Gemini-3-Pro. These results show that a specialized, physics-informed, lightweight model can surpass closed-source models on domain-specific tasks.

  </details>



- **OmniEarth: A Benchmark for Evaluating Vision-Language Models in Geospatial Tasks**  
  Ronghao Fu, Haoran Liu, Weijie Zhang, Zhiwen Lin, Xiao Yang, Peng Zhang, Bo Yang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09471v1  
  <details><summary>Abstract</summary>

  Vision-Language Models (VLMs) have demonstrated effective perception and reasoning capabilities on general-domain tasks, leading to growing interest in their application to Earth observation. However, a systematic benchmark for comprehensively evaluating remote sensing vision-language models (RSVLMs) remains lacking. To address this gap, we introduce OmniEarth, a benchmark for evaluating RSVLMs under realistic Earth observation scenarios. OmniEarth organizes tasks along three capability dimensions: perception, reasoning, and robustness. It defines 28 fine-grained tasks covering multi-source sensing data and diverse geospatial contexts. The benchmark supports two task formulations: multiple-choice VQA and open-ended VQA. The latter includes pure text outputs for captioning tasks, bounding box outputs for visual grounding tasks, and mask outputs for segmentation tasks. To reduce linguistic bias and examine whether model predictions rely on visual evidence, OmniEarth adopts a blind test protocol and a quintuple semantic consistency requirement. OmniEarth includes 9,275 carefully quality-controlled images, including proprietary satellite imagery from Jilin-1 (JL-1), along with 44,210 manually verified instructions. We conduct a systematic evaluation of contrastive learning-based models, general closed-source and open-source VLMs, as well as RSVLMs. Results show that existing VLMs still struggle with geospatially complex tasks, revealing clear gaps that need to be addressed for remote sensing applications. OmniEarth is publicly available at https://huggingface.co/datasets/sjeeudd/OmniEarth.

  </details>



- **The Patrologia Graeca Corpus: OCR, Annotation, and Open Release of Noisy Nineteenth-Century Polytonic Greek Editions**  
  Chahan Vidal-Gorène, Bastien Kindt  
  _2026-03-10_ · https://arxiv.org/abs/2603.09470v1  
  <details><summary>Abstract</summary>

  We present the Patrologia Graeca Corpus, the first large-scale open OCR and linguistic resource for nineteenthcentury editions of Ancient Greek. The collection covers the remaining undigitized volumes of the Patrologia Graeca (PG), printed in complex bilingual (Greek-Latin) layouts and characterized by highly degraded polytonic Greek typography. Through a dedicated pipeline combining YOLO-based layout detection and CRNN-based text recognition, we achieve a character error rate (CER) of 1.05% and a word error rate (WER) of 4.69%, largely outperforming existing OCR systems for polytonic Greek. The resulting corpus contains around six million lemmatized and part-of-speech tagged tokens, aligned with full OCR and layout annotations. Beyond its philological value, this corpus establishes a new benchmark for OCR on noisy polytonic Greek and provides training material for future models, including LLMs.

  </details>



- **Stein Variational Ergodic Surface Coverage with SE(3) Constraints**  
  Jiayun Li, Yufeng Jin, Sangli Teng, Dejian Gong, Georgia Chalvatzaki  
  _2026-03-10_ · https://arxiv.org/abs/2603.09458v1  
  <details><summary>Abstract</summary>

  Surface manipulation tasks require robots to generate trajectories that comprehensively cover complex 3D surfaces while maintaining precise end-effector poses. Existing ergodic trajectory optimization (TO) methods demonstrate success in coverage tasks, while struggling with point-cloud targets due to the nonconvex optimization landscapes and the inadequate handling of SE(3) constraints in sampling-as-optimization (SAO) techniques. In this work, we introduce a preconditioned SE(3) Stein Variational Gradient Descent (SVGD) approach for SAO ergodic trajectory generation. Our proposed approach comprises multiple innovations. First, we reformulate point-cloud ergodic coverage as a manifold-aware sampling problem. Second, we derive SE(3)-specific SVGD particle updates, and, third, we develop a preconditioner to accelerate TO convergence. Our sampling-based framework consistently identifies superior local optima compared to strong optimization-based and SAO baselines while preserving the SE(3) geometric structure. Experiments on a 3D point-cloud surface coverage benchmark and robotic surface drawing tasks demonstrate that our method achieves superior coverage quality with tractable computation in our setting relative to existing TO and SAO approaches, and is validated in real-world robot experiments.

  </details>



- **MetaDAT: Generalizable Trajectory Prediction via Meta Pre-training and Data-Adaptive Test-Time Updating**  
  Yuning Wang, Pu Zhang, Yuan He, Ke Wang, Jianru Xue  
  _2026-03-10_ · https://arxiv.org/abs/2603.09419v1  
  <details><summary>Abstract</summary>

  Existing trajectory prediction methods exhibit significant performance degradation under distribution shifts during test time. Although test-time training techniques have been explored to enable adaptation, current approaches rely on an offline pre-trained predictor that lacks online learning flexibility. Moreover, they depend on fixed online model updating rules that do not accommodate the specific characteristics of test data. To address these limitations, we first propose a meta-learning framework to directly optimize the predictor for fast and accurate online adaptation, which performs bi-level optimization on the performance of simulated test-time adaptation tasks during pre-training. Furthermore, at test time, we introduce a data-adaptive model updating mechanism that dynamically adjusts the predefined learning rates and updating frequencies based on online partial derivatives and hard sample selection. This mechanism enables the online learning rate to suit the test data, and focuses on informative hard samples to enhance efficiency. Experiments are conducted on various challenging cross-dataset distribution shift scenarios, including nuScenes, Lyft, and Waymo. Results demonstrate that our method achieves superior adaptation accuracy, surpassing state-of-the-art test-time training methods for trajectory prediction. Additionally, our method excels under suboptimal learning rates and high FPS demands, showcasing its robustness and practicality.

  </details>



- **CIGPose: Causal Intervention Graph Neural Network for Whole-Body Pose Estimation**  
  Bohao Li, Zhicheng Cao, Huixian Li, Yangming Guo  
  _2026-03-10_ · https://arxiv.org/abs/2603.09418v1  
  <details><summary>Abstract</summary>

  State-of-the-art whole-body pose estimators often lack robustness, producing anatomically implausible predictions in challenging scenes. We posit this failure stems from spurious correlations learned from visual context, a problem we formalize using a Structural Causal Model (SCM). The SCM identifies visual context as a confounder that creates a non-causal backdoor path, corrupting the model's reasoning. We introduce the Causal Intervention Graph Pose (CIGPose) framework to address this by approximating the true causal effect between visual evidence and pose. The core of CIGPose is a novel Causal Intervention Module: it first identifies confounded keypoint representations via predictive uncertainty and then replaces them with learned, context-invariant canonical embeddings. These deconfounded embeddings are processed by a hierarchical graph neural network that reasons over the human skeleton at both local and global semantic levels to enforce anatomical plausibility. Extensive experiments show CIGPose achieves a new state-of-the-art on COCO-WholeBody. Notably, our CIGPose-x model achieves 67.0\% AP, surpassing prior methods that rely on extra training data. With the additional UBody dataset, CIGPose-x is further boosted to 67.5\% AP, demonstrating superior robustness and data efficiency. The codes and models are publicly available at https://github.com/53mins/CIGPose.

  </details>



- **YOLO-NAS-Bench: A Surrogate Benchmark with Self-Evolving Predictors for YOLO Architecture Search**  
  Zhe Li, Xiaoyu Ding, Jiaxin Zheng, Yongtao Wang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09405v1  
  <details><summary>Abstract</summary>

  Neural Architecture Search (NAS) for object detection is severely bottlenecked by high evaluation cost, as fully training each candidate YOLO architecture on COCO demands days of GPU time. Meanwhile, existing NAS benchmarks largely target image classification, leaving the detection community without a comparable benchmark for NAS evaluation. To address this gap, we introduce YOLO-NAS-Bench, the first surrogate benchmark tailored to YOLO-style detectors. YOLO-NAS-Bench defines a search space spanning channel width, block depth, and operator type across both backbone and neck, covering the core modules of YOLOv8 through YOLO12. We sample 1,000 architectures via random, stratified, and Latin Hypercube strategies, train them on COCO-mini, and build a LightGBM surrogate predictor. To sharpen the predictor in the high-performance regime most relevant to NAS, we propose a Self-Evolving Mechanism that progressively aligns the predictor's training distribution with the high-performance frontier, by using the predictor itself to discover and evaluate informative architectures in each iteration. This method grows the pool to 1,500 architectures and raises the ensemble predictor's R2 from 0.770 to 0.815 and Sparse Kendall Tau from 0.694 to 0.752, demonstrating strong predictive accuracy and ranking consistency. Using the final predictor as the fitness function for evolutionary search, we discover architectures that surpass all official YOLOv8-YOLO12 baselines at comparable latency on COCO-mini, confirming the predictor's discriminative power for top-performing detection architectures.

  </details>



- **ICDAR 2025 Competition on End-to-End Document Image Machine Translation Towards Complex Layouts**  
  Yaping Zhang, Yupu Liang, Zhiyang Zhang, Zhiyuan Chen, Lu Xiang, Yang Zhao, Yu Zhou, Chengqing Zong  
  _2026-03-10_ · https://arxiv.org/abs/2603.09392v1  
  <details><summary>Abstract</summary>

  Document Image Machine Translation (DIMT) seeks to translate text embedded in document images from one language to another by jointly modeling both textual content and page layout, bridging optical character recognition (OCR) and natural language processing (NLP). The DIMT 2025 Challenge advances research on end-to-end document image translation, a rapidly evolving area within multimodal document understanding. The competition features two tracks, OCR-free and OCR-based, each with two subtasks for small (less than 1B parameters) and large (greater than 1B parameters) models. Participants submit a single unified DIMT system, with the option to incorporate provided OCR transcripts. Running from December 10, 2024 to April 20, 2025, the competition attracted 69 teams and 27 valid submissions in total. Track 1 had 34 teams and 13 valid submissions, while Track 2 had 35 teams and 14 valid submissions. In this report, we present the challenge motivation, dataset construction, task definitions, evaluation protocol, and a summary of results. Our analysis shows that large-model approaches establish a promising new paradigm for translating complex-layout document images and highlight substantial opportunities for future research.

  </details>



- **EventVGGT: Exploring Cross-Modal Distillation for Consistent Event-based Depth Estimation**  
  Yinrui Ren, Jinjing Zhu, Kanghao Chen, Zhuoxiao Li, Jing Ou, Zidong Cao, Tongyan Hua, Peilun Shi, Yingchun Fu, Wufan Zhao, et al.  
  _2026-03-10_ · https://arxiv.org/abs/2603.09385v1  
  <details><summary>Abstract</summary>

  Event cameras offer superior sensitivity to high-speed motion and extreme lighting, making event-based monocular depth estimation a promising approach for robust 3D perception in challenging conditions. However, progress is severely hindered by the scarcity of dense depth annotations. While recent annotation-free approaches mitigate this by distilling knowledge from Vision Foundation Models (VFMs), a critical limitation persists: they process event streams as independent frames. By neglecting the inherent temporal continuity of event data, these methods fail to leverage the rich temporal priors encoded in VFMs, ultimately yielding temporally inconsistent and less accurate depth predictions. To address this, we introduce EventVGGT, a novel framework that explicitly models the event stream as a coherent video sequence. To the best of our knowledge, we are the first to distill spatio-temporal and multi-view geometric priors from the Visual Geometry Grounded Transformer (VGGT) into the event domain. We achieve this via a comprehensive tri-level distillation strategy: (i) Cross-Modal Feature Mixture (CMFM) bridges the modality gap at the output level by fusing RGB and event features to generate auxiliary depth predictions; (ii) Spatio-Temporal Feature Distillation (STFD) distills VGGT's powerful spatio-temporal representations at the feature level; and (iii) Temporal Consistency Distillation (TCD) enforces cross-frame coherence at the temporal level by aligning inter-frame depth changes. Extensive experiments demonstrate that EventVGGT consistently outperforms existing methods -- reducing the absolute mean depth error at 30m by over 53\% on EventScape (from 2.30 to 1.06) -- while exhibiting robust zero-shot generalization on the unseen DENSE and MVSEC datasets.

  </details>



- **SinGeo: Unlock Single Model's Potential for Robust Cross-View Geo-Localization**  
  Yang Chen, Xieyuanli Chen, Junxiang Li, Jie Tang, Tao Wu  
  _2026-03-10_ · https://arxiv.org/abs/2603.09377v1  
  <details><summary>Abstract</summary>

  Robust cross-view geo-localization (CVGL) remains challenging despite the surge in recent progress. Existing methods still rely on field-of-view (FoV)-specific training paradigms, where models are optimized under a fixed FoV but collapse when tested on unseen FoVs and unknown orientations. This limitation necessitates deploying multiple models to cover diverse variations. Although studies have explored dynamic FoV training by simply randomizing FoVs, they failed to achieve robustness across diverse conditions -- implicitly assuming all FoVs are equally difficult. To address this gap, we present SinGeo, a simple yet powerful framework that enables a single model to realize robust cross-view geo-localization without additional modules or explicit transformations. SinGeo employs a dual discriminative learning architecture that enhances intra-view discriminability within both ground and satellite branches, and is the first to introduce a curriculum learning strategy to achieve robust CVGL. Extensive evaluations on four benchmark datasets reveal that SinGeo sets state-of-the-art (SOTA) results under diverse conditions, and notably outperforms methods specifically trained for extreme FoVs. Beyond superior performance, SinGeo also exhibits cross-architecture transferability. Furthermore, we propose a consistency evaluation method to quantitatively assess model stability under varying views, providing an explainable perspective for understanding and advancing robustness in future CVGL research. Codes will be available upon acceptance.

  </details>



- **Evidential Perfusion Physics-Informed Neural Networks with Residual Uncertainty Quantification**  
  Junhyeok Lee, Minseo Choi, Han Jang, Young Hun Jeon, Heeseong Eum, Joon Jang, Chul-Ho Sohn, Kyu Sung Choi  
  _2026-03-10_ · https://arxiv.org/abs/2603.09359v1  
  <details><summary>Abstract</summary>

  Physics-informed neural networks (PINNs) have shown promise in addressing the ill-posed deconvolution problem in computed tomography perfusion (CTP) imaging for acute ischemic stroke assessment. However, existing PINN-based approaches remain deterministic and do not quantify uncertainty associated with violations of physics constraints, limiting reliability assessment. We propose Evidential Perfusion Physics-Informed Neural Networks (EPPINN), a framework that integrates evidential deep learning with physics-informed modeling to enable uncertainty-aware perfusion parameter estimation. EPPINN models arterial input, tissue concentration, and perfusion parameters using coordinate-based networks, and places a Normal--Inverse--Gamma distribution over the physics residual to characterize voxel-wise aleatoric and epistemic uncertainty in physics consistency without requiring Bayesian sampling or ensemble inference. The framework further incorporates physiologically constrained parameterization and stabilization strategies to promote robust per-case optimization. We evaluate EPPINN on digital phantom data, the ISLES 2018 benchmark, and a clinical cohort. On the evaluated datasets, EPPINN achieves lower normalized mean absolute error than classical deconvolution and PINN baselines, particularly under sparse temporal sampling and low signal-to-noise conditions, while providing conservative uncertainty estimates with high empirical coverage. On clinical data, EPPINN attains the highest voxel-level and case-level infarct-core detection sensitivity. These results suggest that evidential physics-informed learning can improve both accuracy and reliability of CTP analysis for time-critical stroke assessment.

  </details>



- **Robust Provably Secure Image Steganography via Latent Iterative Optimization**  
  Yanan Li, Zixuan Wang, Qiyang Xiao, Yanzhen Ren  
  _2026-03-10_ · https://arxiv.org/abs/2603.09348v1  
  <details><summary>Abstract</summary>

  We propose a robust and provably secure image steganography framework based on latent-space iterative optimization. Within this framework, the receiver treats the transmitted image as a fixed reference and iteratively refines a latent variable to minimize the reconstruction error, thereby improving message extraction accuracy. Unlike prior methods, our approach preserves the provable security of the embedding while markedly enhancing robustness under various compression and image processing scenarios. On benchmark datasets, the experimental results demonstrate that the proposed iterative optimization not only improves robustness against image compression while preserving provable security, but can also be applied as an independent module to further reinforce robustness in other provably secure steganographic schemes. This highlights the practicality and promise of latent-space optimization for building reliable, robust, and secure steganographic systems.

  </details>



- **Beyond Scaling: Assessing Strategic Reasoning and Rapid Decision-Making Capability of LLMs in Zero-sum Environments**  
  Yang Li, Xing Chen, Yutao Liu, Gege Qi, Yanxian BI, Zizhe Wang, Yunjian Zhang, Yao Zhu  
  _2026-03-10_ · https://arxiv.org/abs/2603.09337v1  
  <details><summary>Abstract</summary>

  Large Language Models (LLMs) have achieved strong performance on static reasoning benchmarks, yet their effectiveness as interactive agents operating in adversarial, time-sensitive environments remains poorly understood. Existing evaluations largely treat reasoning as a single-shot capability, overlooking the challenges of opponent-aware decision-making, temporal constraints, and execution under pressure. This paper introduces Strategic Tactical Agent Reasoning (STAR) Benchmark, a multi-agent evaluation framework that assesses LLMs through 1v1 zero-sum competitive interactions, framing reasoning as an iterative, adaptive decision-making process. STAR supports both turn-based and real-time settings, enabling controlled analysis of long-horizon strategic planning and fast-paced tactical execution within a unified environment. Built on a modular architecture with a standardized API and fully implemented execution engine, STAR facilitates reproducible evaluation and flexible task customization. To move beyond binary win-loss outcomes, we introduce a Strategic Evaluation Suite that assesses not only competitive success but also the quality of strategic behavior, such as execution efficiency and outcome stability. Extensive pairwise evaluations reveal a pronounced strategy-execution gap: while reasoning-intensive models dominate turn-based settings, their inference latency often leads to inferior performance in real-time scenarios, where faster instruction-tuned models prevail. These results show that strategic intelligence in interactive environments depends not only on reasoning depth, but also on the ability to translate plans into timely actions, positioning STAR as a principled benchmark for studying this trade-off in competitive, dynamic settings.

  </details>



- **OddGridBench: Exposing the Lack of Fine-Grained Visual Discrepancy Sensitivity in Multimodal Large Language Models**  
  Tengjin Weng, Wenhao Jiang, Jingyi Wang, Ming Li, Lin Ma, Zhong Ming  
  _2026-03-10_ · https://arxiv.org/abs/2603.09326v1  
  <details><summary>Abstract</summary>

  Multimodal large language models (MLLMs) have achieved remarkable performance across a wide range of vision language tasks. However, their ability in low-level visual perception, particularly in detecting fine-grained visual discrepancies, remains underexplored and lacks systematic analysis. In this work, we introduce OddGridBench, a controllable benchmark for evaluating the visual discrepancy sensitivity of MLLMs. OddGridBench comprises over 1,400 grid-based images, where a single element differs from all others by one or multiple visual attributes such as color, size, rotation, or position. Experiments reveal that all evaluated MLLMs, including open-source families such as Qwen3-VL and InternVL3.5, and proprietary systems like Gemini-2.5-Pro and GPT-5, perform far below human levels in visual discrepancy detection. We further propose OddGrid-GRPO, a reinforcement learning framework that integrates curriculum learning and distance-aware reward. By progressively controlling the difficulty of training samples and incorporating spatial proximity constraints into the reward design, OddGrid-GRPO significantly enhances the model's fine-grained visual discrimination ability. We hope OddGridBench and OddGrid-GRPO will lay the groundwork for advancing perceptual grounding and visual discrepancy sensitivity in multimodal intelligence. Code and dataset are available at https://wwwtttjjj.github.io/OddGridBench/.

  </details>



- **SpaceSense-Bench: A Large-Scale Multi-Modal Benchmark for Spacecraft Perception and Pose Estimation**  
  Aodi Wu, Jianhong Zuo, Zeyuan Zhao, Xubo Luo, Ruisuo Wang, Xue Wan  
  _2026-03-10_ · https://arxiv.org/abs/2603.09320v1  
  <details><summary>Abstract</summary>

  Autonomous space operations such as on-orbit servicing and active debris removal demand robust part-level semantic understanding and precise relative navigation of target spacecraft, yet collecting large-scale real data in orbit remains impractical due to cost and access constraints. Existing synthetic datasets, moreover, suffer from limited target diversity, single-modality sensing, and incomplete ground-truth annotations. We present \textbf{SpaceSense-Bench}, a large-scale multi-modal benchmark for spacecraft perception encompassing 136~satellite models with approximately 70~GB of data. Each frame provides time-synchronized 1024$\times$1024 RGB images, millimeter-precision depth maps, and 256-beam LiDAR point clouds, together with dense 7-class part-level semantic labels at both the pixel and point level as well as accurate 6-DoF pose ground truth. The dataset is generated through a high-fidelity space simulation built in Unreal Engine~5 and a fully automated pipeline covering data acquisition, multi-stage quality control, and conversion to mainstream formats. We benchmark five representative tasks (object detection, 2D semantic segmentation, RGB--LiDAR fusion-based 3D point cloud segmentation, monocular depth estimation, and orientation estimation) and identify two key findings: (i)~perceiving small-scale components (\emph{e.g.}, thrusters and omni-antennas) and generalizing to entirely unseen spacecraft in a zero-shot setting remain critical bottlenecks for current methods, and (ii)~scaling up the number of training satellites yields substantial performance gains on novel targets, underscoring the value of large-scale, diverse datasets for space perception research. The dataset, code, and toolkit are publicly available at https://github.com/wuaodi/SpaceSense-Bench.

  </details>



- **CLoE: Expert Consistency Learning for Missing Modality Segmentation**  
  Xinyu Tong, Meihua Zhou, Bowu Fan, Haitao Li  
  _2026-03-10_ · https://arxiv.org/abs/2603.09316v1  
  <details><summary>Abstract</summary>

  Multimodal medical image segmentation often faces missing modalities at inference, which induces disagreement among modality experts and makes fusion unstable, particularly on small foreground structures. We propose Consistency Learning of Experts (CLoE), a consistency-driven framework for missing-modality segmentation that preserves strong performance when all modalities are available. CLoE formulates robustness as decision-level expert consistency control and introduces a dual-branch Expert Consistency Learning objective. Modality Expert Consistency enforces global agreement among expert predictions to reduce case-wise drift under partial inputs, while Region Expert Consistency emphasizes agreement on clinically critical foreground regions to avoid background-dominated regularization. We further map consistency scores to modality reliability weights using a lightweight gating network, enabling reliability-aware feature recalibration before fusion. Extensive experiments on BraTS 2020 and MSD Prostate demonstrate that CLoE outperforms state-of-the-art methods in incomplete multimodal segmentation, while exhibiting strong cross-dataset generalization and improving robustness on clinically critical structures.

  </details>



- **IntroSVG: Learning from Rendering Feedback for Text-to-SVG Generation via an Introspective Generator-Critic Framework**  
  Feiyu Wang, Jiayuan Yang, Zhiyuan Zhao, Da Zhang, Bingyu Li, Peng Liu, Junyu Gao  
  _2026-03-10_ · https://arxiv.org/abs/2603.09312v1  
  <details><summary>Abstract</summary>

  Scalable Vector Graphics (SVG) are central to digital design due to their inherent scalability and editability. Despite significant advancements in content generation enabled by Visual Language Models (VLMs), existing text-to-SVG generation methods are limited by a core challenge: the autoregressive training process does not incorporate visual perception of the final rendered image, which fundamentally constrains generation quality. To address this limitation, we propose an Introspective SVG Generation Framework (IntroSVG). At its core, the framework instantiates a unified VLM that operates in a closed loop, assuming dual roles of both generator and critic. Specifically, through Supervised Fine-Tuning (SFT), the model learns to draft SVGs and to provide feedback on their rendered outputs; moreover, we systematically convert early-stage failures into high-quality error-correction training data, thereby enhancing model robustness. Subsequently, we leverage a high-capacity teacher VLM to construct a preference dataset and further align the generator's policy through Direct Preference Optimization (DPO). During inference, the optimized generator and critic operate collaboratively in an iterative "generate-review-refine" cycle, starting from imperfect intermediate drafts to autonomously improve output quality. Experimental results demonstrate that our method achieves state-of-the-art performance across several key evaluation metrics, generating SVGs with more complex structures, stronger semantic alignment, and greater editability. These results corroborate the effectiveness of incorporating explicit visual feedback into the generation loop.

  </details>



- **See, Plan, Rewind: Progress-Aware Vision-Language-Action Models for Robust Robotic Manipulation**  
  Tingjun Dai, Mingfei Han, Tingwen Du, Zhiheng Liu, Zhihui Li, Salman Khan, Jun Yu, Xiaojun Chang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09292v1  
  <details><summary>Abstract</summary>

  Measurement of task progress through explicit, actionable milestones is critical for robust robotic manipulation. This progress awareness enables a model to ground its current task status, anticipate verifiable intermediate states, and detect and recover from failures when progress stalls. To embody this capability, we introduce See, Plan, Rewind (SPR), a progress-aware vision-language-action framework that dynamically grounds language instructions into a sequence of spatial subgoals. SPR operates through a continuous core cycle, Seeing the current state and upcoming milestone, Planning a trajectory towards the next 2D waypoint, and Rewinding to a recoverable state upon failure by monitoring progress against the expected sequence. This closed-loop approach enables robust error correction without requiring additional training data or auxiliary models. Extensive experiments demonstrate the framework's effectiveness, generalization and robustness: SPR outperforms the MolmoAct baseline by 5\% on the LIBERO benchmark. On the challenging LIBERO-Plus benchmark with unseen instructions and initial states, SPR achieves state-of-the-art robustness with the smallest performance drop, surpassing OpenVLA-OFT and UniVLA, demonstrating superior out-of-distribution robustness.

  </details>



- **DenoiseSplat: Feed-Forward Gaussian Splatting for Noisy 3D Scene Reconstruction**  
  Fuzhen Jiang, Zhuoran Li, Yinlin Zhang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09291v1  
  <details><summary>Abstract</summary>

  3D scene reconstruction and novel-view synthesis are fundamental for VR, robotics, and content creation. However, most NeRF and 3D Gaussian Splatting pipelines assume clean inputs and degrade under real noise and artifacts. We therefore propose DenoiseSplat, a feed-forward 3D Gaussian splatting method for noisy multi-view images. We build a large-scale, scene-consistent noisy--clean benchmark on RE10K by injecting Gaussian, Poisson, speckle, and salt-and-pepper noise with controlled intensities. With a lightweight MVSplat-style feed-forward backbone, we train end-to-end using only clean 2D renderings as supervision and no 3D ground truth. On noisy RE10K, DenoiseSplat outperforms vanilla MVSplat and a strong two-stage baseline (IDF + MVSplat) in PSNR/SSIM and LPIPS across noise types and levels.

  </details>


