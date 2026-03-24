# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-03-24 07:17 UTC_

Total papers shown: **50**


---

- **3D-Layout-R1: Structured Reasoning for Language-Instructed Spatial Editing**  
  Haoyu Zhen, Xiaolong Li, Yilin Zhao, Han Zhang, Sifei Liu, Kaichun Mo, Chuang Gan, Subhashree Radhakrishnan  
  _2026-03-23_ · https://arxiv.org/abs/2603.22279v1  
  <details><summary>Abstract</summary>

  Large Language Models (LLMs) and Vision Language Models (VLMs) have shown impressive reasoning abilities, yet they struggle with spatial understanding and layout consistency when performing fine-grained visual editing. We introduce a Structured Reasoning framework that performs text-conditioned spatial layout editing via scene-graph reasoning. Given an input scene graph and a natural-language instruction, the model reasons over the graph to generate an updated scene graph that satisfies the text condition while maintaining spatial coherence. By explicitly guiding the reasoning process through structured relational representations, our approach improves both interpretability and control over spatial relationships. We evaluate our method on a new text-guided layout editing benchmark encompassing sorting, spatial alignment, and room-editing tasks. Our training paradigm yields an average 15% improvement in IoU and 25% reduction in center-distance error compared to Chain of Thought Fine-tuning (CoT-SFT) and vanilla GRPO baselines. Compared to SOTA zero-shot LLMs, our best models achieve up to 20% higher mIoU, demonstrating markedly improved spatial precision.

  </details>



- **GenOpticalFlow: A Generative Approach to Unsupervised Optical Flow Learning**  
  Yixuan Luo, Feng Qiao, Zhexiao Xiong, Yanjing Li, Nathan Jacobs  
  _2026-03-23_ · https://arxiv.org/abs/2603.22270v1  
  <details><summary>Abstract</summary>

  Optical flow estimation is a fundamental problem in computer vision, yet the reliance on expensive ground-truth annotations limits the scalability of supervised approaches. Although unsupervised and semi-supervised methods alleviate this issue, they often suffer from unreliable supervision signals based on brightness constancy and smoothness assumptions, leading to inaccurate motion estimation in complex real-world scenarios. To overcome these limitations, we introduce \textbf{\modelname}, a novel framework that synthesizes large-scale, perfectly aligned frame--flow data pairs for supervised optical flow training without human annotations. Specifically, our method leverages a pre-trained depth estimation network to generate pseudo optical flows, which serve as conditioning inputs for a next-frame generation model trained to produce high-fidelity, pixel-aligned subsequent frames. This process enables the creation of abundant, high-quality synthetic data with precise motion correspondence. Furthermore, we propose an \textit{inconsistent pixel filtering} strategy that identifies and removes unreliable pixels in generated frames, effectively enhancing fine-tuning performance on real-world datasets. Extensive experiments on KITTI2012, KITTI2015, and Sintel demonstrate that \textbf{\modelname} achieves competitive or superior results compared to existing unsupervised and semi-supervised approaches, highlighting its potential as a scalable and annotation-free solution for optical flow learning. We will release our code upon acceptance.

  </details>



- **UniDex: A Robot Foundation Suite for Universal Dexterous Hand Control from Egocentric Human Videos**  
  Gu Zhang, Qicheng Xu, Haozhe Zhang, Jianhan Ma, Long He, Yiming Bao, Zeyu Ping, Zhecheng Yuan, Chenhao Lu, Chengbo Yuan, et al.  
  _2026-03-23_ · https://arxiv.org/abs/2603.22264v1  
  <details><summary>Abstract</summary>

  Dexterous manipulation remains challenging due to the cost of collecting real-robot teleoperation data, the heterogeneity of hand embodiments, and the high dimensionality of control. We present UniDex, a robot foundation suite that couples a large-scale robot-centric dataset with a unified vision-language-action (VLA) policy and a practical human-data capture setup for universal dexterous hand control. First, we construct UniDex-Dataset, a robot-centric dataset over 50K trajectories across eight dexterous hands (6--24 DoFs), derived from egocentric human video datasets. To transform human data into robot-executable trajectories, we employ a human-in-the-loop retargeting procedure to align fingertip trajectories while preserving plausible hand-object contacts, and we operate on explicit 3D pointclouds with human hands masked to narrow kinematic and visual gaps. Second, we introduce the Function-Actuator-Aligned Space (FAAS), a unified action space that maps functionally similar actuators to shared coordinates, enabling cross-hand transfer. Leveraging FAAS as the action parameterization, we train UniDex-VLA, a 3D VLA policy pretrained on UniDex-Dataset and finetuned with task demonstrations. In addition, we build UniDex-Cap, a simple portable capture setup that records synchronized RGB-D streams and human hand poses and converts them into robot-executable trajectories to enable human-robot data co-training that reduces reliance on costly robot demonstrations. On challenging tool-use tasks across two different hands, UniDex-VLA achieves 81% average task progress and outperforms prior VLA baselines by a large margin, while exhibiting strong spatial, object, and zero-shot cross-hand generalization. Together, UniDex-Dataset, UniDex-VLA, and UniDex-Cap provide a scalable foundation suite for universal dexterous manipulation.

  </details>



- **EgoGroups: A Benchmark For Detecting Social Groups of People in the Wild**  
  Jeffri Murrugarra-Llerena, Pranav Chitale, Zicheng Liu, Kai Ao, Yujin Ham, Guha Balakrishnan, Paola Cascante-Bonilla  
  _2026-03-23_ · https://arxiv.org/abs/2603.22249v1  
  <details><summary>Abstract</summary>

  Social group detection, or the identification of humans involved in reciprocal interpersonal interactions (e.g., family members, friends, and customers and merchants), is a crucial component of social intelligence needed for agents transacting in the world. The few existing benchmarks for social group detection are limited by low scene diversity and reliance on third-person camera sources (e.g., surveillance footage). Consequently, these benchmarks generally lack real-world evaluation on how groups form and evolve in diverse cultural contexts and unconstrained settings. To address this gap, we introduce EgoGroups, a first-person view dataset that captures social dynamics in cities around the world. EgoGroups spans 65 countries covering low, medium, and high-crowd settings under four weather/time-of-day conditions. We include dense human annotations for person and social groups, along with rich geographic and scene metadata. Using this dataset, we performed an extensive evaluation of state-of-the-art VLM/LLMs and supervised models on their group detection capabilities. We found several interesting findings, including VLMs and LLMs can outperform supervised baselines in a zero-shot setting, while crowd density and cultural regions clearly influence model performance.

  </details>



- **Riverine Land Cover Mapping through Semantic Segmentation of Multispectral Point Clouds**  
  Sopitta Thurachen, Josef Taher, Matti Lehtomäki, Leena Matikainen, Linnea Blåfield, Mikel Calle Navarro, Antero Kukko, Tomi Westerlund, Harri Kaartinen  
  _2026-03-23_ · https://arxiv.org/abs/2603.22230v1  
  <details><summary>Abstract</summary>

  Accurate land cover mapping in riverine environments is essential for effective river management, ecological understanding, and geomorphic change monitoring. This study explores the use of Point Transformer v2 (PTv2), an advanced deep neural network architecture designed for point cloud data, for land cover mapping through semantic segmentation of multispectral LiDAR data in real-world riverine environments. We utilize the geometric and spectral information from the 3-channel LiDAR point cloud to map land cover classes, including sand, gravel, low vegetation, high vegetation, forest floor, and water. The PTv2 model was trained and evaluated on point cloud data from the Oulanka river in northern Finland using both geometry and spectral features. To improve the model's generalization in new riverine environments, we additionally investigate multi-dataset training that adds sparsely annotated data from an additional river dataset. Results demonstrated that using the full-feature configuration resulted in performance with a mean Intersection over Union (mIoU) of 0.950, significantly outperforming the geometry baseline. Other ablation studies revealed that intensity and reflectance features were the key for accurate land cover mapping. The multi-dataset training experiment showed improved generalization performance, suggesting potential for developing more robust models despite limited high-quality annotated data. Our work demonstrates the potential of applying transformer-based architectures to multispectral point clouds in riverine environments. The approach offers new capabilities for monitoring sediment transport and other river management applications.

  </details>



- **SpatialReward: Verifiable Spatial Reward Modeling for Fine-Grained Spatial Consistency in Text-to-Image Generation**  
  Sashuai Zhou, Qiang Zhou, Junpeng Ma, Yue Cao, Ruofan Hu, Ziang Zhang, Xiaoda Yang, Zhibin Wang, Jun Song, Cheng Yu, et al.  
  _2026-03-23_ · https://arxiv.org/abs/2603.22228v1  
  <details><summary>Abstract</summary>

  Recent advances in text-to-image (T2I) generation via reinforcement learning (RL) have benefited from reward models that assess semantic alignment and visual quality. However, most existing reward models pay limited attention to fine-grained spatial relationships, often producing images that appear plausible overall yet contain inaccuracies in object positioning. In this work, we present \textbf{SpatialReward}, a verifiable reward model explicitly designed to evaluate spatial layouts in generated images. SpatialReward adopts a multi-stage pipeline: a \emph{Prompt Decomposer} extracts entities, attributes, and spatial metadata from free-form prompts; expert detectors provide accurate visual grounding of object positions and attributes; and a vision-language model applies chain-of-thought reasoning over grounded observations to assess complex spatial relations that are challenging for rule-based methods. To more comprehensively evaluate spatial relationships in generated images, we introduce \textbf{SpatRelBench}, a benchmark covering object attributes, orientation, inter-object relations, and rendered text placement. Experiments on Stable Diffusion and FLUX show that incorporating SpatialReward into RL training consistently improves spatial consistency and overall generation quality, with results aligned more closely to human judgments. These findings indicate that verifiable reward models hold considerable potential for enabling more accurate and controllable optimization in text-to-image generation models.

  </details>



- **Omni-WorldBench: Towards a Comprehensive Interaction-Centric Evaluation for World Models**  
  Meiqi Wu, Zhixin Cai, Fufangchen Zhao, Xiaokun Feng, Rujing Dang, Bingze Song, Ruitian Tian, Jiashu Zhu, Jiachen Lei, Hao Dou, et al.  
  _2026-03-23_ · https://arxiv.org/abs/2603.22212v1  
  <details><summary>Abstract</summary>

  Video--based world models have emerged along two dominant paradigms: video generation and 3D reconstruction. However, existing evaluation benchmarks either focus narrowly on visual fidelity and text--video alignment for generative models, or rely on static 3D reconstruction metrics that fundamentally neglect temporal dynamics. We argue that the future of world modeling lies in 4D generation, which jointly models spatial structure and temporal evolution. In this paradigm, the core capability is interactive response: the ability to faithfully reflect how interaction actions drive state transitions across space and time. Yet no existing benchmark systematically evaluates this critical dimension. To address this gap, we propose Omni--WorldBench, a comprehensive benchmark specifically designed to evaluate the interactive response capabilities of world models in 4D settings. Omni--WorldBench comprises two key components: Omni--WorldSuite, a systematic prompt suite spanning diverse interaction levels and scene types; and Omni--Metrics, an agent-based evaluation framework that quantifies world modeling capabilities by measuring the causal impact of interaction actions on both final outcomes and intermediate state evolution trajectories. We conduct extensive evaluations of 18 representative world models across multiple paradigms. Our analysis reveals critical limitations of current world models in interactive response, providing actionable insights for future research. Omni-WorldBench will be publicly released to foster progress in interactive 4D world modeling.

  </details>



- **A Backbone Benchmarking Study on Self-supervised Learning as a Auxiliary Task with Texture-based Local Descriptors for Face Analysis**  
  Shukesh Reddy, Abhijit Das  
  _2026-03-23_ · https://arxiv.org/abs/2603.22190v1  
  <details><summary>Abstract</summary>

  In this work, we benchmark with different backbones and study their impact for self-supervised learning (SSL) as an auxiliary task to blend texture-based local descriptors into feature modelling for efficient face analysis. It is established in previous work that combining a primary task and a self-supervised auxiliary task enables more robust and discriminative representation learning. We employed different shallow to deep backbones for the SSL task of Masked Auto-Encoder (MAE) as an auxiliary objective to reconstruct texture features such as local patterns alongside the primary task in local pattern SSAT (L-SSAT), ensuring robust and unbiased face analysis. To expand the benchmark, we conducted a comprehensive comparative analysis across multiple model configurations within the proposed framework. To this end, we address the three research questions: "What is the role of the backbone in performance L-SSAT?", "What type of backbone is effective for different face analysis tasks?", and "Is there any generalized backbone for effective face analysis with L-SSAT?". Towards answering these questions, we provide a detailed study and experiments. The performance evaluation demonstrates that the backbone for the proposed method is highly dependent on the downstream task, achieving average accuracies of 0.94 on FaceForensics++, 0.87 on CelebA, and 0.88 on AffectNet. For consistency of feature representation quality and generalisation capability across various face analysis paradigms, including face attribute prediction, emotion classification, and deepfake detection, there is no unified backbone.

  </details>



- **Beyond Matching to Tiles: Bridging Unaligned Aerial and Satellite Views for Vision-Only UAV Navigation**  
  Kejia Liu, Haoyang Zhou, Ruoyu Xu, Peicheng Wang, Mingli Song, Haofei Zhang  
  _2026-03-23_ · https://arxiv.org/abs/2603.22153v1  
  <details><summary>Abstract</summary>

  Recent advances in cross-view geo-localization (CVGL) methods have shown strong potential for supporting unmanned aerial vehicle (UAV) navigation in GNSS-denied environments. However, existing work predominantly focuses on matching UAV views to onboard map tiles, which introduces an inherent trade-off between accuracy and storage overhead, and overlooks the importance of the UAV's heading during navigation. Moreover, the substantial discrepancies and varying overlaps in cross-view scenarios have been insufficiently considered, limiting their generalization to real-world scenarios. In this paper, we present Bearing-UAV, a purely vision-driven cross-view navigation method that jointly predicts UAV absolute location and heading from neighboring features, enabling accurate, lightweight, and robust navigation in the wild. Our method leverages global and local structural features and explicitly encodes relative spatial relationships, making it robust to cross-view variations, misalignment, and feature-sparse conditions. We also present Bearing-UAV-90k, a multi-city benchmark for evaluating cross-view localization and navigation. Extensive experiments show encouraging results that Bearing-UAV yields lower localization error than previous matching/retrieval paradigm across diverse terrains. Our code and dataset will be made publicly available.

  </details>



- **OpenEarth-Agent: From Tool Calling to Tool Creation for Open-Environment Earth Observation**  
  Sijie Zhao, Feng Liu, Xueliang Zhang, Hao Chen, Xinyu Gu, Zhe Jiang, Fenghua Ling, Ben Fei, Wenlong Zhang, Junjue Wang, et al.  
  _2026-03-23_ · https://arxiv.org/abs/2603.22148v1  
  <details><summary>Abstract</summary>

  Earth Observation (EO) is essential for perceiving dynamic land surface changes, yet deploying autonomous EO in open environments is hindered by the immense diversity of multi-source data and heterogeneous tasks. While remote sensing agents have emerged to streamline EO workflows, existing tool-calling agents are confined to closed environments. They rely on pre-defined tools and are restricted to narrow scope, limiting their generalization to the diverse data and tasks. To overcome these limitations, we introduce OpenEarth-Agent, the first tool-creation agent framework tailored for open-environment EO. Rather than calling predefined tools, OpenEarth-Agent employs adaptive workflow planning and tool creation to generalize to unseen data and tasks. This adaptability is bolstered by an open-ended integration of multi-stage tools and cross-domain knowledge bases, enabling robust execution in the entire EO pipeline across multiple application domains. To comprehensively evaluate EO agents in open environments, we propose OpenEarth-Bench, a novel benchmark comprising 596 real-world, full-pipeline cases across seven application domains, explicitly designed to assess agents' adaptive planning and tool creation capabilities. Only essential pre-trained model tools are provided in this benchmark, devoid of any other predefined task-specific tools. Extensive experiments demonstrate that OpenEarth-Agent successfully masters full-pipeline EO across multiple domains in the open environment. Notably, on the cross-benchmark Earth-Bench, our tool-creating agent equipped with 6 essential pre-trained models achieves performance comparable to tool-calling agents relying on 104 specialized tools, and significantly outperforms them when provided with the complete toolset. In several cases, the created tools exhibit superior robustness to data anomalies compared to human-engineered counterparts.

  </details>



- **ROBOGATE: Adaptive Failure Discovery for Safe Robot Policy Deployment via Two-Stage Boundary-Focused Sampling**  
  Byungjin Kim  
  _2026-03-23_ · https://arxiv.org/abs/2603.22126v1  
  <details><summary>Abstract</summary>

  Deploying learned robot manipulation policies in industrial settings requires rigorous pre-deployment validation, yet exhaustive testing across high-dimensional parameter spaces is intractable. We present ROBOGATE, a deployment risk management framework that combines physics-based simulation with a two-stage adaptive sampling strategy to efficiently discover failure boundaries in the operational parameter space. Stage 1 employs Latin Hypercube Sampling (LHS) across an 8-dimensional parameter space to establish a coarse failure landscape from 20,000 uniformly distributed experiments. Stage 2 applies boundary-focused sampling that concentrates 10,000 additional experiments in the 30-70% success rate transition zone, enabling precise failure boundary mapping. Using NVIDIA Isaac Sim with Newton physics, we evaluate a scripted pick-and-place controller on two robot embodiments -- Franka Panda (7-DOF) and UR5e (6-DOF) -- across 30,000 total experiments. Our logistic regression risk model achieves an AUC of 0.780 on the combined dataset (vs. 0.754 for Stage 1 alone), identifies a closed-form failure boundary equation, and reveals four universal danger zones affecting both robot platforms. We further demonstrate the framework on VLA (Vision-Language-Action) model evaluation, where Octo-Small achieves 0.0% success rate on 68 adversarial scenarios versus 100% for the scripted baseline -- a 100-point gap that underscores the challenge of deploying foundation models in industrial settings. ROBOGATE is open-source and runs on a single GPU workstation.

  </details>



- **Mamba-VMR: Multimodal Query Augmentation via Generated Videos for Precise Temporal Grounding**  
  Yunzhuo Sun, Xinyue Liu, Yanyang Li, Nanding Wu, Yifang Xu, Linlin Zong, Xianchao Zhang, Wenxin Liang  
  _2026-03-23_ · https://arxiv.org/abs/2603.22121v1  
  <details><summary>Abstract</summary>

  Text-driven video moment retrieval (VMR) remains challenging due to limited capture of hidden temporal dynamics in untrimmed videos, leading to imprecise grounding in long sequences. Traditional methods rely on natural language queries (NLQs) or static image augmentations, overlooking motion sequences and suffering from high computational costs in Transformer-based architectures. Existing approaches fail to integrate subtitle contexts and generated temporal priors effectively, we therefore propose a novel two-stage framework for enhanced temporal grounding. In the first stage, LLM-guided subtitle matching identifies relevant textual cues from video subtitles, fused with the query to generate auxiliary short videos via text-to-video models, capturing implicit motion information as temporal priors. In the second stage, augmented queries are processed through a multi-modal controlled Mamba network, extending text-controlled selection with video-guided gating for efficient fusion of generated priors and long sequences while filtering noise. Our framework is agnostic to base retrieval models and widely applicable for multimodal VMR. Experimental evaluations on the TVR benchmark demonstrate significant improvements over state-of-the-art methods, including reduced computational overhead and higher recall in long-sequence grounding.

  </details>



- **FontCrafter: High-Fidelity Element-Driven Artistic Font Creation with Visual In-Context Generation**  
  Wuyang Luo, Chengkai Tan, Chang Ge, Binye Hong, Su Yang, Yongjiu Ma  
  _2026-03-23_ · https://arxiv.org/abs/2603.22054v1  
  <details><summary>Abstract</summary>

  Artistic font generation aims to synthesize stylized glyphs based on a reference style. However, existing approaches suffer from limited style diversity and coarse control. In this work, we explore the potential of element-driven artistic font generation. Elements are the fundamental visual units of a font, serving as reference images for the desired style. Conceptually, we categorize elements into object elements (e.g., flowers or stones) with distinct structures and amorphous elements (e.g., flames or clouds) with unstructured textures. We introduce FontCrafter, an element-driven framework for font creation, and construct a large-scale dataset, ElementFont, which contains diverse element types and high-quality glyph images. However, achieving high-fidelity reconstruction of both texture and structure of reference elements remains challenging. To address this, we propose an in-context generation strategy that treats element images as visual context and uses an inpainting model to transfer element styles into glyph regions at the pixel level. To further control glyph shapes, we design a lightweight Context-aware Mask Adapter (CMA) that injects shape information. Moreover, a training-free attention redirection mechanism enables region-aware style control and suppresses stroke hallucination. In addition, edge repainting is applied to make boundaries more natural. Extensive experiments demonstrate that FontCrafter achieves strong zero-shot generation performance, particularly in preserving structural and textural fidelity, while also supporting flexible controls such as style mixture.

  </details>



- **GTSR: Subsurface Scattering Awared 3D Gaussians for Translucent Surface Reconstruction**  
  Youwen Yuan, Xi Zhao  
  _2026-03-23_ · https://arxiv.org/abs/2603.22036v1  
  <details><summary>Abstract</summary>

  Reconstructing translucent objects from multi-view images is a difficult problem. Previously, researchers have used differentiable path tracing and the neural implicit field, which require relatively large computational costs. Recently, many works have achieved good reconstruction results for opaque objects based on a 3DGS pipeline with much higher efficiency. However, such methods have difficulty dealing with translucent objects, because they do not consider the optical properties of translucent objects. In this paper, we propose a novel 3DGS-based pipeline (GTSR) to reconstruct the surface geometry of translucent objects. GTSR combines two sets of Gaussians, surface and interior Gaussians, which are used to model the surface and scattering color when lights pass translucent objects. To render the appearance of translucent objects, we introduce a method that uses the Fresnel term to blend two sets of Gaussians. Furthermore, to improve the reconstructed details of non-contour areas, we introduce the Disney BSDF model with deferred rendering to enhance constraints of the normal and depth. Experimental results demonstrate that our method outperforms baseline reconstruction methods on the NeuralTO Syn dataset while showing great real-time rendering performance. We also extend the dataset with new translucent objects of varying material properties and demonstrate our method can adapt to different translucent materials.

  </details>



- **VP-VLA: Visual Prompting as an Interface for Vision-Language-Action Models**  
  Zixuan Wang, Yuxin Chen, Yuqi Liu, Jinhui Ye, Pengguang Chen, Changsheng Lu, Shu Liu, Jiaya Jia  
  _2026-03-23_ · https://arxiv.org/abs/2603.22003v1  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models typically map visual observations and linguistic instructions directly to robotic control signals. This "black-box" mapping forces a single forward pass to simultaneously handle instruction interpretation, spatial grounding, and low-level control, often leading to poor spatial precision and limited robustness in out-of-distribution scenarios. To address these limitations, we propose VP-VLA, a dual-system framework that decouples high-level reasoning and low-level execution via a structured visual prompting interface. Specifically, a "System 2 Planner" decomposes complex instructions into sub-tasks and identifies relevant target objects and goal locations. These spatial anchors are then overlaid directly onto visual observations as structured visual prompts, such as crosshairs and bounding boxes. Guided by these prompts and enhanced by a novel auxiliary visual grounding objective during training, a "System 1 Controller" reliably generates precise low-level execution motions. Experiments on the Robocasa-GR1-Tabletop benchmark and SimplerEnv simulation demonstrate that VP-VLA improves success rates by 5% and 8.3%, surpassing competitive baselines including QwenOFT and GR00T-N1.6.

  </details>



- **LRC-WeatherNet: LiDAR, RADAR, and Camera Fusion Network for Real-time Weather-type Classification in Autonomous Driving**  
  Nour Alhuda Albashir, Lars Pernickel, Danial Hamoud, Idriss Gouigah, Eren Erdal Aksoy  
  _2026-03-23_ · https://arxiv.org/abs/2603.21987v1  
  <details><summary>Abstract</summary>

  Autonomous vehicles face major perception and navigation challenges in adverse weather such as rain, fog, and snow, which degrade the performance of LiDAR, RADAR, and RGB camera sensors. While each sensor type offers unique strengths, such as RADAR robustness in poor visibility and LiDAR precision in clear conditions, they also suffer distinct limitations when exposed to environmental obstructions. This study proposes LRC-WeatherNet, a novel multi-sensor fusion framework that integrates LiDAR, RADAR, and camera data for real-time classification of weather conditions. By employing both early fusion using a unified Bird's Eye View representation and mid-level gated fusion of modality-specific feature maps, our approach adapts to the varying reliability of each sensor under changing weather. Evaluated on the extensive MSU-4S dataset covering nine weather types, LRC-WeatherNet achieves superior classification performance and computational efficiency, significantly outperforming unimodal baselines in adverse conditions. This work is the first to combine all three modalities for robust, real-time weather classification in autonomous driving. We release our trained models and source code in https://github.com/nouralhudaalbashir/LRC-WeatherNet.

  </details>



- **GeoFusion-CAD: Structure-Aware Diffusion with Geometric State Space for Parametric 3D Design**  
  Xiaolei Zhou, Chuangjie Fang, Jie Wu, Jingyi Yang, Boyi Lin, Jianwei Zheng  
  _2026-03-23_ · https://arxiv.org/abs/2603.21978v1  
  <details><summary>Abstract</summary>

  Parametric Computer-Aided Design (CAD) is fundamental to modern 3D modeling, yet existing methods struggle to generate long command sequences, especially under complex geometric and topological dependencies. Transformer-based architectures dominate CAD sequence generation due to their strong dependency modeling, but their quadratic attention cost and limited context windowing hinder scalability to long programs. We propose GeoFusion-CAD, an end-to-end diffusion framework for scalable and structure-aware generation. Our proposal encodes CAD programs as hierarchical trees, jointly capturing geometry and topology within a state-space diffusion process. Specifically, a lightweight C-Mamba block models long-range structural dependencies through selective state transitions, enabling coherent generation across extended command sequences. To support long-sequence evaluation, we introduce DeepCAD-240, an extended benchmark that increases the sequence length ranging from 40 to 240 while preserving sketch-extrusion semantics from the ABC dataset. Extensive experiments demonstrate that GeoFusion-CAD achieves superior performance on both short and long command ranges, maintaining high geometric fidelity and topological consistency where Transformer-based models degrade. Our approach sets new state-of-the-art scores for long-sequence parametric CAD generation, establishing a scalable foundation for next-generation CAD modeling systems. Code and datasets are available at GitHub.

  </details>



- **BHDD: A Burmese Handwritten Digit Dataset**  
  Swan Htet Aung, Hein Htet, Htoo Say Wah Khaing, Thuya Myo Nyunt  
  _2026-03-23_ · https://arxiv.org/abs/2603.21966v1  
  <details><summary>Abstract</summary>

  We introduce the Burmese Handwritten Digit Dataset (BHDD), a collection of 87,561 grayscale images of handwritten Burmese digits in ten classes. Each image is 28x28 pixels, following the MNIST format. The training set has 60,000 samples split evenly across classes; the test set has 27,561 samples with class frequencies as they arose during collection. Over 150 people of different ages and backgrounds contributed samples. We analyze the dataset's class distribution, pixel statistics, and morphological variation, and identify digit pairs that are easily confused due to the round shapes of the Myanmar script. Simple baselines (an MLP, a two-layer CNN, and an improved CNN with batch normalization and augmentation) reach 99.40%, 99.75%, and 99.83% test accuracy respectively. BHDD is available under CC BY-SA 4.0 at https://github.com/baseresearch/BHDD

  </details>



- **MultiBind: A Benchmark for Attribute Misbinding in Multi-Subject Generation**  
  Wenqing Tian, Hanyi Mao, Zhaocheng Liu, Lihua Zhang, Qiang Liu, Jian Wu, Liang Wang  
  _2026-03-23_ · https://arxiv.org/abs/2603.21937v1  
  <details><summary>Abstract</summary>

  Subject-driven image generation is increasingly expected to support fine-grained control over multiple entities within a single image. In multi-reference workflows, users may provide several subject images, a background reference, and long, entity-indexed prompts to control multiple people within one scene. In this setting, a key failure mode is cross-subject attribute misbinding: attributes are preserved, edited, or transferred to the wrong subject. Existing benchmarks and metrics largely emphasize holistic fidelity or per-subject self-similarity, making such failures hard to diagnose. We introduce MultiBind, a benchmark built from real multi-person photographs. Each instance provides slot-ordered subject crops with masks and bounding boxes, canonicalized subject references, an inpainted background reference, and a dense entity-indexed prompt derived from structured annotations. We also propose a dimension-wise confusion evaluation protocol that matches generated subjects to ground-truth slots and measures slot-to-slot similarity using specialists for face identity, appearance, pose, and expression. By subtracting the corresponding ground-truth similarity matrices, our method separates self-degradation from true cross-subject interference and exposes interpretable failure patterns such as drift, swap, dominance, and blending. Experiments on modern multi-reference generators show that MultiBind reveals binding failures that conventional reconstruction metrics miss.

  </details>



- **Chronological Contrastive Learning: Few-Shot Progression Assessment in Irreversible Diseases**  
  Clemens Watzenböck, Daniel Aletaha, Michaël Deman, Thomas Deimel, Jana Eder, Ivana Janickova, Robert Janiczek, Peter Mandl, Philipp Seeböck, Gabriela Supp, et al.  
  _2026-03-23_ · https://arxiv.org/abs/2603.21935v1  
  <details><summary>Abstract</summary>

  Quantitative disease severity scoring in medical imaging is costly, time-consuming, and subject to inter-reader variability. At the same time, clinical archives contain far more longitudinal imaging data than expert-annotated severity scores. Existing self-supervised methods typically ignore this chronological structure. We introduce ChronoCon, a contrastive learning approach that replaces label-based ranking losses with rankings derived solely from the visitation order of a patient's longitudinal scans. Under the clinically plausible assumption of monotonic progression in irreversible diseases, the method learns disease-relevant representations without using any expert labels. This generalizes the idea of Rank-N-Contrast from label distances to temporal ordering. Evaluated on rheumatoid arthritis radiographs for severity assessment, the learned representations substantially improve label efficiency. In low-label settings, ChronoCon significantly outperforms a fully supervised baseline initialized from ImageNet weights. In a few-shot learning experiment, fine-tuning ChronoCon on expert scores from only five patients yields an intraclass correlation coefficient of 86% for severity score prediction. These results demonstrate the potential of chronological contrastive learning to exploit routinely available imaging metadata to reduce annotation requirements in the irreversible disease domain. Code is available at https://github.com/cirmuw/ChronoCon.

  </details>



- **SatGeo-NeRF: Geometrically Regularized NeRF for Satellite Imagery**  
  Valentin Wagner, Sebastian Bullinger, Michael Arens, Rainer Stiefelhagen  
  _2026-03-23_ · https://arxiv.org/abs/2603.21931v1  
  <details><summary>Abstract</summary>

  We present SatGeo-NeRF, a geometrically regularized NeRF for satellite imagery that mitigates overfitting-induced geometric artifacts observed in current state-of-the-art models using three model-agnostic regularizers. Gravity-Aligned Planarity Regularization aligns depth-inferred, approximated surface normals with the gravity axis to promote local planarity, coupling adjacent rays via a corresponding surface approximation to facilitate cross-ray gradient flow. Granularity Regularization enforces a coarse-to-fine geometry-learning scheme, and Depth-Supervised Regularization stabilizes early training for improved geometric accuracy. On the DFC2019 satellite reconstruction benchmark, SatGeo-NeRF improves the Mean Altitude Error by 13.9% and 11.7% relative to state-of-the-art baselines such as EO-NeRF and EO-GS.

  </details>



- **HMS-VesselNet: Hierarchical Multi-Scale Attention Network with Topology-Preserving Loss for Retinal Vessel Segmentation**  
  Amarnath R  
  _2026-03-23_ · https://arxiv.org/abs/2603.21891v1  
  <details><summary>Abstract</summary>

  Retinal vessel segmentation methods based on standard overlap losses tend to miss thin peripheral vessels because these structures occupy very few pixels and have low contrast against the background. We propose HMS-VesselNet, a hierarchical multi-scale network that processes fundus images across four parallel branches at different resolutions and combines their outputs using learned fusion weights. The training loss combines Dice, binary cross-entropy, and centerline Dice to jointly optimize area overlap and vessel continuity. Hard example mining is applied from epoch 20 onward to concentrate gradient updates on the most difficult training images. Tested on 68 images from DRIVE, STARE, and CHASE_DB1 using 5-fold cross-validation, the model achieves a mean Dice of 88.72 +/- 0.67%, Sensitivity of 90.78 +/- 1.42%, and AUC of 98.25 +/- 0.21%. In leave-one-dataset-out experiments, AUC remains above 95% on each unseen dataset. The largest improvement is in the recall of thin peripheral vessels, which are the structures most frequently missed by standard methods and most critical for early detection of diabetic retinopathy.

  </details>



- **SteelDefectX: A Coarse-to-Fine Vision-Language Dataset and Benchmark for Generalizable Steel Surface Defect Detection**  
  Shuxian Zhao, Jie Gui, Baosheng Yu, Lu Dong, Zhipeng Gui  
  _2026-03-23_ · https://arxiv.org/abs/2603.21824v1  
  <details><summary>Abstract</summary>

  Steel surface defect detection is essential for ensuring product quality and reliability in modern manufacturing. Current methods often rely on basic image classification models trained on label-only datasets, which limits their interpretability and generalization. To address these challenges, we introduce SteelDefectX, a vision-language dataset containing 7,778 images across 25 defect categories, annotated with coarse-to-fine textual descriptions. At the coarse-grained level, the dataset provides class-level information, including defect categories, representative visual attributes, and associated industrial causes. At the fine-grained level, it captures sample-specific attributes, such as shape, size, depth, position, and contrast, enabling models to learn richer and more detailed defect representations. We further establish a benchmark comprising four tasks, vision-only classification, vision-language classification, few/zero-shot recognition, and zero-shot transfer, to evaluate model performance and generalization. Experiments with several baseline models demonstrate that coarse-to-fine textual annotations significantly improve interpretability, generalization, and transferability. We hope that SteelDefectX will serve as a valuable resource for advancing research on explainable, generalizable steel surface defect detection. The data will be publicly available on https://github.com/Zhaosxian/SteelDefectX.

  </details>



- **Beyond Strict Pairing: Arbitrarily Paired Training for High-Performance Infrared and Visible Image Fusion**  
  Yanglin Deng, Tianyang Xu, Chunyang Cheng, Hui Li, Xiao-jun Wu, Josef Kittler  
  _2026-03-23_ · https://arxiv.org/abs/2603.21820v1  
  <details><summary>Abstract</summary>

  Infrared and visible image fusion(IVIF) combines complementary modalities while preserving natural textures and salient thermal signatures. Existing solutions predominantly rely on extensive sets of rigidly aligned image pairs for training. However, acquiring such data is often impractical due to the costly and labour-intensive alignment process. Besides, maintaining a rigid pairing setting during training restricts the volume of cross-modal relationships, thereby limiting generalisation performance. To this end, this work challenges the necessity of Strictly Paired Training Paradigm (SPTP) by systematically investigating UnPaired and Arbitrarily Paired Training Paradigms (UPTP and APTP) for high-performance IVIF. We establish a theoretical objective of APTP, reflecting the complementary nature between UPTP and SPTP. More importantly, we develop a practical framework capable of significantly enriching cross-modal relationships even with severely limited and unaligned training data. To validate our propositions, three end-to-end lightweight baselines, alongside a set of innovative loss functions, are designed to cover three classic frameworks (CNN, Transformer, GAN). Comprehensive experiments demonstrate that the proposed APTP and UPTP are feasible and capable of training models on a severely limited and content-inconsistent infrared and visible dataset, achieving performance comparable to that of a dataset 100$\times$ larger in SPTP. This finding fundamentally alleviates the cost and difficulty of data collection while enhancing model robustness from the data perspective, delivering a feasible solution for IVIF studies. The code is available at \href{https://github.com/yanglinDeng/IVIF_unpair}{\textcolor{blue}{https://github.com/yanglinDeng/IVIF\_unpair}}.

  </details>



- **Ctrl-A: Control-Driven Online Data Augmentation**  
  Jesper B. Christensen, Ciaran Bench, Spencer A. Thomas, Hüsnü Aslan, David Balslev-Harder, Nadia A. S. Smith, Alessandra Manzin  
  _2026-03-23_ · https://arxiv.org/abs/2603.21819v1  
  <details><summary>Abstract</summary>

  We introduce ControlAugment (Ctrl-A), an automated data augmentation algorithm for image-vision tasks, which incorporates principles from control theory for online adjustment of augmentation strength distributions during model training. Ctrl-A eliminates the need for initialization of individual augmentation strengths. Instead, augmentation strength distributions are dynamically, and individually, adapted during training based on a control-loop architecture and what we define as relative operation response curves. Using an operation-dependent update procedure provides Ctrl-A with the potential to suppress augmentation styles that negatively impact model performance, alleviating the need for manually engineering augmentation policies for new image-vision tasks. Experiments on the CIFAR-10, CIFAR-100, and SVHN-core benchmark datasets using the common WideResNet-28-10 architecture demonstrate that Ctrl-A is highly competitive with existing state-of-the-art data augmentation strategies.

  </details>



- **Clinical Graph-Mediated Distillation for Unpaired MRI-to-CFI Hypertension Prediction**  
  Dillan Imans, Phuoc-Nguyen Bui, Duc-Tai Le, Hyunseung Choo  
  _2026-03-23_ · https://arxiv.org/abs/2603.21809v1  
  <details><summary>Abstract</summary>

  Retinal fundus imaging enables low-cost and scalable hypertension (HTN) screening, but HTN-related retinal cues are subtle, yielding high-variance predictions. Brain MRI provides stronger vascular and small-vessel-disease markers of HTN, yet it is expensive and rarely acquired alongside fundus images, resulting in modality-siloed datasets with disjoint MRI and fundus cohorts. We study this unpaired MRI-fundus regime and introduce Clinical Graph-Mediated Distillation (CGMD), a framework that transfers MRI-derived HTN knowledge to a fundus model without paired multimodal data. CGMD leverages shared structured biomarkers as a bridge by constructing a clinical similarity kNN graph spanning both cohorts. We train an MRI teacher, propagate its representations over the graph, and impute brain-informed representation targets for fundus patients. A fundus student is then trained with a joint objective combining HTN supervision, target distillation, and relational distillation. Experiments on our newly collected unpaired MRI-fundus-biomarker dataset show that CGMD consistently improves fundus-based HTN prediction over standard distillation and non-graph imputation baselines, with ablations confirming the importance of clinically grounded graph connectivity. Code is available at https://github.com/DillanImans/CGMD-unpaired-distillation.

  </details>



- **Benchmarking Recurrent Event-Based Object Detection for Industrial Multi-Class Recognition on MTEvent**  
  Lokeshwaran Manohar, Moritz Roidl  
  _2026-03-23_ · https://arxiv.org/abs/2603.21787v1  
  <details><summary>Abstract</summary>

  Event cameras are attractive for industrial robotics because they provide high temporal resolution, high dynamic range, and reduced motion blur. However, most event-based object detection studies focus on outdoor driving scenarios or limited class settings. In this work, we benchmark recurrent ReYOLOv8s on MTEvent for industrial multi-class recognition and use a non-recurrent YOLOv8s variant as a baseline to analyze the effect of temporal memory. On the MTEvent validation split, the best scratch recurrent model (C21) reaches 0.285 mAP50, corresponding to a 9.6% relative improvement over the nonrecurrent YOLOv8s baseline (0.260). Event-domain pretraining has a stronger effect: GEN1-initialized fine-tuning yields the best overall result of 0.329 mAP50 at clip length 21, and unlike scratch training, GEN1-pretrained models improve consistently with clip length. PEDRo initialization drops to 0.251, indicating that mismatched source-domain pretraining can be less effective than training from scratch. Persistent failure modes are dominated by class imbalance and human-object interaction. Overall, we position this work as a focused benchmarking and analysis study of recurrent event-based detection in industrial environments.

  </details>



- **The Universal Normal Embedding**  
  Chen Tasker, Roy Betser, Eyal Gofer, Meir Yossef Levi, Guy Gilboa  
  _2026-03-23_ · https://arxiv.org/abs/2603.21786v1  
  <details><summary>Abstract</summary>

  Generative models and vision encoders have largely advanced on separate tracks, optimized for different goals and grounded in different mathematical principles. Yet, they share a fundamental property: latent space Gaussianity. Generative models map Gaussian noise to images, while encoders map images to semantic embeddings whose coordinates empirically behave as Gaussian. We hypothesize that both are views of a shared latent source, the Universal Normal Embedding (UNE): an approximately Gaussian latent space from which encoder embeddings and DDIM-inverted noise arise as noisy linear projections. To test our hypothesis, we introduce NoiseZoo, a dataset of per-image latents comprising DDIM-inverted diffusion noise and matching encoder representations (CLIP, DINO). On CelebA, linear probes in both spaces yield strong, aligned attribute predictions, indicating that generative noise encodes meaningful semantics along linear directions. These directions further enable faithful, controllable edits (e.g., smile, gender, age) without architectural changes, where simple orthogonalization mitigates spurious entanglements. Taken together, our results provide empirical support for the UNE hypothesis and reveal a shared Gaussian-like latent geometry that concretely links encoding and generation. Code and data are available https://rbetser.github.io/UNE/

  </details>



- **Cycle Inverse-Consistent TransMorph: A Balanced Deep Learning Framework for Brain MRI Registration**  
  Jiaqi Shang, Haojin Wu, Yinyi Lai, Zongyu Li, Chenghao Zhang, Jia Guo  
  _2026-03-23_ · https://arxiv.org/abs/2603.21760v1  
  <details><summary>Abstract</summary>

  Deformable image registration plays a fundamental role in medical image analysis by enabling spatial alignment of anatomical structures across subjects. While recent deep learning-based approaches have significantly improved computational efficiency, many existing methods remain limited in capturing long-range anatomical correspondence and maintaining deformation consistency. In this work, we present a cycle inverse-consistent transformer-based framework for deformable brain MRI registration. The model integrates a Swin-UNet architecture with bidirectional consistency constraints, enabling the joint estimation of forward and backward deformation fields. This design allows the framework to capture both local anatomical details and global spatial relationships while improving deformation stability. We conduct a comprehensive evaluation of the proposed framework on a large multi-center dataset consisting of 2851 T1-weighted brain MRI scans aggregated from 13 public datasets. Experimental results demonstrate that the proposed framework achieves strong and balanced performance across multiple quantitative evaluation metrics while maintaining stable and physically plausible deformation fields. Detailed quantitative comparisons with baseline methods, including ANTs, ICNet, and VoxelMorph, are provided in the appendix. Experimental results demonstrate that CICTM achieves consistently strong performance across multiple evaluation criteria while maintaining stable and physically plausible deformation fields. These properties make the proposed framework suitable for large-scale neuroimaging datasets where both accuracy and deformation stability are critical.

  </details>



- **PRM-as-a-Judge: A Dense Evaluation Paradigm for Fine-Grained Robotic Auditing**  
  Yuheng Ji, Yuyang Liu, Huajie Tan, Xuchuan Huang, Fanding Huang, Yijie Xu, Cheng Chi, Yuting Zhao, Huaihai Lyu, Peterson Co, et al.  
  _2026-03-23_ · https://arxiv.org/abs/2603.21669v1  
  <details><summary>Abstract</summary>

  Current robotic evaluation is still largely dominated by binary success rates, which collapse rich execution processes into a single outcome and obscure critical qualities such as progress, efficiency, and stability. To address this limitation, we propose PRM-as-a-Judge, a dense evaluation paradigm that leverages Process Reward Models (PRMs) to audit policy execution directly from trajectory videos by estimating task progress from observation sequences. Central to this paradigm is the OPD (Outcome-Process-Diagnosis) metric system, which explicitly formalizes execution quality via a task-aligned progress potential. We characterize dense robotic evaluation through two axiomatic properties: macro-consistency, which requires additive and path-consistent aggregation, and micro-resolution, which requires sensitivity to fine-grained physical evolution. Under this formulation, potential-based PRM judges provide a natural instantiation of dense evaluation, with macro-consistency following directly from the induced scalar potential. We empirically validate the micro-resolution property using RoboPulse, a diagnostic benchmark specifically designed for probing micro-scale progress discrimination, where several trajectory-trained PRM judges outperform discriminative similarity-based methods and general-purpose foundation-model judges. Finally, leveraging PRM-as-a-Judge and the OPD metric system, we conduct a structured audit of mainstream policy paradigms across long-horizon tasks, revealing behavioral signatures and failure modes that are invisible to outcome-only metrics.

  </details>



- **HumanOmni-Speaker: Identifying Who said What and When**  
  Detao Bai, Shimin Yao, Weixuan Chen, Xihan Wei, Zhiheng Ma  
  _2026-03-23_ · https://arxiv.org/abs/2603.21664v1  
  <details><summary>Abstract</summary>

  While Omni-modal Large Language Models have made strides in joint sensory processing, they fundamentally struggle with a cornerstone of human interaction: deciphering complex, multi-person conversational dynamics to accurately answer ``Who said what and when.'' Current models suffer from an ``illusion of competence'' -- they exploit visual biases in conventional benchmarks to bypass genuine cross-modal alignment, while relying on sparse, low-frame-rate visual sampling that destroys crucial high-frequency dynamics like lip movements. To shatter this illusion, we introduce Visual-Registered Speaker Diarization and Recognition (VR-SDR) and the HumanOmni-Speaker Benchmark. By strictly eliminating visual shortcuts, this rigorous paradigm demands true end-to-end spatio-temporal identity binding using only natural language queries. To overcome the underlying architectural perception gap, we propose HumanOmni-Speaker, powered by a Visual Delta Encoder. By sampling raw video at 25 fps and explicitly compressing inter-frame motion residuals into just 6 tokens per frame, it captures fine-grained visemes and speaker trajectories without triggering a catastrophic token explosion. Ultimately, HumanOmni-Speaker demonstrates strong multimodal synergy, natively enabling end-to-end lip-reading and high-precision spatial localization without intrusive cropping, and achieving superior performance across a wide spectrum of speaker-centric tasks.

  </details>



- **OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging**  
  Meilin Liu, Jiaying Wang, Jing Shan  
  _2026-03-23_ · https://arxiv.org/abs/2603.21660v1  
  <details><summary>Abstract</summary>

  Federated learning (FL) has become a promising paradigm for collaborative medical image analysis, yet existing frameworks remain tightly coupled to task-specific backbones and are fragile under heterogeneous imaging modalities. Such constraints hinder real-world deployment, where institutions vary widely in modality distributions and must support diverse downstream tasks. To address this limitation, we propose OmniFM, a modality- and task-agnostic FL framework that unifies training across classification, segmentation, super-resolution, visual question answering, and multimodal fusion without re-engineering the optimization pipeline. OmniFM builds on a key frequency-domain insight: low-frequency spectral components exhibit strong cross-modality consistency and encode modality-invariant anatomical structures. Accordingly, OmniFM integrates (i) Global Spectral Knowledge Retrieval to inject global frequency priors, (ii) Embedding-wise Cross-Attention Fusion to align representations, and (iii) Prefix-Suffix Spectral Prompting to jointly condition global and personalized cues, together regularized by a Spectral-Proximal Alignment objective that stabilizes aggregation. Experiments on real-world datasets show that OmniFM consistently surpasses state-of-the-art FL baselines across intra- and cross-modality heterogeneity, achieving superior results under both fine-tuning and training-from-scratch setups.

  </details>



- **No Dense Tensors Needed: Fully Sparse Object Detection on Event-Camera Voxel Grids**  
  Mohamad Yazan Sadoun, Sarah Sharif, Yaser Mike Banad  
  _2026-03-23_ · https://arxiv.org/abs/2603.21638v1  
  <details><summary>Abstract</summary>

  Event cameras produce asynchronous, high-dynamic-range streams well suited for detecting small, fast-moving drones, yet most event-based detectors convert the sparse event stream into dense tensors, discarding the representational efficiency of neuromorphic sensing. We propose SparseVoxelDet, to our knowledge the first fully sparse object detector for event cameras, in which backbone feature extraction, feature pyramid fusion, and the detection head all operate exclusively on occupied voxel positions through 3D sparse convolutions; no dense feature tensor is instantiated at any stage of the pipeline. On the FRED benchmark (629,832 annotated frames), SparseVoxelDet achieves 83.38% mAP at 50 while processing only 14,900 active voxels per frame (0.23% of the T.H.W grid), compared to 409,600 pixels for the dense YOLOv11 baseline (87.68% mAP at 50). Relaxing the IoU threshold from 0.50 to 0.40 recovers mAP to 89.26%, indicating that the remaining accuracy gap is dominated by box regression precision rather than detection capability. The sparse representation yields 858 times GPU memory compression and 3,670 times storage reduction relative to the equivalent dense 3D voxel tensor, with data-structure size that scales with scene dynamics rather than sensor resolution. Error forensics across 119,459 test frames confirms that 71 percent of failures are localization near-misses rather than missed targets. These results demonstrate that native sparse processing is a viable paradigm for event-camera object detection, exploiting the structural sparsity of neuromorphic sensor data without requiring neuromorphic computing hardware, and providing a framework whose representation cost is governed by scene activity rather than pixel count, a property that becomes increasingly valuable as event cameras scale to higher resolutions.

  </details>



- **Dual-level Adaptation for Multi-Object Tracking: Building Test-Time Calibration from Experience and Intuition**  
  Wen Guo, Pengfei Zhao, Zongmeng Wang, Yufan Hu, Junyu Gao  
  _2026-03-23_ · https://arxiv.org/abs/2603.21629v1  
  <details><summary>Abstract</summary>

  Multiple Object Tracking (MOT) has long been a fundamental task in computer vision, with broad applications in various real-world scenarios. However, due to distribution shifts in appearance, motion pattern, and catagory between the training and testing data, model performance degrades considerably during online inference in MOT. Test-Time Adaptation (TTA) has emerged as a promising paradigm to alleviate such distribution shifts. However, existing TTA methods often fail to deliver satisfactory results in MOT, as they primarily focus solely on frame-level adaptation while neglecting temporal consistency and identity association across frames and videos. Inspired by human decision-making process, this paper propose a Test-time Calibration from Experience and Intuition (TCEI) framework. In this framework, the Intuitive system utilizes transient memory to recall recently observed objects for rapid predictions, while the Experiential system leverages the accumulated experience from prior test videos to reassess and calibrate these intuitive predictions. Furthermore, both confident and uncertain objects during online testing are exploited as historical priors and reflective cases, respectively, enabling the model to adapt to the testing environment and alleviate performance degradation. Extensive experiments demonstrate that the proposed TCEI framework consistently achieves superior performance across multiple benchmark datasets and significantly enhances the model's adaptability under distribution shifts. The code will be released at https://github.com/1941Zpf/TCEI.

  </details>



- **Efficient Zero-Shot AI-Generated Image Detection**  
  Ryosuke Sonoda, Ramya Srinivasan  
  _2026-03-23_ · https://arxiv.org/abs/2603.21619v1  
  <details><summary>Abstract</summary>

  The rapid progress of text-to-image models has made AI-generated images increasingly realistic, posing significant challenges for accurate detection of generated content. While training-based detectors often suffer from limited generalization to unseen images, training-free approaches offer better robustness, yet struggle to capture subtle discrepancies between real and synthetic images. In this work, we propose a training-free AI-generated image detection method that measures representation sensitivity to structured frequency perturbations, enabling detection of minute manipulations. The proposed method is computationally lightweight, as perturbation generation requires only a single Fourier transform for an input image. As a result, it achieves one to two orders of magnitude faster inference than most training-free detectors.Extensive experiments on challenging benchmarks demonstrate the efficacy of our method over state-of-the-art (SoTA). In particular, on OpenFake benchmark, our method improves AUC by nearly $10\%$ compared to SoTA, while maintaining substantially lower computational cost.

  </details>



- **4DGS360: 360° Gaussian Reconstruction of Dynamic Objects from a Single Video**  
  Jae Won Jang, Yeonjin Chang, Wonsik Shin, Juhwan Cho, Nojun Kwak  
  _2026-03-23_ · https://arxiv.org/abs/2603.21618v1  
  <details><summary>Abstract</summary>

  We introduce 4DGS360, a diffusion-free framework for 360$^{\circ}$ dynamic object reconstruction from casual monocular video. Existing methods often fail to reconstruct consistent 360$^{\circ}$ geometry, as their heavy reliance on 2D-native priors causes initial points to overfit to visible surface in each training view. 4DGS360 addresses this challenge through a advanced 3D-native initialization that mitigates the geometric ambiguity of occluded regions. Our proposed 3D tracker, AnchorTAP3D, produces reinforced 3D point trajectories by leveraging confident 2D track points as anchors, suppressing drift and providing reliable initialization that preserves geometry in occluded regions. This initialization, combined with optimization, yields coherent 360$^{\circ}$ 4D reconstructions. We further present iPhone360, a new benchmark where test cameras are placed up to 135$^{\circ}$ apart from training views, enabling 360$^{\circ}$ evaluation that existing datasets cannot provide. Experiments show that 4DGS360 achieves state-of-the-art performance on the iPhone360, iPhone, and DAVIS datasets, both qualitatively and quantitatively.

  </details>



- **AdaEdit: Adaptive Temporal and Channel Modulation for Flow-Based Image Editing**  
  Guandong Li, Zhaobin Chu  
  _2026-03-23_ · https://arxiv.org/abs/2603.21615v1  
  <details><summary>Abstract</summary>

  Inversion-based image editing in flow matching models has emerged as a powerful paradigm for training-free, text-guided image manipulation. A central challenge in this paradigm is the injection dilemma: injecting source features during denoising preserves the background of the original image but simultaneously suppresses the model's ability to synthesize edited content. Existing methods address this with fixed injection strategies -- binary on/off temporal schedules, uniform spatial mixing ratios, and channel-agnostic latent perturbation -- that ignore the inherently heterogeneous nature of injection demand across both the temporal and channel dimensions. In this paper, we present AdaEdit, a training-free adaptive editing framework that resolves this dilemma through two complementary innovations. First, we propose a Progressive Injection Schedule that replaces hard binary cutoffs with continuous decay functions (sigmoid, cosine, or linear), enabling a smooth transition from source-feature preservation to target-feature generation and eliminating feature discontinuity artifacts. Second, we introduce Channel-Selective Latent Perturbation, which estimates per-channel importance based on the distributional gap between the inverted and random latents and applies differentiated perturbation strengths accordingly -- strongly perturbing edit-relevant channels while preserving structure-encoding channels. Extensive experiments on the PIE-Bench benchmark (700 images, 10 editing types) demonstrate that AdaEdit achieves an 8.7% reduction in LPIPS, a 2.6% improvement in SSIM, and a 2.3% improvement in PSNR over strong baselines, while maintaining competitive CLIP similarity. AdaEdit is fully plug-and-play and compatible with multiple ODE solvers including Euler, RF-Solver, and FireFlow. Code is available at https://github.com/leeguandong/AdaEdit

  </details>



- **A Multidisciplinary AI Board for Multimodal Dementia Characterization and Risk Assessment**  
  Sheng Liu, Long Chen, Zeyun Zhao, Qinglin Gou, Qingyue Wei, Arjun Masurkar, Kevin M. Spiegler, Philip Kuball, Stefania C. Bray, Megan Bernath, et al.  
  _2026-03-23_ · https://arxiv.org/abs/2603.21597v1  
  <details><summary>Abstract</summary>

  Modern clinical practice increasingly depends on reasoning over heterogeneous, evolving, and incomplete patient data. Although recent advances in multimodal foundation models have improved performance on various clinical tasks, most existing models remain static, opaque, and poorly aligned with real-world clinical workflows. We present Cerebra, an interactive multi-agent AI team that coordinates specialized agents for EHR, clinical notes, and medical imaging analysis. These outputs are synthesized into a clinician-facing dashboard that combines visual analytics with a conversational interface, enabling clinicians to interrogate predictions and contextualize risk at the point of care. Cerebra supports privacy-preserving deployment by operating on structured representations and remains robust when modalities are incomplete. We evaluated Cerebra using a massive multi-institutional dataset spanning 3 million patients from four independent healthcare systems. Cerebra consistently outperformed both state-of-the-art single-modality models and large multimodal language model baselines. In dementia risk prediction, it achieved AUROCs up to 0.80, compared with 0.74 for the strongest single-modality model and 0.68 for language model baselines. For dementia diagnosis, it achieved an AUROC of 0.86, and for survival prediction, a C-index of 0.81. In a reader study with experienced physicians, Cerebra significantly improved expert performance, increasing accuracy by 17.5 percentage points in prospective dementia risk estimation. These results demonstrate Cerebra's potential for interpretable, robust decision support in clinical care.

  </details>



- **Rethinking Visual Privacy: A Compositional Privacy Risk Framework for Severity Assessment with VLMs**  
  Efthymios Tsaprazlis, Tiantian Feng, Anil Ramakrishna, Sai Praneeth Karimireddy, Rahul Gupta, Shrikanth Narayanan  
  _2026-03-23_ · https://arxiv.org/abs/2603.21573v1  
  <details><summary>Abstract</summary>

  Existing visual privacy benchmarks largely treat privacy as a binary property, labeling images as private or non-private based on visible sensitive content. We argue that privacy is fundamentally compositional. Attributes that are benign in isolation may combine to produce severe privacy violations. We introduce the Compositional Privacy Risk Taxonomy (CPRT), a regulation-aware framework that organizes visual attributes according to standalone identifiability and compositional harm potential. CPRT defines four graded severity levels and is paired with an interpretable scoring function that assigns continuous privacy severity scores. We further construct a taxonomy-aligned dataset of 6.7K images and derive ground-truth compositional risk scores. By evaluating frontier and open-weight VLMs we find that frontier models align well with compositional severity when provided structured guidance, but systematically underestimate composition-driven risks. Smaller models struggle to internalize graded privacy reasoning. To bridge this gap, we introduce a deployable 8B supervised fine-tuned (SFT) model that closely matches frontier-level performance on compositional privacy assessment.

  </details>



- **CataractSAM-2: A Domain-Adapted Model for Anterior Segment Surgery Segmentation and Scalable Ground-Truth Annotation**  
  Mohammad Eslami, Dhanvinkumar Ganeshkumar, Saber Kazeminasab, Michael G. Morley, Michael V. Boland, Michael M. Lin, John B. Miller, David S. Friedman, Nazlee Zebardast, Lucia Sobrin, et al.  
  _2026-03-23_ · https://arxiv.org/abs/2603.21566v1  
  <details><summary>Abstract</summary>

  We present CataractSAM-2, a domain-adapted extension of Meta's Segment Anything Model 2, designed for real-time semantic segmentation of cataract ophthalmic surgery videos with high accuracy. Positioned at the intersection of computer vision and medical robotics, CataractSAM-2 enables precise intraoperative perception crucial for robotic-assisted and computer-guided surgical systems. Furthermore, to alleviate the burden of manual labeling, we introduce an interactive annotation framework that combines sparse prompts with video-based mask propagation. This tool significantly reduces annotation time and facilitates the scalable creation of high-quality ground-truth masks, accelerating dataset development for ocular anterior segment surgeries. We also demonstrate the model's strong zero-shot generalization to glaucoma trabeculectomy procedures, confirming its cross-procedural utility and potential for broader surgical applications. The trained model and annotation toolkit are released as open-source resources, establishing CataractSAM-2 as a foundation for expanding anterior ophthalmic surgical datasets and advancing real-time AI-driven solutions in medical robotics, as well as surgical video understanding.

  </details>



- **Exploring Multimodal Prompts For Unsupervised Continuous Anomaly Detection**  
  Mingle Zhou, Jiahui Liu, Jin Wan, Gang Li, Min Li  
  _2026-03-23_ · https://arxiv.org/abs/2603.21562v1  
  <details><summary>Abstract</summary>

  Unsupervised Continuous Anomaly Detection (UCAD) is gaining attention for effectively addressing the catastrophic forgetting and heavy computational burden issues in traditional Unsupervised Anomaly Detection (UAD). However, existing UCAD approaches that rely solely on visual information are insufficient to capture the manifold of normality in complex scenes, thereby impeding further gains in anomaly detection accuracy. To overcome this limitation, we propose an unsupervised continual anomaly detection framework grounded in multimodal prompting. Specifically, we introduce a Continual Multimodal Prompt Memory Bank (CMPMB) that progressively distills and retains prototypical normal patterns from both visual and textual domains across consecutive tasks, yielding a richer representation of normality. Furthermore, we devise a Defect-Semantic-Guided Adaptive Fusion Mechanism (DSG-AFM) that integrates an Adaptive Normalization Module (ANM) with a Dynamic Fusion Strategy (DFS) to jointly enhance detection accuracy and adversarial robustness. Benchmark experiments on MVTec AD and VisA datasets show that our approach achieves state-of-the-art (SOTA) performance on image-level AUROC and pixel-level AUPR metrics.

  </details>



- **Revisiting Weakly-Supervised Video Scene Graph Generation via Pair Affinity Learning**  
  Minseok Kang, Minhyeok Lee, Minjung Kim, Jungho Lee, Donghyeong Kim, Sungmin Woo, Inseok Jeon, Sangyoun Lee  
  _2026-03-23_ · https://arxiv.org/abs/2603.21559v1  
  <details><summary>Abstract</summary>

  Weakly-supervised video scene graph generation (WS-VSGG) aims to parse video content into structured relational triplets without bounding box annotations and with only sparse temporal labeling, significantly reducing annotation costs. Without ground-truth bounding boxes, these methods rely on off-the-shelf detectors to generate object proposals, yet largely overlook a fundamental discrepancy from fullysupervised pipelines. Fully-supervised detectors implicitly filter out noninteractive objects, while off-the-shelf detectors indiscriminately detect all visible objects, overwhelming relation models with noisy pairs.We address this by introducing a learnable pair affinity that estimates the likelihood of interaction between subject-object pairs. Through Pair Affinity Learning and Scoring (PALS), pair affinity is incorporated into inferencetime ranking and further integrated into contextual reasoning through Pair Affinity Modulation (PAM), enabling the model to suppress noninteractive pairs and focus on relationally meaningful ones. To provide cleaner supervision for pair affinity learning, we further propose Relation- Aware Matching (RAM), which leverages vision-language grounding to resolve class-level ambiguity in pseudo-label generation. Extensive experiments on Action Genome demonstrate that our approach consistently yields substantial improvements across different baselines and backbones, achieving state-of-the-art WS-VSGG performance.

  </details>



- **VIGIL: Part-Grounded Structured Reasoning for Generalizable Deepfake Detection**  
  Xinghan Li, Junhao Xu, Jingjing Chen  
  _2026-03-23_ · https://arxiv.org/abs/2603.21526v1  
  <details><summary>Abstract</summary>

  Multimodal large language models (MLLMs) offer a promising path toward interpretable deepfake detection by generating textual explanations. However, the reasoning process of current MLLM-based methods combines evidence generation and manipulation localization into a unified step. This combination blurs the boundary between faithful observations and hallucinated explanations, leading to unreliable conclusions. Building on this, we present VIGIL, a part-centric structured forensic framework inspired by expert forensic practice through a plan-then-examine pipeline: the model first plans which facial parts warrant inspection based on global visual cues, then examines each part with independently sourced forensic evidence. A stage-gated injection mechanism delivers part-level forensic evidence only during examination, ensuring that part selection remains driven by the model's own perception rather than biased by external signals. We further propose a progressive three-stage training paradigm whose reinforcement learning stage employs part-aware rewards to enforce anatomical validity and evidence--conclusion coherence. To enable rigorous generalizability evaluation, we construct OmniFake, a hierarchical 5-Level benchmark where the model, trained on only three foundational generators, is progressively tested up to in-the-wild social-media data. Extensive experiments on OmniFake and cross-dataset evaluations demonstrate that VIGIL consistently outperforms both expert detectors and concurrent MLLM-based methods across all generalizability levels.

  </details>



- **Parameter-efficient Prompt Tuning and Hierarchical Textual Guidance for Few-shot Whole Slide Image Classification**  
  Jayanie Bogahawatte, Sachith Seneviratne, Saman Halgamuge  
  _2026-03-23_ · https://arxiv.org/abs/2603.21504v1  
  <details><summary>Abstract</summary>

  Whole Slide Images (WSIs) are giga-pixel in scale and are typically partitioned into small instances in WSI classification pipelines for computational feasibility. However, obtaining extensive instance level annotations is costly, making few-shot weakly supervised WSI classification (FSWC) crucial for learning from limited slide-level labels. Recently, pre-trained vision-language models (VLMs) have been adopted in FSWC, yet they exhibit several limitations. Existing prompt tuning methods in FSWC substantially increase both the number of trainable parameters and inference overhead. Moreover, current methods discard instances with low alignment to text embeddings from VLMs, potentially leading to information loss. To address these challenges, we propose two key contributions. First, we introduce a new parameter efficient prompt tuning method by scaling and shifting features in text encoder, which significantly reduces the computational cost. Second, to leverage not only the pre-trained knowledge of VLMs, but also the inherent hierarchical structure of WSIs, we introduce a WSI representation learning approach with a soft hierarchical textual guidance strategy without utilizing hard instance filtering. Comprehensive evaluations on pathology datasets covering breast, lung, and ovarian cancer types demonstrate consistent improvements up-to 10.9%, 7.8%, and 13.8% respectively, over the state-of-the-art methods in FSWC. Our method reduces the number of trainable parameters by 18.1% on both breast and lung cancer datasets, and 5.8% on the ovarian cancer dataset, while also excelling at weakly-supervised tumor localization. Code at https://github.com/Jayanie/HIPSS.

  </details>



- **StreamingEval: A Unified Evaluation Protocol towards Realistic Streaming Video Understanding**  
  Guowei Tang, Tianwen Qian, Huanran Zheng, Yifei Wang, Xiaoling Wang  
  _2026-03-23_ · https://arxiv.org/abs/2603.21493v1  
  <details><summary>Abstract</summary>

  Real-time, continuous understanding of visual signals is essential for real-world interactive AI applications, and poses a fundamental system-level challenge. Existing research on streaming video understanding, however, typically focuses on isolated aspects such as question-answering accuracy under limited visual context or improvements in encoding efficiency, while largely overlooking practical deployability under realistic resource constraints. To bridge this gap, we introduce StreamingEval, a unified evaluation framework for assessing the streaming video understanding capabilities of Video-LLMs under realistic constraints. StreamingEval benchmarks both mainstream offline models and recent online video models under a standardized protocol, explicitly characterizing the trade-off between efficiency, storage and accuracy. Specifically, we adopt a fixed-capacity memory bank to normalize accessible historical visual context, and jointly evaluate visual encoding efficiency, text decoding latency, and task performance to quantify overall system deployability. Extensive experiments across multiple datasets reveal substantial gaps between current Video-LLMs and the requirements of realistic streaming applications, providing a systematic basis for future research in this direction. Codes will be released at https://github.com/wwgTang-111/StreamingEval1.

  </details>



- **EpiMask: Leveraging Epipolar Distance Based Masks in Cross-Attention for Satellite Image Matching**  
  Rahul Deshmukh, Aditya Chauhan, Avinash Kak  
  _2026-03-23_ · https://arxiv.org/abs/2603.21463v1  
  <details><summary>Abstract</summary>

  The deep-learning based image matching networks can now handle significantly larger variations in viewpoints and illuminations while providing matched pairs of pixels with sub-pixel precision. These networks have been trained with ground-based image datasets and, implicitly, their performance is optimized for the pinhole camera geometry. Consequently, you get suboptimal performance when such networks are used to match satellite images since those images are synthesized as a moving satellite camera records one line at a time of the points on the ground. In this paper, we present EpiMask, a semi-dense image matching network for satellite images that (1) Incorporates patch-wise affine approximations to the camera modeling geometry; (2) Uses an epipolar distance-based attention mask to restrict cross-attention to geometrically plausible regions; and (3) That fine-tunes a foundational pretrained image encoder for robust feature extraction. Experiments on the SatDepth dataset demonstrate up to 30% improvement in matching accuracy compared to re-trained ground-based models.

  </details>



- **Bayesian Active Object Recognition and 6D Pose Estimation from Multimodal Contact Sensing**  
  Haodong Zheng, Gabriele M. Caddeo, Andrei C. Jalba, Wijnand A. IJsselsteijn, Lorenzo Natale, Raymond H. Cuijpers  
  _2026-03-22_ · https://arxiv.org/abs/2603.21410v1  
  <details><summary>Abstract</summary>

  We present an active tactile exploration framework for joint object recognition and 6D pose estimation. The proposed method integrates wrist force/torque sensing, GelSight tactile sensing, and free-space constraints within a Bayesian inference framework that maintains a belief over object class and pose during active tactile exploration. By combining contact and non-contact evidence, the framework reduces ambiguity and improves robustness in the joint class-pose estimation problem. To enable efficient inference in the large hypothesis space, we employ a customized particle filter that progressively samples particles based on new observations. The inferred belief is further used to guide active exploration by selecting informative next touches under reachability constraints. For effective data collection, a motion planning and control framework is developed to plan and execute feasible paths for tactile exploration, handle unexpected contacts and GelSight-surface alignment with tactile servoing. We evaluate the framework in simulation and on a Franka Panda robot using 11 YCB objects. Results show that incorporating tactile and free-space information substantially improves recognition and pose estimation accuracy and stability, while reducing the number of action cycles compared with force/torque-only baselines. Code, dataset, and supplementary material will be made available online.

  </details>



- **Respiratory Status Detection with Video Transformers**  
  Thomas Savage, Evan Madill  
  _2026-03-22_ · https://arxiv.org/abs/2603.21349v1  
  <details><summary>Abstract</summary>

  Recognition of respiratory distress through visual inspection is a life saving clinical skill. Clinicians can detect early signs of respiratory deterioration, creating a valuable window for earlier intervention. In this study, we evaluate whether recent advances in video transformers can enable Artificial Intelligence systems to recognize the signs of respiratory distress from video. We collected videos of healthy volunteers recovering after strenuous exercise and used the natural recovery of each participants respiratory status to create a labeled dataset for respiratory distress. Splitting the video into short clips, with earlier clips corresponding to more shortness of breath, we designed a temporal ordering challenge to assess whether an AI system can detect respiratory distress. We found a ViViT encoder augmented with Lie Relative Encodings (LieRE) and Motion Guided Masking, combined with an embedding based comparison strategy, can achieve an F1 score of 0.81 on this task. Our findings suggest that modern video transformers can recognize subtle changes in respiratory mechanics.

  </details>



- **Privacy-Preserving Federated Action Recognition via Differentially Private Selective Tuning and Efficient Communication**  
  Idris Zakariyya, Pai Chet Ng, Kaushik Bhargav Sivangi, S. Mohammad Sheikholeslami, Konstantinos N. Plataniotis, Fani Deligianni  
  _2026-03-22_ · https://arxiv.org/abs/2603.21305v1  
  <details><summary>Abstract</summary>

  Federated video action recognition enables collaborative model training without sharing raw video data, yet remains vulnerable to two key challenges: \textit{model exposure} and \textit{communication overhead}. Gradients exchanged between clients and the server can leak private motion patterns, while full-model synchronization of high-dimensional video networks causes significant bandwidth and communication costs. To address these issues, we propose \textit{Federated Differential Privacy with Selective Tuning and Efficient Communication for Action Recognition}, namely \textit{FedDP-STECAR}. Our \textit{FedDP-STECAR} framework selectively fine-tunes and perturbs only a small subset of task-relevant layers under Differential Privacy (DP), reducing the surface of information leakage while preserving temporal coherence in video features. By transmitting only the tuned layers during aggregation, communication traffic is reduced by over 99\% compared to full-model updates. Experiments on the UCF-101 dataset using the MViT-B-16x4 transformer show that \textit{FedDP-STECAR} achieves up to \textbf{70.2\% higher accuracy} under strict privacy ($ε=0.65$) in centralized settings and \textbf{48\% faster training} with \textbf{73.1\% accuracy} in federated setups, enabling scalable and privacy-preserving video action recognition. Code available at https://github.com/izakariyya/mvit-federated-videodp

  </details>



- **Identity-Consistent Video Generation under Large Facial-Angle Variations**  
  Bin Hu, Zipeng Qi, Guoxi Huang, Zunnan Xu, Ruicheng Zhang, Chongjie Ye, Jun Zhou, Xiu Li, Jingdong Wang  
  _2026-03-22_ · https://arxiv.org/abs/2603.21299v1  
  <details><summary>Abstract</summary>

  Single-view reference-to-video methods often struggle to preserve identity consistency under large facial-angle variations. This limitation naturally motivates the incorporation of multi-view facial references. However, simply introducing additional reference images exacerbates the \textit{copy-paste} problem, particularly the \textbf{\textit{view-dependent copy-paste}} artifact, which reduces facial motion naturalness. Although cross-paired data can alleviate this issue, collecting such data is costly. To balance the consistency and naturalness, we propose $\mathrm{Mv}^2\mathrm{ID}$, a multi-view conditioned framework under in-paired supervision. We introduce a region-masking training strategy to prevent shortcut learning and extract essential identity features by encouraging the model to aggregate complementary identity cues across views. In addition, we design a reference decoupled-RoPE mechanism that assigns distinct positional encoding to video and conditioning tokens for better modeling of their heterogeneous properties. Furthermore, we construct a large-scale dataset with diverse facial-angle variations and propose dedicated evaluation metrics for identity consistency and motion naturalness. Extensive experiments demonstrate that our method significantly improves identity consistency while maintaining motion naturalness, outperforming existing approaches trained with cross-paired data.

  </details>


