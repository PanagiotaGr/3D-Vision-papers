# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-02-10 07:20 UTC_

Total papers shown: **50**


---

- **Robustness Is a Function, Not a Number: A Factorized Comprehensive Study of OOD Robustness in Vision-Based Driving**  
  Amir Mallak, Alaa Maalouf  
  _2026-02-09_ · https://arxiv.org/abs/2602.09018v1  
  <details><summary>Abstract</summary>

  Out of distribution (OOD) robustness in autonomous driving is often reduced to a single number, hiding what breaks a policy. We decompose environments along five axes: scene (rural/urban), season, weather, time (day/night), and agent mix; and measure performance under controlled $k$-factor perturbations ($k \in \{0,1,2,3\}$). Using closed loop control in VISTA, we benchmark FC, CNN, and ViT policies, train compact ViT heads on frozen foundation-model (FM) features, and vary ID support in scale, diversity, and temporal context. (1) ViT policies are markedly more OOD-robust than comparably sized CNN/FC, and FM features yield state-of-the-art success at a latency cost. (2) Naive temporal inputs (multi-frame) do not beat the best single-frame baseline. (3) The largest single factor drops are rural $\rightarrow$ urban and day $\rightarrow$ night ($\sim 31\%$ each); actor swaps $\sim 10\%$, moderate rain $\sim 7\%$; season shifts can be drastic, and combining a time flip with other changes further degrades performance. (4) FM-feature policies stay above $85\%$ under three simultaneous changes; non-FM single-frame policies take a large first-shift hit, and all no-FM models fall below $50\%$ by three changes. (5) Interactions are non-additive: some pairings partially offset, whereas season-time combinations are especially harmful. (6) Training on winter/snow is most robust to single-factor shifts, while a rural+summer baseline gives the best overall OOD performance. (7) Scaling traces/views improves robustness ($+11.8$ points from $5$ to $14$ traces), yet targeted exposure to hard conditions can substitute for scale. (8) Using multiple ID environments broadens coverage and strengthens weak cases (urban OOD $60.6\% \rightarrow 70.1\%$) with a small ID drop; single-ID preserves peak performance but in a narrow domain. These results yield actionable design rules for OOD-robust driving policies.

  </details>



- **Contact-Anchored Policies: Contact Conditioning Creates Strong Robot Utility Models**  
  Zichen Jeff Cui, Omar Rayyan, Haritheja Etukuru, Bowen Tan, Zavier Andrianarivo, Zicheng Teng, Yihang Zhou, Krish Mehta, Nicholas Wojno, Kevin Yuanbo Wu, et al.  
  _2026-02-09_ · https://arxiv.org/abs/2602.09017v1  
  <details><summary>Abstract</summary>

  The prevalent paradigm in robot learning attempts to generalize across environments, embodiments, and tasks with language prompts at runtime. A fundamental tension limits this approach: language is often too abstract to guide the concrete physical understanding required for robust manipulation. In this work, we introduce Contact-Anchored Policies (CAP), which replace language conditioning with points of physical contact in space. Simultaneously, we structure CAP as a library of modular utility models rather than a monolithic generalist policy. This factorization allows us to implement a real-to-sim iteration cycle: we build EgoGym, a lightweight simulation benchmark, to rapidly identify failure modes and refine our models and datasets prior to real-world deployment. We show that by conditioning on contact and iterating via simulation, CAP generalizes to novel environments and embodiments out of the box on three fundamental manipulation skills while using only 23 hours of demonstration data, and outperforms large, state-of-the-art VLAs in zero-shot evaluations by 56%. All model checkpoints, codebase, hardware, simulation, and datasets will be open-sourced. Project page: https://cap-policy.github.io/

  </details>



- **GEBench: Benchmarking Image Generation Models as GUI Environments**  
  Haodong Li, Jingwei Wu, Quan Sun, Guopeng Li, Juanxi Tian, Huanyu Zhang, Yanlin Lai, Ruichuan An, Hongbo Peng, Yuhong Dai, et al.  
  _2026-02-09_ · https://arxiv.org/abs/2602.09007v1  
  <details><summary>Abstract</summary>

  Recent advancements in image generation models have enabled the prediction of future Graphical User Interface (GUI) states based on user instructions. However, existing benchmarks primarily focus on general domain visual fidelity, leaving the evaluation of state transitions and temporal coherence in GUI-specific contexts underexplored. To address this gap, we introduce GEBench, a comprehensive benchmark for evaluating dynamic interaction and temporal coherence in GUI generation. GEBench comprises 700 carefully curated samples spanning five task categories, covering both single-step interactions and multi-step trajectories across real-world and fictional scenarios, as well as grounding point localization. To support systematic evaluation, we propose GE-Score, a novel five-dimensional metric that assesses Goal Achievement, Interaction Logic, Content Consistency, UI Plausibility, and Visual Quality. Extensive evaluations on current models indicate that while they perform well on single-step transitions, they struggle significantly with maintaining temporal coherence and spatial grounding over longer interaction sequences. Our findings identify icon interpretation, text rendering, and localization precision as critical bottlenecks. This work provides a foundation for systematic assessment and suggests promising directions for future research toward building high-fidelity generative GUI environments. The code is available at: https://github.com/stepfun-ai/GEBench.

  </details>



- **CLUE: Crossmodal disambiguation via Language-vision Understanding with attEntion**  
  Mouad Abrini, Mohamed Chetouani  
  _2026-02-09_ · https://arxiv.org/abs/2602.08999v1  
  <details><summary>Abstract</summary>

  With the increasing integration of robots into daily life, human-robot interaction has become more complex and multifaceted. A critical component of this interaction is Interactive Visual Grounding (IVG), through which robots must interpret human intentions and resolve ambiguity. Existing IVG models generally lack a mechanism to determine when to ask clarification questions, as they implicitly rely on their learned representations. CLUE addresses this gap by converting the VLM's cross-modal attention into an explicit, spatially grounded signal for deciding when to ask. We extract text to image attention maps and pass them to a lightweight CNN to detect referential ambiguity, while a LoRA fine-tuned decoder conducts the dialog and emits grounding location tokens. We train on a real-world interactive dataset for IVG, and a mixed ambiguity set for the detector. With InViG-only supervision, our model surpasses a state-of-the-art method while using parameter-efficient fine-tuning. Similarly, the ambiguity detector outperforms prior baselines. Overall, CLUE turns the internal cross-modal attention of a VLM into an explicit, spatially grounded signal for deciding when to ask. The data and code are publicly available at: mouadabrini.github.io/clue

  </details>



- **WorldArena: A Unified Benchmark for Evaluating Perception and Functional Utility of Embodied World Models**  
  Yu Shang, Zhuohang Li, Yiding Ma, Weikang Su, Xin Jin, Ziyou Wang, Xin Zhang, Yinzhou Tang, Chen Gao, Wei Wu, et al.  
  _2026-02-09_ · https://arxiv.org/abs/2602.08971v1  
  <details><summary>Abstract</summary>

  While world models have emerged as a cornerstone of embodied intelligence by enabling agents to reason about environmental dynamics through action-conditioned prediction, their evaluation remains fragmented. Current evaluation of embodied world models has largely focused on perceptual fidelity (e.g., video generation quality), overlooking the functional utility of these models in downstream decision-making tasks. In this work, we introduce WorldArena, a unified benchmark designed to systematically evaluate embodied world models across both perceptual and functional dimensions. WorldArena assesses models through three dimensions: video perception quality, measured with 16 metrics across six sub-dimensions; embodied task functionality, which evaluates world models as data engines, policy evaluators, and action planners integrating with subjective human evaluation. Furthermore, we propose EWMScore, a holistic metric integrating multi-dimensional performance into a single interpretable index. Through extensive experiments on 14 representative models, we reveal a significant perception-functionality gap, showing that high visual quality does not necessarily translate into strong embodied task capability. WorldArena benchmark with the public leaderboard is released at https://worldarena.ai, providing a framework for tracking progress toward truly functional world models in embodied AI.

  </details>



- **Modeling 3D Pedestrian-Vehicle Interactions for Vehicle-Conditioned Pose Forecasting**  
  Guangxun Zhu, Xuan Liu, Nicolas Pugeault, Chongfeng Wei, Edmond S. L. Ho  
  _2026-02-09_ · https://arxiv.org/abs/2602.08962v1  
  <details><summary>Abstract</summary>

  Accurately predicting pedestrian motion is crucial for safe and reliable autonomous driving in complex urban environments. In this work, we present a 3D vehicle-conditioned pedestrian pose forecasting framework that explicitly incorporates surrounding vehicle information. To support this, we enhance the Waymo-3DSkelMo dataset with aligned 3D vehicle bounding boxes, enabling realistic modeling of multi-agent pedestrian-vehicle interactions. We introduce a sampling scheme to categorize scenes by pedestrian and vehicle count, facilitating training across varying interaction complexities. Our proposed network adapts the TBIFormer architecture with a dedicated vehicle encoder and pedestrian-vehicle interaction cross-attention module to fuse pedestrian and vehicle features, allowing predictions to be conditioned on both historical pedestrian motion and surrounding vehicles. Extensive experiments demonstrate substantial improvements in forecasting accuracy and validate different approaches for modeling pedestrian-vehicle interactions, highlighting the importance of vehicle-aware 3D pose prediction for autonomous driving. Code is available at: https://github.com/GuangxunZhu/VehCondPose3D

  </details>



- **Designing Multi-Robot Ground Video Sensemaking with Public Safety Professionals**  
  Puqi Zhou, Ali Asgarov, Aafiya Hussain, Wonjoon Park, Amit Paudyal, Sameep Shrestha, Chia-wei Tang, Michael F. Lighthiser, Michael R. Hieb, Xuesu Xiao, et al.  
  _2026-02-09_ · https://arxiv.org/abs/2602.08882v1  
  <details><summary>Abstract</summary>

  Videos from fleets of ground robots can advance public safety by providing scalable situational awareness and reducing professionals' burden. Yet little is known about how to design and integrate multi-robot videos into public safety workflows. Collaborating with six police agencies, we examined how such videos could be made practical. In Study 1, we presented the first testbed for multi-robot ground video sensemaking. The testbed includes 38 events-of-interest (EoI) relevant to public safety, a dataset of 20 robot patrol videos (10 day/night pairs) covering EoI types, and 6 design requirements aimed at improving current video sensemaking practices. In Study 2, we built MRVS, a tool that augments multi-robot patrol video streams with a prompt-engineered video understanding model. Participants reported reduced manual workload and greater confidence with LLM-based explanations, while noting concerns about false alarms and privacy. We conclude with implications for designing future multi-robot video sensemaking tools. The testbed is available at https://github.com/Puqi7/MRVS\_VideoSensemaking

  </details>



- **karl. -- A Research Vehicle for Automated and Connected Driving**  
  Jean-Pierre Busch, Lukas Ostendorf, Guido Linden, Lennart Reiher, Till Beemelmanns, Bastian Lampe, Timo Woopen, Lutz Eckstein  
  _2026-02-09_ · https://arxiv.org/abs/2602.08842v1  
  <details><summary>Abstract</summary>

  As highly automated driving is transitioning from single-vehicle closed-access testing to commercial deployments of public ride-hailing in selected areas (e.g., Waymo), automated driving and connected cooperative intelligent transport systems (C-ITS) remain active fields of research. Even though simulation is omnipresent in the development and validation life cycle of automated and connected driving technology, the complex nature of public road traffic and software that masters it still requires real-world integration and testing with actual vehicles. Dedicated vehicles for research and development allow testing and validation of software and hardware components under real-world conditions early on. They also enable collecting and publishing real-world datasets that let others conduct research without vehicle access, and support early demonstration of futuristic use cases. In this paper, we present karl., our new research vehicle for automated and connected driving. Apart from major corporations, few institutions worldwide have access to their own L4-capable research vehicles, restricting their ability to carry out independent research. This paper aims to help bridge that gap by sharing the reasoning, design choices, and technical details that went into making karl. a flexible and powerful platform for research, engineering, and validation in the context of automated and connected driving. More impressions of karl. are available at https://karl.ac.

  </details>



- **VideoVeritas: AI-Generated Video Detection via Perception Pretext Reinforcement Learning**  
  Hao Tan, Jun Lan, Senyuan Shi, Zichang Tan, Zijian Yu, Huijia Zhu, Weiqiang Wang, Jun Wan, Zhen Lei  
  _2026-02-09_ · https://arxiv.org/abs/2602.08828v1  
  <details><summary>Abstract</summary>

  The growing capability of video generation poses escalating security risks, making reliable detection increasingly essential. In this paper, we introduce VideoVeritas, a framework that integrates fine-grained perception and fact-based reasoning. We observe that while current multi-modal large language models (MLLMs) exhibit strong reasoning capacity, their granular perception ability remains limited. To mitigate this, we introduce Joint Preference Alignment and Perception Pretext Reinforcement Learning (PPRL). Specifically, rather than directly optimizing for detection task, we adopt general spatiotemporal grounding and self-supervised object counting in the RL stage, enhancing detection performance with simple perception pretext tasks. To facilitate robust evaluation, we further introduce MintVid, a light yet high-quality dataset containing 3K videos from 9 state-of-the-art generators, along with a real-world collected subset that has factual errors in content. Experimental results demonstrate that existing methods tend to bias towards either superficial reasoning or mechanical analysis, while VideoVeritas achieves more balanced performance across diverse benchmarks.

  </details>



- **Omni-Video 2: Scaling MLLM-Conditioned Diffusion for Unified Video Generation and Editing**  
  Hao Yang, Zhiyu Tan, Jia Gong, Luozheng Qin, Hesen Chen, Xiaomeng Yang, Yuqing Sun, Yuetan Lin, Mengping Yang, Hao Li  
  _2026-02-09_ · https://arxiv.org/abs/2602.08820v1  
  <details><summary>Abstract</summary>

  We present Omni-Video 2, a scalable and computationally efficient model that connects pretrained multimodal large-language models (MLLMs) with video diffusion models for unified video generation and editing. Our key idea is to exploit the understanding and reasoning capabilities of MLLMs to produce explicit target captions to interpret user instructions. In this way, the rich contextual representations from the understanding model are directly used to guide the generative process, thereby improving performance on complex and compositional editing. Moreover, a lightweight adapter is developed to inject multimodal conditional tokens into pretrained text-to-video diffusion models, allowing maximum reuse of their powerful generative priors in a parameter-efficient manner. Benefiting from these designs, we scale up Omni-Video 2 to a 14B video diffusion model on meticulously curated training data with quality, supporting high quality text-to-video generation and various video editing tasks such as object removal, addition, background change, complex motion editing, \emph{etc.} We evaluate the performance of Omni-Video 2 on the FiVE benchmark for fine-grained video editing and the VBench benchmark for text-to-video generation. The results demonstrate its superior ability to follow complex compositional instructions in video editing, while also achieving competitive or superior quality in video generation tasks.

  </details>



- **Addressing data annotation scarcity in Brain Tumor Segmentation on 3D MRI scan Using a Semi-Supervised Teacher-Student Framework**  
  Jiaming Liu, Cheng Ding, Daoqiang Zhang  
  _2026-02-09_ · https://arxiv.org/abs/2602.08797v1  
  <details><summary>Abstract</summary>

  Accurate brain tumor segmentation from MRI is limited by expensive annotations and data heterogeneity across scanners and sites. We propose a semi-supervised teacher-student framework that combines an uncertainty-aware pseudo-labeling teacher with a progressive, confidence-based curriculum for the student. The teacher produces probabilistic masks and per-pixel uncertainty; unlabeled scans are ranked by image-level confidence and introduced in stages, while a dual-loss objective trains the student to learn from high-confidence regions and unlearn low-confidence ones. Agreement-based refinement further improves pseudo-label quality. On BraTS 2021, validation DSC increased from 0.393 (10% data) to 0.872 (100%), with the largest gains in early stages, demonstrating data efficiency. The teacher reached a validation DSC of 0.922, and the student surpassed the teacher on tumor subregions (e.g., NCR/NET 0.797 and Edema 0.980); notably, the student recovered the Enhancing class (DSC 0.620) where the teacher failed. These results show that confidence-driven curricula and selective unlearning provide robust segmentation under limited supervision and noisy pseudo-labels.

  </details>



- **Multimodal Learning for Arcing Detection in Pantograph-Catenary Systems**  
  Hao Dong, Eleni Chatzi, Olga Fink  
  _2026-02-09_ · https://arxiv.org/abs/2602.08792v1  
  <details><summary>Abstract</summary>

  The pantograph-catenary interface is essential for ensuring uninterrupted and reliable power delivery in electrified rail systems. However, electrical arcing at this interface poses serious risks, including accelerated wear of contact components, degraded system performance, and potential service disruptions. Detecting arcing events at the pantograph-catenary interface is challenging due to their transient nature, noisy operating environment, data scarcity, and the difficulty of distinguishing arcs from other similar transient phenomena. To address these challenges, we propose a novel multimodal framework that combines high-resolution image data with force measurements to more accurately and robustly detect arcing events. First, we construct two arcing detection datasets comprising synchronized visual and force measurements. One dataset is built from data provided by the Swiss Federal Railways (SBB), and the other is derived from publicly available videos of arcing events in different railway systems and synthetic force data that mimic the characteristics observed in the real dataset. Leveraging these datasets, we propose MultiDeepSAD, an extension of the DeepSAD algorithm for multiple modalities with a new loss formulation. Additionally, we introduce tailored pseudo-anomaly generation techniques specific to each data type, such as synthetic arc-like artifacts in images and simulated force irregularities, to augment training data and improve the discriminative ability of the model. Through extensive experiments and ablation studies, we demonstrate that our framework significantly outperforms baseline approaches, exhibiting enhanced sensitivity to real arcing events even under domain shifts and limited availability of real arcing observations.

  </details>



- **GaussianCaR: Gaussian Splatting for Efficient Camera-Radar Fusion**  
  Santiago Montiel-Marín, Miguel Antunes-García, Fabio Sánchez-García, Angel Llamazares, Holger Caesar, Luis M. Bergasa  
  _2026-02-09_ · https://arxiv.org/abs/2602.08784v1  
  <details><summary>Abstract</summary>

  Robust and accurate perception of dynamic objects and map elements is crucial for autonomous vehicles performing safe navigation in complex traffic scenarios. While vision-only methods have become the de facto standard due to their technical advances, they can benefit from effective and cost-efficient fusion with radar measurements. In this work, we advance fusion methods by repurposing Gaussian Splatting as an efficient universal view transformer that bridges the view disparity gap, mapping both image pixels and radar points into a common Bird's-Eye View (BEV) representation. Our main contribution is GaussianCaR, an end-to-end network for BEV segmentation that, unlike prior BEV fusion methods, leverages Gaussian Splatting to map raw sensor information into latent features for efficient camera-radar fusion. Our architecture combines multi-scale fusion with a transformer decoder to efficiently extract BEV features. Experimental results demonstrate that our approach achieves performance on par with, or even surpassing, the state of the art on BEV segmentation tasks (57.3%, 82.9%, and 50.1% IoU for vehicles, roads, and lane dividers) on the nuScenes dataset, while maintaining a 3.2x faster inference runtime. Code and project page are available online.

  </details>



- **Efficient Brain Extraction of MRI Scans with Mild to Moderate Neuropathology**  
  Hjalti Thrastarson, Lotta M. Ellingsen  
  _2026-02-09_ · https://arxiv.org/abs/2602.08764v1  
  <details><summary>Abstract</summary>

  Skull stripping magnetic resonance images (MRI) of the human brain is an important process in many image processing techniques, such as automatic segmentation of brain structures. Numerous methods have been developed to perform this task, however, they often fail in the presence of neuropathology and can be inconsistent in defining the boundary of the brain mask. Here, we propose a novel approach to skull strip T1-weighted images in a robust and efficient manner, aiming to consistently segment the outer surface of the brain, including the sulcal cerebrospinal fluid (CSF), while excluding the full extent of the subarachnoid space and meninges. We train a modified version of the U-net on silver-standard ground truth data using a novel loss function based on the signed-distance transform (SDT). We validate our model both qualitatively and quantitatively using held-out data from the training dataset, as well as an independent external dataset. The brain masks used for evaluation partially or fully include the subarachnoid space, which may introduce bias into the comparison; nonetheless, our model demonstrates strong performance on the held-out test data, achieving a consistent mean Dice similarity coefficient (DSC) of 0.964$\pm$0.006 and an average symmetric surface distance (ASSD) of 1.4mm$\pm$0.2mm. Performance on the external dataset is comparable, with a DSC of 0.958$\pm$0.006 and an ASSD of 1.7$\pm$0.2mm. Our method achieves performance comparable to or better than existing state-of-the-art methods for brain extraction, particularly in its highly consistent preservation of the brain's outer surface. The method is publicly available on GitHub.

  </details>



- **Shifting the Breaking Point of Flow Matching for Multi-Instance Editing**  
  Carmine Zaccagnino, Fabio Quattrini, Enis Simsar, Marta Tintoré Gazulla, Rita Cucchiara, Alessio Tonioni, Silvia Cascianelli  
  _2026-02-09_ · https://arxiv.org/abs/2602.08749v1  
  <details><summary>Abstract</summary>

  Flow matching models have recently emerged as an efficient alternative to diffusion, especially for text-guided image generation and editing, offering faster inference through continuous-time dynamics. However, existing flow-based editors predominantly support global or single-instruction edits and struggle with multi-instance scenarios, where multiple parts of a reference input must be edited independently without semantic interference. We identify this limitation as a consequence of globally conditioned velocity fields and joint attention mechanisms, which entangle concurrent edits. To address this issue, we introduce Instance-Disentangled Attention, a mechanism that partitions joint attention operations, enforcing binding between instance-specific textual instructions and spatial regions during velocity field estimation. We evaluate our approach on both natural image editing and a newly introduced benchmark of text-dense infographics with region-level editing instructions. Experimental results demonstrate that our approach promotes edit disentanglement and locality while preserving global output coherence, enabling single-pass, instance-level editing.

  </details>



- **SynSacc: A Blender-to-V2E Pipeline for Synthetic Neuromorphic Eye-Movement Data and Sim-to-Real Spiking Model Training**  
  Khadija Iddrisu, Waseem Shariff, Suzanne Little, Noel OConnor  
  _2026-02-09_ · https://arxiv.org/abs/2602.08726v1  
  <details><summary>Abstract</summary>

  The study of eye movements, particularly saccades and fixations, are fundamental to understanding the mechanisms of human cognition and perception. Accurate classification of these movements requires sensing technologies capable of capturing rapid dynamics without distortion. Event cameras, also known as Dynamic Vision Sensors (DVS), provide asynchronous recordings of changes in light intensity, thereby eliminating motion blur inherent in conventional frame-based cameras and offering superior temporal resolution and data efficiency. In this study, we introduce a synthetic dataset generated with Blender to simulate saccades and fixations under controlled conditions. Leveraging Spiking Neural Networks (SNNs), we evaluate its robustness by training two architectures and finetuning on real event data. The proposed models achieve up to 0.83 accuracy and maintain consistent performance across varying temporal resolutions, demonstrating stability in eye movement classification. Moreover, the use of SNNs with synthetic event streams yields substantial computational efficiency gains over artificial neural network (ANN) counterparts, underscoring the utility of synthetic data augmentation in advancing event-based vision. All code and datasets associated with this work is available at https: //github.com/Ikhadija-5/SynSacc-Dataset.

  </details>



- **TimeChat-Captioner: Scripting Multi-Scene Videos with Time-Aware and Structural Audio-Visual Captions**  
  Linli Yao, Yuancheng Wei, Yaojie Zhang, Lei Li, Xinlong Chen, Feifan Song, Ziyue Wang, Kun Ouyang, Yuanxin Liu, Lingpeng Kong, et al.  
  _2026-02-09_ · https://arxiv.org/abs/2602.08711v1  
  <details><summary>Abstract</summary>

  This paper proposes Omni Dense Captioning, a novel task designed to generate continuous, fine-grained, and structured audio-visual narratives with explicit timestamps. To ensure dense semantic coverage, we introduce a six-dimensional structural schema to create "script-like" captions, enabling readers to vividly imagine the video content scene by scene, akin to a cinematographic screenplay. To facilitate research, we construct OmniDCBench, a high-quality, human-annotated benchmark, and propose SodaM, a unified metric that evaluates time-aware detailed descriptions while mitigating scene boundary ambiguity. Furthermore, we construct a training dataset, TimeChatCap-42K, and present TimeChat-Captioner-7B, a strong baseline trained via SFT and GRPO with task-specific rewards. Extensive experiments demonstrate that TimeChat-Captioner-7B achieves state-of-the-art performance, surpassing Gemini-2.5-Pro, while its generated dense descriptions significantly boost downstream capabilities in audio-visual reasoning (DailyOmni and WorldSense) and temporal grounding (Charades-STA). All datasets, models, and code will be made publicly available at https://github.com/yaolinli/TimeChat-Captioner.

  </details>



- **ALIVE: Animate Your World with Lifelike Audio-Video Generation**  
  Ying Guo, Qijun Gan, Yifu Zhang, Jinlai Liu, Yifei Hu, Pan Xie, Dongjun Qian, Yu Zhang, Ruiqi Li, Yuqi Zhang, et al.  
  _2026-02-09_ · https://arxiv.org/abs/2602.08682v1  
  <details><summary>Abstract</summary>

  Video generation is rapidly evolving towards unified audio-video generation. In this paper, we present ALIVE, a generation model that adapts a pretrained Text-to-Video (T2V) model to Sora-style audio-video generation and animation. In particular, the model unlocks the Text-to-Video&Audio (T2VA) and Reference-to-Video&Audio (animation) capabilities compared to the T2V foundation models. To support the audio-visual synchronization and reference animation, we augment the popular MMDiT architecture with a joint audio-video branch which includes TA-CrossAttn for temporally-aligned cross-modal fusion and UniTemp-RoPE for precise audio-visual alignment. Meanwhile, a comprehensive data pipeline consisting of audio-video captioning, quality control, etc., is carefully designed to collect high-quality finetuning data. Additionally, we introduce a new benchmark to perform a comprehensive model test and comparison. After continue pretraining and finetuning on million-level high-quality data, ALIVE demonstrates outstanding performance, consistently outperforming open-source models and matching or surpassing state-of-the-art commercial solutions. With detailed recipes and benchmarks, we hope ALIVE helps the community develop audio-video generation models more efficiently. Official page: https://github.com/FoundationVision/Alive.

  </details>



- **WiFlow: A Lightweight WiFi-based Continuous Human Pose Estimation Network with Spatio-Temporal Feature Decoupling**  
  Yi Dao, Lankai Zhang, Hao Liu, Haiwei Zhang, Wenbo Wang  
  _2026-02-09_ · https://arxiv.org/abs/2602.08661v1  
  <details><summary>Abstract</summary>

  Human pose estimation is fundamental to intelligent perception in the Internet of Things (IoT), enabling applications ranging from smart healthcare to human-computer interaction. While WiFi-based methods have gained traction, they often struggle with continuous motion and high computational overhead. This work presents WiFlow, a novel framework for continuous human pose estimation using WiFi signals. Unlike vision-based approaches such as two-dimensional deep residual networks that treat Channel State Information (CSI) as images, WiFlow employs an encoder-decoder architecture. The encoder captures spatio-temporal features of CSI using temporal and asymmetric convolutions, preserving the original sequential structure of signals. It then refines keypoint features of human bodies to be tracked and capture their structural dependencies via axial attention. The decoder subsequently maps the encoded high-dimensional features into keypoint coordinates. Trained on a self-collected dataset of 360,000 synchronized CSI-pose samples from 5 subjects performing continuous sequences of 8 daily activities, WiFlow achieves a Percentage of Correct Keypoints (PCK) of 97.00% at a threshold of 20% (PCK@20) and 99.48% at PCK@50, with a mean per-joint position error of 0.008m. With only 4.82M parameters, WiFlow significantly reduces model complexity and computational cost, establishing a new performance baseline for practical WiFi-based human pose estimation. Our code and datasets are available at https://github.com/DY2434/WiFlow-WiFi-Pose-Estimation-with-Spatio-Temporal-Decoupling.git.

  </details>



- **High-Speed Vision-Based Flight in Clutter with Safety-Shielded Reinforcement Learning**  
  Jiarui Zhang, Chengyong Lei, Chengjiang Dai, Lijie Wang, Zhichao Han, Fei Gao  
  _2026-02-09_ · https://arxiv.org/abs/2602.08653v1  
  <details><summary>Abstract</summary>

  Quadrotor unmanned aerial vehicles (UAVs) are increasingly deployed in complex missions that demand reliable autonomous navigation and robust obstacle avoidance. However, traditional modular pipelines often incur cumulative latency, whereas purely reinforcement learning (RL) approaches typically provide limited formal safety guarantees. To bridge this gap, we propose an end-to-end RL framework augmented with model-based safety mechanisms. We incorporate physical priors in both training and deployment. During training, we design a physics-informed reward structure that provides global navigational guidance. During deployment, we integrate a real-time safety filter that projects the policy outputs onto a provably safe set to enforce strict collision-avoidance constraints. This hybrid architecture reconciles high-speed flight with robust safety assurances. Benchmark evaluations demonstrate that our method outperforms both traditional planners and recent end-to-end obstacle avoidance approaches based on differentiable physics. Extensive experiments demonstrate strong generalization, enabling reliable high-speed navigation in dense clutter and challenging outdoor forest environments at velocities up to 7.5m/s.

  </details>



- **Deep Learning-Based Fixation Type Prediction for Quality Assurance in Digital Pathology**  
  Oskar Thaeter, Tanja Niedermair, Johannes Raffler, Ralf Huss, Peter J. Schüffler  
  _2026-02-09_ · https://arxiv.org/abs/2602.08652v1  
  <details><summary>Abstract</summary>

  Accurate annotation of fixation type is a critical step in slide preparation for pathology laboratories. However, this manual process is prone to errors, impacting downstream analyses and diagnostic accuracy. Existing methods for verifying formalin-fixed, paraffin-embedded (FFPE), and frozen section (FS) fixation types typically require full-resolution whole-slide images (WSIs), limiting scalability for high-throughput quality control. We propose a deep-learning model to predict fixation types using low-resolution, pre-scan thumbnail images. The model was trained on WSIs from the TUM Institute of Pathology (n=1,200, Leica GT450DX) and evaluated on a class-balanced subset of The Cancer Genome Atlas dataset (TCGA, n=8,800, Leica AT2), as well as on class-balanced datasets from Augsburg (n=695 [392 FFPE, 303 FS], Philips UFS) and Regensburg (n=202, 3DHISTECH P1000). Our model achieves an AUROC of 0.88 on TCGA, outperforming comparable pre-scan methods by 4.8%. It also achieves AUROCs of 0.72 on Regensburg and Augsburg slides, underscoring challenges related to scanner-induced domain shifts. Furthermore, the model processes each slide in 21 ms, $400\times$ faster than existing high-magnification, full-resolution methods, enabling rapid, high-throughput processing. This approach provides an efficient solution for detecting labelling errors without relying on high-magnification scans, offering a valuable tool for quality control in high-throughput pathology workflows. Future work will improve and evaluate the model's generalisation to additional scanner types. Our findings suggest that this method can increase accuracy and efficiency in digital pathology workflows and may be extended to other low-resolution slide annotations.

  </details>



- **Thegra: Graph-based SLAM for Thermal Imagery**  
  Anastasiia Kornilova, Ivan Moskalenko, Arabella Gromova, Gonzalo Ferrer, Alexander Menshchikov  
  _2026-02-09_ · https://arxiv.org/abs/2602.08531v1  
  <details><summary>Abstract</summary>

  Thermal imaging provides a practical sensing modality for visual SLAM in visually degraded environments such as low illumination, smoke, or adverse weather. However, thermal imagery often exhibits low texture, low contrast, and high noise, complicating feature-based SLAM. In this work, we propose a sparse monocular graph-based SLAM system for thermal imagery that leverages general-purpose learned features -- the SuperPoint detector and LightGlue matcher, trained on large-scale visible-spectrum data to improve cross-domain generalization. To adapt these components to thermal data, we introduce a preprocessing pipeline to enhance input suitability and modify core SLAM modules to handle sparse and outlier-prone feature matches. We further incorporate keypoint confidence scores from SuperPoint into a confidence-weighted factor graph to improve estimation robustness. Evaluations on public thermal datasets demonstrate that the proposed system achieves reliable performance without requiring dataset-specific training or fine-tuning a desired feature detector, given the scarcity of quality thermal data. Code will be made available upon publication.

  </details>



- **Are Vision Foundation Models Foundational for Electron Microscopy Image Segmentation?**  
  Caterina Fuster-Barceló, Virginie Uhlmann  
  _2026-02-09_ · https://arxiv.org/abs/2602.08505v1  
  <details><summary>Abstract</summary>

  Although vision foundation models (VFMs) are increasingly reused for biomedical image analysis, it remains unclear whether the latent representations they provide are general enough to support effective transfer and reuse across heterogeneous microscopy image datasets. Here, we study this question for the problem of mitochondria segmentation in electron microscopy (EM) images, using two popular public EM datasets (Lucchi++ and VNC) and three recent representative VFMs (DINOv2, DINOv3, and OpenCLIP). We evaluate two practical model adaptation regimes: a frozen-backbone setting in which only a lightweight segmentation head is trained on top of the VFM, and parameter-efficient fine-tuning (PEFT) via Low-Rank Adaptation (LoRA) in which the VFM is fine-tuned in a targeted manner to a specific dataset. Across all backbones, we observe that training on a single EM dataset yields good segmentation performance (quantified as foreground Intersection-over-Union), and that LoRA consistently improves in-domain performance. In contrast, training on multiple EM datasets leads to severe performance degradation for all models considered, with only marginal gains from PEFT. Exploration of the latent representation space through various techniques (PCA, Fréchet Dinov2 distance, and linear probes) reveals a pronounced and persistent domain mismatch between the two considered EM datasets in spite of their visual similarity, which is consistent with the observed failure of paired training. These results suggest that, while VFMs can deliver competitive results for EM segmentation within a single domain under lightweight adaptation, current PEFT strategies are insufficient to obtain a single robust model across heterogeneous EM datasets without additional domain-alignment mechanisms.

  </details>



- **Enhanced Food Category Recognition under Illumination-Induced Domain Shift**  
  Keonvin Park, Aditya Pal, Jin Hong Mok  
  _2026-02-09_ · https://arxiv.org/abs/2602.08491v1  
  <details><summary>Abstract</summary>

  Visual food recognition systems deployed in real-world environments, such as automated conveyor-belt inspection, are highly sensitive to domain shifts caused by illumination changes. While recent studies have shown that lighting variations can significantly distort food perception by both humans and AI, existing works are often limited to single food categories or controlled settings, and most public food datasets lack explicit illumination annotations. In this work, we investigate illumination-induced domain shift in multi-class food category recognition using two widely adopted datasets, Food-101 and Fruits-360. We demonstrate substantial accuracy degradation under cross-dataset evaluation due to mismatched visual conditions. To address this challenge, we construct synthetic illumination-augmented datasets by systematically varying light temperature and intensity, enabling controlled robustness analysis without additional labels. We further evaluate cross-dataset transfer learning and domain generalization, with a focus on illumination-sensitive target categories such as apple-based classes. Experimental results show that illumination-aware augmentation significantly improves recognition robustness under domain shift while preserving real-time performance. Our findings highlight the importance of illumination robustness and provide practical insights for deploying reliable food recognition systems in real-world inspection scenarios.

  </details>



- **Gesture Matters: Pedestrian Gesture Recognition for AVs Through Skeleton Pose Evaluation**  
  Alif Rizqullah Mahdi, Mahdi Rezaei, Natasha Merat  
  _2026-02-09_ · https://arxiv.org/abs/2602.08479v1  
  <details><summary>Abstract</summary>

  Gestures are a key component of non-verbal communication in traffic, often helping pedestrian-to-driver interactions when formal traffic rules may be insufficient. This problem becomes more apparent when autonomous vehicles (AVs) struggle to interpret such gestures. In this study, we present a gesture classification framework using 2D pose estimation applied to real-world video sequences from the WIVW dataset. We categorise gestures into four primary classes (Stop, Go, Thank & Greet, and No Gesture) and extract 76 static and dynamic features from normalised keypoints. Our analysis demonstrates that hand position and movement velocity are especially discriminative in distinguishing between gesture classes, achieving a classification accuracy score of 87%. These findings not only improve the perceptual capabilities of AV systems but also contribute to the broader understanding of pedestrian behaviour in traffic contexts.

  </details>



- **TriC-Motion: Tri-Domain Causal Modeling Grounded Text-to-Motion Generation**  
  Yiyang Cao, Yunze Deng, Ziyu Lin, Bin Feng, Xinggang Wang, Wenyu Liu, Dandan Zheng, Jingdong Chen  
  _2026-02-09_ · https://arxiv.org/abs/2602.08462v1  
  <details><summary>Abstract</summary>

  Text-to-motion generation, a rapidly evolving field in computer vision, aims to produce realistic and text-aligned motion sequences. Current methods primarily focus on spatial-temporal modeling or independent frequency domain analysis, lacking a unified framework for joint optimization across spatial, temporal, and frequency domains. This limitation hinders the model's ability to leverage information from all domains simultaneously, leading to suboptimal generation quality. Additionally, in motion generation frameworks, motion-irrelevant cues caused by noise are often entangled with features that contribute positively to generation, thereby leading to motion distortion. To address these issues, we propose Tri-Domain Causal Text-to-Motion Generation (TriC-Motion), a novel diffusion-based framework integrating spatial-temporal-frequency-domain modeling with causal intervention. TriC-Motion includes three core modeling modules for domain-specific modeling, namely Temporal Motion Encoding, Spatial Topology Modeling, and Hybrid Frequency Analysis. After comprehensive modeling, a Score-guided Tri-domain Fusion module integrates valuable information from the triple domains, simultaneously ensuring temporal consistency, spatial topology, motion trends, and dynamics. Moreover, the Causality-based Counterfactual Motion Disentangler is meticulously designed to expose motion-irrelevant cues to eliminate noise, disentangling the real modeling contributions of each domain for superior generation. Extensive experimental results validate that TriC-Motion achieves superior performance compared to state-of-the-art methods, attaining an outstanding R@1 of 0.612 on the HumanML3D dataset. These results demonstrate its capability to generate high-fidelity, coherent, diverse, and text-aligned motion sequences. Code is available at: https://caoyiyang1105.github.io/TriC-Motion/.

  </details>



- **SteerVLA: Steering Vision-Language-Action Models in Long-Tail Driving Scenarios**  
  Tian Gao, Celine Tan, Catherine Glossop, Timothy Gao, Jiankai Sun, Kyle Stachowicz, Shirley Wu, Oier Mees, Dorsa Sadigh, Sergey Levine, et al.  
  _2026-02-09_ · https://arxiv.org/abs/2602.08440v1  
  <details><summary>Abstract</summary>

  A fundamental challenge in autonomous driving is the integration of high-level, semantic reasoning for long-tail events with low-level, reactive control for robust driving. While large vision-language models (VLMs) trained on web-scale data offer powerful common-sense reasoning, they lack the grounded experience necessary for safe vehicle control. We posit that an effective autonomous agent should leverage the world knowledge of VLMs to guide a steerable driving policy toward robust control in driving scenarios. To this end, we propose SteerVLA, which leverages the reasoning capabilities of VLMs to produce fine-grained language instructions that steer a vision-language-action (VLA) driving policy. Key to our method is this rich language interface between the high-level VLM and low-level VLA, which allows the high-level policy to more effectively ground its reasoning in the control outputs of the low-level policy. To provide fine-grained language supervision aligned with vehicle control, we leverage a VLM to augment existing driving data with detailed language annotations, which we find to be essential for effective reasoning and steerability. We evaluate SteerVLA on a challenging closed-loop benchmark, where it outperforms state-of-the-art methods by 4.77 points in overall driving score and by 8.04 points on a long-tail subset. The project website is available at: https://steervla.github.io/.

  </details>



- **Demo-ICL: In-Context Learning for Procedural Video Knowledge Acquisition**  
  Yuhao Dong, Shulin Tian, Shuai Liu, Shuangrui Ding, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Jiaqi Wang, Ziwei Liu  
  _2026-02-09_ · https://arxiv.org/abs/2602.08439v1  
  <details><summary>Abstract</summary>

  Despite the growing video understanding capabilities of recent Multimodal Large Language Models (MLLMs), existing video benchmarks primarily assess understanding based on models' static, internal knowledge, rather than their ability to learn and adapt from dynamic, novel contexts from few examples. To bridge this gap, we present Demo-driven Video In-Context Learning, a novel task focused on learning from in-context demonstrations to answer questions about the target videos. Alongside this, we propose Demo-ICL-Bench, a challenging benchmark designed to evaluate demo-driven video in-context learning capabilities. Demo-ICL-Bench is constructed from 1200 instructional YouTube videos with associated questions, from which two types of demonstrations are derived: (i) summarizing video subtitles for text demonstration; and (ii) corresponding instructional videos as video demonstrations. To effectively tackle this new challenge, we develop Demo-ICL, an MLLM with a two-stage training strategy: video-supervised fine-tuning and information-assisted direct preference optimization, jointly enhancing the model's ability to learn from in-context examples. Extensive experiments with state-of-the-art MLLMs confirm the difficulty of Demo-ICL-Bench, demonstrate the effectiveness of Demo-ICL, and thereby unveil future research directions.

  </details>



- **Bi-Adapt: Few-shot Bimanual Adaptation for Novel Categories of 3D Objects via Semantic Correspondence**  
  Jinxian Zhou, Ruihai Wu, Yiwei Liu, Yiwen Hou, Xunzhe Zhou, Checheng Yu, Licheng Zhong, Lin Shao  
  _2026-02-09_ · https://arxiv.org/abs/2602.08425v1  
  <details><summary>Abstract</summary>

  Bimanual manipulation is imperative yet challenging for robots to execute complex tasks, requiring coordinated collaboration between two arms. However, existing methods for bimanual manipulation often rely on costly data collection and training, struggling to generalize to unseen objects in novel categories efficiently. In this paper, we present Bi-Adapt, a novel framework designed for efficient generalization for bimanual manipulation via semantic correspondence. Bi-Adapt achieves cross-category affordance mapping by leveraging the strong capability of vision foundation models. Fine-tuning with restricted data on novel categories, Bi-Adapt exhibits notable generalization to out-of-category objects in a zero-shot manner. Extensive experiments conducted in both simulation and real-world environments validate the effectiveness of our approach and demonstrate its high efficiency, achieving a high success rate on different benchmark tasks across novel categories with limited data. Project website: https://biadapt-project.github.io/

  </details>



- **Decentralized Intent-Based Multi-Robot Task Planner with LLM Oracles on Hyperledger Fabric**  
  Farhad Keramat, Salma Salimi, Tomi Westerlund  
  _2026-02-09_ · https://arxiv.org/abs/2602.08421v1  
  <details><summary>Abstract</summary>

  Large language models (LLMs) have opened new opportunities for transforming natural language user intents into executable actions. This capability enables embodied AI agents to perform complex tasks, without involvement of an expert, making human-robot interaction (HRI) more convenient. However these developments raise significant security and privacy challenges such as self-preferencing, where a single LLM service provider dominates the market and uses this power to promote their own preferences. LLM oracles have been recently proposed as a mechanism to decentralize LLMs by executing multiple LLMs from different vendors and aggregating their outputs to obtain a more reliable and trustworthy final result. However, the accuracy of these approaches highly depends on the aggregation method. The current aggregation methods mostly use semantic similarity between various LLM outputs, not suitable for robotic task planning, where the temporal order of tasks is important. To fill the gap, we propose an LLM oracle with a new aggregation method for robotic task planning. In addition, we propose a decentralized multi-robot infrastructure based on Hyperledger Fabric that can host the proposed oracle. The proposed infrastructure enables users to express their natural language intent to the system, which then can be decomposed into subtasks. These subtasks require coordinating different robots from different vendors, while enforcing fine-grained access control management on the data. To evaluate our methodology, we created the SkillChain-RTD benchmark made it publicly available. Our experimental results demonstrate the feasibility of the proposed architecture, and the proposed aggregation method outperforms other aggregation methods currently in use.

  </details>



- **RealSynCol: a high-fidelity synthetic colon dataset for 3D reconstruction applications**  
  Chiara Lena, Davide Milesi, Alessandro Casella, Luca Carlini, Joseph C. Norton, James Martin, Bruno Scaglioni, Keith L. Obstein, Roberto De Sire, Marco Spadaccini, et al.  
  _2026-02-09_ · https://arxiv.org/abs/2602.08397v1  
  <details><summary>Abstract</summary>

  Deep learning has the potential to improve colonoscopy by enabling 3D reconstruction of the colon, providing a comprehensive view of mucosal surfaces and lesions, and facilitating the identification of unexplored areas. However, the development of robust methods is limited by the scarcity of large-scale ground truth data. We propose RealSynCol, a highly realistic synthetic dataset designed to replicate the endoscopic environment. Colon geometries extracted from 10 CT scans were imported into a virtual environment that closely mimics intraoperative conditions and rendered with realistic vascular textures. The resulting dataset comprises 28\,130 frames, paired with ground truth depth maps, optical flow, 3D meshes, and camera trajectories. A benchmark study was conducted to evaluate the available synthetic colon datasets for the tasks of depth and pose estimation. Results demonstrate that the high realism and variability of RealSynCol significantly enhance generalization performance on clinical images, proving it to be a powerful tool for developing deep learning algorithms to support endoscopic diagnosis.

  </details>



- **BiManiBench: A Hierarchical Benchmark for Evaluating Bimanual Coordination of Multimodal Large Language Models**  
  Xin Wu, Zhixuan Liang, Yue Ma, Mengkang Hu, Zhiyuan Qin, Xiu Li  
  _2026-02-09_ · https://arxiv.org/abs/2602.08392v1  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLMs) have significantly advanced embodied AI, and using them to benchmark robotic intelligence has become a pivotal trend. However, existing frameworks remain predominantly confined to single-arm manipulation, failing to capture the spatio-temporal coordination required for bimanual tasks like lifting a heavy pot. To address this, we introduce BiManiBench, a hierarchical benchmark evaluating MLLMs across three tiers: fundamental spatial reasoning, high-level action planning, and low-level end-effector control. Our framework isolates unique bimanual challenges, such as arm reachability and kinematic constraints, thereby distinguishing perceptual hallucinations from planning failures. Analysis of over 30 state-of-the-art models reveals that despite high-level reasoning proficiency, MLLMs struggle with dual-arm spatial grounding and control, frequently resulting in mutual interference and sequencing errors. These findings suggest the current paradigm lacks a deep understanding of mutual kinematic constraints, highlighting the need for future research to focus on inter-arm collision-avoidance and fine-grained temporal sequencing.

  </details>



- **Geometric Image Editing via Effects-Sensitive In-Context Inpainting with Diffusion Transformers**  
  Shuo Zhang, Wenzhuo Wu, Huayu Zhang, Jiarong Cheng, Xianghao Zang, Chao Ban, Hao Sun, Zhongjiang He, Tianwei Cao, Kongming Liang, et al.  
  _2026-02-09_ · https://arxiv.org/abs/2602.08388v1  
  <details><summary>Abstract</summary>

  Recent advances in diffusion models have significantly improved image editing. However, challenges persist in handling geometric transformations, such as translation, rotation, and scaling, particularly in complex scenes. Existing approaches suffer from two main limitations: (1) difficulty in achieving accurate geometric editing of object translation, rotation, and scaling; (2) inadequate modeling of intricate lighting and shadow effects, leading to unrealistic results. To address these issues, we propose GeoEdit, a framework that leverages in-context generation through a diffusion transformer module, which integrates geometric transformations for precise object edits. Moreover, we introduce Effects-Sensitive Attention, which enhances the modeling of intricate lighting and shadow effects for improved realism. To further support training, we construct RS-Objects, a large-scale geometric editing dataset containing over 120,000 high-quality image pairs, enabling the model to learn precise geometric editing while generating realistic lighting and shadows. Extensive experiments on public benchmarks demonstrate that GeoEdit consistently outperforms state-of-the-art methods in terms of visual quality, geometric accuracy, and realism.

  </details>



- **E-VAds: An E-commerce Short Videos Understanding Benchmark for MLLMs**  
  Xianjie Liu, Yiman Hu, Liang Wu, Ping Hu, Yixiong Zou, Jian Xu, Bo Zheng  
  _2026-02-09_ · https://arxiv.org/abs/2602.08355v1  
  <details><summary>Abstract</summary>

  E-commerce short videos represent a high-revenue segment of the online video industry characterized by a goal-driven format and dense multi-modal signals. Current models often struggle with these videos because existing benchmarks focus primarily on general-purpose tasks and neglect the reasoning of commercial intent. In this work, we first propose a \textbf{multi-modal information density assessment framework} to quantify the complexity of this domain. Our evaluation reveals that e-commerce content exhibits substantially higher density across visual, audio, and textual modalities compared to mainstream datasets, establishing a more challenging frontier for video understanding. To address this gap, we introduce \textbf{E-commerce Video Ads Benchmark (E-VAds)}, which is the first benchmark specifically designed for e-commerce short video understanding. We curated 3,961 high-quality videos from Taobao covering a wide range of product categories and used a multi-agent system to generate 19,785 open-ended Q&A pairs. These questions are organized into two primary dimensions, namely Perception and Cognition and Reasoning, which consist of five distinct tasks. Finally, we develop \textbf{E-VAds-R1}, an RL-based reasoning model featuring a multi-grained reward design called \textbf{MG-GRPO}. This strategy provides smooth guidance for early exploration while creating a non-linear incentive for expert-level precision. Experimental results demonstrate that E-VAds-R1 achieves a 109.2% performance gain in commercial intent reasoning with only a few hundred training samples.

  </details>



- **What, Whether and How? Unveiling Process Reward Models for Thinking with Images Reasoning**  
  Yujin Zhou, Pengcheng Wen, Jiale Chen, Boqin Yin, Han Zhu, Jiaming Ji, Juntao Dai, Chi-Min Chan, Sirui Han  
  _2026-02-09_ · https://arxiv.org/abs/2602.08346v1  
  <details><summary>Abstract</summary>

  The rapid advancement of Large Vision Language Models (LVLMs) has demonstrated excellent abilities in various visual tasks. Building upon these developments, the thinking with images paradigm has emerged, enabling models to dynamically edit and re-encode visual information at each reasoning step, mirroring human visual processing. However, this paradigm introduces significant challenges as diverse errors may occur during reasoning processes. This necessitates Process Reward Models (PRMs) for distinguishing positive and negative reasoning steps, yet existing benchmarks for PRMs are predominantly text-centric and lack comprehensive assessment under this paradigm. To address these gaps, this work introduces the first comprehensive benchmark specifically designed for evaluating PRMs under the thinking with images paradigm. Our main contributions are: (1) Through extensive analysis of reasoning trajectories and guided search experiments with PRMs, we define 7 fine-grained error types and demonstrate both the necessity for specialized PRMs and the potential for improvement. (2) We construct a comprehensive benchmark comprising 1,206 manually annotated thinking with images reasoning trajectories spanning 4 categories and 16 subcategories for fine-grained evaluation of PRMs. (3) Our experimental analysis reveals that current LVLMs fall short as effective PRMs, exhibiting limited capabilities in visual reasoning process evaluation with significant performance disparities across error types, positive evaluation bias, and sensitivity to reasoning step positions. These findings demonstrate the effectiveness of our benchmark and establish crucial foundations for advancing PRMs in LVLMs.

  </details>



- **UrbanGraphEmbeddings: Learning and Evaluating Spatially Grounded Multimodal Embeddings for Urban Science**  
  Jie Zhang, Xingtong Yu, Yuan Fang, Rudi Stouffs, Zdravko Trivic  
  _2026-02-09_ · https://arxiv.org/abs/2602.08342v1  
  <details><summary>Abstract</summary>

  Learning transferable multimodal embeddings for urban environments is challenging because urban understanding is inherently spatial, yet existing datasets and benchmarks lack explicit alignment between street-view images and urban structure. We introduce UGData, a spatially grounded dataset that anchors street-view images to structured spatial graphs and provides graph-aligned supervision via spatial reasoning paths and spatial context captions, exposing distance, directionality, connectivity, and neighborhood context beyond image content. Building on UGData, we propose UGE, a two-stage training strategy that progressively and stably aligns images, text, and spatial structures by combining instruction-guided contrastive learning with graph-based spatial encoding. We finally introduce UGBench, a comprehensive benchmark to evaluate how spatially grounded embeddings support diverse urban understanding tasks -- including geolocation ranking, image retrieval, urban perception, and spatial grounding. We develop UGE on multiple state-of-the-art VLM backbones, including Qwen2-VL, Qwen2.5-VL, Phi-3-Vision, and LLaVA1.6-Mistral, and train fixed-dimensional spatial embeddings with LoRA tuning. UGE built upon Qwen2.5-VL-7B backbone achieves up to 44% improvement in image retrieval and 30% in geolocation ranking on training cities, and over 30% and 22% gains respectively on held-out cities, demonstrating the effectiveness of explicit spatial grounding for spatially intensive urban tasks.

  </details>



- **CoTZero: Annotation-Free Human-Like Vision Reasoning via Hierarchical Synthetic CoT**  
  Chengyi Du, Yazhe Niu, Dazhong Shen, Luxin Xu  
  _2026-02-09_ · https://arxiv.org/abs/2602.08339v1  
  <details><summary>Abstract</summary>

  Recent advances in vision-language models (VLMs) have markedly improved image-text alignment, yet they still fall short of human-like visual reasoning. A key limitation is that many VLMs rely on surface correlations rather than building logically coherent structured representations, which often leads to missed higher-level semantic structure and non-causal relational understanding, hindering compositional and verifiable reasoning. To address these limitations by introducing human models into the reasoning process, we propose CoTZero, an annotation-free paradigm with two components: (i) a dual-stage data synthesis approach and (ii) a cognition-aligned training method. In the first component, we draw inspiration from neurocognitive accounts of compositional productivity and global-to-local analysis. In the bottom-up stage, CoTZero extracts atomic visual primitives and incrementally composes them into diverse, structured question-reasoning forms. In the top-down stage, it enforces hierarchical reasoning by using coarse global structure to guide the interpretation of local details and causal relations. In the cognition-aligned training component, built on the synthesized CoT data, we introduce Cognitively Coherent Verifiable Rewards (CCVR) in Reinforcement Fine-Tuning (RFT) to further strengthen VLMs' hierarchical reasoning and generalization, providing stepwise feedback on reasoning coherence and factual correctness. Experiments show that CoTZero achieves an F1 score of 83.33 percent on our multi-level semantic inconsistency benchmark with lexical-perturbation negatives, across both in-domain and out-of-domain settings. Ablations confirm that each component contributes to more interpretable and human-aligned visual reasoning.

  </details>



- **UReason: Benchmarking the Reasoning Paradox in Unified Multimodal Models**  
  Cheng Yang, Chufan Shi, Bo Shui, Yaokang Wu, Muzi Tao, Huijuan Wang, Ivan Yee Lee, Yong Liu, Xuezhe Ma, Taylor Berg-Kirkpatrick  
  _2026-02-09_ · https://arxiv.org/abs/2602.08336v1  
  <details><summary>Abstract</summary>

  To elicit capabilities for addressing complex and implicit visual requirements, recent unified multimodal models increasingly adopt chain-of-thought reasoning to guide image generation. However, the actual effect of reasoning on visual synthesis remains unclear. We present UReason, a diagnostic benchmark for reasoning-driven image generation that evaluates whether reasoning can be faithfully executed in pixels. UReason contains 2,000 instances across five task families: Code, Arithmetic, Spatial, Attribute, and Text reasoning. To isolate the role of reasoning traces, we introduce an evaluation framework comparing direct generation, reasoning-guided generation, and de-contextualized generation which conditions only on the refined prompt. Across eight open-source unified models, we observe a consistent Reasoning Paradox: Reasoning traces generally improve performance over direct generation, yet retaining intermediate thoughts as conditioning context often hinders visual synthesis, and conditioning only on the refined prompt yields substantial gains. Our analysis suggests that the bottleneck lies in contextual interference rather than insufficient reasoning capacity. UReason provides a principled testbed for studying reasoning in unified models and motivates future methods that effectively integrate reasoning for visual generation while mitigating interference.

  </details>



- **Benchmarking Autonomous Vehicles: A Driver Foundation Model Framework**  
  Yuxin Zhang, Cheng Wang, Hubert P. H. Shum  
  _2026-02-09_ · https://arxiv.org/abs/2602.08298v1  
  <details><summary>Abstract</summary>

  Autonomous vehicles (AVs) are poised to revolutionize global transportation systems. However, its widespread acceptance and market penetration remain significantly below expectations. This gap is primarily driven by persistent challenges in safety, comfort, commuting efficiency and energy economy when compared to the performance of experienced human drivers. We hypothesize that these challenges can be addressed through the development of a driver foundation model (DFM). Accordingly, we propose a framework for establishing DFMs to comprehensively benchmark AVs. Specifically, we describe a large-scale dataset collection strategy for training a DFM, discuss the core functionalities such a model should possess, and explore potential technical solutions to realize these functionalities. We further present the utility of the DFM across the operational spectrum, from defining human-centric safety envelopes to establishing benchmarks for energy economy. Overall, We aim to formalize the DFM concept and introduce a new paradigm for the systematic specification, verification and validation of AVs.

  </details>



- **Tighnari v2: Mitigating Label Noise and Distribution Shift in Multimodal Plant Distribution Prediction via Mixture of Experts and Weakly Supervised Learning**  
  Haixu Liu, Yufei Wang, Tianxiang Xu, Chuancheng Shi, Hongsheng Xing  
  _2026-02-09_ · https://arxiv.org/abs/2602.08282v1  
  <details><summary>Abstract</summary>

  Large-scale, cross-species plant distribution prediction plays a crucial role in biodiversity conservation, yet modeling efforts in this area still face significant challenges due to the sparsity and bias of observational data. Presence-Absence (PA) data provide accurate and noise-free labels, but are costly to obtain and limited in quantity; Presence-Only (PO) data, by contrast, offer broad spatial coverage and rich spatiotemporal distribution, but suffer from severe label noise in negative samples. To address these real-world constraints, this paper proposes a multimodal fusion framework that fully leverages the strengths of both PA and PO data. We introduce an innovative pseudo-label aggregation strategy for PO data based on the geographic coverage of satellite imagery, enabling geographic alignment between the label space and remote sensing feature space. In terms of model architecture, we adopt Swin Transformer Base as the backbone for satellite imagery, utilize the TabM network for tabular feature extraction, retain the Temporal Swin Transformer for time-series modeling, and employ a stackable serial tri-modal cross-attention mechanism to optimize the fusion of heterogeneous modalities. Furthermore, empirical analysis reveals significant geographic distribution shifts between PA training and test samples, and models trained by directly mixing PO and PA data tend to experience performance degradation due to label noise in PO data. To address this, we draw on the mixture-of-experts paradigm: test samples are partitioned according to their spatial proximity to PA samples, and different models trained on distinct datasets are used for inference and post-processing within each partition. Experiments on the GeoLifeCLEF 2025 dataset demonstrate that our approach achieves superior predictive performance in scenarios with limited PA coverage and pronounced distribution shifts.

  </details>



- **PISCO: Precise Video Instance Insertion with Sparse Control**  
  Xiangbo Gao, Renjie Li, Xinghao Chen, Yuheng Wu, Suofei Feng, Qing Yin, Zhengzhong Tu  
  _2026-02-09_ · https://arxiv.org/abs/2602.08277v1  
  <details><summary>Abstract</summary>

  The landscape of AI video generation is undergoing a pivotal shift: moving beyond general generation - which relies on exhaustive prompt-engineering and "cherry-picking" - towards fine-grained, controllable generation and high-fidelity post-processing. In professional AI-assisted filmmaking, it is crucial to perform precise, targeted modifications. A cornerstone of this transition is video instance insertion, which requires inserting a specific instance into existing footage while maintaining scene integrity. Unlike traditional video editing, this task demands several requirements: precise spatial-temporal placement, physically consistent scene interaction, and the faithful preservation of original dynamics - all achieved under minimal user effort. In this paper, we propose PISCO, a video diffusion model for precise video instance insertion with arbitrary sparse keyframe control. PISCO allows users to specify a single keyframe, start-and-end keyframes, or sparse keyframes at arbitrary timestamps, and automatically propagates object appearance, motion, and interaction. To address the severe distribution shift induced by sparse conditioning in pretrained video diffusion models, we introduce Variable-Information Guidance for robust conditioning and Distribution-Preserving Temporal Masking to stabilize temporal generation, together with geometry-aware conditioning for realistic scene adaptation. We further construct PISCO-Bench, a benchmark with verified instance annotations and paired clean background videos, and evaluate performance using both reference-based and reference-free perceptual metrics. Experiments demonstrate that PISCO consistently outperforms strong inpainting and video editing baselines under sparse control, and exhibits clear, monotonic performance improvements as additional control signals are provided. Project page: xiangbogaobarry.github.io/PISCO.

  </details>



- **Informative Object-centric Next Best View for Object-aware 3D Gaussian Splatting in Cluttered Scenes**  
  Seunghoon Jeong, Eunho Lee, Jeongyun Kim, Ayoung Kim  
  _2026-02-09_ · https://arxiv.org/abs/2602.08266v1  
  <details><summary>Abstract</summary>

  In cluttered scenes with inevitable occlusions and incomplete observations, selecting informative viewpoints is essential for building a reliable representation. In this context, 3D Gaussian Splatting (3DGS) offers a distinct advantage, as it can explicitly guide the selection of subsequent viewpoints and then refine the representation with new observations. However, existing approaches rely solely on geometric cues, neglect manipulation-relevant semantics, and tend to prioritize exploitation over exploration. To tackle these limitations, we introduce an instance-aware Next Best View (NBV) policy that prioritizes underexplored regions by leveraging object features. Specifically, our object-aware 3DGS distills instancelevel information into one-hot object vectors, which are used to compute confidence-weighted information gain that guides the identification of regions associated with erroneous and uncertain Gaussians. Furthermore, our method can be easily adapted to an object-centric NBV, which focuses view selection on a target object, thereby improving reconstruction robustness to object placement. Experiments demonstrate that our NBV policy reduces depth error by up to 77.14% on the synthetic dataset and 34.10% on the real-world GraspNet dataset compared to baselines. Moreover, compared to targeting the entire scene, performing NBV on a specific object yields an additional reduction of 25.60% in depth error for that object. We further validate the effectiveness of our approach through real-world robotic manipulation tasks.

  </details>



- **Moving Beyond Functional Connectivity: Time-Series Modeling for fMRI-Based Brain Disorder Classification**  
  Guoqi Yu, Xiaowei Hu, Angelica I. Aviles-Rivero, Anqi Qiu, Shujun Wang  
  _2026-02-09_ · https://arxiv.org/abs/2602.08262v1  
  <details><summary>Abstract</summary>

  Functional magnetic resonance imaging (fMRI) enables non-invasive brain disorder classification by capturing blood-oxygen-level-dependent (BOLD) signals. However, most existing methods rely on functional connectivity (FC) via Pearson correlation, which reduces 4D BOLD signals to static 2D matrices, discarding temporal dynamics and capturing only linear inter-regional relationships. In this work, we benchmark state-of-the-art temporal models (e.g., time-series models such as PatchTST, TimesNet, and TimeMixer) on raw BOLD signals across five public datasets. Results show these models consistently outperform traditional FC-based approaches, highlighting the value of directly modeling temporal information such as cycle-like oscillatory fluctuations and drift-like slow baseline trends. Building on this insight, we propose DeCI, a simple yet effective framework that integrates two key principles: (i) Cycle and Drift Decomposition to disentangle cycle and drift within each ROI (Region of Interest); and (ii) Channel-Independence to model each ROI separately, improving robustness and reducing overfitting. Extensive experiments demonstrate that DeCI achieves superior classification accuracy and generalization compared to both FC-based and temporal baselines. Our findings advocate for a shift toward end-to-end temporal modeling in fMRI analysis to better capture complex brain dynamics. The code is available at https://github.com/Levi-Ackman/DeCI.

  </details>



- **A Unified Framework for Multimodal Image Reconstruction and Synthesis using Denoising Diffusion Models**  
  Weijie Gan, Xucheng Wang, Tongyao Wang, Wenshang Wang, Chunwei Ying, Yuyang Hu, Yasheng Chen, Hongyu An, Ulugbek S. Kamilov  
  _2026-02-09_ · https://arxiv.org/abs/2602.08249v1  
  <details><summary>Abstract</summary>

  Image reconstruction and image synthesis are important for handling incomplete multimodal imaging data, but existing methods require various task-specific models, complicating training and deployment workflows. We introduce Any2all, a unified framework that addresses this limitation by formulating these disparate tasks as a single virtual inpainting problem. We train a single, unconditional diffusion model on the complete multimodal data stack. This model is then adapted at inference time to ``inpaint'' all target modalities from any combination of inputs of available clean images or noisy measurements. We validated Any2all on a PET/MR/CT brain dataset. Our results show that Any2all can achieve excellent performance on both multimodal reconstruction and synthesis tasks, consistently yielding images with competitive distortion-based performance and superior perceptual quality over specialized methods.

  </details>



- **STEP: Warm-Started Visuomotor Policies with Spatiotemporal Consistency Prediction**  
  Jinhao Li, Yuxuan Cong, Yingqiao Wang, Hao Xia, Shan Huang, Yijia Zhang, Ningyi Xu, Guohao Dai  
  _2026-02-09_ · https://arxiv.org/abs/2602.08245v1  
  <details><summary>Abstract</summary>

  Diffusion policies have recently emerged as a powerful paradigm for visuomotor control in robotic manipulation due to their ability to model the distribution of action sequences and capture multimodality. However, iterative denoising leads to substantial inference latency, limiting control frequency in real-time closed-loop systems. Existing acceleration methods either reduce sampling steps, bypass diffusion through direct prediction, or reuse past actions, but often struggle to jointly preserve action quality and achieve consistently low latency. In this work, we propose STEP, a lightweight spatiotemporal consistency prediction mechanism to construct high-quality warm-start actions that are both distributionally close to the target action and temporally consistent, without compromising the generative capability of the original diffusion policy. Then, we propose a velocity-aware perturbation injection mechanism that adaptively modulates actuation excitation based on temporal action variation to prevent execution stall especially for real-world tasks. We further provide a theoretical analysis showing that the proposed prediction induces a locally contractive mapping, ensuring convergence of action errors during diffusion refinement. We conduct extensive evaluations on nine simulated benchmarks and two real-world tasks. Notably, STEP with 2 steps can achieve an average 21.6% and 27.5% higher success rate than BRIDGER and DDIM on the RoboMimic benchmark and real-world tasks, respectively. These results demonstrate that STEP consistently advances the Pareto frontier of inference latency and success rate over existing methods.

  </details>



- **When and How Much to Imagine: Adaptive Test-Time Scaling with World Models for Visual Spatial Reasoning**  
  Shoubin Yu, Yue Zhang, Zun Wang, Jaehong Yoon, Huaxiu Yao, Mingyu Ding, Mohit Bansal  
  _2026-02-09_ · https://arxiv.org/abs/2602.08236v1  
  <details><summary>Abstract</summary>

  Despite rapid progress in Multimodal Large Language Models (MLLMs), visual spatial reasoning remains unreliable when correct answers depend on how a scene would appear under unseen or alternative viewpoints. Recent work addresses this by augmenting reasoning with world models for visual imagination, but questions such as when imagination is actually necessary, how much of it is beneficial, and when it becomes harmful, remain poorly understood. In practice, indiscriminate imagination can increase computation and even degrade performance by introducing misleading evidence. In this work, we present an in-depth analysis of test-time visual imagination as a controllable resource for spatial reasoning. We study when static visual evidence is sufficient, when imagination improves reasoning, and how excessive or unnecessary imagination affects accuracy and efficiency. To support this analysis, we introduce AVIC, an adaptive test-time framework with world models that explicitly reasons about the sufficiency of current visual evidence before selectively invoking and scaling visual imagination. Across spatial reasoning benchmarks (SAT, MMSI) and an embodied navigation benchmark (R2R), our results reveal clear scenarios where imagination is critical, marginal, or detrimental, and show that selective control can match or outperform fixed imagination strategies with substantially fewer world-model calls and language tokens. Overall, our findings highlight the importance of analyzing and controlling test-time imagination for efficient and reliable spatial reasoning.

  </details>



- **DAS-SK: An Adaptive Model Integrating Dual Atrous Separable and Selective Kernel CNN for Agriculture Semantic Segmentation**  
  Mei Ling Chee, Thangarajah Akilan, Aparna Ravindra Phalke, Kanchan Keisham  
  _2026-02-09_ · https://arxiv.org/abs/2602.08168v1  
  <details><summary>Abstract</summary>

  Semantic segmentation in high-resolution agricultural imagery demands models that strike a careful balance between accuracy and computational efficiency to enable deployment in practical systems. In this work, we propose DAS-SK, a novel lightweight architecture that retrofits selective kernel convolution (SK-Conv) into the dual atrous separable convolution (DAS-Conv) module to strengthen multi-scale feature learning. The model further enhances the atrous spatial pyramid pooling (ASPP) module, enabling the capture of fine-grained local structures alongside global contextual information. Built upon a modified DeepLabV3 framework with two complementary backbones - MobileNetV3-Large and EfficientNet-B3, the DAS-SK model mitigates limitations associated with large dataset requirements, limited spectral generalization, and the high computational cost that typically restricts deployment on UAVs and other edge devices. Comprehensive experiments across three benchmarks: LandCover.ai, VDD, and PhenoBench, demonstrate that DAS-SK consistently achieves state-of-the-art performance, while being more efficient than CNN-, transformer-, and hybrid-based competitors. Notably, DAS-SK requires up to 21x fewer parameters and 19x fewer GFLOPs than top-performing transformer models. These findings establish DAS-SK as a robust, efficient, and scalable solution for real-time agricultural robotics and high-resolution remote sensing, with strong potential for broader deployment in other vision domains.

  </details>



- **Self-Supervised Bootstrapping of Action-Predictive Embodied Reasoning**  
  Milan Ganai, Katie Luo, Jonas Frey, Clark Barrett, Marco Pavone  
  _2026-02-09_ · https://arxiv.org/abs/2602.08167v1  
  <details><summary>Abstract</summary>

  Embodied Chain-of-Thought (CoT) reasoning has significantly enhanced Vision-Language-Action (VLA) models, yet current methods rely on rigid templates to specify reasoning primitives (e.g., objects in the scene, high-level plans, structural affordances). These templates can force policies to process irrelevant information that distracts from critical action-prediction signals. This creates a bottleneck: without successful policies, we cannot verify reasoning quality; without quality reasoning, we cannot build robust policies. We introduce R&B-EnCoRe, which enables models to bootstrap embodied reasoning from internet-scale knowledge through self-supervised refinement. By treating reasoning as a latent variable within importance-weighted variational inference, models can generate and distill a refined reasoning training dataset of embodiment-specific strategies without external rewards, verifiers, or human annotation. We validate R&B-EnCoRe across manipulation (Franka Panda in simulation, WidowX in hardware), legged navigation (bipedal, wheeled, bicycle, quadruped), and autonomous driving embodiments using various VLA architectures with 1B, 4B, 7B, and 30B parameters. Our approach achieves 28% gains in manipulation success, 101% improvement in navigation scores, and 21% reduction in collision-rate metric over models that indiscriminately reason about all available primitives. R&B-EnCoRe enables models to distill reasoning that is predictive of successful control, bypassing manual annotation engineering while grounding internet-scale knowledge in physical execution.

  </details>



- **Building Damage Detection using Satellite Images and Patch-Based Transformer Methods**  
  Smriti Siva, Jan Cross-Zamirski  
  _2026-02-08_ · https://arxiv.org/abs/2602.08117v1  
  <details><summary>Abstract</summary>

  Rapid building damage assessment is critical for post-disaster response. Damage classification models built on satellite imagery provide a scalable means of obtaining situational awareness. However, label noise and severe class imbalance in satellite data create major challenges. The xBD dataset offers a standardized benchmark for building-level damage across diverse geographic regions. In this study, we evaluate Vision Transformer (ViT) model performance on the xBD dataset, specifically investigating how these models distinguish between types of structural damage when training on noisy, imbalanced data. In this study, we specifically evaluate DINOv2-small and DeiT for multi-class damage classification. We propose a targeted patch-based pre-processing pipeline to isolate structural features and minimize background noise in training. We adopt a frozen-head fine-tuning strategy to keep computational requirements manageable. Model performance is evaluated through accuracy, precision, recall, and macro-averaged F1 scores. We show that small ViT architectures with our novel training method achieves competitive macro-averaged F1 relative to prior CNN baselines for disaster classification.

  </details>



- **MMLSv2: A Multimodal Dataset for Martian Landslide Detection in Remote Sensing Imagery**  
  Sidike Paheding, Abel Reyes-Angulo, Leo Thomas Ramos, Angel D. Sappa, Rajaneesh A., Hiral P. B., Sajin Kumar K. S., Thomas Oommen  
  _2026-02-08_ · https://arxiv.org/abs/2602.08112v1  
  <details><summary>Abstract</summary>

  We present MMLSv2, a dataset for landslide segmentation on Martian surfaces. MMLSv2 consists of multimodal imagery with seven bands: RGB, digital elevation model, slope, thermal inertia, and grayscale channels. MMLSv2 comprises 664 images distributed across training, validation, and test splits. In addition, an isolated test set of 276 images from a geographically disjoint region from the base dataset is released to evaluate spatial generalization. Experiments conducted with multiple segmentation models show that the dataset supports stable training and achieves competitive performance, while still posing challenges in fragmented, elongated, and small-scale landslide regions. Evaluation on the isolated test set leads to a noticeable performance drop, indicating increased difficulty and highlighting its value for assessing model robustness and generalization beyond standard in-distribution settings. Dataset will be available at: https://github.com/MAIN-Lab/MMLS_v2

  </details>


