# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-02-02 07:16 UTC_

Total papers shown: **28**


---

- **Denoising the Deep Sky: Physics-Based CCD Noise Formation for Astronomical Imaging**  
  Shuhong Liu, Xining Ge, Ziying Gu, Lin Gu, Ziteng Cui, Xuangeng Chu, Jun Liu, Dong Li, Tatsuya Harada  
  _2026-01-30_ · https://arxiv.org/abs/2601.23276v1  
  <details><summary>Abstract</summary>

  Astronomical imaging remains noise-limited under practical observing constraints, while standard calibration pipelines mainly remove structured artifacts and leave stochastic noise largely unresolved. Learning-based denoising is promising, yet progress is hindered by scarce paired training data and the need for physically interpretable and reproducible models in scientific workflows. We propose a physics-based noise synthesis framework tailored to CCD noise formation. The pipeline models photon shot noise, photo-response non-uniformity, dark-current noise, readout effects, and localized outliers arising from cosmic-ray hits and hot pixels. To obtain low-noise inputs for synthesis, we average multiple unregistered exposures to produce high-SNR bases. Realistic noisy counterparts synthesized from these bases using our noise model enable the construction of abundant paired datasets for supervised learning. We further introduce a real-world dataset across multi-bands acquired with two twin ground-based telescopes, providing paired raw frames and instrument-pipeline calibrated frames, together with calibration data and stacked high-SNR bases for real-world evaluation.

  </details>



- **IRL-DAL: Safe and Adaptive Trajectory Planning for Autonomous Driving via Energy-Guided Diffusion Models**  
  Seyed Ahmad Hosseini Miangoleh, Amin Jalal Aghdasian, Farzaneh Abdollahi  
  _2026-01-30_ · https://arxiv.org/abs/2601.23266v1  
  <details><summary>Abstract</summary>

  This paper proposes a novel inverse reinforcement learning framework using a diffusion-based adaptive lookahead planner (IRL-DAL) for autonomous vehicles. Training begins with imitation from an expert finite state machine (FSM) controller to provide a stable initialization. Environment terms are combined with an IRL discriminator signal to align with expert goals. Reinforcement learning (RL) is then performed with a hybrid reward that combines diffuse environmental feedback and targeted IRL rewards. A conditional diffusion model, which acts as a safety supervisor, plans safe paths. It stays in its lane, avoids obstacles, and moves smoothly. Then, a learnable adaptive mask (LAM) improves perception. It shifts visual attention based on vehicle speed and nearby hazards. After FSM-based imitation, the policy is fine-tuned with Proximal Policy Optimization (PPO). Training is run in the Webots simulator with a two-stage curriculum. A 96\% success rate is reached, and collisions are reduced to 0.05 per 1k steps, marking a new benchmark for safe navigation. By applying the proposed approach, the agent not only drives in lane but also handles unsafe conditions at an expert level, increasing robustness.We make our code publicly available.

  </details>



- **Training-Free Test-Time Adaptation with Brownian Distance Covariance in Vision-Language Models**  
  Yi Zhang, Chun-Wun Cheng, Angelica I. Aviles-Rivero, Zhihai He, Liang-Jie Zhang  
  _2026-01-30_ · https://arxiv.org/abs/2601.23253v1  
  <details><summary>Abstract</summary>

  Vision-language models suffer performance degradation under domain shift, limiting real-world applicability. Existing test-time adaptation methods are computationally intensive, rely on back-propagation, and often focus on single modalities. To address these issues, we propose Training-free Test-Time Adaptation with Brownian Distance Covariance (TaTa). TaTa leverages Brownian Distance Covariance-a powerful statistical measure that captures both linear and nonlinear dependencies via pairwise distances-to dynamically adapt VLMs to new domains without training or back-propagation. This not only improves efficiency but also enhances stability by avoiding disruptive weight updates. TaTa further integrates attribute-enhanced prompting to improve vision-language inference with descriptive visual cues. Combined with dynamic clustering and pseudo-label refinement, it effectively recalibrates the model for novel visual contexts. Experiments across diverse datasets show that TaTa significantly reduces computational cost while achieving state-of-the-art performance in domain and cross-dataset generalization.

  </details>



- **Structured Over Scale: Learning Spatial Reasoning from Educational Video**  
  Bishoy Galoaa, Xiangyu Bai, Sarah Ostadabbas  
  _2026-01-30_ · https://arxiv.org/abs/2601.23251v1  
  <details><summary>Abstract</summary>

  Vision-language models (VLMs) demonstrate impressive performance on standard video understanding benchmarks yet fail systematically on simple reasoning tasks that preschool children can solve, including counting, spatial reasoning, and compositional understanding. We hypothesize that the pedagogically-structured content of educational videos provides an ideal training signal for improving these capabilities. We introduce DoraVQA, a dataset of 5,344 question-answer pairs automatically extracted from 8 seasons of Dora the Explorer with precise timestamp alignment. Each episode follows a consistent \textit{context-question-pause-answer} structure that creates a self-contained learning environment analogous to interactive tutoring. We fine-tune both Qwen2 and Qwen3 using Group Relative Policy Optimization (GRPO), leveraging the clear correctness signals and structured reasoning traces inherent in educational content. Despite training exclusively on 38 hours of children's educational videos, our approach achieves improvements of 8-14 points on DoraVQA and state-of-the-art 86.16\% on CVBench, with strong transfer to Video-MME and NExT-QA, demonstrating effective generalization from narrow pedagogical content to broad multimodal understanding. Through cross-domain benchmarks, we show that VLMs can perform tasks that require robust reasoning learned from structured educational content, suggesting that content structure matters as much as content scale.

  </details>



- **ShotFinder: Imagination-Driven Open-Domain Video Shot Retrieval via Web Search**  
  Tao Yu, Haopeng Jin, Hao Wang, Shenghua Chai, Yujia Yang, Junhao Gong, Jiaming Guo, Minghui Zhang, Xinlong Chen, Zhenghao Zhang, et al.  
  _2026-01-30_ · https://arxiv.org/abs/2601.23232v1  
  <details><summary>Abstract</summary>

  In recent years, large language models (LLMs) have made rapid progress in information retrieval, yet existing research has mainly focused on text or static multimodal settings. Open-domain video shot retrieval, which involves richer temporal structure and more complex semantics, still lacks systematic benchmarks and analysis. To fill this gap, we introduce ShotFinder, a benchmark that formalizes editing requirements as keyframe-oriented shot descriptions and introduces five types of controllable single-factor constraints: Temporal order, Color, Visual style, Audio, and Resolution. We curate 1,210 high-quality samples from YouTube across 20 thematic categories, using large models for generation with human verification. Based on the benchmark, we propose ShotFinder, a text-driven three-stage retrieval and localization pipeline: (1) query expansion via video imagination, (2) candidate video retrieval with a search engine, and (3) description-guided temporal localization. Experiments on multiple closed-source and open-source models reveal a significant gap to human performance, with clear imbalance across constraints: temporal localization is relatively tractable, while color and visual style remain major challenges. These results reveal that open-domain video shot retrieval is still a critical capability that multimodal large models have yet to overcome.

  </details>



- **Region-Normalized DPO for Medical Image Segmentation under Noisy Judges**  
  Hamza Kalisch, Constantin Seibold, Jens Kleesiek, Ken Herrmann, Frederic Jonske  
  _2026-01-30_ · https://arxiv.org/abs/2601.23222v1  
  <details><summary>Abstract</summary>

  While dense pixel-wise annotations remain the gold standard for medical image segmentation, they are costly to obtain and limit scalability. In contrast, many deployed systems already produce inexpensive automatic quality-control (QC) signals like model agreement, uncertainty measures, or learned mask-quality scores which can be used for further model training without additional ground-truth annotation. However, these signals can be noisy and biased, making preference-based fine-tuning susceptible to harmful updates. We study Direct Preference Optimization (DPO) for segmentation from such noisy judges using proposals generated by a supervised base segmenter trained on a small labeled set. We find that outcomes depend strongly on how preference pairs are mined: selecting the judge's top-ranked proposal can improve peak performance when the judge is reliable, but can amplify harmful errors under weaker judges. We propose Region-Normalized DPO (RN-DPO), a segmentation-aware objective which normalizes preference updates by the size of the disagreement region between masks, reducing the leverage of harmful comparisons and improving optimization stability. Across two medical datasets and multiple regimes, RN-DPO improves sustained performance and stabilizes preference-based fine-tuning, outperforming standard DPO and strong baselines without requiring additional pixel annotations.

  </details>



- **Med-Scout: Curing MLLMs' Geometric Blindness in Medical Perception via Geometry-Aware RL Post-Training**  
  Anglin Liu, Ruichao Chen, Yi Lu, Hongxia Xu, Jintai Chen  
  _2026-01-30_ · https://arxiv.org/abs/2601.23220v1  
  <details><summary>Abstract</summary>

  Despite recent Multimodal Large Language Models (MLLMs)' linguistic prowess in medical diagnosis, we find even state-of-the-art MLLMs suffer from a critical perceptual deficit: geometric blindness. This failure to ground outputs in objective geometric constraints leads to plausible yet factually incorrect hallucinations, rooted in training paradigms that prioritize linguistic fluency over geometric fidelity. This paper introduces Med-Scout, a novel framework that "cures" this blindness via Reinforcement Learning (RL) that leverages the intrinsic geometric logic latent within unlabeled medical images. Instead of relying on costly expert annotations, Med-Scout derives verifiable supervision signals through three strategic proxy tasks: Hierarchical Scale Localization, Topological Jigsaw Reconstruction, and Anomaly Consistency Detection. To rigorously quantify this deficit, we present Med-Scout-Bench, a new benchmark specifically designed to evaluate geometric perception. Extensive evaluations show that Med-Scout significantly mitigates geometric blindness, outperforming leading proprietary and open-source MLLMs by over 40% on our benchmark. Furthermore, this enhanced geometric perception generalizes to broader medical understanding, achieving superior results on radiological and comprehensive medical VQA tasks.

  </details>



- **FlowCalib: LiDAR-to-Vehicle Miscalibration Detection using Scene Flows**  
  Ilir Tahiraj, Peter Wittal, Markus Lienkamp  
  _2026-01-30_ · https://arxiv.org/abs/2601.23107v1  
  <details><summary>Abstract</summary>

  Accurate sensor-to-vehicle calibration is essential for safe autonomous driving. Angular misalignments of LiDAR sensors can lead to safety-critical issues during autonomous operation. However, current methods primarily focus on correcting sensor-to-sensor errors without considering the miscalibration of individual sensors that cause these errors in the first place. We introduce FlowCalib, the first framework that detects LiDAR-to-vehicle miscalibration using motion cues from the scene flow of static objects. Our approach leverages the systematic bias induced by rotational misalignment in the flow field generated from sequential 3D point clouds, eliminating the need for additional sensors. The architecture integrates a neural scene flow prior for flow estimation and incorporates a dual-branch detection network that fuses learned global flow features with handcrafted geometric descriptors. These combined representations allow the system to perform two complementary binary classification tasks: a global binary decision indicating whether misalignment is present and separate, axis-specific binary decisions indicating whether each rotational axis is misaligned. Experiments on the nuScenes dataset demonstrate FlowCalib's ability to robustly detect miscalibration, establishing a benchmark for sensor-to-vehicle miscalibration detection.

  </details>



- **HierLoc: Hyperbolic Entity Embeddings for Hierarchical Visual Geolocation**  
  Hari Krishna Gadi, Daniel Matos, Hongyi Luo, Lu Liu, Yongliang Wang, Yanfeng Zhang, Liqiu Meng  
  _2026-01-30_ · https://arxiv.org/abs/2601.23064v1  
  <details><summary>Abstract</summary>

  Visual geolocalization, the task of predicting where an image was taken, remains challenging due to global scale, visual ambiguity, and the inherently hierarchical structure of geography. Existing paradigms rely on either large-scale retrieval, which requires storing a large number of image embeddings, grid-based classifiers that ignore geographic continuity, or generative models that diffuse over space but struggle with fine detail. We introduce an entity-centric formulation of geolocation that replaces image-to-image retrieval with a compact hierarchy of geographic entities embedded in Hyperbolic space. Images are aligned directly to country, region, subregion, and city entities through Geo-Weighted Hyperbolic contrastive learning by directly incorporating haversine distance into the contrastive objective. This hierarchical design enables interpretable predictions and efficient inference with 240k entity embeddings instead of over 5 million image embeddings on the OSV5M benchmark, on which our method establishes a new state-of-the-art performance. Compared to the current methods in the literature, it reduces mean geodesic error by 19.5\%, while improving the fine-grained subregion accuracy by 43%. These results demonstrate that geometry-aware hierarchical embeddings provide a scalable and conceptually new alternative for global image geolocation.

  </details>



- **Leveraging Multi-Rater Annotations to Calibrate Object Detectors in Microscopy Imaging**  
  Francesco Campi, Lucrezia Tondo, Ekin Karabati, Johannes Betge, Marie Piraud  
  _2026-01-30_ · https://arxiv.org/abs/2601.23007v1  
  <details><summary>Abstract</summary>

  Deep learning-based object detectors have achieved impressive performance in microscopy imaging, yet their confidence estimates often lack calibration, limiting their reliability for biomedical applications. In this work, we introduce a new approach to improve model calibration by leveraging multi-rater annotations. We propose to train separate models on the annotations from single experts and aggregate their predictions to emulate consensus. This improves upon label sampling strategies, where models are trained on mixed annotations, and offers a more principled way to capture inter-rater variability. Experiments on a colorectal organoid dataset annotated by two experts demonstrate that our rater-specific ensemble strategy improves calibration performance while maintaining comparable detection accuracy. These findings suggest that explicitly modelling rater disagreement can lead to more trustworthy object detectors in biomedical imaging.

  </details>



- **About an Automating Annotation Method for Robot Markers**  
  Wataru Uemura, Takeru Nagashima  
  _2026-01-30_ · https://arxiv.org/abs/2601.22982v1  
  <details><summary>Abstract</summary>

  Factory automation has become increasingly important due to labor shortages, leading to the introduction of autonomous mobile robots for tasks such as material transportation. Markers are commonly used for robot self-localization and object identification. In the RoboCup Logistics League (RCLL), ArUco markers are employed both for robot localization and for identifying processing modules. Conventional recognition relies on OpenCV-based image processing, which detects black-and-white marker patterns. However, these methods often fail under noise, motion blur, defocus, or varying illumination conditions. Deep-learning-based recognition offers improved robustness under such conditions, but requires large amounts of annotated data. Annotation must typically be done manually, as the type and position of objects cannot be detected automatically, making dataset preparation a major bottleneck. In contrast, ArUco markers include built-in recognition modules that provide both ID and positional information, enabling automatic annotation. This paper proposes an automated annotation method for training deep-learning models on ArUco marker images. By leveraging marker detection results obtained from the ArUco module, the proposed approach eliminates the need for manual labeling. A YOLO-based model is trained using the automatically annotated dataset, and its performance is evaluated under various conditions. Experimental results demonstrate that the proposed method improves recognition performance compared with conventional image-processing techniques, particularly for images affected by blur or defocus. Automatic annotation also reduces human effort and ensures consistent labeling quality. Future work will investigate the relationship between confidence thresholds and recognition performance.

  </details>



- **Self-Imitated Diffusion Policy for Efficient and Robust Visual Navigation**  
  Runhua Zhang, Junyi Hou, Changxu Cheng, Qiyi Chen, Tao Wang, Wuyue Zhao  
  _2026-01-30_ · https://arxiv.org/abs/2601.22965v1  
  <details><summary>Abstract</summary>

  Diffusion policies (DP) have demonstrated significant potential in visual navigation by capturing diverse multi-modal trajectory distributions. However, standard imitation learning (IL), which most DP methods rely on for training, often inherits sub-optimality and redundancy from expert demonstrations, thereby necessitating a computationally intensive "generate-then-filter" pipeline that relies on auxiliary selectors during inference. To address these challenges, we propose Self-Imitated Diffusion Policy (SIDP), a novel framework that learns improved planning by selectively imitating a set of trajectories sampled from itself. Specifically, SIDP introduces a reward-guided self-imitation mechanism that encourages the policy to consistently produce high-quality trajectories efficiently, rather than outputs of inconsistent quality, thereby reducing reliance on extensive sampling and post-filtering. During training, we employ a reward-driven curriculum learning paradigm to mitigate inefficient data utility, and goal-agnostic exploration for trajectory augmentation to improve planning robustness. Extensive evaluations on a comprehensive simulation benchmark show that SIDP significantly outperforms previous methods, with real-world experiments confirming its effectiveness across multiple robotic platforms. On Jetson Orin Nano, SIDP delivers a 2.5$\times$ faster inference than the baseline NavDP, i.e., 110ms VS 273ms, enabling efficient real-time deployment.

  </details>



- **Improving Supervised Machine Learning Performance in Optical Quality Control via Generative AI for Dataset Expansion**  
  Dennis Sprute, Hanna Senke, Holger Flatt  
  _2026-01-30_ · https://arxiv.org/abs/2601.22961v1  
  <details><summary>Abstract</summary>

  Supervised machine learning algorithms play a crucial role in optical quality control within industrial production. These approaches require representative datasets for effective model training. However, while non-defective components are frequent, defective parts are rare in production, resulting in highly imbalanced datasets that adversely impact model performance. Existing strategies to address this challenge, such as specialized loss functions or traditional data augmentation techniques, have limitations, including the need for careful hyperparameter tuning or the alteration of only simple image features. Therefore, this work explores the potential of generative artificial intelligence (GenAI) as an alternative method for expanding limited datasets and enhancing supervised machine learning performance. Specifically, we investigate Stable Diffusion and CycleGAN as image generation models, focusing on the segmentation of combine harvester components in thermal images for subsequent defect detection. Our results demonstrate that dataset expansion using Stable Diffusion yields the most significant improvement, enhancing segmentation performance by 4.6 %, resulting in a Mean Intersection over Union (Mean IoU) of 84.6 %.

  </details>



- **MTDrive: Multi-turn Interactive Reinforcement Learning for Autonomous Driving**  
  Xidong Li, Mingyu Guo, Chenchao Xu, Bailin Li, Wenjing Zhu, Yangang Zou, Rui Chen, Zehuan Wang  
  _2026-01-30_ · https://arxiv.org/abs/2601.22930v1  
  <details><summary>Abstract</summary>

  Trajectory planning is a core task in autonomous driving, requiring the prediction of safe and comfortable paths across diverse scenarios. Integrating Multi-modal Large Language Models (MLLMs) with Reinforcement Learning (RL) has shown promise in addressing "long-tail" scenarios. However, existing methods are constrained to single-turn reasoning, limiting their ability to handle complex tasks requiring iterative refinement. To overcome this limitation, we present MTDrive, a multi-turn framework that enables MLLMs to iteratively refine trajectories based on environmental feedback. MTDrive introduces Multi-Turn Group Relative Policy Optimization (mtGRPO), which mitigates reward sparsity by computing relative advantages across turns. We further construct an interactive trajectory understanding dataset from closed-loop simulation to support multi-turn training. Experiments on the NAVSIM benchmark demonstrate superior performance compared to existing methods, validating the effectiveness of our multi-turn reasoning paradigm. Additionally, we implement system-level optimizations to reduce data transfer overhead caused by high-resolution images and multi-turn sequences, achieving 2.5x training throughput. Our data, models, and code will be made available soon.

  </details>



- **Deep in the Jungle: Towards Automating Chimpanzee Population Estimation**  
  Tom Raynes, Otto Brookes, Timm Haucke, Lukas Bösch, Anne-Sophie Crunchant, Hjalmar Kühl, Sara Beery, Majid Mirmehdi, Tilo Burghardt  
  _2026-01-30_ · https://arxiv.org/abs/2601.22917v1  
  <details><summary>Abstract</summary>

  The estimation of abundance and density in unmarked populations of great apes relies on statistical frameworks that require animal-to-camera distance measurements. In practice, acquiring these distances depends on labour-intensive manual interpretation of animal observations across large camera trap video corpora. This study introduces and evaluates an only sparsely explored alternative: the integration of computer vision-based monocular depth estimation (MDE) pipelines directly into ecological camera trap workflows for great ape conservation. Using a real-world dataset of 220 camera trap videos documenting a wild chimpanzee population, we combine two MDE models, Dense Prediction Transformers and Depth Anything, with multiple distance sampling strategies. These components are used to generate detection distance estimates, from which population density and abundance are inferred. Comparative analysis against manually derived ground-truth distances shows that calibrated DPT consistently outperforms Depth Anything. This advantage is observed in both distance estimation accuracy and downstream density and abundance inference. Nevertheless, both models exhibit systematic biases. We show that, given complex forest environments, they tend to overestimate detection distances and consequently underestimate density and abundance relative to conventional manual approaches. We further find that failures in animal detection across distance ranges are a primary factor limiting estimation accuracy. Overall, this work provides a case study that shows MDE-driven camera trap distance sampling is a viable and practical alternative to manual distance estimation. The proposed approach yields population estimates within 22% of those obtained using traditional methods.

  </details>



- **Development of Domain-Invariant Visual Enhancement and Restoration (DIVER) Approach for Underwater Images**  
  Rajini Makam, Sharanya Patil, Dhatri Shankari T M, Suresh Sundaram, Narasimhan Sundararajan  
  _2026-01-30_ · https://arxiv.org/abs/2601.22878v1  
  <details><summary>Abstract</summary>

  Underwater images suffer severe degradation due to wavelength-dependent attenuation, scattering, and illumination non-uniformity that vary across water types and depths. We propose an unsupervised Domain-Invariant Visual Enhancement and Restoration (DIVER) framework that integrates empirical correction with physics-guided modeling for robust underwater image enhancement. DIVER first applies either IlluminateNet for adaptive luminance enhancement or a Spectral Equalization Filter for spectral normalization. An Adaptive Optical Correction Module then refines hue and contrast using channel-adaptive filtering, while Hydro-OpticNet employs physics-constrained learning to compensate for backscatter and wavelength-dependent attenuation. The parameters of IlluminateNet and Hydro-OpticNet are optimized via unsupervised learning using a composite loss function. DIVER is evaluated on eight diverse datasets covering shallow, deep, and highly turbid environments, including both naturally low-light and artificially illuminated scenes, using reference and non-reference metrics. While state-of-the-art methods such as WaterNet, UDNet, and Phaseformer perform reasonably in shallow water, their performance degrades in deep, unevenly illuminated, or artificially lit conditions. In contrast, DIVER consistently achieves best or near-best performance across all datasets, demonstrating strong domain-invariant capability. DIVER yields at least a 9% improvement over SOTA methods in UCIQE. On the low-light SeaThru dataset, where color-palette references enable direct evaluation of color restoration, DIVER achieves at least a 4.9% reduction in GPMAE compared to existing methods. Beyond visual quality, DIVER also improves robotic perception by enhancing ORB-based keypoint repeatability and matching performance, confirming its robustness across diverse underwater environments.

  </details>



- **When Anomalies Depend on Context: Learning Conditional Compatibility for Anomaly Detection**  
  Shashank Mishra, Didier Stricker, Jason Rambach  
  _2026-01-30_ · https://arxiv.org/abs/2601.22868v1  
  <details><summary>Abstract</summary>

  Anomaly detection is often formulated under the assumption that abnormality is an intrinsic property of an observation, independent of context. This assumption breaks down in many real-world settings, where the same object or action may be normal or anomalous depending on latent contextual factors (e.g., running on a track versus on a highway). We revisit \emph{contextual anomaly detection}, classically defined as context-dependent abnormality, and operationalize it in the visual domain, where anomaly labels depend on subject--context compatibility rather than intrinsic appearance. To enable systematic study of this setting, we introduce CAAD-3K, a benchmark that isolates contextual anomalies by controlling subject identity while varying context. We further propose a conditional compatibility learning framework that leverages vision--language representations to model subject--context relationships under limited supervision. Our method substantially outperforms existing approaches on CAAD-3K and achieves state-of-the-art performance on MVTec-AD and VisA, demonstrating that modeling context dependence complements traditional structural anomaly detection. Our code and dataset will be publicly released.

  </details>



- **Neural Clothing Tryer: Customized Virtual Try-On via Semantic Enhancement and Controlling Diffusion Model**  
  Zhijing Yang, Weiwei Zhang, Mingliang Yang, Siyuan Peng, Yukai Shi, Junpeng Tan, Tianshui Chen, Liruo Zhong  
  _2026-01-30_ · https://arxiv.org/abs/2601.22838v1  
  <details><summary>Abstract</summary>

  This work aims to address a novel Customized Virtual Try-ON (Cu-VTON) task, enabling the superimposition of a specified garment onto a model that can be customized in terms of appearance, posture, and additional attributes. Compared with traditional VTON task, it enables users to tailor digital avatars to their individual preferences, thereby enhancing the virtual fitting experience with greater flexibility and engagement. To address this task, we introduce a Neural Clothing Tryer (NCT) framework, which exploits the advanced diffusion models equipped with semantic enhancement and controlling modules to better preserve semantic characterization and textural details of the garment and meanwhile facilitating the flexible editing of the model's postures and appearances. Specifically, NCT introduces a semantic-enhanced module to take semantic descriptions of garments and utilizes a visual-language encoder to learn aligned features across modalities. The aligned features are served as condition input to the diffusion model to enhance the preservation of the garment's semantics. Then, a semantic controlling module is designed to take the garment image, tailored posture image, and semantic description as input to maintain garment details while simultaneously editing model postures, expressions, and various attributes. Extensive experiments on the open available benchmark demonstrate the superior performance of the proposed NCT framework.

  </details>



- **A Comparative Evaluation of Large Vision-Language Models for 2D Object Detection under SOTIF Conditions**  
  Ji Zhou, Yilin Ding, Yongqi Zhao, Jiachen Xu, Arno Eichberger  
  _2026-01-30_ · https://arxiv.org/abs/2601.22830v1  
  <details><summary>Abstract</summary>

  Reliable environmental perception remains one of the main obstacles for safe operation of automated vehicles. Safety of the Intended Functionality (SOTIF) concerns safety risks from perception insufficiencies, particularly under adverse conditions where conventional detectors often falter. While Large Vision-Language Models (LVLMs) demonstrate promising semantic reasoning, their quantitative effectiveness for safety-critical 2D object detection is underexplored. This paper presents a systematic evaluation of ten representative LVLMs using the PeSOTIF dataset, a benchmark specifically curated for long-tail traffic scenarios and environmental degradations. Performance is quantitatively compared against the classical perception approach, a YOLO-based detector. Experimental results reveal a critical trade-off: top-performing LVLMs (e.g., Gemini 3, Doubao) surpass the YOLO baseline in recall by over 25% in complex natural scenarios, exhibiting superior robustness to visual degradation. Conversely, the baseline retains an advantage in geometric precision for synthetic perturbations. These findings highlight the complementary strengths of semantic reasoning versus geometric regression, supporting the use of LVLMs as high-level safety validators in SOTIF-oriented automated driving systems.

  </details>



- **FarmMind: Reasoning-Query-Driven Dynamic Segmentation for Farmland Remote Sensing Images**  
  Haiyang Wu, Weiliang Mu, Jipeng Zhang, Zhong Dandan, Zhuofei Du, Haifeng Li, Tao Chao  
  _2026-01-30_ · https://arxiv.org/abs/2601.22809v1  
  <details><summary>Abstract</summary>

  Existing methods for farmland remote sensing image (FRSI) segmentation generally follow a static segmentation paradigm, where analysis relies solely on the limited information contained within a single input patch. Consequently, their reasoning capability is limited when dealing with complex scenes characterized by ambiguity and visual uncertainty. In contrast, human experts, when interpreting remote sensing images in such ambiguous cases, tend to actively query auxiliary images (such as higher-resolution, larger-scale, or temporally adjacent data) to conduct cross-verification and achieve more comprehensive reasoning. Inspired by this, we propose a reasoning-query-driven dynamic segmentation framework for FRSIs, named FarmMind. This framework breaks through the limitations of the static segmentation paradigm by introducing a reasoning-query mechanism, which dynamically and on-demand queries external auxiliary images to compensate for the insufficient information in a single input image. Unlike direct queries, this mechanism simulates the thinking process of human experts when faced with segmentation ambiguity: it first analyzes the root causes of segmentation ambiguities through reasoning, and then determines what type of auxiliary image needs to be queried based on this analysis. Extensive experiments demonstrate that FarmMind achieves superior segmentation performance and stronger generalization ability compared with existing methods. The source code and dataset used in this work are publicly available at: https://github.com/WithoutOcean/FarmMind.

  </details>



- **Diachronic Stereo Matching for Multi-Date Satellite Imagery**  
  Elías Masquil, Luca Savant Aira, Roger Marí, Thibaud Ehret, Pablo Musé, Gabriele Facciolo  
  _2026-01-30_ · https://arxiv.org/abs/2601.22808v1  
  <details><summary>Abstract</summary>

  Recent advances in image-based satellite 3D reconstruction have progressed along two complementary directions. On one hand, multi-date approaches using NeRF or Gaussian-splatting jointly model appearance and geometry across many acquisitions, achieving accurate reconstructions on opportunistic imagery with numerous observations. On the other hand, classical stereoscopic reconstruction pipelines deliver robust and scalable results for simultaneous or quasi-simultaneous image pairs. However, when the two images are captured months apart, strong seasonal, illumination, and shadow changes violate standard stereoscopic assumptions, causing existing pipelines to fail. This work presents the first Diachronic Stereo Matching method for satellite imagery, enabling reliable 3D reconstruction from temporally distant pairs. Two advances make this possible: (1) fine-tuning a state-of-the-art deep stereo network that leverages monocular depth priors, and (2) exposing it to a dataset specifically curated to include a diverse set of diachronic image pairs. In particular, we start from a pretrained MonSter model, trained initially on a mix of synthetic and real datasets such as SceneFlow and KITTI, and fine-tune it on a set of stereo pairs derived from the DFC2019 remote sensing challenge. This dataset contains both synchronic and diachronic pairs under diverse seasonal and illumination conditions. Experiments on multi-date WorldView-3 imagery demonstrate that our approach consistently surpasses classical pipelines and unadapted deep stereo models on both synchronic and diachronic settings. Fine-tuning on temporally diverse images, together with monocular priors, proves essential for enabling 3D reconstruction from previously incompatible acquisition dates. Left image (winter) Right image (autumn) DSM geometry Ours (1.23 m) Zero-shot (3.99 m) LiDAR GT Figure 1. Output geometry for a winter-autumn image pair from Omaha (OMA 331 test scene). Our method recovers accurate geometry despite the diachronic nature of the pair, exhibiting strong appearance changes, which cause existing zero-shot methods to fail. Missing values due to perspective shown in black. Mean altitude error in parentheses; lower is better.

  </details>



- **Lingua-SafetyBench: A Benchmark for Safety Evaluation of Multilingual Vision-Language Models**  
  Enyi Shi, Pengyang Shao, Yanxin Zhang, Chenhang Cui, Jiayi Lyu, Xu Xie, Xiaobo Xia, Fei Shen, Tat-Seng Chua  
  _2026-01-30_ · https://arxiv.org/abs/2601.22737v1  
  <details><summary>Abstract</summary>

  Robust safety of vision-language large models (VLLMs) under joint multilingual and multimodal inputs remains underexplored. Existing benchmarks are typically multilingual but text-only, or multimodal but monolingual. Recent multilingual multimodal red-teaming efforts render harmful prompts into images, yet rely heavily on typography-style visuals and lack semantically grounded image-text pairs, limiting coverage of realistic cross-modal interactions. We introduce Lingua-SafetyBench, a benchmark of 100,440 harmful image-text pairs across 10 languages, explicitly partitioned into image-dominant and text-dominant subsets to disentangle risk sources. Evaluating 11 open-source VLLMs reveals a consistent asymmetry: image-dominant risks yield higher ASR in high-resource languages, while text-dominant risks are more severe in non-high-resource languages. A controlled study on the Qwen series shows that scaling and version upgrades reduce Attack Success Rate (ASR) overall but disproportionately benefit HRLs, widening the gap between HRLs and Non-HRLs under text-dominant risks. This underscores the necessity of language- and modality-aware safety alignment beyond mere scaling.To facilitate reproducibility and future research, we will publicly release our benchmark, model checkpoints, and source code.The code and dataset will be available at https://github.com/zsxr15/Lingua-SafetyBench.Warning: this paper contains examples with unsafe content.

  </details>



- **Active Learning-Driven Lightweight YOLOv9: Enhancing Efficiency in Smart Agriculture**  
  Hung-Chih Tu, Bo-Syun Chen, Yun-Chien Cheng  
  _2026-01-30_ · https://arxiv.org/abs/2601.22732v1  
  <details><summary>Abstract</summary>

  This study addresses the demand for real-time detection of tomatoes and tomato flowers by agricultural robots deployed on edge devices in greenhouse environments. Under practical imaging conditions, object detection systems often face challenges such as large scale variations caused by varying camera distances, severe occlusion from plant structures, and highly imbalanced class distributions. These factors make conventional object detection approaches that rely on fully annotated datasets difficult to simultaneously achieve high detection accuracy and deployment efficiency. To overcome these limitations, this research proposes an active learning driven lightweight object detection framework, integrating data analysis, model design, and training strategy. First, the size distribution of objects in raw agricultural images is analyzed to redefine an operational target range, thereby improving learning stability under real-world conditions. Second, an efficient feature extraction module is incorporated to reduce computational cost, while a lightweight attention mechanism is introduced to enhance feature representation under multi-scale and occluded scenarios. Finally, an active learning strategy is employed to iteratively select high-information samples for annotation and training under a limited labeling budget, effectively improving the recognition performance of minority and small-object categories. Experimental results demonstrate that, while maintaining a low parameter count and inference cost suitable for edge-device deployment, the proposed method effectively improves the detection performance of tomatoes and tomato flowers in raw images. Under limited annotation conditions, the framework achieves an overall detection accuracy of 67.8% mAP, validating its practicality and feasibility for intelligent agricultural applications.

  </details>



- **OpenVTON-Bench: A Large-Scale High-Resolution Benchmark for Controllable Virtual Try-On Evaluation**  
  Jin Li, Tao Chen, Shuai Jiang, Weijie Wang, Jingwen Luo, Chenhui Wu  
  _2026-01-30_ · https://arxiv.org/abs/2601.22725v1  
  <details><summary>Abstract</summary>

  Recent advances in diffusion models have significantly elevated the visual fidelity of Virtual Try-On (VTON) systems, yet reliable evaluation remains a persistent bottleneck. Traditional metrics struggle to quantify fine-grained texture details and semantic consistency, while existing datasets fail to meet commercial standards in scale and diversity. We present OpenVTON-Bench, a large-scale benchmark comprising approximately 100K high-resolution image pairs (up to $1536 \times 1536$). The dataset is constructed using DINOv3-based hierarchical clustering for semantically balanced sampling and Gemini-powered dense captioning, ensuring a uniform distribution across 20 fine-grained garment categories. To support reliable evaluation, we propose a multi-modal protocol that measures VTON quality along five interpretable dimensions: background consistency, identity fidelity, texture fidelity, shape plausibility, and overall realism. The protocol integrates VLM-based semantic reasoning with a novel Multi-Scale Representation Metric based on SAM3 segmentation and morphological erosion, enabling the separation of boundary alignment errors from internal texture artifacts. Experimental results show strong agreement with human judgments (Kendall's $τ$ of 0.833 vs. 0.611 for SSIM), establishing a robust benchmark for VTON evaluation.

  </details>



- **Vision-Language Models Unlock Task-Centric Latent Actions**  
  Alexander Nikulin, Ilya Zisman, Albina Klepach, Denis Tarasov, Alexander Derevyagin, Andrei Polubarov, Lyubaykin Nikita, Vladislav Kurenkov  
  _2026-01-30_ · https://arxiv.org/abs/2601.22714v1  
  <details><summary>Abstract</summary>

  Latent Action Models (LAMs) have rapidly gained traction as an important component in the pre-training pipelines of leading Vision-Language-Action models. However, they fail when observations contain action-correlated distractors, often encoding noise instead of meaningful latent actions. Humans, on the other hand, can effortlessly distinguish task-relevant motions from irrelevant details in any video given only a brief task description. In this work, we propose to utilize the common-sense reasoning abilities of Vision-Language Models (VLMs) to provide promptable representations, effectively separating controllable changes from the noise in unsupervised way. We use these representations as targets during LAM training and benchmark a wide variety of popular VLMs, revealing substantial variation in the quality of promptable representations as well as their robustness to different prompts and hyperparameters. Interestingly, we find that more recent VLMs may perform worse than older ones. Finally, we show that simply asking VLMs to ignore distractors can substantially improve latent action quality, yielding up to a six-fold increase in downstream success rates on Distracting MetaWorld.

  </details>



- **DAVIS: OOD Detection via Dominant Activations and Variance for Increased Separation**  
  Abid Hassan, Tuan Ngo, Saad Shafiq, Nenad Medvidovic  
  _2026-01-30_ · https://arxiv.org/abs/2601.22703v1  
  <details><summary>Abstract</summary>

  Detecting out-of-distribution (OOD) inputs is a critical safeguard for deploying machine learning models in the real world. However, most post-hoc detection methods operate on penultimate feature representations derived from global average pooling (GAP) -- a lossy operation that discards valuable distributional statistics from activation maps prior to global average pooling. We contend that these overlooked statistics, particularly channel-wise variance and dominant (maximum) activations, are highly discriminative for OOD detection. We introduce DAVIS, a simple and broadly applicable post-hoc technique that enriches feature vectors by incorporating these crucial statistics, directly addressing the information loss from GAP. Extensive evaluations show DAVIS sets a new benchmark across diverse architectures, including ResNet, DenseNet, and EfficientNet. It achieves significant reductions in the false positive rate (FPR95), with improvements of 48.26\% on CIFAR-10 using ResNet-18, 38.13\% on CIFAR-100 using ResNet-34, and 26.83\% on ImageNet-1k benchmarks using MobileNet-v2. Our analysis reveals the underlying mechanism for this improvement, providing a principled basis for moving beyond the mean in OOD detection.

  </details>



- **PEAR: Pixel-aligned Expressive humAn mesh Recovery**  
  Jiahao Wu, Yunfei Liu, Lijian Lin, Ye Zhu, Lei Zhu, Jingyi Li, Yu Li  
  _2026-01-30_ · https://arxiv.org/abs/2601.22693v1  
  <details><summary>Abstract</summary>

  Reconstructing detailed 3D human meshes from a single in-the-wild image remains a fundamental challenge in computer vision. Existing SMPLX-based methods often suffer from slow inference, produce only coarse body poses, and exhibit misalignments or unnatural artifacts in fine-grained regions such as the face and hands. These issues make current approaches difficult to apply to downstream tasks. To address these challenges, we propose PEAR-a fast and robust framework for pixel-aligned expressive human mesh recovery. PEAR explicitly tackles three major limitations of existing methods: slow inference, inaccurate localization of fine-grained human pose details, and insufficient facial expression capture. Specifically, to enable real-time SMPLX parameter inference, we depart from prior designs that rely on high resolution inputs or multi-branch architectures. Instead, we adopt a clean and unified ViT-based model capable of recovering coarse 3D human geometry. To compensate for the loss of fine-grained details caused by this simplified architecture, we introduce pixel-level supervision to optimize the geometry, significantly improving the reconstruction accuracy of fine-grained human details. To make this approach practical, we further propose a modular data annotation strategy that enriches the training data and enhances the robustness of the model. Overall, PEAR is a preprocessing-free framework that can simultaneously infer EHM-s (SMPLX and scaled-FLAME) parameters at over 100 FPS. Extensive experiments on multiple benchmark datasets demonstrate that our method achieves substantial improvements in pose estimation accuracy compared to previous SMPLX-based approaches. Project page: https://wujh2001.github.io/PEAR

  </details>



- **Visual Personalization Turing Test**  
  Rameen Abdal, James Burgess, Sergey Tulyakov, Kuan-Chieh Jackson Wang  
  _2026-01-30_ · https://arxiv.org/abs/2601.22680v1  
  <details><summary>Abstract</summary>

  We introduce the Visual Personalization Turing Test (VPTT), a new paradigm for evaluating contextual visual personalization based on perceptual indistinguishability, rather than identity replication. A model passes the VPTT if its output (image, video, 3D asset, etc.) is indistinguishable to a human or calibrated VLM judge from content a given person might plausibly create or share. To operationalize VPTT, we present the VPTT Framework, integrating a 10k-persona benchmark (VPTT-Bench), a visual retrieval-augmented generator (VPRAG), and the VPTT Score, a text-only metric calibrated against human and VLM judgments. We show high correlation across human, VLM, and VPTT evaluations, validating the VPTT Score as a reliable perceptual proxy. Experiments demonstrate that VPRAG achieves the best alignment-originality balance, offering a scalable and privacy-safe foundation for personalized generative AI.

  </details>


