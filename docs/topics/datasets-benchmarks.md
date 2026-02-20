# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-02-20 07:12 UTC_

Total papers shown: **50**


---

- **When Vision Overrides Language: Evaluating and Mitigating Counterfactual Failures in VLAs**  
  Yu Fang, Yuchun Feng, Dong Jing, Jiaqi Liu, Yue Yang, Zhenyu Wei, Daniel Szafir, Mingyu Ding  
  _2026-02-19_ · https://arxiv.org/abs/2602.17659v1  
  <details><summary>Abstract</summary>

  Vision-Language-Action models (VLAs) promise to ground language instructions in robot control, yet in practice often fail to faithfully follow language. When presented with instructions that lack strong scene-specific supervision, VLAs suffer from counterfactual failures: they act based on vision shortcuts induced by dataset biases, repeatedly executing well-learned behaviors and selecting objects frequently seen during training regardless of language intent. To systematically study it, we introduce LIBERO-CF, the first counterfactual benchmark for VLAs that evaluates language following capability by assigning alternative instructions under visually plausible LIBERO layouts. Our evaluation reveals that counterfactual failures are prevalent yet underexplored across state-of-the-art VLAs. We propose Counterfactual Action Guidance (CAG), a simple yet effective dual-branch inference scheme that explicitly regularizes language conditioning in VLAs. CAG combines a standard VLA policy with a language-unconditioned Vision-Action (VA) module, enabling counterfactual comparison during action selection. This design reduces reliance on visual shortcuts, improves robustness on under-observed tasks, and requires neither additional demonstrations nor modifications to existing architectures or pretrained models. Extensive experiments demonstrate its plug-and-play integration across diverse VLAs and consistent improvements. For example, on LIBERO-CF, CAG improves $π_{0.5}$ by 9.7% in language following accuracy and 3.6% in task success on under-observed tasks using a training-free strategy, with further gains of 15.5% and 8.5%, respectively, when paired with a VA model. In real-world evaluations, CAG reduces counterfactual failures of 9.4% and improves task success by 17.2% on average.

  </details>



- **IntRec: Intent-based Retrieval with Contrastive Refinement**  
  Pourya Shamsolmoali, Masoumeh Zareapoor, Eric Granger, Yue Lu  
  _2026-02-19_ · https://arxiv.org/abs/2602.17639v1  
  <details><summary>Abstract</summary>

  Retrieving user-specified objects from complex scenes remains a challenging task, especially when queries are ambiguous or involve multiple similar objects. Existing open-vocabulary detectors operate in a one-shot manner, lacking the ability to refine predictions based on user feedback. To address this, we propose IntRec, an interactive object retrieval framework that refines predictions based on user feedback. At its core is an Intent State (IS) that maintains dual memory sets for positive anchors (confirmed cues) and negative constraints (rejected hypotheses). A contrastive alignment function ranks candidate objects by maximizing similarity to positive cues while penalizing rejected ones, enabling fine-grained disambiguation in cluttered scenes. Our interactive framework provides substantial improvements in retrieval accuracy without additional supervision. On LVIS, IntRec achieves 35.4 AP, outperforming OVMR, CoDet, and CAKE by +2.3, +3.7, and +0.5, respectively. On the challenging LVIS-Ambiguous benchmark, it improves performance by +7.9 AP over its one-shot baseline after a single corrective feedback, with less than 30 ms of added latency per interaction.

  </details>



- **CORAL: Correspondence Alignment for Improved Virtual Try-On**  
  Jiyoung Kim, Youngjin Shin, Siyoon Jin, Dahyun Chung, Jisu Nam, Tongmin Kim, Jongjae Park, Hyeonwoo Kang, Seungryong Kim  
  _2026-02-19_ · https://arxiv.org/abs/2602.17636v1  
  <details><summary>Abstract</summary>

  Existing methods for Virtual Try-On (VTON) often struggle to preserve fine garment details, especially in unpaired settings where accurate person-garment correspondence is required. These methods do not explicitly enforce person-garment alignment and fail to explain how correspondence emerges within Diffusion Transformers (DiTs). In this paper, we first analyze full 3D attention in DiT-based architecture and reveal that the person-garment correspondence critically depends on precise person-garment query-key matching within the full 3D attention. Building on this insight, we then introduce CORrespondence ALignment (CORAL), a DiT-based framework that explicitly aligns query-key matching with robust external correspondences. CORAL integrates two complementary components: a correspondence distillation loss that aligns reliable matches with person-garment attention, and an entropy minimization loss that sharpens the attention distribution. We further propose a VLM-based evaluation protocol to better reflect human preference. CORAL consistently improves over the baseline, enhancing both global shape transfer and local detail preservation. Extensive ablations validate our design choices.

  </details>



- **Adapting Actively on the Fly: Relevance-Guided Online Meta-Learning with Latent Concepts for Geospatial Discovery**  
  Jowaria Khan, Anindya Sarkar, Yevgeniy Vorobeychik, Elizabeth Bondi-Kelly  
  _2026-02-19_ · https://arxiv.org/abs/2602.17605v1  
  <details><summary>Abstract</summary>

  In many real-world settings, such as environmental monitoring, disaster response, or public health, with costly and difficult data collection and dynamic environments, strategically sampling from unobserved regions is essential for efficiently uncovering hidden targets under tight resource constraints. Yet, sparse and biased geospatial ground truth limits the applicability of existing learning-based methods, such as reinforcement learning. To address this, we propose a unified geospatial discovery framework that integrates active learning, online meta-learning, and concept-guided reasoning. Our approach introduces two key innovations built on a shared notion of *concept relevance*, which captures how domain-specific factors influence target presence: a *concept-weighted uncertainty sampling strategy*, where uncertainty is modulated by learned relevance based on readily-available domain-specific concepts (e.g., land cover, source proximity); and a *relevance-aware meta-batch formation strategy* that promotes semantic diversity during online-meta updates, improving generalization in dynamic environments. Our experiments include testing on a real-world dataset of cancer-causing PFAS (Per- and polyfluoroalkyl substances) contamination, showcasing our method's reliability at uncovering targets with limited data and a varying environment.

  </details>



- **Art2Mus: Artwork-to-Music Generation via Visual Conditioning and Large-Scale Cross-Modal Alignment**  
  Ivan Rinaldi, Matteo Mendula, Nicola Fanelli, Florence Levé, Matteo Testi, Giovanna Castellano, Gennaro Vessio  
  _2026-02-19_ · https://arxiv.org/abs/2602.17599v1  
  <details><summary>Abstract</summary>

  Music generation has advanced markedly through multimodal deep learning, enabling models to synthesize audio from text and, more recently, from images. However, existing image-conditioned systems suffer from two fundamental limitations: (i) they are typically trained on natural photographs, limiting their ability to capture the richer semantic, stylistic, and cultural content of artworks; and (ii) most rely on an image-to-text conversion stage, using language as a semantic shortcut that simplifies conditioning but prevents direct visual-to-audio learning. Motivated by these gaps, we introduce ArtSound, a large-scale multimodal dataset of 105,884 artwork-music pairs enriched with dual-modality captions, obtained by extending ArtGraph and the Free Music Archive. We further propose ArtToMus, the first framework explicitly designed for direct artwork-to-music generation, which maps digitized artworks to music without image-to-text translation or language-based semantic supervision. The framework projects visual embeddings into the conditioning space of a latent diffusion model, enabling music synthesis guided solely by visual information. Experimental results show that ArtToMus generates musically coherent and stylistically consistent outputs that reflect salient visual cues of the source artworks. While absolute alignment scores remain lower than those of text-conditioned systems-as expected given the substantially increased difficulty of removing linguistic supervision-ArtToMus achieves competitive perceptual quality and meaningful cross-modal correspondence. This work establishes direct visual-to-music generation as a distinct and challenging research direction, and provides resources that support applications in multimedia art, cultural heritage, and AI-assisted creative practice. Code and dataset will be publicly released upon acceptance.

  </details>



- **Conditional Flow Matching for Continuous Anomaly Detection in Autonomous Driving on a Manifold-Aware Spectral Space**  
  Antonio Guillen-Perez  
  _2026-02-19_ · https://arxiv.org/abs/2602.17586v1  
  <details><summary>Abstract</summary>

  Safety validation for Level 4 autonomous vehicles (AVs) is currently bottlenecked by the inability to scale the detection of rare, high-risk long-tail scenarios using traditional rule-based heuristics. We present Deep-Flow, an unsupervised framework for safety-critical anomaly detection that utilizes Optimal Transport Conditional Flow Matching (OT-CFM) to characterize the continuous probability density of expert human driving behavior. Unlike standard generative approaches that operate in unstable, high-dimensional coordinate spaces, Deep-Flow constrains the generative process to a low-rank spectral manifold via a Principal Component Analysis (PCA) bottleneck. This ensures kinematic smoothness by design and enables the computation of the exact Jacobian trace for numerically stable, deterministic log-likelihood estimation. To resolve multi-modal ambiguity at complex junctions, we utilize an Early Fusion Transformer encoder with lane-aware goal conditioning, featuring a direct skip-connection to the flow head to maintain intent-integrity throughout the network. We introduce a kinematic complexity weighting scheme that prioritizes high-energy maneuvers (quantified via path tortuosity and jerk) during the simulation-free training process. Evaluated on the Waymo Open Motion Dataset (WOMD), our framework achieves an AUC-ROC of 0.766 against a heuristic golden set of safety-critical events. More significantly, our analysis reveals a fundamental distinction between kinematic danger and semantic non-compliance. Deep-Flow identifies a critical predictability gap by surfacing out-of-distribution behaviors, such as lane-boundary violations and non-normative junction maneuvers, that traditional safety filters overlook. This work provides a mathematically rigorous foundation for defining statistical safety gates, enabling objective, data-driven validation for the safe deployment of autonomous fleets.

  </details>



- **FR-GESTURE: An RGBD Dataset For Gesture-based Human-Robot Interaction In First Responder Operations**  
  Konstantinos Foteinos, Georgios Angelidis, Aggelos Psiris, Vasileios Argyriou, Panagiotis Sarigiannidis, Georgios Th. Papadopoulos  
  _2026-02-19_ · https://arxiv.org/abs/2602.17573v1  
  <details><summary>Abstract</summary>

  The ever increasing intensity and number of disasters make even more difficult the work of First Responders (FRs). Artificial intelligence and robotics solutions could facilitate their operations, compensating these difficulties. To this end, we propose a dataset for gesture-based UGV control by FRs, introducing a set of 12 commands, drawing inspiration from existing gestures used by FRs and tactical hand signals and refined after incorporating feedback from experienced FRs. Then we proceed with the data collection itself, resulting in 3312 RGBD pairs captured from 2 viewpoints and 7 distances. To the best of our knowledge, this is the first dataset especially intended for gesture-based UGV guidance by FRs. Finally we define evaluation protocols for our RGBD dataset, termed FR-GESTURE, and we perform baseline experiments, which are put forward for improvement. We have made data publicly available to promote future research on the domain: https://doi.org/10.5281/zenodo.18131333.

  </details>



- **RetouchIQ: MLLM Agents for Instruction-Based Image Retouching with Generalist Reward**  
  Qiucheng Wu, Jing Shi, Simon Jenni, Kushal Kafle, Tianyu Wang, Shiyu Chang, Handong Zhao  
  _2026-02-19_ · https://arxiv.org/abs/2602.17558v1  
  <details><summary>Abstract</summary>

  Recent advances in multimodal large language models (MLLMs) have shown great potential for extending vision-language reasoning to professional tool-based image editing, enabling intuitive and creative editing. A promising direction is to use reinforcement learning (RL) to enable MLLMs to reason about and execute optimal tool-use plans within professional image-editing software. However, training remains challenging due to the lack of reliable, verifiable reward signals that can reflect the inherently subjective nature of creative editing. In this work, we introduce RetouchIQ, a framework that performs instruction-based executable image editing through MLLM agents guided by a generalist reward model. RetouchIQ interprets user-specified editing intentions and generates corresponding, executable image adjustments, bridging high-level aesthetic goals with precise parameter control. To move beyond conventional, rule-based rewards that compute similarity against a fixed reference image using handcrafted metrics, we propose a generalist reward model, an RL fine-tuned MLLM that evaluates edited results through a set of generated metrics on a case-by-case basis. Then, the reward model provides scalar feedback through multimodal reasoning, enabling reinforcement learning with high-quality, instruction-consistent gradients. We curate an extended dataset with 190k instruction-reasoning pairs and establish a new benchmark for instruction-based image editing. Experiments show that RetouchIQ substantially improves both semantic consistency and perceptual quality over previous MLLM-based and diffusion-based editing systems. Our findings demonstrate the potential of generalist reward-driven MLLM agents as flexible, explainable, and executable assistants for professional image editing.

  </details>



- **Tracing Copied Pixels and Regularizing Patch Affinity in Copy Detection**  
  Yichen Lu, Siwei Nie, Minlong Lu, Xudong Yang, Xiaobo Zhang, Peng Zhang  
  _2026-02-19_ · https://arxiv.org/abs/2602.17484v1  
  <details><summary>Abstract</summary>

  Image Copy Detection (ICD) aims to identify manipulated content between image pairs through robust feature representation learning. While self-supervised learning (SSL) has advanced ICD systems, existing view-level contrastive methods struggle with sophisticated edits due to insufficient fine-grained correspondence learning. We address this limitation by exploiting the inherent geometric traceability in edited content through two key innovations. First, we propose PixTrace - a pixel coordinate tracking module that maintains explicit spatial mappings across editing transformations. Second, we introduce CopyNCE, a geometrically-guided contrastive loss that regularizes patch affinity using overlap ratios derived from PixTrace's verified mappings. Our method bridges pixel-level traceability with patch-level similarity learning, suppressing supervision noise in SSL training. Extensive experiments demonstrate not only state-of-the-art performance (88.7% uAP / 83.9% RP90 for matcher, 72.6% uAP / 68.4% RP90 for descriptor on DISC21 dataset) but also better interpretability over existing methods.

  </details>



- **QuPAINT: Physics-Aware Instruction Tuning Approach to Quantum Material Discovery**  
  Xuan-Bac Nguyen, Hoang-Quan Nguyen, Sankalp Pandey, Tim Faltermeier, Nicholas Borys, Hugh Churchill, Khoa Luu  
  _2026-02-19_ · https://arxiv.org/abs/2602.17478v1  
  <details><summary>Abstract</summary>

  Characterizing two-dimensional quantum materials from optical microscopy images is challenging due to the subtle layer-dependent contrast, limited labeled data, and significant variation across laboratories and imaging setups. Existing vision models struggle in this domain since they lack physical priors and cannot generalize to new materials or hardware conditions. This work presents a new physics-aware multimodal framework that addresses these limitations from both the data and model perspectives. We first present Synthia, a physics-based synthetic data generator that simulates realistic optical responses of quantum material flakes under thin-film interference. Synthia produces diverse and high-quality samples, helping reduce the dependence on expert manual annotation. We introduce QMat-Instruct, the first large-scale instruction dataset for quantum materials, comprising multimodal, physics-informed question-answer pairs designed to teach Multimodal Large Language Models (MLLMs) to understand the appearance and thickness of flakes. Then, we propose Physics-Aware Instruction Tuning (QuPAINT), a multimodal architecture that incorporates a Physics-Informed Attention module to fuse visual embeddings with optical priors, enabling more robust and discriminative flake representations. Finally, we establish QF-Bench, a comprehensive benchmark spanning multiple materials, substrates, and imaging settings, offering standardized protocols for fair and reproducible evaluation.

  </details>



- **Attachment Anchors: A Novel Framework for Laparoscopic Grasping Point Prediction in Colorectal Surgery**  
  Dennis N. Schneider, Lars Wagner, Daniel Rueckert, Dirk Wilhelm  
  _2026-02-19_ · https://arxiv.org/abs/2602.17310v1  
  <details><summary>Abstract</summary>

  Accurate grasping point prediction is a key challenge for autonomous tissue manipulation in minimally invasive surgery, particularly in complex and variable procedures such as colorectal interventions. Due to their complexity and prolonged duration, colorectal procedures have been underrepresented in current research. At the same time, they pose a particularly interesting learning environment due to repetitive tissue manipulation, making them a promising entry point for autonomous, machine learning-driven support. Therefore, in this work, we introduce attachment anchors, a structured representation that encodes the local geometric and mechanical relationships between tissue and its anatomical attachments in colorectal surgery. This representation reduces uncertainty in grasping point prediction by normalizing surgical scenes into a consistent local reference frame. We demonstrate that attachment anchors can be predicted from laparoscopic images and incorporated into a grasping framework based on machine learning. Experiments on a dataset of 90 colorectal surgeries demonstrate that attachment anchors improve grasping point prediction compared to image-only baselines. There are particularly strong gains in out-of-distribution settings, including unseen procedures and operating surgeons. These results suggest that attachment anchors are an effective intermediate representation for learning-based tissue manipulation in colorectal surgery.

  </details>



- **Physics Encoded Spatial and Temporal Generative Adversarial Network for Tropical Cyclone Image Super-resolution**  
  Ruoyi Zhang, Jiawei Yuan, Lujia Ye, Runling Yu, Liling Zhao  
  _2026-02-19_ · https://arxiv.org/abs/2602.17277v1  
  <details><summary>Abstract</summary>

  High-resolution satellite imagery is indispensable for tracking the genesis, intensification, and trajectory of tropical cyclones (TCs). However, existing deep learning-based super-resolution (SR) methods often treat satellite image sequences as generic videos, neglecting the underlying atmospheric physical laws governing cloud motion. To address this, we propose a Physics Encoded Spatial and Temporal Generative Adversarial Network (PESTGAN) for TC image super-resolution. Specifically, we design a disentangled generator architecture incorporating a PhyCell module, which approximates the vorticity equation via constrained convolutions and encodes the resulting approximate physical dynamics as implicit latent representations to separate physical dynamics from visual textures. Furthermore, a dual-discriminator framework is introduced, employing a temporal discriminator to enforce motion consistency alongside spatial realism. Experiments on the Digital Typhoon dataset for 4$\times$ upscaling demonstrate that PESTGAN establishes a better performance in structural fidelity and perceptual quality. While maintaining competitive pixel-wise accuracy compared to existing approaches, our method significantly excels in reconstructing meteorologically plausible cloud structures with superior physical fidelity.

  </details>



- **EA-Swin: An Embedding-Agnostic Swin Transformer for AI-Generated Video Detection**  
  Hung Mai, Loi Dinh, Duc Hai Nguyen, Dat Do, Luong Doan, Khanh Nguyen Quoc, Huan Vu, Phong Ho, Naeem Ul Islam, Tuan Do  
  _2026-02-19_ · https://arxiv.org/abs/2602.17260v1  
  <details><summary>Abstract</summary>

  Recent advances in foundation video generators such as Sora2, Veo3, and other commercial systems have produced highly realistic synthetic videos, exposing the limitations of existing detection methods that rely on shallow embedding trajectories, image-based adaptation, or computationally heavy MLLMs. We propose EA-Swin, an Embedding-Agnostic Swin Transformer that models spatiotemporal dependencies directly on pretrained video embeddings via a factorized windowed attention design, making it compatible with generic ViT-style patch-based encoders. Alongside the model, we construct the EA-Video dataset, a benchmark dataset comprising 130K videos that integrates newly collected samples with curated existing datasets, covering diverse commercial and open-source generators and including unseen-generator splits for rigorous cross-distribution evaluation. Extensive experiments show that EA-Swin achieves 0.97-0.99 accuracy across major generators, outperforming prior SoTA methods (typically 0.8-0.9) by a margin of 5-20%, while maintaining strong generalization to unseen distributions, establishing a scalable and robust solution for modern AI-generated video detection.

  </details>



- **FRAPPE: Infusing World Modeling into Generalist Policies via Multiple Future Representation Alignment**  
  Han Zhao, Jingbo Wang, Wenxuan Song, Shuai Chen, Yang Liu, Yan Wang, Haoang Li, Donglin Wang  
  _2026-02-19_ · https://arxiv.org/abs/2602.17259v1  
  <details><summary>Abstract</summary>

  Enabling VLA models to predict environmental dynamics, known as world modeling, has been recognized as essential for improving robotic reasoning and generalization. However, current approaches face two main issues: 1. The training objective forces models to over-emphasize pixel-level reconstruction, which constrains semantic learning and generalization 2. Reliance on predicted future observations during inference often leads to error accumulation. To address these challenges, we introduce Future Representation Alignment via Parallel Progressive Expansion (FRAPPE). Our method adopts a two-stage fine-tuning strategy: In the mid-training phase, the model learns to predict the latent representations of future observations; In the post-training phase, we expand the computational workload in parallel and align the representation simultaneously with multiple different visual foundation models. By significantly improving fine-tuning efficiency and reducing dependence on action-annotated data, FRAPPE provides a scalable and data-efficient pathway to enhance world-awareness in generalist robotic policies. Experiments on the RoboTwin benchmark and real-world tasks demonstrate that FRAPPE outperforms state-of-the-art approaches and shows strong generalization in long-horizon and unseen scenarios.

  </details>



- **Benchmarking the Effects of Object Pose Estimation and Reconstruction on Robotic Grasping Success**  
  Varun Burde, Pavel Burget, Torsten Sattler  
  _2026-02-19_ · https://arxiv.org/abs/2602.17101v1  
  <details><summary>Abstract</summary>

  3D reconstruction serves as the foundational layer for numerous robotic perception tasks, including 6D object pose estimation and grasp pose generation. Modern 3D reconstruction methods for objects can produce visually and geometrically impressive meshes from multi-view images, yet standard geometric evaluations do not reflect how reconstruction quality influences downstream tasks such as robotic manipulation performance. This paper addresses this gap by introducing a large-scale, physics-based benchmark that evaluates 6D pose estimators and 3D mesh models based on their functional efficacy in grasping. We analyze the impact of model fidelity by generating grasps on various reconstructed 3D meshes and executing them on the ground-truth model, simulating how grasp poses generated with an imperfect model affect interaction with the real object. This assesses the combined impact of pose error, grasp robustness, and geometric inaccuracies from 3D reconstruction. Our results show that reconstruction artifacts significantly decrease the number of grasp pose candidates but have a negligible effect on grasping performance given an accurately estimated pose. Our results also reveal that the relationship between grasp success and pose error is dominated by spatial error, and even a simple translation error provides insight into the success of the grasping pose of symmetric objects. This work provides insight into how perception systems relate to object manipulation using robots.

  </details>



- **Cholec80-port: A Geometrically Consistent Trocar Port Segmentation Dataset for Robust Surgical Scene Understanding**  
  Shunsuke Kikuchi, Atsushi Kouno, Hiroki Matsuzaki  
  _2026-02-19_ · https://arxiv.org/abs/2602.17060v1  
  <details><summary>Abstract</summary>

  Trocar ports are camera-fixed, pseudo-static structures that can persistently occlude laparoscopic views and attract disproportionate feature points due to specular, textured surfaces. This makes ports particularly detrimental to geometry-based downstream pipelines such as image stitching, 3D reconstruction, and visual SLAM, where dynamic or non-anatomical outliers degrade alignment and tracking stability. Despite this practical importance, explicit port labels are rare in public surgical datasets, and existing annotations often violate geometric consistency by masking the central lumen (opening), even when anatomical regions are visible through it. We present Cholec80-port, a high-fidelity trocar port segmentation dataset derived from Cholec80, together with a rigorous standard operating procedure (SOP) that defines a port-sleeve mask excluding the central opening. We additionally cleanse and unify existing public datasets under the same SOP. Experiments demonstrate that geometrically consistent annotations substantially improve cross-dataset robustness beyond what dataset size alone provides.

  </details>



- **Characterizing the Predictive Impact of Modalities with Supervised Latent-Variable Modeling**  
  Divyam Madaan, Sumit Chopra, Kyunghyun Cho  
  _2026-02-19_ · https://arxiv.org/abs/2602.16979v1  
  <details><summary>Abstract</summary>

  Despite the recent success of Multimodal Large Language Models (MLLMs), existing approaches predominantly assume the availability of multiple modalities during training and inference. In practice, multimodal data is often incomplete because modalities may be missing, collected asynchronously, or available only for a subset of examples. In this work, we propose PRIMO, a supervised latent-variable imputation model that quantifies the predictive impact of any missing modality within the multimodal learning setting. PRIMO enables the use of all available training examples, whether modalities are complete or partial. Specifically, it models the missing modality through a latent variable that captures its relationship with the observed modality in the context of prediction. During inference, we draw many samples from the learned distribution over the missing modality to both obtain the marginal predictive distribution (for the purpose of prediction) and analyze the impact of the missing modalities on the prediction for each instance. We evaluate PRIMO on a synthetic XOR dataset, Audio-Vision MNIST, and MIMIC-III for mortality and ICD-9 prediction. Across all datasets, PRIMO obtains performance comparable to unimodal baselines when a modality is fully missing and to multimodal baselines when all modalities are available. PRIMO quantifies the predictive impact of a modality at the instance level using a variance-based metric computed from predictions across latent completions. We visually demonstrate how varying completions of the missing modality result in a set of plausible labels.

  </details>



- **StereoAdapter-2: Globally Structure-Consistent Underwater Stereo Depth Estimation**  
  Zeyu Ren, Xiang Li, Yiran Wang, Zeyu Zhang, Hao Tang  
  _2026-02-18_ · https://arxiv.org/abs/2602.16915v1  
  <details><summary>Abstract</summary>

  Stereo depth estimation is fundamental to underwater robotic perception, yet suffers from severe domain shifts caused by wavelength-dependent light attenuation, scattering, and refraction. Recent approaches leverage monocular foundation models with GRU-based iterative refinement for underwater adaptation; however, the sequential gating and local convolutional kernels in GRUs necessitate multiple iterations for long-range disparity propagation, limiting performance in large-disparity and textureless underwater regions. In this paper, we propose StereoAdapter-2, which replaces the conventional ConvGRU updater with a novel ConvSS2D operator based on selective state space models. The proposed operator employs a four-directional scanning strategy that naturally aligns with epipolar geometry while capturing vertical structural consistency, enabling efficient long-range spatial propagation within a single update step at linear computational complexity. Furthermore, we construct UW-StereoDepth-80K, a large-scale synthetic underwater stereo dataset featuring diverse baselines, attenuation coefficients, and scattering parameters through a two-stage generative pipeline combining semantic-aware style transfer and geometry-consistent novel view synthesis. Combined with dynamic LoRA adaptation inherited from StereoAdapter, our framework achieves state-of-the-art zero-shot performance on underwater benchmarks with 17% improvement on TartanAir-UW and 7.2% improvment on SQUID, with real-world validation on the BlueROV2 platform demonstrates the robustness of our approach. Code: https://github.com/AIGeeksGroup/StereoAdapter-2. Website: https://aigeeksgroup.github.io/StereoAdapter-2.

  </details>



- **Boreas Road Trip: A Multi-Sensor Autonomous Driving Dataset on Challenging Roads**  
  Daniil Lisus, Katya M. Papais, Cedric Le Gentil, Elliot Preston-Krebs, Andrew Lambert, Keith Y. K. Leung, Timothy D. Barfoot  
  _2026-02-18_ · https://arxiv.org/abs/2602.16870v1  
  <details><summary>Abstract</summary>

  The Boreas Road Trip (Boreas-RT) dataset extends the multi-season Boreas dataset to new and diverse locations that pose challenges for modern autonomous driving algorithms. Boreas-RT comprises 60 sequences collected over 9 real-world routes, totalling 643 km of driving. Each route is traversed multiple times, enabling evaluation in identical environments under varying traffic and, in some cases, weather conditions. The data collection platform includes a 5MP FLIR Blackfly S camera, a 360 degree Navtech RAS6 Doppler-enabled spinning radar, a 128-channel 360 degree Velodyne Alpha Prime lidar, an Aeva Aeries II FMCW Doppler-enabled lidar, a Silicon Sensing DMU41 inertial measurement unit, and a Dynapar wheel encoder. Centimetre-level ground truth is provided via post-processed Applanix POS LV GNSS-INS data. The dataset includes precise extrinsic and intrinsic calibrations, a publicly available development kit, and a live leaderboard for odometry and metric localization. Benchmark results show that many state-of-the-art odometry and localization algorithms overfit to simple driving environments and degrade significantly on the more challenging Boreas-RT routes. Boreas-RT provides a unified dataset for evaluating multi-modal algorithms across diverse road conditions. The dataset, leaderboard, and development kit are available at www.boreas.utias.utoronto.ca.

  </details>



- **Analytic Score Optimization for Multi Dimension Video Quality Assessment**  
  Boda Lin, Yongjie Zhu, Wenyu Qin, Meng Wang, Pengfei Wan  
  _2026-02-18_ · https://arxiv.org/abs/2602.16856v1  
  <details><summary>Abstract</summary>

  Video Quality Assessment (VQA) is evolving beyond single-number mean opinion score toward richer, multi-faceted evaluations of video content. In this paper, we present a large-scale multi-dimensional VQA dataset UltraVQA that encompasses diverse User-Generated Content~(UGC) annotated across five key quality dimensions: Motion Quality, Motion Amplitude, Aesthetic Quality, Content Quality, and Clarity Quality. Each video in our dataset is scored by over 3 human raters on these dimensions, with fine-grained sub-attribute labels, and accompanied by an explanatory rationale generated by GPT based on the collective human judgments. To better leverage these rich annotations and improve discrete quality score assessment, we introduce Analytic Score Optimization (ASO), a theoretically grounded post-training objective derived for multi-dimensional VQA. By reframing quality assessment as a regularized decision-making process, we obtain a closed-form solution that naturally captures the ordinal nature of human ratings, ensuring alignment with human ranking preferences. In experiments, our method outperforms most baselines including closed-source APIs and open-source models, while also reducing mean absolute error (MAE) in quality prediction. Our work highlights the importance of multi-dimensional, interpretable annotations and reinforcement-based alignment in advancing video quality assessment.

  </details>



- **Are Object-Centric Representations Better At Compositional Generalization?**  
  Ferdinand Kapl, Amir Mohammad Karimi Mamaghan, Maximilian Seitzer, Karl Henrik Johansson, Carsten Marr, Stefan Bauer, Andrea Dittadi  
  _2026-02-18_ · https://arxiv.org/abs/2602.16689v1  
  <details><summary>Abstract</summary>

  Compositional generalization, the ability to reason about novel combinations of familiar concepts, is fundamental to human cognition and a critical challenge for machine learning. Object-centric (OC) representations, which encode a scene as a set of objects, are often argued to support such generalization, but systematic evidence in visually rich settings is limited. We introduce a Visual Question Answering benchmark across three controlled visual worlds (CLEVRTex, Super-CLEVR, and MOVi-C) to measure how well vision encoders, with and without object-centric biases, generalize to unseen combinations of object properties. To ensure a fair and comprehensive comparison, we carefully account for training data diversity, sample size, representation size, downstream model capacity, and compute. We use DINOv2 and SigLIP2, two widely used vision encoders, as the foundation models and their OC counterparts. Our key findings reveal that (1) OC approaches are superior in harder compositional generalization settings; (2) original dense representations surpass OC only on easier settings and typically require substantially more downstream compute; and (3) OC models are more sample efficient, achieving stronger generalization with fewer images, whereas dense encoders catch up or surpass them only with sufficient data and diversity. Overall, object-centric representations offer stronger compositional generalization when any one of dataset size, training data diversity, or downstream compute is constrained.

  </details>



- **Learning Situated Awareness in the Real World**  
  Chuhan Li, Ruilin Han, Joy Hsu, Yongyuan Liang, Rajiv Dhawan, Jiajun Wu, Ming-Hsuan Yang, Xin Eric Wang  
  _2026-02-18_ · https://arxiv.org/abs/2602.16682v1  
  <details><summary>Abstract</summary>

  A core aspect of human perception is situated awareness, the ability to relate ourselves to the surrounding physical environment and reason over possible actions in context. However, most existing benchmarks for multimodal foundation models (MFMs) emphasize environment-centric spatial relations (relations among objects in a scene), while largely overlooking observer-centric relationships that require reasoning relative to agent's viewpoint, pose, and motion. To bridge this gap, we introduce SAW-Bench (Situated Awareness in the Real World), a novel benchmark for evaluating egocentric situated awareness using real-world videos. SAW-Bench comprises 786 self-recorded videos captured with Ray-Ban Meta (Gen 2) smart glasses spanning diverse indoor and outdoor environments, and over 2,071 human-annotated question-answer pairs. It probes a model's observer-centric understanding with six different awareness tasks. Our comprehensive evaluation reveals a human-model performance gap of 37.66%, even with the best-performing MFM, Gemini 3 Flash. Beyond this gap, our in-depth analysis uncovers several notable findings; for example, while models can exploit partial geometric cues in egocentric videos, they often fail to infer a coherent camera geometry, leading to systematic spatial reasoning errors. We position SAW-Bench as a benchmark for situated spatial intelligence, moving beyond passive observation to understanding physically grounded, observer-centric dynamics.

  </details>



- **Style-Aware Gloss Control for Generative Non-Photorealistic Rendering**  
  Santiago Jimenez-Navarro, Belen Masia, Ana Serrano  
  _2026-02-18_ · https://arxiv.org/abs/2602.16611v2  
  <details><summary>Abstract</summary>

  Humans can infer material characteristics of objects from their visual appearance, and this ability extends to artistic depictions, where similar perceptual strategies guide the interpretation of paintings or drawings. Among the factors that define material appearance, gloss, along with color, is widely regarded as one of the most important, and recent studies indicate that humans can perceive gloss independently of the artistic style used to depict an object. To investigate how gloss and artistic style are represented in learned models, we train an unsupervised generative model on a newly curated dataset of painterly objects designed to systematically vary such factors. Our analysis reveals a hierarchical latent space in which gloss is disentangled from other appearance factors, allowing for a detailed study of how gloss is represented and varies across artistic styles. Building on this representation, we introduce a lightweight adapter that connects our style- and gloss-aware latent space to a latent-diffusion model, enabling the synthesis of non-photorealistic images with fine-grained control of these factors. We compare our approach with previous models and observe improved disentanglement and controllability of the learned factors.

  </details>



- **A Contrastive Learning Framework Empowered by Attention-based Feature Adaptation for Street-View Image Classification**  
  Qi You, Yitai Cheng, Zichao Zeng, James Haworth  
  _2026-02-18_ · https://arxiv.org/abs/2602.16590v1  
  <details><summary>Abstract</summary>

  Street-view image attribute classification is a vital downstream task of image classification, enabling applications such as autonomous driving, urban analytics, and high-definition map construction. It remains computationally demanding whether training from scratch, initialising from pre-trained weights, or fine-tuning large models. Although pre-trained vision-language models such as CLIP offer rich image representations, existing adaptation or fine-tuning methods often rely on their global image embeddings, limiting their ability to capture fine-grained, localised attributes essential in complex, cluttered street scenes. To address this, we propose CLIP-MHAdapter, a variant of the current lightweight CLIP adaptation paradigm that appends a bottleneck MLP equipped with multi-head self-attention operating on patch tokens to model inter-patch dependencies. With approximately 1.4 million trainable parameters, CLIP-MHAdapter achieves superior or competitive accuracy across eight attribute classification tasks on the Global StreetScapes dataset, attaining new state-of-the-art results while maintaining low computational cost. The code is available at https://github.com/SpaceTimeLab/CLIP-MHAdapter.

  </details>



- **Benchmarking Adversarial Robustness and Adversarial Training Strategies for Object Detection**  
  Alexis Winter, Jean-Vincent Martini, Romaric Audigier, Angelique Loesch, Bertrand Luvison  
  _2026-02-18_ · https://arxiv.org/abs/2602.16494v1  
  <details><summary>Abstract</summary>

  Object detection models are critical components of automated systems, such as autonomous vehicles and perception-based robots, but their sensitivity to adversarial attacks poses a serious security risk. Progress in defending these models lags behind classification, hindered by a lack of standardized evaluation. It is nearly impossible to thoroughly compare attack or defense methods, as existing work uses different datasets, inconsistent efficiency metrics, and varied measures of perturbation cost. This paper addresses this gap by investigating three key questions: (1) How can we create a fair benchmark to impartially compare attacks? (2) How well do modern attacks transfer across different architectures, especially from Convolutional Neural Networks to Vision Transformers? (3) What is the most effective adversarial training strategy for robust defense? To answer these, we first propose a unified benchmark framework focused on digital, non-patch-based attacks. This framework introduces specific metrics to disentangle localization and classification errors and evaluates attack cost using multiple perceptual metrics. Using this benchmark, we conduct extensive experiments on state-of-the-art attacks and a wide range of detectors. Our findings reveal two major conclusions: first, modern adversarial attacks against object detection models show a significant lack of transferability to transformer-based architectures. Second, we demonstrate that the most robust adversarial training strategy leverages a dataset composed of a mix of high-perturbation attacks with different objectives (e.g., spatial and semantic), which outperforms training on any single attack.

  </details>



- **MMA: Multimodal Memory Agent**  
  Yihao Lu, Wanru Cheng, Zeyu Zhang, Hao Tang  
  _2026-02-18_ · https://arxiv.org/abs/2602.16493v1  
  <details><summary>Abstract</summary>

  Long-horizon multimodal agents depend on external memory; however, similarity-based retrieval often surfaces stale, low-credibility, or conflicting items, which can trigger overconfident errors. We propose Multimodal Memory Agent (MMA), which assigns each retrieved memory item a dynamic reliability score by combining source credibility, temporal decay, and conflict-aware network consensus, and uses this signal to reweight evidence and abstain when support is insufficient. We also introduce MMA-Bench, a programmatically generated benchmark for belief dynamics with controlled speaker reliability and structured text-vision contradictions. Using this framework, we uncover the "Visual Placebo Effect", revealing how RAG-based agents inherit latent visual biases from foundation models. On FEVER, MMA matches baseline accuracy while reducing variance by 35.2% and improving selective utility; on LoCoMo, a safety-oriented configuration improves actionable accuracy and reduces wrong answers; on MMA-Bench, MMA reaches 41.18% Type-B accuracy in Vision mode, while the baseline collapses to 0.0% under the same protocol. Code: https://github.com/AIGeeksGroup/MMA.

  </details>



- **Visual Self-Refine: A Pixel-Guided Paradigm for Accurate Chart Parsing**  
  Jinsong Li, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Jiaqi Wang, Dahua Lin  
  _2026-02-18_ · https://arxiv.org/abs/2602.16455v1  
  <details><summary>Abstract</summary>

  While Large Vision-Language Models (LVLMs) have demonstrated remarkable capabilities for reasoning and self-correction at the textual level, these strengths provide minimal benefits for complex tasks centered on visual perception, such as Chart Parsing. Existing models often struggle with visually dense charts, leading to errors like data omission, misalignment, and hallucination. Inspired by the human strategy of using a finger as a ``visual anchor'' to ensure accuracy when reading complex charts, we propose a new paradigm named Visual Self-Refine (VSR). The core idea of VSR is to enable a model to generate pixel-level localization outputs, visualize them, and then feed these visualizations back to itself, allowing it to intuitively inspect and correct its own potential visual perception errors. We instantiate the VSR paradigm in the domain of Chart Parsing by proposing ChartVSR. This model decomposes the parsing process into two stages: a Refine Stage, where it iteratively uses visual feedback to ensure the accuracy of all data points' Pixel-level Localizations, and a Decode Stage, where it uses these verified localizations as precise visual anchors to parse the final structured data. To address the limitations of existing benchmarks, we also construct ChartP-Bench, a new and highly challenging benchmark for chart parsing. Our work also highlights VSR as a general-purpose visual feedback mechanism, offering a promising new direction for enhancing accuracy on a wide range of vision-centric tasks.

  </details>



- **Parameter-Free Adaptive Multi-Scale Channel-Spatial Attention Aggregation framework for 3D Indoor Semantic Scene Completion Toward Assisting Visually Impaired**  
  Qi He, XiangXiang Wang, Jingtao Zhang, Yongbin Yu, Hongxiang Chu, Manping Fan, JingYe Cai, Zhenglin Yang  
  _2026-02-18_ · https://arxiv.org/abs/2602.16385v2  
  <details><summary>Abstract</summary>

  In indoor assistive perception for visually impaired users, 3D Semantic Scene Completion (SSC) is expected to provide structurally coherent and semantically consistent occupancy under strictly monocular vision for safety-critical scene understanding. However, existing monocular SSC approaches often lack explicit modeling of voxel-feature reliability and regulated cross-scale information propagation during 2D-3D projection and multi-scale fusion, making them vulnerable to projection diffusion and feature entanglement and thus limiting structural stability. To address these challenges, this paper presents an Adaptive Multi-scale Attention Aggregation (AMAA) framework built upon the MonoScene pipeline. Rather than introducing a heavier backbone, AMAA focuses on reliability-oriented feature regulation within a monocular SSC framework. Specifically, lifted voxel features are jointly calibrated in semantic and spatial dimensions through parallel channel-spatial attention aggregation, while multi-scale encoder-decoder fusion is stabilized via a hierarchical adaptive feature-gating strategy that regulates information injection across scales. Experiments on the NYUv2 benchmark demonstrate consistent improvements over MonoScene without significantly increasing system complexity: AMAA achieves 27.25% SSC mIoU (+0.31) and 43.10% SC IoU (+0.59). In addition, system-level deployment on an NVIDIA Jetson platform verifies that the complete AMAA framework can be executed stably on embedded hardware. Overall, AMAA improves monocular SSC quality and provides a reliable and deployable perception framework for indoor assistive systems targeting visually impaired users.

  </details>



- **Articulated 3D Scene Graphs for Open-World Mobile Manipulation**  
  Martin Büchner, Adrian Röfer, Tim Engelbracht, Tim Welschehold, Zuria Bauer, Hermann Blum, Marc Pollefeys, Abhinav Valada  
  _2026-02-18_ · https://arxiv.org/abs/2602.16356v1  
  <details><summary>Abstract</summary>

  Semantics has enabled 3D scene understanding and affordance-driven object interaction. However, robots operating in real-world environments face a critical limitation: they cannot anticipate how objects move. Long-horizon mobile manipulation requires closing the gap between semantics, geometry, and kinematics. In this work, we present MoMa-SG, a novel framework for building semantic-kinematic 3D scene graphs of articulated scenes containing a myriad of interactable objects. Given RGB-D sequences containing multiple object articulations, we temporally segment object interactions and infer object motion using occlusion-robust point tracking. We then lift point trajectories into 3D and estimate articulation models using a novel unified twist estimation formulation that robustly estimates revolute and prismatic joint parameters in a single optimization pass. Next, we associate objects with estimated articulations and detect contained objects by reasoning over parent-child relations at identified opening states. We also introduce the novel Arti4D-Semantic dataset, which uniquely combines hierarchical object semantics including parent-child relation labels with object axis annotations across 62 in-the-wild RGB-D sequences containing 600 object interactions and three distinct observation paradigms. We extensively evaluate the performance of MoMa-SG on two datasets and ablate key design choices of our approach. In addition, real-world experiments on both a quadruped and a mobile manipulator demonstrate that our semantic-kinematic scene graphs enable robust manipulation of articulated objects in everyday home environments. We provide code and data at: https://momasg.cs.uni-freiburg.de.

  </details>



- **SIT-LMPC: Safe Information-Theoretic Learning Model Predictive Control for Iterative Tasks**  
  Zirui Zang, Ahmad Amine, Nick-Marios T. Kokolakis, Truong X. Nghiem, Ugo Rosolia, Rahul Mangharam  
  _2026-02-18_ · https://arxiv.org/abs/2602.16187v1  
  <details><summary>Abstract</summary>

  Robots executing iterative tasks in complex, uncertain environments require control strategies that balance robustness, safety, and high performance. This paper introduces a safe information-theoretic learning model predictive control (SIT-LMPC) algorithm for iterative tasks. Specifically, we design an iterative control framework based on an information-theoretic model predictive control algorithm to address a constrained infinite-horizon optimal control problem for discrete-time nonlinear stochastic systems. An adaptive penalty method is developed to ensure safety while balancing optimality. Trajectories from previous iterations are utilized to learn a value function using normalizing flows, which enables richer uncertainty modeling compared to Gaussian priors. SIT-LMPC is designed for highly parallel execution on graphics processing units, allowing efficient real-time optimization. Benchmark simulations and hardware experiments demonstrate that SIT-LMPC iteratively improves system performance while robustly satisfying system constraints.

  </details>



- **Evaluating Demographic Misrepresentation in Image-to-Image Portrait Editing**  
  Huichan Seo, Minki Hong, Sieun Choi, Jihie Kim, Jean Oh  
  _2026-02-18_ · https://arxiv.org/abs/2602.16149v1  
  <details><summary>Abstract</summary>

  Demographic bias in text-to-image (T2I) generation is well studied, yet demographic-conditioned failures in instruction-guided image-to-image (I2I) editing remain underexplored. We examine whether identical edit instructions yield systematically different outcomes across subject demographics in open-weight I2I editors. We formalize two failure modes: Soft Erasure, where edits are silently weakened or ignored in the output image, and Stereotype Replacement, where edits introduce unrequested, stereotype-consistent attributes. We introduce a controlled benchmark that probes demographic-conditioned behavior by generating and editing portraits conditioned on race, gender, and age using a diagnostic prompt set, and evaluate multiple editors with vision-language model (VLM) scoring and human evaluation. Our analysis shows that identity preservation failures are pervasive, demographically uneven, and shaped by implicit social priors, including occupation-driven gender inference. Finally, we demonstrate that a prompt-level identity constraint, without model updates, can substantially reduce demographic change for minority groups while leaving majority-group portraits largely unchanged, revealing asymmetric identity priors in current editors. Together, our findings establish identity preservation as a central and demographically uneven failure mode in I2I editing and motivate demographic-robust editing systems. Project page: https://seochan99.github.io/i2i-demographic-bias

  </details>



- **IRIS: Intent Resolution via Inference-time Saccades for Open-Ended VQA in Large Vision-Language Models**  
  Parsa Madinei, Srijita Karmakar, Russell Cohen Hoffing, Felix Gervitz, Miguel P. Eckstein  
  _2026-02-18_ · https://arxiv.org/abs/2602.16138v1  
  <details><summary>Abstract</summary>

  We introduce IRIS (Intent Resolution via Inference-time Saccades), a novel training-free approach that uses eye-tracking data in real-time to resolve ambiguity in open-ended VQA. Through a comprehensive user study with 500 unique image-question pairs, we demonstrate that fixations closest to the time participants start verbally asking their questions are the most informative for disambiguation in Large VLMs, more than doubling the accuracy of responses on ambiguous questions (from 35.2% to 77.2%) while maintaining performance on unambiguous queries. We evaluate our approach across state-of-the-art VLMs, showing consistent improvements when gaze data is incorporated in ambiguous image-question pairs, regardless of architectural differences. We release a new benchmark dataset to use eye movement data for disambiguated VQA, a novel real-time interactive protocol, and an evaluation suite.

  </details>



- **OmniCT: Towards a Unified Slice-Volume LVLM for Comprehensive CT Analysis**  
  Tianwei Lin, Zhongwei Qiu, Wenqiao Zhang, Jiang Liu, Yihan Xie, Mingjian Gao, Zhenxuan Fan, Zhaocheng Li, Sijing Li, Zhongle Xie, et al.  
  _2026-02-18_ · https://arxiv.org/abs/2602.16110v1  
  <details><summary>Abstract</summary>

  Computed Tomography (CT) is one of the most widely used and diagnostically information-dense imaging modalities, covering critical organs such as the heart, lungs, liver, and colon. Clinical interpretation relies on both slice-driven local features (e.g., sub-centimeter nodules, lesion boundaries) and volume-driven spatial representations (e.g., tumor infiltration, inter-organ anatomical relations). However, existing Large Vision-Language Models (LVLMs) remain fragmented in CT slice versus volumetric understanding: slice-driven LVLMs show strong generalization but lack cross-slice spatial consistency, while volume-driven LVLMs explicitly capture volumetric semantics but suffer from coarse granularity and poor compatibility with slice inputs. The absence of a unified modeling paradigm constitutes a major bottleneck for the clinical translation of medical LVLMs. We present OmniCT, a powerful unified slice-volume LVLM for CT scenarios, which makes three contributions: (i) Spatial Consistency Enhancement (SCE): volumetric slice composition combined with tri-axial positional embedding that introduces volumetric consistency, and an MoE hybrid projection enables efficient slice-volume adaptation; (ii) Organ-level Semantic Enhancement (OSE): segmentation and ROI localization explicitly align anatomical regions, emphasizing lesion- and organ-level semantics; (iii) MedEval-CT: the largest slice-volume CT dataset and hybrid benchmark integrates comprehensive metrics for unified evaluation. OmniCT consistently outperforms existing methods with a substantial margin across diverse clinical tasks and satisfies both micro-level detail sensitivity and macro-level spatial reasoning. More importantly, it establishes a new paradigm for cross-modal medical imaging understanding.

  </details>



- **ScenicRules: An Autonomous Driving Benchmark with Multi-Objective Specifications and Abstract Scenarios**  
  Kevin Kai-Chun Chang, Ekin Beyazit, Alberto Sangiovanni-Vincentelli, Tichakorn Wongpiromsarn, Sanjit A. Seshia  
  _2026-02-17_ · https://arxiv.org/abs/2602.16073v1  
  <details><summary>Abstract</summary>

  Developing autonomous driving systems for complex traffic environments requires balancing multiple objectives, such as avoiding collisions, obeying traffic rules, and making efficient progress. In many situations, these objectives cannot be satisfied simultaneously, and explicit priority relations naturally arise. Also, driving rules require context, so it is important to formally model the environment scenarios within which such rules apply. Existing benchmarks for evaluating autonomous vehicles lack such combinations of multi-objective prioritized rules and formal environment models. In this work, we introduce ScenicRules, a benchmark for evaluating autonomous driving systems in stochastic environments under prioritized multi-objective specifications. We first formalize a diverse set of objectives to serve as quantitative evaluation metrics. Next, we design a Hierarchical Rulebook framework that encodes multiple objectives and their priority relations in an interpretable and adaptable manner. We then construct a compact yet representative collection of scenarios spanning diverse driving contexts and near-accident situations, formally modeled in the Scenic language. Experimental results show that our formalized objectives and Hierarchical Rulebooks align well with human driving judgments and that our benchmark effectively exposes agent failures with respect to the prioritized objectives. Our benchmark can be accessed at https://github.com/BerkeleyLearnVerify/ScenicRules/.

  </details>



- **The Impact of Class Uncertainty Propagation in Perception-Based Motion Planning**  
  Jibran Iqbal Shah, Andrei Ivanovic, Kelly Zhu, Masha Itkina, Rowan McAllister, Igor Gilitschenski, Florian Shkurti  
  _2026-02-17_ · https://arxiv.org/abs/2602.16035v1  
  <details><summary>Abstract</summary>

  Autonomous vehicles (AVs) are being increasingly deployed in urban environments. In order to operate safely and reliably, AVs need to account for the inherent uncertainty associated with perceiving the world through sensor data and incorporate that into their decision-making process. Uncertainty-aware planners have recently been developed to account for upstream perception and prediction uncertainty. However, such planners may be sensitive to prediction uncertainty miscalibration, the magnitude of which has not yet been characterized. Towards this end, we perform a detailed analysis on the impact that perceptual uncertainty propagation and calibration has on perception-based motion planning. We do so by comparing two novel prediction-planning pipelines with varying levels of uncertainty propagation on the recently-released nuPlan planning benchmark. We study the impact of upstream uncertainty calibration using closed-loop evaluation on the nuPlan challenge scenarios. We find that the method incorporating upstream uncertainty propagation demonstrates superior generalization to complex closed-loop scenarios.

  </details>



- **MedProbCLIP: Probabilistic Adaptation of Vision-Language Foundation Model for Reliable Radiograph-Report Retrieval**  
  Ahmad Elallaf, Yu Zhang, Yuktha Priya Masupalli, Jeong Yang, Young Lee, Zechun Cao, Gongbo Liang  
  _2026-02-17_ · https://arxiv.org/abs/2602.16019v1  
  <details><summary>Abstract</summary>

  Vision-language foundation models have emerged as powerful general-purpose representation learners with strong potential for multimodal understanding, but their deterministic embeddings often fail to provide the reliability required for high-stakes biomedical applications. This work introduces MedProbCLIP, a probabilistic vision-language learning framework for chest X-ray and radiology report representation learning and bidirectional retrieval. MedProbCLIP models image and text representations as Gaussian embeddings through a probabilistic contrastive objective that explicitly captures uncertainty and many-to-many correspondences between radiographs and clinical narratives. A variational information bottleneck mitigates overconfident predictions, while MedProbCLIP employs multi-view radiograph encoding and multi-section report encoding during training to provide fine-grained supervision for clinically aligned correspondence, yet requires only a single radiograph and a single report at inference. Evaluated on the MIMIC-CXR dataset, MedProbCLIP outperforms deterministic and probabilistic baselines, including CLIP, CXR-CLIP, and PCME++, in both retrieval and zero-shot classification. Beyond accuracy, MedProbCLIP demonstrates superior calibration, risk-coverage behavior, selective retrieval reliability, and robustness to clinically relevant corruptions, underscoring the value of probabilistic vision-language modeling for improving the trustworthiness and safety of radiology image-text retrieval systems.

  </details>



- **BTReport: A Framework for Brain Tumor Radiology Report Generation with Clinically Relevant Features**  
  Juampablo E. Heras Rivera, Dickson T. Chen, Tianyi Ren, Daniel K. Low, Asma Ben Abacha, Alberto Santamaria-Pang, Mehmet Kurt  
  _2026-02-17_ · https://arxiv.org/abs/2602.16006v1  
  <details><summary>Abstract</summary>

  Recent advances in radiology report generation (RRG) have been driven by large paired image-text datasets; however, progress in neuro-oncology has been limited due to a lack of open paired image-report datasets. Here, we introduce BTReport, an open-source framework for brain tumor RRG that constructs natural language radiology reports using deterministically extracted imaging features. Unlike existing approaches that rely on large general-purpose or fine-tuned vision-language models for both image interpretation and report composition, BTReport performs deterministic feature extraction for image analysis and uses large language models only for syntactic structuring and narrative formatting. By separating RRG into a deterministic feature extraction step and a report generation step, the generated reports are completely interpretable and less prone to hallucinations. We show that the features used for report generation are predictive of key clinical outcomes, including survival and IDH mutation status, and reports generated by BTReport are more closely aligned with reference clinical reports than existing baselines for RRG. Finally, we introduce BTReport-BraTS, a companion dataset that augments BraTS imaging with synthetically generated radiology reports produced with BTReport. Code for this project can be found at https://github.com/KurtLabUW/BTReport.

  </details>



- **ODYN: An All-Shifted Non-Interior-Point Method for Quadratic Programming in Robotics and AI**  
  Jose Rojas, Aristotelis Papatheodorou, Sergi Martinez, Ioannis Havoutis, Carlos Mastalli  
  _2026-02-17_ · https://arxiv.org/abs/2602.16005v1  
  <details><summary>Abstract</summary>

  We introduce ODYN, a novel all-shifted primal-dual non-interior-point quadratic programming (QP) solver designed to efficiently handle challenging dense and sparse QPs. ODYN combines all-shifted nonlinear complementarity problem (NCP) functions with proximal method of multipliers to robustly address ill-conditioned and degenerate problems, without requiring linear independence of the constraints. It exhibits strong warm-start performance and is well suited to both general-purpose optimization, and robotics and AI applications, including model-based control, estimation, and kernel-based learning methods. We provide an open-source implementation and benchmark ODYN on the Maros-Mészáros test set, demonstrating state-of-the-art convergence performance in small-to-high-scale problems. The results highlight ODYN's superior warm-starting capabilities, which are critical in sequential and real-time settings common in robotics and AI. These advantages are further demonstrated by deploying ODYN as the backend of an SQP-based predictive control framework (OdynSQP), as the implicitly differentiable optimization layer for deep learning (ODYNLayer), and the optimizer of a contact-dynamics simulation (ODYNSim).

  </details>



- **SAM 3D Body: Robust Full-Body Human Mesh Recovery**  
  Xitong Yang, Devansh Kukreja, Don Pinkus, Anushka Sagar, Taosha Fan, Jinhyung Park, Soyong Shin, Jinkun Cao, Jiawei Liu, Nicolas Ugrinovic, et al.  
  _2026-02-17_ · https://arxiv.org/abs/2602.15989v1  
  <details><summary>Abstract</summary>

  We introduce SAM 3D Body (3DB), a promptable model for single-image full-body 3D human mesh recovery (HMR) that demonstrates state-of-the-art performance, with strong generalization and consistent accuracy in diverse in-the-wild conditions. 3DB estimates the human pose of the body, feet, and hands. It is the first model to use a new parametric mesh representation, Momentum Human Rig (MHR), which decouples skeletal structure and surface shape. 3DB employs an encoder-decoder architecture and supports auxiliary prompts, including 2D keypoints and masks, enabling user-guided inference similar to the SAM family of models. We derive high-quality annotations from a multi-stage annotation pipeline that uses various combinations of manual keypoint annotation, differentiable optimization, multi-view geometry, and dense keypoint detection. Our data engine efficiently selects and processes data to ensure data diversity, collecting unusual poses and rare imaging conditions. We present a new evaluation dataset organized by pose and appearance categories, enabling nuanced analysis of model behavior. Our experiments demonstrate superior generalization and substantial improvements over prior methods in both qualitative user preference studies and traditional quantitative analysis. Both 3DB and MHR are open-source.

  </details>



- **LAND: A Longitudinal Analysis of Neuromorphic Datasets**  
  Gregory Cohen, Alexandre Marcireau  
  _2026-02-17_ · https://arxiv.org/abs/2602.15973v1  
  <details><summary>Abstract</summary>

  Neuromorphic engineering has a data problem. Despite the meteoric rise in the number of neuromorphic datasets published over the past ten years, the conclusion of a significant portion of neuromorphic research papers still states that there is a need for yet more data and even larger datasets. Whilst this need is driven in part by the sheer volume of data required by modern deep learning approaches, it is also fuelled by the current state of the available neuromorphic datasets and the difficulties in finding them, understanding their purpose, and determining the nature of their underlying task. This is further compounded by practical difficulties in downloading and using these datasets. This review starts by capturing a snapshot of the existing neuromorphic datasets, covering over 423 datasets, and then explores the nature of their tasks and the underlying structure of the presented data. Analysing these datasets shows the difficulties arising from their size, the lack of standardisation, and difficulties in accessing the actual data. This paper also highlights the growth in the size of individual datasets and the complexities involved in working with the data. However, a more important concern is the rise of synthetic datasets, created by either simulation or video-to-events methods. This review explores the benefits of simulated data for testing existing algorithms and applications, highlighting the potential pitfalls for exploring new applications of neuromorphic technologies. This review also introduces the concepts of meta-datasets, created from existing datasets, as a way of both reducing the need for more data, and to remove potential bias arising from defining both the dataset and the task.

  </details>



- **Automated Re-Identification of Holstein-Friesian Cattle in Dense Crowds**  
  Phoenix Yu, Tilo Burghardt, Andrew W Dowsey, Neill W Campbell  
  _2026-02-17_ · https://arxiv.org/abs/2602.15962v1  
  <details><summary>Abstract</summary>

  Holstein-Friesian detection and re-identification (Re-ID) methods capture individuals well when targets are spatially separate. However, existing approaches, including YOLO-based species detection, break down when cows group closely together. This is particularly prevalent for species which have outline-breaking coat patterns. To boost both effectiveness and transferability in this setting, we propose a new detect-segment-identify pipeline that leverages the Open-Vocabulary Weight-free Localisation and the Segment Anything models as pre-processing stages alongside Re-ID networks. To evaluate our approach, we publish a collection of nine days CCTV data filmed on a working dairy farm. Our methodology overcomes detection breakdown in dense animal groupings, resulting in a 98.93% accuracy. This significantly outperforms current oriented bounding box-driven, as well as SAM species detection baselines with accuracy improvements of 47.52% and 27.13%, respectively. We show that unsupervised contrastive learning can build on this to yield 94.82% Re-ID accuracy on our test data. Our work demonstrates that Re-ID in crowded scenarios is both practical as well as reliable in working farm settings with no manual intervention. Code and dataset are provided for reproducibility.

  </details>



- **Position-Aware Scene-Appearance Disentanglement for Bidirectional Photoacoustic Microscopy Registration**  
  Yiwen Wang, Jiahao Qin  
  _2026-02-17_ · https://arxiv.org/abs/2602.15959v1  
  <details><summary>Abstract</summary>

  High-speed optical-resolution photoacoustic microscopy (OR-PAM) with bidirectional raster scanning doubles imaging speed but introduces coupled domain shift and geometric misalignment between forward and backward scan lines. Existing registration methods, constrained by brightness constancy assumptions, achieve limited alignment quality, while recent generative approaches address domain shift through complex architectures that lack temporal awareness across frames. We propose GPEReg-Net, a scene-appearance disentanglement framework that separates domain-invariant scene features from domain-specific appearance codes via Adaptive Instance Normalization (AdaIN), enabling direct image-to-image registration without explicit deformation field estimation. To exploit temporal structure in sequential acquisitions, we introduce a Global Position Encoding (GPE) module that combines learnable position embeddings with sinusoidal encoding and cross-frame attention, allowing the network to leverage context from neighboring frames for improved temporal coherence. On the OR-PAM-Reg-4K benchmark (432 test samples), GPEReg-Net achieves NCC of 0.953, SSIM of 0.932, and PSNR of 34.49dB, surpassing the state-of-the-art by 3.8% in SSIM and 1.99dB in PSNR while maintaining competitive NCC. Code is available at https://github.com/JiahaoQin/GPEReg-Net.

  </details>



- **DocSplit: A Comprehensive Benchmark Dataset and Evaluation Approach for Document Packet Recognition and Splitting**  
  Md Mofijul Islam, Md Sirajus Salekin, Nivedha Balakrishnan, Vincil C. Bishop, Niharika Jain, Spencer Romo, Bob Strahan, Boyi Xie, Diego A. Socolinsky  
  _2026-02-17_ · https://arxiv.org/abs/2602.15958v1  
  <details><summary>Abstract</summary>

  Document understanding in real-world applications often requires processing heterogeneous, multi-page document packets containing multiple documents stitched together. Despite recent advances in visual document understanding, the fundamental task of document packet splitting, which involves separating a document packet into individual units, remains largely unaddressed. We present the first comprehensive benchmark dataset, DocSplit, along with novel evaluation metrics for assessing the document packet splitting capabilities of large language models. DocSplit comprises five datasets of varying complexity, covering diverse document types, layouts, and multimodal settings. We formalize the DocSplit task, which requires models to identify document boundaries, classify document types, and maintain correct page ordering within a document packet. The benchmark addresses real-world challenges, including out-of-order pages, interleaved documents, and documents lacking clear demarcations. We conduct extensive experiments evaluating multimodal LLMs on our datasets, revealing significant performance gaps in current models' ability to handle complex document splitting tasks. The DocSplit benchmark datasets and proposed novel evaluation metrics provide a systematic framework for advancing document understanding capabilities essential for legal, financial, healthcare, and other document-intensive domains. We release the datasets to facilitate future research in document packet processing.

  </details>



- **Hybrid Model Predictive Control with Physics-Informed Neural Network for Satellite Attitude Control**  
  Carlo Cena, Mauro Martini, Marcello Chiaberge  
  _2026-02-17_ · https://arxiv.org/abs/2602.15954v1  
  <details><summary>Abstract</summary>

  Reliable spacecraft attitude control depends on accurate prediction of attitude dynamics, particularly when model-based strategies such as Model Predictive Control (MPC) are employed, where performance is limited by the quality of the internal system model. For spacecraft with complex dynamics, obtaining accurate physics-based models can be difficult, time-consuming, or computationally heavy. Learning-based system identification presents a compelling alternative; however, models trained exclusively on data frequently exhibit fragile stability properties and limited extrapolation capability. This work explores Physics-Informed Neural Networks (PINNs) for modeling spacecraft attitude dynamics and contrasts it with a conventional data-driven approach. A comprehensive dataset is generated using high-fidelity numerical simulations, and two learning methodologies are investigated: a purely data-driven pipeline and a physics-regularized approach that incorporates prior knowledge into the optimization process. The results indicate that embedding physical constraints during training leads to substantial improvements in predictive reliability, achieving a 68.17% decrease in mean relative error relative. When deployed within an MPC architecture, the physics-informed models yield superior closed-loop tracking performance and improved robustness to uncertainty. Furthermore, a hybrid control formulation that merges the learned nonlinear dynamics with a nominal linear model enables consistent steady-state convergence and significantly faster response, reducing settling times by 61.52%-76.42% under measurement noise and reaction wheel friction.

  </details>



- **Task-Agnostic Continual Learning for Chest Radiograph Classification**  
  Muthu Subash Kavitha, Anas Zafar, Amgad Muneer, Jia Wu  
  _2026-02-17_ · https://arxiv.org/abs/2602.15811v1  
  <details><summary>Abstract</summary>

  Clinical deployment of chest radiograph classifiers requires models that can be updated as new datasets become available without retraining on previously ob- served data or degrading validated performance. We study, for the first time, a task-incremental continual learning setting for chest radiograph classification, in which heterogeneous chest X-ray datasets arrive sequentially and task identifiers are unavailable at inference. We propose a continual adapter-based routing learning strategy for Chest X-rays (CARL-XRay) that maintains a fixed high-capacity backbone and incrementally allocates lightweight task-specific adapters and classifier heads. A latent task selector operates on task-adapted features and leverages both current and historical context preserved through compact prototypes and feature-level experience replay. This design supports stable task identification and adaptation across sequential updates while avoiding raw-image storage. Experiments on large-scale public chest radiograph datasets demonstrate robust performance retention and reliable task-aware inference under continual dataset ingestion. CARL-XRay outperforms joint training under task-unknown deployment, achieving higher routing accuracy (75.0\% vs.\ 62.5\%), while maintaining competitive diagnostic performance with AUROC of 0.74 in the oracle setting with ground-truth task identity and 0.75 under task-unknown inference, using significantly fewer trainable parameters. Finally, the proposed framework provides a practical alternative to joint training and repeated full retraining in continual clinical deployment.

  </details>



- **A Study on Real-time Object Detection using Deep Learning**  
  Ankita Bose, Jayasravani Bhumireddy, Naveen N  
  _2026-02-17_ · https://arxiv.org/abs/2602.15926v1  
  <details><summary>Abstract</summary>

  Object detection has compelling applications over a range of domains, including human-computer interfaces, security and video surveillance, navigation and road traffic monitoring, transportation systems, industrial automation healthcare, the world of Augmented Reality (AR) and Virtual Reality (VR), environment monitoring and activity identification. Applications of real time object detection in all these areas provide dynamic analysis of the visual information that helps in immediate decision making. Furthermore, advanced deep learning algorithms leverage the progress in the field of object detection providing more accurate and efficient solutions. There are some outstanding deep learning algorithms for object detection which includes, Faster R CNN(Region-based Convolutional Neural Network),Mask R-CNN, Cascade R-CNN, YOLO (You Only Look Once), SSD (Single Shot Multibox Detector), RetinaNet etc. This article goes into great detail on how deep learning algorithms are used to enhance real time object recognition. It provides information on the different object detection models available, open benchmark datasets, and studies on the use of object detection models in a range of applications. Additionally, controlled studies are provided to compare various strategies and produce some illuminating findings. Last but not least, a number of encouraging challenges and approaches are offered as suggestions for further investigation in both relevant deep learning approaches and object recognition.

  </details>



- **Learning to Retrieve Navigable Candidates for Efficient Vision-and-Language Navigation**  
  Shutian Gu, Chengkai Huang, Ruoyu Wang, Lina Yao  
  _2026-02-17_ · https://arxiv.org/abs/2602.15724v1  
  <details><summary>Abstract</summary>

  Vision-and-Language Navigation (VLN) requires an agent to follow natural-language instructions and navigate through previously unseen environments. Recent approaches increasingly employ large language models (LLMs) as high-level navigators due to their flexibility and reasoning capability. However, prompt-based LLM navigation often suffers from inefficient decision-making, as the model must repeatedly interpret instructions from scratch and reason over noisy and verbose navigable candidates at each step. In this paper, we propose a retrieval-augmented framework to improve the efficiency and stability of LLM-based VLN without modifying or fine-tuning the underlying language model. Our approach introduces retrieval at two complementary levels. At the episode level, an instruction-level embedding retriever selects semantically similar successful navigation trajectories as in-context exemplars, providing task-specific priors for instruction grounding. At the step level, an imitation-learned candidate retriever prunes irrelevant navigable directions before LLM inference, reducing action ambiguity and prompt complexity. Both retrieval modules are lightweight, modular, and trained independently of the LLM. We evaluate our method on the Room-to-Room (R2R) benchmark. Experimental results demonstrate consistent improvements in Success Rate, Oracle Success Rate, and SPL on both seen and unseen environments. Ablation studies further show that instruction-level exemplar retrieval and candidate pruning contribute complementary benefits to global guidance and step-wise decision efficiency. These results indicate that retrieval-augmented decision support is an effective and scalable strategy for enhancing LLM-based vision-and-language navigation.

  </details>



- **Bayesian Optimization for Design Parameters of 3D Image Data Analysis**  
  David Exler, Joaquin Eduardo Urrutia Gómez, Martin Krüger, Maike Schliephake, John Jbeily, Mario Vitacolonna, Rüdiger Rudolf, Markus Reischl  
  _2026-02-17_ · https://arxiv.org/abs/2602.15660v1  
  <details><summary>Abstract</summary>

  Deep learning-based segmentation and classification are crucial to large-scale biomedical imaging, particularly for 3D data, where manual analysis is impractical. Although many methods exist, selecting suitable models and tuning parameters remains a major bottleneck in practice. Hence, we introduce the 3D data Analysis Optimization Pipeline, a method designed to facilitate the design and parameterization of segmentation and classification using two Bayesian Optimization stages. First, the pipeline selects a segmentation model and optimizes postprocessing parameters using a domain-adapted syntactic benchmark dataset. To ensure a concise evaluation of segmentation performance, we introduce a segmentation quality metric that serves as the objective function. Second, the pipeline optimizes design choices of a classifier, such as encoder and classifier head architectures, incorporation of prior knowledge, and pretraining strategies. To reduce manual annotation effort, this stage includes an assisted class-annotation workflow that extracts predicted instances from the segmentation results and sequentially presents them to the operator, eliminating the need for manual tracking. In four case studies, the 3D data Analysis Optimization Pipeline efficiently identifies effective model and parameter configurations for individual datasets.

  </details>



- **A Novel Public Dataset for Strawberry (Fragaria x ananassa) Ripeness Detection and Comparative Evaluation of YOLO-Based Models**  
  Mustafa Yurdakul, Zeynep Sena Bastug, Ali Emre Gok, Sakir Taşdemir  
  _2026-02-17_ · https://arxiv.org/abs/2602.15656v2  
  <details><summary>Abstract</summary>

  The strawberry (Fragaria x ananassa), known worldwide for its economic value and nutritional richness, is a widely cultivated fruit. Determining the correct ripeness level during the harvest period is crucial for both preventing losses for producers and ensuring consumers receive a quality product. However, traditional methods, i.e., visual assessments alone, can be subjective and have a high margin of error. Therefore, computer-assisted systems are needed. However, the scarcity of comprehensive datasets accessible to everyone in the literature makes it difficult to compare studies in this field. In this study, a new and publicly available strawberry ripeness dataset, consisting of 566 images and 1,201 labeled objects, prepared under variable light and environmental conditions in two different greenhouses in Turkey, is presented to the literature. Comparative tests conducted on the data set using YOLOv8, YOLOv9, and YOLO11-based models showed that the highest precision value was 90.94% in the YOLOv9c model, while the highest recall value was 83.74% in the YOLO11s model. In terms of the general performance criterion mAP@50, YOLOv8s was the best performing model with a success rate of 86.09%. The results show that small and medium-sized models work more balanced and efficiently on this type of dataset, while also establishing a fundamental reference point for smart agriculture applications.

  </details>



- **An Industrial Dataset for Scene Acquisitions and Functional Schematics Alignment**  
  Flavien Armangeon, Thibaud Ehret, Enric Meinhardt-Llopis, Rafael Grompone von Gioi, Guillaume Thibault, Marc Petit, Gabriele Facciolo  
  _2026-02-17_ · https://arxiv.org/abs/2602.15584v1  
  <details><summary>Abstract</summary>

  Aligning functional schematics with 2D and 3D scene acquisitions is crucial for building digital twins, especially for old industrial facilities that lack native digital models. Current manual alignment using images and LiDAR data does not scale due to tediousness and complexity of industrial sites. Inconsistencies between schematics and reality, and the scarcity of public industrial datasets, make the problem both challenging and underexplored. This paper introduces IRIS-v2, a comprehensive dataset to support further research. It includes images, point clouds, 2D annotated boxes and segmentation masks, a CAD model, 3D pipe routing information, and the P&ID (Piping and Instrumentation Diagram). The alignment is experimented on a practical case study, aiming at reducing the time required for this task by combining segmentation and graph matching.

  </details>


