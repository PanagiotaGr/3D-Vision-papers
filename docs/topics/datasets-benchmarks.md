# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-02-19 07:14 UTC_

Total papers shown: **50**


---

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
  _2026-02-18_ · https://arxiv.org/abs/2602.16611v1  
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
  _2026-02-18_ · https://arxiv.org/abs/2602.16385v1  
  <details><summary>Abstract</summary>

  In indoor assistive perception for visually impaired users, 3D Semantic Scene Completion (SSC) is expected to provide structurally coherent and semantically consistent occupancy under strictly monocular vision for safety-critical scene understanding. However, existing monocular SSC approaches often lack explicit modeling of voxel-feature reliability and regulated cross-scale information propagation during 2D-3D projection and multi-scale fusion, making them vulnerable to projection diffusion and feature entanglement and thus limiting structural stability.To address these challenges, this paper presents an Adaptive Multi-scale Attention Aggregation (AMAA) framework built upon the MonoScene pipeline. Rather than introducing a heavier backbone, AMAA focuses on reliability-oriented feature regulation within a monocular SSC framework. Specifically, lifted voxel features are jointly calibrated in semantic and spatial dimensions through parallel channel-spatial attention aggregation, while multi-scale encoder-decoder fusion is stabilized via a hierarchical adaptive feature-gating strategy that regulates information injection across scales.Experiments on the NYUv2 benchmark demonstrate consistent improvements over MonoScene without significantly increasing system complexity: AMAA achieves 27.25% SSC mIoU (+0.31) and 43.10% SC IoU (+0.59). In addition, system-level deployment on an NVIDIA Jetson platform verifies that the complete AMAA framework can be executed stably on embedded hardware. Overall, AMAA improves monocular SSC quality and provides a reliable and deployable perception framework for indoor assistive systems targeting visually impaired users.

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



- **Intracoronary Optical Coherence Tomography Image Processing and Vessel Classification Using Machine Learning**  
  Amal Lahchim, Lambros Athanasiou  
  _2026-02-17_ · https://arxiv.org/abs/2602.15579v1  
  <details><summary>Abstract</summary>

  Intracoronary Optical Coherence Tomography (OCT) enables high-resolution visualization of coronary vessel anatomy but presents challenges due to noise, imaging artifacts, and complex tissue structures. This paper proposes a fully automated pipeline for vessel segmentation and classification in OCT images using machine learning techniques. The proposed method integrates image preprocessing, guidewire artifact removal, polar-to-Cartesian transformation, unsupervised K-means clustering, and local feature extraction. These features are used to train Logistic Regression and Support Vector Machine classifiers for pixel-wise vessel classification. Experimental results demonstrate excellent performance, achieving precision, recall, and F1-score values up to 1.00 and overall classification accuracy of 99.68%. The proposed approach provides accurate vessel boundary detection while maintaining low computational complexity and requiring minimal manual annotation. This method offers a reliable and efficient solution for automated OCT image analysis and has potential applications in clinical decision support and real-time medical image processing.

  </details>



- **Selective Perception for Robot: Task-Aware Attention in Multimodal VLA**  
  Young-Chae Son, Jung-Woo Lee, Yoon-Ji Choi, Dae-Kwan Ko, Soo-Chul Lim  
  _2026-02-17_ · https://arxiv.org/abs/2602.15543v1  
  <details><summary>Abstract</summary>

  In robotics, Vision-Language-Action (VLA) models that integrate diverse multimodal signals from multi-view inputs have emerged as an effective approach. However, most prior work adopts static fusion that processes all visual inputs uniformly, which incurs unnecessary computational overhead and allows task-irrelevant background information to act as noise. Inspired by the principles of human active perception, we propose a dynamic information fusion framework designed to maximize the efficiency and robustness of VLA models. Our approach introduces a lightweight adaptive routing architecture that analyzes the current text prompt and observations from a wrist-mounted camera in real-time to predict the task-relevance of multiple camera views. By conditionally attenuating computations for views with low informational utility and selectively providing only essential visual features to the policy network, Our framework achieves computation efficiency proportional to task relevance. Furthermore, to efficiently secure large-scale annotation data for router training, we established an automated labeling pipeline utilizing Vision-Language Models (VLMs) to minimize data collection and annotation costs. Experimental results in real-world robotic manipulation scenarios demonstrate that the proposed approach achieves significant improvements in both inference efficiency and control performance compared to existing VLA models, validating the effectiveness and practicality of dynamic information fusion in resource-constrained, real-time robot control environments.

  </details>



- **Semantic-Guided 3D Gaussian Splatting for Transient Object Removal**  
  Aditi Prabakaran, Priyesh Shukla  
  _2026-02-17_ · https://arxiv.org/abs/2602.15516v1  
  <details><summary>Abstract</summary>

  Transient objects in casual multi-view captures cause ghosting artifacts in 3D Gaussian Splatting (3DGS) reconstruction. Existing solutions relied on scene decomposition at significant memory cost or on motion-based heuristics that were vulnerable to parallax ambiguity. A semantic filtering framework was proposed for category-aware transient removal using vision-language models. CLIP similarity scores between rendered views and distractor text prompts were accumulated per-Gaussian across training iterations. Gaussians exceeding a calibrated threshold underwent opacity regularization and periodic pruning. Unlike motion-based approaches, semantic classification resolved parallax ambiguity by identifying object categories independently of motion patterns. Experiments on the RobustNeRF benchmark demonstrated consistent improvement in reconstruction quality over vanilla 3DGS across four sequences, while maintaining minimal memory overhead and real-time rendering performance. Threshold calibration and comparisons with baselines validated semantic guidance as a practical strategy for transient removal in scenarios with predictable distractor categories.

  </details>



- **LEADER: Lightweight End-to-End Attention-Gated Dual Autoencoder for Robust Minutiae Extraction**  
  Raffaele Cappelli, Matteo Ferrara  
  _2026-02-17_ · https://arxiv.org/abs/2602.15493v1  
  <details><summary>Abstract</summary>

  Minutiae extraction, a fundamental stage in fingerprint recognition, is increasingly shifting toward deep learning. However, truly end-to-end methods that eliminate separate preprocessing and postprocessing steps remain scarce. This paper introduces LEADER (Lightweight End-to-end Attention-gated Dual autoencodER), a neural network that maps raw fingerprint images to minutiae descriptors, including location, direction, and type. The proposed architecture integrates non-maximum suppression and angular decoding to enable complete end-to-end inference using only 0.9M parameters. It employs a novel "Castle-Moat-Rampart" ground-truth encoding and a dual-autoencoder structure, interconnected through an attention-gating mechanism. Experimental evaluations demonstrate state-of-the-art accuracy on plain fingerprints and robust cross-domain generalization to latent impressions. Specifically, LEADER attains a 34% higher F1-score on the NIST SD27 dataset compared to specialized latent minutiae extractors. Sample-level analysis on this challenging benchmark reveals an average rank of 2.07 among all compared methods, with LEADER securing the first-place position in 47% of the samples-more than doubling the frequency of the second-best extractor. The internal representations learned by the model align with established fingerprint domain features, such as segmentation masks, orientation fields, frequency maps, and skeletons. Inference requires 15ms on GPU and 322ms on CPU, outperforming leading commercial software in computational efficiency. The source code and pre-trained weights are publicly released to facilitate reproducibility.

  </details>



- **Emergent Morphing Attack Detection in Open Multi-modal Large Language Models**  
  Marija Ivanovska, Vitomir Štruc  
  _2026-02-17_ · https://arxiv.org/abs/2602.15461v1  
  <details><summary>Abstract</summary>

  Face morphing attacks threaten biometric verification, yet most morphing attack detection (MAD) systems require task-specific training and generalize poorly to unseen attack types. Meanwhile, open-source multimodal large language models (MLLMs) have demonstrated strong visual-linguistic reasoning, but their potential in biometric forensics remains underexplored. In this paper, we present the first systematic zero-shot evaluation of open-source MLLMs for single-image MAD, using publicly available weights and a standardized, reproducible protocol. Across diverse morphing techniques, many MLLMs show non-trivial discriminative ability without any fine-tuning or domain adaptation, and LLaVA1.6-Mistral-7B achieves state-of-the-art performance, surpassing highly competitive task-specific MAD baselines by at least 23% in terms of equal error rate (EER). The results indicate that multimodal pretraining can implicitly encode fine-grained facial inconsistencies indicative of morphing artifacts, enabling zero-shot forensic sensitivity. Our findings position open-source MLLMs as reproducible, interpretable, and competitive foundations for biometric security and forensic image analysis. This emergent capability also highlights new opportunities to develop state-of-the-art MAD systems through targeted fine-tuning or lightweight adaptation, further improving accuracy and efficiency while preserving interpretability. To support future research, all code and evaluation protocols will be released upon publication.

  </details>



- **Bridging Day and Night: Target-Class Hallucination Suppression in Unpaired Image Translation**  
  Shuwei Li, Lei Tan, Robby T. Tan  
  _2026-02-17_ · https://arxiv.org/abs/2602.15383v1  
  <details><summary>Abstract</summary>

  Day-to-night unpaired image translation is important to downstream tasks but remains challenging due to large appearance shifts and the lack of direct pixel-level supervision. Existing methods often introduce semantic hallucinations, where objects from target classes such as traffic signs and vehicles, as well as man-made light effects, are incorrectly synthesized. These hallucinations significantly degrade downstream performance. We propose a novel framework that detects and suppresses hallucinations of target-class features during unpaired translation. To detect hallucination, we design a dual-head discriminator that additionally performs semantic segmentation to identify hallucinated content in background regions. To suppress these hallucinations, we introduce class-specific prototypes, constructed by aggregating features of annotated target-domain objects, which act as semantic anchors for each class. Built upon a Schrodinger Bridge-based translation model, our framework performs iterative refinement, where detected hallucination features are explicitly pushed away from class prototypes in feature space, thus preserving object semantics across the translation trajectory.Experiments show that our method outperforms existing approaches both qualitatively and quantitatively. On the BDD100K dataset, it improves mAP by 15.5% for day-to-night domain adaptation, with a notable 31.7% gain for classes such as traffic lights that are prone to hallucinations.

  </details>



- **Automatic Funny Scene Extraction from Long-form Cinematic Videos**  
  Sibendu Paul, Haotian Jiang, Caren Chen  
  _2026-02-17_ · https://arxiv.org/abs/2602.15381v1  
  <details><summary>Abstract</summary>

  Automatically extracting engaging and high-quality humorous scenes from cinematic titles is pivotal for creating captivating video previews and snackable content, boosting user engagement on streaming platforms. Long-form cinematic titles, with their extended duration and complex narratives, challenge scene localization, while humor's reliance on diverse modalities and its nuanced style add further complexity. This paper introduces an end-to-end system for automatically identifying and ranking humorous scenes from long-form cinematic titles, featuring shot detection, multimodal scene localization, and humor tagging optimized for cinematic content. Key innovations include a novel scene segmentation approach combining visual and textual cues, improved shot representations via guided triplet mining, and a multimodal humor tagging framework leveraging both audio and text. Our system achieves an 18.3% AP improvement over state-of-the-art scene detection on the OVSD dataset and an F1 score of 0.834 for detecting humor in long text. Extensive evaluations across five cinematic titles demonstrate 87% of clips extracted by our pipeline are intended to be funny, while 98% of scenes are accurately localized. With successful generalization to trailers, these results showcase the pipeline's potential to enhance content creation workflows, improve user engagement, and streamline snackable content generation for diverse cinematic media formats.

  </details>



- **EarthSpatialBench: Benchmarking Spatial Reasoning Capabilities of Multimodal LLMs on Earth Imagery**  
  Zelin Xu, Yupu Zhang, Saugat Adhikari, Saiful Islam, Tingsong Xiao, Zibo Liu, Shigang Chen, Da Yan, Zhe Jiang  
  _2026-02-17_ · https://arxiv.org/abs/2602.15918v1  
  <details><summary>Abstract</summary>

  Benchmarking spatial reasoning in multimodal large language models (MLLMs) has attracted growing interest in computer vision due to its importance for embodied AI and other agentic systems that require precise interaction with the physical world. However, spatial reasoning on Earth imagery has lagged behind, as it uniquely involves grounding objects in georeferenced images and quantitatively reasoning about distances, directions, and topological relations using both visual cues and vector geometry coordinates (e.g., 2D bounding boxes, polylines, and polygons). Existing benchmarks for Earth imagery primarily focus on 2D spatial grounding, image captioning, and coarse spatial relations (e.g., simple directional or proximity cues). They lack support for quantitative direction and distance reasoning, systematic topological relations, and complex object geometries beyond bounding boxes. To fill this gap, we propose \textbf{EarthSpatialBench}, a comprehensive benchmark for evaluating spatial reasoning in MLLMs on Earth imagery. The benchmark contains over 325K question-answer pairs spanning: (1) qualitative and quantitative reasoning about spatial distance and direction; (2) systematic topological relations; (3) single-object queries, object-pair queries, and compositional aggregate group queries; and (4) object references expressed via textual descriptions, visual overlays, and explicit geometry coordinates, including 2D bounding boxes, polylines, and polygons. We conducted extensive experiments on both open-source and proprietary models to identify limitations in the spatial reasoning of MLLMs.

  </details>



- **CREMD: Crowd-Sourced Emotional Multimodal Dogs Dataset**  
  Jinho Baek, Houwei Cao, Kate Blackwell  
  _2026-02-17_ · https://arxiv.org/abs/2602.15349v1  
  <details><summary>Abstract</summary>

  Dog emotion recognition plays a crucial role in enhancing human-animal interactions, veterinary care, and the development of automated systems for monitoring canine well-being. However, accurately interpreting dog emotions is challenging due to the subjective nature of emotional assessments and the absence of standardized ground truth methods. We present the CREMD (Crowd-sourced Emotional Multimodal Dogs Dataset), a comprehensive dataset exploring how different presentation modes (e.g., context, audio, video) and annotator characteristics (e.g., dog ownership, gender, professional experience) influence the perception and labeling of dog emotions. The dataset consists of 923 video clips presented in three distinct modes: without context or audio, with context but no audio, and with both context and audio. We analyze annotations from diverse participants, including dog owners, professionals, and individuals with varying demographic backgrounds and experience levels, to identify factors that influence reliable dog emotion recognition. Our findings reveal several key insights: (1) while adding visual context significantly improved annotation agreement, our findings regarding audio cues are inconclusive due to design limitations (specifically, the absence of a no-context-with-audio condition and limited clean audio availability); (2) contrary to expectations, non-owners and male annotators showed higher agreement levels than dog owners and female annotators, respectively, while professionals showed higher agreement levels, aligned with our initial hypothesis; and (3) the presence of audio substantially increased annotators' confidence in identifying specific emotions, particularly anger and fear.

  </details>



- **Benchmarking Self-Supervised Models for Cardiac Ultrasound View Classification**  
  Youssef Megahed, Salma I. Megahed, Robin Ducharme, Inok Lee, Adrian D. C. Chan, Mark C. Walker, Steven Hawken  
  _2026-02-17_ · https://arxiv.org/abs/2602.15339v1  
  <details><summary>Abstract</summary>

  Reliable interpretation of cardiac ultrasound images is essential for accurate clinical diagnosis and assessment. Self-supervised learning has shown promise in medical imaging by leveraging large unlabelled datasets to learn meaningful representations. In this study, we evaluate and compare two self-supervised learning frameworks, USF-MAE, developed by our team, and MoCo v3, on the recently introduced CACTUS dataset (37,736 images) for automated simulated cardiac view (A4C, PL, PSAV, PSMV, Random, and SC) classification. Both models used 5-fold cross-validation, enabling robust assessment of generalization performance across multiple random splits. The CACTUS dataset provides expert-annotated cardiac ultrasound images with diverse views. We adopt an identical training protocol for both models to ensure a fair comparison. Both models are configured with a learning rate of 0.0001 and a weight decay of 0.01. For each fold, we record performance metrics including ROC-AUC, accuracy, F1-score, and recall. Our results indicate that USF-MAE consistently outperforms MoCo v3 across metrics. The average testing AUC for USF-MAE is 99.99% (+/-0.01% 95% CI), compared to 99.97% (+/-0.01%) for MoCo v3. USF-MAE achieves a mean testing accuracy of 99.33% (+/-0.18%), higher than the 98.99% (+/-0.28%) reported for MoCo v3. Similar trends are observed for the F1-score and recall, with improvements statistically significant across folds (paired t-test, p=0.0048 < 0.01). This proof-of-concept analysis suggests that USF-MAE learns more discriminative features for cardiac view classification than MoCo v3 when applied to this dataset. The enhanced performance across multiple metrics highlights the potential of USF-MAE for improving automated cardiac ultrasound classification.

  </details>



- **Accelerating Large-Scale Dataset Distillation via Exploration-Exploitation Optimization**  
  Muhammad J. Alahmadi, Peng Gao, Feiyi Wang, Dongkuan, Xu  
  _2026-02-17_ · https://arxiv.org/abs/2602.15277v1  
  <details><summary>Abstract</summary>

  Dataset distillation compresses the original data into compact synthetic datasets, reducing training time and storage while retaining model performance, enabling deployment under limited resources. Although recent decoupling-based distillation methods enable dataset distillation at large-scale, they continue to face an efficiency gap: optimization-based decoupling methods achieve higher accuracy but demand intensive computation, whereas optimization-free decoupling methods are efficient but sacrifice accuracy. To overcome this trade-off, we propose Exploration-Exploitation Distillation (E^2D), a simple, practical method that minimizes redundant computation through an efficient pipeline that begins with full-image initialization to preserve semantic integrity and feature diversity. It then uses a two-phase optimization strategy: an exploration phase that performs uniform updates and identifies high-loss regions, and an exploitation phase that focuses updates on these regions to accelerate convergence. We evaluate E^2D on large-scale benchmarks, surpassing the state-of-the-art on ImageNet-1K while being 18x faster, and on ImageNet-21K, our method substantially improves accuracy while remaining 4.3x faster. These results demonstrate that targeted, redundancy-reducing updates, rather than brute-force optimization, bridge the gap between accuracy and efficiency in large-scale dataset distillation. Code is available at https://github.com/ncsu-dk-lab.

  </details>



- **How to Train Your Long-Context Visual Document Model**  
  Austin Veselka  
  _2026-02-16_ · https://arxiv.org/abs/2602.15257v1  
  <details><summary>Abstract</summary>

  We present the first comprehensive, large-scale study of training long-context vision language models up to 344K context, targeting long-document visual question answering with measured transfer to long-context text. While several such strong are open-weight, namely Qwen3 VL and GLM 4.5/6V, their training recipes and data pipelines are not reproducible. We systematically study continued pretraining, supervised finetuning, and preference optimization for 24B and 32B parameter models, backed by extensive LC evaluations and ablations to bridge this gap, and achieve state-of-the-art performance on MMLongBenchDoc for both parameter scales. In addition to this, our key findings include: (i) training on context lengths that match evaluation context lengths outperforms training on longer contexts, (ii) training and evaluating with page indices provides a simple, high-impact boost to long-document performance, (iii) our synthetic data pipelines enable self-improvement via continued pretraining and supervised finetuning, and (iv) we extend the known text-to-visual long context transfer to the reverse, showing that visual long context training transfers to long-context text performance. We also release MMLBD-C, a manually corrected version of MMLongBenchDoc to reduce erroneous and low quality examples in the benchmark.

  </details>



- **DexEvolve: Evolutionary Optimization for Robust and Diverse Dexterous Grasp Synthesis**  
  René Zurbrügg, Andrei Cramariuc, Marco Hutter  
  _2026-02-16_ · https://arxiv.org/abs/2602.15201v1  
  <details><summary>Abstract</summary>

  Dexterous grasping is fundamental to robotics, yet data-driven grasp prediction heavily relies on large, diverse datasets that are costly to generate and typically limited to a narrow set of gripper morphologies. Analytical grasp synthesis can be used to scale data collection, but necessary simplifying assumptions often yield physically infeasible grasps that need to be filtered in high-fidelity simulators, significantly reducing the total number of grasps and their diversity. We propose a scalable generate-and-refine pipeline for synthesizing large-scale, diverse, and physically feasible grasps. Instead of using high-fidelity simulators solely for verification and filtering, we leverage them as an optimization stage that continuously improves grasp quality without discarding precomputed candidates. More specifically, we initialize an evolutionary search with a seed set of analytically generated, potentially suboptimal grasps. We then refine these proposals directly in a high-fidelity simulator (Isaac Sim) using an asynchronous, gradient-free evolutionary algorithm, improving stability while maintaining diversity. In addition, this refinement stage can be guided toward human preferences and/or domain-specific quality metrics without requiring a differentiable objective. We further distill the refined grasp distribution into a diffusion model for robust real-world deployment, and highlight the role of diversity for both effective training and during deployment. Experiments on a newly introduced Handles dataset and a DexGraspNet subset demonstrate that our approach achieves over 120 distinct stable grasps per object (a 1.7-6x improvement over unrefined analytical methods) while outperforming diffusion-based alternatives by 46-60\% in unique grasp coverage.

  </details>



- **Distributional Deep Learning for Super-Resolution of 4D Flow MRI under Domain Shift**  
  Xiaoyi Wen, Fei Jiang  
  _2026-02-16_ · https://arxiv.org/abs/2602.15167v1  
  <details><summary>Abstract</summary>

  Super-resolution is widely used in medical imaging to enhance low-quality data, reducing scan time and improving abnormality detection. Conventional super-resolution approaches typically rely on paired datasets of downsampled and original high resolution images, training models to reconstruct high resolution images from their artificially degraded counterparts. However, in real-world clinical settings, low resolution data often arise from acquisition mechanisms that differ significantly from simple downsampling. As a result, these inputs may lie outside the domain of the training data, leading to poor model generalization due to domain shift. To address this limitation, we propose a distributional deep learning framework that improves model robustness and domain generalization. We develop this approch for enhancing the resolution of 4D Flow MRI (4DF). This is a novel imaging modality that captures hemodynamic flow velocity and clinically relevant metrics such as vessel wall stress. These metrics are critical for assessing aneurysm rupture risk. Our model is initially trained on high resolution computational fluid dynamics (CFD) simulations and their downsampled counterparts. It is then fine-tuned on a small, harmonized dataset of paired 4D Flow MRI and CFD samples. We derive the theoretical properties of our distributional estimators and demonstrate that our framework significantly outperforms traditional deep learning approaches through real data applications. This highlights the effectiveness of distributional learning in addressing domain shift and improving super-resolution performance in clinically realistic scenarios.

  </details>



- **A ROS2 Benchmarking Framework for Hierarchical Control Strategies in Mobile Robots for Mediterranean Greenhouses**  
  Fernando Cañadas-Aránega, Francisco J. Mañas-Álvarez, José L- Guzmán, José C. Moreno, José L. Blanco-Claraco  
  _2026-02-16_ · https://arxiv.org/abs/2602.15162v1  
  <details><summary>Abstract</summary>

  Mobile robots operating in agroindustrial environments, such as Mediterranean greenhouses, are subject to challenging conditions, including uneven terrain, variable friction, payload changes, and terrain slopes, all of which significantly affect control performance and stability. Despite the increasing adoption of robotic platforms in agriculture, the lack of standardized, reproducible benchmarks impedes fair comparisons and systematic evaluations of control strategies under realistic operating conditions. This paper presents a comprehensive benchmarking framework for evaluating mobile robot controllers in greenhouse environments. The proposed framework integrates an accurate three dimensional model of the environment, a physics based simulator, and a hierarchical control architecture comprising low, mid, and high level control layers. Three benchmark categories are defined to enable modular assessment, ranging from actuator level control to full autonomous navigation. Additionally, three disturbance scenarios payload variation, terrain type, and slope are explicitly modeled to replicate real world agricultural conditions. To ensure objective and reproducible evaluation, standardized performance metrics are introduced, including the Squared Absolute Error (SAE), the Squared Control Input (SCI), and composite performance indices. Statistical analysis based on repeated trials is employed to mitigate the influence of sensor noise and environmental variability. The framework is further enhanced by a plugin based architecture that facilitates seamless integration of user defined controllers and planners. The proposed benchmark provides a robust and extensible tool for the quantitative comparison of classical, predictive, and planning based control strategies in realistic conditions, bridging the gap between simulation based analysis and real world agroindustrial applications.

  </details>



- **Loss Knows Best: Detecting Annotation Errors in Videos via Loss Trajectories**  
  Praditha Alwis, Soumyadeep Chandra, Deepak Ravikumar, Kaushik Roy  
  _2026-02-16_ · https://arxiv.org/abs/2602.15154v1  
  <details><summary>Abstract</summary>

  High-quality video datasets are foundational for training robust models in tasks like action recognition, phase detection, and event segmentation. However, many real-world video datasets suffer from annotation errors such as *mislabeling*, where segments are assigned incorrect class labels, and *disordering*, where the temporal sequence does not follow the correct progression. These errors are particularly harmful in phase-annotated tasks, where temporal consistency is critical. We propose a novel, model-agnostic method for detecting annotation errors by analyzing the Cumulative Sample Loss (CSL)--defined as the average loss a frame incurs when passing through model checkpoints saved across training epochs. This per-frame loss trajectory acts as a dynamic fingerprint of frame-level learnability. Mislabeled or disordered frames tend to show consistently high or irregular loss patterns, as they remain difficult for the model to learn throughout training, while correctly labeled frames typically converge to low loss early. To compute CSL, we train a video segmentation model and store its weights at each epoch. These checkpoints are then used to evaluate the loss of each frame in a test video. Frames with persistently high CSL are flagged as likely candidates for annotation errors, including mislabeling or temporal misalignment. Our method does not require ground truth on annotation errors and is generalizable across datasets. Experiments on EgoPER and Cholec80 demonstrate strong detection performance, effectively identifying subtle inconsistencies such as mislabeling and frame disordering. The proposed approach provides a powerful tool for dataset auditing and improving training reliability in video-based machine learning.

  </details>



- **CGRA-DeBERTa Concept Guided Residual Augmentation Transformer for Theologically Islamic Understanding**  
  Tahir Hussain, Saddam Hussain Khan  
  _2026-02-16_ · https://arxiv.org/abs/2602.15139v1  
  <details><summary>Abstract</summary>

  Accurate QA over classical Islamic texts remains challenging due to domain specific semantics, long context dependencies, and concept sensitive reasoning. Therefore, a new CGRA DeBERTa, a concept guided residual domain augmentation transformer framework, is proposed that enhances theological QA over Hadith corpora. The CGRA DeBERTa builds on a customized DeBERTa transformer backbone with lightweight LoRA based adaptations and a residual concept aware gating mechanism. The customized DeBERTa embedding block learns global and positional context, while Concept Guided Residual Blocks incorporate theological priors from a curated Islamic Concept Dictionary of 12 core terms. Moreover, the Concept Gating Mechanism selectively amplifies semantically critical tokens via importance weighted attention, applying differential scaling from 1.04 to 3.00. This design preserves contextual integrity, strengthens domain-specific semantic representations, and enables accurate, efficient span extraction while maintaining computational efficiency. This paper reports the results of training CGRA using a specially constructed dataset of 42591 QA pairs from the text of Sahih alBukhari and Sahih Muslim. While BERT achieved an EM score of 75.87 and DeBERTa one of 89.77, our model scored 97.85 and thus surpassed them by 8.08 on an absolute scale, all while adding approximately 8 inference overhead due to parameter efficient gating. The qualitative evaluation noted better extraction and discrimination and theological precision. This study presents Hadith QA systems that are efficient, interpretable, and accurate and that scale provide educational materials with necessary theological nuance.

  </details>



- **Zero-shot HOI Detection with MLLM-based Detector-agnostic Interaction Recognition**  
  Shiyu Xuan, Dongkai Wang, Zechao Li, Jinhui Tang  
  _2026-02-16_ · https://arxiv.org/abs/2602.15124v1  
  <details><summary>Abstract</summary>

  Zero-shot Human-object interaction (HOI) detection aims to locate humans and objects in images and recognize their interactions. While advances in open-vocabulary object detection provide promising solutions for object localization, interaction recognition (IR) remains challenging due to the combinatorial diversity of interactions. Existing methods, including two-stage methods, tightly couple IR with a specific detector and rely on coarse-grained vision-language model (VLM) features, which limit generalization to unseen interactions. In this work, we propose a decoupled framework that separates object detection from IR and leverages multi-modal large language models (MLLMs) for zero-shot IR. We introduce a deterministic generation method that formulates IR as a visual question answering task and enforces deterministic outputs, enabling training-free zero-shot IR. To further enhance performance and efficiency by fine-tuning the model, we design a spatial-aware pooling module that integrates appearance and pairwise spatial cues, and a one-pass deterministic matching method that predicts all candidate interactions in a single forward pass. Extensive experiments on HICO-DET and V-COCO demonstrate that our method achieves superior zero-shot performance, strong cross-dataset generalization, and the flexibility to integrate with any object detectors without retraining. The codes are publicly available at https://github.com/SY-Xuan/DA-HOI.

  </details>



- **ThermEval: A Structured Benchmark for Evaluation of Vision-Language Models on Thermal Imagery**  
  Ayush Shrivastava, Kirtan Gangani, Laksh Jain, Mayank Goel, Nipun Batra  
  _2026-02-16_ · https://arxiv.org/abs/2602.14989v1  
  <details><summary>Abstract</summary>

  Vision language models (VLMs) achieve strong performance on RGB imagery, but they do not generalize to thermal images. Thermal sensing plays a critical role in settings where visible light fails, including nighttime surveillance, search and rescue, autonomous driving, and medical screening. Unlike RGB imagery, thermal images encode physical temperature rather than color or texture, requiring perceptual and reasoning capabilities that existing RGB-centric benchmarks do not evaluate. We introduce ThermEval-B, a structured benchmark of approximately 55,000 thermal visual question answering pairs designed to assess the foundational primitives required for thermal vision language understanding. ThermEval-B integrates public datasets with our newly collected ThermEval-D, the first dataset to provide dense per-pixel temperature maps with semantic body-part annotations across diverse indoor and outdoor environments. Evaluating 25 open-source and closed-source VLMs, we find that models consistently fail at temperature-grounded reasoning, degrade under colormap transformations, and default to language priors or fixed responses, with only marginal gains from prompting or supervised fine-tuning. These results demonstrate that thermal understanding requires dedicated evaluation beyond RGB-centric assumptions, positioning ThermEval as a benchmark to drive progress in thermal vision language modeling.

  </details>



- **DM0: An Embodied-Native Vision-Language-Action Model towards Physical AI**  
  En Yu, Haoran Lv, Jianjian Sun, Kangheng Lin, Ruitao Zhang, Yukang Shi, Yuyang Chen, Ze Chen, Ziheng Zhang, Fan Jia, et al.  
  _2026-02-16_ · https://arxiv.org/abs/2602.14974v1  
  <details><summary>Abstract</summary>

  Moving beyond the traditional paradigm of adapting internet-pretrained models to physical tasks, we present DM0, an Embodied-Native Vision-Language-Action (VLA) framework designed for Physical AI. Unlike approaches that treat physical grounding as a fine-tuning afterthought, DM0 unifies embodied manipulation and navigation by learning from heterogeneous data sources from the onset. Our methodology follows a comprehensive three-stage pipeline: Pretraining, Mid-Training, and Post-Training. First, we conduct large-scale unified pretraining on the Vision-Language Model (VLM) using diverse corpora--seamlessly integrating web text, autonomous driving scenarios, and embodied interaction logs-to jointly acquire semantic knowledge and physical priors. Subsequently, we build a flow-matching action expert atop the VLM. To reconcile high-level reasoning with low-level control, DM0 employs a hybrid training strategy: for embodied data, gradients from the action expert are not backpropagated to the VLM to preserve generalized representations, while the VLM remains trainable on non-embodied data. Furthermore, we introduce an Embodied Spatial Scaffolding strategy to construct spatial Chain-of-Thought (CoT) reasoning, effectively constraining the action solution space. Experiments on the RoboChallenge benchmark demonstrate that DM0 achieves state-of-the-art performance in both Specialist and Generalist settings on Table30.

  </details>


