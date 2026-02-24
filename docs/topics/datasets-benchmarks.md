# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-02-24 07:14 UTC_

Total papers shown: **50**


---

- **A Very Big Video Reasoning Suite**  
  Maijunxian Wang, Ruisi Wang, Juyi Lin, Ran Ji, Thaddäus Wiedemer, Qingying Gao, Dezhi Luo, Yaoyao Qian, Lianyu Huang, Zelong Hong, et al.  
  _2026-02-23_ · https://arxiv.org/abs/2602.20159v1  
  <details><summary>Abstract</summary>

  Rapid progress in video models has largely focused on visual quality, leaving their reasoning capabilities underexplored. Video reasoning grounds intelligence in spatiotemporally consistent visual environments that go beyond what text can naturally capture, enabling intuitive reasoning over spatiotemporal structure such as continuity, interaction, and causality. However, systematically studying video reasoning and its scaling behavior is hindered by the lack of large-scale training data. To address this gap, we introduce the Very Big Video Reasoning (VBVR) Dataset, an unprecedentedly large-scale resource spanning 200 curated reasoning tasks following a principled taxonomy and over one million video clips, approximately three orders of magnitude larger than existing datasets. We further present VBVR-Bench, a verifiable evaluation framework that moves beyond model-based judging by incorporating rule-based, human-aligned scorers, enabling reproducible and interpretable diagnosis of video reasoning capabilities. Leveraging the VBVR suite, we conduct one of the first large-scale scaling studies of video reasoning and observe early signs of emergent generalization to unseen reasoning tasks. Together, VBVR lays a foundation for the next stage of research in generalizable video reasoning. The data, benchmark toolkit, and models are publicly available at https://video-reason.com/ .

  </details>



- **Do Large Language Models Understand Data Visualization Rules?**  
  Martin Sinnona, Valentin Bonas, Emmanuel Iarussi, Viviana Siless  
  _2026-02-23_ · https://arxiv.org/abs/2602.20137v1  
  <details><summary>Abstract</summary>

  Data visualization rules-derived from decades of research in design and perception-ensure trustworthy chart communication. While prior work has shown that large language models (LLMs) can generate charts or flag misleading figures, it remains unclear whether they can reason about and enforce visualization rules directly. Constraint-based systems such as Draco encode these rules as logical constraints for precise automated checks, but maintaining symbolic encodings requires expert effort, motivating the use of LLMs as flexible rule validators. In this paper, we present the first systematic evaluation of LLMs against visualization rules using hard-verification ground truth derived from Answer Set Programming (ASP). We translated a subset of Draco's constraints into natural-language statements and generated a controlled dataset of 2,000 Vega-Lite specifications annotated with explicit rule violations. LLMs were evaluated on both accuracy in detecting violations and prompt adherence, which measures whether outputs follow the required structured format. Results show that frontier models achieve high adherence (Gemma 3 4B / 27B: 100%, GPT-oss 20B: 98%) and reliably detect common violations (F1 up to 0.82),yet performance drops for subtler perceptual rules (F1 < 0.15 for some categories) and for outputs generated from technical ASP formulations.Translating constraints into natural language improved performance by up to 150% for smaller models. These findings demonstrate the potential of LLMs as flexible, language-driven validators while highlighting their current limitations compared to symbolic solvers.

  </details>



- **NovaPlan: Zero-Shot Long-Horizon Manipulation via Closed-Loop Video Language Planning**  
  Jiahui Fu, Junyu Nan, Lingfeng Sun, Hongyu Li, Jianing Qian, Jennifer L. Barry, Kris Kitani, George Konidaris  
  _2026-02-23_ · https://arxiv.org/abs/2602.20119v1  
  <details><summary>Abstract</summary>

  Solving long-horizon tasks requires robots to integrate high-level semantic reasoning with low-level physical interaction. While vision-language models (VLMs) and video generation models can decompose tasks and imagine outcomes, they often lack the physical grounding necessary for real-world execution. We introduce NovaPlan, a hierarchical framework that unifies closed-loop VLM and video planning with geometrically grounded robot execution for zero-shot long-horizon manipulation. At the high level, a VLM planner decomposes tasks into sub-goals and monitors robot execution in a closed loop, enabling the system to recover from single-step failures through autonomous re-planning. To compute low-level robot actions, we extract and utilize both task-relevant object keypoints and human hand poses as kinematic priors from the generated videos, and employ a switching mechanism to choose the better one as a reference for robot actions, maintaining stable execution even under heavy occlusion or depth inaccuracy. We demonstrate the effectiveness of NovaPlan on three long-horizon tasks and the Functional Manipulation Benchmark (FMB). Our results show that NovaPlan can perform complex assembly tasks and exhibit dexterous error recovery behaviors without any prior demonstrations or training. Project page: https://nova-plan.github.io/

  </details>



- **Benchmarking Unlearning for Vision Transformers**  
  Kairan Zhao, Iurie Luca, Peter Triantafillou  
  _2026-02-23_ · https://arxiv.org/abs/2602.20114v1  
  <details><summary>Abstract</summary>

  Research in machine unlearning (MU) has gained strong momentum: MU is now widely regarded as a critical capability for building safe and fair AI. In parallel, research into transformer architectures for computer vision tasks has been highly successful: Increasingly, Vision Transformers (VTs) emerge as strong alternatives to CNNs. Yet, MU research for vision tasks has largely centered on CNNs, not VTs. While benchmarking MU efforts have addressed LLMs, diffusion models, and CNNs, none exist for VTs. This work is the first to attempt this, benchmarking MU algorithm performance in different VT families (ViT and Swin-T) and at different capacities. The work employs (i) different datasets, selected to assess the impacts of dataset scale and complexity; (ii) different MU algorithms, selected to represent fundamentally different approaches for MU; and (iii) both single-shot and continual unlearning protocols. Additionally, it focuses on benchmarking MU algorithms that leverage training data memorization, since leveraging memorization has been recently discovered to significantly improve the performance of previously SOTA algorithms. En route, the work characterizes how VTs memorize training data relative to CNNs, and assesses the impact of different memorization proxies on performance. The benchmark uses unified evaluation metrics that capture two complementary notions of forget quality along with accuracy on unseen (test) data and on retained data. Overall, this work offers a benchmarking basis, enabling reproducible, fair, and comprehensive comparisons of existing (and future) MU algorithms on VTs. And, for the first time, it sheds light on how well existing algorithms work in VT settings, establishing a promising reference performance baseline.

  </details>



- **Transcending the Annotation Bottleneck: AI-Powered Discovery in Biology and Medicine**  
  Soumick Chatterjee  
  _2026-02-23_ · https://arxiv.org/abs/2602.20100v1  
  <details><summary>Abstract</summary>

  The dependence on expert annotation has long constituted the primary rate-limiting step in the application of artificial intelligence to biomedicine. While supervised learning drove the initial wave of clinical algorithms, a paradigm shift towards unsupervised and self-supervised learning (SSL) is currently unlocking the latent potential of biobank-scale datasets. By learning directly from the intrinsic structure of data - whether pixels in a magnetic resonance image (MRI), voxels in a volumetric scan, or tokens in a genomic sequence - these methods facilitate the discovery of novel phenotypes, the linkage of morphology to genetics, and the detection of anomalies without human bias. This article synthesises seminal and recent advances in "learning without labels," highlighting how unsupervised frameworks can derive heritable cardiac traits, predict spatial gene expression in histology, and detect pathologies with performance that rivals or exceeds supervised counterparts.

  </details>



- **Do Large Language Models Understand Data Visualization Principles?**  
  Martin Sinnona, Valentin Bonas, Viviana Siless, Emmanuel Iarussi  
  _2026-02-23_ · https://arxiv.org/abs/2602.20084v1  
  <details><summary>Abstract</summary>

  Data visualization principles, derived from decades of research in design and perception, ensure proper visual communication. While prior work has shown that large language models (LLMs) can generate charts or flag misleading figures, it remains unclear whether they and their vision-language counterparts (VLMs) can reason about and enforce visualization principles directly. Constraint based systems encode these principles as logical rules for precise automated checks, but translating them into formal specifications demands expert knowledge. This motivates leveraging LLMs and VLMs as principle checkers that can reason about visual design directly, bypassing the need for symbolic rule specification. In this paper, we present the first systematic evaluation of both LLMs and VLMs on their ability to reason about visualization principles, using hard verification ground truth derived from Answer Set Programming (ASP). We compiled a set of visualization principles expressed as natural-language statements and generated a controlled dataset of approximately 2,000 Vega-Lite specifications annotated with explicit principle violations, complemented by over 300 real-world Vega-Lite charts. We evaluated both checking and fixing tasks, assessing how well models detect principle violations and correct flawed chart specifications. Our work highlights both the promise of large (vision-)language models as flexible validators and editors of visualization designs and the persistent gap with symbolic solvers on more nuanced aspects of visual perception. They also reveal an interesting asymmetry: frontier models tend to be more effective at correcting violations than at detecting them reliably.

  </details>



- **The Invisible Gorilla Effect in Out-of-distribution Detection**  
  Harry Anthony, Ziyun Liang, Hermione Warr, Konstantinos Kamnitsas  
  _2026-02-23_ · https://arxiv.org/abs/2602.20068v1  
  <details><summary>Abstract</summary>

  Deep Neural Networks achieve high performance in vision tasks by learning features from regions of interest (ROI) within images, but their performance degrades when deployed on out-of-distribution (OOD) data that differs from training data. This challenge has led to OOD detection methods that aim to identify and reject unreliable predictions. Although prior work shows that OOD detection performance varies by artefact type, the underlying causes remain underexplored. To this end, we identify a previously unreported bias in OOD detection: for hard-to-detect artefacts (near-OOD), detection performance typically improves when the artefact shares visual similarity (e.g. colour) with the model's ROI and drops when it does not - a phenomenon we term the Invisible Gorilla Effect. For example, in a skin lesion classifier with red lesion ROI, we show the method Mahalanobis Score achieves a 31.5% higher AUROC when detecting OOD red ink (similar to ROI) compared to black ink (dissimilar) annotations. We annotated artefacts by colour in 11,355 images from three public datasets (e.g. ISIC) and generated colour-swapped counterfactuals to rule out dataset bias. We then evaluated 40 OOD methods across 7 benchmarks and found significant performance drops for most methods when artefacts differed from the ROI. Our findings highlight an overlooked failure mode in OOD detection and provide guidance for more robust detectors. Code and annotations are available at: https://github.com/HarryAnthony/Invisible_Gorilla_Effect.

  </details>



- **MeanFuser: Fast One-Step Multi-Modal Trajectory Generation and Adaptive Reconstruction via MeanFlow for End-to-End Autonomous Driving**  
  Junli Wang, Xueyi Liu, Yinan Zheng, Zebing Xing, Pengfei Li, Guang Li, Kun Ma, Guang Chen, Hangjun Ye, Zhongpu Xia, et al.  
  _2026-02-23_ · https://arxiv.org/abs/2602.20060v1  
  <details><summary>Abstract</summary>

  Generative models have shown great potential in trajectory planning. Recent studies demonstrate that anchor-guided generative models are effective in modeling the uncertainty of driving behaviors and improving overall performance. However, these methods rely on discrete anchor vocabularies that must sufficiently cover the trajectory distribution during testing to ensure robustness, inducing an inherent trade-off between vocabulary size and model performance. To overcome this limitation, we propose MeanFuser, an end-to-end autonomous driving method that enhances both efficiency and robustness through three key designs. (1) We introduce Gaussian Mixture Noise (GMN) to guide generative sampling, enabling a continuous representation of the trajectory space and eliminating the dependency on discrete anchor vocabularies. (2) We adapt ``MeanFlow Identity" to end-to-end planning, which models the mean velocity field between GMN and trajectory distribution instead of the instantaneous velocity field used in vanilla flow matching methods, effectively eliminating numerical errors from ODE solvers and significantly accelerating inference. (3) We design a lightweight Adaptive Reconstruction Module (ARM) that enables the model to implicitly select from all sampled proposals or reconstruct a new trajectory when none is satisfactory via attention weights. Experiments on the NAVSIM closed-loop benchmark demonstrate that MeanFuser achieves outstanding performance without the supervision of the PDM Score. and exceptional inference efficiency, offering a robust and efficient solution for end-to-end autonomous driving. Our code and model are available at https://github.com/wjl2244/MeanFuser.

  </details>



- **SEAL-pose: Enhancing 3D Human Pose Estimation via a Learned Loss for Structural Consistency**  
  Yeonsung Kim, Junggeun Do, Seunguk Do, Sangmin Kim, Jaesik Park, Jay-Yoon Lee  
  _2026-02-23_ · https://arxiv.org/abs/2602.20051v1  
  <details><summary>Abstract</summary>

  3D human pose estimation (HPE) is characterized by intricate local and global dependencies among joints. Conventional supervised losses are limited in capturing these correlations because they treat each joint independently. Previous studies have attempted to promote structural consistency through manually designed priors or rule-based constraints; however, these approaches typically require manual specification and are often non-differentiable, limiting their use as end-to-end training objectives. We propose SEAL-pose, a data-driven framework in which a learnable loss-net trains a pose-net by evaluating structural plausibility. Rather than relying on hand-crafted priors, our joint-graph-based design enables the loss-net to learn complex structural dependencies directly from data. Extensive experiments on three 3D HPE benchmarks with eight backbones show that SEAL-pose reduces per-joint errors and improves pose plausibility compared with the corresponding backbones across all settings. Beyond improving each backbone, SEAL-pose also outperforms models with explicit structural constraints, despite not enforcing any such constraints. Finally, we analyze the relationship between the loss-net and structural consistency, and evaluate SEAL-pose in cross-dataset and in-the-wild settings.

  </details>



- **EEG-Driven Intention Decoding: Offline Deep Learning Benchmarking on a Robotic Rover**  
  Ghadah Alosaimi, Maha Alsayyari, Yixin Sun, Stamos Katsigiannis, Amir Atapour-Abarghouei, Toby P. Breckon  
  _2026-02-23_ · https://arxiv.org/abs/2602.20041v1  
  <details><summary>Abstract</summary>

  Brain-computer interfaces (BCIs) provide a hands-free control modality for mobile robotics, yet decoding user intent during real-world navigation remains challenging. This work presents a brain-robot control framework for offline decoding of driving commands during robotic rover operation. A 4WD Rover Pro platform was remotely operated by 12 participants who navigated a predefined route using a joystick, executing the commands forward, reverse, left, right, and stop. Electroencephalogram (EEG) signals were recorded with a 16-channel OpenBCI cap and aligned with motor actions at Delta = 0 ms and future prediction horizons (Delta > 0 ms). After preprocessing, several deep learning models were benchmarked, including convolutional neural networks, recurrent neural networks, and Transformer architectures. ShallowConvNet achieved the highest performance for both action prediction and intent prediction. By combining real-world robotic control with multi-horizon EEG intention decoding, this study introduces a reproducible benchmark and reveals key design insights for predictive deep learning-based BCI systems.

  </details>



- **RADE-Net: Robust Attention Network for Radar-Only Object Detection in Adverse Weather**  
  Christof Leitgeb, Thomas Puchleitner, Max Peter Ronecker, Daniel Watzenig  
  _2026-02-23_ · https://arxiv.org/abs/2602.19994v1  
  <details><summary>Abstract</summary>

  Automotive perception systems are obligated to meet high requirements. While optical sensors such as Camera and Lidar struggle in adverse weather conditions, Radar provides a more robust perception performance, effectively penetrating fog, rain, and snow. Since full Radar tensors have large data sizes and very few datasets provide them, most Radar-based approaches work with sparse point clouds or 2D projections, which can result in information loss. Additionally, deep learning methods show potential to extract richer and more dense features from low level Radar data and therefore significantly increase the perception performance. Therefore, we propose a 3D projection method for fast-Fourier-transformed 4D Range-Azimuth-Doppler-Elevation (RADE) tensors. Our method preserves rich Doppler and Elevation features while reducing the required data size for a single frame by 91.9% compared to a full tensor, thus achieving higher training and inference speed as well as lower model complexity. We introduce RADE-Net, a lightweight model tailored to 3D projections of the RADE tensor. The backbone enables exploitation of low-level and high-level cues of Radar tensors with spatial and channel-attention. The decoupled detection heads predict object center-points directly in the Range-Azimuth domain and regress rotated 3D bounding boxes from rich feature maps in the cartesian scene. We evaluate the model on scenes with multiple different road users and under various weather conditions on the large-scale K-Radar dataset and achieve a 16.7% improvement compared to their baseline, as well as 6.5% improvement over current Radar-only models. Additionally, we outperform several Lidar approaches in scenarios with adverse weather conditions. The code is available under https://github.com/chr-is-tof/RADE-Net.

  </details>



- **RL-RIG: A Generative Spatial Reasoner via Intrinsic Reflection**  
  Tianyu Wang, Zhiyuan Ma, Qian Wang, Xinyi Zhang, Xinwei Long, Bowen Zhou  
  _2026-02-23_ · https://arxiv.org/abs/2602.19974v1  
  <details><summary>Abstract</summary>

  Recent advancements in image generation have achieved impressive results in producing high-quality images. However, existing image generation models still generally struggle with a spatial reasoning dilemma, lacking the ability to accurately capture fine-grained spatial relationships from the prompt and correctly generate scenes with structural integrity. To mitigate this dilemma, we propose RL-RIG, a Reinforcement Learning framework for Reflection-based Image Generation. Our architecture comprises four primary components: Diffuser, Checker, Actor, and Inverse Diffuser, following a Generate-Reflect-Edit paradigm to spark the Chain of Thought reasoning ability in image generation for addressing the dilemma. To equip the model with better intuition over generation trajectories, we further develop Reflection-GRPO to train the VLM Actor for edit prompts and the Image Editor for better image quality under a given prompt, respectively. Unlike traditional approaches that solely produce visually stunning yet structurally unreasonable content, our evaluation metrics prioritize spatial accuracy, utilizing Scene Graph IoU and employing a VLM-as-a-Judge strategy to assess the spatial consistency of generated images on LAION-SG dataset. Experimental results show that RL-RIG outperforms existing state-of-the-art open-source models by up to 11% in terms of controllable and precise spatial reasoning in image generation.

  </details>



- **When Pretty Isn't Useful: Investigating Why Modern Text-to-Image Models Fail as Reliable Training Data Generators**  
  Krzysztof Adamkiewicz, Brian Moser, Stanislav Frolov, Tobias Christian Nauen, Federico Raue, Andreas Dengel  
  _2026-02-23_ · https://arxiv.org/abs/2602.19946v1  
  <details><summary>Abstract</summary>

  Recent text-to-image (T2I) diffusion models produce visually stunning images and demonstrate excellent prompt following. But do they perform well as synthetic vision data generators? In this work, we revisit the promise of synthetic data as a scalable substitute for real training sets and uncover a surprising performance regression. We generate large-scale synthetic datasets using state-of-the-art T2I models released between 2022 and 2025, train standard classifiers solely on this synthetic data, and evaluate them on real test data. Despite observable advances in visual fidelity and prompt adherence, classification accuracy on real test data consistently declines with newer T2I models as training data generators. Our analysis reveals a hidden trend: These models collapse to a narrow, aesthetic-centric distribution that undermines diversity and label-image alignment. Overall, our findings challenge a growing assumption in vision research, namely that progress in generative realism implies progress in data realism. We thus highlight an urgent need to rethink the capabilities of modern T2I models as reliable training data generators.

  </details>



- **Scaling Law of Neural Koopman Operators**  
  Abulikemu Abuduweili, Yuyang Pang, Feihan Li, Changliu Liu  
  _2026-02-23_ · https://arxiv.org/abs/2602.19943v1  
  <details><summary>Abstract</summary>

  Data-driven neural Koopman operator theory has emerged as a powerful tool for linearizing and controlling nonlinear robotic systems. However, the performance of these data-driven models fundamentally depends on the trade-off between sample size and model dimensions, a relationship for which the scaling laws have remained unclear. This paper establishes a rigorous framework to address this challenge by deriving and empirically validating scaling laws that connect sample size, latent space dimension, and downstream control quality. We derive a theoretical upper bound on the Koopman approximation error, explicitly decomposing it into sampling error and projection error. We show that these terms decay at specific rates relative to dataset size and latent dimension, providing a rigorous basis for the scaling law. Based on the theoretical results, we introduce two lightweight regularizers for the neural Koopman operator: a covariance loss to help stabilize the learned latent features and an inverse control loss to ensure the model aligns with physical actuation. The results from systematic experiments across six robotic environments confirm that model fitting error follows the derived scaling laws, and the regularizers improve dynamic model fitting fidelity, with enhanced closed-loop control performance. Together, our results provide a simple recipe for allocating effort between data collection and model capacity when learning Koopman dynamics for control.

  </details>



- **Uncertainty-Aware Rank-One MIMO Q Network Framework for Accelerated Offline Reinforcement Learning**  
  Thanh Nguyen, Tung Luu, Tri Ton, Sungwoong Kim, Chang D. Yoo  
  _2026-02-23_ · https://arxiv.org/abs/2602.19917v1  
  <details><summary>Abstract</summary>

  Offline reinforcement learning (RL) has garnered significant interest due to its safe and easily scalable paradigm. However, training under this paradigm presents its own challenge: the extrapolation error stemming from out-of-distribution (OOD) data. Existing methodologies have endeavored to address this issue through means like penalizing OOD Q-values or imposing similarity constraints on the learned policy and the behavior policy. Nonetheless, these approaches are often beset by limitations such as being overly conservative in utilizing OOD data, imprecise OOD data characterization, and significant computational overhead. To address these challenges, this paper introduces an Uncertainty-Aware Rank-One Multi-Input Multi-Output (MIMO) Q Network framework. The framework aims to enhance Offline Reinforcement Learning by fully leveraging the potential of OOD data while still ensuring efficiency in the learning process. Specifically, the framework quantifies data uncertainty and harnesses it in the training losses, aiming to train a policy that maximizes the lower confidence bound of the corresponding Q-function. Furthermore, a Rank-One MIMO architecture is introduced to model the uncertainty-aware Q-function, \TP{offering the same ability for uncertainty quantification as an ensemble of networks but with a cost nearly equivalent to that of a single network}. Consequently, this framework strikes a harmonious balance between precision, speed, and memory efficiency, culminating in improved overall performance. Extensive experimentation on the D4RL benchmark demonstrates that the framework attains state-of-the-art performance while remaining computationally efficient. By incorporating the concept of uncertainty quantification, our framework offers a promising avenue to alleviate extrapolation errors and enhance the efficiency of offline RL.

  </details>



- **Multi-Modal Representation Learning via Semi-Supervised Rate Reduction for Generalized Category Discovery**  
  Wei He, Xianghan Meng, Zhiyuan Huang, Xianbiao Qi, Rong Xiao, Chun-Guang Li  
  _2026-02-23_ · https://arxiv.org/abs/2602.19910v1  
  <details><summary>Abstract</summary>

  Generalized Category Discovery (GCD) aims to identify both known and unknown categories, with only partial labels given for the known categories, posing a challenging open-set recognition problem. State-of-the-art approaches for GCD task are usually built on multi-modality representation learning, which is heavily dependent upon inter-modality alignment. However, few of them cast a proper intra-modality alignment to generate a desired underlying structure of representation distributions. In this paper, we propose a novel and effective multi-modal representation learning framework for GCD via Semi-Supervised Rate Reduction, called SSR$^2$-GCD, to learn cross-modality representations with desired structural properties based on emphasizing to properly align intra-modality relationships. Moreover, to boost knowledge transfer, we integrate prompt candidates by leveraging the inter-modal alignment offered by Vision Language Models. We conduct extensive experiments on generic and fine-grained benchmark datasets demonstrating superior performance of our approach.

  </details>



- **Monocular Mesh Recovery and Body Measurement of Female Saanen Goats**  
  Bo Jin, Shichao Zhao, Jin Lyu, Bin Zhang, Tao Yu, Liang An, Yebin Liu, Meili Wang  
  _2026-02-23_ · https://arxiv.org/abs/2602.19896v1  
  <details><summary>Abstract</summary>

  The lactation performance of Saanen dairy goats, renowned for their high milk yield, is intrinsically linked to their body size, making accurate 3D body measurement essential for assessing milk production potential, yet existing reconstruction methods lack goat-specific authentic 3D data. To address this limitation, we establish the FemaleSaanenGoat dataset containing synchronized eight-view RGBD videos of 55 female Saanen goats (6-18 months). Using multi-view DynamicFusion, we fuse noisy, non-rigid point cloud sequences into high-fidelity 3D scans, overcoming challenges from irregular surfaces and rapid movement. Based on these scans, we develop SaanenGoat, a parametric 3D shape model specifically designed for female Saanen goats. This model features a refined template with 41 skeletal joints and enhanced udder representation, registered with our scan data. A comprehensive shape space constructed from 48 goats enables precise representation of diverse individual variations. With the help of SaanenGoat model, we get high-precision 3D reconstruction from single-view RGBD input, and achieve automated measurement of six critical body dimensions: body length, height, chest width, chest girth, hip width, and hip height. Experimental results demonstrate the superior accuracy of our method in both 3D reconstruction and body measurement, presenting a novel paradigm for large-scale 3D vision applications in precision livestock farming.

  </details>



- **Using Unsupervised Domain Adaptation Semantic Segmentation for Pulmonary Embolism Detection in Computed Tomography Pulmonary Angiogram (CTPA) Images**  
  Wen-Liang Lin, Yun-Chien Cheng  
  _2026-02-23_ · https://arxiv.org/abs/2602.19891v1  
  <details><summary>Abstract</summary>

  While deep learning has demonstrated considerable promise in computer-aided diagnosis for pulmonary embolism (PE), practical deployment in Computed Tomography Pulmonary Angiography (CTPA) is often hindered by "domain shift" and the prohibitive cost of expert annotations. To address these challenges, an unsupervised domain adaptation (UDA) framework is proposed, utilizing a Transformer backbone and a Mean-Teacher architecture for cross-center semantic segmentation. The primary focus is placed on enhancing pseudo-label reliability by learning deep structural information within the feature space. Specifically, three modules are integrated and designed for this task: (1) a Prototype Alignment (PA) mechanism to reduce category-level distribution discrepancies; (2) Global and Local Contrastive Learning (GLCL) to capture both pixel-level topological relationships and global semantic representations; and (3) an Attention-based Auxiliary Local Prediction (AALP) module designed to reinforce sensitivity to small PE lesions by automatically extracting high-information slices from Transformer attention maps. Experimental validation conducted on cross-center datasets (FUMPE and CAD-PE) demonstrates significant performance gains. In the FUMPE -> CAD-PE task, the IoU increased from 0.1152 to 0.4153, while the CAD-PE -> FUMPE task saw an improvement from 0.1705 to 0.4302. Furthermore, the proposed method achieved a 69.9% Dice score in the CT -> MRI cross-modality task on the MMWHS dataset without utilizing any target-domain labels for model selection, confirming its robustness and generalizability for diverse clinical environments.

  </details>



- **BigMaQ: A Big Macaque Motion and Animation Dataset Bridging Image and 3D Pose Representations**  
  Lucas Martini, Alexander Lappe, Anna Bognár, Rufin Vogels, Martin A. Giese  
  _2026-02-23_ · https://arxiv.org/abs/2602.19874v1  
  <details><summary>Abstract</summary>

  The recognition of dynamic and social behavior in animals is fundamental for advancing ethology, ecology, medicine and neuroscience. Recent progress in deep learning has enabled automated behavior recognition from video, yet an accurate reconstruction of the three-dimensional (3D) pose and shape has not been integrated into this process. Especially for non-human primates, mesh-based tracking efforts lag behind those for other species, leaving pose descriptions restricted to sparse keypoints that are unable to fully capture the richness of action dynamics. To address this gap, we introduce the $\textbf{Big Ma}$ca$\textbf{Q}$ue 3D Motion and Animation Dataset ($\texttt{BigMaQ}$), a large-scale dataset comprising more than 750 scenes of interacting rhesus macaques with detailed 3D pose descriptions. Extending previous surface-based animal tracking methods, we construct subject-specific textured avatars by adapting a high-quality macaque template mesh to individual monkeys. This allows us to provide pose descriptions that are more accurate than previous state-of-the-art surface-based animal tracking methods. From the original dataset, we derive BigMaQ500, an action recognition benchmark that links surface-based pose vectors to single frames across multiple individual monkeys. By pairing features extracted from established image and video encoders with and without our pose descriptors, we demonstrate substantial improvements in mean average precision (mAP) when pose information is included. With these contributions, $\texttt{BigMaQ}$ establishes the first dataset that both integrates dynamic 3D pose-shape representations into the learning task of animal action recognition and provides a rich resource to advance the study of visual appearance, posture, and social interaction in non-human primates. The code and data are publicly available at https://martinivis.github.io/BigMaQ/ .

  </details>



- **TactiVerse: Generalizing Multi-Point Tactile Sensing in Soft Robotics Using Single-Point Data**  
  Junhui Lee, Hyosung Kim, Saekwang Nam  
  _2026-02-23_ · https://arxiv.org/abs/2602.19850v1  
  <details><summary>Abstract</summary>

  Real-time prediction of deformation in highly compliant soft materials remains a significant challenge in soft robotics. While vision-based soft tactile sensors can track internal marker displacements, learning-based models for 3D contact estimation heavily depend on their training datasets, inherently limiting their ability to generalize to complex scenarios such as multi-point sensing. To address this limitation, we introduce TactiVerse, a U-Net-based framework that formulates contact geometry estimation as a spatial heatmap prediction task. Even when trained exclusively on a limited dataset of single-point indentations, our architecture achieves highly accurate single-point sensing, yielding a superior mean absolute error of 0.0589 mm compared to the 0.0612 mm of a conventional regression-based CNN baseline. Furthermore, we demonstrate that augmenting the training dataset with multi-point contact data substantially enhances the sensor's multi-point sensing capabilities, significantly improving the overall mean MAE for two-point discrimination from 1.214 mm to 0.383 mm. By successfully extrapolating complex contact geometries from fundamental interactions, this methodology unlocks advanced multi-point and large-area shape sensing. Ultimately, it significantly streamlines the development of marker-based soft sensors, offering a highly scalable solution for real-world tactile perception.

  </details>



- **M3S-Net: Multimodal Feature Fusion Network Based on Multi-scale Data for Ultra-short-term PV Power Forecasting**  
  Penghui Niu, Taotao Cai, Suqi Zhang, Junhua Gu, Ping Zhang, Qiqi Liu, Jianxin Li  
  _2026-02-23_ · https://arxiv.org/abs/2602.19832v1  
  <details><summary>Abstract</summary>

  The inherent intermittency and high-frequency variability of solar irradiance, particularly during rapid cloud advection, present significant stability challenges to high-penetration photovoltaic grids. Although multimodal forecasting has emerged as a viable mitigation strategy, existing architectures predominantly rely on shallow feature concatenation and binary cloud segmentation, thereby failing to capture the fine-grained optical features of clouds and the complex spatiotemporal coupling between visual and meteorological modalities. To bridge this gap, this paper proposes M3S-Net, a novel multimodal feature fusion network based on multi-scale data for ultra-short-term PV power forecasting. First, a multi-scale partial channel selection network leverages partial convolutions to explicitly isolate the boundary features of optically thin clouds, effectively transcending the precision limitations of coarse-grained binary masking. Second, a multi-scale sequence to image analysis network employs Fast Fourier Transform (FFT)-based time-frequency representation to disentangle the complex periodicity of meteorological data across varying time horizons. Crucially, the model incorporates a cross-modal Mamba interaction module featuring a novel dynamic C-matrix swapping mechanism. By exchanging state-space parameters between visual and temporal streams, this design conditions the state evolution of one modality on the context of the other, enabling deep structural coupling with linear computational complexity, thus overcoming the limitations of shallow concatenation. Experimental validation on the newly constructed fine-grained PV power dataset demonstrates that M3S-Net achieves a mean absolute error reduction of 6.2% in 10-minute forecasts compared to state-of-the-art baselines. The dataset and source code will be available at https://github.com/she1110/FGPD.

  </details>



- **TextShield-R1: Reinforced Reasoning for Tampered Text Detection**  
  Chenfan Qu, Yiwu Zhong, Jian Liu, Xuekang Zhu, Bohan Yu, Lianwen Jin  
  _2026-02-23_ · https://arxiv.org/abs/2602.19828v1  
  <details><summary>Abstract</summary>

  The growing prevalence of tampered images poses serious security threats, highlighting the urgent need for reliable detection methods. Multimodal large language models (MLLMs) demonstrate strong potential in analyzing tampered images and generating interpretations. However, they still struggle with identifying micro-level artifacts, exhibit low accuracy in localizing tampered text regions, and heavily rely on expensive annotations for forgery interpretation. To this end, we introduce TextShield-R1, the first reinforcement learning based MLLM solution for tampered text detection and reasoning. Specifically, our approach introduces Forensic Continual Pre-training, an easy-to-hard curriculum that well prepares the MLLM for tampered text detection by harnessing the large-scale cheap data from natural image forensic and OCR tasks. During fine-tuning, we perform Group Relative Policy Optimization with novel reward functions to reduce annotation dependency and improve reasoning capabilities. At inference time, we enhance localization accuracy via OCR Rectification, a method that leverages the MLLM's strong text recognition abilities to refine its predictions. Furthermore, to support rigorous evaluation, we introduce the Text Forensics Reasoning (TFR) benchmark, comprising over 45k real and tampered images across 16 languages, 10 tampering techniques, and diverse domains. Rich reasoning-style annotations are included, allowing for comprehensive assessment. Our TFR benchmark simultaneously addresses seven major limitations of existing benchmarks and enables robust evaluation under cross-style, cross-method, and cross-language conditions. Extensive experiments demonstrate that TextShield-R1 significantly advances the state of the art in interpretable tampered text detection.

  </details>



- **TraceVision: Trajectory-Aware Vision-Language Model for Human-Like Spatial Understanding**  
  Fan Yang, Shurong Zheng, Hongyin Zhao, Yufei Zhan, Xin Li, Yousong Zhu, Chaoyang Zhao Ming Tang, Jinqiao Wang  
  _2026-02-23_ · https://arxiv.org/abs/2602.19768v1  
  <details><summary>Abstract</summary>

  Recent Large Vision-Language Models (LVLMs) demonstrate remarkable capabilities in image understanding and natural language generation. However, current approaches focus predominantly on global image understanding, struggling to simulate human visual attention trajectories and explain associations between descriptions and specific regions. We propose TraceVision, a unified vision-language model integrating trajectory-aware spatial understanding in an end-to-end framework. TraceVision employs a Trajectory-aware Visual Perception (TVP) module for bidirectional fusion of visual features and trajectory information. We design geometric simplification to extract semantic keypoints from raw trajectories and propose a three-stage training pipeline where trajectories guide description generation and region localization. We extend TraceVision to trajectory-guided segmentation and video scene understanding, enabling cross-frame tracking and temporal attention analysis. We construct the Reasoning-based Interactive Localized Narratives (RILN) dataset to enhance logical reasoning and interpretability. Extensive experiments on trajectory-guided captioning, text-guided trajectory prediction, understanding, and segmentation demonstrate that TraceVision achieves state-of-the-art performance, establishing a foundation for intuitive spatial interaction and interpretable visual understanding.

  </details>



- **One2Scene: Geometric Consistent Explorable 3D Scene Generation from a Single Image**  
  Pengfei Wang, Liyi Chen, Zhiyuan Ma, Yanjun Guo, Guowen Zhang, Lei Zhang  
  _2026-02-23_ · https://arxiv.org/abs/2602.19766v1  
  <details><summary>Abstract</summary>

  Generating explorable 3D scenes from a single image is a highly challenging problem in 3D vision. Existing methods struggle to support free exploration, often producing severe geometric distortions and noisy artifacts when the viewpoint moves far from the original perspective. We introduce \textbf{One2Scene}, an effective framework that decomposes this ill-posed problem into three tractable sub-tasks to enable immersive explorable scene generation. We first use a panorama generator to produce anchor views from a single input image as initialization. Then, we lift these 2D anchors into an explicit 3D geometric scaffold via a generalizable, feed-forward Gaussian Splatting network. Instead of treating the panorama as a single image for reconstruction, we project it into multiple sparse anchor views and reformulate the reconstruction task as multi-view stereo matching, which allows us to leverage robust geometric priors learned from large-scale multi-view datasets. A bidirectional feature fusion module is used to enforce cross-view consistency, yielding an efficient and geometrically reliable scaffold. Finally, the scaffold serves as a strong prior for a novel view generator to produce photorealistic and geometrically accurate views at arbitrary cameras. By explicitly conditioning on a 3D-consistent scaffold to perform reconstruction, One2Scene works stably under large camera motions, supporting immersive scene exploration. Extensive experiments show that One2Scene substantially outperforms state-of-the-art methods in panorama depth estimation, feed-forward 360° reconstruction, and explorable 3D scene generation. Code and models will be released.

  </details>



- **Training Deep Stereo Matching Networks on Tree Branch Imagery: A Benchmark Study for Real-Time UAV Forestry Applications**  
  Yida Lin, Bing Xue, Mengjie Zhang, Sam Schofield, Richard Green  
  _2026-02-23_ · https://arxiv.org/abs/2602.19763v1  
  <details><summary>Abstract</summary>

  Autonomous drone-based tree pruning needs accurate, real-time depth estimation from stereo cameras. Depth is computed from disparity maps using $Z = f B/d$, so even small disparity errors cause noticeable depth mistakes at working distances. Building on our earlier work that identified DEFOM-Stereo as the best reference disparity generator for vegetation scenes, we present the first study to train and test ten deep stereo matching networks on real tree branch images. We use the Canterbury Tree Branches dataset -- 5,313 stereo pairs from a ZED Mini camera at 1080P and 720P -- with DEFOM-generated disparity maps as training targets. The ten methods cover step-by-step refinement, 3D convolution, edge-aware attention, and lightweight designs. Using perceptual metrics (SSIM, LPIPS, ViTScore) and structural metrics (SIFT/ORB feature matching), we find that BANet-3D produces the best overall quality (SSIM = 0.883, LPIPS = 0.157), while RAFT-Stereo scores highest on scene-level understanding (ViTScore = 0.799). Testing on an NVIDIA Jetson Orin Super (16 GB, independently powered) mounted on our drone shows that AnyNet reaches 6.99 FPS at 1080P -- the only near-real-time option -- while BANet-2D gives the best quality-speed balance at 1.21 FPS. We also compare 720P and 1080P processing times to guide resolution choices for forestry drone systems.

  </details>



- **Multimodal Dataset Distillation Made Simple by Prototype-Guided Data Synthesis**  
  Junhyeok Choi, Sangwoo Mo, Minwoo Chae  
  _2026-02-23_ · https://arxiv.org/abs/2602.19756v1  
  <details><summary>Abstract</summary>

  Recent advances in multimodal learning have achieved remarkable success across diverse vision-language tasks. However, such progress heavily relies on large-scale image-text datasets, making training costly and inefficient. Prior efforts in dataset filtering and pruning attempt to mitigate this issue, but still require relatively large subsets to maintain performance and fail under very small subsets. Dataset distillation offers a promising alternative, yet existing multimodal dataset distillation methods require full-dataset training and joint optimization of image pixels and text features, making them architecture-dependent and limiting cross-architecture generalization. To overcome this, we propose a learning-free dataset distillation framework that eliminates the need for large-scale training and optimization while enhancing generalization across architectures. Our method uses CLIP to extract aligned image-text embeddings, obtains prototypes, and employs an unCLIP decoder to synthesize images, enabling efficient and scalable multimodal dataset distillation. Extensive experiments demonstrate that our approach consistently outperforms optimization-based dataset distillation and subset selection methods, achieving state-of-the-art cross-architecture generalization.

  </details>



- **Towards Personalized Multi-Modal MRI Synthesis across Heterogeneous Datasets**  
  Yue Zhang, Zhizheng Zhuo, Siyao Xu, Shan Lv, Zhaoxi Liu, Jun Qiu, Qiuli Wang, Yaou Liu, S. Kevin Zhou  
  _2026-02-23_ · https://arxiv.org/abs/2602.19723v1  
  <details><summary>Abstract</summary>

  Synthesizing missing modalities in multi-modal magnetic resonance imaging (MRI) is vital for ensuring diagnostic completeness, particularly when full acquisitions are infeasible due to time constraints, motion artifacts, and patient tolerance. Recent unified synthesis models have enabled flexible synthesis tasks by accommodating various input-output configurations. However, their training and evaluation are typically restricted to a single dataset, limiting their generalizability across diverse clinical datasets and impeding practical deployment. To address this limitation, we propose PMM-Synth, a personalized MRI synthesis framework that not only supports various synthesis tasks but also generalizes effectively across heterogeneous datasets. PMM-Synth is jointly trained on multiple multi-modal MRI datasets that differ in modality coverage, disease types, and intensity distributions. It achieves cross-dataset generalization through three core innovations: a Personalized Feature Modulation module that dynamically adapts feature representations based on dataset identifier to mitigate the impact of distributional shifts; a Modality-Consistent Batch Scheduler that facilitates stable and efficient batch training under inconsistent modality conditions; and a selective supervision loss to ensure effective learning when ground truth modalities are partially missing. Evaluated on four clinical multi-modal MRI datasets, PMM-Synth consistently outperforms state-of-the-art methods in both one-to-one and many-to-one synthesis tasks, achieving superior PSNR and SSIM scores. Qualitative results further demonstrate improved preservation of anatomical structures and pathological details. Additionally, downstream tumor segmentation and radiological reporting studies suggest that PMM-Synth holds potential for supporting reliable diagnosis under real-world modality-missing scenarios.

  </details>



- **Generative 6D Pose Estimation via Conditional Flow Matching**  
  Amir Hamza, Davide Boscaini, Weihang Li, Benjamin Busam, Fabio Poiesi  
  _2026-02-23_ · https://arxiv.org/abs/2602.19719v1  
  <details><summary>Abstract</summary>

  Existing methods for instance-level 6D pose estimation typically rely on neural networks that either directly regress the pose in $\mathrm{SE}(3)$ or estimate it indirectly via local feature matching. The former struggle with object symmetries, while the latter fail in the absence of distinctive local features. To overcome these limitations, we propose a novel formulation of 6D pose estimation as a conditional flow matching problem in $\mathbb{R}^3$. We introduce Flose, a generative method that infers object poses via a denoising process conditioned on local features. While prior approaches based on conditional flow matching perform denoising solely based on geometric guidance, Flose integrates appearance-based semantic features to mitigate ambiguities caused by object symmetries. We further incorporate RANSAC-based registration to handle outliers. We validate Flose on five datasets from the established BOP benchmark. Flose outperforms prior methods with an average improvement of +4.5 Average Recall. Project Website : https://tev-fbk.github.io/Flose/

  </details>



- **Pixels Don't Lie (But Your Detector Might): Bootstrapping MLLM-as-a-Judge for Trustworthy Deepfake Detection and Reasoning Supervision**  
  Kartik Kuckreja, Parul Gupta, Muhammad Haris Khan, Abhinav Dhall  
  _2026-02-23_ · https://arxiv.org/abs/2602.19715v1  
  <details><summary>Abstract</summary>

  Deepfake detection models often generate natural-language explanations, yet their reasoning is frequently ungrounded in visual evidence, limiting reliability. Existing evaluations measure classification accuracy but overlook reasoning fidelity. We propose DeepfakeJudge, a framework for scalable reasoning supervision and evaluation, that integrates an out-of-distribution benchmark containing recent generative and editing forgeries, a human-annotated subset with visual reasoning labels, and a suite of evaluation models, that specialize in evaluating reasoning rationales without the need for explicit ground truth reasoning rationales. The Judge is optimized through a bootstrapped generator-evaluator process that scales human feedback into structured reasoning supervision and supports both pointwise and pairwise evaluation. On the proposed meta-evaluation benchmark, our reasoning-bootstrapped model achieves an accuracy of 96.2\%, outperforming \texttt{30x} larger baselines. The reasoning judge attains very high correlation with human ratings and 98.9\% percent pairwise agreement on the human-annotated meta-evaluation subset. These results establish reasoning fidelity as a quantifiable dimension of deepfake detection and demonstrate scalable supervision for interpretable deepfake reasoning. Our user study shows that participants preferred the reasonings generated by our framework 70\% of the time, in terms of faithfulness, groundedness, and usefulness, compared to those produced by other models and datasets. All of our datasets, models, and codebase are \href{https://github.com/KjAeRsTuIsK/DeepfakeJudge}{open-sourced}.

  </details>



- **Universal Pose Pretraining for Generalizable Vision-Language-Action Policies**  
  Haitao Lin, Hanyang Yu, Jingshun Huang, He Zhang, Yonggen Ling, Ping Tan, Xiangyang Xue, Yanwei Fu  
  _2026-02-23_ · https://arxiv.org/abs/2602.19710v1  
  <details><summary>Abstract</summary>

  Existing Vision-Language-Action (VLA) models often suffer from feature collapse and low training efficiency because they entangle high-level perception with sparse, embodiment-specific action supervision. Since these models typically rely on VLM backbones optimized for Visual Question Answering (VQA), they excel at semantic identification but often overlook subtle 3D state variations that dictate distinct action patterns. To resolve these misalignments, we propose Pose-VLA, a decoupled paradigm that separates VLA training into a pre-training phase for extracting universal 3D spatial priors in a unified camera-centric space, and a post-training phase for efficient embodiment alignment within robot-specific action space. By introducing discrete pose tokens as a universal representation, Pose-VLA seamlessly integrates spatial grounding from diverse 3D datasets with geometry-level trajectories from robotic demonstrations. Our framework follows a two-stage pre-training pipeline, establishing fundamental spatial grounding via poses followed by motion alignment through trajectory supervision. Extensive evaluations demonstrate that Pose-VLA achieves state-of-the-art results on RoboTwin 2.0 with a 79.5% average success rate and competitive performance on LIBERO at 96.0%. Real-world experiments further showcase robust generalization across diverse objects using only 100 demonstrations per task, validating the efficiency of our pre-training paradigm.

  </details>



- **ChimeraLoRA: Multi-Head LoRA-Guided Synthetic Datasets**  
  Hoyoung Kim, Minwoo Jang, Jabin Koo, Sangdoo Yun, Jungseul Ok  
  _2026-02-23_ · https://arxiv.org/abs/2602.19708v1  
  <details><summary>Abstract</summary>

  Beyond general recognition tasks, specialized domains including privacy-constrained medical applications and fine-grained settings often encounter data scarcity, especially for tail classes. To obtain less biased and more reliable models under such scarcity, practitioners leverage diffusion models to supplement underrepresented regions of real data. Specifically, recent studies fine-tune pretrained diffusion models with LoRA on few-shot real sets to synthesize additional images. While an image-wise LoRA trained on a single image captures fine-grained details yet offers limited diversity, a class-wise LoRA trained over all shots produces diverse images as it encodes class priors yet tends to overlook fine details. To combine both benefits, we separate the adapter into a class-shared LoRA~$A$ for class priors and per-image LoRAs~$\mathcal{B}$ for image-specific characteristics. To expose coherent class semantics in the shared LoRA~$A$, we propose a semantic boosting by preserving class bounding boxes during training. For generation, we compose $A$ with a mixture of $\mathcal{B}$ using coefficients drawn from a Dirichlet distribution. Across diverse datasets, our synthesized images are both diverse and detail-rich while closely aligning with the few-shot real distribution, yielding robust gains in downstream classification accuracy.

  </details>



- **RAID: Retrieval-Augmented Anomaly Detection**  
  Mingxiu Cai, Zhe Zhang, Gaochang Wu, Tianyou Chai, Xiatian Zhu  
  _2026-02-23_ · https://arxiv.org/abs/2602.19611v1  
  <details><summary>Abstract</summary>

  Unsupervised Anomaly Detection (UAD) aims to identify abnormal regions by establishing correspondences between test images and normal templates. Existing methods primarily rely on image reconstruction or template retrieval but face a fundamental challenge: matching between test images and normal templates inevitably introduces noise due to intra-class variations, imperfect correspondences, and limited templates. Observing that Retrieval-Augmented Generation (RAG) leverages retrieved samples directly in the generation process, we reinterpret UAD through this lens and introduce \textbf{RAID}, a retrieval-augmented UAD framework designed for noise-resilient anomaly detection and localization. Unlike standard RAG that enriches context or knowledge, we focus on using retrieved normal samples to guide noise suppression in anomaly map generation. RAID retrieves class-, semantic-, and instance-level representations from a hierarchical vector database, forming a coarse-to-fine pipeline. A matching cost volume correlates the input with retrieved exemplars, followed by a guided Mixture-of-Experts (MoE) network that leverages the retrieved samples to adaptively suppress matching noise and produce fine-grained anomaly maps. RAID achieves state-of-the-art performance across full-shot, few-shot, and multi-dataset settings on MVTec, VisA, MPDD, and BTAD benchmarks. \href{https://github.com/Mingxiu-Cai/RAID}{https://github.com/Mingxiu-Cai/RAID}.

  </details>



- **Satellite-Based Detection of Looted Archaeological Sites Using Machine Learning**  
  Girmaw Abebe Tadesse, Titien Bartette, Andrew Hassanali, Allen Kim, Jonathan Chemla, Andrew Zolli, Yves Ubelmann, Caleb Robinson, Inbal Becker-Reshef, Juan Lavista Ferres  
  _2026-02-23_ · https://arxiv.org/abs/2602.19608v1  
  <details><summary>Abstract</summary>

  Looting at archaeological sites poses a severe risk to cultural heritage, yet monitoring thousands of remote locations remains operationally difficult. We present a scalable and satellite-based pipeline to detect looted archaeological sites, using PlanetScope monthly mosaics (4.7m/pixel) and a curated dataset of 1,943 archaeological sites in Afghanistan (898 looted, 1,045 preserved) with multi-year imagery (2016--2023) and site-footprint masks. We compare (i) end-to-end CNN classifiers trained on raw RGB patches and (ii) traditional machine learning (ML) trained on handcrafted spectral/texture features and embeddings from recent remote-sensing foundation models. Results indicate that ImageNet-pretrained CNNs combined with spatial masking reach an F1 score of 0.926, clearly surpassing the strongest traditional ML setup, which attains an F1 score of 0.710 using SatCLIP-V+RF+Mean, i.e., location and vision embeddings fed into a Random Forest with mean-based temporal aggregation. Ablation studies demonstrate that ImageNet pretraining (even in the presence of domain shift) and spatial masking enhance performance. In contrast, geospatial foundation model embeddings perform competitively with handcrafted features, suggesting that looting signatures are extremely localized. The repository is available at https://github.com/microsoft/looted_site_detection.

  </details>



- **Chasing Ghosts: A Simulation-to-Real Olfactory Navigation Stack with Optional Vision Augmentation**  
  Kordel K. France, Ovidiu Daescu, Latifur Khan, Rohith Peddi  
  _2026-02-23_ · https://arxiv.org/abs/2602.19577v1  
  <details><summary>Abstract</summary>

  Autonomous odor source localization remains a challenging problem for aerial robots due to turbulent airflow, sparse and delayed sensory signals, and strict payload and compute constraints. While prior unmanned aerial vehicle (UAV)-based olfaction systems have demonstrated gas distribution mapping or reactive plume tracing, they rely on predefined coverage patterns, external infrastructure, or extensive sensing and coordination. In this work, we present a complete, open-source UAV system for online odor source localization using a minimal sensor suite. The system integrates custom olfaction hardware, onboard sensing, and a learning-based navigation policy trained in simulation and deployed on a real quadrotor. Through our minimal framework, the UAV is able to navigate directly toward an odor source without constructing an explicit gas distribution map or relying on external positioning systems. Vision is incorporated as an optional complementary modality to accelerate navigation under certain conditions. We validate the proposed system through real-world flight experiments in a large indoor environment using an ethanol source, demonstrating consistent source-finding behavior under realistic airflow conditions. The primary contribution of this work is a reproducible system and methodological framework for UAV-based olfactory navigation and source finding under minimal sensing assumptions. We elaborate on our hardware design and open source our UAV firmware, simulation code, olfaction-vision dataset, and circuit board to the community. Code, data, and designs will be made available at https://github.com/KordelFranceTech/ChasingGhosts.

  </details>



- **HOCA-Bench: Beyond Semantic Perception to Predictive World Modeling via Hegelian Ontological-Causal Anomalies**  
  Chang Liu, Yunfan Ye, Qingyang Zhou, Xichen Tan, Mengxuan Luo, Zhenyu Qiu, Wei Peng, Zhiping Cai  
  _2026-02-23_ · https://arxiv.org/abs/2602.19571v1  
  <details><summary>Abstract</summary>

  Video-LLMs have improved steadily on semantic perception, but they still fall short on predictive world modeling, which is central to physically grounded intelligence. We introduce HOCA-Bench, a benchmark that frames physical anomalies through a Hegelian lens. HOCA-Bench separates anomalies into two types: ontological anomalies, where an entity violates its own definition or persistence, and causal anomalies, where interactions violate physical relations. Using state-of-the-art generative video models as adversarial simulators, we build a testbed of 1,439 videos (3,470 QA pairs). Evaluations on 17 Video-LLMs show a clear cognitive lag: models often identify static ontological violations (e.g., shape mutations) but struggle with causal mechanisms (e.g., gravity or friction), with performance dropping by more than 20% on causal tasks. System-2 "Thinking" modes improve reasoning, but they do not close the gap, suggesting that current architectures recognize visual patterns more readily than they apply basic physical laws.

  </details>



- **DICArt: Advancing Category-level Articulated Object Pose Estimation in Discrete State-Spaces**  
  Li Zhang, Mingyu Mei, Ailing Wang, Xianhui Meng, Yan Zhong, Xinyuan Song, Liu Liu, Rujing Wang, Zaixing He, Cewu Lu  
  _2026-02-23_ · https://arxiv.org/abs/2602.19565v1  
  <details><summary>Abstract</summary>

  Articulated object pose estimation is a core task in embodied AI. Existing methods typically regress poses in a continuous space, but often struggle with 1) navigating a large, complex search space and 2) failing to incorporate intrinsic kinematic constraints. In this work, we introduce DICArt (DIsCrete Diffusion for Articulation Pose Estimation), a novel framework that formulates pose estimation as a conditional discrete diffusion process. Instead of operating in a continuous domain, DICArt progressively denoises a noisy pose representation through a learned reverse diffusion procedure to recover the GT pose. To improve modeling fidelity, we propose a flexible flow decider that dynamically determines whether each token should be denoised or reset, effectively balancing the real and noise distributions during diffusion. Additionally, we incorporate a hierarchical kinematic coupling strategy, estimating the pose of each rigid part hierarchically to respect the object's kinematic structure. We validate DICArt on both synthetic and real-world datasets. Experimental results demonstrate its superior performance and robustness. By integrating discrete generative modeling with structural priors, DICArt offers a new paradigm for reliable category-level 6D pose estimation in complex environments.

  </details>



- **A Multimodal Framework for Aligning Human Linguistic Descriptions with Visual Perceptual Data**  
  Joseph Bingham  
  _2026-02-23_ · https://arxiv.org/abs/2602.19562v1  
  <details><summary>Abstract</summary>

  Establishing stable mappings between natural language expressions and visual percepts is a foundational problem for both cognitive science and artificial intelligence. Humans routinely ground linguistic reference in noisy, ambiguous perceptual contexts, yet the mechanisms supporting such cross-modal alignment remain poorly understood. In this work, we introduce a computational framework designed to model core aspects of human referential interpretation by integrating linguistic utterances with perceptual representations derived from large-scale, crowd-sourced imagery. The system approximates human perceptual categorization by combining scale-invariant feature transform (SIFT) alignment with the Universal Quality Index (UQI) to quantify similarity in a cognitively plausible feature space, while a set of linguistic preprocessing and query-transformation operations captures pragmatic variability in referring expressions. We evaluate the model on the Stanford Repeated Reference Game corpus (15,000 utterances paired with tangram stimuli), a paradigm explicitly developed to probe human-level perceptual ambiguity and coordination. Our framework achieves robust referential grounding. It requires 65\% fewer utterances than human interlocutors to reach stable mappings and can correctly identify target objects from single referring expressions 41.66\% of the time (versus 20\% for humans).These results suggest that relatively simple perceptual-linguistic alignment mechanisms can yield human-competitive behavior on a classic cognitive benchmark, and offers insights into models of grounded communication, perceptual inference, and cross-modal concept formation. Code is available at https://anonymous.4open.science/r/metasequoia-9D13/README.md .

  </details>



- **Can a Teenager Fool an AI? Evaluating Low-Cost Cosmetic Attacks on Age Estimation Systems**  
  Xingyu Shen, Tommy Duong, Xiaodong An, Zengqi Zhao, Zebang Hu, Haoyu Hu, Ziyou Wang, Finn Guo, Simiao Ren  
  _2026-02-23_ · https://arxiv.org/abs/2602.19539v1  
  <details><summary>Abstract</summary>

  Age estimation systems are increasingly deployed as gatekeepers for age-restricted online content, yet their robustness to cosmetic modifications has not been systematically evaluated. We investigate whether simple, household-accessible cosmetic changes, including beards, grey hair, makeup, and simulated wrinkles, can cause AI age estimators to classify minors as adults. To study this threat at scale without ethical concerns, we simulate these physical attacks on 329 facial images of individuals aged 10 to 21 using a VLM image editor (Gemini 2.5 Flash Image). We then evaluate eight models from our prior benchmark: five specialized architectures (MiVOLO, Custom-Best, Herosan, MiViaLab, DEX) and three vision-language models (Gemini 3 Flash, Gemini 2.5 Flash, GPT-5-Nano). We introduce the Attack Conversion Rate (ACR), defined as the fraction of images predicted as minor at baseline that flip to adult after attack, a population-agnostic metric that does not depend on the ratio of minors to adults in the test set. Our results reveal that a synthetic beard alone achieves 28 to 69 percent ACR across all eight models; combining all four attacks shifts predicted age by +7.7 years on average across all 329 subjects and reaches up to 83 percent ACR; and vision-language models exhibit lower ACR (59 to 71 percent) than specialized models (63 to 83 percent) under the full attack, although the ACR ranges overlap and the difference is not statistically tested. These findings highlight a critical vulnerability in deployed age-verification pipelines and call for adversarial robustness evaluation as a mandatory criterion for model selection.

  </details>



- **OSInsert: Towards High-authenticity and High-fidelity Image Composition**  
  Jingyuan Wang, Li Niu  
  _2026-02-23_ · https://arxiv.org/abs/2602.19523v1  
  <details><summary>Abstract</summary>

  Generative image composition aims to regenerate the given foreground object in the background image to produce a realistic composite image. Some high-authenticity methods can adjust foreground pose/view to be compatible with background, while some high-fidelity methods can preserve the foreground details accurately. However, existing methods can hardly achieve both goals at the same time. In this work, we propose a two-stage strategy to achieve both goals. In the first stage, we use high-authenticity method to generate reasonable foreground shape, serving as the condition of high-fidelity method in the second stage. The experiments on MureCOM dataset verify the effectiveness of our two-stage strategy. The code and model have been released at https://github.com/bcmi/OSInsert-Image-Composition.

  </details>



- **Classroom Final Exam: An Instructor-Tested Reasoning Benchmark**  
  Chongyang Gao, Diji Yang, Shuyan Zhou, Xichen Yan, Luchuan Song, Shuo Li, Kezhen Chen  
  _2026-02-23_ · https://arxiv.org/abs/2602.19517v1  
  <details><summary>Abstract</summary>

  We introduce \CFE{} (\textbf{C}lassroom \textbf{F}inal \textbf{E}xam), a multimodal benchmark for evaluating the reasoning capabilities of large language models across more than 20 STEM domains. \CFE{} is curated from repeatedly used, authentic university homework and exam problems, together with reference solutions provided by course instructors. \CFE{} presents a significant challenge even for frontier models: the newly released Gemini-3.1-pro-preview achieves an overall accuracy of 59.69\%, while the second-best model, Gemini-3-flash-preview, reaches 55.46\%, leaving considerable room for improvement. Beyond leaderboard results, we perform a diagnostic analysis by decomposing reference solutions into reasoning flows. We find that although frontier models can often answer intermediate sub-questions correctly, they struggle to reliably derive and maintain correct intermediate states throughout multi-step solutions. We further observe that model-generated solutions typically have more reasoning steps than those provided by the instructor, indicating suboptimal step efficiency and a higher risk of error accumulation. The data and code are available at https://github.com/Analogy-AI/CFE_Bench.

  </details>



- **A Text-Guided Vision Model for Enhanced Recognition of Small Instances**  
  Hyun-Ki Jung  
  _2026-02-23_ · https://arxiv.org/abs/2602.19503v1  
  <details><summary>Abstract</summary>

  As drone-based object detection technology continues to evolve, the demand is shifting from merely detecting objects to enabling users to accurately identify specific targets. For example, users can input particular targets as prompts to precisely detect desired objects. To address this need, an efficient text-guided object detection model has been developed to enhance the detection of small objects. Specifically, an improved version of the existing YOLO-World model is introduced. The proposed method replaces the C2f layer in the YOLOv8 backbone with a C3k2 layer, enabling more precise representation of local features, particularly for small objects or those with clearly defined boundaries. Additionally, the proposed architecture improves processing speed and efficiency through parallel processing optimization, while also contributing to a more lightweight model design. Comparative experiments on the VisDrone dataset show that the proposed model outperforms the original YOLO-World model, with precision increasing from 40.6% to 41.6%, recall from 30.8% to 31%, F1 score from 35% to 35.5%, and mAP@0.5 from 30.4% to 30.7%, confirming its enhanced accuracy. Furthermore, the model demonstrates superior lightweight performance, with the parameter count reduced from 4 million to 3.8 million and FLOPs decreasing from 15.7 billion to 15.2 billion. These results indicate that the proposed approach provides a practical and effective solution for precise object detection in drone-based applications.

  </details>



- **MICON-Bench: Benchmarking and Enhancing Multi-Image Context Image Generation in Unified Multimodal Models**  
  Mingrui Wu, Hang Liu, Jiayi Ji, Xiaoshuai Sun, Rongrong Ji  
  _2026-02-23_ · https://arxiv.org/abs/2602.19497v1  
  <details><summary>Abstract</summary>

  Recent advancements in Unified Multimodal Models (UMMs) have enabled remarkable image understanding and generation capabilities. However, while models like Gemini-2.5-Flash-Image show emerging abilities to reason over multiple related images, existing benchmarks rarely address the challenges of multi-image context generation, focusing mainly on text-to-image or single-image editing tasks. In this work, we introduce \textbf{MICON-Bench}, a comprehensive benchmark covering six tasks that evaluate cross-image composition, contextual reasoning, and identity preservation. We further propose an MLLM-driven Evaluation-by-Checkpoint framework for automatic verification of semantic and visual consistency, where multimodal large language model (MLLM) serves as a verifier. Additionally, we present \textbf{Dynamic Attention Rebalancing (DAR)}, a training-free, plug-and-play mechanism that dynamically adjusts attention during inference to enhance coherence and reduce hallucinations. Extensive experiments on various state-of-the-art open-source models demonstrate both the rigor of MICON-Bench in exposing multi-image reasoning challenges and the efficacy of DAR in improving generation quality and cross-image coherence. Github: https://github.com/Angusliuuu/MICON-Bench.

  </details>



- **FinSight-Net:A Physics-Aware Decoupled Network with Frequency-Domain Compensation for Underwater Fish Detection in Smart Aquaculture**  
  Jinsong Yang, Zeyuan Hu, Yichen Li, Hong Yu  
  _2026-02-23_ · https://arxiv.org/abs/2602.19437v1  
  <details><summary>Abstract</summary>

  Underwater fish detection (UFD) is a core capability for smart aquaculture and marine ecological monitoring. While recent detectors improve accuracy by stacking feature extractors or introducing heavy attention modules, they often incur substantial computational overhead and, more importantly, neglect the physics that fundamentally limits UFD: wavelength-dependent absorption and turbidity-induced scattering significantly degrade contrast, blur fine structures, and introduce backscattering noise, leading to unreliable localization and recognition. To address these challenges, we propose FinSight-Net, an efficient and physics-aware detection framework tailored for complex aquaculture environments. FinSight-Net introduces a Multi-Scale Decoupled Dual-Stream Processing (MS-DDSP) bottleneck that explicitly targets frequency-specific information loss via heterogeneous convolutional branches, suppressing backscattering artifacts while compensating distorted biological cues through scale-aware and channel-weighted pathways. We further design an Efficient Path Aggregation FPN (EPA-FPN) as a detail-filling mechanism: it restores high-frequency spatial information typically attenuated in deep layers by establishing long-range skip connections and pruning redundant fusion routes, enabling robust detection of non-rigid fish targets under severe blur and turbidity. Extensive experiments on DeepFish, AquaFishSet, and our challenging UW-BlurredFish benchmark demonstrate that FinSight-Net achieves state-of-the-art performance. In particular, on UW-BlurredFish, FinSight-Net reaches 92.8% mAP, outperforming YOLOv11s by 4.8% while reducing parameters by 29.0%, providing a strong and lightweight solution for real-time automated monitoring in smart aquaculture.

  </details>



- **CountEx: Fine-Grained Counting via Exemplars and Exclusion**  
  Yifeng Huang, Gia Khanh Nguyen, Minh Hoai  
  _2026-02-23_ · https://arxiv.org/abs/2602.19432v1  
  <details><summary>Abstract</summary>

  This paper presents CountEx, a discriminative visual counting framework designed to address a key limitation of existing prompt-based methods: the inability to explicitly exclude visually similar distractors. While current approaches allow users to specify what to count via inclusion prompts, they often struggle in cluttered scenes with confusable object categories, leading to ambiguity and overcounting. CountEx enables users to express both inclusion and exclusion intent, specifying what to count and what to ignore, through multimodal prompts including natural language descriptions and optional visual exemplars. At the core of CountEx is a novel Discriminative Query Refinement module, which jointly reasons over inclusion and exclusion cues by first identifying shared visual features, then isolating exclusion-specific patterns, and finally applying selective suppression to refine the counting query. To support systematic evaluation of fine-grained counting methods, we introduce CoCount, a benchmark comprising 1,780 videos and 10,086 annotated frames across 97 category pairs. Experiments show that CountEx achieves substantial improvements over state-of-the-art methods for counting objects from both known and novel categories. The data and code are available at https://github.com/bbvisual/CountEx.

  </details>



- **TherA: Thermal-Aware Visual-Language Prompting for Controllable RGB-to-Thermal Infrared Translation**  
  Dong-Guw Lee, Tai Hyoung Rhee, Hyunsoo Jang, Young-Sik Shin, Ukcheol Shin, Ayoung Kim  
  _2026-02-23_ · https://arxiv.org/abs/2602.19430v1  
  <details><summary>Abstract</summary>

  Despite the inherent advantages of thermal infrared(TIR) imaging, large-scale data collection and annotation remain a major bottleneck for TIR-based perception. A practical alternative is to synthesize pseudo TIR data via image translation; however, most RGB-to-TIR approaches heavily rely on RGB-centric priors that overlook thermal physics, yielding implausible heat distributions. In this paper, we introduce TherA, a controllable RGB-to-TIR translation framework that produces diverse and thermally plausible images at both scene and object level. TherA couples TherA-VLM with a latent-diffusion-based translator. Given a single RGB image and a user-prompted condition pair, TherA-VLM yields a thermal-aware embedding that encodes scene, object, material, and heat-emission context reflecting the input scene-condition pair. Conditioning the diffusion model on this embedding enables realistic TIR synthesis and fine-grained control across time of day, weather, and object state. Compared to other baselines, TherA achieves state-of-the-art translation performance, demonstrating improved zero-shot translation performance up to 33% increase averaged across all metrics.

  </details>



- **Hepato-LLaVA: An Expert MLLM with Sparse Topo-Pack Attention for Hepatocellular Pathology Analysis on Whole Slide Images**  
  Yuxuan Yang, Zhonghao Yan, Yi Zhang, Bo Yun, Muxi Diao, Guowei Zhao, Kongming Liang, Wenbin Li, Zhanyu Ma  
  _2026-02-23_ · https://arxiv.org/abs/2602.19424v1  
  <details><summary>Abstract</summary>

  Hepatocellular Carcinoma diagnosis relies heavily on the interpretation of gigapixel Whole Slide Images. However, current computational approaches are constrained by fixed-resolution processing mechanisms and inefficient feature aggregation, which inevitably lead to either severe information loss or high feature redundancy. To address these challenges, we propose Hepato-LLaVA, a specialized Multi-modal Large Language Model designed for fine-grained hepatocellular pathology analysis. We introduce a novel Sparse Topo-Pack Attention mechanism that explicitly models 2D tissue topology. This mechanism effectively aggregates local diagnostic evidence into semantic summary tokens while preserving global context. Furthermore, to overcome the lack of multi-scale data, we present HepatoPathoVQA, a clinically grounded dataset comprising 33K hierarchically structured question-answer pairs validated by expert pathologists. Our experiments demonstrate that Hepato-LLaVA achieves state-of-the-art performance on HCC diagnosis and captioning tasks, significantly outperforming existing methods. Our code and implementation details are available at https://pris-cv.github.io/Hepto-LLaVA/.

  </details>



- **Prefer-DAS: Learning from Local Preferences and Sparse Prompts for Domain Adaptive Segmentation of Electron Microscopy**  
  Jiabao Chen, Shan Xiong, Jialin Peng  
  _2026-02-23_ · https://arxiv.org/abs/2602.19423v1  
  <details><summary>Abstract</summary>

  Domain adaptive segmentation (DAS) is a promising paradigm for delineating intracellular structures from various large-scale electron microscopy (EM) without incurring extensive annotated data in each domain. However, the prevalent unsupervised domain adaptation (UDA) strategies often demonstrate limited and biased performance, which hinders their practical applications. In this study, we explore sparse points and local human preferences as weak labels in the target domain, thereby presenting a more realistic yet annotation-efficient setting. Specifically, we develop Prefer-DAS, which pioneers sparse promptable learning and local preference alignment. The Prefer-DAS is a promptable multitask model that integrates self-training and prompt-guided contrastive learning. Unlike SAM-like methods, the Prefer-DAS allows for the use of full, partial, and even no point prompts during both training and inference stages and thus enables interactive segmentation. Instead of using image-level human preference alignment for segmentation, we introduce Local direct Preference Optimization (LPO) and sparse LPO (SLPO), plug-and-play solutions for alignment with spatially varying human feedback or sparse feedback. To address potential missing feedback, we also introduce Unsupervised Preference Optimization (UPO), which leverages self-learned preferences. As a result, the Prefer-DAS model can effectively perform both weakly-supervised and unsupervised DAS, depending on the availability of points and human preferences. Comprehensive experiments on four challenging DAS tasks demonstrate that our model outperforms SAM-like methods as well as unsupervised and weakly-supervised DAS methods in both automatic and interactive segmentation modes, highlighting strong generalizability and flexibility. Additionally, the performance of our model is very close to or even exceeds that of supervised models.

  </details>



- **Referring Layer Decomposition**  
  Fangyi Chen, Yaojie Shen, Lu Xu, Ye Yuan, Shu Zhang, Yulei Niu, Longyin Wen  
  _2026-02-22_ · https://arxiv.org/abs/2602.19358v1  
  <details><summary>Abstract</summary>

  Precise, object-aware control over visual content is essential for advanced image editing and compositional generation. Yet, most existing approaches operate on entire images holistically, limiting the ability to isolate and manipulate individual scene elements. In contrast, layered representations, where scenes are explicitly separated into objects, environmental context, and visual effects, provide a more intuitive and structured framework for interpreting and editing visual content. To bridge this gap and enable both compositional understanding and controllable editing, we introduce the Referring Layer Decomposition (RLD) task, which predicts complete RGBA layers from a single RGB image, conditioned on flexible user prompts, such as spatial inputs (e.g., points, boxes, masks), natural language descriptions, or combinations thereof. At the core is the RefLade, a large-scale dataset comprising 1.11M image-layer-prompt triplets produced by our scalable data engine, along with 100K manually curated, high-fidelity layers. Coupled with a perceptually grounded, human-preference-aligned automatic evaluation protocol, RefLade establishes RLD as a well-defined and benchmarkable research task. Building on this foundation, we present RefLayer, a simple baseline designed for prompt-conditioned layer decomposition, achieving high visual fidelity and semantic alignment. Extensive experiments show our approach enables effective training, reliable evaluation, and high-quality image decomposition, while exhibiting strong zero-shot generalization capabilities.

  </details>



- **MentalBlackboard: Evaluating Spatial Visualization via Mathematical Transformations**  
  Nilay Yilmaz, Maitreya Patel, Naga Sai Abhiram Kusumba, Yixuan He, Yezhou Yang  
  _2026-02-22_ · https://arxiv.org/abs/2602.19357v1  
  <details><summary>Abstract</summary>

  Spatial visualization is the mental ability to imagine, transform, and manipulate the spatial characteristics of objects and actions. This intelligence is a part of human cognition where actions and perception are connected on a mental level. To explore whether state-of-the-art Vision-Language Models (VLMs) exhibit this ability, we develop MentalBlackboard, an open-ended spatial visualization benchmark for Paper Folding and Hole Punching tests within two core tasks: prediction and planning. Our prediction experiments reveal that models struggle with applying symmetrical transformations, even when they predict the sequence of unfolding steps correctly. Also, rotations introduce a significant challenge to the physical situational awareness for models. The planning task reveals limitations of models in analyzing symmetrical relationships and in implementing the multi-stage symmetry process, with Claude Opus 4.1 achieving the highest planning score at an accuracy of 10\%. The top-performing model, o3, attains a peak performance of 71.6\% on the generalization task, which does not require spatial visualization but transfers spatial data; however, it achieves only 25\% accuracy on text-based prediction tasks.

  </details>



- **UP-Fuse: Uncertainty-guided LiDAR-Camera Fusion for 3D Panoptic Segmentation**  
  Rohit Mohan, Florian Drews, Yakov Miron, Daniele Cattaneo, Abhinav Valada  
  _2026-02-22_ · https://arxiv.org/abs/2602.19349v1  
  <details><summary>Abstract</summary>

  LiDAR-camera fusion enhances 3D panoptic segmentation by leveraging camera images to complement sparse LiDAR scans, but it also introduces a critical failure mode. Under adverse conditions, degradation or failure of the camera sensor can significantly compromise the reliability of the perception system. To address this problem, we introduce UP-Fuse, a novel uncertainty-aware fusion framework in the 2D range-view that remains robust under camera sensor degradation, calibration drift, and sensor failure. Raw LiDAR data is first projected into the range-view and encoded by a LiDAR encoder, while camera features are simultaneously extracted and projected into the same shared space. At its core, UP-Fuse employs an uncertainty-guided fusion module that dynamically modulates cross-modal interaction using predicted uncertainty maps. These maps are learned by quantifying representational divergence under diverse visual degradations, ensuring that only reliable visual cues influence the fused representation. The fused range-view features are decoded by a novel hybrid 2D-3D transformer that mitigates spatial ambiguities inherent to the 2D projection and directly predicts 3D panoptic segmentation masks. Extensive experiments on Panoptic nuScenes, SemanticKITTI, and our introduced Panoptic Waymo benchmark demonstrate the efficacy and robustness of UP-Fuse, which maintains strong performance even under severe visual corruption or misalignment, making it well suited for robotic perception in safety-critical settings.

  </details>


