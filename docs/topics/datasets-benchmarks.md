# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-02-17 07:13 UTC_

Total papers shown: **50**


---

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



- **Kalman Filtering Based Flight Management System Modeling for AAM Aircraft**  
  Balram Kandoria, Aryaman Singh Samyal  
  _2026-02-16_ · https://arxiv.org/abs/2602.14948v1  
  <details><summary>Abstract</summary>

  Advanced Aerial Mobility (AAM) operations require strategic flight planning services that predict both spatial and temporal uncertainties to safely validate flight plans against hazards such as weather cells, restricted airspaces, and CNS disruption areas. Current uncertainty estimation methods for AAM vehicles rely on conservative linear models due to limited real-world performance data. This paper presents a novel Kalman Filter-based uncertainty propagation method that models AAM Flight Management System (FMS) architectures through sigmoid-blended measurement noise covariance. Unlike existing approaches with fixed uncertainty thresholds, our method continuously adapts the filter's measurement trust based on progress toward waypoints, enabling FMS correction behavior to emerge naturally. The approach scales proportionally with control inputs and is tunable to match specific aircraft characteristics or route conditions. We validate the method using real ADS-B data from general aviation aircraft divided into training and verification sets. Uncertainty propagation parameters were tuned on the training set, achieving 76% accuracy in predicting arrival times when compared against the verification dataset, demonstrating the method's effectiveness for strategic flight plan validation in AAM operations.

  </details>



- **Wrivinder: Towards Spatial Intelligence for Geo-locating Ground Images onto Satellite Imagery**  
  Chandrakanth Gudavalli, Tajuddin Manhar Mohammed, Abhay Yadav, Ananth Vishnu Bhaskar, Hardik Prajapati, Cheng Peng, Rama Chellappa, Shivkumar Chandrasekaran, B. S. Manjunath  
  _2026-02-16_ · https://arxiv.org/abs/2602.14929v1  
  <details><summary>Abstract</summary>

  Aligning ground-level imagery with geo-registered satellite maps is crucial for mapping, navigation, and situational awareness, yet remains challenging under large viewpoint gaps or when GPS is unreliable. We introduce Wrivinder, a zero-shot, geometry-driven framework that aggregates multiple ground photographs to reconstruct a consistent 3D scene and align it with overhead satellite imagery. Wrivinder combines SfM reconstruction, 3D Gaussian Splatting, semantic grounding, and monocular depth--based metric cues to produce a stable zenith-view rendering that can be directly matched to satellite context for metrically accurate camera geo-localization. To support systematic evaluation of this task, which lacks suitable benchmarks, we also release MC-Sat, a curated dataset linking multi-view ground imagery with geo-registered satellite tiles across diverse outdoor environments. Together, Wrivinder and MC-Sat provide a first comprehensive baseline and testbed for studying geometry-centered cross-view alignment without paired supervision. In zero-shot experiments, Wrivinder achieves sub-30\,m geolocation accuracy across both dense and large-area scenes, highlighting the promise of geometry-based aggregation for robust ground-to-satellite localization.

  </details>



- **Picking the Right Specialist: Attentive Neural Process-based Selection of Task-Specialized Models as Tools for Agentic Healthcare Systems**  
  Pramit Saha, Joshua Strong, Mohammad Alsharid, Divyanshu Mishra, J. Alison Noble  
  _2026-02-16_ · https://arxiv.org/abs/2602.14901v1  
  <details><summary>Abstract</summary>

  Task-specialized models form the backbone of agentic healthcare systems, enabling the agents to answer clinical queries across tasks such as disease diagnosis, localization, and report generation. Yet, for a given task, a single "best" model rarely exists. In practice, each task is better served by multiple competing specialist models where different models excel on different data samples. As a result, for any given query, agents must reliably select the right specialist model from a heterogeneous pool of tool candidates. To this end, we introduce ToolSelect, which adaptively learns model selection for tools by minimizing a population risk over sampled specialist tool candidates using a consistent surrogate of the task-conditional selection loss. Concretely, we propose an Attentive Neural Process-based selector conditioned on the query and per-model behavioral summaries to choose among the specialist models. Motivated by the absence of any established testbed, we, for the first time, introduce an agentic Chest X-ray environment equipped with a diverse suite of task-specialized models (17 disease detection, 19 report generation, 6 visual grounding, and 13 VQA) and develop ToolSelectBench, a benchmark of 1448 queries. Our results demonstrate that ToolSelect consistently outperforms 10 SOTA methods across four different task families.

  </details>



- **CT-Bench: A Benchmark for Multimodal Lesion Understanding in Computed Tomography**  
  Qingqing Zhu, Qiao Jin, Tejas S. Mathai, Yin Fang, Zhizheng Wang, Yifan Yang, Maame Sarfo-Gyamfi, Benjamin Hou, Ran Gu, Praveen T. S. Balamuralikrishna, et al.  
  _2026-02-16_ · https://arxiv.org/abs/2602.14879v1  
  <details><summary>Abstract</summary>

  Artificial intelligence (AI) can automatically delineate lesions on computed tomography (CT) and generate radiology report content, yet progress is limited by the scarcity of publicly available CT datasets with lesion-level annotations. To bridge this gap, we introduce CT-Bench, a first-of-its-kind benchmark dataset comprising two components: a Lesion Image and Metadata Set containing 20,335 lesions from 7,795 CT studies with bounding boxes, descriptions, and size information, and a multitask visual question answering benchmark with 2,850 QA pairs covering lesion localization, description, size estimation, and attribute categorization. Hard negative examples are included to reflect real-world diagnostic challenges. We evaluate multiple state-of-the-art multimodal models, including vision-language and medical CLIP variants, by comparing their performance to radiologist assessments, demonstrating the value of CT-Bench as a comprehensive benchmark for lesion analysis. Moreover, fine-tuning models on the Lesion Image and Metadata Set yields significant performance gains across both components, underscoring the clinical utility of CT-Bench.

  </details>



- **Debiasing Central Fixation Confounds Reveals a Peripheral "Sweet Spot" for Human-like Scanpaths in Hard-Attention Vision**  
  Pengcheng Pan, Yonekura Shogo, Yasuo Kuniyosh  
  _2026-02-16_ · https://arxiv.org/abs/2602.14834v1  
  <details><summary>Abstract</summary>

  Human eye movements in visual recognition reflect a balance between foveal sampling and peripheral context. Task-driven hard-attention models for vision are often evaluated by how well their scanpaths match human gaze. However, common scanpath metrics can be strongly confounded by dataset-specific center bias, especially on object-centric datasets. Using Gaze-CIFAR-10, we show that a trivial center-fixation baseline achieves surprisingly strong scanpath scores, approaching many learned policies. This makes standard metrics optimistic and blurs the distinction between genuine behavioral alignment and mere central tendency. We then analyze a hard-attention classifier under constrained vision by sweeping foveal patch size and peripheral context, revealing a peripheral sweet spot: only a narrow range of sensory constraints yields scanpaths that are simultaneously (i) above the center baseline after debiasing and (ii) temporally human-like in movement statistics. To address center bias, we propose GCS (Gaze Consistency Score), a center-debiased composite metric augmented with movement similarity. GCS uncovers a robust sweet spot at medium patch size with both foveal and peripheral vision, that is not obvious from raw scanpath metrics or accuracy alone, and also highlights a "shortcut regime" when the field-of-view becomes too large. We discuss implications for evaluating active perception on object-centric datasets and for designing gaze benchmarks that better separate behavioral alignment from center bias.

  </details>



- **Exposing Diversity Bias in Deep Generative Models: Statistical Origins and Correction of Diversity Error**  
  Farzan Farnia, Mohammad Jalali, Azim Ospanov  
  _2026-02-16_ · https://arxiv.org/abs/2602.14682v1  
  <details><summary>Abstract</summary>

  Deep generative models have achieved great success in producing high-quality samples, making them a central tool across machine learning applications. Beyond sample quality, an important yet less systematically studied question is whether trained generative models faithfully capture the diversity of the underlying data distribution. In this work, we address this question by directly comparing the diversity of samples generated by state-of-the-art models with that of test samples drawn from the target data distribution, using recently proposed reference-free entropy-based diversity scores, Vendi and RKE. Across multiple benchmark datasets, we find that test data consistently attains substantially higher Vendi and RKE diversity scores than the generated samples, suggesting a systematic downward diversity bias in modern generative models. To understand the origin of this bias, we analyze the finite-sample behavior of entropy-based diversity scores and show that their expected values increase with sample size, implying that diversity estimated from finite training sets could inherently underestimate the diversity of the true distribution. As a result, optimizing the generators to minimize divergence to empirical data distributions would induce a loss of diversity. Finally, we discuss potential diversity-aware regularization and guidance strategies based on Vendi and RKE as principled directions for mitigating this bias, and provide empirical evidence suggesting their potential to improve the results.

  </details>



- **MeFEm: Medical Face Embedding model**  
  Yury Borets, Stepan Botman  
  _2026-02-16_ · https://arxiv.org/abs/2602.14672v1  
  <details><summary>Abstract</summary>

  We present MeFEm, a vision model based on a modified Joint Embedding Predictive Architecture (JEPA) for biometric and medical analysis from facial images. Key modifications include an axial stripe masking strategy to focus learning on semantically relevant regions, a circular loss weighting scheme, and the probabilistic reassignment of the CLS token for high quality linear probing. Trained on a consolidated dataset of curated images, MeFEm outperforms strong baselines like FaRL and Franca on core anthropometric tasks despite using significantly less data. It also shows promising results on Body Mass Index (BMI) estimation, evaluated on a novel, consolidated closed-source dataset that addresses the domain bias prevalent in existing data. Model weights are available at https://huggingface.co/boretsyury/MeFEm , offering a strong baseline for future work in this domain.

  </details>



- **VIGIL: Tackling Hallucination Detection in Image Recontextualization**  
  Joanna Wojciechowicz, Maria Łubniewska, Jakub Antczak, Justyna Baczyńska, Wojciech Gromski, Wojciech Kozłowski, Maciej Zięba  
  _2026-02-16_ · https://arxiv.org/abs/2602.14633v1  
  <details><summary>Abstract</summary>

  We introduce VIGIL (Visual Inconsistency & Generative In-context Lucidity), the first benchmark dataset and framework providing a fine-grained categorization of hallucinations in the multimodal image recontextualization task for large multimodal models (LMMs). While existing research often treats hallucinations as a uniform issue, our work addresses a significant gap in multimodal evaluation by decomposing these errors into five categories: pasted object hallucinations, background hallucinations, object omission, positional & logical inconsistencies, and physical law violations. To address these complexities, we propose a multi-stage detection pipeline. Our architecture processes recontextualized images through a series of specialized steps targeting object-level fidelity, background consistency, and omission detection, leveraging a coordinated ensemble of open-source models, whose effectiveness is demonstrated through extensive experimental evaluations. Our approach enables a deeper understanding of where the models fail with an explanation; thus, we fill a gap in the field, as no prior methods offer such categorization and decomposition for this task. To promote transparency and further exploration, we openly release VIGIL, along with the detection pipeline and benchmark code, through our GitHub repository: https://github.com/mlubneuskaya/vigil and Data repository: https://huggingface.co/datasets/joannaww/VIGIL.

  </details>



- **OmniVTON++: Training-Free Universal Virtual Try-On with Principal Pose Guidance**  
  Zhaotong Yang, Yong Du, Shengfeng He, Yuhui Li, Xinzhe Li, Yangyang Xu, Junyu Dong, Jian Yang  
  _2026-02-16_ · https://arxiv.org/abs/2602.14552v1  
  <details><summary>Abstract</summary>

  Image-based Virtual Try-On (VTON) concerns the synthesis of realistic person imagery through garment re-rendering under human pose and body constraints. In practice, however, existing approaches are typically optimized for specific data conditions, making their deployment reliant on retraining and limiting their generalization as a unified solution. We present OmniVTON++, a training-free VTON framework designed for universal applicability. It addresses the intertwined challenges of garment alignment, human structural coherence, and boundary continuity by coordinating Structured Garment Morphing for correspondence-driven garment adaptation, Principal Pose Guidance for step-wise structural regulation during diffusion sampling, and Continuous Boundary Stitching for boundary-aware refinement, forming a cohesive pipeline without task-specific retraining. Experimental results demonstrate that OmniVTON++ achieves state-of-the-art performance across diverse generalization settings, including cross-dataset and cross-garment-type evaluations, while reliably operating across scenarios and diffusion backbones within a single formulation. In addition to single-garment, single-human cases, the framework supports multi-garment, multi-human, and anime character virtual try-on, expanding the scope of virtual try-on applications. The source code will be released to the public.

  </details>



- **Architectural Insights for Post-Tornado Damage Recognition**  
  Robinson Umeike, Thang Dao, Shane Crawford, John van de Lindt, Blythe Johnston, Wanting, Wang, Trung Do, Ajibola Mofikoya, Sarbesh Banjara, et al.  
  _2026-02-16_ · https://arxiv.org/abs/2602.14523v1  
  <details><summary>Abstract</summary>

  Rapid and accurate building damage assessment in the immediate aftermath of tornadoes is critical for coordinating life-saving search and rescue operations, optimizing emergency resource allocation, and accelerating community recovery. However, current automated methods struggle with the unique visual complexity of tornado-induced wreckage, primarily due to severe domain shift from standard pre-training datasets and extreme class imbalance in real-world disaster data. To address these challenges, we introduce a systematic experimental framework evaluating 79 open-source deep learning models, encompassing both Convolutional Neural Networks (CNNs) and Vision Transformers, across over 2,300 controlled experiments on our newly curated Quad-State Tornado Damage (QSTD) benchmark dataset. Our findings reveal that achieving operational-grade performance hinges on a complex interaction between architecture and optimization, rather than architectural selection alone. Most strikingly, we demonstrate that optimizer choice can be more consequential than architecture: switching from Adam to SGD provided dramatic F1 gains of +25 to +38 points for Vision Transformer and Swin Transformer families, fundamentally reversing their ranking from bottom-tier to competitive with top-performing CNNs. Furthermore, a low learning rate of 1x10^(-4) proved universally critical, boosting average F1 performance by +10.2 points across all architectures. Our champion model, ConvNeXt-Base trained with these optimized settings, demonstrated strong cross-event generalization on the held-out Tuscaloosa-Moore Tornado Damage (TMTD) dataset, achieving 46.4% Macro F1 (+34.6 points over baseline) and retaining 85.5% Ordinal Top-1 Accuracy despite temporal and sensor domain shifts.

  </details>



- **MedVAR: Towards Scalable and Efficient Medical Image Generation via Next-scale Autoregressive Prediction**  
  Zhicheng He, Yunpeng Zhao, Junde Wu, Ziwei Niu, Zijun Li, Lanfen Lin, Yueming Jin  
  _2026-02-16_ · https://arxiv.org/abs/2602.14512v1  
  <details><summary>Abstract</summary>

  Medical image generation is pivotal in applications like data augmentation for low-resource clinical tasks and privacy-preserving data sharing. However, developing a scalable generative backbone for medical imaging requires architectural efficiency, sufficient multi-organ data, and principled evaluation, yet current approaches leave these aspects unresolved. Therefore, we introduce MedVAR, the first autoregressive-based foundation model that adopts the next-scale prediction paradigm to enable fast and scale-up-friendly medical image synthesis. MedVAR generates images in a coarse-to-fine manner and produces structured multi-scale representations suitable for downstream use. To support hierarchical generation, we curate a harmonized dataset of around 440,000 CT and MRI images spanning six anatomical regions. Comprehensive experiments across fidelity, diversity, and scalability show that MedVAR achieves state-of-the-art generative performance and offers a promising architectural direction for future medical generative foundation models.

  </details>



- **RoboSolver: A Multi-Agent Large Language Model Framework for Solving Robotic Arm Problems**  
  Hamid Khabazi, Ali F. Meghdari, Alireza Taheri  
  _2026-02-16_ · https://arxiv.org/abs/2602.14438v1  
  <details><summary>Abstract</summary>

  This study proposes an intelligent multi-agent framework built on LLMs and VLMs and specifically tailored to robotics. The goal is to integrate the strengths of LLMs and VLMs with computational tools to automatically analyze and solve problems related to robotic manipulators. Our developed framework accepts both textual and visual inputs and can automatically perform forward and inverse kinematics, compute velocities and accelerations of key points, generate 3D simulations of the robot, and ultimately execute motion control within the simulated environment, all according to the user's query. To evaluate the framework, three benchmark tests were designed, each consisting of ten questions. In the first benchmark test, the framework was evaluated while connected to GPT-4o, DeepSeek-V3.2, and Claude-Sonnet-4.5, as well as their corresponding raw models. The objective was to extract the forward kinematics of robots directly from textual descriptions. The results showed that the framework integrated with GPT-4o achieved the highest accuracy, reaching 0.97 in computing the final solution, whereas the raw model alone attained an accuracy of only 0.30 for the same task. Similarly, for the other two models, the framework consistently outperformed the corresponding raw models in terms of accuracy. The second benchmark test was identical to the first, except that the input was provided in visual form. In this test, the GPT-4o LLM was used alongside the Gemini 2.5 Pro VLM. The results showed that the framework achieved an accuracy of 0.93 in obtaining the final answer, which is approximately 20% higher than that of the corresponding raw model. The third benchmark test encompassed a range of robotic tasks, including simulation, control, velocity and acceleration computation, as well as inverse kinematics and Jacobian calculation, for which the framework achieved an accuracy of 0.97.

  </details>



- **A Soft Wrist with Anisotropic and Selectable Stiffness for Robust Robot Learning in Contact-rich Manipulation**  
  Steven Oh, Tomoya Takahashi, Cristian C. Beltran-Hernandez, Yuki Kuroda, Masashi Hamaya  
  _2026-02-16_ · https://arxiv.org/abs/2602.14434v1  
  <details><summary>Abstract</summary>

  Contact-rich manipulation tasks in unstructured environments pose significant robustness challenges for robot learning, where unexpected collisions can cause damage and hinder policy acquisition. Existing soft end-effectors face fundamental limitations: they either provide a limited deformation range, lack directional stiffness control, or require complex actuation systems that compromise practicality. This study introduces CLAW (Compliant Leaf-spring Anisotropic soft Wrist), a novel soft wrist mechanism that addresses these limitations through a simple yet effective design using two orthogonal leaf springs and rotary joints with a locking mechanism. CLAW provides large 6-degree-of-freedom deformation (40mm lateral, 20mm vertical), anisotropic stiffness that is tunable across three distinct modes, while maintaining lightweight construction (330g) at low cost ($550). Experimental evaluations using imitation learning demonstrate that CLAW achieves 76% success rate in benchmark peg-insertion tasks, outperforming both the Fin Ray gripper (43%) and rigid gripper alternatives (36%). CLAW successfully handles diverse contact-rich scenarios, including precision assembly with tight tolerances and delicate object manipulation, demonstrating its potential to enable robust robot learning in contact-rich domains. Project page: https://project-page-manager.github.io/CLAW/

  </details>



- **Learning Proposes, Geometry Disposes: A Modular Framework for Efficient Spatial Reasoning**  
  Haichao Zhu, Zhaorui Yang, Qian Zhang  
  _2026-02-16_ · https://arxiv.org/abs/2602.14409v1  
  <details><summary>Abstract</summary>

  Spatial perception aims to estimate camera motion and scene structure from visual observations, a problem traditionally addressed through geometric modeling and physical consistency constraints. Recent learning-based methods have demonstrated strong representational capacity for geometric perception and are increasingly used to augment classical geometry-centric systems in practice. However, whether learning components should directly replace geometric estimation or instead serve as intermediate modules within such pipelines remains an open question. In this work, we address this gap and investigate an end-to-end modular framework for effective spatial reasoning, where learning proposes geometric hypotheses, while geometric algorithms dispose estimation decisions. In particular, we study this principle in the context of relative camera pose estimation on RGB-D sequences. Using VGGT as a representative learning model, we evaluate learning-based pose and depth proposals under varying motion magnitudes and scene dynamics, followed by a classical point-to-plane RGB-D ICP as the geometric backend. Our experiments on the TUM RGB-D benchmark reveal three consistent findings: (1) learning-based pose proposals alone are unreliable; (2) learning-proposed geometry, when improperly aligned with camera intrinsics, can degrade performance; and (3) when learning-proposed depth is geometrically aligned and followed by a geometric disposal stage, consistent improvements emerge in moderately challenging rigid settings. These results demonstrate that geometry is not merely a refinement component, but an essential arbiter that validates and absorbs learning-based geometric observations. Our study highlights the importance of modular, geometry-aware system design for robust spatial perception.

  </details>



- **Feature Recalibration Based Olfactory-Visual Multimodal Model for Fine-Grained Rice Deterioration Detection**  
  Rongqiang Zhao, Hengrui Hu, Yijing Wang, Mingchun Sun, Jie Liu  
  _2026-02-16_ · https://arxiv.org/abs/2602.14408v1  
  <details><summary>Abstract</summary>

  Multimodal methods are widely used in rice deterioration detection, which exhibit limited capability in representing and extracting fine-grained abnormal features. Moreover, these methods rely on devices, such as hyperspectral cameras and mass spectrometers, increasing detection costs and prolonging data acquisition time. To address these issues, we propose a feature recalibration based olfactory-visual multimodal model for fine-grained rice deterioration detection. The fine-grained deterioration embedding constructor (FDEC) is proposed to reconstruct the labeled multimodal embedded-feature dataset, enhancing sample representation. The fine-grained deterioration recalibration attention network (FDRA-Net) is proposed to emphasize signal variations and increase sensitivity to fine-grained deterioration on the rice surface. Experiments show that the proposed method achieves a classification accuracy of 99.89%. Compared with state-of-the-art methods, the detection accuracy is improved and the procedure is simplified. Furthermore, field detection demonstrates the advantages of accuracy and operational simplicity. The proposed method can also be extended to other agrifood in agriculture and food industry.

  </details>



- **Event-based Visual Deformation Measurement**  
  Yuliang Wu, Wei Zhai, Yuxin Cui, Tiesong Zhao, Yang Cao, Zheng-Jun Zha  
  _2026-02-16_ · https://arxiv.org/abs/2602.14376v1  
  <details><summary>Abstract</summary>

  Visual Deformation Measurement (VDM) aims to recover dense deformation fields by tracking surface motion from camera observations. Traditional image-based methods rely on minimal inter-frame motion to constrain the correspondence search space, which limits their applicability to highly dynamic scenes or necessitates high-speed cameras at the cost of prohibitive storage and computational overhead. We propose an event-frame fusion framework that exploits events for temporally dense motion cues and frames for spatially dense precise estimation. Revisiting the solid elastic modeling prior, we propose an Affine Invariant Simplicial (AIS) framework. It partitions the deformation field into linearized sub-regions with low-parametric representation, effectively mitigating motion ambiguities arising from sparse and noisy events. To speed up parameter searching and reduce error accumulation, a neighborhood-greedy optimization strategy is introduced, enabling well-converged sub-regions to guide their poorly-converged neighbors, effectively suppress local error accumulation in long-term dense tracking. To evaluate the proposed method, a benchmark dataset with temporally aligned event streams and frames is established, encompassing over 120 sequences spanning diverse deformation scenarios. Experimental results show that our method outperforms the state-of-the-art baseline by 1.6% in survival rate. Remarkably, it achieves this using only 18.9% of the data storage and processing resources of high-speed video methods.

  </details>



- **Image-based Joint-level Detection for Inflammation in Rheumatoid Arthritis from Small and Imbalanced Data**  
  Shun Kato, Yasushi Kondo, Shuntaro Saito, Yoshimitsu Aoki, Mariko Isogawa  
  _2026-02-16_ · https://arxiv.org/abs/2602.14365v1  
  <details><summary>Abstract</summary>

  Rheumatoid arthritis (RA) is an autoimmune disease characterized by systemic joint inflammation. Early diagnosis and tight follow-up are essential to the management of RA, as ongoing inflammation can cause irreversible joint damage. The detection of arthritis is important for diagnosis and assessment of disease activity; however, it often takes a long time for patients to receive appropriate specialist care. Therefore, there is a strong need to develop systems that can detect joint inflammation easily using RGB images captured at home. Consequently, we tackle the task of RA inflammation detection from RGB hand images. This task is highly challenging due to general issues in medical imaging, such as the scarcity of positive samples, data imbalance, and the inherent difficulty of the task itself. However, to the best of our knowledge, no existing work has explicitly addressed these challenges in RGB-based RA inflammation detection. This paper quantitatively demonstrates the difficulty of visually detecting inflammation by constructing a dedicated dataset, and we propose a inflammation detection framework with global local encoder that combines self-supervised pretraining on large-scale healthy hand images with imbalance-aware training to detect RA-related joint inflammation from RGB hand images. Our experiments demonstrated that the proposed approach improves F1-score by 0.2 points and Gmean by 0.25 points compared with the baseline model.

  </details>



- **A Generative AI Approach for Reducing Skin Tone Bias in Skin Cancer Classification**  
  Areez Muhammed Shabu, Mohammad Samar Ansari, Asra Aslam  
  _2026-02-16_ · https://arxiv.org/abs/2602.14356v1  
  <details><summary>Abstract</summary>

  Skin cancer is one of the most common cancers worldwide and early detection is critical for effective treatment. However, current AI diagnostic tools are often trained on datasets dominated by lighter skin tones, leading to reduced accuracy and fairness for people with darker skin. The International Skin Imaging Collaboration (ISIC) dataset, one of the most widely used benchmarks, contains over 70% light skin images while dark skins fewer than 8%. This imbalance poses a significant barrier to equitable healthcare delivery and highlights the urgent need for methods that address demographic diversity in medical imaging. This paper addresses this challenge of skin tone imbalance in automated skin cancer detection using dermoscopic images. To overcome this, we present a generative augmentation pipeline that fine-tunes a pre-trained Stable Diffusion model using Low-Rank Adaptation (LoRA) on the image dark-skin subset of the ISIC dataset and generates synthetic dermoscopic images conditioned on lesion type and skin tone. In this study, we investigated the utility of these images on two downstream tasks: lesion segmentation and binary classification. For segmentation, models trained on the augmented dataset and evaluated on held-out real images show consistent improvements in IoU, Dice coefficient, and boundary accuracy. These evalutions provides the verification of Generated dataset. For classification, an EfficientNet-B0 model trained on the augmented dataset achieved 92.14% accuracy. This paper demonstrates that synthetic data augmentation with Generative AI integration can substantially reduce bias with increase fairness in conventional dermatological diagnostics and open challenges for future directions.

  </details>



- **Moving Beyond Sparse Grounding with Complete Screen Parsing Supervision**  
  A. Said Gurbuz, Sunghwan Hong, Ahmed Nassar, Marc Pollefeys, Peter Staar  
  _2026-02-15_ · https://arxiv.org/abs/2602.14276v1  
  <details><summary>Abstract</summary>

  Modern computer-use agents (CUA) must perceive a screen as a structured state, what elements are visible, where they are, and what text they contain, before they can reliably ground instructions and act. Yet, most available grounding datasets provide sparse supervision, with insufficient and low-diversity labels that annotate only a small subset of task-relevant elements per screen, which limits both coverage and generalization; moreover, practical deployment requires efficiency to enable low-latency, on-device use. We introduce ScreenParse, a large-scale dataset for complete screen parsing, with dense annotations of all visible UI elements (boxes, 55-class types, and text) across 771K web screenshots (21M elements). ScreenParse is generated by Webshot, an automated, scalable pipeline that renders diverse urls, extracts annotations and applies VLM-based relabeling and quality filtering. Using ScreenParse, we train ScreenVLM, a compact, 316M-parameter vision language model (VLM) that decodes a compact ScreenTag markup representation with a structure-aware loss that upweights structure-critical tokens. ScreenVLM substantially outperforms much larger foundation VLMs on dense parsing (e.g., 0.592 vs. 0.294 PageIoU on ScreenParse) and shows strong transfer to public benchmarks. Moreover, finetuning foundation VLMs on ScreenParse consistently improves their grounding performance, suggesting that dense screen supervision provides transferable structural priors for UI understanding. Project page: https://saidgurbuz.github.io/screenparse/.

  </details>



- **AbracADDbra: Touch-Guided Object Addition by Decoupling Placement and Editing Subtasks**  
  Kunal Swami, Raghu Chittersu, Yuvraj Rathore, Rajeev Irny, Shashavali Doodekula, Alok Shukla  
  _2026-02-15_ · https://arxiv.org/abs/2602.14237v1  
  <details><summary>Abstract</summary>

  Instruction-based object addition is often hindered by the ambiguity of text-only prompts or the tedious nature of mask-based inputs. To address this usability gap, we introduce AbracADDbra, a user-friendly framework that leverages intuitive touch priors to spatially ground succinct instructions for precise placement. Our efficient, decoupled architecture uses a vision-language transformer for touch-guided placement, followed by a diffusion model that jointly generates the object and an instance mask for high-fidelity blending. To facilitate standardized evaluation, we contribute the Touch2Add benchmark for this interactive task. Our extensive evaluations, where our placement model significantly outperforms both random placement and general-purpose VLM baselines, confirm the framework's ability to produce high-fidelity edits. Furthermore, our analysis reveals a strong correlation between initial placement accuracy and final edit quality, validating our decoupled approach. This work thus paves the way for more accessible and efficient creative tools.

  </details>



- **Learning Significant Persistent Homology Features for 3D Shape Understanding**  
  Prachi Kudeshia, Jiju Poovvancheri  
  _2026-02-15_ · https://arxiv.org/abs/2602.14228v1  
  <details><summary>Abstract</summary>

  Geometry and topology constitute complementary descriptors of three-dimensional shape, yet existing benchmark datasets primarily capture geometric information while neglecting topological structure. This work addresses this limitation by introducing topologically-enriched versions of ModelNet40 and ShapeNet, where each point cloud is augmented with its corresponding persistent homology features. These benchmarks with the topological signatures establish a foundation for unified geometry-topology learning and enable systematic evaluation of topology-aware deep learning architectures for 3D shape analysis. Building on this foundation, we propose a deep learning-based significant persistent point selection method, \textit{TopoGAT}, that learns to identify the most informative topological features directly from input data and the corresponding topological signatures, circumventing the limitations of hand-crafted statistical selection criteria. A comparative study verifies the superiority of the proposed method over traditional statistical approaches in terms of stability and discriminative power. Integrating the selected significant persistent points into standard point cloud classification and part-segmentation pipelines yields improvements in both classification accuracy and segmentation metrics. The presented topologically-enriched datasets, coupled with our learnable significant feature selection approach, enable the broader integration of persistent homology into the practical deep learning workflows for 3D point cloud analysis.

  </details>



- **Freq-DP Net: A Dual-Branch Network for Fence Removal using Dual-Pixel and Fourier Priors**  
  Kunal Swami, Sudha Velusamy, Chandra Sekhar Seelamantula  
  _2026-02-15_ · https://arxiv.org/abs/2602.14226v1  
  <details><summary>Abstract</summary>

  Removing fence occlusions from single images is a challenging task that degrades visual quality and limits downstream computer vision applications. Existing methods often fail on static scenes or require motion cues from multiple frames. To overcome these limitations, we introduce the first framework to leverage dual-pixel (DP) sensors for this problem. We propose Freq-DP Net, a novel dual-branch network that fuses two complementary priors: a geometric prior from defocus disparity, modeled using an explicit cost volume, and a structural prior of the fence's global pattern, learned via Fast Fourier Convolution (FFC). An attention mechanism intelligently merges these cues for highly accurate fence segmentation. To validate our approach, we build and release a diverse benchmark with different fence varieties. Experiments demonstrate that our method significantly outperforms strong general-purpose baselines, establishing a new state-of-the-art for single-image, DP-based fence removal.

  </details>



- **HiVid: LLM-Guided Video Saliency For Content-Aware VOD And Live Streaming**  
  Jiahui Chen, Bo Peng, Lianchen Jia, Zeyu Zhang, Tianchi Huang, Lifeng Sun  
  _2026-02-15_ · https://arxiv.org/abs/2602.14214v1  
  <details><summary>Abstract</summary>

  Content-aware streaming requires dynamic, chunk-level importance weights to optimize subjective quality of experience (QoE). However, direct human annotation is prohibitively expensive while vision-saliency models generalize poorly. We introduce HiVid, the first framework to leverage Large Language Models (LLMs) as a scalable human proxy to generate high-fidelity weights for both Video-on-Demand (VOD) and live streaming. We address 3 non-trivial challenges: (1) To extend LLMs' limited modality and circumvent token limits, we propose a perception module to assess frames in a local context window, autoregressively building a coherent understanding of the video. (2) For VOD with rating inconsistency across local windows, we propose a ranking module to perform global re-ranking with a novel LLM-guided merge-sort algorithm. (3) For live streaming which requires low-latency, online inference without future knowledge, we propose a prediction module to predict future weights with a multi-modal time series model, which comprises a content-aware attention and adaptive horizon to accommodate asynchronous LLM inference. Extensive experiments show HiVid improves weight prediction accuracy by up to 11.5\% for VOD and 26\% for live streaming over SOTA baselines. Real-world user study validates HiVid boosts streaming QoE correlation by 14.7\%.

  </details>



- **GeoEyes: On-Demand Visual Focusing for Evidence-Grounded Understanding of Ultra-High-Resolution Remote Sensing Imagery**  
  Fengxiang Wang, Mingshuo Chen, Yueying Li, Yajie Yang, Yifan Zhang, Long Lan, Xue Yang, Hongda Sun, Yulin Wang, Di Wang, et al.  
  _2026-02-15_ · https://arxiv.org/abs/2602.14201v1  
  <details><summary>Abstract</summary>

  The "thinking-with-images" paradigm enables multimodal large language models (MLLMs) to actively explore visual scenes via zoom-in tools. This is essential for ultra-high-resolution (UHR) remote sensing VQA, where task-relevant cues are sparse and tiny. However, we observe a consistent failure mode in existing zoom-enabled MLLMs: Tool Usage Homogenization, where tool calls collapse into task-agnostic patterns, limiting effective evidence acquisition. To address this, we propose GeoEyes, a staged training framework consisting of (1) a cold-start SFT dataset, UHR Chain-of-Zoom (UHR-CoZ), which covers diverse zooming regimes, and (2) an agentic reinforcement learning method, AdaZoom-GRPO, that explicitly rewards evidence gain and answer improvement during zoom interactions. The resulting model learns on-demand zooming with proper stopping behavior and achieves substantial improvements on UHR remote sensing benchmarks, with 54.23% accuracy on XLRS-Bench.

  </details>



- **Learning Part-Aware Dense 3D Feature Field for Generalizable Articulated Object Manipulation**  
  Yue Chen, Muqing Jiang, Kaifeng Zheng, Jiaqi Liang, Chenrui Tie, Haoran Lu, Ruihai Wu, Hao Dong  
  _2026-02-15_ · https://arxiv.org/abs/2602.14193v1  
  <details><summary>Abstract</summary>

  Articulated object manipulation is essential for various real-world robotic tasks, yet generalizing across diverse objects remains a major challenge. A key to generalization lies in understanding functional parts (e.g., door handles and knobs), which indicate where and how to manipulate across diverse object categories and shapes. Previous works attempted to achieve generalization by introducing foundation features, while these features are mostly 2D-based and do not specifically consider functional parts. When lifting these 2D features to geometry-profound 3D space, challenges arise, such as long runtimes, multi-view inconsistencies, and low spatial resolution with insufficient geometric information. To address these issues, we propose Part-Aware 3D Feature Field (PA3FF), a novel dense 3D feature with part awareness for generalizable articulated object manipulation. PA3FF is trained by 3D part proposals from a large-scale labeled dataset, via a contrastive learning formulation. Given point clouds as input, PA3FF predicts a continuous 3D feature field in a feedforward manner, where the distance between point features reflects the proximity of functional parts: points with similar features are more likely to belong to the same part. Building on this feature, we introduce the Part-Aware Diffusion Policy (PADP), an imitation learning framework aimed at enhancing sample efficiency and generalization for robotic manipulation. We evaluate PADP on several simulated and real-world tasks, demonstrating that PA3FF consistently outperforms a range of 2D and 3D representations in manipulation scenarios, including CLIP, DINOv2, and Grounded-SAM. Beyond imitation learning, PA3FF enables diverse downstream methods, including correspondence learning and segmentation tasks, making it a versatile foundation for robotic manipulation. Project page: https://pa3ff.github.io

  </details>



- **Detection of On-Ground Chestnuts Using Artificial Intelligence Toward Automated Picking**  
  Kaixuan Fang, Yuzhen Lu, Xinyang Mu  
  _2026-02-15_ · https://arxiv.org/abs/2602.14140v1  
  <details><summary>Abstract</summary>

  Traditional mechanized chestnut harvesting is too costly for small producers, non-selective, and prone to damaging nuts. Accurate, reliable detection of chestnuts on the orchard floor is crucial for developing low-cost, vision-guided automated harvesting technology. However, developing a reliable chestnut detection system faces challenges in complex environments with shading, varying natural light conditions, and interference from weeds, fallen leaves, stones, and other foreign on-ground objects, which have remained unaddressed. This study collected 319 images of chestnuts on the orchard floor, containing 6524 annotated chestnuts. A comprehensive set of 29 state-of-the-art real-time object detectors, including 14 in the YOLO (v11-13) and 15 in the RT-DETR (v1-v4) families at varied model scales, was systematically evaluated through replicated modeling experiments for chestnut detection. Experimental results show that the YOLOv12m model achieves the best mAP@0.5 of 95.1% among all the evaluated models, while the RT-DETRv2-R101 was the most accurate variant among RT-DETR models, with mAP@0.5 of 91.1%. In terms of mAP@[0.5:0.95], the YOLOv11x model achieved the best accuracy of 80.1%. All models demonstrate significant potential for real-time chestnut detection, and YOLO models outperformed RT-DETR models in terms of both detection accuracy and inference, making them better suited for on-board deployment. Both the dataset and software programs in this study have been made publicly available at https://github.com/AgFood-Sensing-and-Intelligence-Lab/ChestnutDetection.

  </details>



- **EgoSound: Benchmarking Sound Understanding in Egocentric Videos**  
  Bingwen Zhu, Yuqian Fu, Qiaole Dong, Guolei Sun, Tianwen Qian, Yuzheng Wu, Danda Pani Paudel, Xiangyang Xue, Yanwei Fu  
  _2026-02-15_ · https://arxiv.org/abs/2602.14122v1  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLMs) have recently achieved remarkable progress in vision-language understanding. Yet, human perception is inherently multisensory, integrating sight, sound, and motion to reason about the world. Among these modalities, sound provides indispensable cues about spatial layout, off-screen events, and causal interactions, particularly in egocentric settings where auditory and visual signals are tightly coupled. To this end, we introduce EgoSound, the first benchmark designed to systematically evaluate egocentric sound understanding in MLLMs. EgoSound unifies data from Ego4D and EgoBlind, encompassing both sighted and sound-dependent experiences. It defines a seven-task taxonomy spanning intrinsic sound perception, spatial localization, causal inference, and cross-modal reasoning. Constructed through a multi-stage auto-generative pipeline, EgoSound contains 7315 validated QA pairs across 900 videos. Comprehensive experiments on nine state-of-the-art MLLMs reveal that current models exhibit emerging auditory reasoning abilities but remain limited in fine-grained spatial and causal understanding. EgoSound establishes a challenging foundation for advancing multisensory egocentric intelligence, bridging the gap between seeing and truly hearing the world.

  </details>



- **Bidirectional Temporal Dynamics Modeling for EEG-based Driving Fatigue Recognition**  
  YipTin Po, Jianming Wang, Yutao Miao, Jiayan Zhang, Yunxu Zhao, Xiaomin Ouyang, Zhihong Li, Nevin L. Zhang  
  _2026-02-15_ · https://arxiv.org/abs/2602.14071v1  
  <details><summary>Abstract</summary>

  Driving fatigue is a major contributor to traffic accidents and poses a serious threat to road safety. Electroencephalography (EEG) provides a direct measurement of neural activity, yet EEG-based fatigue recognition is hindered by strong non-stationarity and asymmetric neural dynamics. To address these challenges, we propose DeltaGateNet, a novel framework that explicitly captures Bidirectional temporal dynamics for EEG-based driving fatigue recognition. Our key idea is to introduce a Bidirectional Delta module that decomposes first-order temporal differences into positive and negative components, enabling explicit modeling of asymmetric neural activation and suppression patterns. Furthermore, we design a Gated Temporal Convolution module to capture long-term temporal dependencies for each EEG channel using depthwise temporal convolutions and residual learning, preserving channel-wise specificity while enhancing temporal representation robustness. Extensive experiments conducted under both intra-subject and inter-subject evaluation settings on the public SEED-VIG and SADT driving fatigue datasets demonstrate that DeltaGateNet consistently outperforms existing methods. On SEED-VIG, DeltaGateNet achieves an intra-subject accuracy of 81.89% and an inter-subject accuracy of 55.55%. On the balanced SADT 2022 dataset, it attains intra-subject and inter-subject accuracies of 96.81% and 83.21%, respectively, while on the unbalanced SADT 2952 dataset, it achieves 96.84% intra-subject and 84.49% inter-subject accuracy. These results indicate that explicitly modeling Bidirectional temporal dynamics yields robust and generalizable performance under varying subject and class-distribution conditions.

  </details>



- **Restoration Adaptation for Semantic Segmentation on Low Quality Images**  
  Kai Guan, Rongyuan Wu, Shuai Li, Wentao Zhu, Wenjun Zeng, Lei Zhang  
  _2026-02-15_ · https://arxiv.org/abs/2602.14042v1  
  <details><summary>Abstract</summary>

  In real-world scenarios, the performance of semantic segmentation often deteriorates when processing low-quality (LQ) images, which may lack clear semantic structures and high-frequency details. Although image restoration techniques offer a promising direction for enhancing degraded visual content, conventional real-world image restoration (Real-IR) models primarily focus on pixel-level fidelity and often fail to recover task-relevant semantic cues, limiting their effectiveness when directly applied to downstream vision tasks. Conversely, existing segmentation models trained on high-quality data lack robustness under real-world degradations. In this paper, we propose Restoration Adaptation for Semantic Segmentation (RASS), which effectively integrates semantic image restoration into the segmentation process, enabling high-quality semantic segmentation on the LQ images directly. Specifically, we first propose a Semantic-Constrained Restoration (SCR) model, which injects segmentation priors into the restoration model by aligning its cross-attention maps with segmentation masks, encouraging semantically faithful image reconstruction. Then, RASS transfers semantic restoration knowledge into segmentation through LoRA-based module merging and task-specific fine-tuning, thereby enhancing the model's robustness to LQ images. To validate the effectiveness of our framework, we construct a real-world LQ image segmentation dataset with high-quality annotations, and conduct extensive experiments on both synthetic and real-world LQ benchmarks. The results show that SCR and RASS significantly outperform state-of-the-art methods in segmentation and restoration tasks. Code, models, and datasets will be available at https://github.com/Ka1Guan/RASS.git.

  </details>



- **RoboAug: One Annotation to Hundreds of Scenes via Region-Contrastive Data Augmentation for Robotic Manipulation**  
  Xinhua Wang, Kun Wu, Zhen Zhao, Hu Cao, Yinuo Zhao, Zhiyuan Xu, Meng Li, Shichao Fan, Di Wu, Yixue Zhang, et al.  
  _2026-02-15_ · https://arxiv.org/abs/2602.14032v1  
  <details><summary>Abstract</summary>

  Enhancing the generalization capability of robotic learning to enable robots to operate effectively in diverse, unseen scenes is a fundamental and challenging problem. Existing approaches often depend on pretraining with large-scale data collection, which is labor-intensive and time-consuming, or on semantic data augmentation techniques that necessitate an impractical assumption of flawless upstream object detection in real-world scenarios. In this work, we propose RoboAug, a novel generative data augmentation framework that significantly minimizes the reliance on large-scale pretraining and the perfect visual recognition assumption by requiring only the bounding box annotation of a single image during training. Leveraging this minimal information, RoboAug employs pre-trained generative models for precise semantic data augmentation and integrates a plug-and-play region-contrastive loss to help models focus on task-relevant regions, thereby improving generalization and boosting task success rates. We conduct extensive real-world experiments on three robots, namely UR-5e, AgileX, and Tien Kung 2.0, spanning over 35k rollouts. Empirical results demonstrate that RoboAug significantly outperforms state-of-the-art data augmentation baselines. Specifically, when evaluating generalization capabilities in unseen scenes featuring diverse combinations of backgrounds, distractors, and lighting conditions, our method achieves substantial gains over the baseline without augmentation. The success rates increase from 0.09 to 0.47 on UR-5e, from 0.16 to 0.60 on AgileX, and from 0.19 to 0.67 on Tien Kung 2.0. These results highlight the superior generalization and effectiveness of RoboAug in real-world manipulation tasks. Our project is available at https://x-roboaug.github.io/.

  </details>



- **A Deployment-Friendly Foundational Framework for Efficient Computational Pathology**  
  Yu Cai, Cheng Jin, Jiabo Ma, Fengtao Zhou, Yingxue Xu, Zhengrui Guo, Yihui Wang, Zhengyu Zhang, Ling Liang, Yonghao Tan, et al.  
  _2026-02-15_ · https://arxiv.org/abs/2602.14010v1  
  <details><summary>Abstract</summary>

  Pathology foundation models (PFMs) have enabled robust generalization in computational pathology through large-scale datasets and expansive architectures, but their substantial computational cost, particularly for gigapixel whole slide images, limits clinical accessibility and scalability. Here, we present LitePath, a deployment-friendly foundational framework designed to mitigate model over-parameterization and patch level redundancy. LitePath integrates LiteFM, a compact model distilled from three large PFMs (Virchow2, H-Optimus-1 and UNI2) using 190 million patches, and the Adaptive Patch Selector (APS), a lightweight component for task-specific patch selection. The framework reduces model parameters by 28x and lowers FLOPs by 403.5x relative to Virchow2, enabling deployment on low-power edge hardware such as the NVIDIA Jetson Orin Nano Super. On this device, LitePath processes 208 slides per hour, 104.5x faster than Virchow2, and consumes 0.36 kWh per 3,000 slides, 171x lower than Virchow2 on an RTX3090 GPU. We validated accuracy using 37 cohorts across four organs and 26 tasks (26 internal, 9 external, and 2 prospective), comprising 15,672 slides from 9,808 patients disjoint from the pretraining data. LitePath ranks second among 19 evaluated models and outperforms larger models including H-Optimus-1, mSTAR, UNI2 and GPFM, while retaining 99.71% of the AUC of Virchow2 on average. To quantify the balance between accuracy and efficiency, we propose the Deployability Score (D-Score), defined as the weighted geometric mean of normalized AUC and normalized FLOP, where LitePath achieves the highest value, surpassing Virchow2 by 10.64%. These results demonstrate that LitePath enables rapid, cost-effective and energy-efficient pathology image analysis on accessible hardware while maintaining accuracy comparable to state-of-the-art PFMs and reducing the carbon footprint of AI deployment.

  </details>



- **MarsRetrieval: Benchmarking Vision-Language Models for Planetary-Scale Geospatial Retrieval on Mars**  
  Shuoyuan Wang, Yiran Wang, Hongxin Wei  
  _2026-02-15_ · https://arxiv.org/abs/2602.13961v1  
  <details><summary>Abstract</summary>

  Data-driven approaches like deep learning are rapidly advancing planetary science, particularly in Mars exploration. Despite recent progress, most existing benchmarks remain confined to closed-set supervised visual tasks and do not support text-guided retrieval for geospatial discovery. We introduce MarsRetrieval, a retrieval benchmark for evaluating vision-language models for Martian geospatial discovery. MarsRetrieval includes three tasks: (1) paired image-text retrieval, (2) landform retrieval, and (3) global geo-localization, covering multiple spatial scales and diverse geomorphic origins. We propose a unified retrieval-centric protocol to benchmark multimodal embedding architectures, including contrastive dual-tower encoders and generative vision-language models. Our evaluation shows MarsRetrieval is challenging: even strong foundation models often fail to capture domain-specific geomorphic distinctions. We further show that domain-specific fine-tuning is critical for generalizable geospatial discovery in planetary settings. Our code is available at https://github.com/ml-stat-Sustech/MarsRetrieval

  </details>



- **Fusing Pixels and Genes: Spatially-Aware Learning in Computational Pathology**  
  Minghao Han, Dingkang Yang, Linhao Qu, Zizhi Chen, Gang Li, Han Wang, Jiacong Wang, Lihua Zhang  
  _2026-02-15_ · https://arxiv.org/abs/2602.13944v1  
  <details><summary>Abstract</summary>

  Recent years have witnessed remarkable progress in multimodal learning within computational pathology. Existing models primarily rely on vision and language modalities; however, language alone lacks molecular specificity and offers limited pathological supervision, leading to representational bottlenecks. In this paper, we propose STAMP, a Spatial Transcriptomics-Augmented Multimodal Pathology representation learning framework that integrates spatially-resolved gene expression profiles to enable molecule-guided joint embedding of pathology images and transcriptomic data. Our study shows that self-supervised, gene-guided training provides a robust and task-agnostic signal for learning pathology image representations. Incorporating spatial context and multi-scale information further enhances model performance and generalizability. To support this, we constructed SpaVis-6M, the largest Visium-based spatial transcriptomics dataset to date, and trained a spatially-aware gene encoder on this resource. Leveraging hierarchical multi-scale contrastive alignment and cross-scale patch localization mechanisms, STAMP effectively aligns spatial transcriptomics with pathology images, capturing spatial structure and molecular variation. We validate STAMP across six datasets and four downstream tasks, where it consistently achieves strong performance. These results highlight the value and necessity of integrating spatially resolved molecular supervision for advancing multimodal learning in computational pathology. The code is included in the supplementary materials. The pretrained weights and SpaVis-6M are available at: https://github.com/Hanminghao/STAMP.

  </details>



- **RPGD: RANSAC-P3P Gradient Descent for Extrinsic Calibration in 3D Human Pose Estimation**  
  Zhanyu Tuo  
  _2026-02-14_ · https://arxiv.org/abs/2602.13901v1  
  <details><summary>Abstract</summary>

  In this paper, we propose RPGD (RANSAC-P3P Gradient Descent), a human-pose-driven extrinsic calibration framework that robustly aligns MoCap-based 3D skeletal data with monocular or multi-view RGB cameras using only natural human motion. RPGD formulates extrinsic calibration as a coarse-to-fine problem tailored to human poses, combining the global robustness of RANSAC-P3P with Gradient-Descent-based refinement. We evaluate RPGD on three large-scale public 3D HPE datasets as well as on a self-collected in-the-wild dataset. Experimental results demonstrate that RPGD consistently recovers extrinsic parameters with accuracy comparable to the provided ground truth, achieving sub-pixel MPJPE reprojection error even in challenging, noisy settings. These results indicate that RPGD provides a practical and automatic solution for reliable extrinsic calibration of large-scale 3D HPE dataset collection.

  </details>



- **UAV-SEAD: State Estimation Anomaly Dataset for UAVs**  
  Aykut Kabaoglu, Sanem Sariel  
  _2026-02-14_ · https://arxiv.org/abs/2602.13900v1  
  <details><summary>Abstract</summary>

  Accurate state estimation in Unmanned Aerial Vehicles (UAVs) is crucial for ensuring reliable and safe operation, as anomalies occurring during mission execution may induce discrepancies between expected and observed system behaviors, thereby compromising mission success or posing potential safety hazards. It is essential to continuously monitor and detect such conditions in order to ensure a timely response and maintain system reliability. In this work, we focus on UAV state estimation anomalies and provide a large-scale real-world UAV dataset to facilitate research aimed at improving the development of anomaly detection. Unlike existing datasets that primarily rely on injected faults into simulated data, this dataset comprises 1396 real flight logs totaling over 52 hours of flight time, collected across diverse indoor and outdoor environments using a collection of PX4-based UAVs equipped with a variety of sensor configurations. The dataset comprises both normal and anomalous flights without synthetic manipulation, making it uniquely suitable for realistic anomaly detection tasks. A structured classification is proposed that categorizes UAV state estimation anomalies into four classes: mechanical and electrical, external position, global position, and altitude anomalies. These classifications reflect collective, contextual, and outlier anomalies observed in multivariate sensor data streams, including IMU, GPS, barometer, magnetometer, distance sensors, visual odometry, and optical flow, that can be found in the PX4 logging mechanism. It is anticipated that this dataset will play a key role in the development, training, and evaluation of anomaly detection and isolation systems to address the critical gap in UAV reliability research.

  </details>



- **Parameter-Efficient Fine-Tuning of DINOv2 for Large-Scale Font Classification**  
  Daniel Chen, Zaria Zinn, Marcus Lowe  
  _2026-02-14_ · https://arxiv.org/abs/2602.13889v1  
  <details><summary>Abstract</summary>

  We present a font classification system capable of identifying 394 font families from rendered text images. Our approach fine-tunes a DINOv2 Vision Transformer using Low-Rank Adaptation (LoRA), achieving approximately 86% top-1 accuracy while training fewer than 1% of the model's 87.2M parameters. We introduce a synthetic dataset generation pipeline that renders Google Fonts at scale with diverse augmentations including randomized colors, alignment, line wrapping, and Gaussian noise, producing training images that generalize to real-world typographic samples. The model incorporates built-in preprocessing to ensure consistency between training and inference, and is deployed as a HuggingFace Inference Endpoint. We release the model, dataset, and full training pipeline as open-source resources.

  </details>



- **Low-Pass Filtering Improves Behavioral Alignment of Vision Models**  
  Max Wolff, Thomas Klein, Evgenia Rusak, Felix Wichmann, Wieland Brendel  
  _2026-02-14_ · https://arxiv.org/abs/2602.13859v1  
  <details><summary>Abstract</summary>

  Despite their impressive performance on computer vision benchmarks, Deep Neural Networks (DNNs) still fall short of adequately modeling human visual behavior, as measured by error consistency and shape bias. Recent work hypothesized that behavioral alignment can be drastically improved through \emph{generative} -- rather than \emph{discriminative} -- classifiers, with far-reaching implications for models of human vision. Here, we instead show that the increased alignment of generative models can be largely explained by a seemingly innocuous resizing operation in the generative model which effectively acts as a low-pass filter. In a series of controlled experiments, we show that removing high-frequency spatial information from discriminative models like CLIP drastically increases their behavioral alignment. Simply blurring images at test-time -- rather than training on blurred images -- achieves a new state-of-the-art score on the model-vs-human benchmark, halving the current alignment gap between DNNs and human observers. Furthermore, low-pass filters are likely optimal, which we demonstrate by directly optimizing filters for alignment. To contextualize the performance of optimal filters, we compute the frontier of all possible pareto-optimal solutions to the benchmark, which was formerly unknown. We explain our findings by observing that the frequency spectrum of optimal Gaussian filters roughly matches the spectrum of band-pass filters implemented by the human visual system. We show that the contrast sensitivity function, describing the inverse of the contrast threshold required for humans to detect a sinusoidal grating as a function of spatiotemporal frequency, is approximated well by Gaussian filters of the specific width that also maximizes error consistency.

  </details>



- **Humanoid Hanoi: Investigating Shared Whole-Body Control for Skill-Based Box Rearrangement**  
  Minku Kim, Kuan-Chia Chen, Aayam Shrestha, Li Fuxin, Stefan Lee, Alan Fern  
  _2026-02-14_ · https://arxiv.org/abs/2602.13850v1  
  <details><summary>Abstract</summary>

  We investigate a skill-based framework for humanoid box rearrangement that enables long-horizon execution by sequencing reusable skills at the task level. In our architecture, all skills execute through a shared, task-agnostic whole-body controller (WBC), providing a consistent closed-loop interface for skill composition, in contrast to non-shared designs that use separate low-level controllers per skill. We find that naively reusing the same pretrained WBC can reduce robustness over long horizons, as new skills and their compositions induce shifted state and command distributions. We address this with a simple data aggregation procedure that augments shared-WBC training with rollouts from closed-loop skill execution under domain randomization. To evaluate the approach, we introduce \emph{Humanoid Hanoi}, a long-horizon Tower-of-Hanoi box rearrangement benchmark, and report results in simulation and on the Digit V3 humanoid robot, demonstrating fully autonomous rearrangement over extended horizons and quantifying the benefits of the shared-WBC approach over non-shared baselines.

  </details>



- **Cardiac Output Prediction from Echocardiograms: Self-Supervised Learning with Limited Data**  
  Adson Duarte, Davide Vitturini, Emanuele Milillo, Andrea Bragagnolo, Carlo Alberto Barbano, Riccardo Renzulli, Michele Cannito, Federico Giacobbe, Francesco Bruno, Ovidio de Filippo, et al.  
  _2026-02-14_ · https://arxiv.org/abs/2602.13846v1  
  <details><summary>Abstract</summary>

  Cardiac Output (CO) is a key parameter in the diagnosis and management of cardiovascular diseases. However, its accurate measurement requires right-heart catheterization, an invasive and time-consuming procedure, motivating the development of reliable non-invasive alternatives using echocardiography. In this work, we propose a self-supervised learning (SSL) pretraining strategy based on SimCLR to improve CO prediction from apical four-chamber echocardiographic videos. The pretraining is performed using the same limited dataset available for the downstream task, demonstrating the potential of SSL even under data scarcity. Our results show that SSL mitigates overfitting and improves representation learning, achieving an average Pearson correlation of 0.41 on the test set and outperforming PanEcho, a model trained on over one million echocardiographic exams. Source code is available at https://github.com/EIDOSLAB/cardiac-output.

  </details>



- **Synthetic Dataset Generation and Validation for Robotic Surgery Instrument Segmentation**  
  Giorgio Chiesa, Rossella Borra, Vittorio Lauro, Sabrina De Cillis, Daniele Amparore, Cristian Fiori, Riccardo Renzulli, Marco Grangetto  
  _2026-02-14_ · https://arxiv.org/abs/2602.13844v1  
  <details><summary>Abstract</summary>

  This paper presents a comprehensive workflow for generating and validating a synthetic dataset designed for robotic surgery instrument segmentation. A 3D reconstruction of the Da Vinci robotic arms was refined and animated in Autodesk Maya through a fully automated Python-based pipeline capable of producing photorealistic, labeled video sequences. Each scene integrates randomized motion patterns, lighting variations, and synthetic blood textures to mimic intraoperative variability while preserving pixel-accurate ground truth masks. To validate the realism and effectiveness of the generated data, several segmentation models were trained under controlled ratios of real and synthetic data. Results demonstrate that a balanced composition of real and synthetic samples significantly improves model generalization compared to training on real data only, while excessive reliance on synthetic data introduces a measurable domain shift. The proposed framework provides a reproducible and scalable tool for surgical computer vision, supporting future research in data augmentation, domain adaptation, and simulation-based pretraining for robotic-assisted surgery. Data and code are available at https://github.com/EIDOSLAB/Sintetic-dataset-DaVinci.

  </details>



- **Automated Prediction of Paravalvular Regurgitation before Transcatheter Aortic Valve Implantation**  
  Michele Cannito, Riccardo Renzulli, Adson Duarte, Farzad Nikfam, Carlo Alberto Barbano, Enrico Chiesa, Francesco Bruno, Federico Giacobbe, Wojciech Wanha, Arturo Giordano, et al.  
  _2026-02-14_ · https://arxiv.org/abs/2602.13842v1  
  <details><summary>Abstract</summary>

  Severe aortic stenosis is a common and life-threatening condition in elderly patients, often treated with Transcatheter Aortic Valve Implantation (TAVI). Despite procedural advances, paravalvular aortic regurgitation (PVR) remains one of the most frequent post-TAVI complications, with a proven impact on long-term prognosis. In this work, we investigate the potential of deep learning to predict the occurrence of PVR from preoperative cardiac CT. To this end, a dataset of preoperative TAVI patients was collected, and 3D convolutional neural networks were trained on isotropic CT volumes. The results achieved suggest that volumetric deep learning can capture subtle anatomical features from pre-TAVI imaging, opening new perspectives for personalized risk assessment and procedural optimization. Source code is available at https://github.com/EIDOSLAB/tavi.

  </details>



- **Gaussian Sequences with Multi-Scale Dynamics for 4D Reconstruction from Monocular Casual Videos**  
  Can Li, Jie Gu, Jingmin Chen, Fangzhou Qiu, Lei Sun  
  _2026-02-14_ · https://arxiv.org/abs/2602.13806v1  
  <details><summary>Abstract</summary>

  Understanding dynamic scenes from casual videos is critical for scalable robot learning, yet four-dimensional (4D) reconstruction under strictly monocular settings remains highly ill-posed. To address this challenge, our key insight is that real-world dynamics exhibits a multi-scale regularity from object to particle level. To this end, we design the multi-scale dynamics mechanism that factorizes complex motion fields. Within this formulation, we propose Gaussian sequences with multi-scale dynamics, a novel representation for dynamic 3D Gaussians derived through compositions of multi-level motion. This layered structure substantially alleviates ambiguity of reconstruction and promotes physically plausible dynamics. We further incorporate multi-modal priors from vision foundation models to establish complementary supervision, constraining the solution space and improving the reconstruction fidelity. Our approach enables accurate and globally consistent 4D reconstruction from monocular casual videos. Experiments of dynamic novel-view synthesis (NVS) on benchmark and real-world manipulation datasets demonstrate considerable improvements over existing methods.

  </details>



- **Foundation Model-Driven Semantic Change Detection in Remote Sensing Imagery**  
  Hengtong Shen, Li Yan, Hong Xie, Yaxuan Wei, Xinhao Li, Wenfei Shen, Peixian Lv, Fei Tan  
  _2026-02-14_ · https://arxiv.org/abs/2602.13780v1  
  <details><summary>Abstract</summary>

  Remote sensing (RS) change detection methods can extract critical information on surface dynamics and are an essential means for humans to understand changes in the earth's surface and environment. Among these methods, semantic change detection (SCD) can more effectively interpret the multi-class information contained in bi-temporal RS imagery, providing semantic-level predictions that support dynamic change monitoring. However, due to the limited semantic understanding capability of the model and the inherent complexity of the SCD tasks, existing SCD methods face significant challenges in both performance and paradigm complexity. In this paper, we propose PerASCD, a SCD method driven by RS foundation model PerA, designed to enhance the multi-scale semantic understanding and overall performance. We introduce a modular Cascaded Gated Decoder (CG-Decoder) that simplifies complex SCD decoding pipelines while promoting effective multi-level feature interaction and fusion. In addition, we propose a Soft Semantic Consistency Loss (SSCLoss) to mitigate the numerical instability commonly encountered during SCD training. We further explore the applicability of multiple existing RS foundation models on the SCD task when equipped with the proposed decoder. Experimental results demonstrate that our decoder not only effectively simplifies the paradigm of SCD, but also achieves seamless adaptation across various vision encoders. Our method achieves state-of-the-art (SOTA) performance on two public benchmark datasets, validating its effectiveness. The code is available at https://github.com/SathShen/PerASCD.git.

  </details>



- **OmniScience: A Large-scale Multi-modal Dataset for Scientific Image Understanding**  
  Haoyi Tao, Chaozheng Huang, Nan Wang, Han Lyu, Linfeng Zhang, Guolin Ke, Xi Fang  
  _2026-02-14_ · https://arxiv.org/abs/2602.13758v1  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models demonstrate strong performance on natural image understanding, yet exhibit limited capability in interpreting scientific images, including but not limited to schematic diagrams, experimental characterizations, and analytical charts. This limitation is particularly pronounced in open-source MLLMs. The gap largely stems from existing datasets with limited domain coverage, coarse structural annotations, and weak semantic grounding. We introduce OmniScience, a large-scale, high-fidelity multi-modal dataset comprising 1.5 million figure-caption-context triplets, spanning more than 10 major scientific disciplines. To obtain image caption data with higher information density and accuracy for multi-modal large-model training, we develop a dynamic model-routing re-captioning pipeline that leverages state-of-the-art multi-modal large language models to generate dense, self-contained descriptions by jointly synthesizing visual features, original figure captions, and corresponding in-text references authored by human scientists. The pipeline is further reinforced with rigorous quality filtering and alignment with human expert judgments, ensuring both factual accuracy and semantic completeness, and boosts the image-text multi-modal similarity score from 0.769 to 0.956. We further propose a caption QA protocol as a proxy task for evaluating visual understanding. Under this setting, Qwen2.5-VL-3B model finetuned on OmniScience show substantial gains over baselines, achieving a gain of 0.378 on MM-MT-Bench and a gain of 0.140 on MMMU.

  </details>



- **T2MBench: A Benchmark for Out-of-Distribution Text-to-Motion Generation**  
  Bin Yang, Rong Ou, Weisheng Xu, Jiaqi Xiong, Xintao Li, Taowen Wang, Luyu Zhu, Xu Jiang, Jing Tan, Renjing Xu  
  _2026-02-14_ · https://arxiv.org/abs/2602.13751v1  
  <details><summary>Abstract</summary>

  Most existing evaluations of text-to-motion generation focus on in-distribution textual inputs and a limited set of evaluation criteria, which restricts their ability to systematically assess model generalization and motion generation capabilities under complex out-of-distribution (OOD) textual conditions. To address this limitation, we propose a benchmark specifically designed for OOD text-to-motion evaluation, which includes a comprehensive analysis of 14 representative baseline models and the two datasets derived from evaluation results. Specifically, we construct an OOD prompt dataset consisting of 1,025 textual descriptions. Based on this prompt dataset, we introduce a unified evaluation framework that integrates LLM-based Evaluation, Multi-factor Motion evaluation, and Fine-grained Accuracy Evaluation. Our experimental results reveal that while different baseline models demonstrate strengths in areas such as text-to-motion semantic alignment, motion generalizability, and physical quality, most models struggle to achieve strong performance with Fine-grained Accuracy Evaluation. These findings highlight the limitations of existing methods in OOD scenarios and offer practical guidance for the design and evaluation of future production-level text-to-motion models.

  </details>



- **RMPL: Relation-aware Multi-task Progressive Learning with Stage-wise Training for Multimedia Event Extraction**  
  Yongkang Jin, Jianwen Luo, Jingjing Wang, Jianmin Yao, Yu Hong  
  _2026-02-14_ · https://arxiv.org/abs/2602.13748v1  
  <details><summary>Abstract</summary>

  Multimedia Event Extraction (MEE) aims to identify events and their arguments from documents that contain both text and images. It requires grounding event semantics across different modalities. Progress in MEE is limited by the lack of annotated training data. M2E2 is the only established benchmark, but it provides annotations only for evaluation. This makes direct supervised training impractical. Existing methods mainly rely on cross-modal alignment or inference-time prompting with Vision--Language Models (VLMs). These approaches do not explicitly learn structured event representations and often produce weak argument grounding in multimodal settings. To address these limitations, we propose RMPL, a Relation-aware Multi-task Progressive Learning framework for MEE under low-resource conditions. RMPL incorporates heterogeneous supervision from unimodal event extraction and multimedia relation extraction with stage-wise training. The model is first trained with a unified schema to learn shared event-centric representations across modalities. It is then fine-tuned for event mention identification and argument role extraction using mixed textual and visual data. Experiments on the M2E2 benchmark with multiple VLMs show consistent improvements across different modality settings.

  </details>



- **Generative Latent Representations of 3D Brain MRI for Multi-Task Downstream Analysis in Down Syndrome**  
  Jordi Malé, Juan Fortea, Mateus Rozalem-Aranha, Neus Martínez-Abadías, Xavier Sevillano  
  _2026-02-14_ · https://arxiv.org/abs/2602.13731v1  
  <details><summary>Abstract</summary>

  Generative models have emerged as powerful tools in medical imaging, enabling tasks such as segmentation, anomaly detection, and high-quality synthetic data generation. These models typically rely on learning meaningful latent representations, which are particularly valuable given the high-dimensional nature of 3D medical images like brain magnetic resonance imaging (MRI) scans. Despite their potential, latent representations remain underexplored in terms of their structure, information content, and applicability to downstream clinical tasks. Investigating these representations is crucial for advancing the use of generative models in neuroimaging research and clinical decision-making. In this work, we develop multiple variational autoencoders (VAEs) to encode 3D brain MRI scans into compact latent space representations for generative and predictive applications. We systematically evaluate the effectiveness of the learned representations through three key analyses: (i) a quantitative and qualitative assessment of MRI reconstruction quality, (ii) a visualisation of the latent space structure using Principal Component Analysis, and (iii) downstream classification tasks on a proprietary dataset of euploid and Down syndrome individuals brain MRI scans. Our results demonstrate that the VAE successfully captures essential brain features while maintaining high reconstruction fidelity. The latent space exhibits clear clustering patterns, particularly in distinguishing individuals with Down syndrome from euploid controls.

  </details>



- **A WDLoRA-Based Multimodal Generative Framework for Clinically Guided Corneal Confocal Microscopy Image Synthesis in Diabetic Neuropathy**  
  Xin Zhang, Liangxiu Han, Yue Shi, Yalin Zheng, Uazman Alam, Maryam Ferdousi, Rayaz Malik  
  _2026-02-14_ · https://arxiv.org/abs/2602.13693v1  
  <details><summary>Abstract</summary>

  Corneal Confocal Microscopy (CCM) is a sensitive tool for assessing small-fiber damage in Diabetic Peripheral Neuropathy (DPN), yet the development of robust, automated deep learning-based diagnostic models is limited by scarce labelled data and fine-grained variability in corneal nerve morphology. Although Artificial Intelligence (AI)-driven foundation generative models excel at natural image synthesis, they often struggle in medical imaging due to limited domain-specific training, compromising the anatomical fidelity required for clinical analysis. To overcome these limitations, we propose a Weight-Decomposed Low-Rank Adaptation (WDLoRA)-based multimodal generative framework for clinically guided CCM image synthesis. WDLoRA is a parameter-efficient fine-tuning (PEFT) mechanism that decouples magnitude and directional weight updates, enabling foundation generative models to independently learn the orientation (nerve topology) and intensity (stromal contrast) required for medical realism. By jointly conditioning on nerve segmentation masks and disease-specific clinical prompts, the model synthesises anatomically coherent images across the DPN spectrum (Control, T1NoDPN, T1DPN). A comprehensive three-pillar evaluation demonstrates that the proposed framework achieves state-of-the-art visual fidelity (Fréchet Inception Distance (FID): 5.18) and structural integrity (Structural Similarity Index Measure (SSIM): 0.630), significantly outperforming GAN and standard diffusion baselines. Crucially, the synthetic images preserve gold-standard clinical biomarkers and are statistically equivalent to real patient data. When used to train automated diagnostic models, the synthetic dataset improves downstream diagnostic accuracy by 2.1% and segmentation performance by 2.2%, validating the framework's potential to alleviate data bottlenecks in medical AI.

  </details>


