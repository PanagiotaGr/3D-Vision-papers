# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-02-18 07:15 UTC_

Total papers shown: **50**


---

- **Task-Agnostic Continual Learning for Chest Radiograph Classification**  
  Muthu Subash Kavitha, Anas Zafar, Amgad Muneer, Jia Wu  
  _2026-02-17_ · https://arxiv.org/abs/2602.15811v1  
  <details><summary>Abstract</summary>

  Clinical deployment of chest radiograph classifiers requires models that can be updated as new datasets become available without retraining on previously ob- served data or degrading validated performance. We study, for the first time, a task-incremental continual learning setting for chest radiograph classification, in which heterogeneous chest X-ray datasets arrive sequentially and task identifiers are unavailable at inference. We propose a continual adapter-based routing learning strategy for Chest X-rays (CARL-XRay) that maintains a fixed high-capacity backbone and incrementally allocates lightweight task-specific adapters and classifier heads. A latent task selector operates on task-adapted features and leverages both current and historical context preserved through compact prototypes and feature-level experience replay. This design supports stable task identification and adaptation across sequential updates while avoiding raw-image storage. Experiments on large-scale public chest radiograph datasets demonstrate robust performance retention and reliable task-aware inference under continual dataset ingestion. CARL-XRay outperforms joint training under task-unknown deployment, achieving higher routing accuracy (75.0\% vs.\ 62.5\%), while maintaining competitive diagnostic performance with AUROC of 0.74 in the oracle setting with ground-truth task identity and 0.75 under task-unknown inference, using significantly fewer trainable parameters. Finally, the proposed framework provides a practical alternative to joint training and repeated full retraining in continual clinical deployment.

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
  _2026-02-17_ · https://arxiv.org/abs/2602.15656v1  
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



- **StrokeNeXt: A Siamese-encoder Approach for Brain Stroke Classification in Computed Tomography Imagery**  
  Leo Thomas Ramos, Angel D. Sappa  
  _2026-02-16_ · https://arxiv.org/abs/2602.15087v1  
  <details><summary>Abstract</summary>

  We present StrokeNeXt, a model for stroke classification in 2D Computed Tomography (CT) images. StrokeNeXt employs a dual-branch design with two ConvNeXt encoders, whose features are fused through a lightweight convolutional decoder based on stacked 1D operations, including a bottleneck projection and transformation layers, and a compact classification head. The model is evaluated on a curated dataset of 6,774 CT images, addressing both stroke detection and subtype classification between ischemic and hemorrhage cases. StrokeNeXt consistently outperforms convolutional and Transformer-based baselines, reaching accuracies and F1-scores of up to 0.988. Paired statistical tests confirm that the performance gains are statistically significant, while class-wise sensitivity and specificity demonstrate robust behavior across diagnostic categories. Calibration analysis shows reduced prediction error compared to competing methods, and confusion matrix results indicate low misclassification rates. In addition, the model exhibits low inference time and fast convergence.

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


