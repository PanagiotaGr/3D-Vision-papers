# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-04-01 07:51 UTC_

Total papers shown: **50**


---

- **Benchmarking PhD-Level Coding in 3D Geometric Computer Vision**  
  Wenyi Li, Renkai Luo, Yue Yu, Huan-ang Gao, Mingju Gao, Li Yuan, Chaoyou Fu, Hao Zhao  
  _2026-03-31_ · https://arxiv.org/abs/2603.30038v1  
  <details><summary>Abstract</summary>

  AI-assisted coding has rapidly reshaped software practice and research workflows, yet today's models still struggle to produce correct code for complex 3D geometric vision. If models could reliably write such code, the research of our community would change substantially. To measure progress toward that goal, we introduce GeoCodeBench, a PhD-level benchmark that evaluates coding for 3D vision. Each problem is a fill-in-the-function implementation task curated from representative papers at recent venues: we first let a tool propose candidate functions from official repositories, then perform careful human screening to select core 3D geometric components. For every target, we generate diverse, edge-case unit tests, enabling fully automatic, reproducible scoring. We evaluate eight representative open- and closed-source models to reflect the current ecosystem. The best model, GPT-5, attains only 36.6% pass rate, revealing a large gap between current capabilities and dependable 3D scientific coding. GeoCodeBench organizes tasks into a two-level hierarchy: General 3D capability (geometric transformations and mechanics/optics formulation) and Research capability (novel algorithm implementation and geometric logic routing). Scores are positively correlated across these axes, but research-oriented tasks are markedly harder. Context ablations further show that "more paper text" is not always better: cutting off at the Method section statistically outperforms full-paper inputs, highlighting unresolved challenges in long-context scientific comprehension. Together, these findings position GeoCodeBench as a rigorous testbed for advancing from generic coding to trustworthy 3D geometric vision coding.

  </details>



- **Learning Structural-Functional Brain Representations through Multi-Scale Adaptive Graph Attention for Cognitive Insight**  
  Badhan Mazumder, Sir-Lord Wiafe, Aline Kotoski, Vince D. Calhoun, Dong Hye Ye  
  _2026-03-31_ · https://arxiv.org/abs/2603.29967v1  
  <details><summary>Abstract</summary>

  Understanding how brain structure and function interact is key to explaining intelligence yet modeling them jointly is challenging as the structural and functional connectome capture complementary aspects of organization. We introduced Multi-scale Adaptive Graph Network (MAGNet), a Transformer-style graph neural network framework that adaptively learns structure-function interactions. MAGNet leverages source-based morphometry from structural MRI to extract inter-regional morphological features and fuses them with functional network connectivity from resting-state fMRI. A hybrid graph integrates direct and indirect pathways, while local-global attention refines connectivity importance and a joint loss simultaneously enforces cross-modal coherence and optimizes the prediction objective end-to-end. On the ABCD dataset, MAGNet outperformed relevant baselines, demonstrating effective multimodal integration for advancing our understanding of cognitive function.

  </details>



- **Scaling Video Pretraining for Surgical Foundation Models**  
  Sicheng Lu, Zikai Xiao, Jianhui Wei, Danyu Sun, Qi Lu, Keli Hu, Yang Feng, Jian Wu, Zongxin Yang, Zuozhu Liu  
  _2026-03-31_ · https://arxiv.org/abs/2603.29966v1  
  <details><summary>Abstract</summary>

  Surgical video understanding is essential for computer-assisted interventions, yet existing surgical foundation models remain constrained by limited data scale, procedural diversity, and inconsistent evaluation, often lacking a reproducible training pipeline. We propose SurgRec, a scalable and reproducible pretraining recipe for surgical video understanding, instantiated with two variants: SurgRec-MAE and SurgRec-JEPA. We curate a large multi-source corpus of 10,535 videos and 214.5M frames spanning endoscopy, laparoscopy, cataract, and robotic surgery. Building on this corpus, we develop a unified pretraining pipeline with balanced sampling and standardize a reproducible benchmark across 16 downstream datasets and four clinical domains with consistent data splits. Across extensive comparisons against SSL baselines and vision-language models, SurgRec consistently achieves superior performance across downstream datasets. In contrast, VLMs prove unreliable for fine-grained temporal recognition, exhibiting both performance gaps and sensitivity to prompt phrasing. Our work provides a reproducible, scalable foundation for the community to build more general surgical video models. All code, models, and data will be publicly released.

  </details>



- **SurgTEMP: Temporal-Aware Surgical Video Question Answering with Text-guided Visual Memory for Laparoscopic Cholecystectomy**  
  Shi Li, Vinkle Srivastav, Nicolas Chanel, Saurav Sharma, Nabani Banik, Lorenzo Arboit, Kun Yuan, Pietro Mascagni, Nicolas Padoy  
  _2026-03-31_ · https://arxiv.org/abs/2603.29962v1  
  <details><summary>Abstract</summary>

  Surgical procedures are inherently complex and risky, requiring extensive expertise and constant focus to well navigate evolving intraoperative scenes. Computer-assisted systems such as surgical visual question answering (VQA) offer promises for education and intraoperative support. Current surgical VQA research largely focuses on static frame analysis, overlooking rich temporal semantics. Surgical video question answering is further challenged by low visual contrast, its highly knowledge-driven nature, diverse analytical needs spanning scattered temporal windows, and the hierarchy from basic perception to high-level intraoperative assessment. To address these challenges, we propose SurgTEMP, a multimodal LLM framework featuring (i) a query-guided token selection module that builds hierarchical visual memory (spatial and temporal memory banks) and (ii) a Surgical Competency Progression (SCP) training scheme. Together, these components enable effective modeling of variable-length surgical videos while preserving procedure-relevant cues and temporal coherence, and better support diverse downstream assessment tasks. To support model development, we introduce CholeVidQA-32K, a surgical video question answering dataset comprising 32K open-ended QA pairs and 3,855 video segments (approximately 128 h total) from laparoscopic cholecystectomy. The dataset is organized into a three-level hierarchy -- Perception, Assessment, and Reasoning -- spanning 11 tasks from instrument/action/anatomy perception to Critical View of Safety (CVS), intraoperative difficulty, skill proficiency, and adverse event assessment. In comprehensive evaluations against state-of-the-art open-source multimodal and video LLMs (fine-tuned and zero-shot), SurgTEMP achieves substantial performance improvements, advancing the state of video-based surgical VQA.

  </details>



- **EC-Bench: Enumeration and Counting Benchmark for Ultra-Long Videos**  
  Fumihiko Tsuchiya, Taiki Miyanishi, Mahiro Ukai, Nakamasa Inoue, Shuhei Kurita, Yusuke Iwasawa, Yutaka Matsuo  
  _2026-03-31_ · https://arxiv.org/abs/2603.29943v1  
  <details><summary>Abstract</summary>

  Counting in long videos remains a fundamental yet underexplored challenge in computer vision. Real-world recordings often span tens of minutes or longer and contain sparse, diverse events, making long-range temporal reasoning particularly difficult. However, most existing video counting benchmarks focus on short clips and evaluate only the final numerical answer, providing little insight into what should be counted or whether models consistently identify relevant instances across time. We introduce EC-Bench, a benchmark that jointly evaluates enumeration, counting, and temporal evidence grounding in long-form videos. EC-Bench contains 152 videos longer than 30 minutes and 1,699 queries paired with explicit evidence spans. Across 22 multimodal large language models (MLLMs), the best model achieves only 29.98% accuracy on Enumeration and 23.74% on Counting, while human performance reaches 78.57% and 82.97%, respectively. Our analysis reveals strong relationships between enumeration accuracy, temporal grounding, and counting performance. These results highlight fundamental limitations of current MLLMs and establish EC-Bench as a challenging benchmark for long-form quantitative video reasoning.

  </details>



- **Better than Average: Spatially-Aware Aggregation of Segmentation Uncertainty Improves Downstream Performance**  
  Vanessa Emanuela Guarino, Claudia Winklmayr, Jannik Franzen, Josef Lorenz Rumberger, Manuel Pfeuffer, Sonja Greven, Klaus Maier-Hein, Carsten T. Lüth, Christoph Karg, Dagmar Kainmueller  
  _2026-03-31_ · https://arxiv.org/abs/2603.29941v1  
  <details><summary>Abstract</summary>

  Uncertainty Quantification (UQ) is crucial for ensuring the reliability of automated image segmentations in safety-critical domains like biomedical image analysis or autonomous driving. In segmentation, UQ generates pixel-wise uncertainty scores that must be aggregated into image-level scores for downstream tasks like Out-of-Distribution (OoD) or failure detection. Despite routine use of aggregation strategies, their properties and impact on downstream task performance have not yet been comprehensively studied. Global Average is the default choice, yet it does not account for spatial and structural features of segmentation uncertainty. Alternatives like patch-, class- and threshold-based strategies exist, but lack systematic comparison, leading to inconsistent reporting and unclear best practices. We address this gap by (1) formally analyzing properties, limitations, and pitfalls of common strategies; (2) proposing novel strategies that incorporate spatial uncertainty structure and (3) benchmarking their performance on OoD and failure detection across ten datasets that vary in image geometry and structure. We find that aggregators leveraging spatial structure yield stronger performance in both downstream tasks studied. However, the performance of individual aggregators depends heavily on dataset characteristics, so we (4) propose a meta-aggregator that integrates multiple aggregators and performs robustly across datasets.

  </details>



- **End-to-End Image Compression with Segmentation Guided Dual Coding for Wind Turbines**  
  Raül Pérez-Gonzalo, Andreas Espersen, Søren Forchhammer, Antonio Agudo  
  _2026-03-31_ · https://arxiv.org/abs/2603.29927v1  
  <details><summary>Abstract</summary>

  Transferring large volumes of high-resolution images during wind turbine inspections introduces a bottleneck in assessing and detecting severe defects. Efficient coding must preserve high fidelity in blade regions while aggressively compressing the background. In this work, we propose an end-to-end deep learning framework that jointly performs segmentation and dual-mode (lossy and lossless) compression. The segmentation module accurately identifies the blade region, after which our region-of-interest (ROI) compressor encodes it at superior quality compared to the rest of the image. Unlike conventional ROI schemes that merely allocate more bits to salient areas, our framework integrates: (i) a robust segmentation network (BU-Netv2+P) with a CRF-regularized loss for precise blade localization, (ii) a hyperprior-based autoencoder optimized for lossy compression, and (iii) an extended bits-back coder with hierarchical models for fully lossless blade reconstruction. Furthermore, our ROI framework removes the sequential dependency in bits-back coding by reusing background-coded bits, enabling parallelized and efficient dual-mode compression. To the best of our knowledge, this is the first fully integrated learning-based ROI codec combining segmentation, lossy, and lossless compression, ensuring that subsequent defect detection is not compromised. Experiments on a large-scale wind turbine dataset demonstrate superior compression performance and efficiency, offering a practical solution for automated inspections.

  </details>



- **Training deep learning based dynamic MR image reconstruction using synthetic fractals**  
  Anirudh Raman, Olivier Jaubert, Mark Wrobel, Tina Yao, Ruaraidh Campbell, Rebecca Baker, Ruta Virsinskaite, Daniel Knight, Michael Quail, Jennifer Steeden, et al.  
  _2026-03-31_ · https://arxiv.org/abs/2603.29922v1  
  <details><summary>Abstract</summary>

  Purpose: To investigate whether synthetically generated fractal data can be used to train deep learning (DL) models for dynamic MRI reconstruction, thereby avoiding the privacy, licensing, and availability limitations associated with cardiac MR training datasets. Methods: A training dataset was generated using quaternion Julia fractals to produce 2D+time images. Multi-coil MRI acquisition was simulated to generate paired fully sampled and radially undersampled k-space data. A 3D UNet deep artefact suppression model was trained using these fractal data (F-DL) and compared with an identical model trained on cardiac MRI data (CMR-DL). Both models were evaluated on prospectively acquired radial real-time cardiac MRI from 10 patients. Reconstructions were compared against compressed sensing(CS) and low-rank deep image prior (LR-DIP). All reconstrctuions were ranked for image quality, while ventricular volumes and ejection fraction were compared with reference breath-hold cine MRI. Results: There was no significant difference in qualitative ranking between F-DL and CMR-DL (p=0.9), while both outperformed CS and LR-DIP (p<0.001). Ventricular volumes and function derived from F-DL were similar to CMR-DL, showing no significant bias and accptable limits of agreement compared to reference cine imaging. However, LR-DIP had a signifcant bias (p=0.016) and wider lmits of agreement. Conclusion: DL models trained using synthetic fractal data can reconstruct real-time cardiac MRI with image quality and clinical measurements comparable to models trained on true cardiac MRI data. Fractal training data provide an open, scalable alternative to clinical datasets and may enable development of more generalisable DL reconstruction models for dynamic MRI.

  </details>



- **Less Is More? Selective Visual Attention to High-Importance Regions for Multimodal Radiology Summarization**  
  Mst. Fahmida Sultana Naznin, Adnan Ibney Faruq, Mushfiqur Rahman, Niloy Kumar Mondal, Md. Mehedi Hasan Shawon, Md Rakibul Hasan  
  _2026-03-31_ · https://arxiv.org/abs/2603.29901v1  
  <details><summary>Abstract</summary>

  Automated radiology report summarization aims to distill verbose findings into concise clinical impressions, but existing multimodal models often struggle with visual noise and fail to meaningfully improve over strong text-only baselines in the FINDINGS $\to$ IMPRESSION transformation. We challenge two prevailing assumptions: (1) that more visual input is always better, and (2) that multimodal models add limited value when findings already contain rich image-derived detail. Through controlled ablations on MIMIC-CXR benchmark, we show that selectively focusing on pathology-relevant visual patches rather than full images yields substantially better performance. We introduce ViTAS, Visual-Text Attention Summarizer, a multi-stage pipeline that combines ensemble-guided MedSAM2 lung segmentation, bidirectional cross-attention for multi-view fusion, Shapley-guided adaptive patch clustering, and hierarchical visual tokenization feeding a ViT. ViTAS achieves SOTA results with 29.25% BLEU-4 and 69.83% ROUGE-L, improved factual alignment in qualitative analysis, and the highest expert-rated human evaluation scores. Our findings demonstrate that less but more relevant visual input is not only sufficient but superior for multimodal radiology summarization.

  </details>



- **DIAL: Decoupling Intent and Action via Latent World Modeling for End-to-End VLA**  
  Yi Chen, Yuying Ge, Hui Zhou, Mingyu Ding, Yixiao Ge, Xihui Liu  
  _2026-03-31_ · https://arxiv.org/abs/2603.29844v1  
  <details><summary>Abstract</summary>

  The development of Vision-Language-Action (VLA) models has been significantly accelerated by pre-trained Vision-Language Models (VLMs). However, most existing end-to-end VLAs treat the VLM primarily as a multimodal encoder, directly mapping vision-language features to low-level actions. This paradigm underutilizes the VLM's potential in high-level decision making and introduces training instability, frequently degrading its rich semantic representations. To address these limitations, we introduce DIAL, a framework bridging high-level decision making and low-level motor execution through a differentiable latent intent bottleneck. Specifically, a VLM-based System-2 performs latent world modeling by synthesizing latent visual foresight within the VLM's native feature space; this foresight explicitly encodes intent and serves as the structural bottleneck. A lightweight System-1 policy then decodes this predicted intent together with the current observation into precise robot actions via latent inverse dynamics. To ensure optimization stability, we employ a two-stage training paradigm: a decoupled warmup phase where System-2 learns to predict latent futures while System-1 learns motor control under ground-truth future guidance within a unified feature space, followed by seamless end-to-end joint optimization. This enables action-aware gradients to refine the VLM backbone in a controlled manner, preserving pre-trained knowledge. Extensive experiments on the RoboCasa GR1 Tabletop benchmark show that DIAL establishes a new state-of-the-art, achieving superior performance with 10x fewer demonstrations than prior methods. Furthermore, by leveraging heterogeneous human demonstrations, DIAL learns physically grounded manipulation priors and exhibits robust zero-shot generalization to unseen objects and novel configurations during real-world deployment on a humanoid robot.

  </details>



- **Toward Generalizable Whole Brain Representations with High-Resolution Light-Sheet Data**  
  Minyoung E. Kim, Dae Hee Yun, Aditi V. Patel, Madeline Hon, Webster Guan, Taegeon Lee, Brian Nguyen  
  _2026-03-31_ · https://arxiv.org/abs/2603.29842v1  
  <details><summary>Abstract</summary>

  Unprecedented visual details of biological structures are being revealed by subcellular-resolution whole-brain 3D microscopy data, enabled by recent advances in intact tissue processing and light-sheet fluorescence microscopy (LSFM). These volumetric data offer rich morphological and spatial cellular information, however, the lack of scalable data processing and analysis methods tailored to these petabyte-scale data poses a substantial challenge for accurate interpretation. Further, existing models for visual tasks such as object detection and classification struggle to generalize to this type of data. To accelerate the development of suitable methods and foundational models, we present CANVAS, a comprehensive set of high-resolution whole mouse brain LSFM benchmark data, encompassing six neuronal and immune cell-type markers, along with cell annotations and a leaderboard. We also demonstrate challenges in generalization of baseline models built on existing architectures, especially due to the heterogeneity in cellular morphology across phenotypes and anatomical locations in the brain. To the best of our knowledge, CANVAS is the first and largest LSFM benchmark that captures intact mouse brain tissue at subcellular level, and includes extensive annotations of cells throughout the brain.

  </details>



- **AutoFormBench: Benchmark Dataset for Automating Form Understanding**  
  Gaurab Baral, Junxiu Zhou  
  _2026-03-31_ · https://arxiv.org/abs/2603.29832v1  
  <details><summary>Abstract</summary>

  Automated processing of structured documents such as government forms, healthcare records, and enterprise invoices remains a persistent challenge due to the high degree of layout variability encountered in real-world settings. This paper introduces AutoFormBench, a benchmark dataset of 407 annotated real-world forms spanning government, healthcare, and enterprise domains, designed to train and evaluate form element detection models. We present a systematic comparison of classical OpenCV approaches and four YOLO architectures (YOLOv8, YOLOv11, YOLOv26-s, and YOLOv26-l) for localizing and classifying fillable form elements. specifically checkboxes, input lines, and text boxes across diverse PDF document types. YOLOv11 demonstrates consistently superior performance in both F1 score and Jaccard accuracy across all element classes and tolerance levels.

  </details>



- **Multi-Feature Fusion Approach for Generative AI Images Detection**  
  Abderrezzaq Sendjasni, Mohamed-Chaker Larabi  
  _2026-03-31_ · https://arxiv.org/abs/2603.29788v1  
  <details><summary>Abstract</summary>

  The rapid evolution of Generative AI (GenAI) models has led to synthetic images of unprecedented realism, challenging traditional methods for distinguishing them from natural photographs. While existing detectors often rely on single-feature spaces, such as statistical regularities, semantic embeddings, or texture patterns, these approaches tend to lack robustness when confronted with diverse and evolving generative models. In this work, we investigate and systematically evaluate a multi-feature fusion framework that combines complementary cues from three distinct spaces: (1) Mean Subtracted Contrast Normalized (MSCN) features capturing low-level statistical deviations; (2) CLIP embeddings encoding high-level semantic coherence; and (3) Multi-scale Local Binary Patterns (MLBP) characterizing mid-level texture anomalies. Through extensive experiments on four benchmark datasets covering a wide range of generative models, we show that individual feature spaces exhibit significant performance variability across different generators. Crucially, the fusion of all three representations yields superior and more consistent performance, particularly in a challenging mixed-model scenario. Compared to state-of-the-art methods, the proposed framework yields consistently improved performance across all evaluated datasets. Overall, this work highlights the importance of hybrid representations for robust GenAI image detection and provides a principled framework for integrating complementary visual cues.

  </details>



- **TSHA: A Benchmark for Visual Language Models in Trustworthy Safety Hazard Assessment Scenarios**  
  Qiucheng Yu, Ruijie Xu, Mingang Chen, Xuequan Lu, Jianfeng Dong, Chaochao Lu, Xin Tan  
  _2026-03-31_ · https://arxiv.org/abs/2603.29759v1  
  <details><summary>Abstract</summary>

  Recent advances in vision-language models (VLMs) have accelerated their application to indoor safety hazards assessment. However, existing benchmarks suffer from three fundamental limitations: (1) heavy reliance on synthetic datasets constructed via simulation software, creating a significant domain gap with real-world environments; (2) oversimplified safety tasks with artificial constraints on hazard and scene types, thereby limiting model generalization; and (3) absence of rigorous evaluation protocols to thoroughly assess model capabilities in complex home safety scenarios. To address these challenges, we introduce TSHA (\textbf{T}rustworthy \textbf{S}afety \textbf{H}azards \textbf{A}ssessment), a comprehensive benchmark comprising 81,809 carefully curated training samples drawn from four complementary sources: existing indoor datasets, internet images, AIGC images, and newly captured images. This benchmark set also includes a highly challenging test set with 1707 samples, comprising not only a carefully selected subset from the training distribution but also newly added videos and panoramic images containing multiple safety hazards, used to evaluate the model's robustness in complex safety scenarios. Extensive experiments on 23 popular VLMs demonstrate that current VLMs lack robust capabilities for safety hazard assessment. Importantly, models trained on the TSHA training set not only achieve a significant performance improvement of up to +18.3 points on the TSHA test set but also exhibit enhanced generalizability across other benchmarks, underscoring the substantial contribution and importance of the TSHA benchmark.

  </details>



- **GRVS: a Generalizable and Recurrent Approach to Monocular Dynamic View Synthesis**  
  Thomas Tanay, Mohammed Brahimi, Michal Nazarczuk, Qingwen Zhang, Sibi Catley-Chandar, Arthur Moreau, Zhensong Zhang, Eduardo Pérez-Pellitero  
  _2026-03-31_ · https://arxiv.org/abs/2603.29734v1  
  <details><summary>Abstract</summary>

  Synthesizing novel views from monocular videos of dynamic scenes remains a challenging problem. Scene-specific methods that optimize 4D representations with explicit motion priors often break down in highly dynamic regions where multi-view information is hard to exploit. Diffusion-based approaches that integrate camera control into large pre-trained models can produce visually plausible videos but frequently suffer from geometric inconsistencies across both static and dynamic areas. Both families of methods also require substantial computational resources. Building on the success of generalizable models for static novel view synthesis, we adapt the framework to dynamic inputs and propose a new model with two key components: (1) a recurrent loop that enables unbounded and asynchronous mapping between input and target videos and (2) an efficient use of plane sweeps over dynamic inputs to disentangle camera and scene motion, and achieve fine-grained, six-degrees-of-freedom camera controls. We train and evaluate our model on the UCSD dataset and on Kubric-4D-dyn, a new monocular dynamic dataset featuring longer, higher resolution sequences with more complex scene dynamics than existing alternatives. Our model outperforms four Gaussian Splatting-based scene-specific approaches, as well as two diffusion-based approaches in reconstructing fine-grained geometric details across both static and dynamic regions.

  </details>



- **Leveraging Synthetic Data for Enhancing Egocentric Hand-Object Interaction Detection**  
  Rosario Leonardi, Antonino Furnari, Francesco Ragusa, Giovanni Maria Farinella  
  _2026-03-31_ · https://arxiv.org/abs/2603.29733v1  
  <details><summary>Abstract</summary>

  In this work, we explore the role of synthetic data in improving the detection of Hand-Object Interactions from egocentric images. Through extensive experimentation and comparative analysis on VISOR, EgoHOS, and ENIGMA-51 datasets, our findings demonstrate the potential of synthetic data to significantly improve HOI detection, particularly when real labeled data are scarce or unavailable. By using synthetic data and only 10% of the real labeled data, we achieve improvements in Overall AP over models trained exclusively on real data, with gains of +5.67% on VISOR, +8.24% on EgoHOS, and +11.69% on ENIGMA-51. Furthermore, we systematically study how aligning synthetic data to specific real-world benchmarks with respect to objects, grasps, and environments, showing that the effectiveness of synthetic data consistently improves with better synthetic-real alignment. As a result of this work, we release a new data generation pipeline and the new HOI-Synth benchmark, which augments existing datasets with synthetic images of hand-object interaction. These data are automatically annotated with hand-object contact states, bounding boxes, and pixel-wise segmentation masks. All data, code, and tools for synthetic data generation are available at: https://fpv-iplab.github.io/HOI-Synth/.

  </details>



- **FED-Bench: A Cross-Granular Benchmark for Disentangled Evaluation of Facial Expression Editing**  
  Fengjian Xue, Xuecheng Wu, Heli Sun, Yunyun Shi, Shi Chen, Liangyu Fu, Jinheng Xie, Dingkang Yang, Hao Wang, Junxiao Xue, et al.  
  _2026-03-31_ · https://arxiv.org/abs/2603.29697v1  
  <details><summary>Abstract</summary>

  Facial expression image editing requires fine-grained control to strictly preserve human identity and background while precisely manipulating expression. However, existing editing benchmarks primarily focus on general scenarios, lacking high-quality facial images and corresponding editing instructions. Furthermore, current evaluation metrics exhibit systemic biases in this task, often favoring lazy editing or overfit editing. To bridge these gaps, we propose FED-Bench, a comprehensive benchmark featuring rigorous testing and an accurate evaluation suite. First, we carefully construct a benchmark of 747 triplets through a cascaded and scalable pipeline, each comprising an original image, an editing instruction, and a ground-truth image for precise evaluation. Second, we introduce FED-Score, a cross-granularity evaluation protocol that disentangles assessment into three dimensions: Alignment for verifying instruction following, Fidelity for testing image quality and identity preservation, and Relative Expression Gain for quantifying the magnitude of expression changes, effectively mitigating the aforementioned evaluation biases. Third, we benchmark 18 image editing models, revealing that current approaches struggle to simultaneously achieve high fidelity and accurate expression manipulation, with fine-grained instruction following identified as the primary bottleneck. Finally, leveraging the scalable characteristic of introduced benchmark engine, we provide a 20k+ in-the-wild facial training set and demonstrate its effectiveness by fine-tuning a baseline model that achieves significant performance gains. Our benchmark and related code will be made publicly open soon.

  </details>



- **CoRe-DA: Contrastive Regression for Unsupervised Domain Adaptation in Surgical Skill Assessment**  
  Dimitrios Anastasiou, Razvan Caramalau, Jialang Xu, Runlong He, Freweini Tesfai, Matthew Boal, Nader Francis, Danail Stoyanov, Evangelos B. Mazomenos  
  _2026-03-31_ · https://arxiv.org/abs/2603.29666v1  
  <details><summary>Abstract</summary>

  Vision-based surgical skill assessment (SSA) enables objective and scalable evaluation of operative performance. Progress in this field is constrained by the high cost and time demands for manual annotation of quantitative skill scores, as well as the poor generalization of existing regression models to new surgical tasks and environments. Meanwhile, appreciable volumes of unlabeled video data are now available, motivating the development of unsupervised domain adaptation (UDA) methods for SSA. We introduce the first benchmark for UDA in SSA regression, spanning four datasets across dry-lab and clinical settings as well as open and robotic surgery. We evaluate eight representative models under challenging domain shifts and propose CoRe-DA, a novel contrastive regression-based adaptation framework. Our method learns domain-invariant representations through relative-score supervision and target-domain self-training. Comprehensive experiments across two UDA settings show that CoRe-DA is superior to state-of-the-art methods, achieving Spearman Correlation Coefficients of 0.46 and 0.41 on dry-lab and clinical target datasets, respectively, without using any labeled target data for training. Overall, CoRe-DA enables scalable SSA with reliable cross-domain generalization, where existing methods underperform. Our code and datasets will be released at https://github.com/anastadimi/CoRe-DA.

  </details>



- **BigEarthNet.txt: A Large-Scale Multi-Sensor Image-Text Dataset and Benchmark for Earth Observation**  
  Johann-Ludwig Herzog, Mathis Jürgen Adler, Leonard Hackel, Yan Shu, Angelos Zavras, Ioannis Papoutsis, Paolo Rota, Begüm Demir  
  _2026-03-31_ · https://arxiv.org/abs/2603.29630v1  
  <details><summary>Abstract</summary>

  Vision-langugage models (VLMs) have shown strong performance in computer vision (CV), yet their performance on remote sensing (RS) data remains limited due to the lack of large-scale, multi-sensor RS image-text datasets with diverse textual annotations. Existing datasets predominantly include aerial Red-Green-Blue imagery, with short or weakly grounded captions, and provide limited diversity in annotation types. To address this limitation, we introduce BigEarthNet.txt, a large-scale, multi-sensor image-text dataset designed to advance instruction-driven image-text learning in Earth observation across multiple tasks. BigEarthNet.txt contains 464044 co-registered Sentinel-1 synthetic aperture radar and Sentinel-2 multispectral images with 9.6M text annotations, including: i) geographically anchored captions describing land-use/land-cover (LULC) classes, their spatial relations, and environmental context; ii) visual question answering pairs relevant for different tasks; and iii) referring expression detection instructions for bounding box prediction. Through a comparative statistical analysis, we demonstrate that BigEarthNet.txt surpasses existing RS image-text datasets in textual richness and annotation type variety. We further establish a manually-verified benchmark split to evaluate VLMs in RS and CV. The results show the limitations of these models on tasks that involve complex LULC classes, whereas fine-tuning using BigEarthNet.txt results in consistent performance gains across all considered tasks.

  </details>



- **Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis**  
  Shuang Chen, Quanxin Shou, Hangting Chen, Yucheng Zhou, Kaituo Feng, Wenbo Hu, Yi-Fan Zhang, Yunlong Lin, Wenxuan Huang, Mingyang Song, et al.  
  _2026-03-31_ · https://arxiv.org/abs/2603.29620v1  
  <details><summary>Abstract</summary>

  Unified multimodal models provide a natural and promising architecture for understanding diverse and complex real-world knowledge while generating high-quality images. However, they still rely primarily on frozen parametric knowledge, which makes them struggle with real-world image generation involving long-tail and knowledge-intensive concepts. Inspired by the broad success of agents on real-world tasks, we explore agentic modeling to address this limitation. Specifically, we present Unify-Agent, a unified multimodal agent for world-grounded image synthesis, which reframes image generation as an agentic pipeline consisting of prompt understanding, multimodal evidence searching, grounded recaptioning, and final synthesis. To train our model, we construct a tailored multimodal data pipeline and curate 143K high-quality agent trajectories for world-grounded image synthesis, enabling effective supervision over the full agentic generation process. We further introduce FactIP, a benchmark covering 12 categories of culturally significant and long-tail factual concepts that explicitly requires external knowledge grounding. Extensive experiments show that our proposed Unify-Agent substantially improves over its base unified model across diverse benchmarks and real world generation tasks, while approaching the world knowledge capabilities of the strongest closed-source models. As an early exploration of agent-based modeling for world-grounded image synthesis, our work highlights the value of tightly coupling reasoning, searching, and generation for reliable open-world agentic image synthesis.

  </details>



- **Video-Oasis: Rethinking Evaluation of Video Understanding**  
  Geuntaek Lim, Minho Shim, Sungjune Park, Jaeyun Lee, Inwoong Lee, Taeoh Kim, Dongyoon Wee, Yukyung Choi  
  _2026-03-31_ · https://arxiv.org/abs/2603.29616v1  
  <details><summary>Abstract</summary>

  The inherent complexity of video understanding makes it difficult to attribute whether performance gains stem from visual perception, linguistic reasoning, or knowledge priors. While many benchmarks have emerged to assess high-level reasoning, the essential criteria that constitute video understanding remain largely overlooked. Instead of introducing yet another benchmark, we take a step back to re-examine the current landscape of video understanding. In this work, we provide Video-Oasis, a sustainable diagnostic suite designed to systematically evaluate existing evaluations and distill spatio-temporal challenges for video understanding. Our analysis reveals two critical findings: (1) 54% of existing benchmark samples are solvable without visual input or temporal context, and (2) on the remaining samples, state-of-the-art models exhibit performance barely exceeding random guessing. To bridge this gap, we investigate which algorithmic design choices contribute to robust video understanding, providing practical guidelines for future research. We hope our work serves as a standard guideline for benchmark construction and the rigorous evaluation of architecture development. Code is available at https://github.com/sejong-rcv/Video-Oasis.

  </details>



- **FlowID : Enhancing Forensic Identification with Latent Flow-Matching Models**  
  Jules Ripoll, David Bertoin, Alasdair Newson, Charles Dossal, Jose Pablo Baraybar  
  _2026-03-31_ · https://arxiv.org/abs/2603.29591v1  
  <details><summary>Abstract</summary>

  Every day, many people die under violent circumstances, whether from crimes, war, migration, or climate disasters. Medico-legal and law enforcement institutions document many portraits of the deceased for evidence, but cannot immediately carry out identification on them. While traditional image editing tools can process these photos for public release, the workflow is lengthy and produces suboptimal results. In this work, we leverage advances in image generation models, which can now produce photorealistic human portraits, to introduce FlowID, an identity-preserving facial reconstruction method. Our approach combines single-image fine-tuning, which adapts the generative model to out-of-distribution injured faces, with attention-based masking that localizes edits to damaged regions while preserving identity-critical features. Together, these components enable the removal of artifacts from violent death while retaining sufficient identity information to support identification. To evaluate our method, we introduce InjuredFaces, a novel benchmark for identity-preserving facial reconstruction under severe facial damage. Beyond serving as an evaluation tool for this work, InjuredFaces provides a standardized resource for the community to study and compare methods addressing facial reconstruction in extreme conditions. Experimental results show that FlowID outperforms state-of-the-art open-source methods while maintaining low memory requirements, making it suitable for local deployment without compromising data privacy.

  </details>



- **GraSP-STL: A Graph-Based Framework for Zero-Shot Signal Temporal Logic Planning via Offline Goal-Conditioned Reinforcement Learning**  
  Ancheng Hou, Ruijia Liu, Xiang Yin  
  _2026-03-31_ · https://arxiv.org/abs/2603.29533v1  
  <details><summary>Abstract</summary>

  This paper studies offline, zero-shot planning under Signal Temporal Logic (STL) specifications. We assume access only to an offline dataset of state-action-state transitions collected by a task-agnostic behavior policy, with no analytical dynamics model, no further environment interaction, and no task-specific retraining. The objective is to synthesize a control strategy whose resulting trajectory satisfies an arbitrary unseen STL specification. To this end, we propose GraSP-STL, a graph-search-based framework for zero-shot STL planning from offline trajectories. The method learns a goal-conditioned value function from offline data and uses it to induce a finite-horizon reachability metric over the state space. Based on this metric, it constructs a directed graph abstraction whose nodes represent representative states and whose edges encode feasible short-horizon transitions. Planning is then formulated as a graph search over waypoint sequences, evaluated using arithmetic-geometric mean robustness and its interval semantics, and executed by a learned goal-conditioned policy. The proposed framework separates reusable reachability learning from task-conditioned planning, enabling zero-shot generalization to unseen STL tasks and long-horizon planning through the composition of short-horizon behaviors from offline data. Experimental results demonstrate its effectiveness on a range of offline STL planning tasks.

  </details>



- **NeoNet: An End-to-End 3D MRI-Based Deep Learning Framework for Non-Invasive Prediction of Perineural Invasion via Generation-Driven Classification**  
  Youngung Han, Minkyung Cha, Kyeonghun Kim, Induk Um, Myeongbin Sho, Joo Young Bae, Jaewon Jung, Jung Hyeok Park, Seojun Lee, Nam-Joon Kim, et al.  
  _2026-03-31_ · https://arxiv.org/abs/2603.29449v1  
  <details><summary>Abstract</summary>

  Minimizing invasive diagnostic procedures to reduce the risk of patient injury and infection is a central goal in medical imaging. And yet, noninvasive diagnosis of perineural invasion (PNI), a critical prognostic factor involving infiltration of tumor cells along the surrounding nerve, still remains challenging, due to the lack of clear and consistent imaging criteria criteria for identifying PNI. To address this challenge, we present NeoNet, an integrated end-to-end 3D deep learning framework for PNI prediction in cholangiocarcinoma that does not rely on predefined image features. NeoNet integrates three modules: (1) NeoSeg, utilizing a Tumor-Localized ROI Crop (TLCR) algorithm; (2) NeoGen, a 3D Latent Diffusion Model (LDM) with ControlNet, conditioned on anatomical masks to generate synthetic image patches, specifically balancing the dataset to a 1:1 ratio; and (3) NeoCls, the final prediction module. For NeoCls, we developed the PNI-Attention Network (PattenNet), which uses the frozen LDM encoder and specialized 3D Dual Attention Blocks (DAB) designed to detect subtle intensity variations and spatial patterns indicative of PNI. In 5-fold cross-validation, NeoNet outperformed baseline 3D models and achieved the highest performance with a maximum AUC of 0.7903.

  </details>



- **A2BFR: Attribute-Aware Blind Face Restoration**  
  Chenxin Zhu, Yushun Fang, Lu Liu, Shibo Yin, Xiaohong Liu, Xiaoyun Zhang, Qiang Hu, Guangtao Zhai  
  _2026-03-31_ · https://arxiv.org/abs/2603.29423v1  
  <details><summary>Abstract</summary>

  Blind face restoration (BFR) aims to recover high-quality facial images from degraded inputs, yet its inherently ill-posed nature leads to ambiguous and uncontrollable solutions. Recent diffusion-based BFR methods improve perceptual quality but remain uncontrollable, whereas text-guided face editing enables attribute manipulation without reliable restoration. To address these issues, we propose A$^2$BFR, an attribute-aware blind face restoration framework that unifies high-fidelity reconstruction with prompt-controllable generation. Built upon a Diffusion Transformer backbone with unified image-text cross-modal attention, A$^2$BFR jointly conditions the denoising trajectory on both degraded inputs and textual prompts. To inject semantic priors, we introduce attribute-aware learning, which supervises denoising latents using facial attribute embeddings extracted by an attribute-aware encoder. To further enhance prompt controllability, we introduce semantic dual-training, which leverages the pairwise attribute variations in our newly curated AttrFace-90K dataset to enforce attribute discrimination while preserving fidelity. Extensive experiments demonstrate that A$^2$BFR achieves state-of-the-art performance in both restoration fidelity and instruction adherence, outperforming diffusion-based BFR baselines by -0.0467 LPIPS and +52.58% attribute accuracy, while enabling fine-grained, prompt-controllable restoration even under severe degradations.

  </details>



- **CLaD: Planning with Grounded Foresight via Cross-Modal Latent Dynamics**  
  Andrew Jeong, Jaemin Kim, Sebin Lee, Sung-Eui Yoon  
  _2026-03-31_ · https://arxiv.org/abs/2603.29409v1  
  <details><summary>Abstract</summary>

  Robotic manipulation involves kinematic and semantic transitions that are inherently coupled via underlying actions. However, existing approaches plan within either semantic or latent space without explicitly aligning these cross-modal transitions. To address this, we propose CLaD, a framework that models how proprioceptive and semantic states jointly evolve under actions through asymmetric cross-attention that allows kinematic transitions to query semantic ones. CLaD predicts grounded latent foresights via self-supervised objectives with EMA target encoders and auxiliary reconstruction losses, preventing representation collapse while anchoring predictions to observable states. Predicted foresights are modulated with observations to condition a diffusion policy for action generation. On LIBERO-LONG benchmark, CLaD achieves 94.7\% success rate, competitive with large VLAs with significantly fewer parameters.

  </details>



- **Learning Semantic Priorities for Autonomous Target Search**  
  Max Lodel, Nils Wilde, Robert Babuška, Javier Alonso-Mora  
  _2026-03-31_ · https://arxiv.org/abs/2603.29391v1  
  <details><summary>Abstract</summary>

  The use of semantic features can improve the efficiency of target search in unknown environments for robotic search and rescue missions. Current target search methods rely on training with large datasets of similar domains, which limits the adaptability to diverse environments. However, human experts possess high-level knowledge about semantic relationships necessary to effectively guide a robot during target search missions in diverse and previously unseen environments. In this paper, we propose a target search method that leverages expert input to train a model of semantic priorities. By employing the learned priorities in a frontier exploration planner using combinatorial optimization, our approach achieves efficient target search driven by semantic features while ensuring robustness and complete coverage. The proposed semantic priority model is trained with several synthetic datasets of simulated expert guidance for target search. Simulation tests in previously unseen environments show that our method consistently achieves faster target recovery than a coverage-driven exploration planner.

  </details>



- **PromptForge-350k: A Large-Scale Dataset and Contrastive Framework for Prompt-Based AI Image Forgery Localization**  
  Jianpeng Wang, Haoyu Wang, Baoying Chen, Jishen Zeng, Yiming Qin, Yiqi Yang, Zhongjie Ba  
  _2026-03-31_ · https://arxiv.org/abs/2603.29386v1  
  <details><summary>Abstract</summary>

  The rapid democratization of prompt-based AI image editing has recently exacerbated the risks associated with malicious content fabrication and misinformation. However, forgery localization methods targeting these emerging editing techniques remain significantly under-explored. To bridge this gap, we first introduce a fully automated mask annotating framework that leverages keypoint alignment and semantic space similarity to generate precise ground-truth masks for edited regions. Based on this framework, we construct PromptForge-350k, a large-scale forgery localization dataset covering four state-of-the-art prompt-based AI image editing models, thereby mitigating the data scarcity in this domain. Furthermore, we propose ICL-Net, an effective forgery localization network featuring a triple-stream backbone and intra-image contrastive learning. This design enables the model to capture highly robust and generalizable forensic features. Extensive experiments demonstrate that our method achieves an IoU of 62.5% on PromptForge-350k, outperforming SOTA methods by 5.1%. Additionally, it exhibits strong robustness against common degradations with an IoU drop of less than 1%, and shows promising generalization capabilities on unseen editing models, achieving an average IoU of 41.5%.

  </details>



- **Assessing Multimodal Chronic Wound Embeddings with Expert Triplet Agreement**  
  Fabian Kabus, Julia Hindel, Jelena Bratulić, Meropi Karakioulaki, Ayush Gupta, Cristina Has, Thomas Brox, Abhinav Valada, Harald Binder  
  _2026-03-31_ · https://arxiv.org/abs/2603.29376v1  
  <details><summary>Abstract</summary>

  Recessive dystrophic epidermolysis bullosa (RDEB) is a rare genetic skin disorder for which clinicians greatly benefit from finding similar cases using images and clinical text. However, off-the-shelf foundation models do not reliably capture clinically meaningful features for this heterogeneous, long-tail disease, and structured measurement of agreement with experts is challenging. To address these gaps, we propose evaluating embedding spaces with expert ordinal comparisons (triplet judgments), which are fast to collect and encode implicit clinical similarity knowledge. We further introduce TriDerm, a multimodal framework that learns interpretable wound representations from small cohorts by integrating wound imagery, boundary masks, and expert reports. On the vision side, TriDerm adapts visual foundation models to RDEB using wound-level attention pooling and non-contrastive representation learning. For text, we prompt large language models with comparison queries and recover medically meaningful representations via soft ordinal embeddings (SOE). We show that visual and textual modalities capture complementary aspects of wound phenotype, and that fusing both modalities yields 73.5% agreement with experts, outperforming the best off-the-shelf single-modality foundation model by over 5.6 percentage points. We make the expert annotation tool, model code and representative dataset samples publicly available.

  </details>



- **StereoVGGT: A Training-Free Visual Geometry Transformer for Stereo Vision**  
  Ziyang Chen, Yansong Qu, You Shen, Xuan Cheng, Liujuan Cao  
  _2026-03-31_ · https://arxiv.org/abs/2603.29368v1  
  <details><summary>Abstract</summary>

  Driven by the advancement of 3D devices, stereo vision tasks including stereo matching and stereo conversion have emerged as a critical research frontier. Contemporary stereo vision backbones typically rely on either monocular depth estimation (MDE) models or visual foundation models (VFMs). Crucially, these models are predominantly pretrained without explicit supervision of camera poses. Given that such geometric knowledge is indispensable for stereo vision, the absence of explicit spatial constraints constitutes a significant performance bottleneck for existing architectures. Recognizing that the Visual Geometry Grounded Transformer (VGGT) operates as a foundation model pretrained on extensive 3D priors, including camera poses, we investigate its potential as a robust backbone for stereo vision tasks. Nevertheless, empirical results indicate that its direct application to stereo vision yields suboptimal performance. We observe that VGGT suffers from a more significant degradation of geometric details during feature extraction. Such characteristics conflict with the requirements of binocular stereo vision, thereby constraining its efficacy for relative tasks. To bridge this gap, we propose StereoVGGT, a feature backbone specifically tailored for stereo vision. By leveraging the frozen VGGT and introducing a training-free feature adjustment pipeline, we mitigate geometric degradation and harness the latent camera calibration knowledge embedded within the model. StereoVGGT-based stereo matching network achieved the $1^{st}$ rank among all published methods on the KITTI benchmark, validating that StereoVGGT serves as a highly effective backbone for stereo vision.

  </details>



- **Uncertainty-Aware Trajectory Prediction: A Unified Framework Harnessing Positional and Semantic Uncertainties**  
  Jintao Sun, Hu Zhang, Gangyi Ding, Zhedong Zheng  
  _2026-03-31_ · https://arxiv.org/abs/2603.29362v1  
  <details><summary>Abstract</summary>

  Trajectory prediction seeks to forecast the future motion of dynamic entities, such as vehicles and pedestrians, given a temporal horizon of historical movement data and environmental context. A central challenge in this domain is the inherent uncertainty in real-time maps, arising from two primary sources: (1) positional inaccuracies due to sensor limitations or environmental occlusions, and (2) semantic errors stemming from misinterpretations of scene context. To address these challenges, we propose a novel unified framework that jointly models positional and semantic uncertainties and explicitly integrates them into the trajectory prediction pipeline. Our approach employs a dual-head architecture to independently estimate semantic and positional predictions in a dual-pass manner, deriving prediction variances as uncertainty indicators in an end-to-end fashion. These uncertainties are subsequently fused with the semantic and positional predictions to enhance the robustness of trajectory forecasts. We evaluate our uncertainty-aware framework on the nuScenes real-world driving dataset, conducting extensive experiments across four map estimation methods and two trajectory prediction baselines. Results verify that our method (1) effectively quantifies map uncertainties through both positional and semantic dimensions, and (2) consistently improves the performance of existing trajectory prediction models across multiple metrics, including minimum Average Displacement Error (minADE), minimum Final Displacement Error (minFDE), and Miss Rate (MR). Code will available at https://github.com/JT-Sun/UATP.

  </details>



- **FOSCU: Feasibility of Synthetic MRI Generation via Duo-Diffusion Models for Enhancement of 3D U-Nets in Hepatic Segmentation**  
  Youngung Han, Kyeonghun Kim, Seoyoung Ju, Yeonju Jean, Minkyung Cha, Seohyoung Park, Hyeonseok Jung, Nam-Joon Kim, Woo Kyoung Jeong, Ken Ying-Kai Liao, et al.  
  _2026-03-31_ · https://arxiv.org/abs/2603.29343v1  
  <details><summary>Abstract</summary>

  Medical image segmentation faces fundamental challenges including restricted access, costly annotation, and data shortage to clinical datasets through Picture Archiving and Communication Systems (PACS). These systemic barriers significantly impede the development of robust segmentation algorithms. To address these challenges, we propose FOSCU, which integrates Duo-Diffusion, a 3D latent diffusion model with ControlNet that simultaneously generates high-resolution, anatomically realistic synthetic MRI volumes and corresponding segmentation labels, and an enhanced 3D U-Net training pipeline. Duo-Diffusion employs segmentation-conditioned diffusion to ensure spatial consistency and precise anatomical detail in the generated data. Experimental evaluation on 720 abdominal MRI scans shows that models trained with combined real and synthetic data yield a mean Dice score gain of 0.67% over those using only real data, and achieve a 36.4% reduction in Fréchet Inception Distance (FID), reflecting enhanced image fidelity.

  </details>



- **Beyond Corner Patches: Semantics-Aware Backdoor Attack in Federated Learning**  
  Kavindu Herath, Joshua Zhao, Saurabh Bagchi  
  _2026-03-31_ · https://arxiv.org/abs/2603.29328v1  
  <details><summary>Abstract</summary>

  Backdoor attacks on federated learning (FL) are most often evaluated with synthetic corner patches or out-of-distribution (OOD) patterns that are unlikely to arise in practice. In this paper, we revisit the backdoor threat to standard FL (a single global model) under a more realistic setting where triggers must be semantically meaningful, in-distribution, and visually plausible. We propose SABLE, a Semantics-Aware Backdoor for LEarning in federated settings, which constructs natural, content-consistent triggers (e.g., semantic attribute changes such as sunglasses) and optimizes an aggregation-aware malicious objective with feature separation and parameter regularization to keep attacker updates close to benign ones. We instantiate SABLE on CelebA hair-color classification and the German Traffic Sign Recognition Benchmark (GTSRB), poisoning only a small, interpretable subset of each malicious client's local data while otherwise following the standard FL protocol. Across heterogeneous client partitions and multiple aggregation rules (FedAvg, Trimmed Mean, MultiKrum, and FLAME), our semantics-driven triggers achieve high targeted attack success rates while preserving benign test accuracy. These results show that semantics-aligned backdoors remain a potent and practical threat in federated learning, and that robustness claims based solely on synthetic patch triggers can be overly optimistic.

  </details>



- **Self-Consistency for LLM-Based Motion Trajectory Generation and Verification**  
  Jiaju Ma, R. Kenny Jones, Jiajun Wu, Maneesh Agrawala  
  _2026-03-31_ · https://arxiv.org/abs/2603.29301v1  
  <details><summary>Abstract</summary>

  Self-consistency has proven to be an effective technique for improving LLM performance on natural language reasoning tasks in a lightweight, unsupervised manner. In this work, we study how to adapt self-consistency to visual domains. Specifically, we consider the generation and verification of LLM-produced motion graphics trajectories. Given a prompt (e.g., "Move the circle in a spiral path"), we first sample diverse motion trajectories from an LLM, and then identify groups of consistent trajectories via clustering. Our key insight is to model the family of shapes associated with a prompt as a prototype trajectory paired with a group of geometric transformations (e.g., rigid, similarity, and affine). Two trajectories can then be considered consistent if one can be transformed into the other under the warps allowable by the transformation group. We propose an algorithm that automatically recovers a shape family, using hierarchical relationships between a set of candidate transformation groups. Our approach improves the accuracy of LLM-based trajectory generation by 4-6%. We further extend our method to support verification, observing 11% precision gains over VLM baselines. Our code and dataset are available at https://majiaju.io/trajectory-self-consistency .

  </details>



- **GazeCLIP: Gaze-Guided CLIP with Adaptive-Enhanced Fine-Grained Language Prompt for Deepfake Attribution and Detection**  
  Yaning Zhang, Linlin Shen, Zitong Yu, Chunjie Ma, Zan Gao  
  _2026-03-31_ · https://arxiv.org/abs/2603.29295v1  
  <details><summary>Abstract</summary>

  Current deepfake attribution or deepfake detection works tend to exhibit poor generalization to novel generative methods due to the limited exploration in visual modalities alone. They tend to assess the attribution or detection performance of models on unseen advanced generators, coarsely, and fail to consider the synergy of the two tasks. To this end, we propose a novel gaze-guided CLIP with adaptive-enhanced fine-grained language prompts for fine-grained deepfake attribution and detection (DFAD). Specifically, we conduct a novel and fine-grained benchmark to evaluate the DFAD performance of networks on novel generators like diffusion and flow models. Additionally, we introduce a gaze-aware model based on CLIP, which is devised to enhance the generalization to unseen face forgery attacks. Built upon the novel observation that there are significant distribution differences between pristine and forged gaze vectors, and the preservation of the target gaze in facial images generated by GAN and diffusion varies significantly, we design a visual perception encoder to employ the inherent gaze differences to mine global forgery embeddings across appearance and gaze domains. We propose a gaze-aware image encoder (GIE) that fuses forgery gaze prompts extracted via a gaze encoder with common forged image embeddings to capture general attribution patterns, allowing features to be transformed into a more stable and common DFAD feature space. We build a language refinement encoder (LRE) to generate dynamically enhanced language embeddings via an adaptive-enhanced word selector for precise vision-language matching. Extensive experiments on our benchmark show that our model outperforms the state-of-the-art by 6.56% ACC and 5.32% AUC in average performance under the attribution and detection settings, respectively. Codes will be available on GitHub.

  </details>



- **PRISM: A Multi-View Multi-Capability Retail Video Dataset for Embodied Vision-Language Models**  
  Amirreza Rouhi, Parikshit Sakurikar, Satya Sai Reddy, Narsimha Menga, Anirudh Govil, Sri Harsha Chittajallu, Rajat Aggarwal, Anoop Namboodiri, Sashi Reddi  
  _2026-03-31_ · https://arxiv.org/abs/2603.29281v1  
  <details><summary>Abstract</summary>

  A critical gap exists between the general-purpose visual understanding of state-of-the-art physical AI models and the specialized perceptual demands of structured real-world deployment environments. We present PRISM, a 270K-sample multi-view video supervised fine-tuning (SFT) corpus for embodied vision-language-models (VLMs) in real-world retail environments. PRISM is motivated by a simple observation - physical AI systems fail not because of poor visual recognition, but because they do not understand space, physical dynamics and embodied action well enough to operate reliably in the world. To this end, PRISM is grounded in a novel three-dimensional knowledge ontology that spans spatial knowledge, temporal and physical knowledge, and embodied action knowledge. It covers 20+ capability probes across four evaluation dimensions - Embodied Reasoning (ER), Common Sense (CS), Spatial Perception (SP), and Intuitive Physics (IP), and to our knowledge, PRISM is the first dataset to instantiate all three knowledge dimensions within a single real-world deployment domain. The corpus captures data from egocentric, exocentric and 360° viewpoints across five supermarket locations and includes open-ended, chain-of-thought, and multiple-choice supervision. At 4 fps, PRISM spans approximately 11.8M video frames and approximately 730M tokens, placing it among the largest domain-specific video SFT corpora. Fine-tuning on PRISM reduces the error rate across all 20+ probes by 66.6% over the pre-trained baseline, with significant gains in embodied action understanding where the accuracy improves by 36.4%. Our results suggest that ontology-structured, domain specific SFT can meaningfully strengthen embodied VLMs for real-world settings. The PRISM dataset and more details are available at https://dreamvu.ai/prism

  </details>



- **ConInfer: Context-Aware Inference for Training-Free Open-Vocabulary Remote Sensing Segmentation**  
  Wenyang Chen, Zhanxuan Hu, Yaping Zhang, Hailong Ning, Yonghang Tai  
  _2026-03-31_ · https://arxiv.org/abs/2603.29271v1  
  <details><summary>Abstract</summary>

  Training-free open-vocabulary remote sensing segmentation (OVRSS), empowered by vision-language models, has emerged as a promising paradigm for achieving category-agnostic semantic understanding in remote sensing imagery. Existing approaches mainly focus on enhancing feature representations or mitigating modality discrepancies to improve patch-level prediction accuracy. However, such independent prediction schemes are fundamentally misaligned with the intrinsic characteristics of remote sensing data. In real-world applications, remote sensing scenes are typically large-scale and exhibit strong spatial as well as semantic correlations, making isolated patch-wise predictions insufficient for accurate segmentation. To address this limitation, we propose ConInfer, a context-aware inference framework for OVRSS that performs joint prediction across multiple spatial units while explicitly modeling their inter-unit semantic dependencies. By incorporating global contextual cues, our method significantly enhances segmentation consistency, robustness, and generalization in complex remote sensing environments. Extensive experiments on multiple benchmark datasets demonstrate that our approach consistently surpasses state-of-the-art per-pixel VLM-based baselines such as SegEarth-OV, achieving average improvements of 2.80% and 6.13% on open-vocabulary semantic segmentation and object extraction tasks, respectively. The implementation code is available at: https://github.com/Dog-Yang/ConInfer

  </details>



- **SuperGrasp: Single-View Object Grasping via Superquadric Similarity Matching, Evaluation, and Refinement**  
  Lijingze Xiao, Jinhong Du, Yang Cong, Supeng Diao, Yu Ren  
  _2026-03-31_ · https://arxiv.org/abs/2603.29254v1  
  <details><summary>Abstract</summary>

  Robotic grasping from single-view observations remains a critical challenge in manipulation. Existing methods still struggle to generate stable and valid grasp poses when confronted with incomplete geometric information. To address these limitations, we propose SuperGrasp, a novel two-stage framework for single-view grasping with parallel-jaw grippers that decomposes the grasping process into initial grasp pose generation and subsequent grasp evaluation and refinement. In the first stage, we introduce a Similarity Matching Module that efficiently retrieves grasp candidates by matching the input single-view point cloud with a pre-computed primitive dataset based on superquadric coefficients. In the second stage, we propose E-RNet, an end-to-end network that expands the graspaware region and takes the initial grasp closure region as a local anchor region, enabling more accurate and reliable evaluation and refinement of grasp candidates. To enhance generalization, we construct a primitive dataset containing 1.5k primitives for similarity matching and collect a large-scale point cloud dataset with 100k stable grasp labels from 124 objects for network training. Extensive experiments in both simulation and realworld environments demonstrate that our method achieves stable grasping performance and strong generalization across varying scenes and novel objects.

  </details>



- **Monocular Building Height Estimation from PhiSat-2 Imagery: Dataset and Method**  
  Yanjiao Song, Bowen Cai, Timo Balz, Zhenfeng Shao, Neema Simon Sumari, James Magidi, Walter Musakwa  
  _2026-03-31_ · https://arxiv.org/abs/2603.29245v1  
  <details><summary>Abstract</summary>

  Monocular building height estimation from optical imagery is important for urban morphology characterization but remains challenging due to ambiguous height cues, large inter-city variations in building morphology, and the long-tailed distribution of building heights. PhiSat-2 is a promising open-access data source for this task because of its global coverage, 4.75 m spatial resolution, and seven-band spectral observations, yet its potential has not been systematically evaluated. To address this gap, we construct a PhiSat-2-Height dataset (PHDataset) and propose a Two-Stream Ordinal Network (TSONet). PHDataset contains 9,475 co-registered image-label patch pairs from 26 cities worldwide. TSONet jointly models footprint segmentation and height estimation, and introduces a Cross-Stream Exchange Module (CSEM) and a Feature-Enhanced Bin Refinement (FEBR) module for footprint-aware feature interaction and ordinal height refinement. Experiments on PHDataset show that TSONet achieves the best overall performance, reducing MAE and RMSE by 13.2% and 9.7%, and improving IoU and F1-score by 14.0% and 10.1% over the strongest competing results. Ablation studies further verify the effectiveness of CSEM, FEBR, and the joint use of ordinal regression and footprint assistance. Additional analyses indicate that PhiSat-2 benefits monocular building height estimation through its balanced combination of building-relevant spatial detail and multispectral observations. Overall, this study confirms the potential of PhiSat-2 for monocular building height estimation and provides a dedicated dataset and an effective method for future research.

  </details>



- **SyriSign: A Parallel Corpus for Arabic Text to Syrian Arabic Sign Language Translation**  
  Mohammad Amer Khalil, Raghad Nahas, Ahmad Nassar, Khloud Al Jallad  
  _2026-03-31_ · https://arxiv.org/abs/2603.29219v1  
  <details><summary>Abstract</summary>

  Sign language is the primary approach of communication for the Deaf and Hard-of-Hearing (DHH) community. While there are numerous benchmarks for high-resource sign languages, low-resource languages like Arabic remain underrepresented. Currently, there is no publicly available dataset for Syrian Arabic Sign Language (SyArSL). To overcome this gap, we introduce SyriSign, a dataset comprising 1500 video samples across 150 unique lexical signs, designed for text-to-SyArSL translation tasks. This work aims to reduce communication barriers in Syria, as most news are delivered in spoken or written Arabic, which is often inaccessible to the deaf community. We evaluated SyriSign using three deep learning architectures: MotionCLIP for semantic motion generation, T2M-GPT for text-conditioned motion synthesis, and SignCLIP for bilingual embedding alignment. Experimental results indicate that while generative approaches show strong potential for sign representation, the limited dataset size constrains generalization performance. We will release SyriSign publicly, hoping it serves as an initial benchmark.

  </details>



- **LightHarmony3D: Harmonizing Illumination and Shadows for Object Insertion in 3D Gaussian Splatting**  
  Tianyu Huang, Zhenyang Ren, Zhenchen Wan, Jiyang Zheng, Wenjie Wang, Runnan Chen, Mingming Gong, Tongliang Liu  
  _2026-03-31_ · https://arxiv.org/abs/2603.29209v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) enables high-fidelity reconstruction of scene geometry and appearance. Building on this capability, inserting external mesh objects into reconstructed 3DGS scenes enables interactive editing and content augmentation for immersive applications such as AR/VR, virtual staging, and digital content creation. However, achieving physically consistent lighting and shadows for mesh insertion remains challenging, as it requires accurate scene illumination estimation and multi-view consistent rendering. To address this challenge, we present LightHarmony3D, a novel framework for illumination-consistent mesh insertion in 3DGS scenes. Central to our approach is our proposed generative module that predicts a full 360° HDR environment map at the insertion location via a single forward pass. By leveraging generative priors instead of iterative optimization, our method efficiently captures dominant scene illumination and enables physically grounded shading and shadows for inserted meshes while maintaining multi-view coherence. Furthermore, we introduce the first dedicated benchmark for mesh insertion in 3DGS, providing a standardized evaluation framework for assessing lighting consistency and photorealism. Extensive experiments across multiple real-world reconstruction datasets demonstrate that LightHarmony3D achieves state-of-the-art realism and multi-view consistency.

  </details>



- **SLVMEval: Synthetic Meta Evaluation Benchmark for Text-to-Long Video Generation**  
  Ryosuke Matsuda, Keito Kudo, Haruto Yoshida, Nobuyuki Shimizu, Jun Suzuki  
  _2026-03-31_ · https://arxiv.org/abs/2603.29186v1  
  <details><summary>Abstract</summary>

  This paper proposes the synthetic long-video meta-evaluation (SLVMEval), a benchmark for meta-evaluating text-to-video (T2V) evaluation systems. The proposed SLVMEval benchmark focuses on assessing these systems on videos of up to 10,486 s (approximately 3 h). The benchmark targets a fundamental requirement, namely, whether the systems can accurately assess video quality in settings that are easy for humans to assess. We adopt a pairwise comparison-based meta-evaluation framework. Building on dense video-captioning datasets, we synthetically degrade source videos to create controlled "high-quality versus low-quality" pairs across 10 distinct aspects. Then, we employ crowdsourcing to filter and retain only those pairs in which the degradation is clearly perceptible, thereby establishing an effective final testbed. Using this testbed, we assess the reliability of existing evaluation systems in ranking these pairs. Experimental results demonstrate that human evaluators can identify the better long video with 84.7%-96.8% accuracy, and in nine of the 10 aspects, the accuracy of these systems falls short of human assessment, revealing weaknesses in text-to-long-video evaluation.

  </details>



- **Predicting Neuromodulation Outcome for Parkinson's Disease with Generative Virtual Brain Model**  
  Siyuan Du, Siyi Li, Shuwei Bai, Ang Li, Haolin Li, Mingqing Xiao, Yang Pan, Dongsheng Li, Weidi Xie, Yanfeng Wang, et al.  
  _2026-03-31_ · https://arxiv.org/abs/2603.29176v1  
  <details><summary>Abstract</summary>

  Parkinson's disease (PD) affects over ten million people worldwide. Although temporal interference (TI) and deep brain stimulation (DBS) are promising therapies, inter-individual variability limits empirical treatment selection, increasing non-negligible surgical risk and cost. Previous explorations either resort to limited statistical biomarkers that are insufficient to characterize variability, or employ AI-driven methods which is prone to overfitting and opacity. We bridge this gap with a pretraining-finetuning framework to predict outcomes directly from resting-state fMRI. Critically, a generative virtual brain foundation model, pretrained on a collective dataset (2707 subjects, 5621 sessions) to capture universal disorder patterns, was finetuned on PD cohorts receiving TI (n=51) or DBS (n=55) to yield individualized virtual brains with high fidelity to empirical functional connectivity (r=0.935). By constructing counterfactual estimations between pathological and healthy neural states within these personalized models, we predicted clinical responses (TI: AUPR=0.853; DBS: AUPR=0.915), substantially outperforming baselines. External and prospective validations (n=14, n=11) highlight the feasibility of clinical translation. Moreover, our framework provides state-dependent regional patterns linked to response, offering hypothesis-generating mechanistic insights.

  </details>



- **Segmentation of Gray Matters and White Matters from Brain MRI data**  
  Chang Sun, Rui Shi, Tsukasa Koike, Tetsuro Sekine, Akio Morita, Tetsuya Sakai  
  _2026-03-31_ · https://arxiv.org/abs/2603.29171v1  
  <details><summary>Abstract</summary>

  Accurate segmentation of brain tissues such as gray matter and white matter from magnetic resonance imaging is essential for studying brain anatomy, diagnosing neurological disorders, and monitoring disease progression. Traditional methods, such as FSL FAST, produce tissue probability maps but often require task-specific adjustments and face challenges with diverse imaging conditions. Recent foundation models, such as MedSAM, offer a prompt-based approach that leverages large-scale pretraining. In this paper, we propose a modified MedSAM model designed for multi-class brain tissue segmentation. Our preprocessing pipeline includes skull stripping with FSL BET, tissue probability mapping with FSL FAST, and converting these into 2D axial, sagittal, coronal slices with multi-class labels (background, gray matter, and white matter). We extend MedSAM's mask decoder to three classes, freezing the pre-trained image encoder and fine-tuning the prompt encoder and decoder. Experiments on the IXI dataset achieve Dice scores up to 0.8751. This work demonstrates that foundation models like MedSAM can be adapted for multi-class medical image segmentation with minimal architectural modifications. Our findings suggest that such models can be extended to more diverse medical imaging scenarios in future work.

  </details>



- **Enhancing Box and Block Test with Computer Vision for Post-Stroke Upper Extremity Motor Evaluation**  
  David Robinson, Animesh Gupta, Elizabeth Clark, Olga Melnik, Qiushi Fu, Mubarak Shah  
  _2026-03-31_ · https://arxiv.org/abs/2603.29101v1  
  <details><summary>Abstract</summary>

  Standard clinical assessments of upper-extremity motor function after stroke either rely on ordinal scoring, which lacks sensitivity, or time-based task metrics, which do not capture movement quality. In this work, we present a computer vision-based framework for analysis of upper-extremity movement during the Box and Block Test (BBT) through world-aligned joint angles of fingers, arm, and trunk without depth sensors or calibration objects. We apply this framework to a dataset of 136 BBT recordings collected from 48 healthy individuals and 7 individuals post stroke. Using unsupervised dimensionality reduction of joint-angle features, we analyze movement patterns without relying on expert clinical labels. The resulting embeddings show separation between healthy movement patterns and stroke-related movement deviations. Importantly, some patients with the same BBT scores can be separated with different postural patterns. These results show that world-aligned joint angles can capture meaningful information of upper-extremity functions beyond standard time-based BBT scores, with no effort from the clinician other than monocular video recordings of the patient using a phone or camera. This work highlights the potential of a camera-based, calibration-free framework to measure movement quality in clinical assessments without changing the widely adopted clinical routine.

  </details>



- **HCLSM: Hierarchical Causal Latent State Machines for Object-Centric World Modeling**  
  Jaber Jaber, Osama Jaber  
  _2026-03-31_ · https://arxiv.org/abs/2603.29090v1  
  <details><summary>Abstract</summary>

  World models that predict future states from video remain limited by flat latent representations that entangle objects, ignore causal structure, and collapse temporal dynamics into a single scale. We present HCLSM, a world model architecture that operates on three interconnected principles: object-centric decomposition via slot attention with spatial broadcast decoding, hierarchical temporal dynamics through a three-level engine combining selective state space models for continuous physics, sparse transformers for discrete events, and compressed transformers for abstract goals, and causal structure learning through graph neural network interaction patterns. HCLSM introduces a two-stage training protocol where spatial reconstruction forces slot specialization before dynamics prediction begins. We train a 68M-parameter model on the PushT robotic manipulation benchmark from the Open X-Embodiment dataset, achieving 0.008 MSE next-state prediction loss with emerging spatial decomposition (SBD loss: 0.0075) and learned event boundaries. A custom Triton kernel for the SSM scan delivers 38x speedup over sequential PyTorch. The full system spans 8,478 lines of Python across 51 modules with 171 unit tests. Code: https://github.com/rightnow-ai/hclsm

  </details>



- **Generating Humanless Environment Walkthroughs from Egocentric Walking Tour Videos**  
  Yujin Ham, Junho Kim, Vivek Boominathan, Guha Balakrishnan  
  _2026-03-30_ · https://arxiv.org/abs/2603.29036v1  
  <details><summary>Abstract</summary>

  Egocentric "walking tour" videos provide a rich source of image data to develop rich and diverse visual models of environments around the world. However, the significant presence of humans in frames of these videos due to crowds and eye-level camera perspectives mitigates their usefulness in environment modeling applications. We focus on addressing this challenge by developing a generative algorithm that can realistically remove (i.e., inpaint) humans and their associated shadow effects from walking tour videos. Key to our approach is the construction of a rich semi-synthetic dataset of video clip pairs to train this generative model. Each pair in the dataset consists of an environment-only background clip, and a composite clip of walking humans with simulated shadows overlaid on the background. We randomly sourced both foreground and background components from real egocentric walking tour videos around the world to maintain visual diversity. We then used this dataset to fine-tune the state-of-the-art Casper video diffusion model for object and effects inpainting, and demonstrate that the resulting model performs far better than Casper both qualitatively and quantitatively at removing humans from walking tour clips with significant human presence and complex backgrounds. Finally, we show that the resulting generated clips can be used to build successful 3D/4D models of urban locations.

  </details>



- **MMFace-DiT: A Dual-Stream Diffusion Transformer for High-Fidelity Multimodal Face Generation**  
  Bharath Krishnamurthy, Ajita Rattani  
  _2026-03-30_ · https://arxiv.org/abs/2603.29029v1  
  <details><summary>Abstract</summary>

  Recent multimodal face generation models address the spatial control limitations of text-to-image diffusion models by augmenting text-based conditioning with spatial priors such as segmentation masks, sketches, or edge maps. This multimodal fusion enables controllable synthesis aligned with both high-level semantic intent and low-level structural layout. However, most existing approaches typically extend pre-trained text-to-image pipelines by appending auxiliary control modules or stitching together separate uni-modal networks. These ad hoc designs inherit architectural constraints, duplicate parameters, and often fail under conflicting modalities or mismatched latent spaces, limiting their ability to perform synergistic fusion across semantic and spatial domains. We introduce MMFace-DiT, a unified dual-stream diffusion transformer engineered for synergistic multimodal face synthesis. Its core novelty lies in a dual-stream transformer block that processes spatial (mask/sketch) and semantic (text) tokens in parallel, deeply fusing them through a shared Rotary Position-Embedded (RoPE) Attention mechanism. This design prevents modal dominance and ensures strong adherence to both text and structural priors to achieve unprecedented spatial-semantic consistency for controllable face generation. Furthermore, a novel Modality Embedder enables a single cohesive model to dynamically adapt to varying spatial conditions without retraining. MMFace-DiT achieves a 40% improvement in visual fidelity and prompt alignment over six state-of-the-art multimodal face generation models, establishing a flexible new paradigm for end-to-end controllable generative modeling. The code and dataset are available on our project page: https://vcbsl.github.io/MMFace-DiT/

  </details>



- **Stepper: Stepwise Immersive Scene Generation with Multiview Panoramas**  
  Felix Wimbauer, Fabian Manhardt, Michael Oechsle, Nikolai Kalischek, Christian Rupprecht, Daniel Cremers, Federico Tombari  
  _2026-03-30_ · https://arxiv.org/abs/2603.28980v1  
  <details><summary>Abstract</summary>

  The synthesis of immersive 3D scenes from text is rapidly maturing, driven by novel video generative models and feed-forward 3D reconstruction, with vast potential in AR/VR and world modeling. While panoramic images have proven effective for scene initialization, existing approaches suffer from a trade-off between visual fidelity and explorability: autoregressive expansion suffers from context drift, while panoramic video generation is limited to low resolution. We present Stepper, a unified framework for text-driven immersive 3D scene synthesis that circumvents these limitations via stepwise panoramic scene expansion. Stepper leverages a novel multi-view 360° diffusion model that enables consistent, high-resolution expansion, coupled with a geometry reconstruction pipeline that enforces geometric coherence. Trained on a new large-scale, multi-view panorama dataset, Stepper achieves state-of-the-art fidelity and structural consistency, outperforming prior approaches, thereby setting a new standard for immersive scene generation.

  </details>



- **Large Neighborhood Search for Multi-Agent Task Assignment and Path Finding with Precedence Constraints**  
  Viraj Parimi, Brian C. Williams  
  _2026-03-30_ · https://arxiv.org/abs/2603.28968v1  
  <details><summary>Abstract</summary>

  Many multi-robot applications require tasks to be completed efficiently and in the correct order, so that downstream operations can proceed at the right time. Multi-agent path finding with precedence constraints (MAPF-PC) is a well-studied framework for computing collision-free plans that satisfy ordering relations when task sequences are fixed in advance. In many applications, however, solution quality depends not only on how agents move, but also on which agent performs which task. This motivates the lifted problem of task assignment and path finding with precedence constraints (TAPF-PC), which extends MAPF-PC by jointly optimizing assignment, precedence satisfaction, and routing cost. To address the resulting coupled TAPF-PC search space, we develop a large neighborhood search approach that starts from a feasible MAPF-PC seed and iteratively improves it through reassignment-based neighborhood repair, restoring feasibility within each selected neighborhood. Experiments across multiple benchmark families and scaling regimes show that the best-performing configuration improves 89.1% of instances over fixed-assignment seed solutions, demonstrating that large neighborhood search effectively captures the gains from flexible reassignment under precedence constraints.

  </details>


