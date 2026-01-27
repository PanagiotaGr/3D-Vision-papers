# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-01-27 06:53 UTC_

Total papers shown: **50**


---

- **Trustworthy Evaluation of Robotic Manipulation: A New Benchmark and AutoEval Methods**  
  Mengyuan Liu, Juyi Sheng, Peiming Li, Ziyi Wang, Tianming Xu, Tiantian Xu, Hong Liu  
  _2026-01-26_ · https://arxiv.org/abs/2601.18723v1  
  <details><summary>Abstract</summary>

  Driven by the rapid evolution of Vision-Action and Vision-Language-Action models, imitation learning has significantly advanced robotic manipulation capabilities. However, evaluation methodologies have lagged behind, hindering the establishment of Trustworthy Evaluation for these behaviors. Current paradigms rely on binary success rates, failing to address the critical dimensions of trust: Source Authenticity (i.e., distinguishing genuine policy behaviors from human teleoperation) and Execution Quality (e.g., smoothness and safety). To bridge these gaps, we propose a solution that combines the Eval-Actions benchmark and the AutoEval architecture. First, we construct the Eval-Actions benchmark to support trustworthiness analysis. Distinct from existing datasets restricted to successful human demonstrations, Eval-Actions integrates VA and VLA policy execution trajectories alongside human teleoperation data, explicitly including failure scenarios. This dataset is structured around three core supervision signals: Expert Grading (EG), Rank-Guided preferences (RG), and Chain-of-Thought (CoT). Building on this, we propose the AutoEval architecture: AutoEval leverages Spatio-Temporal Aggregation for semantic assessment, augmented by an auxiliary Kinematic Calibration Signal to refine motion smoothness; AutoEval Plus (AutoEval-P) incorporates the Group Relative Policy Optimization (GRPO) paradigm to enhance logical reasoning capabilities. Experiments show AutoEval achieves Spearman's Rank Correlation Coefficients (SRCC) of 0.81 and 0.84 under the EG and RG protocols, respectively. Crucially, the framework possesses robust source discrimination capabilities, distinguishing between policy-generated and teleoperated videos with 99.6% accuracy, thereby establishing a rigorous standard for trustworthy robotic evaluation. Our project and code are available at https://term-bench.github.io/.

  </details>



- **Are Video Generation Models Geographically Fair? An Attraction-Centric Evaluation of Global Visual Knowledge**  
  Xiao Liu, Jiawei Zhang  
  _2026-01-26_ · https://arxiv.org/abs/2601.18698v1  
  <details><summary>Abstract</summary>

  Recent advances in text-to-video generation have produced visually compelling results, yet it remains unclear whether these models encode geographically equitable visual knowledge. In this work, we investigate the geo-equity and geographically grounded visual knowledge of text-to-video models through an attraction-centric evaluation. We introduce Geo-Attraction Landmark Probing (GAP), a systematic framework for assessing how faithfully models synthesize tourist attractions from diverse regions, and construct GEOATTRACTION-500, a benchmark of 500 globally distributed attractions spanning varied regions and popularity levels. GAP integrates complementary metrics that disentangle overall video quality from attraction-specific knowledge, including global structural alignment, fine-grained keypoint-based alignment, and vision-language model judgments, all validated against human evaluation. Applying GAP to the state-of-the-art text-to-video model Sora 2, we find that, contrary to common assumptions of strong geographic bias, the model exhibits a relatively uniform level of geographically grounded visual knowledge across regions, development levels, and cultural groupings, with only weak dependence on attraction popularity. These results suggest that current text-to-video models express global visual knowledge more evenly than expected, highlighting both their promise for globally deployed applications and the need for continued evaluation as such systems evolve.

  </details>



- **A Pragmatic VLA Foundation Model**  
  Wei Wu, Fan Lu, Yunnan Wang, Shuai Yang, Shi Liu, Fangjing Wang, Qian Zhu, He Sun, Yong Wang, Shuailei Ma, et al.  
  _2026-01-26_ · https://arxiv.org/abs/2601.18692v1  
  <details><summary>Abstract</summary>

  Offering great potential in robotic manipulation, a capable Vision-Language-Action (VLA) foundation model is expected to faithfully generalize across tasks and platforms while ensuring cost efficiency (e.g., data and GPU hours required for adaptation). To this end, we develop LingBot-VLA with around 20,000 hours of real-world data from 9 popular dual-arm robot configurations. Through a systematic assessment on 3 robotic platforms, each completing 100 tasks with 130 post-training episodes per task, our model achieves clear superiority over competitors, showcasing its strong performance and broad generalizability. We have also built an efficient codebase, which delivers a throughput of 261 samples per second per GPU with an 8-GPU training setup, representing a 1.5~2.8$\times$ (depending on the relied VLM base model) speedup over existing VLA-oriented codebases. The above features ensure that our model is well-suited for real-world deployment. To advance the field of robot learning, we provide open access to the code, base model, and benchmark data, with a focus on enabling more challenging tasks and promoting sound evaluation standards.

  </details>



- **Scale-Aware Self-Supervised Learning for Segmentation of Small and Sparse Structures**  
  Jorge Quesada, Ghassan AlRegib  
  _2026-01-26_ · https://arxiv.org/abs/2601.18619v1  
  <details><summary>Abstract</summary>

  Self-supervised learning (SSL) has emerged as a powerful strategy for representation learning under limited annotation regimes, yet its effectiveness remains highly sensitive to many factors, especially the nature of the target task. In segmentation, existing pipelines are typically tuned to large, homogeneous regions, but their performance drops when objects are small, sparse, or locally irregular. In this work, we propose a scale-aware SSL adaptation that integrates small-window cropping into the augmentation pipeline, zooming in on fine-scale structures during pretraining. We evaluate this approach across two domains with markedly different data modalities: seismic imaging, where the goal is to segment sparse faults, and neuroimaging, where the task is to delineate small cellular structures. In both settings, our method yields consistent improvements over standard and state-of-the-art baselines under label constraints, improving accuracy by up to 13% for fault segmentation and 5% for cell delineation. In contrast, large-scale features such as seismic facies or tissue regions see little benefit, underscoring that the value of SSL depends critically on the scale of the target objects. Our findings highlight the need to align SSL design with object size and sparsity, offering a general principle for buil ding more effective representation learning pipelines across scientific imaging domains.

  </details>



- **AGSP-DSA: An Adaptive Graph Signal Processing Framework for Robust Multimodal Fusion with Dynamic Semantic Alignment**  
  KV Karthikeya, Ashok Kumar Das, Shantanu Pal, Vivekananda Bhat K, Arun Sekar Rajasekaran  
  _2026-01-26_ · https://arxiv.org/abs/2601.18589v1  
  <details><summary>Abstract</summary>

  In this paper, we introduce an Adaptive Graph Signal Processing with Dynamic Semantic Alignment (AGSP DSA) framework to perform robust multimodal data fusion over heterogeneous sources, including text, audio, and images. The requested approach uses a dual-graph construction to learn both intra-modal and inter-modal relations, spectral graph filtering to boost the informative signals, and effective node embedding with Multi-scale Graph Convolutional Networks (GCNs). Semantic aware attention mechanism: each modality may dynamically contribute to the context with respect to contextual relevance. The experimental outcomes on three benchmark datasets, including CMU-MOSEI, AVE, and MM-IMDB, show that AGSP-DSA performs as the state of the art. More precisely, it achieves 95.3% accuracy, 0.936 F1-score, and 0.924 mAP on CMU-MOSEI, improving MM-GNN by 2.6 percent in accuracy. It gets 93.4% accuracy and 0.911 F1-score on AVE and 91.8% accuracy and 0.886 F1-score on MM-IMDB, which demonstrate good generalization and robustness in the missing modality setting. These findings verify the efficiency of AGSP-DSA in promoting multimodal learning in sentiment analysis, event recognition and multimedia classification.

  </details>



- **Self-Refining Video Sampling**  
  Sangwon Jang, Taekyung Ki, Jaehyeong Jo, Saining Xie, Jaehong Yoon, Sung Ju Hwang  
  _2026-01-26_ · https://arxiv.org/abs/2601.18577v1  
  <details><summary>Abstract</summary>

  Modern video generators still struggle with complex physical dynamics, often falling short of physical realism. Existing approaches address this using external verifiers or additional training on augmented data, which is computationally expensive and still limited in capturing fine-grained motion. In this work, we present self-refining video sampling, a simple method that uses a pre-trained video generator trained on large-scale datasets as its own self-refiner. By interpreting the generator as a denoising autoencoder, we enable iterative inner-loop refinement at inference time without any external verifier or additional training. We further introduce an uncertainty-aware refinement strategy that selectively refines regions based on self-consistency, which prevents artifacts caused by over-refinement. Experiments on state-of-the-art video generators demonstrate significant improvements in motion coherence and physics alignment, achieving over 70\% human preference compared to the default sampler and guidance-based sampler.

  </details>



- **From Cold Start to Active Learning: Embedding-Based Scan Selection for Medical Image Segmentation**  
  Devon Levy, Bar Assayag, Laura Gaspar, Ilan Shimshoni, Bella Specktor-Fadida  
  _2026-01-26_ · https://arxiv.org/abs/2601.18532v1  
  <details><summary>Abstract</summary>

  Accurate segmentation annotations are critical for disease monitoring, yet manual labeling remains a major bottleneck due to the time and expertise required. Active learning (AL) alleviates this burden by prioritizing informative samples for annotation, typically through a diversity-based cold-start phase followed by uncertainty-driven selection. We propose a novel cold-start sampling strategy that combines foundation-model embeddings with clustering, including automatic selection of the number of clusters and proportional sampling across clusters, to construct a diverse and representative initial training. This is followed by an uncertainty-based AL framework that integrates spatial diversity to guide sample selection. The proposed method is intuitive and interpretable, enabling visualization of the feature-space distribution of candidate samples. We evaluate our approach on three datasets spanning X-ray and MRI modalities. On the CheXmask dataset, the cold-start strategy outperforms random selection, improving Dice from 0.918 to 0.929 and reducing the Hausdorff distance from 32.41 to 27.66 mm. In the AL setting, combined entropy and diversity selection improves Dice from 0.919 to 0.939 and reduces the Hausdorff distance from 30.10 to 19.16 mm. On the Montgomery dataset, cold-start gains are substantial, with Dice improving from 0.928 to 0.950 and Hausdorff distance decreasing from 14.22 to 9.38 mm. On the SynthStrip dataset, cold-start selection slightly affects Dice but reduces the Hausdorff distance from 9.43 to 8.69 mm, while active learning improves Dice from 0.816 to 0.826 and reduces the Hausdorff distance from 7.76 to 6.38 mm. Overall, the proposed framework consistently outperforms baseline methods in low-data regimes, improving segmentation accuracy.

  </details>



- **DisasterInsight: A Multimodal Benchmark for Function-Aware and Grounded Disaster Assessment**  
  Sara Tehrani, Yonghao Xu, Leif Haglund, Amanda Berg, Michael Felsberg  
  _2026-01-26_ · https://arxiv.org/abs/2601.18493v1  
  <details><summary>Abstract</summary>

  Timely interpretation of satellite imagery is critical for disaster response, yet existing vision-language benchmarks for remote sensing largely focus on coarse labels and image-level recognition, overlooking the functional understanding and instruction robustness required in real humanitarian workflows. We introduce DisasterInsight, a multimodal benchmark designed to evaluate vision-language models (VLMs) on realistic disaster analysis tasks. DisasterInsight restructures the xBD dataset into approximately 112K building-centered instances and supports instruction-diverse evaluation across multiple tasks, including building-function classification, damage-level and disaster-type classification, counting, and structured report generation aligned with humanitarian assessment guidelines. To establish domain-adapted baselines, we propose DI-Chat, obtained by fine-tuning existing VLM backbones on disaster-specific instruction data using parameter-efficient Low-Rank Adaptation (LoRA). Extensive experiments on state-of-the-art generic and remote-sensing VLMs reveal substantial performance gaps across tasks, particularly in damage understanding and structured report generation. DI-Chat achieves significant improvements on damage-level and disaster-type classification as well as report generation quality, while building-function classification remains challenging for all evaluated models. DisasterInsight provides a unified benchmark for studying grounded multimodal reasoning in disaster imagery.

  </details>



- **AgentDoG: A Diagnostic Guardrail Framework for AI Agent Safety and Security**  
  Dongrui Liu, Qihan Ren, Chen Qian, Shuai Shao, Yuejin Xie, Yu Li, Zhonghao Yang, Haoyu Luo, Peng Wang, Qingyu Liu, et al.  
  _2026-01-26_ · https://arxiv.org/abs/2601.18491v1  
  <details><summary>Abstract</summary>

  The rise of AI agents introduces complex safety and security challenges arising from autonomous tool use and environmental interactions. Current guardrail models lack agentic risk awareness and transparency in risk diagnosis. To introduce an agentic guardrail that covers complex and numerous risky behaviors, we first propose a unified three-dimensional taxonomy that orthogonally categorizes agentic risks by their source (where), failure mode (how), and consequence (what). Guided by this structured and hierarchical taxonomy, we introduce a new fine-grained agentic safety benchmark (ATBench) and a Diagnostic Guardrail framework for agent safety and security (AgentDoG). AgentDoG provides fine-grained and contextual monitoring across agent trajectories. More Crucially, AgentDoG can diagnose the root causes of unsafe actions and seemingly safe but unreasonable actions, offering provenance and transparency beyond binary labels to facilitate effective agent alignment. AgentDoG variants are available in three sizes (4B, 7B, and 8B parameters) across Qwen and Llama model families. Extensive experimental results demonstrate that AgentDoG achieves state-of-the-art performance in agentic safety moderation in diverse and complex interactive scenarios. All models and datasets are openly released.

  </details>



- **3DGesPolicy: Phoneme-Aware Holistic Co-Speech Gesture Generation Based on Action Control**  
  Xuanmeng Sha, Liyun Zhang, Tomohiro Mashita, Naoya Chiba, Yuki Uranishi  
  _2026-01-26_ · https://arxiv.org/abs/2601.18451v1  
  <details><summary>Abstract</summary>

  Generating holistic co-speech gestures that integrate full-body motion with facial expressions suffers from semantically incoherent coordination on body motion and spatially unstable meaningless movements due to existing part-decomposed or frame-level regression methods, We introduce 3DGesPolicy, a novel action-based framework that reformulates holistic gesture generation as a continuous trajectory control problem through diffusion policy from robotics. By modeling frame-to-frame variations as unified holistic actions, our method effectively learns inter-frame holistic gesture motion patterns and ensures both spatially and semantically coherent movement trajectories that adhere to realistic motion manifolds. To further bridge the gap in expressive alignment, we propose a Gesture-Audio-Phoneme (GAP) fusion module that can deeply integrate and refine multi-modal signals, ensuring structured and fine-grained alignment between speech semantics, body motion, and facial expressions. Extensive quantitative and qualitative experiments on the BEAT2 dataset demonstrate the effectiveness of our 3DGesPolicy across other state-of-the-art methods in generating natural, expressive, and highly speech-aligned holistic gestures.

  </details>



- **Comparative Evaluation of Machine Learning Algorithms for Affective State Recognition from Children's Drawings**  
  Aura Loredana Dan  
  _2026-01-26_ · https://arxiv.org/abs/2601.18414v1  
  <details><summary>Abstract</summary>

  Autism spectrum disorder (ASD) represents a neurodevelopmental condition characterized by difficulties in expressing emotions and communication, particularly during early childhood. Understanding the affective state of children at an early age remains challenging, as conventional assessment methods are often intrusive, subjective, or difficult to apply consistently. This paper builds upon previous work on affective state recognition from children's drawings by presenting a comparative evaluation of machine learning models for emotion classification. Three deep learning architectures -- MobileNet, EfficientNet, and VGG16 -- are evaluated within a unified experimental framework to analyze classification performance, robustness, and computational efficiency. The models are trained using transfer learning on a dataset of children's drawings annotated with emotional labels provided by psychological experts. The results highlight important trade-offs between lightweight and deeper architectures when applied to drawing-based affective computing tasks, particularly in mobile and real-time application contexts.

  </details>



- **Noise-Robust AV-ASR Using Visual Features Both in the Whisper Encoder and Decoder**  
  Zhengyang Li, Thomas Graave, Björn Möller, Zehang Wu, Matthias Franz, Tim Fingscheidt  
  _2026-01-26_ · https://arxiv.org/abs/2601.18396v1  
  <details><summary>Abstract</summary>

  In audiovisual automatic speech recognition (AV-ASR) systems, information fusion of visual features in a pre-trained ASR has been proven as a promising method to improve noise robustness. In this work, based on the prominent Whisper ASR, first, we propose a simple and effective visual fusion method -- use of visual features both in encoder and decoder (dual-use) -- to learn the audiovisual interactions in the encoder and to weigh modalities in the decoder. Second, we compare visual fusion methods in Whisper models of various sizes. Our proposed dual-use method shows consistent noise robustness improvement, e.g., a 35% relative improvement (WER: 4.41% vs. 6.83%) based on Whisper small, and a 57% relative improvement (WER: 4.07% vs. 9.53%) based on Whisper medium, compared to typical reference middle fusion in babble noise with a signal-to-noise ratio (SNR) of 0dB. Third, we conduct ablation studies examining the impact of various module designs and fusion options. Fine-tuned on 1929 hours of audiovisual data, our dual-use method using Whisper medium achieves 4.08% (MUSAN babble noise) and 4.43% (NoiseX babble noise) average WER across various SNRs, thereby establishing a new state-of-the-art in noisy conditions on the LRS3 AV-ASR benchmark. Our code is at https://github.com/ifnspaml/Dual-Use-AVASR

  </details>



- **Gaze Prediction in Virtual Reality Without Eye Tracking Using Visual and Head Motion Cues**  
  Christos Petrou, Harris Partaourides, Athanasios Balomenos, Yannis Kopsinis, Sotirios Chatzis  
  _2026-01-26_ · https://arxiv.org/abs/2601.18372v1  
  <details><summary>Abstract</summary>

  Gaze prediction plays a critical role in Virtual Reality (VR) applications by reducing sensor-induced latency and enabling computationally demanding techniques such as foveated rendering, which rely on anticipating user attention. However, direct eye tracking is often unavailable due to hardware limitations or privacy concerns. To address this, we present a novel gaze prediction framework that combines Head-Mounted Display (HMD) motion signals with visual saliency cues derived from video frames. Our method employs UniSal, a lightweight saliency encoder, to extract visual features, which are then fused with HMD motion data and processed through a time-series prediction module. We evaluate two lightweight architectures, TSMixer and LSTM, for forecasting future gaze directions. Experiments on the EHTask dataset, along with deployment on commercial VR hardware, show that our approach consistently outperforms baselines such as Center-of-HMD and Mean Gaze. These results demonstrate the effectiveness of predictive gaze modeling in reducing perceptual lag and enhancing natural interaction in VR environments where direct eye tracking is constrained.

  </details>



- **Q-Bench-Portrait: Benchmarking Multimodal Large Language Models on Portrait Image Quality Perception**  
  Sijing Wu, Yunhao Li, Zicheng Zhang, Qi Jia, Xinyue Li, Huiyu Duan, Xiongkuo Min, Guangtao Zhai  
  _2026-01-26_ · https://arxiv.org/abs/2601.18346v1  
  <details><summary>Abstract</summary>

  Recent advances in multimodal large language models (MLLMs) have demonstrated impressive performance on existing low-level vision benchmarks, which primarily focus on generic images. However, their capabilities to perceive and assess portrait images, a domain characterized by distinct structural and perceptual properties, remain largely underexplored. To this end, we introduce Q-Bench-Portrait, the first holistic benchmark specifically designed for portrait image quality perception, comprising 2,765 image-question-answer triplets and featuring (1) diverse portrait image sources, including natural, synthetic distortion, AI-generated, artistic, and computer graphics images; (2) comprehensive quality dimensions, covering technical distortions, AIGC-specific distortions, and aesthetics; and (3) a range of question formats, including single-choice, multiple-choice, true/false, and open-ended questions, at both global and local levels. Based on Q-Bench-Portrait, we evaluate 20 open-source and 5 closed-source MLLMs, revealing that although current models demonstrate some competence in portrait image perception, their performance remains limited and imprecise, with a clear gap relative to human judgments. We hope that the proposed benchmark will foster further research into enhancing the portrait image perception capabilities of both general-purpose and domain-specific MLLMs.

  </details>



- **Beyond Rigid: Benchmarking Non-Rigid Video Editing**  
  Bingzheng Qu, Kehai Chen, Xuefeng Bai, Jun Yu, Min Zhang  
  _2026-01-26_ · https://arxiv.org/abs/2601.18340v1  
  <details><summary>Abstract</summary>

  Despite the remarkable progress in text-driven video editing, generating coherent non-rigid deformations remains a critical challenge, often plagued by physical distortion and temporal flicker. To bridge this gap, we propose NRVBench, the first dedicated and comprehensive benchmark designed to evaluate non-rigid video editing. First, we curate a high-quality dataset consisting of 180 non-rigid motion videos from six physics-based categories, equipped with 2,340 fine-grained task instructions and 360 multiple-choice questions. Second, we propose NRVE-Acc, a novel evaluation metric based on Vision-Language Models that can rigorously assess physical compliance, temporal consistency, and instruction alignment, overcoming the limitations of general metrics in capturing complex dynamics. Third, we introduce a training-free baseline, VM-Edit, which utilizes a dual-region denoising mechanism to achieve structure-aware control, balancing structural preservation and dynamic deformation. Extensive experiments demonstrate that while current methods have shortcomings in maintaining physical plausibility, our method achieves excellent performance across both standard and proposed metrics. We believe the benchmark could serve as a standard testing platform for advancing physics-aware video editing.

  </details>



- **A Tumor Aware DenseNet Swin Hybrid Learning with Boosted and Hierarchical Feature Spaces for Large-Scale Brain MRI Classification**  
  Muhammad Ali Shah, Muhammad Mansoor Alam, Saddam Hussain Khan  
  _2026-01-26_ · https://arxiv.org/abs/2601.18330v1  
  <details><summary>Abstract</summary>

  This study proposes an efficient Densely Swin Hybrid (EDSH) framework for brain tumor MRI analysis, designed to jointly capture fine grained texture patterns and long range contextual dependencies. Two tumor aware experimental setups are introduced to address class-specific diagnostic challenges. The first setup employs a Boosted Feature Space (BFS), where independently customized DenseNet and Swint branches learn complementary local and global representations that are dimension aligned, fused, and boosted, enabling highly sensitive detection of diffuse glioma patterns by successfully learning the features of irregular shape, poorly defined mass, and heterogeneous texture. The second setup adopts a hierarchical DenseNet Swint architecture with Deep Feature Extraction have Dual Residual connections (DFE and DR), in which DenseNet serves as a stem CNN for structured local feature learning, while Swin_t models global tumor morphology, effectively suppressing false negatives in meningioma and pituitary tumor classification by learning the features of well defined mass, location (outside brain) and enlargments in tumors (dural tail or upward extension). DenseNet is customized at the input level to match MRI spatial characteristics, leveraging dense residual connectivity to preserve texture information and mitigate vanishing-gradient effects. In parallel, Swint is tailored through task aligned patch embedding and shifted-window self attention to efficiently capture hierarchical global dependencies. Extensive evaluation on a large-scale MRI dataset (stringent 40,260 images across four tumor classes) demonstrates consistent superiority over standalone CNNs, Vision Transformers, and hybrids, achieving 98.50 accuracy and recall on the test unseen dataset.

  </details>



- **Integrating Fine-Grained Audio-Visual Evidence for Robust Multimodal Emotion Reasoning**  
  Zhixian Zhao, Wenjie Tian, Xiaohai Tian, Jun Zhang, Lei Xie  
  _2026-01-26_ · https://arxiv.org/abs/2601.18321v1  
  <details><summary>Abstract</summary>

  Multimodal emotion analysis is shifting from static classification to generative reasoning. Beyond simple label prediction, robust affective reasoning must synthesize fine-grained signals such as facial micro-expressions and prosodic which shifts to decode the latent causality within complex social contexts. However, current Multimodal Large Language Models (MLLMs) face significant limitations in fine-grained perception, primarily due to data scarcity and insufficient cross-modal fusion. As a result, these models often exhibit unimodal dominance which leads to hallucinations in complex multimodal interactions, particularly when visual and acoustic cues are subtle, ambiguous, or even contradictory (e.g., in sarcastic scenery). To address this, we introduce SABER-LLM, a framework designed for robust multimodal reasoning. First, we construct SABER, a large-scale emotion reasoning dataset comprising 600K video clips, annotated with a novel six-dimensional schema that jointly captures audiovisual cues and causal logic. Second, we propose the structured evidence decomposition paradigm, which enforces a "perceive-then-reason" separation between evidence extraction and reasoning to alleviate unimodal dominance. The ability to perceive complex scenes is further reinforced by consistency-aware direct preference optimization, which explicitly encourages alignment among modalities under ambiguous or conflicting perceptual conditions. Experiments on EMER, EmoBench-M, and SABER-Test demonstrate that SABER-LLM significantly outperforms open-source baselines and achieves robustness competitive with closed-source models in decoding complex emotional dynamics. The dataset and model are available at https://github.com/zxzhao0/SABER-LLM.

  </details>



- **SwipeGen: Bridging the Execution Gap in GUI Agents via Human-like Swipe Synthesis**  
  Xuan Wang, Siyuan Su, Quantong Fu, Yongxiang Hu, Yangfan Zhou  
  _2026-01-26_ · https://arxiv.org/abs/2601.18305v1  
  <details><summary>Abstract</summary>

  With the widespread adoption of Graphical User Interface (GUI) agents for automating GUI interaction tasks, substantial research focused on improving GUI perception to ground task instructions into concrete action steps. However, the step execution capability of these agents has gradually emerged as a new bottleneck for task completion. In particular, existing GUI agents often adopt overly simplified strategies for handling swipe interactions, preventing them from accurately replicating human-like behavior. To address this limitation, we decompose human swipe gestures into multiple quantifiable dimensions and propose an automated pipeline SwipeGen to synthesize human-like swipe interactions through GUI exploration. Based on this pipeline, we construct and release the first benchmark for evaluating the swipe execution capability of GUI agents. Furthermore, leveraging the synthesized data, we propose GUISwiper, a GUI agent with enhanced interaction execution capabilities. Experimental results demonstrate that GUISwiper achieves a swipe execution accuracy of 69.07%, representing a 214% improvement over existing VLM baselines.

  </details>



- **Contextual Range-View Projection for 3D LiDAR Point Clouds**  
  Seyedali Mousavi, Seyedhamidreza Mousavi, Masoud Daneshtalab  
  _2026-01-26_ · https://arxiv.org/abs/2601.18301v1  
  <details><summary>Abstract</summary>

  Range-view projection provides an efficient method for transforming 3D LiDAR point clouds into 2D range image representations, enabling effective processing with 2D deep learning models. However, a major challenge in this projection is the many-to-one conflict, where multiple 3D points are mapped onto the same pixel in the range image, requiring a selection strategy. Existing approaches typically retain the point with the smallest depth (closest to the LiDAR), disregarding semantic relevance and object structure, which leads to the loss of important contextual information. In this paper, we extend the depth-based selection rule by incorporating contextual information from both instance centers and class labels, introducing two mechanisms: \textit{Centerness-Aware Projection (CAP)} and \textit{Class-Weighted-Aware Projection (CWAP)}. In CAP, point depths are adjusted according to their distance from the instance center, thereby prioritizing central instance points over noisy boundary and background points. In CWAP, object classes are prioritized through user-defined weights, offering flexibility in the projection strategy. Our evaluations on the SemanticKITTI dataset show that CAP preserves more instance points during projection, achieving up to a 3.1\% mIoU improvement compared to the baseline. Furthermore, CWAP enhances the performance of targeted classes while having a negligible impact on the performance of other classes

  </details>



- **Depth to Anatomy: Learning Internal Organ Locations from Surface Depth Images**  
  Eytan Kats, Kai Geissler, Daniel Mensing, Jochen G. Hirsch, Stefan Heldman, Mattias P. Heinrich  
  _2026-01-26_ · https://arxiv.org/abs/2601.18260v1  
  <details><summary>Abstract</summary>

  Automated patient positioning plays an important role in optimizing scanning procedure and improving patient throughput. Leveraging depth information captured by RGB-D cameras presents a promising approach for estimating internal organ positions, thereby enabling more accurate and efficient positioning. In this work, we propose a learning-based framework that directly predicts the 3D locations and shapes of multiple internal organs from single 2D depth images of the body surface. Utilizing a large-scale dataset of full-body MRI scans, we synthesize depth images paired with corresponding anatomical segmentations to train a unified convolutional neural network architecture. Our method accurately localizes a diverse set of anatomical structures, including bones and soft tissues, without requiring explicit surface reconstruction. Experimental results demonstrate the potential of integrating depth sensors into radiology workflows to streamline scanning procedures and enhance patient experience through automated patient positioning.

  </details>



- **A multimodal vision foundation model for generalizable knee pathology**  
  Kang Yu, Dingyu Wang, Zimu Yuan, Nan Zhou, Jiajun Liu, Jiaxin Liu, Shanggui Liu, Yaoyan Zheng, Huishu Yuan, Di Huang, et al.  
  _2026-01-26_ · https://arxiv.org/abs/2601.18250v1  
  <details><summary>Abstract</summary>

  Musculoskeletal disorders represent a leading cause of global disability, creating an urgent demand for precise interpretation of medical imaging. Current artificial intelligence (AI) approaches in orthopedics predominantly rely on task-specific, supervised learning paradigms. These methods are inherently fragmented, require extensive annotated datasets, and often lack generalizability across different modalities and clinical scenarios. The development of foundation models in this field has been constrained by the scarcity of large-scale, curated, and open-source musculoskeletal datasets. To address these challenges, we introduce OrthoFoundation, a multimodal vision foundation model optimized for musculoskeletal pathology. We constructed a pre-training dataset of 1.2 million unlabeled knee X-ray and MRI images from internal and public databases. Utilizing a Dinov3 backbone, the model was trained via self-supervised contrastive learning to capture robust radiological representations. OrthoFoundation achieves state-of-the-art (SOTA) performance across 14 downstream tasks. It attained superior accuracy in X-ray osteoarthritis diagnosis and ranked first in MRI structural injury detection. The model demonstrated remarkable label efficiency, matching supervised baselines using only 50% of labeled data. Furthermore, despite being pre-trained on knee images, OrthoFoundation exhibited exceptional cross-anatomy generalization to the hip, shoulder, and ankle. OrthoFoundation represents a significant advancement toward general-purpose AI for musculoskeletal imaging. By learning fundamental, joint-agnostic radiological semantics from large-scale multimodal data, it overcomes the limitations of conventional models, which provides a robust framework for reducing annotation burdens and enhancing diagnostic accuracy in clinical practice.

  </details>



- **Facial Emotion Recognition on FER-2013 using an EfficientNetB2-Based Approach**  
  Sahil Naik, Soham Bagayatkar, Pavankumar Singh  
  _2026-01-26_ · https://arxiv.org/abs/2601.18228v1  
  <details><summary>Abstract</summary>

  Detection of human emotions based on facial images in real-world scenarios is a difficult task due to low image quality, variations in lighting, pose changes, background distractions, small inter-class variations, noisy crowd-sourced labels, and severe class imbalance, as observed in the FER-2013 dataset of 48x48 grayscale images. Although recent approaches using large CNNs such as VGG and ResNet achieve reasonable accuracy, they are computationally expensive and memory-intensive, limiting their practicality for real-time applications. We address these challenges using a lightweight and efficient facial emotion recognition pipeline based on EfficientNetB2, trained using a two-stage warm-up and fine-tuning strategy. The model is enhanced with AdamW optimization, decoupled weight decay, label smoothing (epsilon = 0.06) to reduce annotation noise, and clipped class weights to mitigate class imbalance, along with dropout, mixed-precision training, and extensive real-time data augmentation. The model is trained using a stratified 87.5%/12.5% train-validation split while keeping the official test set intact, achieving a test accuracy of 68.78% with nearly ten times fewer parameters than VGG16-based baselines. Experimental results, including per-class metrics and learning dynamics, demonstrate stable training and strong generalization, making the proposed approach suitable for real-time and edge-based applications.

  </details>



- **QualiRAG: Retrieval-Augmented Generation for Visual Quality Understanding**  
  Linhan Cao, Wei Sun, Weixia Zhang, Xiangyang Zhu, Kaiwei Zhang, Jun Jia, Dandan Zhu, Guangtao Zhai, Xiongkuo Min  
  _2026-01-26_ · https://arxiv.org/abs/2601.18195v1  
  <details><summary>Abstract</summary>

  Visual quality assessment (VQA) is increasingly shifting from scalar score prediction toward interpretable quality understanding -- a paradigm that demands \textit{fine-grained spatiotemporal perception} and \textit{auxiliary contextual information}. Current approaches rely on supervised fine-tuning or reinforcement learning on curated instruction datasets, which involve labor-intensive annotation and are prone to dataset-specific biases. To address these challenges, we propose \textbf{QualiRAG}, a \textit{training-free} \textbf{R}etrieval-\textbf{A}ugmented \textbf{G}eneration \textbf{(RAG)} framework that systematically leverages the latent perceptual knowledge of large multimodal models (LMMs) for visual quality perception. Unlike conventional RAG that retrieves from static corpora, QualiRAG dynamically generates auxiliary knowledge by decomposing questions into structured requests and constructing four complementary knowledge sources: \textit{visual metadata}, \textit{subject localization}, \textit{global quality summaries}, and \textit{local quality descriptions}, followed by relevance-aware retrieval for evidence-grounded reasoning. Extensive experiments show that QualiRAG achieves substantial improvements over open-source general-purpose LMMs and VQA-finetuned LMMs on visual quality understanding tasks, and delivers competitive performance on visual quality comparison tasks, demonstrating robust quality assessment capabilities without any task-specific training. The code will be publicly available at https://github.com/clh124/QualiRAG.

  </details>



- **YOLO-DS: Fine-Grained Feature Decoupling via Dual-Statistic Synergy Operator for Object Detection**  
  Lin Huang, Yujuan Tan, Weisheng Li, Shitai Shan, Liu Liu, Bo Liu, Linlin Shen, Jing Yu, Yue Niu  
  _2026-01-26_ · https://arxiv.org/abs/2601.18172v1  
  <details><summary>Abstract</summary>

  One-stage object detection, particularly the YOLO series, strikes a favorable balance between accuracy and efficiency. However, existing YOLO detectors lack explicit modeling of heterogeneous object responses within shared feature channels, which limits further performance gains. To address this, we propose YOLO-DS, a framework built around a novel Dual-Statistic Synergy Operator (DSO). The DSO decouples object features by jointly modeling the channel-wise mean and the peak-to-mean difference. Building upon the DSO, we design two lightweight gating modules: the Dual-Statistic Synergy Gating (DSG) module for adaptive channel-wise feature selection, and the Multi-Path Segmented Gating (MSG) module for depth-wise feature weighting. On the MS-COCO benchmark, YOLO-DS consistently outperforms YOLOv8 across five model scales (N, S, M, L, X), achieving AP gains of 1.1% to 1.7% with only a minimal increase in inference latency. Extensive visualization, ablation, and comparative studies validate the effectiveness of our approach, demonstrating its superior capability in discriminating heterogeneous objects with high efficiency.

  </details>



- **Beyond Static Datasets: Robust Offline Policy Optimization via Vetted Synthetic Transitions**  
  Pedram Agand, Mo Chen  
  _2026-01-26_ · https://arxiv.org/abs/2601.18107v1  
  <details><summary>Abstract</summary>

  Offline Reinforcement Learning (ORL) holds immense promise for safety-critical domains like industrial robotics, where real-time environmental interaction is often prohibitive. A primary obstacle in ORL remains the distributional shift between the static dataset and the learned policy, which typically mandates high degrees of conservatism that can restrain potential policy improvements. We present MoReBRAC, a model-based framework that addresses this limitation through Uncertainty-Aware latent synthesis. Instead of relying solely on the fixed data, MoReBRAC utilizes a dual-recurrent world model to synthesize high-fidelity transitions that augment the training manifold. To ensure the reliability of this synthetic data, we implement a hierarchical uncertainty pipeline integrating Variational Autoencoder (VAE) manifold detection, model sensitivity analysis, and Monte Carlo (MC) dropout. This multi-layered filtering process guarantees that only transitions residing within high-confidence regions of the learned dynamics are utilized. Our results on D4RL Gym-MuJoCo benchmarks reveal significant performance gains, particularly in ``random'' and ``suboptimal'' data regimes. We further provide insights into the role of the VAE as a geometric anchor and discuss the distributional trade-offs encountered when learning from near-optimal datasets.

  </details>



- **Spatial-Conditioned Reasoning in Long-Egocentric Videos**  
  James Tribble, Hao Wang, Si-En Hong, Chaoyi Zhou, Ashish Bastola, Siyu Huang, Abolfazl Razi  
  _2026-01-26_ · https://arxiv.org/abs/2601.18100v1  
  <details><summary>Abstract</summary>

  Long-horizon egocentric video presents significant challenges for visual navigation due to viewpoint drift and the absence of persistent geometric context. Although recent vision-language models perform well on image and short-video reasoning, their spatial reasoning capability in long egocentric sequences remains limited. In this work, we study how explicit spatial signals influence VLM-based video understanding without modifying model architectures or inference procedures. We introduce Sanpo-D, a fine-grained re-annotation of the Google Sanpo dataset, and benchmark multiple VLMs on navigation-oriented spatial queries. To examine input-level inductive bias, we further fuse depth maps with RGB frames and evaluate their impact on spatial reasoning. Our results reveal a trade-off between general-purpose accuracy and spatial specialization, showing that depth-aware and spatially grounded representations can improve performance on safety-critical tasks such as pedestrian and obstruction detection.

  </details>



- **Semi-Supervised Hyperspectral Image Classification with Edge-Aware Superpixel Label Propagation and Adaptive Pseudo-Labeling**  
  Yunfei Qiu, Qiqiong Ma, Tianhua Lv, Li Fang, Shudong Zhou, Wei Yao  
  _2026-01-26_ · https://arxiv.org/abs/2601.18049v1  
  <details><summary>Abstract</summary>

  Significant progress has been made in semi-supervised hyperspectral image (HSI) classification regarding feature extraction and classification performance. However, due to high annotation costs and limited sample availability, semi-supervised learning still faces challenges such as boundary label diffusion and pseudo-label instability. To address these issues, this paper proposes a novel semi-supervised hyperspectral classification framework integrating spatial prior information with a dynamic learning mechanism. First, we design an Edge-Aware Superpixel Label Propagation (EASLP) module. By integrating edge intensity penalty with neighborhood correction strategy, it mitigates label diffusion from superpixel segmentation while enhancing classification robustness in boundary regions. Second, we introduce a Dynamic History-Fused Prediction (DHP) method. By maintaining historical predictions and dynamically weighting them with current results, DHP smoothens pseudo-label fluctuations and improves temporal consistency and noise resistance. Concurrently, incorporating condifence and consistency measures, the Adaptive Tripartite Sample Categorization (ATSC) strategy implements hierarchical utilization of easy, ambiguous, and hard samples, leading to enhanced pseudo-label quality and learning efficiency. The Dynamic Reliability-Enhanced Pseudo-Label Framework (DREPL), composed of DHP and ATSC, strengthens pseudo-label stability across temporal and sample domains. Through synergizes operation with EASLP, it achieves spatio-temporal consistency optimization. Evaluations on four benchmark datasets demonstrate its capability to maintain superior classification performance.

  </details>



- **MorphXAI: An Explainable Framework for Morphological Analysis of Parasites in Blood Smear Images**  
  Aqsa Yousaf, Sint Sint Win, Megan Coffee, Habeeb Olufowobi  
  _2026-01-25_ · https://arxiv.org/abs/2601.18001v1  
  <details><summary>Abstract</summary>

  Parasitic infections remain a pressing global health challenge, particularly in low-resource settings where diagnosis still depends on labor-intensive manual inspection of blood smears and the availability of expert domain knowledge. While deep learning models have shown strong performance in automating parasite detection, their clinical usefulness is constrained by limited interpretability. Existing explainability methods are largely restricted to visual heatmaps or attention maps, which highlight regions of interest but fail to capture the morphological traits that clinicians rely on for diagnosis. In this work, we present MorphXAI, an explainable framework that unifies parasite detection with fine-grained morphological analysis. MorphXAI integrates morphological supervision directly into the prediction pipeline, enabling the model to localize parasites while simultaneously characterizing clinically relevant attributes such as shape, curvature, visible dot count, flagellum presence, and developmental stage. To support this task, we curate a clinician-annotated dataset of three parasite species (Leishmania, Trypanosoma brucei, and Trypanosoma cruzi) with detailed morphological labels, establishing a new benchmark for interpretable parasite analysis. Experimental results show that MorphXAI not only improves detection performance over the baseline but also provides structured, biologically meaningful explanations.

  </details>



- **RemEdit: Efficient Diffusion Editing with Riemannian Geometry**  
  Eashan Adhikarla, Brian D. Davison  
  _2026-01-25_ · https://arxiv.org/abs/2601.17927v1  
  <details><summary>Abstract</summary>

  Controllable image generation is fundamental to the success of modern generative AI, yet it faces a critical trade-off between semantic fidelity and inference speed. The RemEdit diffusion-based framework addresses this trade-off with two synergistic innovations. First, for editing fidelity, we navigate the latent space as a Riemannian manifold. A mamba-based module efficiently learns the manifold's structure, enabling direct and accurate geodesic path computation for smooth semantic edits. This control is further refined by a dual-SLERP blending technique and a goal-aware prompt enrichment pass from a Vision-Language Model. Second, for additional acceleration, we introduce a novel task-specific attention pruning mechanism. A lightweight pruning head learns to retain tokens essential to the edit, enabling effective optimization without the semantic degradation common in content-agnostic approaches. RemEdit surpasses prior state-of-the-art editing frameworks while maintaining real-time performance under 50% pruning. Consequently, RemEdit establishes a new benchmark for practical and powerful image editing. Source code: https://www.github.com/eashanadhikarla/RemEdit.

  </details>



- **Feature-Space Generative Models for One-Shot Class-Incremental Learning**  
  Jack Foster, Kirill Paramonov, Mete Ozay, Umberto Michieli  
  _2026-01-25_ · https://arxiv.org/abs/2601.17905v1  
  <details><summary>Abstract</summary>

  Few-shot class-incremental learning (FSCIL) is a paradigm where a model, initially trained on a dataset of base classes, must adapt to an expanding problem space by recognizing novel classes with limited data. We focus on the challenging FSCIL setup where a model receives only a single sample (1-shot) for each novel class and no further training or model alterations are allowed after the base training phase. This makes generalization to novel classes particularly difficult. We propose a novel approach predicated on the hypothesis that base and novel class embeddings have structural similarity. We map the original embedding space into a residual space by subtracting the class prototype (i.e., the average class embedding) of input samples. Then, we leverage generative modeling with VAE or diffusion models to learn the multi-modal distribution of residuals over the base classes, and we use this as a valuable structural prior to improve recognition of novel classes. Our approach, Gen1S, consistently improves novel class recognition over the state of the art across multiple benchmarks and backbone architectures.

  </details>



- **EEG Foundation Models: Progresses, Benchmarking, and Open Problems**  
  Dingkun Liu, Yuheng Chen, Zhu Chen, Zhenyao Cui, Yaozhi Wen, Jiayu An, Jingwei Luo, Dongrui Wu  
  _2026-01-25_ · https://arxiv.org/abs/2601.17883v1  
  <details><summary>Abstract</summary>

  Electroencephalography (EEG) foundation models have recently emerged as a promising paradigm for brain-computer interfaces (BCIs), aiming to learn transferable neural representations from large-scale heterogeneous recordings. Despite rapid progresses, there lacks fair and comprehensive comparisons of existing EEG foundation models, due to inconsistent pre-training objectives, preprocessing choices, and downstream evaluation protocols. This paper fills this gap. We first review 50 representative models and organize their design choices into a unified taxonomic framework including data standardization, model architectures, and self-supervised pre-training strategies. We then evaluate 12 open-source foundation models and competitive specialist baselines across 13 EEG datasets spanning nine BCI paradigms. Emphasizing real-world deployments, we consider both cross-subject generalization under a leave-one-subject-out protocol and rapid calibration under a within-subject few-shot setting. We further compare full-parameter fine-tuning with linear probing to assess the transferability of pre-trained representations, and examine the relationship between model scale and downstream performance. Our results indicate that: 1) linear probing is frequently insufficient; 2) specialist models trained from scratch remain competitive across many tasks; and, 3) larger foundation models do not necessarily yield better generalization performance under current data regimes and training practices.

  </details>



- **Quran-MD: A Fine-Grained Multilingual Multimodal Dataset of the Quran**  
  Muhammad Umar Salman, Mohammad Areeb Qazi, Mohammed Talha Alam  
  _2026-01-25_ · https://arxiv.org/abs/2601.17880v1  
  <details><summary>Abstract</summary>

  We present Quran MD, a comprehensive multimodal dataset of the Quran that integrates textual, linguistic, and audio dimensions at the verse and word levels. For each verse (ayah), the dataset provides its original Arabic text, English translation, and phonetic transliteration. To capture the rich oral tradition of Quranic recitation, we include verse-level audio from 32 distinct reciters, reflecting diverse recitation styles and dialectical nuances. At the word level, each token is paired with its corresponding Arabic script, English translation, transliteration, and an aligned audio recording, allowing fine-grained analysis of pronunciation, phonology, and semantic context. This dataset supports various applications, including natural language processing, speech recognition, text-to-speech synthesis, linguistic analysis, and digital Islamic studies. Bridging text and audio modalities across multiple reciters, this dataset provides a unique resource to advance computational approaches to Quranic recitation and study. Beyond enabling tasks such as ASR, tajweed detection, and Quranic TTS, it lays the foundation for multimodal embeddings, semantic retrieval, style transfer, and personalized tutoring systems that can support both research and community applications. The dataset is available at https://huggingface.co/datasets/Buraaq/quran-audio-text-dataset

  </details>



- **MV-SAM: Multi-view Promptable Segmentation using Pointmap Guidance**  
  Yoonwoo Jeong, Cheng Sun, Yu-Chiang Frank Wang, Minsu Cho, Jaesung Choe  
  _2026-01-25_ · https://arxiv.org/abs/2601.17866v1  
  <details><summary>Abstract</summary>

  Promptable segmentation has emerged as a powerful paradigm in computer vision, enabling users to guide models in parsing complex scenes with prompts such as clicks, boxes, or textual cues. Recent advances, exemplified by the Segment Anything Model (SAM), have extended this paradigm to videos and multi-view images. However, the lack of 3D awareness often leads to inconsistent results, necessitating costly per-scene optimization to enforce 3D consistency. In this work, we introduce MV-SAM, a framework for multi-view segmentation that achieves 3D consistency using pointmaps -- 3D points reconstructed from unposed images by recent visual geometry models. Leveraging the pixel-point one-to-one correspondence of pointmaps, MV-SAM lifts images and prompts into 3D space, eliminating the need for explicit 3D networks or annotated 3D data. Specifically, MV-SAM extends SAM by lifting image embeddings from its pretrained encoder into 3D point embeddings, which are decoded by a transformer using cross-attention with 3D prompt embeddings. This design aligns 2D interactions with 3D geometry, enabling the model to implicitly learn consistent masks across views through 3D positional embeddings. Trained on the SA-1B dataset, our method generalizes well across domains, outperforming SAM2-Video and achieving comparable performance with per-scene optimization baselines on NVOS, SPIn-NeRF, ScanNet++, uCo3D, and DL3DV benchmarks. Code will be released.

  </details>



- **Less Is More: Scalable Visual Navigation from Limited Data**  
  Yves Inglin, Jonas Frey, Changan Chen, Marco Hutter  
  _2026-01-25_ · https://arxiv.org/abs/2601.17815v1  
  <details><summary>Abstract</summary>

  Imitation learning provides a powerful framework for goal-conditioned visual navigation in mobile robots, enabling obstacle avoidance while respecting human preferences and social norms. However, its effectiveness depends critically on the quality and diversity of training data. In this work, we show how classical geometric planners can be leveraged to generate synthetic trajectories that complement costly human demonstrations. We train Less is More (LiMo), a transformer-based visual navigation policy that predicts goal-conditioned SE(2) trajectories from a single RGB observation, and find that augmenting limited expert demonstrations with planner-generated supervision yields substantial performance gains. Through ablations and complementary qualitative and quantitative analyses, we characterize how dataset scale and diversity affect planning performance. We demonstrate real-robot deployment and argue that robust visual navigation is enabled not by simply collecting more demonstrations, but by strategically curating diverse, high-quality datasets. Our results suggest that scalable, embodiment-specific geometric supervision is a practical path toward data-efficient visual navigation.

  </details>



- **MV-S2V: Multi-View Subject-Consistent Video Generation**  
  Ziyang Song, Xinyu Gong, Bangya Liu, Zelin Zhao  
  _2026-01-25_ · https://arxiv.org/abs/2601.17756v1  
  <details><summary>Abstract</summary>

  Existing Subject-to-Video Generation (S2V) methods have achieved high-fidelity and subject-consistent video generation, yet remain constrained to single-view subject references. This limitation renders the S2V task reducible to an S2I + I2V pipeline, failing to exploit the full potential of video subject control. In this work, we propose and address the challenging Multi-View S2V (MV-S2V) task, which synthesizes videos from multiple reference views to enforce 3D-level subject consistency. Regarding the scarcity of training data, we first develop a synthetic data curation pipeline to generate highly customized synthetic data, complemented by a small-scale real-world captured dataset to boost the training of MV-S2V. Another key issue lies in the potential confusion between cross-subject and cross-view references in conditional generation. To overcome this, we further introduce Temporally Shifted RoPE (TS-RoPE) to distinguish between different subjects and distinct views of the same subject in reference conditioning. Our framework achieves superior 3D subject consistency w.r.t. multi-view reference images and high-quality visual outputs, establishing a new meaningful direction for subject-driven video generation. Our project page is available at <a href="https://szy-young.github.io/mv-s2v">this URL</a>

  </details>



- **Bridging Supervision Gaps: A Unified Framework for Remote Sensing Change Detection**  
  Kaixuan Jiang, Chen Wu, Zhenghui Zhao, Chengxi Han  
  _2026-01-25_ · https://arxiv.org/abs/2601.17747v1  
  <details><summary>Abstract</summary>

  Change detection (CD) aims to identify surface changes from multi-temporal remote sensing imagery. In real-world scenarios, Pixel-level change labels are expensive to acquire, and existing models struggle to adapt to scenarios with diverse annotation availability. To tackle this challenge, we propose a unified change detection framework (UniCD), which collaboratively handles supervised, weakly-supervised, and unsupervised tasks through a coupled architecture. UniCD eliminates architectural barriers through a shared encoder and multi-branch collaborative learning mechanism, achieving deep coupling of heterogeneous supervision signals. Specifically, UniCD consists of three supervision-specific branches. In the supervision branch, UniCD introduces the spatial-temporal awareness module (STAM), achieving efficient synergistic fusion of bi-temporal features. In the weakly-supervised branch, we construct change representation regularization (CRR), which steers model convergence from coarse-grained activations toward coherent and separable change modeling. In the unsupervised branch, we propose semantic prior-driven change inference (SPCI), which transforms unsupervised tasks into controlled weakly-supervised path optimization. Experiments on mainstream datasets demonstrate that UniCD achieves optimal performance across three tasks. It exhibits significant accuracy improvements in weakly and unsupervised scenarios, surpassing current state-of-the-art by 12.72% and 12.37% on LEVIR-CD, respectively.

  </details>



- **The Script is All You Need: An Agentic Framework for Long-Horizon Dialogue-to-Cinematic Video Generation**  
  Chenyu Mu, Xin He, Qu Yang, Wanshun Chen, Jiadi Yao, Huang Liu, Zihao Yi, Bo Zhao, Xingyu Chen, Ruotian Ma, et al.  
  _2026-01-25_ · https://arxiv.org/abs/2601.17737v1  
  <details><summary>Abstract</summary>

  Recent advances in video generation have produced models capable of synthesizing stunning visual content from simple text prompts. However, these models struggle to generate long-form, coherent narratives from high-level concepts like dialogue, revealing a ``semantic gap'' between a creative idea and its cinematic execution. To bridge this gap, we introduce a novel, end-to-end agentic framework for dialogue-to-cinematic-video generation. Central to our framework is ScripterAgent, a model trained to translate coarse dialogue into a fine-grained, executable cinematic script. To enable this, we construct ScriptBench, a new large-scale benchmark with rich multimodal context, annotated via an expert-guided pipeline. The generated script then guides DirectorAgent, which orchestrates state-of-the-art video models using a cross-scene continuous generation strategy to ensure long-horizon coherence. Our comprehensive evaluation, featuring an AI-powered CriticAgent and a new Visual-Script Alignment (VSA) metric, shows our framework significantly improves script faithfulness and temporal fidelity across all tested video models. Furthermore, our analysis uncovers a crucial trade-off in current SOTA models between visual spectacle and strict script adherence, providing valuable insights for the future of automated filmmaking.

  </details>



- **A Computational Approach to Visual Metonymy**  
  Saptarshi Ghosh, Linfeng Liu, Tianyu Jiang  
  _2026-01-25_ · https://arxiv.org/abs/2601.17706v1  
  <details><summary>Abstract</summary>

  Images often communicate more than they literally depict: a set of tools can suggest an occupation and a cultural artifact can suggest a tradition. This kind of indirect visual reference, known as visual metonymy, invites viewers to recover a target concept via associated cues rather than explicit depiction. In this work, we present the first computational investigation of visual metonymy. We introduce a novel pipeline grounded in semiotic theory that leverages large language models and text-to-image models to generate metonymic visual representations. Using this framework, we construct ViMET, the first visual metonymy dataset comprising 2,000 multiple-choice questions to evaluate the cognitive reasoning abilities in multimodal language models. Experimental results on our dataset reveal a significant gap between human performance (86.9%) and state-of-the-art vision-language models (65.9%), highlighting limitations in machines' ability to interpret indirect visual references. Our dataset is publicly available at: https://github.com/cincynlp/ViMET.

  </details>



- **An AI-enabled tool for quantifying overlapping red blood cell sickling dynamics in microfluidic assays**  
  Nikhil Kadivar, Guansheng Li, Jianlu Zheng, John M. Higgins, Ming Dao, George Em Karniadakis, Mengjia Xu  
  _2026-01-25_ · https://arxiv.org/abs/2601.17703v1  
  <details><summary>Abstract</summary>

  Understanding sickle cell dynamics requires accurate identification of morphological transitions under diverse biophysical conditions, particularly in densely packed and overlapping cell populations. Here, we present an automated deep learning framework that integrates AI-assisted annotation, segmentation, classification, and instance counting to quantify red blood cell (RBC) populations across varying density regimes in time-lapse microscopy data. Experimental images were annotated using the Roboflow platform to generate labeled dataset for training an nnU-Net segmentation model. The trained network enables prediction of the temporal evolution of the sickle cell fraction, while a watershed algorithm resolves overlapping cells to enhance quantification accuracy. Despite requiring only a limited amount of labeled data for training, the framework achieves high segmentation performance, effectively addressing challenges associated with scarce manual annotations and cell overlap. By quantitatively tracking dynamic changes in RBC morphology, this approach can more than double the experimental throughput via densely packed cell suspensions, capture drug-dependent sickling behavior, and reveal distinct mechanobiological signatures of cellular morphological evolution. Overall, this AI-driven framework establishes a scalable and reproducible computational platform for investigating cellular biomechanics and assessing therapeutic efficacy in microphysiological systems.

  </details>



- **StyleDecoupler: Generalizable Artistic Style Disentanglement**  
  Zexi Jia, Jinchao Zhang, Jie Zhou  
  _2026-01-25_ · https://arxiv.org/abs/2601.17697v1  
  <details><summary>Abstract</summary>

  Representing artistic style is challenging due to its deep entanglement with semantic content. We propose StyleDecoupler, an information-theoretic framework that leverages a key insight: multi-modal vision models encode both style and content, while uni-modal models suppress style to focus on content-invariant features. By using uni-modal representations as content-only references, we isolate pure style features from multi-modal embeddings through mutual information minimization. StyleDecoupler operates as a plug-and-play module on frozen Vision-Language Models without fine-tuning. We also introduce WeART, a large-scale benchmark of 280K artworks across 152 styles and 1,556 artists. Experiments show state-of-the-art performance on style retrieval across WeART and WikiART, while enabling applications like style relationship mapping and generative model evaluation. We release our method and dataset at this url.

  </details>



- **SPACE-CLIP: Spatial Perception via Adaptive CLIP Embeddings for Monocular Depth Estimation**  
  Taewan Cho, Taeryang Kim, Andrew Jaeyong Choi  
  _2026-01-25_ · https://arxiv.org/abs/2601.17657v1  
  <details><summary>Abstract</summary>

  Contrastive Language-Image Pre-training (CLIP) has accomplished extraordinary success for semantic understanding but inherently struggles to perceive geometric structure. Existing methods attempt to bridge this gap by querying CLIP with textual prompts, a process that is often indirect and inefficient. This paper introduces a fundamentally different approach using a dual-pathway decoder. We present SPACE-CLIP, an architecture that unlocks and interprets latent geometric knowledge directly from a frozen CLIP vision encoder, completely bypassing the text encoder and its associated textual prompts. A semantic pathway interprets high-level features, dynamically conditioned on global context using feature-wise linear modulation (FiLM). In addition, a structural pathway extracts fine-grained spatial details from early layers. These complementary streams are hierarchically fused, enabling a robust synthesis of semantic context and precise geometry. Extensive experiments on the KITTI benchmark show that SPACE-CLIP dramatically outperforms previous CLIP-based methods. Our ablation studies validate that the synergistic fusion of our dual pathways is critical to this success. SPACE-CLIP offers a new, efficient, and architecturally elegant blueprint for repurposing large-scale vision models. The proposed method is not just a standalone depth estimator, but a readily integrable spatial perception module for the next generation of embodied AI systems, such as vision-language-action (VLA) models. Our model is available at https://github.com/taewan2002/space-clip

  </details>



- **AVMeme Exam: A Multimodal Multilingual Multicultural Benchmark for LLMs' Contextual and Cultural Knowledge and Thinking**  
  Xilin Jiang, Qiaolin Wang, Junkai Wu, Xiaomin He, Zhongweiyang Xu, Yinghao Ma, Minshuo Piao, Kaiyi Yang, Xiuwen Zheng, Riki Shimizu, et al.  
  _2026-01-25_ · https://arxiv.org/abs/2601.17645v1  
  <details><summary>Abstract</summary>

  Internet audio-visual clips convey meaning through time-varying sound and motion, which extend beyond what text alone can represent. To examine whether AI models can understand such signals in human cultural contexts, we introduce AVMeme Exam, a human-curated benchmark of over one thousand iconic Internet sounds and videos spanning speech, songs, music, and sound effects. Each meme is paired with a unique Q&A assessing levels of understanding from surface content to context and emotion to usage and world knowledge, along with metadata such as original year, transcript, summary, and sensitivity. We systematically evaluate state-of-the-art multimodal large language models (MLLMs) alongside human participants using this benchmark. Our results reveal a consistent limitation: current models perform poorly on textless music and sound effects, and struggle to think in context and in culture compared to surface content. These findings highlight a key gap in human-aligned multimodal intelligence and call for models that can perceive contextually and culturally beyond the surface of what they hear and see. Project page: avmemeexam.github.io/public

  </details>



- **Will It Zero-Shot?: Will It Zero-Shot?: Predicting Zero-Shot Classification Performance For Arbitrary Queries**  
  Kevin Robbins, Xiaotong Liu, Yu Wu, Le Sun, Grady McPeak, Abby Stylianou, Robert Pless  
  _2026-01-24_ · https://arxiv.org/abs/2601.17535v1  
  <details><summary>Abstract</summary>

  Vision-Language Models like CLIP create aligned embedding spaces for text and images, making it possible for anyone to build a visual classifier by simply naming the classes they want to distinguish. However, a model that works well in one domain may fail in another, and non-expert users have no straightforward way to assess whether their chosen VLM will work on their problem. We build on prior work using text-only comparisons to evaluate how well a model works for a given natural language task, and explore approaches that also generate synthetic images relevant to that task to evaluate and refine the prediction of zero-shot accuracy. We show that generated imagery to the baseline text-only scores substantially improves the quality of these predictions. Additionally, it gives a user feedback on the kinds of images that were used to make the assessment. Experiments on standard CLIP benchmark datasets demonstrate that the image-based approach helps users predict, without any labeled examples, whether a VLM will be effective for their application.

  </details>



- **FMIR, a foundation model-based Image Registration Framework for Robust Image Registration**  
  Fengting Zhang, Yue He, Qinghao Liu, Yaonan Wang, Xiang Chen, Hang Zhang  
  _2026-01-24_ · https://arxiv.org/abs/2601.17529v1  
  <details><summary>Abstract</summary>

  Deep learning has revolutionized medical image registration by achieving unprecedented speeds, yet its clinical application is hindered by a limited ability to generalize beyond the training domain, a critical weakness given the typically small scale of medical datasets. In this paper, we introduce FMIR, a foundation model-based registration framework that overcomes this limitation.Combining a foundation model-based feature encoder for extracting anatomical structures with a general registration head, and trained with a channel regularization strategy on just a single dataset, FMIR achieves state-of-the-art(SOTA) in-domain performance while maintaining robust registration on out-of-domain images.Our approach demonstrates a viable path toward building generalizable medical imaging foundation models with limited resources. The code is available at https://github.com/Monday0328/FMIR.git.

  </details>



- **BMDS-Net: A Bayesian Multi-Modal Deep Supervision Network for Robust Brain Tumor Segmentation**  
  Yan Zhou, Zhen Huang, Yingqiu Li, Yue Ouyang, Suncheng Xiang, Zehua Wang  
  _2026-01-24_ · https://arxiv.org/abs/2601.17504v1  
  <details><summary>Abstract</summary>

  Accurate brain tumor segmentation from multi-modal magnetic resonance imaging (MRI) is a prerequisite for precise radiotherapy planning and surgical navigation. While recent Transformer-based models such as Swin UNETR have achieved impressive benchmark performance, their clinical utility is often compromised by two critical issues: sensitivity to missing modalities (common in clinical practice) and a lack of confidence calibration. Merely chasing higher Dice scores on idealized data fails to meet the safety requirements of real-world medical deployment. In this work, we propose BMDS-Net, a unified framework that prioritizes clinical robustness and trustworthiness over simple metric maximization. Our contribution is three-fold. First, we construct a robust deterministic backbone by integrating a Zero-Init Multimodal Contextual Fusion (MMCF) module and a Residual-Gated Deep Decoder Supervision (DDS) mechanism, enabling stable feature learning and precise boundary delineation with significantly reduced Hausdorff Distance, even under modality corruption. Second, and most importantly, we introduce a memory-efficient Bayesian fine-tuning strategy that transforms the network into a probabilistic predictor, providing voxel-wise uncertainty maps to highlight potential errors for clinicians. Third, comprehensive experiments on the BraTS 2021 dataset demonstrate that BMDS-Net not only maintains competitive accuracy but, more importantly, exhibits superior stability in missing-modality scenarios where baseline models fail. The source code is publicly available at https://github.com/RyanZhou168/BMDS-Net.

  </details>



- **SpatialMath: Spatial Comprehension-Infused Symbolic Reasoning for Mathematical Problem-Solving**  
  Ashutosh Bajpai, Akshat Bhandari, Akshay Nambi, Tanmoy Chakraborty  
  _2026-01-24_ · https://arxiv.org/abs/2601.17489v1  
  <details><summary>Abstract</summary>

  Multimodal Small-to-Medium sized Language Models (MSLMs) have demonstrated strong capabilities in integrating visual and textual information but still face significant limitations in visual comprehension and mathematical reasoning, particularly in geometric problems with diverse levels of visual infusion. Current models struggle to accurately decompose intricate visual inputs and connect perception with structured reasoning, leading to suboptimal performance. To address these challenges, we propose SpatialMath, a novel Spatial Comprehension-Infused Symbolic Reasoning Framework designed to integrate spatial representations into structured symbolic reasoning chains. SpatialMath employs a specialized perception module to extract spatially-grounded representations from visual diagrams, capturing critical geometric structures and spatial relationships. These representations are then methodically infused into symbolic reasoning chains, facilitating visual comprehension-aware structured reasoning. To this end, we introduce MATHVERSE-PLUS, a novel dataset containing structured visual interpretations and step-by-step reasoning paths for vision-intensive mathematical problems. SpatialMath significantly outperforms strong multimodal baselines, achieving up to 10 percentage points improvement over supervised fine-tuning with data augmentation in vision-intensive settings. Robustness analysis reveals that enhanced spatial representations directly improve reasoning accuracy, reinforcing the need for structured perception-to-reasoning pipelines in MSLMs.

  </details>



- **Coronary Artery Segmentation and Vessel-Type Classification in X-Ray Angiography**  
  Mehdi Yousefzadeh, Siavash Shirzadeh Barough, Ashkan Fakharifar, Yashar Tayyarazad, Narges Eghbali, Mohaddeseh Mozaffari, Hoda Taeb, Negar Sadat Rafiee Tabatabaee, Parsa Esfahanian, Ghazaleh Sadeghi Gohar, et al.  
  _2026-01-24_ · https://arxiv.org/abs/2601.17429v1  
  <details><summary>Abstract</summary>

  X-ray coronary angiography (XCA) is the clinical reference standard for assessing coronary artery disease, yet quantitative analysis is limited by the difficulty of robust vessel segmentation in routine data. Low contrast, motion, foreshortening, overlap, and catheter confounding degrade segmentation and contribute to domain shift across centers. Reliable segmentation, together with vessel-type labeling, enables vessel-specific coronary analytics and downstream measurements that depend on anatomical localization. From 670 cine sequences (407 subjects), we select a best frame near peak opacification using a low-intensity histogram criterion and apply joint super-resolution and enhancement. We benchmark classical Meijering, Frangi, and Sato vesselness filters under per-image oracle tuning, a single global mean setting, and per-image parameter prediction via Support Vector Regression (SVR). Neural baselines include U-Net, FPN, and a Swin Transformer, trained with coronary-only and merged coronary+catheter supervision. A second stage assigns vessel identity (LAD, LCX, RCA). External evaluation uses the public DCA1 cohort. SVR per-image tuning improves Dice over global means for all classical filters (e.g., Frangi: 0.759 vs. 0.741). Among deep models, FPN attains 0.914+/-0.007 Dice (coronary-only), and merged coronary+catheter labels further improve to 0.931+/-0.006. On DCA1 as a strict external test, Dice drops to 0.798 (coronary-only) and 0.814 (merged), while light in-domain fine-tuning recovers to 0.881+/-0.014 and 0.882+/-0.015. Vessel-type labeling achieves 98.5% accuracy (Dice 0.844) for RCA, 95.4% (0.786) for LAD, and 96.2% (0.794) for LCX. Learned per-image tuning strengthens classical pipelines, while high-resolution FPN models and merged-label supervision improve stability and external transfer with modest adaptation.

  </details>



- **CoT-Seg: Rethinking Segmentation with Chain-of-Thought Reasoning and Self-Correction**  
  Shiu-hong Kao, Chak Ho Huang, Huaiqian Liu, Yu-Wing Tai, Chi-Keung Tang  
  _2026-01-24_ · https://arxiv.org/abs/2601.17420v1  
  <details><summary>Abstract</summary>

  Existing works of reasoning segmentation often fall short in complex cases, particularly when addressing complicated queries and out-of-domain images. Inspired by the chain-of-thought reasoning, where harder problems require longer thinking steps/time, this paper aims to explore a system that can think step-by-step, look up information if needed, generate results, self-evaluate its own results, and refine the results, in the same way humans approach harder questions. We introduce CoT-Seg, a training-free framework that rethinks reasoning segmentation by combining chain-of-thought reasoning with self-correction. Instead of fine-tuning, CoT-Seg leverages the inherent reasoning ability of pre-trained MLLMs (GPT-4o) to decompose queries into meta-instructions, extract fine-grained semantics from images, and identify target objects even under implicit or complex prompts. Moreover, CoT-Seg incorporates a self-correction stage: the model evaluates its own segmentation against the original query and reasoning trace, identifies mismatches, and iteratively refines the mask. This tight integration of reasoning and correction significantly improves reliability and robustness, especially in ambiguous or error-prone cases. Furthermore, our CoT-Seg framework allows easy incorporation of retrieval-augmented reasoning, enabling the system to access external knowledge when the input lacks sufficient information. To showcase CoT-Seg's ability to handle very challenging cases ,we introduce a new dataset ReasonSeg-Hard. Our results highlight that combining chain-of-thought reasoning, self-correction, offers a powerful paradigm for vision-language integration driven segmentation.

  </details>



- **Source-Free Domain Adaptation by Optimizing Batch-Wise Cosine Similarity**  
  Harsharaj Pathak, Vineeth N Balasubramanian  
  _2026-01-24_ · https://arxiv.org/abs/2601.17408v1  
  <details><summary>Abstract</summary>

  Source-Free Domain Adaptation (SFDA) is an emerging area of research that aims to adapt a model trained on a labeled source domain to an unlabeled target domain without accessing the source data. Most of the successful methods in this area rely on the concept of neighborhood consistency but are prone to errors due to misleading neighborhood information. In this paper, we explore this approach from the point of view of learning more informative clusters and mitigating the effect of noisy neighbors using a concept called neighborhood signature, and demonstrate that adaptation can be achieved using just a single loss term tailored to optimize the similarity and dissimilarity of predictions of samples in the target domain. In particular, our proposed method outperforms existing methods in the challenging VisDA dataset while also yielding competitive results on other benchmark datasets.

  </details>



- **ReLE: A Scalable System and Structured Benchmark for Diagnosing Capability Anisotropy in Chinese LLMs**  
  Rui Fang, Jian Li, Wei Chen, Bin Hu, Ying-Cong Chen, Xin Tang, Liang Diao  
  _2026-01-24_ · https://arxiv.org/abs/2601.17399v1  
  <details><summary>Abstract</summary>

  Large Language Models (LLMs) have achieved rapid progress in Chinese language understanding, yet accurately evaluating their capabilities remains challenged by benchmark saturation and prohibitive computational costs. While static leaderboards provide snapshot rankings, they often mask the structural trade-offs between capabilities. In this work, we present ReLE (Robust Efficient Live Evaluation), a scalable system designed to diagnose Capability Anisotropy, the non-uniformity of model performance across domains. Using ReLE, we evaluate 304 models (189 commercial, 115 open-source) across a Domain $\times$ Capability orthogonal matrix comprising 207,843 samples. We introduce two methodological contributions to address current evaluation pitfalls: (1) A Symbolic-Grounded Hybrid Scoring Mechanism that eliminates embedding-based false positives in reasoning tasks; (2) A Dynamic Variance-Aware Scheduler based on Neyman allocation with noise correction, which reduces compute costs by 70\% compared to full-pass evaluations while maintaining a ranking correlation of $ρ=0.96$. Our analysis reveals that aggregate rankings are highly sensitive to weighting schemes: models exhibit a Rank Stability Amplitude (RSA) of 11.4 in ReLE versus $\sim$5.0 in traditional benchmarks, confirming that modern models are highly specialized rather than generally superior. We position ReLE not as a replacement for comprehensive static benchmarks, but as a high-frequency diagnostic monitor for the evolving model landscape.

  </details>


