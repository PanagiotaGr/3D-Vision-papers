# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-02-12 07:16 UTC_

Total papers shown: **50**


---

- **SurfPhase: 3D Interfacial Dynamics in Two-Phase Flows from Sparse Videos**  
  Yue Gao, Hong-Xing Yu, Sanghyeon Chang, Qianxi Fu, Bo Zhu, Yoonjin Won, Juan Carlos Niebles, Jiajun Wu  
  _2026-02-11_ · https://arxiv.org/abs/2602.11154v1  
  <details><summary>Abstract</summary>

  Interfacial dynamics in two-phase flows govern momentum, heat, and mass transfer, yet remain difficult to measure experimentally. Classical techniques face intrinsic limitations near moving interfaces, while existing neural rendering methods target single-phase flows with diffuse boundaries and cannot handle sharp, deformable liquid-vapor interfaces. We propose SurfPhase, a novel model for reconstructing 3D interfacial dynamics from sparse camera views. Our approach integrates dynamic Gaussian surfels with a signed distance function formulation for geometric consistency, and leverages a video diffusion model to synthesize novel-view videos to refine reconstruction from sparse observations. We evaluate on a new dataset of high-speed pool boiling videos, demonstrating high-quality view synthesis and velocity estimation from only two camera views. Project website: https://yuegao.me/SurfPhase.

  </details>



- **GENIUS: Generative Fluid Intelligence Evaluation Suite**  
  Ruichuan An, Sihan Yang, Ziyu Guo, Wei Dai, Zijun Shen, Haodong Li, Renrui Zhang, Xinyu Wei, Guopeng Li, Wenshan Wu, et al.  
  _2026-02-11_ · https://arxiv.org/abs/2602.11144v1  
  <details><summary>Abstract</summary>

  Unified Multimodal Models (UMMs) have shown remarkable progress in visual generation. Yet, existing benchmarks predominantly assess $\textit{Crystallized Intelligence}$, which relies on recalling accumulated knowledge and learned schemas. This focus overlooks $\textit{Generative Fluid Intelligence (GFI)}$: the capacity to induce patterns, reason through constraints, and adapt to novel scenarios on the fly. To rigorously assess this capability, we introduce $\textbf{GENIUS}$ ($\textbf{GEN}$ Fluid $\textbf{I}$ntelligence Eval$\textbf{U}$ation $\textbf{S}$uite). We formalize $\textit{GFI}$ as a synthesis of three primitives. These include $\textit{Inducing Implicit Patterns}$ (e.g., inferring personalized visual preferences), $\textit{Executing Ad-hoc Constraints}$ (e.g., visualizing abstract metaphors), and $\textit{Adapting to Contextual Knowledge}$ (e.g., simulating counter-intuitive physics). Collectively, these primitives challenge models to solve problems grounded entirely in the immediate context. Our systematic evaluation of 12 representative models reveals significant performance deficits in these tasks. Crucially, our diagnostic analysis disentangles these failure modes. It demonstrates that deficits stem from limited context comprehension rather than insufficient intrinsic generative capability. To bridge this gap, we propose a training-free attention intervention strategy. Ultimately, $\textbf{GENIUS}$ establishes a rigorous standard for $\textit{GFI}$, guiding the field beyond knowledge utilization toward dynamic, general-purpose reasoning. Our dataset and code will be released at: $\href{https://github.com/arctanxarc/GENIUS}{https://github.com/arctanxarc/GENIUS}$.

  </details>



- **HairWeaver: Few-Shot Photorealistic Hair Motion Synthesis with Sim-to-Real Guided Video Diffusion**  
  Di Chang, Ji Hou, Aljaz Bozic, Assaf Neuberger, Felix Juefei-Xu, Olivier Maury, Gene Wei-Chin Lin, Tuur Stuyck, Doug Roble, Mohammad Soleymani, et al.  
  _2026-02-11_ · https://arxiv.org/abs/2602.11117v1  
  <details><summary>Abstract</summary>

  We present HairWeaver, a diffusion-based pipeline that animates a single human image with realistic and expressive hair dynamics. While existing methods successfully control body pose, they lack specific control over hair, and as a result, fail to capture the intricate hair motions, resulting in stiff and unrealistic animations. HairWeaver overcomes this limitation using two specialized modules: a Motion-Context-LoRA to integrate motion conditions and a Sim2Real-Domain-LoRA to preserve the subject's photoreal appearance across different data domains. These lightweight components are designed to guide a video diffusion backbone while maintaining its core generative capabilities. By training on a specialized dataset of dynamic human motion generated from a CG simulator, HairWeaver affords fine control over hair motion and ultimately learns to produce highly realistic hair that responds naturally to movement. Comprehensive evaluations demonstrate that our approach sets a new state of the art, producing lifelike human hair animations with dynamic details.

  </details>



- **First International StepUP Competition for Biometric Footstep Recognition: Methods, Results and Remaining Challenges**  
  Robyn Larracy, Eve MacDonald, Angkoon Phinyomark, Saeid Rezaei, Mahdi Laghaei, Ali Hajighasem, Aaron Tabor, Erik Scheme  
  _2026-02-11_ · https://arxiv.org/abs/2602.11086v1  
  <details><summary>Abstract</summary>

  Biometric footstep recognition, based on a person's unique pressure patterns under their feet during walking, is an emerging field with growing applications in security and safety. However, progress in this area has been limited by the lack of large, diverse datasets necessary to address critical challenges such as generalization to new users and robustness to shifts in factors like footwear or walking speed. The recent release of the UNB StepUP-P150 dataset, the largest and most comprehensive collection of high-resolution footstep pressure recordings to date, opens new opportunities for addressing these challenges through deep learning. To mark this milestone, the First International StepUP Competition for Biometric Footstep Recognition was launched. Competitors were tasked with developing robust recognition models using the StepUP-P150 dataset that were then evaluated on a separate, dedicated test set designed to assess verification performance under challenging variations, given limited and relatively homogeneous reference data. The competition attracted global participation, with 23 registered teams from academia and industry. The top-performing team, Saeid_UCC, achieved the best equal error rate (EER) of 10.77% using a generative reward machine (GRM) optimization strategy. Overall, the competition showcased strong solutions, but persistent challenges in generalizing to unfamiliar footwear highlight a critical area for future work.

  </details>



- **Chain-of-Look Spatial Reasoning for Dense Surgical Instrument Counting**  
  Rishikesh Bhyri, Brian R Quaranto, Philip J Seger, Kaity Tung, Brendan Fox, Gene Yang, Steven D. Schwaitzberg, Junsong Yuan, Nan Xi, Peter C W Kim  
  _2026-02-11_ · https://arxiv.org/abs/2602.11024v1  
  <details><summary>Abstract</summary>

  Accurate counting of surgical instruments in Operating Rooms (OR) is a critical prerequisite for ensuring patient safety during surgery. Despite recent progress of large visual-language models and agentic AI, accurately counting such instruments remains highly challenging, particularly in dense scenarios where instruments are tightly clustered. To address this problem, we introduce Chain-of-Look, a novel visual reasoning framework that mimics the sequential human counting process by enforcing a structured visual chain, rather than relying on classic object detection which is unordered. This visual chain guides the model to count along a coherent spatial trajectory, improving accuracy in complex scenes. To further enforce the physical plausibility of the visual chain, we introduce the neighboring loss function, which explicitly models the spatial constraints inherent to densely packed surgical instruments. We also present SurgCount-HD, a new dataset comprising 1,464 high-density surgical instrument images. Extensive experiments demonstrate that our method outperforms state-of-the-art approaches for counting (e.g., CountGD, REC) as well as Multimodality Large Language Models (e.g., Qwen, ChatGPT) in the challenging task of dense surgical instrument counting.

  </details>



- **Interpretable Vision Transformers in Monocular Depth Estimation via SVDA**  
  Vasileios Arampatzakis, George Pavlidis, Nikolaos Mitianoudis, Nikos Papamarkos  
  _2026-02-11_ · https://arxiv.org/abs/2602.11005v1  
  <details><summary>Abstract</summary>

  Monocular depth estimation is a central problem in computer vision with applications in robotics, AR, and autonomous driving, yet the self-attention mechanisms that drive modern Transformer architectures remain opaque. We introduce SVD-Inspired Attention (SVDA) into the Dense Prediction Transformer (DPT), providing the first spectrally structured formulation of attention for dense prediction tasks. SVDA decouples directional alignment from spectral modulation by embedding a learnable diagonal matrix into normalized query-key interactions, enabling attention maps that are intrinsically interpretable rather than post-hoc approximations. Experiments on KITTI and NYU-v2 show that SVDA preserves or slightly improves predictive accuracy while adding only minor computational overhead. More importantly, SVDA unlocks six spectral indicators that quantify entropy, rank, sparsity, alignment, selectivity, and robustness. These reveal consistent cross-dataset and depth-wise patterns in how attention organizes during training, insights that remain inaccessible in standard Transformers. By shifting the role of attention from opaque mechanism to quantifiable descriptor, SVDA redefines interpretability in monocular depth estimation and opens a principled avenue toward transparent dense prediction models.

  </details>



- **Enhancing Predictability of Multi-Tenant DNN Inference for Autonomous Vehicles' Perception**  
  Liangkai Liu, Kang G. Shin, Jinkyu Lee, Chengmo Yang, Weisong Shi  
  _2026-02-11_ · https://arxiv.org/abs/2602.11004v1  
  <details><summary>Abstract</summary>

  Autonomous vehicles (AVs) rely on sensors and deep neural networks (DNNs) to perceive their surrounding environment and make maneuver decisions in real time. However, achieving real-time DNN inference in the AV's perception pipeline is challenging due to the large gap between the computation requirement and the AV's limited resources. Most, if not all, of existing studies focus on optimizing the DNN inference time to achieve faster perception by compressing the DNN model with pruning and quantization. In contrast, we present a Predictable Perception system with DNNs (PP-DNN) that reduce the amount of image data to be processed while maintaining the same level of accuracy for multi-tenant DNNs by dynamically selecting critical frames and regions of interest (ROIs). PP-DNN is based on our key insight that critical frames and ROIs for AVs vary with the AV's surrounding environment. However, it is challenging to identify and use critical frames and ROIs in multi-tenant DNNs for predictable inference. Given image-frame streams, PP-DNN leverages an ROI generator to identify critical frames and ROIs based on the similarities of consecutive frames and traffic scenarios. PP-DNN then leverages a FLOPs predictor to predict multiply-accumulate operations (MACs) from the dynamic critical frames and ROIs. The ROI scheduler coordinates the processing of critical frames and ROIs with multiple DNN models. Finally, we design a detection predictor for the perception of non-critical frames. We have implemented PP-DNN in an ROS-based AV pipeline and evaluated it with the BDD100K and the nuScenes dataset. PP-DNN is observed to significantly enhance perception predictability, increasing the number of fusion frames by up to 7.3x, reducing the fusion delay by >2.6x and fusion-delay variations by >2.3x, improving detection completeness by 75.4% and the cost-effectiveness by up to 98% over the baseline.

  </details>



- **DFIC: Towards a balanced facial image dataset for automatic ICAO compliance verification**  
  Nuno Gonçalves, Diogo Nunes, Carla Guerra, João Marcos  
  _2026-02-11_ · https://arxiv.org/abs/2602.10985v1  
  <details><summary>Abstract</summary>

  Ensuring compliance with ISO/IEC and ICAO standards for facial images in machine-readable travel documents (MRTDs) is essential for reliable identity verification, but current manual inspection methods are inefficient in high-demand environments. This paper introduces the DFIC dataset, a novel comprehensive facial image dataset comprising around 58,000 annotated images and 2706 videos of more than 1000 subjects, that cover a broad range of non-compliant conditions, in addition to compliant portraits. Our dataset provides a more balanced demographic distribution than the existing public datasets, with one partition that is nearly uniformly distributed, facilitating the development of automated ICAO compliance verification methods. Using DFIC, we fine-tuned a novel method that heavily relies on spatial attention mechanisms for the automatic validation of ICAO compliance requirements, and we have compared it with the state-of-the-art aimed at ICAO compliance verification, demonstrating improved results. DFIC dataset is now made public (https://github.com/visteam-isr-uc/DFIC) for the training and validation of new models, offering an unprecedented diversity of faces, that will improve both robustness and adaptability to the intrinsically diverse combinations of faces and props that can be presented to the validation system. These results emphasize the potential of DFIC to enhance automated ICAO compliance methods but it can also be used in many other applications that aim to improve the security, privacy, and fairness of facial recognition systems.

  </details>



- **RADAR: Benchmarking Vision-Language-Action Generalization via Real-World Dynamics, Spatial-Physical Intelligence, and Autonomous Evaluation**  
  Yuhao Chen, Zhihao Zhan, Xiaoxin Lin, Zijian Song, Hao Liu, Qinhan Lyu, Yubo Zu, Xiao Chen, Zhiyuan Liu, Tao Pu, et al.  
  _2026-02-11_ · https://arxiv.org/abs/2602.10980v1  
  <details><summary>Abstract</summary>

  VLA models have achieved remarkable progress in embodied intelligence; however, their evaluation remains largely confined to simulations or highly constrained real-world settings. This mismatch creates a substantial reality gap, where strong benchmark performance often masks poor generalization in diverse physical environments. We identify three systemic shortcomings in current benchmarking practices that hinder fair and reliable model comparison. (1) Existing benchmarks fail to model real-world dynamics, overlooking critical factors such as dynamic object configurations, robot initial states, lighting changes, and sensor noise. (2) Current protocols neglect spatial--physical intelligence, reducing evaluation to rote manipulation tasks that do not probe geometric reasoning. (3) The field lacks scalable fully autonomous evaluation, instead relying on simplistic 2D metrics that miss 3D spatial structure or on human-in-the-loop systems that are costly, biased, and unscalable. To address these limitations, we introduce RADAR (Real-world Autonomous Dynamics And Reasoning), a benchmark designed to systematically evaluate VLA generalization under realistic conditions. RADAR integrates three core components: (1) a principled suite of physical dynamics; (2) dedicated tasks that explicitly test spatial reasoning and physical understanding; and (3) a fully autonomous evaluation pipeline based on 3D metrics, eliminating the need for human supervision. We apply RADAR to audit multiple state-of-the-art VLA models and uncover severe fragility beneath their apparent competence. Performance drops precipitously under modest physical dynamics, with the expectation of 3D IoU declining from 0.261 to 0.068 under sensor noise. Moreover, models exhibit limited spatial reasoning capability. These findings position RADAR as a necessary bench toward reliable and generalizable real-world evaluation of VLA models.

  </details>



- **Healthy Harvests: A Comparative Look at Guava Disease Classification Using InceptionV3**  
  Samanta Ghosh, Shaila Afroz Anika, Umma Habiba Ahmed, B. M. Shahria Alam, Mohammad Tahmid Noor, Nishat Tasnim Niloy  
  _2026-02-11_ · https://arxiv.org/abs/2602.10967v1  
  <details><summary>Abstract</summary>

  Guava fruits often suffer from many diseases. This can harm fruit quality and fruit crop yield. Early identification is important for minimizing damage and ensuring fruit health. This study focuses on 3 different categories for classifying diseases. These are Anthracnose, Fruit flies, and Healthy fruit. The data set used in this study is collected from Mendeley Data. This dataset contains 473 original images of Guava. These images vary in size and format. The original dataset was resized to 256x256 pixels with RGB color mode for better consistency. After this, the Data augmentation process is applied to improve the dataset by generating variations of the original images. The augmented dataset consists of 3784 images using advanced preprocessing techniques. Two deep learning models were implemented to classify the images. The InceptionV3 model is well known for its advanced framework. These apply multiple convolutional filters for obtaining different features effectively. On the other hand, the ResNet50 model helps to train deeper networks by using residual learning. The InceptionV3 model achieved the impressive accuracy of 98.15%, and ResNet50got 94.46% accuracy. Data mixing methods such as CutMix and MixUp were applied to enhance the model's robustness. The confusion matrix was used to evaluate the overall model performance of both InceptionV3 and Resnet50. Additionally, SHAP analysis is used to improve interpretability, which helps to find the significant parts of the image for the model prediction. This study purposes to highlight how advanced models enhan

  </details>



- **Chart Specification: Structural Representations for Incentivizing VLM Reasoning in Chart-to-Code Generation**  
  Minggui He, Mingchen Dai, Jian Zhang, Yilun Liu, Shimin Tao, Pufan Zeng, Osamu Yoshie, Yuya Ieiri  
  _2026-02-11_ · https://arxiv.org/abs/2602.10880v1  
  <details><summary>Abstract</summary>

  Vision-Language Models (VLMs) have shown promise in generating plotting code from chart images, yet achieving structural fidelity remains challenging. Existing approaches largely rely on supervised fine-tuning, encouraging surface-level token imitation rather than faithful modeling of underlying chart structure, which often leads to hallucinated or semantically inconsistent outputs. We propose Chart Specification, a structured intermediate representation that shifts training from text imitation to semantically grounded supervision. Chart Specification filters syntactic noise to construct a structurally balanced training set and supports a Spec-Align Reward that provides fine-grained, verifiable feedback on structural correctness, enabling reinforcement learning to enforce consistent plotting logic. Experiments on three public benchmarks show that our method consistently outperforms prior approaches. With only 3K training samples, we achieve strong data efficiency, surpassing leading baselines by up to 61.7% on complex benchmarks, and scaling to 4K samples establishes new state-of-the-art results across all evaluated metrics. Overall, our results demonstrate that precise structural supervision offers an efficient pathway to high-fidelity chart-to-code generation. Code and dataset are available at: https://github.com/Mighten/chart-specification-paper

  </details>



- **Viewpoint Recommendation for Point Cloud Labeling through Interaction Cost Modeling**  
  Yu Zhang, Xinyi Zhao, Chongke Bi, Siming Chen  
  _2026-02-11_ · https://arxiv.org/abs/2602.10871v1  
  <details><summary>Abstract</summary>

  Semantic segmentation of 3D point clouds is important for many applications, such as autonomous driving. To train semantic segmentation models, labeled point cloud segmentation datasets are essential. Meanwhile, point cloud labeling is time-consuming for annotators, which typically involves tuning the camera viewpoint and selecting points by lasso. To reduce the time cost of point cloud labeling, we propose a viewpoint recommendation approach to reduce annotators' labeling time costs. We adapt Fitts' law to model the time cost of lasso selection in point clouds. Using the modeled time cost, the viewpoint that minimizes the lasso selection time cost is recommended to the annotator. We build a data labeling system for semantic segmentation of 3D point clouds that integrates our viewpoint recommendation approach. The system enables users to navigate to recommended viewpoints for efficient annotation. Through an ablation study, we observed that our approach effectively reduced the data labeling time cost. We also qualitatively compare our approach with previous viewpoint selection approaches on different datasets.

  </details>



- **Hyperspectral Smoke Segmentation via Mixture of Prototypes**  
  Lujian Yao, Haitao Zhao, Xianghai Kong, Yuhan Xu  
  _2026-02-11_ · https://arxiv.org/abs/2602.10858v1  
  <details><summary>Abstract</summary>

  Smoke segmentation is critical for wildfire management and industrial safety applications. Traditional visible-light-based methods face limitations due to insufficient spectral information, particularly struggling with cloud interference and semi-transparent smoke regions. To address these challenges, we introduce hyperspectral imaging for smoke segmentation and present the first hyperspectral smoke segmentation dataset (HSSDataset) with carefully annotated samples collected from over 18,000 frames across 20 real-world scenarios using a Many-to-One annotations protocol. However, different spectral bands exhibit varying discriminative capabilities across spatial regions, necessitating adaptive band weighting strategies. We decompose this into three technical challenges: spectral interaction contamination, limited spectral pattern modeling, and complex weighting router problems. We propose a mixture of prototypes (MoP) network with: (1) Band split for spectral isolation, (2) Prototype-based spectral representation for diverse patterns, and (3) Dual-level router for adaptive spatial-aware band weighting. We further construct a multispectral dataset (MSSDataset) with RGB-infrared images. Extensive experiments validate superior performance across both hyperspectral and multispectral modalities, establishing a new paradigm for spectral-based smoke segmentation.

  </details>



- **Flow caching for autoregressive video generation**  
  Yuexiao Ma, Xuzhe Zheng, Jing Xu, Xiwei Xu, Feng Ling, Xiawu Zheng, Huafeng Kuang, Huixia Li, Xing Wang, Xuefeng Xiao, et al.  
  _2026-02-11_ · https://arxiv.org/abs/2602.10825v1  
  <details><summary>Abstract</summary>

  Autoregressive models, often built on Transformer architectures, represent a powerful paradigm for generating ultra-long videos by synthesizing content in sequential chunks. However, this sequential generation process is notoriously slow. While caching strategies have proven effective for accelerating traditional video diffusion models, existing methods assume uniform denoising across all frames-an assumption that breaks down in autoregressive models where different video chunks exhibit varying similarity patterns at identical timesteps. In this paper, we present FlowCache, the first caching framework specifically designed for autoregressive video generation. Our key insight is that each video chunk should maintain independent caching policies, allowing fine-grained control over which chunks require recomputation at each timestep. We introduce a chunkwise caching strategy that dynamically adapts to the unique denoising characteristics of each chunk, complemented by a joint importance-redundancy optimized KV cache compression mechanism that maintains fixed memory bounds while preserving generation quality. Our method achieves remarkable speedups of 2.38 times on MAGI-1 and 6.7 times on SkyReels-V2, with negligible quality degradation (VBench: 0.87 increase and 0.79 decrease respectively). These results demonstrate that FlowCache successfully unlocks the potential of autoregressive models for real-time, ultra-long video generation-establishing a new benchmark for efficient video synthesis at scale. The code is available at https://github.com/mikeallen39/FlowCache.

  </details>



- **DeepImageSearch: Benchmarking Multimodal Agents for Context-Aware Image Retrieval in Visual Histories**  
  Chenlong Deng, Mengjie Deng, Junjie Wu, Dun Zeng, Teng Wang, Qingsong Xie, Jiadeng Huang, Shengjie Ma, Changwang Zhang, Zhaoxiang Wang, et al.  
  _2026-02-11_ · https://arxiv.org/abs/2602.10809v1  
  <details><summary>Abstract</summary>

  Existing multimodal retrieval systems excel at semantic matching but implicitly assume that query-image relevance can be measured in isolation. This paradigm overlooks the rich dependencies inherent in realistic visual streams, where information is distributed across temporal sequences rather than confined to single snapshots. To bridge this gap, we introduce DeepImageSearch, a novel agentic paradigm that reformulates image retrieval as an autonomous exploration task. Models must plan and perform multi-step reasoning over raw visual histories to locate targets based on implicit contextual cues. We construct DISBench, a challenging benchmark built on interconnected visual data. To address the scalability challenge of creating context-dependent queries, we propose a human-model collaborative pipeline that employs vision-language models to mine latent spatiotemporal associations, effectively offloading intensive context discovery before human verification. Furthermore, we build a robust baseline using a modular agent framework equipped with fine-grained tools and a dual-memory system for long-horizon navigation. Extensive experiments demonstrate that DISBench poses significant challenges to state-of-the-art models, highlighting the necessity of incorporating agentic reasoning into next-generation retrieval systems.

  </details>



- **DMP-3DAD: Cross-Category 3D Anomaly Detection via Realistic Depth Map Projection with Few Normal Samples**  
  Zi Wang, Katsuya Hotta, Koichiro Kamide, Yawen Zou, Jianjian Qin, Chao Zhang, Jun Yu  
  _2026-02-11_ · https://arxiv.org/abs/2602.10806v1  
  <details><summary>Abstract</summary>

  Cross-category anomaly detection for 3D point clouds aims to determine whether an unseen object belongs to a target category using only a few normal examples. Most existing methods rely on category-specific training, which limits their flexibility in few-shot scenarios. In this paper, we propose DMP-3DAD, a training-free framework for cross-category 3D anomaly detection based on multi-view realistic depth map projection. Specifically, by converting point clouds into a fixed set of realistic depth images, our method leverages a frozen CLIP visual encoder to extract multi-view representations and performs anomaly detection via weighted feature similarity, which does not require any fine-tuning or category-dependent adaptation. Extensive experiments on the ShapeNetPart dataset demonstrate that DMP-3DAD achieves state-of-the-art performance under few-shot setting. The results show that the proposed approach provides a simple yet effective solution for practical cross-category 3D anomaly detection.

  </details>



- **RSHallu: Dual-Mode Hallucination Evaluation for Remote-Sensing Multimodal Large Language Models with Domain-Tailored Mitigation**  
  Zihui Zhou, Yong Feng, Yanying Chen, Guofan Duan, Zhenxi Song, Mingliang Zhou, Weijia Jia  
  _2026-02-11_ · https://arxiv.org/abs/2602.10799v1  
  <details><summary>Abstract</summary>

  Multimodal large language models (MLLMs) are increasingly adopted in remote sensing (RS) and have shown strong performance on tasks such as RS visual grounding (RSVG), RS visual question answering (RSVQA), and multimodal dialogue. However, hallucinations, which are responses inconsistent with the input RS images, severely hinder their deployment in high-stakes scenarios (e.g., emergency management and agricultural monitoring) and remain under-explored in RS. In this work, we present RSHallu, a systematic study with three deliverables: (1) we formalize RS hallucinations with an RS-oriented taxonomy and introduce image-level hallucination to capture RS-specific inconsistencies beyond object-centric errors (e.g., modality, resolution, and scene-level semantics); (2) we build a hallucination benchmark RSHalluEval (2,023 QA pairs) and enable dual-mode checking, supporting high-precision cloud auditing and low-cost reproducible local checking via a compact checker fine-tuned on RSHalluCheck dataset (15,396 QA pairs); and (3) we introduce a domain-tailored dataset RSHalluShield (30k QA pairs) for training-friendly mitigation and further propose training-free plug-and-play strategies, including decoding-time logit correction and RS-aware prompting. Across representative RS-MLLMs, our mitigation improves the hallucination-free rate by up to 21.63 percentage points under a unified protocol, while maintaining competitive performance on downstream RS tasks (RSVQA/RSVG). Code and datasets will be released.

  </details>



- **From Steering to Pedalling: Do Autonomous Driving VLMs Generalize to Cyclist-Assistive Spatial Perception and Planning?**  
  Krishna Kanth Nakka, Vedasri Nakka  
  _2026-02-11_ · https://arxiv.org/abs/2602.10771v1  
  <details><summary>Abstract</summary>

  Cyclists often encounter safety-critical situations in urban traffic, highlighting the need for assistive systems that support safe and informed decision-making. Recently, vision-language models (VLMs) have demonstrated strong performance on autonomous driving benchmarks, suggesting their potential for general traffic understanding and navigation-related reasoning. However, existing evaluations are predominantly vehicle-centric and fail to assess perception and reasoning from a cyclist-centric viewpoint. To address this gap, we introduce CyclingVQA, a diagnostic benchmark designed to probe perception, spatio-temporal understanding, and traffic-rule-to-lane reasoning from a cyclist's perspective. Evaluating 31+ recent VLMs spanning general-purpose, spatially enhanced, and autonomous-driving-specialized models, we find that current models demonstrate encouraging capabilities, while also revealing clear areas for improvement in cyclist-centric perception and reasoning, particularly in interpreting cyclist-specific traffic cues and associating signs with the correct navigational lanes. Notably, several driving-specialized models underperform strong generalist VLMs, indicating limited transfer from vehicle-centric training to cyclist-assistive scenarios. Finally, through systematic error analysis, we identify recurring failure modes to guide the development of more effective cyclist-assistive intelligent systems.

  </details>



- **Dual-End Consistency Model**  
  Linwei Dong, Ruoyu Guo, Ge Bai, Zehuan Yuan, Yawei Luo, Changqing Zou  
  _2026-02-11_ · https://arxiv.org/abs/2602.10764v1  
  <details><summary>Abstract</summary>

  The slow iterative sampling nature remains a major bottleneck for the practical deployment of diffusion and flow-based generative models. While consistency models (CMs) represent a state-of-the-art distillation-based approach for efficient generation, their large-scale application is still limited by two key issues: training instability and inflexible sampling. Existing methods seek to mitigate these problems through architectural adjustments or regularized objectives, yet overlook the critical reliance on trajectory selection. In this work, we first conduct an analysis on these two limitations: training instability originates from loss divergence induced by unstable self-supervised term, whereas sampling inflexibility arises from error accumulation. Based on these insights and analysis, we propose the Dual-End Consistency Model (DE-CM) that selects vital sub-trajectory clusters to achieve stable and effective training. DE-CM decomposes the PF-ODE trajectory and selects three critical sub-trajectories as optimization targets. Specifically, our approach leverages continuous-time CMs objectives to achieve few-step distillation and utilizes flow matching as a boundary regularizer to stabilize the training process. Furthermore, we propose a novel noise-to-noisy (N2N) mapping that can map noise to any point, thereby alleviating the error accumulation in the first step. Extensive experimental results show the effectiveness of our method: it achieves a state-of-the-art FID score of 1.70 in one-step generation on the ImageNet 256x256 dataset, outperforming existing CM-based one-step approaches.

  </details>



- **SecureScan: An AI-Driven Multi-Layer Framework for Malware and Phishing Detection Using Logistic Regression and Threat Intelligence Integration**  
  Rumman Firdos, Aman Dangi  
  _2026-02-11_ · https://arxiv.org/abs/2602.10750v1  
  <details><summary>Abstract</summary>

  The growing sophistication of modern malware and phishing campaigns has diminished the effectiveness of traditional signature-based intrusion detection systems. This work presents SecureScan, an AI-driven, triple-layer detection framework that integrates logistic regression-based classification, heuristic analysis, and external threat intelligence via the VirusTotal API for comprehensive triage of URLs, file hashes, and binaries. The proposed architecture prioritizes efficiency by filtering known threats through heuristics, classifying uncertain samples using machine learning, and validating borderline cases with third-party intelligence. On benchmark datasets, SecureScan achieves 93.1 percent accuracy with balanced precision (0.87) and recall (0.92), demonstrating strong generalization and reduced overfitting through threshold-based decision calibration. A calibrated threshold and gray-zone logic (0.45-0.55) were introduced to minimize false positives and enhance real-world stability. Experimental results indicate that a lightweight statistical model, when augmented with calibrated verification and external intelligence, can achieve reliability and performance comparable to more complex deep learning systems.

  </details>



- **Self-Supervised Image Super-Resolution Quality Assessment based on Content-Free Multi-Model Oriented Representation Learning**  
  Kian Majlessi, Amir Masoud Soltani, Mohammad Ebrahim Mahdavi, Aurelien Gourrier, Peyman Adibi  
  _2026-02-11_ · https://arxiv.org/abs/2602.10744v1  
  <details><summary>Abstract</summary>

  Super-resolution (SR) applied to real-world low-resolution (LR) images often results in complex, irregular degradations that stem from the inherent complexity of natural scene acquisition. In contrast to SR artifacts arising from synthetic LR images created under well-defined scenarios, those distortions are highly unpredictable and vary significantly across different real-life contexts. Consequently, assessing the quality of SR images (SR-IQA) obtained from realistic LR, remains a challenging and underexplored problem. In this work, we introduce a no-reference SR-IQA approach tailored for such highly ill-posed realistic settings. The proposed method enables domain-adaptive IQA for real-world SR applications, particularly in data-scarce domains. We hypothesize that degradations in super-resolved images are strongly dependent on the underlying SR algorithms, rather than being solely determined by image content. To this end, we introduce a self-supervised learning (SSL) strategy that first pretrains multiple SR model oriented representations in a pretext stage. Our contrastive learning framework forms positive pairs from images produced by the same SR model and negative pairs from those generated by different methods, independent of image content. The proposed approach S3 RIQA, further incorporates targeted preprocessing to extract complementary quality information and an auxiliary task to better handle the various degradation profiles associated with different SR scaling factors. To this end, we constructed a new dataset, SRMORSS, to support unsupervised pretext training; it includes a wide range of SR algorithms applied to numerous real LR images, which addresses a gap in existing datasets. Experiments on real SR-IQA benchmarks demonstrate that S3 RIQA consistently outperforms most state-of-the-art relevant metrics.

  </details>



- **OccFace: Unified Occlusion-Aware Facial Landmark Detection with Per-Point Visibility**  
  Xinhao Xiang, Zhengxin Li, Saurav Dhakad, Theo Bancroft, Jiawei Zhang, Weiyang Li  
  _2026-02-11_ · https://arxiv.org/abs/2602.10728v1  
  <details><summary>Abstract</summary>

  Accurate facial landmark detection under occlusion remains challenging, especially for human-like faces with large appearance variation and rotation-driven self-occlusion. Existing detectors typically localize landmarks while handling occlusion implicitly, without predicting per-point visibility that downstream applications can benefits. We present OccFace, an occlusion-aware framework for universal human-like faces, including humans, stylized characters, and other non-human designs. OccFace adopts a unified dense 100-point layout and a heatmap-based backbone, and adds an occlusion module that jointly predicts landmark coordinates and per-point visibility by combining local evidence with cross-landmark context. Visibility supervision mixes manual labels with landmark-aware masking that derives pseudo visibility from mask-heatmap overlap. We also create an occlusion-aware evaluation suite reporting NME on visible vs. occluded landmarks and benchmarking visibility with Occ AP, F1@0.5, and ROC-AUC, together with a dataset annotated with 100-point landmarks and per-point visibility. Experiments show improved robustness under external occlusion and large head rotations, especially on occluded regions, while preserving accuracy on visible landmarks.

  </details>



- **(MGS)$^2$-Net: Unifying Micro-Geometric Scale and Macro-Geometric Structure for Cross-View Geo-Localization**  
  Minglei Li, Mengfan He, Chao Chen, Ziyang Meng  
  _2026-02-11_ · https://arxiv.org/abs/2602.10704v1  
  <details><summary>Abstract</summary>

  Cross-view geo-localization (CVGL) is pivotal for GNSS-denied UAV navigation but remains brittle under the drastic geometric misalignment between oblique aerial views and orthographic satellite references. Existing methods predominantly operate within a 2D manifold, neglecting the underlying 3D geometry where view-dependent vertical facades (macro-structure) and scale variations (micro-scale) severely corrupt feature alignment. To bridge this gap, we propose (MGS)$^2$, a geometry-grounded framework. The core of our innovation is the Macro-Geometric Structure Filtering (MGSF) module. Unlike pixel-wise matching sensitive to noise, MGSF leverages dilated geometric gradients to physically filter out high-frequency facade artifacts while enhancing the view-invariant horizontal plane, directly addressing the domain shift. To guarantee robust input for this structural filtering, we explicitly incorporate a Micro-Geometric Scale Adaptation (MGSA) module. MGSA utilizes depth priors to dynamically rectify scale discrepancies via multi-branch feature fusion. Furthermore, a Geometric-Appearance Contrastive Distillation (GACD) loss is designed to strictly discriminate against oblique occlusions. Extensive experiments demonstrate that (MGS)$^2$ achieves state-of-the-art performance, recording a Recall@1 of 97.5\% on University-1652 and 97.02\% on SUES-200. Furthermore, the framework exhibits superior cross-dataset generalization against geometric ambiguity. The code is available at: \href{https://github.com/GabrielLi1473/MGS-Net}{https://github.com/GabrielLi1473/MGS-Net}.

  </details>



- **TwiFF (Think With Future Frames): A Large-Scale Dataset for Dynamic Visual Reasoning**  
  Junhua Liu, Zhangcheng Wang, Zhike Han, Ningli Wang, Guotao Liang, Kun Kuang  
  _2026-02-11_ · https://arxiv.org/abs/2602.10675v1  
  <details><summary>Abstract</summary>

  Visual Chain-of-Thought (VCoT) has emerged as a promising paradigm for enhancing multimodal reasoning by integrating visual perception into intermediate reasoning steps. However, existing VCoT approaches are largely confined to static scenarios and struggle to capture the temporal dynamics essential for tasks such as instruction, prediction, and camera motion. To bridge this gap, we propose TwiFF-2.7M, the first large-scale, temporally grounded VCoT dataset derived from $2.7$ million video clips, explicitly designed for dynamic visual question and answer. Accompanying this, we introduce TwiFF-Bench, a high-quality evaluation benchmark of $1,078$ samples that assesses both the plausibility of reasoning trajectories and the correctness of final answers in open-ended dynamic settings. Building on these foundations, we propose the TwiFF model, a unified modal that synergistically leverages pre-trained video generation and image comprehension capabilities to produce temporally coherent visual reasoning cues-iteratively generating future action frames and textual reasoning. Extensive experiments demonstrate that TwiFF significantly outperforms existing VCoT methods and Textual Chain-of-Thought baselines on dynamic reasoning tasks, which fully validates the effectiveness for visual question answering in dynamic scenarios. Our code and data is available at https://github.com/LiuJunhua02/TwiFF.

  </details>



- **AurigaNet: A Real-Time Multi-Task Network for Enhanced Urban Driving Perception**  
  Kiarash Ghasemzadeh, Sedigheh Dehghani  
  _2026-02-11_ · https://arxiv.org/abs/2602.10660v1  
  <details><summary>Abstract</summary>

  Self-driving cars hold significant potential to reduce traffic accidents, alleviate congestion, and enhance urban mobility. However, developing reliable AI systems for autonomous vehicles remains a substantial challenge. Over the past decade, multi-task learning has emerged as a powerful approach to address complex problems in driving perception. Multi-task networks offer several advantages, including increased computational efficiency, real-time processing capabilities, optimized resource utilization, and improved generalization. In this study, we present AurigaNet, an advanced multi-task network architecture designed to push the boundaries of autonomous driving perception. AurigaNet integrates three critical tasks: object detection, lane detection, and drivable area instance segmentation. The system is trained and evaluated using the BDD100K dataset, renowned for its diversity in driving conditions. Key innovations of AurigaNet include its end-to-end instance segmentation capability, which significantly enhances both accuracy and efficiency in path estimation for autonomous vehicles. Experimental results demonstrate that AurigaNet achieves an 85.2% IoU in drivable area segmentation, outperforming its closest competitor by 0.7%. In lane detection, AurigaNet achieves a remarkable 60.8% IoU, surpassing other models by more than 30%. Furthermore, the network achieves an mAP@0.5:0.95 of 47.6% in traffic object detection, exceeding the next leading model by 2.9%. Additionally, we validate the practical feasibility of AurigaNet by deploying it on embedded devices such as the Jetson Orin NX, where it demonstrates competitive real-time performance. These results underscore AurigaNet's potential as a robust and efficient solution for autonomous driving perception systems. The code can be found here https://github.com/KiaRational/AurigaNet.

  </details>



- **Enhancing YOLOv11n for Reliable Child Detection in Noisy Surveillance Footage**  
  Khanh Linh Tran, Minh Nguyen Dang, Thien Nguyen Trong, Hung Nguyen Quoc, Linh Nguyen Kieu  
  _2026-02-11_ · https://arxiv.org/abs/2602.10592v1  
  <details><summary>Abstract</summary>

  This paper presents a practical and lightweight solution for enhancing child detection in low-quality surveillance footage, a critical component in real-world missing child alert and daycare monitoring systems. Building upon the efficient YOLOv11n architecture, we propose a deployment-ready pipeline that improves detection under challenging conditions including occlusion, small object size, low resolution, motion blur, and poor lighting commonly found in existing CCTV infrastructures. Our approach introduces a domain-specific augmentation strategy that synthesizes realistic child placements using spatial perturbations such as partial visibility, truncation, and overlaps, combined with photometric degradations including lighting variation and noise. To improve recall of small and partially occluded instances, we integrate Slicing Aided Hyper Inference (SAHI) at inference time. All components are trained and evaluated on a filtered, child-only subset of the Roboflow Daycare dataset. Compared to the baseline YOLOv11n, our enhanced system achieves a mean Average Precision at 0.5 IoU (mAP@0.5) of 0.967 and a mean Average Precision averaged over IoU thresholds from 0.5 to 0.95 (mAP@0.5:0.95) of 0.783, yielding absolute improvements of 0.7 percent and 2.3 percent, respectively, without architectural changes. Importantly, the entire pipeline maintains compatibility with low-power edge devices and supports real-time performance, making it particularly well suited for low-cost or resource-constrained industrial surveillance deployments. The example augmented dataset and the source code used to generate it are available at: https://github.com/html-ptit/Data-Augmentation-YOLOv11n-child-detection

  </details>



- **MetaphorStar: Image Metaphor Understanding and Reasoning with End-to-End Visual Reinforcement Learning**  
  Chenhao Zhang, Yazhe Niu, Hongsheng Li  
  _2026-02-11_ · https://arxiv.org/abs/2602.10575v1  
  <details><summary>Abstract</summary>

  Metaphorical comprehension in images remains a critical challenge for Nowadays AI systems. While Multimodal Large Language Models (MLLMs) excel at basic Visual Question Answering (VQA), they consistently struggle to grasp the nuanced cultural, emotional, and contextual implications embedded in visual content. This difficulty stems from the task's demand for sophisticated multi-hop reasoning, cultural context, and Theory of Mind (ToM) capabilities, which current models lack. To fill this gap, we propose MetaphorStar, the first end-to-end visual reinforcement learning (RL) framework for image implication tasks. Our framework includes three core components: the fine-grained dataset TFQ-Data, the visual RL method TFQ-GRPO, and the well-structured benchmark TFQ-Bench. Our fully open-source MetaphorStar family, trained using TFQ-GRPO on TFQ-Data, significantly improves performance by an average of 82.6% on the image implication benchmarks. Compared with 20+ mainstream MLLMs, MetaphorStar-32B achieves state-of-the-art (SOTA) on Multiple-Choice Question and Open-Style Question, significantly outperforms the top closed-source model Gemini-3.0-pro on True-False Question. Crucially, our experiments reveal that learning image implication tasks improves the general understanding ability, especially the complex visual reasoning ability. We further provide a systematic analysis of model parameter scaling, training data scaling, and the impact of different model architectures and training strategies, demonstrating the broad applicability of our method. We open-sourced all model weights, datasets, and method code at https://metaphorstar.github.io.

  </details>



- **LAP: Language-Action Pre-Training Enables Zero-shot Cross-Embodiment Transfer**  
  Lihan Zha, Asher J. Hancock, Mingtong Zhang, Tenny Yin, Yixuan Huang, Dhruv Shah, Allen Z. Ren, Anirudha Majumdar  
  _2026-02-11_ · https://arxiv.org/abs/2602.10556v1  
  <details><summary>Abstract</summary>

  A long-standing goal in robotics is a generalist policy that can be deployed zero-shot on new robot embodiments without per-embodiment adaptation. Despite large-scale multi-embodiment pre-training, existing Vision-Language-Action models (VLAs) remain tightly coupled to their training embodiments and typically require costly fine-tuning. We introduce Language-Action Pre-training (LAP), a simple recipe that represents low-level robot actions directly in natural language, aligning action supervision with the pre-trained vision-language model's input-output distribution. LAP requires no learned tokenizer, no costly annotation, and no embodiment-specific architectural design. Based on LAP, we present LAP-3B, which to the best of our knowledge is the first VLA to achieve substantial zero-shot transfer to previously unseen robot embodiments without any embodiment-specific fine-tuning. Across multiple novel robots and manipulation tasks, LAP-3B attains over 50% average zero-shot success, delivering roughly a 2x improvement over the strongest prior VLAs. We further show that LAP enables efficient adaptation and favorable scaling, while unifying action prediction and VQA in a shared language-action format that yields additional gains through co-training.

  </details>



- **RealHD: A High-Quality Dataset for Robust Detection of State-of-the-Art AI-Generated Images**  
  Hanzhe Yu, Yun Ye, Jintao Rong, Qi Xuan, Chen Ma  
  _2026-02-11_ · https://arxiv.org/abs/2602.10546v1  
  <details><summary>Abstract</summary>

  The rapid advancement of generative AI has raised concerns about the authenticity of digital images, as highly realistic fake images can now be generated at low cost, potentially increasing societal risks. In response, several datasets have been established to train detection models aimed at distinguishing AI-generated images from real ones. However, existing datasets suffer from limited generalization, low image quality, overly simple prompts, and insufficient image diversity. To address these limitations, we propose a high-quality, large-scale dataset comprising over 730,000 images across multiple categories, including both real and AI-generated images. The generated images are synthesized via state-of-the-art methods, including text-to-image generation (guided by over 10,000 carefully designed prompts), image inpainting, image refinement, and face swapping. Each generated image is annotated with its generation method and category. Inpainting images further include binary masks to indicate inpainted regions, providing rich metadata for analysis. Compared to existing datasets, detection models trained on our dataset demonstrate superior generalization capabilities. Our dataset not only serves as a strong benchmark for evaluating detection methods but also contributes to advancing the robustness of AI-generated image detection techniques. Building upon this, we propose a lightweight detection method based on image noise entropy, which transforms the original image into an entropy tensor of Non-Local Means (NLM) noise before classification. Extensive experiments demonstrate that models trained on our dataset achieve strong generalization, and our method delivers competitive performance, establishing a solid baseline for future research. The dataset and source code are publicly available at https://real-hd.github.io.

  </details>



- **MapVerse: A Benchmark for Geospatial Question Answering on Diverse Real-World Maps**  
  Sharat Bhat, Harshita Khandelwal, Tushar Kataria, Vivek Gupta  
  _2026-02-11_ · https://arxiv.org/abs/2602.10518v1  
  <details><summary>Abstract</summary>

  Maps are powerful carriers of structured and contextual knowledge, encompassing geography, demographics, infrastructure, and environmental patterns. Reasoning over such knowledge requires models to integrate spatial relationships, visual cues, real-world context, and domain-specific expertise-capabilities that current large language models (LLMs) and vision-language models (VLMs) still struggle to exhibit consistently. Yet, datasets used to benchmark VLMs on map-based reasoning remain narrow in scope, restricted to specific domains, and heavily reliant on artificially generated content (outputs from LLMs or pipeline-based methods), offering limited depth for evaluating genuine geospatial reasoning. To address this gap, we present MapVerse, a large-scale benchmark built on real-world maps. It comprises 11,837 human-authored question-answer pairs across 1,025 maps, spanning ten diverse map categories and multiple question categories for each. The dataset provides a rich setting for evaluating map reading, interpretation, and multimodal reasoning. We evaluate ten state-of-the-art models against our benchmark to establish baselines and quantify reasoning gaps. Beyond overall performance, we conduct fine-grained categorical analyses to assess model inference across multiple dimensions and investigate the visual factors shaping reasoning outcomes. Our findings reveal that while current VLMs perform competitively on classification-style tasks, both open- and closed-source models fall short on advanced tasks requiring complex spatial reasoning.

  </details>



- **Med-SegLens: Latent-Level Model Diffing for Interpretable Medical Image Segmentation**  
  Salma J. Ahmed, Emad A. Mohammed, Azam Asilian Bidgoli  
  _2026-02-11_ · https://arxiv.org/abs/2602.10508v1  
  <details><summary>Abstract</summary>

  Modern segmentation models achieve strong predictive performance but remain largely opaque, limiting our ability to diagnose failures, understand dataset shift, or intervene in a principled manner. We introduce Med-SegLens, a model-diffing framework that decomposes segmentation model activations into interpretable latent features using sparse autoencoders trained on SegFormer and U-Net. Through cross-architecture and cross-dataset latent alignment across healthy, adult, pediatric, and sub-Saharan African glioma cohorts, we identify a stable backbone of shared representations, while dataset shift is driven by differential reliance on population-specific latents. We show that these latents act as causal bottlenecks for segmentation failures, and that targeted latent-level interventions can correct errors and improve cross-dataset adaption without retraining, recovering performance in 70% of failure cases and improving Dice score from 39.4% to 74.2%. Our results demonstrate that latent-level model diffing provides a practical and mechanistic tool for diagnosing failures and mitigating dataset shift in segmentation models.

  </details>



- **Towards Long-Lived Robots: Continual Learning VLA Models via Reinforcement Fine-Tuning**  
  Yuan Liu, Haoran Li, Shuai Tian, Yuxing Qin, Yuhui Chen, Yupeng Zheng, Yongzhen Huang, Dongbin Zhao  
  _2026-02-11_ · https://arxiv.org/abs/2602.10503v1  
  <details><summary>Abstract</summary>

  Pretrained on large-scale and diverse datasets, VLA models demonstrate strong generalization and adaptability as general-purpose robotic policies. However, Supervised Fine-Tuning (SFT), which serves as the primary mechanism for adapting VLAs to downstream domains, requires substantial amounts of task-specific data and is prone to catastrophic forgetting. To address these limitations, we propose LifeLong-RFT, a simple yet effective Reinforcement Fine-Tuning (RFT) strategy for VLA models independent of online environmental feedback and pre-trained reward models. By integrating chunking-level on-policy reinforcement learning with the proposed Multi-Dimensional Process Reward (MDPR) mechanism, LifeLong-RFT quantifies the heterogeneous contributions of intermediate action chunks across three dimensions to facilitate policy optimization. Specifically, (1) the Quantized Action Consistency Reward (QACR) ensures accurate action prediction within the discrete action space; (2) the Continuous Trajectory Alignment Reward (CTAR) aligns decoded continuous action chunks with reference trajectories to ensure precise control; (3) the Format Compliance Reward (FCR) guarantees the structural validity of outputs. Comprehensive experiments across SimplerEnv, LIBERO, and real-world tasks demonstrate that LifeLong-RFT exhibits strong performance in multi-task learning. Furthermore, for continual learning on the LIBERO benchmark, our method achieves a 22% gain in average success rate over SFT, while effectively adapting to new tasks using only 20% of the training data. Overall, our method provides a promising post-training paradigm for VLAs.

  </details>



- **The Garbage Dataset (GD): A Multi-Class Image Benchmark for Automated Waste Segregation**  
  Suman Kunwar  
  _2026-02-11_ · https://arxiv.org/abs/2602.10500v1  
  <details><summary>Abstract</summary>

  This study introduces the Garbage Dataset (GD), a publicly available image dataset designed to advance automated waste segregation through machine learning and computer vision. It's a diverse dataset covering 10 common household waste categories: metal, glass, biological, paper, battery, trash, cardboard, shoes, clothes, and plastic. The dataset comprises 13,348 labeled images collected through multiple methods, including DWaste mobile app and curated web sources. Methods included rigorous validation through checksums and outlier detection, analysis of class imbalance and visual separability via PCA/t-SNE, and assessment of background complexity using entropy and saliency measures. The dataset was benchmarked using state-of-the-art deep learning models (EfficientNetV2M, EfficientNetV2S, MobileNet, ResNet50, ResNet101) evaluated on performance metrics and operational carbon emissions. Experiment results indicate EfficientNetV2S achieved the highest performance with 96.19% accuracy and a 0.96 F1-score, though with a moderate carbon cost. Analysis revealed inherent dataset characteristics including class imbalance, a skew toward high-outlier classes (plastic, cardboard, paper), and brightness variations that require consideration. The main conclusion is that GD provides a valuable, real-world benchmark for waste classification research while highlighting important challenges such as class imbalance, background complexity, and environmental trade-offs in model selection that must be addressed for practical deployment. The dataset is publicly released to support further research in environmental sustainability applications.

  </details>



- **Towards Remote Sensing Change Detection with Neural Memory**  
  Zhenyu Yang, Gensheng Pei, Yazhou Yao, Tianfei Zhou, Lizhong Ding, Fumin Shen  
  _2026-02-11_ · https://arxiv.org/abs/2602.10491v1  
  <details><summary>Abstract</summary>

  Remote sensing change detection is essential for environmental monitoring, urban planning, and related applications. However, current methods often struggle to capture long-range dependencies while maintaining computational efficiency. Although Transformers can effectively model global context, their quadratic complexity poses scalability challenges, and existing linear attention approaches frequently fail to capture intricate spatiotemporal relationships. Drawing inspiration from the recent success of Titans in language tasks, we present ChangeTitans, the Titans-based framework for remote sensing change detection. Specifically, we propose VTitans, the first Titans-based vision backbone that integrates neural memory with segmented local attention, thereby capturing long-range dependencies while mitigating computational overhead. Next, we present a hierarchical VTitans-Adapter to refine multi-scale features across different network layers. Finally, we introduce TS-CBAM, a two-stream fusion module leveraging cross-temporal attention to suppress pseudo-changes and enhance detection accuracy. Experimental evaluations on four benchmark datasets (LEVIR-CD, WHU-CD, LEVIR-CD+, and SYSU-CD) demonstrate that ChangeTitans achieves state-of-the-art results, attaining \textbf{84.36\%} IoU and \textbf{91.52\%} F1-score on LEVIR-CD, while remaining computationally competitive.

  </details>



- **HII-DPO: Eliminate Hallucination via Accurate Hallucination-Inducing Counterfactual Images**  
  Yilin Yang, Zhenghui Guo, Yuke Wang, Omprakash Gnawali, Sheng Di, Chengming Zhang  
  _2026-02-11_ · https://arxiv.org/abs/2602.10425v1  
  <details><summary>Abstract</summary>

  Large Vision-Language Models (VLMs) have achieved remarkable success across diverse multimodal tasks but remain vulnerable to hallucinations rooted in inherent language bias. Despite recent progress, existing hallucination mitigation methods often overlook the underlying hallucination patterns driven by language bias. In this work, we design a novel pipeline to accurately synthesize Hallucination-Inducing Images (HIIs). Using synthesized HIIs, we reveal a consistent scene-conditioned hallucination pattern: models tend to mention objects that are highly typical of the scene even when visual evidence is removed. To quantify the susceptibility of VLMs to this hallucination pattern, we establish the Masked-Object-Hallucination (MOH) benchmark to rigorously evaluate existing state-of-the-art alignment frameworks. Finally, we leverage HIIs to construct high-quality preference datasets for fine-grained alignment. Experimental results demonstrate that our approach effectively mitigates hallucinations while preserving general model capabilities. Specifically, our method achieves up to a 38% improvement over the current state-of-the-art on standard hallucination benchmarks.

  </details>



- **ENIGMA: EEG-to-Image in 15 Minutes Using Less Than 1% of the Parameters**  
  Reese Kneeland, Wangshu Jiang, Ugo Bruzadin Nunes, Paul Steven Scotti, Arnaud Delorme, Jonathan Xu  
  _2026-02-10_ · https://arxiv.org/abs/2602.10361v1  
  <details><summary>Abstract</summary>

  To be practical for real-life applications, models for brain-computer interfaces must be easily and quickly deployable on new subjects, effective on affordable scanning hardware, and small enough to run locally on accessible computing resources. To directly address these current limitations, we introduce ENIGMA, a multi-subject electroencephalography (EEG)-to-Image decoding model that reconstructs seen images from EEG recordings and achieves state-of-the-art (SOTA) performance on the research-grade THINGS-EEG2 and consumer-grade AllJoined-1.6M benchmarks, while fine-tuning effectively on new subjects with as little as 15 minutes of data. ENIGMA boasts a simpler architecture and requires less than 1% of the trainable parameters necessary for previous approaches. Our approach integrates a subject-unified spatio-temporal backbone along with a set of multi-subject latent alignment layers and an MLP projector to map raw EEG signals to a rich visual latent space. We evaluate our approach using a broad suite of image reconstruction metrics that have been standardized in the adjacent field of fMRI-to-Image research, and we describe the first EEG-to-Image study to conduct extensive behavioral evaluations of our reconstructions using human raters. Our simple and robust architecture provides a significant performance boost across both research-grade and consumer-grade EEG hardware, and a substantial improvement in fine-tuning efficiency and inference cost. Finally, we provide extensive ablations to determine the architectural choices most responsible for our performance gains in both single and multi-subject cases across multiple benchmark datasets. Collectively, our work provides a substantial step towards the development of practical brain-computer interface applications.

  </details>



- **Beyond Calibration: Confounding Pathology Limits Foundation Model Specificity in Abdominal Trauma CT**  
  Jineel H Raythatha, Shuchang Ye, Jeremy Hsu, Jinman Kim  
  _2026-02-10_ · https://arxiv.org/abs/2602.10359v1  
  <details><summary>Abstract</summary>

  Purpose: Translating foundation models into clinical practice requires evaluating their performance under compound distribution shift, where severe class imbalance coexists with heterogeneous imaging appearances. This challenge is relevant for traumatic bowel injury, a rare but high-mortality diagnosis. We investigated whether specificity deficits in foundation models are associated with heterogeneity in the negative class. Methods: This retrospective study used the multi-institutional, RSNA Abdominal Traumatic Injury CT dataset (2019-2023), comprising scans from 23 centres. Two foundation models (MedCLIP, zero-shot; RadDINO, linear probe) were compared against three task-specific approaches (CNN, Transformer, Ensemble). Models were trained on 3,147 patients (2.3% bowel injury prevalence) and evaluated on an enriched 100-patient test set. To isolate negative-class effects, specificity was assessed in patients without bowel injury who had concurrent solid organ injury (n=58) versus no abdominal pathology (n=50). Results: Foundation models achieved equivalent discrimination to task-specific models (AUC, 0.64-0.68 versus 0.58-0.64) with higher sensitivity (79-91% vs 41-74%) but lower specificity (33-50% vs 50-88%). All models demonstrated high specificity in patients without abdominal pathology (84-100%). When solid organ injuries were present, specificity declined substantially for foundation models (50-51 percentage points) compared with smaller reductions of 12-41 percentage points for task-specific models. Conclusion: Foundation models matched task-specific discrimination without task-specific training, but their specificity deficits were driven primarily by confounding negative-class heterogeneity rather than prevalence alone. Susceptibility to negative-class heterogeneity decreased progressively with labelled training, suggesting adaptation is required before clinical implementation.

  </details>



- **Conditional Uncertainty-Aware Political Deepfake Detection with Stochastic Convolutional Neural Networks**  
  Rafael-Petruţ Gardoş  
  _2026-02-10_ · https://arxiv.org/abs/2602.10343v1  
  <details><summary>Abstract</summary>

  Recent advances in generative image models have enabled the creation of highly realistic political deepfakes, posing risks to information integrity, public trust, and democratic processes. While automated deepfake detectors are increasingly deployed in moderation and investigative pipelines, most existing systems provide only point predictions and fail to indicate when outputs are unreliable, being an operationally critical limitation in high-stakes political contexts. This work investigates conditional, uncertainty-aware political deepfake detection using stochastic convolutional neural networks within an empirical, decision-oriented reliability framework. Rather than treating uncertainty as a purely Bayesian construct, it is evaluated through observable criteria, including calibration quality, proper scoring rules, and its alignment with prediction errors under both global and confidence-conditioned analyses. A politically focused binary image dataset is constructed via deterministic metadata filtering from a large public real-synthetic corpus. Two pretrained CNN backbones (ResNet-18 and EfficientNet-B4) are fully fine-tuned for classification. Deterministic inference is compared with single-pass stochastic prediction, Monte Carlo dropout with multiple forward passes, temperature scaling, and ensemble-based uncertainty surrogates. Evaluation reports ROC-AUC, thresholded confusion matrices, calibration metrics, and generator-disjoint out-of-distribution performance. Results demonstrate that calibrated probabilistic outputs and uncertainty estimates enable risk-aware moderation policies. A systematic confidence-band analysis further clarifies when uncertainty provides operational value beyond predicted confidence, delineating both the benefits and limitations of uncertainty-aware deepfake detection in political settings.

  </details>



- **Uncertainty-Aware Ordinal Deep Learning for cross-Dataset Diabetic Retinopathy Grading**  
  Ali El Bellaj, Aya Benradi, Salman El Youssoufi, Taha El Marzouki, Mohammed-Amine Cheddadi  
  _2026-02-10_ · https://arxiv.org/abs/2602.10315v1  
  <details><summary>Abstract</summary>

  Diabetes mellitus is a chronic metabolic disorder characterized by persistent hyperglycemia due to insufficient insulin production or impaired insulin utilization. One of its most severe complications is diabetic retinopathy (DR), a progressive retinal disease caused by microvascular damage, leading to hemorrhages, exudates, and potential vision loss. Early and reliable detection of DR is therefore critical for preventing irreversible blindness. In this work, we propose an uncertainty-aware deep learning framework for automated DR severity grading that explicitly models the ordinal nature of disease progression. Our approach combines a convolutional backbone with lesion-query attention pooling and an evidential Dirichlet-based ordinal regression head, enabling both accurate severity prediction and principled estimation of predictive uncertainty. The model is trained using an ordinal evidential loss with annealed regularization to encourage calibrated confidence under domain shift. We evaluate the proposed method on a multi-domain training setup combining APTOS, Messidor-2, and a subset of EyePACS fundus datasets. Experimental results demonstrate strong cross-dataset generalization, achieving competitive classification accuracy and high quadratic weighted kappa on held-out test sets, while providing meaningful uncertainty estimates for low-confidence cases. These results suggest that ordinal evidential learning is a promising direction for robust and clinically reliable diabetic retinopathy grading.

  </details>



- **Adaptive Time Step Flow Matching for Autonomous Driving Motion Planning**  
  Ananya Trivedi, Anjian Li, Mohamed Elnoor, Yusuf Umut Ciftci, Avinash Singh, Jovin D'sa, Sangjae Bae, David Isele, Taskin Padir, Faizan M. Tariq  
  _2026-02-10_ · https://arxiv.org/abs/2602.10285v1  
  <details><summary>Abstract</summary>

  Autonomous driving requires reasoning about interactions with surrounding traffic. A prevailing approach is large-scale imitation learning on expert driving datasets, aimed at generalizing across diverse real-world scenarios. For online trajectory generation, such methods must operate at real-time rates. Diffusion models require hundreds of denoising steps at inference, resulting in high latency. Consistency models mitigate this issue but rely on carefully tuned noise schedules to capture the multimodal action distributions common in autonomous driving. Adapting the schedule, typically requires expensive retraining. To address these limitations, we propose a framework based on conditional flow matching that jointly predicts future motions of surrounding agents and plans the ego trajectory in real time. We train a lightweight variance estimator that selects the number of inference steps online, removing the need for retraining to balance runtime and imitation learning performance. To further enhance ride quality, we introduce a trajectory post-processing step cast as a convex quadratic program, with negligible computational overhead. Trained on the Waymo Open Motion Dataset, the framework performs maneuvers such as lane changes, cruise control, and navigating unprotected left turns without requiring scenario-specific tuning. Our method maintains a 20 Hz update rate on an NVIDIA RTX 3070 GPU, making it suitable for online deployment. Compared to transformer, diffusion, and consistency model baselines, we achieve improved trajectory smoothness and better adherence to dynamic constraints. Experiment videos and code implementations can be found at https://flow-matching-self-driving.github.io/.

  </details>



- **ERGO: Excess-Risk-Guided Optimization for High-Fidelity Monocular 3D Gaussian Splatting**  
  Zehua Ma, Hanhui Li, Zhenyu Xie, Xiaonan Luo, Michael Kampffmeyer, Feng Gao, Xiaodan Liang  
  _2026-02-10_ · https://arxiv.org/abs/2602.10278v1  
  <details><summary>Abstract</summary>

  Generating 3D content from a single image remains a fundamentally challenging and ill-posed problem due to the inherent absence of geometric and textural information in occluded regions. While state-of-the-art generative models can synthesize auxiliary views to provide additional supervision, these views inevitably contain geometric inconsistencies and textural misalignments that propagate and amplify artifacts during 3D reconstruction. To effectively harness these imperfect supervisory signals, we propose an adaptive optimization framework guided by excess risk decomposition, termed ERGO. Specifically, ERGO decomposes the optimization losses in 3D Gaussian splatting into two components, i.e., excess risk that quantifies the suboptimality gap between current and optimal parameters, and Bayes error that models the irreducible noise inherent in synthesized views. This decomposition enables ERGO to dynamically estimate the view-specific excess risk and adaptively adjust loss weights during optimization. Furthermore, we introduce geometry-aware and texture-aware objectives that complement the excess-risk-derived weighting mechanism, establishing a synergistic global-local optimization paradigm. Consequently, ERGO demonstrates robustness against supervision noise while consistently enhancing both geometric fidelity and textural quality of the reconstructed 3D content. Extensive experiments on the Google Scanned Objects dataset and the OmniObject3D dataset demonstrate the superiority of ERGO over existing state-of-the-art methods.

  </details>



- **Colorimeter-Supervised Skin Tone Estimation from Dermatoscopic Images for Fairness Auditing**  
  Marin Benčević, Krešimir Romić, Ivana Hartmann Tolić, Irena Galić  
  _2026-02-10_ · https://arxiv.org/abs/2602.10265v1  
  <details><summary>Abstract</summary>

  Neural-network-based diagnosis from dermatoscopic images is increasingly used for clinical decision support, yet studies report performance disparities across skin tones. Fairness auditing of these models is limited by the lack of reliable skin-tone annotations in public dermatoscopy datasets. We address this gap with neural networks that predict Fitzpatrick skin type via ordinal regression and the Individual Typology Angle (ITA) via color regression, using in-person Fitzpatrick labels and colorimeter measurements as targets. We further leverage extensive pretraining on synthetic and real dermatoscopic and clinical images. The Fitzpatrick model achieves agreement comparable to human crowdsourced annotations, and ITA predictions show high concordance with colorimeter-derived ITA, substantially outperforming pixel-averaging approaches. Applying these estimators to ISIC 2020 and MILK10k, we find that fewer than 1% of subjects belong to Fitzpatrick types V and VI. We release code and pretrained models as an open-source tool for rapid skin-tone annotation and bias auditing. This is, to our knowledge, the first dermatoscopic skin-tone estimation neural network validated against colorimeter measurements, and it supports growing evidence of clinically relevant performance gaps across skin-tone groups.

  </details>



- **PMMA: The Polytechnique Montreal Mobility Aids Dataset**  
  Qingwu Liu, Nicolas Saunier, Guillaume-Alexandre Bilodeau  
  _2026-02-10_ · https://arxiv.org/abs/2602.10259v1  
  <details><summary>Abstract</summary>

  This study introduces a new object detection dataset of pedestrians using mobility aids, named PMMA. The dataset was collected in an outdoor environment, where volunteers used wheelchairs, canes, and walkers, resulting in nine categories of pedestrians: pedestrians, cane users, two types of walker users, whether walking or resting, five types of wheelchair users, including wheelchair users, people pushing empty wheelchairs, and three types of users pushing occupied wheelchairs, including the entire pushing group, the pusher and the person seated on the wheelchair. To establish a benchmark, seven object detection models (Faster R-CNN, CenterNet, YOLOX, DETR, Deformable DETR, DINO, and RT-DETR) and three tracking algorithms (ByteTrack, BOT-SORT, and OC-SORT) were implemented under the MMDetection framework. Experimental results show that YOLOX, Deformable DETR, and Faster R-CNN achieve the best detection performance, while the differences among the three trackers are relatively small. The PMMA dataset is publicly available at https://doi.org/10.5683/SP3/XJPQUG, and the video processing and model training code is available at https://github.com/DatasetPMMA/PMMA.

  </details>



- **When the Prompt Becomes Visual: Vision-Centric Jailbreak Attacks for Large Image Editing Models**  
  Jiacheng Hou, Yining Sun, Ruochong Jin, Haochen Han, Fangming Liu, Wai Kin Victor Chan, Alex Jinpeng Wang  
  _2026-02-10_ · https://arxiv.org/abs/2602.10179v1  
  <details><summary>Abstract</summary>

  Recent advances in large image editing models have shifted the paradigm from text-driven instructions to vision-prompt editing, where user intent is inferred directly from visual inputs such as marks, arrows, and visual-text prompts. While this paradigm greatly expands usability, it also introduces a critical and underexplored safety risk: the attack surface itself becomes visual. In this work, we propose Vision-Centric Jailbreak Attack (VJA), the first visual-to-visual jailbreak attack that conveys malicious instructions purely through visual inputs. To systematically study this emerging threat, we introduce IESBench, a safety-oriented benchmark for image editing models. Extensive experiments on IESBench demonstrate that VJA effectively compromises state-of-the-art commercial models, achieving attack success rates of up to 80.9% on Nano Banana Pro and 70.1% on GPT-Image-1.5. To mitigate this vulnerability, we propose a training-free defense based on introspective multimodal reasoning, which substantially improves the safety of poorly aligned models to a level comparable with commercial systems, without auxiliary guard models and with negligible computational overhead. Our findings expose new vulnerabilities, provide both a benchmark and practical defense to advance safe and trustworthy modern image editing systems. Warning: This paper contains offensive images created by large image editing models.

  </details>



- **SAGE: Scalable Agentic 3D Scene Generation for Embodied AI**  
  Hongchi Xia, Xuan Li, Zhaoshuo Li, Qianli Ma, Jiashu Xu, Ming-Yu Liu, Yin Cui, Tsung-Yi Lin, Wei-Chiu Ma, Shenlong Wang, et al.  
  _2026-02-10_ · https://arxiv.org/abs/2602.10116v1  
  <details><summary>Abstract</summary>

  Real-world data collection for embodied agents remains costly and unsafe, calling for scalable, realistic, and simulator-ready 3D environments. However, existing scene-generation systems often rely on rule-based or task-specific pipelines, yielding artifacts and physically invalid scenes. We present SAGE, an agentic framework that, given a user-specified embodied task (e.g., "pick up a bowl and place it on the table"), understands the intent and automatically generates simulation-ready environments at scale. The agent couples multiple generators for layout and object composition with critics that evaluate semantic plausibility, visual realism, and physical stability. Through iterative reasoning and adaptive tool selection, it self-refines the scenes until meeting user intent and physical validity. The resulting environments are realistic, diverse, and directly deployable in modern simulators for policy training. Policies trained purely on this data exhibit clear scaling trends and generalize to unseen objects and layouts, demonstrating the promise of simulation-driven scaling for embodied AI. Code, demos, and the SAGE-10k dataset can be found on the project page here: https://nvlabs.github.io/sage.

  </details>



- **Quantum Multiple Rotation Averaging**  
  Shuteng Wang, Natacha Kuete Meli, Michael Möller, Vladislav Golyanik  
  _2026-02-10_ · https://arxiv.org/abs/2602.10115v1  
  <details><summary>Abstract</summary>

  Multiple rotation averaging (MRA) is a fundamental optimization problem in 3D vision and robotics that aims to recover globally consistent absolute rotations from noisy relative measurements. Established classical methods, such as L1-IRLS and Shonan, face limitations including local minima susceptibility and reliance on convex relaxations that fail to preserve the exact manifold geometry, leading to reduced accuracy in high-noise scenarios. We introduce IQARS (Iterative Quantum Annealing for Rotation Synchronization), the first algorithm that reformulates MRA as a sequence of local quadratic non-convex sub-problems executable on quantum annealers after binarization, to leverage inherent hardware advantages. IQARS removes convex relaxation dependence and better preserves non-Euclidean rotation manifold geometry while leveraging quantum tunneling and parallelism for efficient solution space exploration. We evaluate IQARS's performance on synthetic and real-world datasets. While current annealers remain in their nascent phase and only support solving problems of limited scale with constrained performance, we observed that IQARS on D-Wave annealers can already achieve ca. 12% higher accuracy than Shonan, i.e., the best-performing classical method evaluated empirically.

  </details>



- **ConsID-Gen: View-Consistent and Identity-Preserving Image-to-Video Generation**  
  Mingyang Wu, Ashirbad Mishra, Soumik Dey, Shuo Xing, Naveen Ravipati, Hansi Wu, Binbin Li, Zhengzhong Tu  
  _2026-02-10_ · https://arxiv.org/abs/2602.10113v1  
  <details><summary>Abstract</summary>

  Image-to-Video generation (I2V) animates a static image into a temporally coherent video sequence following textual instructions, yet preserving fine-grained object identity under changing viewpoints remains a persistent challenge. Unlike text-to-video models, existing I2V pipelines often suffer from appearance drift and geometric distortion, artifacts we attribute to the sparsity of single-view 2D observations and weak cross-modal alignment. Here we address this problem from both data and model perspectives. First, we curate ConsIDVid, a large-scale object-centric dataset built with a scalable pipeline for high-quality, temporally aligned videos, and establish ConsIDVid-Bench, where we present a novel benchmarking and evaluation framework for multi-view consistency using metrics sensitive to subtle geometric and appearance deviations. We further propose ConsID-Gen, a view-assisted I2V generation framework that augments the first frame with unposed auxiliary views and fuses semantic and structural cues via a dual-stream visual-geometric encoder as well as a text-visual connector, yielding unified conditioning for a Diffusion Transformer backbone. Experiments across ConsIDVid-Bench demonstrate that ConsID-Gen consistently outperforms in multiple metrics, with the best overall performance surpassing leading video generation models like Wan2.1 and HunyuanVideo, delivering superior identity fidelity and temporal coherence under challenging real-world scenarios. We will release our model and dataset at https://myangwu.github.io/ConsID-Gen.

  </details>



- **VideoWorld 2: Learning Transferable Knowledge from Real-world Videos**  
  Zhongwei Ren, Yunchao Wei, Xiao Yu, Guixun Luo, Yao Zhao, Bingyi Kang, Jiashi Feng, Xiaojie Jin  
  _2026-02-10_ · https://arxiv.org/abs/2602.10102v1  
  <details><summary>Abstract</summary>

  Learning transferable knowledge from unlabeled video data and applying it in new environments is a fundamental capability of intelligent agents. This work presents VideoWorld 2, which extends VideoWorld and offers the first investigation into learning transferable knowledge directly from raw real-world videos. At its core, VideoWorld 2 introduces a dynamic-enhanced Latent Dynamics Model (dLDM) that decouples action dynamics from visual appearance: a pretrained video diffusion model handles visual appearance modeling, enabling the dLDM to learn latent codes that focus on compact and meaningful task-related dynamics. These latent codes are then modeled autoregressively to learn task policies and support long-horizon reasoning. We evaluate VideoWorld 2 on challenging real-world handcraft making tasks, where prior video generation and latent-dynamics models struggle to operate reliably. Remarkably, VideoWorld 2 achieves up to 70% improvement in task success rate and produces coherent long execution videos. In robotics, we show that VideoWorld 2 can acquire effective manipulation knowledge from the Open-X dataset, which substantially improves task performance on CALVIN. This study reveals the potential of learning transferable world knowledge directly from raw videos, with all code, data, and models to be open-sourced for further research.

  </details>



- **Robo3R: Enhancing Robotic Manipulation with Accurate Feed-Forward 3D Reconstruction**  
  Sizhe Yang, Linning Xu, Hao Li, Juncheng Mu, Jia Zeng, Dahua Lin, Jiangmiao Pang  
  _2026-02-10_ · https://arxiv.org/abs/2602.10101v1  
  <details><summary>Abstract</summary>

  3D spatial perception is fundamental to generalizable robotic manipulation, yet obtaining reliable, high-quality 3D geometry remains challenging. Depth sensors suffer from noise and material sensitivity, while existing reconstruction models lack the precision and metric consistency required for physical interaction. We introduce Robo3R, a feed-forward, manipulation-ready 3D reconstruction model that predicts accurate, metric-scale scene geometry directly from RGB images and robot states in real time. Robo3R jointly infers scale-invariant local geometry and relative camera poses, which are unified into the scene representation in the canonical robot frame via a learned global similarity transformation. To meet the precision demands of manipulation, Robo3R employs a masked point head for sharp, fine-grained point clouds, and a keypoint-based Perspective-n-Point (PnP) formulation to refine camera extrinsics and global alignment. Trained on Robo3R-4M, a curated large-scale synthetic dataset with four million high-fidelity annotated frames, Robo3R consistently outperforms state-of-the-art reconstruction methods and depth sensors. Across downstream tasks including imitation learning, sim-to-real transfer, grasp synthesis, and collision-free motion planning, we observe consistent gains in performance, suggesting the promise of this alternative 3D sensing module for robotic manipulation.

  </details>



- **UniVTAC: A Unified Simulation Platform for Visuo-Tactile Manipulation Data Generation, Learning, and Benchmarking**  
  Baijun Chen, Weijie Wan, Tianxing Chen, Xianda Guo, Congsheng Xu, Yuanyang Qi, Haojie Zhang, Longyan Wu, Tianling Xu, Zixuan Li, et al.  
  _2026-02-10_ · https://arxiv.org/abs/2602.10093v1  
  <details><summary>Abstract</summary>

  Robotic manipulation has seen rapid progress with vision-language-action (VLA) policies. However, visuo-tactile perception is critical for contact-rich manipulation, as tasks such as insertion are difficult to complete robustly using vision alone. At the same time, acquiring large-scale and reliable tactile data in the physical world remains costly and challenging, and the lack of a unified evaluation platform further limits policy learning and systematic analysis. To address these challenges, we propose UniVTAC, a simulation-based visuo-tactile data synthesis platform that supports three commonly used visuo-tactile sensors and enables scalable and controllable generation of informative contact interactions. Based on this platform, we introduce the UniVTAC Encoder, a visuo-tactile encoder trained on large-scale simulation-synthesized data with designed supervisory signals, providing tactile-centric visuo-tactile representations for downstream manipulation tasks. In addition, we present the UniVTAC Benchmark, which consists of eight representative visuo-tactile manipulation tasks for evaluating tactile-driven policies. Experimental results show that integrating the UniVTAC Encoder improves average success rates by 17.1% on the UniVTAC Benchmark, while real-world robotic experiments further demonstrate a 25% improvement in task success. Our webpage is available at https://univtac.github.io/.

  </details>


