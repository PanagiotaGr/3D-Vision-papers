# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-02-09 07:20 UTC_

Total papers shown: **29**


---

- **CineScene: Implicit 3D as Effective Scene Representation for Cinematic Video Generation**  
  Kaiyi Huang, Yukun Huang, Yu Li, Jianhong Bai, Xintao Wang, Zinan Lin, Xuefei Ning, Jiwen Yu, Pengfei Wan, Yu Wang, et al.  
  _2026-02-06_ · https://arxiv.org/abs/2602.06959v1  
  <details><summary>Abstract</summary>

  Cinematic video production requires control over scene-subject composition and camera movement, but live-action shooting remains costly due to the need for constructing physical sets. To address this, we introduce the task of cinematic video generation with decoupled scene context: given multiple images of a static environment, the goal is to synthesize high-quality videos featuring dynamic subject while preserving the underlying scene consistency and following a user-specified camera trajectory. We present CineScene, a framework that leverages implicit 3D-aware scene representation for cinematic video generation. Our key innovation is a novel context conditioning mechanism that injects 3D-aware features in an implicit way: By encoding scene images into visual representations through VGGT, CineScene injects spatial priors into a pretrained text-to-video generation model by additional context concatenation, enabling camera-controlled video synthesis with consistent scenes and dynamic subjects. To further enhance the model's robustness, we introduce a simple yet effective random-shuffling strategy for the input scene images during training. To address the lack of training data, we construct a scene-decoupled dataset with Unreal Engine 5, containing paired videos of scenes with and without dynamic subjects, panoramic images representing the underlying static scene, along with their camera trajectories. Experiments show that CineScene achieves state-of-the-art performance in scene-consistent cinematic video generation, handling large camera movements and demonstrating generalization across diverse environments.

  </details>



- **DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos**  
  Shenyuan Gao, William Liang, Kaiyuan Zheng, Ayaan Malik, Seonghyeon Ye, Sihyun Yu, Wei-Cheng Tseng, Yuzhu Dong, Kaichun Mo, Chen-Hsuan Lin, et al.  
  _2026-02-06_ · https://arxiv.org/abs/2602.06949v1  
  <details><summary>Abstract</summary>

  Being able to simulate the outcomes of actions in varied environments will revolutionize the development of generalist agents at scale. However, modeling these world dynamics, especially for dexterous robotics tasks, poses significant challenges due to limited data coverage and scarce action labels. As an endeavor towards this end, we introduce DreamDojo, a foundation world model that learns diverse interactions and dexterous controls from 44k hours of egocentric human videos. Our data mixture represents the largest video dataset to date for world model pretraining, spanning a wide range of daily scenarios with diverse objects and skills. To address the scarcity of action labels, we introduce continuous latent actions as unified proxy actions, enhancing interaction knowledge transfer from unlabeled videos. After post-training on small-scale target robot data, DreamDojo demonstrates a strong understanding of physics and precise action controllability. We also devise a distillation pipeline that accelerates DreamDojo to a real-time speed of 10.81 FPS and further improves context consistency. Our work enables several important applications based on generative world models, including live teleoperation, policy evaluation, and model-based planning. Systematic evaluation on multiple challenging out-of-distribution (OOD) benchmarks verifies the significance of our method for simulating open-world, contact-rich tasks, paving the way for general-purpose robot world models.

  </details>



- **Strategizing at Speed: A Learned Model Predictive Game for Multi-Agent Drone Racing**  
  Andrei-Carlo Papuc, Lasse Peters, Sihao Sun, Laura Ferranti, Javier Alonso-Mora  
  _2026-02-06_ · https://arxiv.org/abs/2602.06925v1  
  <details><summary>Abstract</summary>

  Autonomous drone racing pushes the boundaries of high-speed motion planning and multi-agent strategic decision-making. Success in this domain requires drones not only to navigate at their limits but also to anticipate and counteract competitors' actions. In this paper, we study a fundamental question that arises in this domain: how deeply should an agent strategize before taking an action? To this end, we compare two planning paradigms: the Model Predictive Game (MPG), which finds interaction-aware strategies at the expense of longer computation times, and contouring Model Predictive Control (MPC), which computes strategies rapidly but does not reason about interactions. We perform extensive experiments to study this trade-off, revealing that MPG outperforms MPC at moderate velocities but loses its advantage at higher speeds due to latency. To address this shortcoming, we propose a Learned Model Predictive Game (LMPG) approach that amortizes model predictive gameplay to reduce latency. In both simulation and hardware experiments, we benchmark our approach against MPG and MPC in head-to-head races, finding that LMPG outperforms both baselines.

  </details>



- **Seeing Beyond Redundancy: Task Complexity's Role in Vision Token Specialization in VLLMs**  
  Darryl Hannan, John Cooper, Dylan White, Yijing Watkins  
  _2026-02-06_ · https://arxiv.org/abs/2602.06914v1  
  <details><summary>Abstract</summary>

  Vision capabilities in vision large language models (VLLMs) have consistently lagged behind their linguistic capabilities. In particular, numerous benchmark studies have demonstrated that VLLMs struggle when fine-grained visual information or spatial reasoning is required. However, we do not yet understand exactly why VLLMs struggle so much with these tasks relative to others. Some works have focused on visual redundancy as an explanation, where high-level visual information is uniformly spread across numerous tokens and specific, fine-grained visual information is discarded. In this work, we investigate this premise in greater detail, seeking to better understand exactly how various types of visual information are processed by the model and what types of visual information are discarded. To do so, we introduce a simple synthetic benchmark dataset that is specifically constructed to probe various visual features, along with a set of metrics for measuring visual redundancy, allowing us to better understand the nuances of their relationship. Then, we explore fine-tuning VLLMs on a number of complex visual tasks to better understand how redundancy and compression change based upon the complexity of the data that a model is trained on. We find that there is a connection between task complexity and visual compression, implying that having a sufficient ratio of high complexity visual data is crucial for altering the way that VLLMs distribute their visual representation and consequently improving their performance on complex visual tasks. We hope that this work will provide valuable insights for training the next generation of VLLMs.

  </details>



- **PANC: Prior-Aware Normalized Cut for Object Segmentation**  
  Juan Gutiérrez, Victor Gutiérrez-Garcia, José Luis Blanco-Murillo  
  _2026-02-06_ · https://arxiv.org/abs/2602.06912v1  
  <details><summary>Abstract</summary>

  Fully unsupervised segmentation pipelines naively seek the most salient object, should this be present. As a result, most of the methods reported in the literature deliver non-deterministic partitions that are sensitive to initialization, seed order, and threshold heuristics. We propose PANC, a weakly supervised spectral segmentation framework that uses a minimal set of annotated visual tokens to produce stable, controllable, and reproducible object masks. From the TokenCut approach, we augment the token-token affinity graph with a handful of priors coupled to anchor nodes. By manipulating the graph topology, we bias the spectral eigenspace toward partitions that are consistent with the annotations. Our approach preserves the global grouping enforced by dense self-supervised visual features, trading annotated tokens for significant gains in reproducibility, user control, and segmentation quality. Using 5 to 30 annotations per dataset, our training-free method achieves state-of-the-art performance among weakly and unsupervised approaches on standard benchmarks (e.g., DUTS-TE, ECSSD, MS COCO). Contrarily, it excels in domains where dense labels are costly or intra-class differences are subtle. We report strong and reliable results on homogeneous, fine-grained, and texture-limited domains, achieving 96.8% (+14.43% over SotA), 78.0% (+0.2%), and 78.8% (+0.37%) average mean intersection-over-union (mIoU) on CrackForest (CFD), CUB-200-2011, and HAM10000 datasets, respectively. For multi-object benchmarks, the framework showcases explicit, user-controllable semantic segmentation.

  </details>



- **RFDM: Residual Flow Diffusion Model for Efficient Causal Video Editing**  
  Mohammadreza Salehi, Mehdi Noroozi, Luca Morreale, Ruchika Chavhan, Malcolm Chadwick, Alberto Gil Ramos, Abhinav Mehrotra  
  _2026-02-06_ · https://arxiv.org/abs/2602.06871v1  
  <details><summary>Abstract</summary>

  Instructional video editing applies edits to an input video using only text prompts, enabling intuitive natural-language control. Despite rapid progress, most methods still require fixed-length inputs and substantial compute. Meanwhile, autoregressive video generation enables efficient variable-length synthesis, yet remains under-explored for video editing. We introduce a causal, efficient video editing model that edits variable-length videos frame by frame. For efficiency, we start from a 2D image-to-image (I2I) diffusion model and adapt it to video-to-video (V2V) editing by conditioning the edit at time step t on the model's prediction at t-1. To leverage videos' temporal redundancy, we propose a new I2I diffusion forward process formulation that encourages the model to predict the residual between the target output and the previous prediction. We call this Residual Flow Diffusion Model (RFDM), which focuses the denoising process on changes between consecutive frames. Moreover, we propose a new benchmark that better ranks state-of-the-art methods for editing tasks. Trained on paired video data for global/local style transfer and object removal, RFDM surpasses I2I-based methods and competes with fully spatiotemporal (3D) V2V models, while matching the compute of image models and scaling independently of input video length. More content can be found in: https://smsd75.github.io/RFDM_page/

  </details>



- **DynaRetarget: Dynamically-Feasible Retargeting using Sampling-Based Trajectory Optimization**  
  Victor Dhedin, Ilyass Taouil, Shafeef Omar, Dian Yu, Kun Tao, Angela Dai, Majid Khadiv  
  _2026-02-06_ · https://arxiv.org/abs/2602.06827v1  
  <details><summary>Abstract</summary>

  In this paper, we introduce DynaRetarget, a complete pipeline for retargeting human motions to humanoid control policies. The core component of DynaRetarget is a novel Sampling-Based Trajectory Optimization (SBTO) framework that refines imperfect kinematic trajectories into dynamically feasible motions. SBTO incrementally advances the optimization horizon, enabling optimization over the entire trajectory for long-horizon tasks. We validate DynaRetarget by successfully retargeting hundreds of humanoid-object demonstrations and achieving higher success rates than the state of the art. The framework also generalizes across varying object properties, such as mass, size, and geometry, using the same tracking objective. This ability to robustly retarget diverse demonstrations opens the door to generating large-scale synthetic datasets of humanoid loco-manipulation trajectories, addressing a major bottleneck in real-world data collection.

  </details>



- **Machine Learning for Detection and Severity Estimation of Sweetpotato Weevil Damage in Field and Lab Conditions**  
  Doreen M. Chelangat, Sudi Murindanyi, Bruce Mugizi, Paul Musana, Benard Yada, Milton A. Otema, Florence Osaru, Andrew Katumba, Joyce Nakatumba-Nabende  
  _2026-02-06_ · https://arxiv.org/abs/2602.06786v1  
  <details><summary>Abstract</summary>

  Sweetpotato weevils (Cylas spp.) are considered among the most destructive pests impacting sweetpotato production, particularly in sub-Saharan Africa. Traditional methods for assessing weevil damage, predominantly relying on manual scoring, are labour-intensive, subjective, and often yield inconsistent results. These challenges significantly hinder breeding programs aimed at developing resilient sweetpotato varieties. This study introduces a computer vision-based approach for the automated evaluation of weevil damage in both field and laboratory contexts. In the field settings, we collected data to train classification models to predict root-damage severity levels, achieving a test accuracy of 71.43%. Additionally, we established a laboratory dataset and designed an object detection pipeline employing YOLO12, a leading real-time detection model. This methodology incorporated a two-stage laboratory pipeline that combined root segmentation with a tiling strategy to improve the detectability of small objects. The resulting model demonstrated a mean average precision of 77.7% in identifying minute weevil feeding holes. Our findings indicate that computer vision technologies can provide efficient, objective, and scalable assessment tools that align seamlessly with contemporary breeding workflows. These advancements represent a significant improvement in enhancing phenotyping efficiency within sweetpotato breeding programs and play a crucial role in mitigating the detrimental effects of weevils on food security.

  </details>



- **Revisiting Emotions Representation for Recognition in the Wild**  
  Joao Baptista Cardia Neto, Claudio Ferrari, Stefano Berretti  
  _2026-02-06_ · https://arxiv.org/abs/2602.06778v1  
  <details><summary>Abstract</summary>

  Facial emotion recognition has been typically cast as a single-label classification problem of one out of six prototypical emotions. However, that is an oversimplification that is unsuitable for representing the multifaceted spectrum of spontaneous emotional states, which are most often the result of a combination of multiple emotions contributing at different intensities. Building on this, a promising direction that was explored recently is to cast emotion recognition as a distribution learning problem. Still, such approaches are limited in that research datasets are typically annotated with a single emotion class. In this paper, we contribute a novel approach to describe complex emotional states as probability distributions over a set of emotion classes. To do so, we propose a solution to automatically re-label existing datasets by exploiting the result of a study in which a large set of both basic and compound emotions is mapped to probability distributions in the Valence-Arousal-Dominance (VAD) space. In this way, given a face image annotated with VAD values, we can estimate the likelihood of it belonging to each of the distributions, so that emotional states can be described as a mixture of emotions, enriching their description, while also accounting for the ambiguous nature of their perception. In a preliminary set of experiments, we illustrate the advantages of this solution and a new possible direction of investigation. Data annotations are available at https://github.com/jbcnrlz/affectnet-b-annotation.

  </details>



- **Orientation-Robust Latent Motion Trajectory Learning for Annotation-free Cardiac Phase Detection in Fetal Echocardiography**  
  Yingyu Yang, Qianye Yang, Can Peng, Elena D'Alberti, Olga Patey, Aris T. Papageorghiou, J. Alison Noble  
  _2026-02-06_ · https://arxiv.org/abs/2602.06761v1  
  <details><summary>Abstract</summary>

  Fetal echocardiography is essential for detecting congenital heart disease (CHD), facilitating pregnancy management, optimized delivery planning, and timely postnatal interventions. Among standard imaging planes, the four-chamber (4CH) view provides comprehensive information for CHD diagnosis, where clinicians carefully inspect the end-diastolic (ED) and end-systolic (ES) phases to evaluate cardiac structure and motion. Automated detection of these cardiac phases is thus a critical component toward fully automated CHD analysis. Yet, in the absence of fetal electrocardiography (ECG), manual identification of ED and ES frames remains a labor-intensive bottleneck. We present ORBIT (Orientation-Robust Beat Inference from Trajectories), a self-supervised framework that identifies cardiac phases without manual annotations under various fetal heart orientation. ORBIT employs registration as self-supervision task and learns a latent motion trajectory of cardiac deformation, whose turning points capture transitions between cardiac relaxation and contraction, enabling accurate and orientation-robust localization of ED and ES frames across diverse fetal positions. Trained exclusively on normal fetal echocardiography videos, ORBIT achieves consistent performance on both normal (MAE = 1.9 frames for ED and 1.6 for ES) and CHD cases (MAE = 2.4 frames for ED and 2.1 for ES), outperforming existing annotation-free approaches constrained by fixed orientation assumptions. These results highlight the potential of ORBIT to facilitate robust cardiac phase detection directly from 4CH fetal echocardiography.

  </details>



- **Gold Exploration using Representations from a Multispectral Autoencoder**  
  Argyro Tsandalidou, Konstantinos Dogeas, Eleftheria Tetoula Tsonga, Elisavet Parselia, Georgios Tsimiklis, George Arvanitakis  
  _2026-02-06_ · https://arxiv.org/abs/2602.06748v1  
  <details><summary>Abstract</summary>

  Satellite imagery is employed for large-scale prospectivity mapping due to the high cost and typically limited availability of on-site mineral exploration data. In this work, we present a proof-of-concept framework that leverages generative representations learned from multispectral Sentinel-2 imagery to identify gold-bearing regions from space. An autoencoder foundation model, called Isometric, which is pretrained on the large-scale FalconSpace-S2 v1.0 dataset, produces information-dense spectral-spatial representations that serve as inputs to a lightweight XGBoost classifier. We compare this representation-based approach with a raw spectral input baseline using a dataset of 63 Sentinel-2 images from known gold and non-gold locations. The proposed method improves patch-level accuracy from 0.51 to 0.68 and image-level accuracy from 0.55 to 0.73, demonstrating that generative embeddings capture transferable mineralogical patterns even with limited labeled data. These results highlight the potential of foundation-model representations to make mineral exploration more efficient, scalable, and globally applicable.

  </details>



- **Clinical-Prior Guided Multi-Modal Learning with Latent Attention Pooling for Gait-Based Scoliosis Screening**  
  Dong Chen, Zizhuang Wei, Jialei Xu, Xinyang Sun, Zonglin He, Meiru An, Huili Peng, Yong Hu, Kenneth MC Cheung  
  _2026-02-06_ · https://arxiv.org/abs/2602.06743v1  
  <details><summary>Abstract</summary>

  Adolescent Idiopathic Scoliosis (AIS) is a prevalent spinal deformity whose progression can be mitigated through early detection. Conventional screening methods are often subjective, difficult to scale, and reliant on specialized clinical expertise. Video-based gait analysis offers a promising alternative, but current datasets and methods frequently suffer from data leakage, where performance is inflated by repeated clips from the same individual, or employ oversimplified models that lack clinical interpretability. To address these limitations, we introduce ScoliGait, a new benchmark dataset comprising 1,572 gait video clips for training and 300 fully independent clips for testing. Each clip is annotated with radiographic Cobb angles and descriptive text based on clinical kinematic priors. We propose a multi-modal framework that integrates a clinical-prior-guided kinematic knowledge map for interpretable feature representation, alongside a latent attention pooling mechanism to fuse video, text, and knowledge map modalities. Our method establishes a new state-of-the-art, demonstrating a significant performance gap on a realistic, non-repeating subject benchmark. Our approach establishes a new state of the art, showing a significant performance gain on a realistic, subject-independent benchmark. This work provides a robust, interpretable, and clinically grounded foundation for scalable, non-invasive AIS assessment.

  </details>



- **Crowd-FM: Learned Optimal Selection of Conditional Flow Matching-generated Trajectories for Crowd Navigation**  
  Antareep Singha, Laksh Nanwani, Mathai Mathew P., Samkit Jain, Phani Teja Singamaneni, Arun Kumar Singh, K. Madhava Krishna  
  _2026-02-06_ · https://arxiv.org/abs/2602.06698v1  
  <details><summary>Abstract</summary>

  Safe and computationally efficient local planning for mobile robots in dense, unstructured human crowds remains a fundamental challenge. Moreover, ensuring that robot trajectories are similar to how a human moves will increase the acceptance of the robot in human environments. In this paper, we present Crowd-FM, a learning-based approach to address both safety and human-likeness challenges. Our approach has two novel components. First, we train a Conditional Flow-Matching (CFM) policy over a dataset of optimally controlled trajectories to learn a set of collision-free primitives that a robot can choose at any given scenario. The chosen optimal control solver can generate multi-modal collision-free trajectories, allowing the CFM policy to learn a diverse set of maneuvers. Secondly, we learn a score function over a dataset of human demonstration trajectories that provides a human-likeness score for the flow primitives. At inference time, computing the optimal trajectory requires selecting the one with the highest score. Our approach improves the state-of-the-art by showing that our CFM policy alone can produce collision-free navigation with a higher success rate than existing learning-based baselines. Furthermore, when augmented with inference-time refinement, our approach can outperform even expensive optimisation-based planning approaches. Finally, we validate that our scoring network can select trajectories closer to the expert data than a manually designed cost function.

  </details>



- **Can We Build a Monolithic Model for Fake Image Detection? SICA: Semantic-Induced Constrained Adaptation for Unified-Yet-Discriminative Artifact Feature Space Reconstruction**  
  Bo Du, Xiaochen Ma, Xuekang Zhu, Zhe Yang, Chaogun Niu, Jian Liu, Ji-Zhe Zhou  
  _2026-02-06_ · https://arxiv.org/abs/2602.06676v1  
  <details><summary>Abstract</summary>

  Fake Image Detection (FID), aiming at unified detection across four image forensic subdomains, is critical in real-world forensic scenarios. Compared with ensemble approaches, monolithic FID models are theoretically more promising, but to date, consistently yield inferior performance in practice. In this work, by discovering the ``heterogeneous phenomenon'', which is the intrinsic distinctness of artifacts across subdomains, we diagnose the cause of this underperformance for the first time: the collapse of the artifact feature space driven by such phenomenon. The core challenge for developing a practical monolithic FID model thus boils down to the ``unified-yet-discriminative" reconstruction of the artifact feature space. To address this paradoxical challenge, we hypothesize that high-level semantics can serve as a structural prior for the reconstruction, and further propose Semantic-Induced Constrained Adaptation (SICA), the first monolithic FID paradigm. Extensive experiments on our OpenMMSec dataset demonstrate that SICA outperforms 15 state-of-the-art methods and reconstructs the target unified-yet-discriminative artifact feature space in a near-orthogonal manner, thus firmly validating our hypothesis. The code and dataset are available at:https: //github.com/scu-zjz/SICA_OpenMMSec.

  </details>



- **CytoCrowd: A Multi-Annotator Benchmark Dataset for Cytology Image Analysis**  
  Yonghao Si, Xingyuan Zeng, Zhao Chen, Libin Zheng, Caleb Chen Cao, Lei Chen, Jian Yin  
  _2026-02-06_ · https://arxiv.org/abs/2602.06674v1  
  <details><summary>Abstract</summary>

  High-quality annotated datasets are crucial for advancing machine learning in medical image analysis. However, a critical gap exists: most datasets either offer a single, clean ground truth, which hides real-world expert disagreement, or they provide multiple annotations without a separate gold standard for objective evaluation. To bridge this gap, we introduce CytoCrowd, a new public benchmark for cytology analysis. The dataset features 446 high-resolution images, each with two key components: (1) raw, conflicting annotations from four independent pathologists, and (2) a separate, high-quality gold-standard ground truth established by a senior expert. This dual structure makes CytoCrowd a versatile resource. It serves as a benchmark for standard computer vision tasks, such as object detection and classification, using the ground truth. Simultaneously, it provides a realistic testbed for evaluating annotation aggregation algorithms that must resolve expert disagreements. We provide comprehensive baseline results for both tasks. Our experiments demonstrate the challenges presented by CytoCrowd and establish its value as a resource for developing the next generation of models for medical image analysis.

  </details>



- **PlanViz: Evaluating Planning-Oriented Image Generation and Editing for Computer-Use Tasks**  
  Junxian Li, Kai Liu, Leyang Chen, Weida Wang, Zhixin Wang, Jiaqi Xu, Fan Li, Renjing Pei, Linghe Kong, Yulun Zhang  
  _2026-02-06_ · https://arxiv.org/abs/2602.06663v1  
  <details><summary>Abstract</summary>

  Unified multimodal models (UMMs) have shown impressive capabilities in generating natural images and supporting multimodal reasoning. However, their potential in supporting computer-use planning tasks, which are closely related to our lives, remain underexplored. Image generation and editing in computer-use tasks require capabilities like spatial reasoning and procedural understanding, and it is still unknown whether UMMs have these capabilities to finish these tasks or not. Therefore, we propose PlanViz, a new benchmark designed to evaluate image generation and editing for computer-use tasks. To achieve the goal of our evaluation, we focus on sub-tasks which frequently involve in daily life and require planning steps. Specifically, three new sub-tasks are designed: route planning, work diagramming, and web&UI displaying. We address challenges in data quality ensuring by curating human-annotated questions and reference images, and a quality control process. For challenges of comprehensive and exact evaluation, a task-adaptive score, PlanScore, is proposed. The score helps understanding the correctness, visual quality and efficiency of generated images. Through experiments, we highlight key limitations and opportunities for future research on this topic.

  </details>



- **CauCLIP: Bridging the Sim-to-Real Gap in Surgical Video Understanding via Causality-Inspired Vision-Language Modeling**  
  Yuxin He, An Li, Cheng Xue  
  _2026-02-06_ · https://arxiv.org/abs/2602.06619v1  
  <details><summary>Abstract</summary>

  Surgical phase recognition is a critical component for context-aware decision support in intelligent operating rooms, yet training robust models is hindered by limited annotated clinical videos and large domain gaps between synthetic and real surgical data. To address this, we propose CauCLIP, a causality-inspired vision-language framework that leverages CLIP to learn domain-invariant representations for surgical phase recognition without access to target domain data. Our approach integrates a frequency-based augmentation strategy to perturb domain-specific attributes while preserving semantic structures, and a causal suppression loss that mitigates non-causal biases and reinforces causal surgical features. These components are combined in a unified training framework that enables the model to focus on stable causal factors underlying surgical workflows. Experiments on the SurgVisDom hard adaptation benchmark demonstrate that our method substantially outperforms all competing approaches, highlighting the effectiveness of causality-guided vision-language models for domain-generalizable surgical video understanding.

  </details>



- **ProtoQuant: Quantization of Prototypical Parts For General and Fine-Grained Image Classification**  
  Mikołaj Janusz, Adam Wróbel, Bartosz Zieliński, Dawid Rymarczyk  
  _2026-02-06_ · https://arxiv.org/abs/2602.06592v1  
  <details><summary>Abstract</summary>

  Prototypical parts-based models offer a "this looks like that" paradigm for intrinsic interpretability, yet they typically struggle with ImageNet-scale generalization and often require computationally expensive backbone finetuning. Furthermore, existing methods frequently suffer from "prototype drift," where learned prototypes lack tangible grounding in the training distribution and change their activation under small perturbations. We present ProtoQuant, a novel architecture that achieves prototype stability and grounded interpretability through latent vector quantization. By constraining prototypes to a discrete learned codebook within the latent space, we ensure they remain faithful representations of the training data without the need to update the backbone. This design allows ProtoQuant to function as an efficient, interpretable head that scales to large-scale datasets. We evaluate ProtoQuant on ImageNet and several fine-grained benchmarks (CUB-200, Cars-196). Our results demonstrate that ProtoQuant achieves competitive classification accuracy while generalizing to ImageNet and comparable interpretability metrics to other prototypical-parts-based methods.

  </details>



- **SPARC: Separating Perception And Reasoning Circuits for Test-time Scaling of VLMs**  
  Niccolo Avogaro, Nayanika Debnath, Li Mi, Thomas Frick, Junling Wang, Zexue He, Hang Hua, Konrad Schindler, Mattia Rigotti  
  _2026-02-06_ · https://arxiv.org/abs/2602.06566v1  
  <details><summary>Abstract</summary>

  Despite recent successes, test-time scaling - i.e., dynamically expanding the token budget during inference as needed - remains brittle for vision-language models (VLMs): unstructured chains-of-thought about images entangle perception and reasoning, leading to long, disorganized contexts where small perceptual mistakes may cascade into completely wrong answers. Moreover, expensive reinforcement learning with hand-crafted rewards is required to achieve good performance. Here, we introduce SPARC (Separating Perception And Reasoning Circuits), a modular framework that explicitly decouples visual perception from reasoning. Inspired by sequential sensory-to-cognitive processing in the brain, SPARC implements a two-stage pipeline where the model first performs explicit visual search to localize question-relevant regions, then conditions its reasoning on those regions to produce the final answer. This separation enables independent test-time scaling with asymmetric compute allocation (e.g., prioritizing perceptual processing under distribution shift), supports selective optimization (e.g., improving the perceptual stage alone when it is the bottleneck for end-to-end performance), and accommodates compressed contexts by running global search at lower image resolutions and allocating high-resolution processing only to selected regions, thereby reducing total visual tokens count and compute. Across challenging visual reasoning benchmarks, SPARC outperforms monolithic baselines and strong visual-grounding approaches. For instance, SPARC improves the accuracy of Qwen3VL-4B on the $V^*$ VQA benchmark by 6.7 percentage points, and it surpasses "thinking with images" by 4.6 points on a challenging OOD task despite requiring a 200$\times$ lower token budget.

  </details>



- **LIBERO-X: Robustness Litmus for Vision-Language-Action Models**  
  Guodong Wang, Chenkai Zhang, Qingjie Liu, Jinjin Zhang, Jiancheng Cai, Junjie Liu, Xinmin Liu  
  _2026-02-06_ · https://arxiv.org/abs/2602.06556v1  
  <details><summary>Abstract</summary>

  Reliable benchmarking is critical for advancing Vision-Language-Action (VLA) models, as it reveals their generalization, robustness, and alignment of perception with language-driven manipulation tasks. However, existing benchmarks often provide limited or misleading assessments due to insufficient evaluation protocols that inadequately capture real-world distribution shifts. This work systematically rethinks VLA benchmarking from both evaluation and data perspectives, introducing LIBERO-X, a benchmark featuring: 1) A hierarchical evaluation protocol with progressive difficulty levels targeting three core capabilities: spatial generalization, object recognition, and task instruction understanding. This design enables fine-grained analysis of performance degradation under increasing environmental and task complexity; 2) A high-diversity training dataset collected via human teleoperation, where each scene supports multiple fine-grained manipulation objectives to bridge the train-evaluation distribution gap. Experiments with representative VLA models reveal significant performance drops under cumulative perturbations, exposing persistent limitations in scene comprehension and instruction grounding. By integrating hierarchical evaluation with diverse training data, LIBERO-X offers a more reliable foundation for assessing and advancing VLA development.

  </details>



- **NECromancer: Breathing Life into Skeletons via BVH Animation**  
  Mingxi Xu, Qi Wang, Zhengyu Wen, Phong Dao Thien, Zhengyu Li, Ning Zhang, Xiaoyu He, Wei Zhao, Kehong Gong, Mingyuan Zhang  
  _2026-02-06_ · https://arxiv.org/abs/2602.06548v1  
  <details><summary>Abstract</summary>

  Motion tokenization is a key component of generalizable motion models, yet most existing approaches are restricted to species-specific skeletons, limiting their applicability across diverse morphologies. We propose NECromancer (NEC), a universal motion tokenizer that operates directly on arbitrary BVH skeletons. NEC consists of three components: (1) an Ontology-aware Skeletal Graph Encoder (OwO) that encodes structural priors from BVH files, including joint semantics, rest-pose offsets, and skeletal topology, into skeletal embeddings; (2) a Topology-Agnostic Tokenizer (TAT) that compresses motion sequences into a universal, topology-invariant discrete representation; and (3) the Unified BVH Universe (UvU), a large-scale dataset aggregating BVH motions across heterogeneous skeletons. Experiments show that NEC achieves high-fidelity reconstruction under substantial compression and effectively disentangles motion from skeletal structure. The resulting token space supports cross-species motion transfer, composition, denoising, generation with token-based models, and text-motion retrieval, establishing a unified framework for motion analysis and synthesis across diverse morphologies. Demo page: https://animotionlab.github.io/NECromancer/

  </details>



- **Universal Anti-forensics Attack against Image Forgery Detection via Multi-modal Guidance**  
  Haipeng Li, Rongxuan Peng, Anwei Luo, Shunquan Tan, Changsheng Chen, Anastasia Antsiferova  
  _2026-02-06_ · https://arxiv.org/abs/2602.06530v1  
  <details><summary>Abstract</summary>

  The rapid advancement of AI-Generated Content (AIGC) technologies poses significant challenges for authenticity assessment. However, existing evaluation protocols largely overlook anti-forensics attack, failing to ensure the comprehensive robustness of state-of-the-art AIGC detectors in real-world applications. To bridge this gap, we propose ForgeryEraser, a framework designed to execute universal anti-forensics attack without access to the target AIGC detectors. We reveal an adversarial vulnerability stemming from the systemic reliance on Vision-Language Models (VLMs) as shared backbones (e.g., CLIP), where downstream AIGC detectors inherit the feature space of these publicly accessible models. Instead of traditional logit-based optimization, we design a multi-modal guidance loss to drive forged image embeddings within the VLM feature space toward text-derived authentic anchors to erase forgery traces, while repelling them from forgery anchors. Extensive experiments demonstrate that ForgeryEraser causes substantial performance degradation to advanced AIGC detectors on both global synthesis and local editing benchmarks. Moreover, ForgeryEraser induces explainable forensic models to generate explanations consistent with authentic images for forged images. Our code will be made publicly available.

  </details>



- **AdaptOVCD: Training-Free Open-Vocabulary Remote Sensing Change Detection via Adaptive Information Fusion**  
  Mingyu Dou, Shi Qiu, Ming Hu, Yifan Chen, Huping Ye, Xiaohan Liao, Zhe Sun  
  _2026-02-06_ · https://arxiv.org/abs/2602.06529v1  
  <details><summary>Abstract</summary>

  Remote sensing change detection plays a pivotal role in domains such as environmental monitoring, urban planning, and disaster assessment. However, existing methods typically rely on predefined categories and large-scale pixel-level annotations, which limit their generalization and applicability in open-world scenarios. To address these limitations, this paper proposes AdaptOVCD, a training-free Open-Vocabulary Change Detection (OVCD) architecture based on dual-dimensional multi-level information fusion. The framework integrates multi-level information fusion across data, feature, and decision levels vertically while incorporating targeted adaptive designs horizontally, achieving deep synergy among heterogeneous pre-trained models to effectively mitigate error propagation. Specifically, (1) at the data level, Adaptive Radiometric Alignment (ARA) fuses radiometric statistics with original texture features and synergizes with SAM-HQ to achieve radiometrically consistent segmentation; (2) at the feature level, Adaptive Change Thresholding (ACT) combines global difference distributions with edge structure priors and leverages DINOv3 to achieve robust change detection; (3) at the decision level, Adaptive Confidence Filtering (ACF) integrates semantic confidence with spatial constraints and collaborates with DGTRS-CLIP to achieve high-confidence semantic identification. Comprehensive evaluations across nine scenarios demonstrate that AdaptOVCD detects arbitrary category changes in a zero-shot manner, significantly outperforming existing training-free methods. Meanwhile, it achieves 84.89\% of the fully-supervised performance upper bound in cross-dataset evaluations and exhibits superior generalization capabilities. The code is available at https://github.com/Dmygithub/AdaptOVCD.

  </details>



- **World-VLA-Loop: Closed-Loop Learning of Video World Model and VLA Policy**  
  Xiaokang Liu, Zechen Bai, Hai Ci, Kevin Yuchen Ma, Mike Zheng Shou  
  _2026-02-06_ · https://arxiv.org/abs/2602.06508v1  
  <details><summary>Abstract</summary>

  Recent progress in robotic world models has leveraged video diffusion transformers to predict future observations conditioned on historical states and actions. While these models can simulate realistic visual outcomes, they often exhibit poor action-following precision, hindering their utility for downstream robotic learning. In this work, we introduce World-VLA-Loop, a closed-loop framework for the joint refinement of world models and Vision-Language-Action (VLA) policies. We propose a state-aware video world model that functions as a high-fidelity interactive simulator by jointly predicting future observations and reward signals. To enhance reliability, we introduce the SANS dataset, which incorporates near-success trajectories to improve action-outcome alignment within the world model. This framework enables a closed-loop for reinforcement learning (RL) post-training of VLA policies entirely within a virtual environment. Crucially, our approach facilitates a co-evolving cycle: failure rollouts generated by the VLA policy are iteratively fed back to refine the world model precision, which in turn enhances subsequent RL optimization. Evaluations across simulation and real-world tasks demonstrate that our framework significantly boosts VLA performance with minimal physical interaction, establishing a mutually beneficial relationship between world modeling and policy learning for general-purpose robotics. Project page: https://showlab.github.io/World-VLA-Loop/.

  </details>



- **FloorplanVLM: A Vision-Language Model for Floorplan Vectorization**  
  Yuanqing Liu, Ziming Yang, Yulong Li, Yue Yang  
  _2026-02-06_ · https://arxiv.org/abs/2602.06507v1  
  <details><summary>Abstract</summary>

  Converting raster floorplans into engineering-grade vector graphics is challenging due to complex topology and strict geometric constraints. To address this, we present FloorplanVLM, a unified framework that reformulates floorplan vectorization as an image-conditioned sequence modeling task. Unlike pixel-based methods that rely on fragile heuristics or query-based transformers that generate fragmented rooms, our model directly outputs structured JSON sequences representing the global topology. This 'pixels-to-sequence' paradigm enables the precise and holistic constraint satisfaction of complex geometries, such as slanted walls and curved arcs. To support this data-hungry approach, we introduce a scalable data engine: we construct a large-scale dataset (Floorplan-2M) and a high-fidelity subset (Floorplan-HQ-300K) to balance geometric diversity and pixel-level precision. We then employ a progressive training strategy, using Supervised Fine-Tuning (SFT) for structural grounding and quality annealing, followed by Group Relative Policy Optimization (GRPO) for strict geometric alignment. To standardize evaluation on complex layouts, we establish and open-source FPBench-2K. Evaluated on this rigorous benchmark, FloorplanVLM demonstrates exceptional structural validity, achieving $\textbf{92.52%}$ external-wall IoU and robust generalization across non-Manhattan architectures.

  </details>



- **DreamHome-Pano: Design-Aware and Conflict-Free Panoramic Interior Generation**  
  Lulu Chen, Yijiang Hu, Yuanqing Liu, Yulong Li, Yue Yang  
  _2026-02-06_ · https://arxiv.org/abs/2602.06494v1  
  <details><summary>Abstract</summary>

  In modern interior design, the generation of personalized spaces frequently necessitates a delicate balance between rigid architectural structural constraints and specific stylistic preferences. However, existing multi-condition generative frameworks often struggle to harmonize these inputs, leading to "condition conflicts" where stylistic attributes inadvertently compromise the geometric precision of the layout. To address this challenge, we present DreamHome-Pano, a controllable panoramic generation framework designed for high-fidelity interior synthesis. Our approach introduces a Prompt-LLM that serves as a semantic bridge, effectively translating layout constraints and style references into professional descriptive prompts to achieve precise cross-modal alignment. To safeguard architectural integrity during the generative process, we develop a Conflict-Free Control architecture that incorporates structural-aware geometric priors and a multi-condition decoupling strategy, effectively suppressing stylistic interference from eroding the spatial layout. Furthermore, we establish a comprehensive panoramic interior benchmark alongside a multi-stage training pipeline, encompassing progressive Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL). Experimental results demonstrate that DreamHome-Pano achieves a superior balance between aesthetic quality and structural consistency, offering a robust and professional-grade solution for panoramic interior visualization.

  </details>



- **Rebenchmarking Unsupervised Monocular 3D Occupancy Prediction**  
  Zizhan Guo, Yi Feng, Mengtan Zhang, Haoran Zhang, Wei Ye, Rui Fan  
  _2026-02-06_ · https://arxiv.org/abs/2602.06488v1  
  <details><summary>Abstract</summary>

  Inferring the 3D structure from a single image, particularly in occluded regions, remains a fundamental yet unsolved challenge in vision-centric autonomous driving. Existing unsupervised approaches typically train a neural radiance field and treat the network outputs as occupancy probabilities during evaluation, overlooking the inconsistency between training and evaluation protocols. Moreover, the prevalent use of 2D ground truth fails to reveal the inherent ambiguity in occluded regions caused by insufficient geometric constraints. To address these issues, this paper presents a reformulated benchmark for unsupervised monocular 3D occupancy prediction. We first interpret the variables involved in the volume rendering process and identify the most physically consistent representation of the occupancy probability. Building on these analyses, we improve existing evaluation protocols by aligning the newly identified representation with voxel-wise 3D occupancy ground truth, thereby enabling unsupervised methods to be evaluated in a manner consistent with that of supervised approaches. Additionally, to impose explicit constraints in occluded regions, we introduce an occlusion-aware polarization mechanism that incorporates multi-view visual cues to enhance discrimination between occupied and free spaces in these regions. Extensive experiments demonstrate that our approach not only significantly outperforms existing unsupervised approaches but also matches the performance of supervised ones. Our source code and evaluation protocol will be made available upon publication.

  </details>



- **User-Centric Object Navigation: A Benchmark with Integrated User Habits for Personalized Embodied Object Search**  
  Hongcheng Wang, Jinyu Zhu, Hao Dong  
  _2026-02-06_ · https://arxiv.org/abs/2602.06459v1  
  <details><summary>Abstract</summary>

  In the evolving field of robotics, the challenge of Object Navigation (ON) in household environments has attracted significant interest. Existing ON benchmarks typically place objects in locations guided by general scene priors, without accounting for the specific placement habits of individual users. This omission limits the adaptability of navigation agents in personalized household environments. To address this, we introduce User-centric Object Navigation (UcON), a new benchmark that incorporates user-specific object placement habits, referred to as user habits. This benchmark requires agents to leverage these user habits for more informed decision-making during navigation. UcON encompasses approximately 22,600 user habits across 489 object categories. UcON is, to our knowledge, the first benchmark that explicitly formalizes and evaluates habit-conditioned object navigation at scale and covers the widest range of target object categories. Additionally, we propose a habit retrieval module to extract and utilize habits related to target objects, enabling agents to infer their likely locations more effectively. Experimental results demonstrate that current SOTA methods exhibit substantial performance degradation under habit-driven object placement, while integrating user habits consistently improves success rates. Code is available at https://github.com/whcpumpkin/User-Centric-Object-Navigation.

  </details>



- **What Is Wrong with Synthetic Data for Scene Text Recognition? A Strong Synthetic Engine with Diverse Simulations and Self-Evolution**  
  Xingsong Ye, Yongkun Du, JiaXin Zhang, Chen Li, Jing LYU, Zhineng Chen  
  _2026-02-06_ · https://arxiv.org/abs/2602.06450v1  
  <details><summary>Abstract</summary>

  Large-scale and categorical-balanced text data is essential for training effective Scene Text Recognition (STR) models, which is hard to achieve when collecting real data. Synthetic data offers a cost-effective and perfectly labeled alternative. However, its performance often lags behind, revealing a significant domain gap between real and current synthetic data. In this work, we systematically analyze mainstream rendering-based synthetic datasets and identify their key limitations: insufficient diversity in corpus, font, and layout, which restricts their realism in complex scenarios. To address these issues, we introduce UnionST, a strong data engine synthesizes text covering a union of challenging samples and better aligns with the complexity observed in the wild. We then construct UnionST-S, a large-scale synthetic dataset with improved simulations in challenging scenarios. Furthermore, we develop a self-evolution learning (SEL) framework for effective real data annotation. Experiments show that models trained on UnionST-S achieve significant improvements over existing synthetic datasets. They even surpass real-data performance in certain scenarios. Moreover, when using SEL, the trained models achieve competitive performance by only seeing 9% of real data labels.

  </details>


