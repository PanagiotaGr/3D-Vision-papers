# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-03-19 07:12 UTC_

Total papers shown: **50**


---

- **GMT: Goal-Conditioned Multimodal Transformer for 6-DOF Object Trajectory Synthesis in 3D Scenes**  
  Huajian Zeng, Abhishek Saroha, Daniel Cremers, Xi Wang  
  _2026-03-18_ · https://arxiv.org/abs/2603.17993v1  
  <details><summary>Abstract</summary>

  Synthesizing controllable 6-DOF object manipulation trajectories in 3D environments is essential for enabling robots to interact with complex scenes, yet remains challenging due to the need for accurate spatial reasoning, physical feasibility, and multimodal scene understanding. Existing approaches often rely on 2D or partial 3D representations, limiting their ability to capture full scene geometry and constraining trajectory precision. We present GMT, a multimodal transformer framework that generates realistic and goal-directed object trajectories by jointly leveraging 3D bounding box geometry, point cloud context, semantic object categories, and target end poses. The model represents trajectories as continuous 6-DOF pose sequences and employs a tailored conditioning strategy that fuses geometric, semantic, contextual, and goaloriented information. Extensive experiments on synthetic and real-world benchmarks demonstrate that GMT outperforms state-of-the-art human motion and human-object interaction baselines, such as CHOIS and GIMO, achieving substantial gains in spatial accuracy and orientation control. Our method establishes a new benchmark for learningbased manipulation planning and shows strong generalization to diverse objects and cluttered 3D environments. Project page: https://huajian- zeng.github. io/projects/gmt/.

  </details>



- **Interpretable Traffic Responsibility from Dashcam Video via Legal Multi Agent Reasoning**  
  Jingchun Yang, Jinchang Zhang  
  _2026-03-18_ · https://arxiv.org/abs/2603.17930v1  
  <details><summary>Abstract</summary>

  The widespread adoption of dashcams has made video evidence in traffic accidents increasingly abundant, yet transforming "what happened in the video" into "who is responsible under which legal provisions" still relies heavily on human experts. Existing ego-view traffic accident studies mainly focus on perception and semantic understanding, while LLM-based legal methods are mostly built on textual case descriptions and rarely incorporate video evidence, leaving a clear gap between the two. We first propose C-TRAIL, a multimodal legal dataset that, under the Chinese traffic regulation system, explicitly aligns dashcam videos and textual descriptions with a closed set of responsibility modes and their corresponding Chinese traffic statutes. On this basis, we introduce a two-stage framework: (1) a traffic accident understanding module that generates textual video descriptions; and (2) a legal multi-agent framework that outputs responsibility modes, statute sets, and complete judgment reports. Experimental results on C-TRAIL and MM-AU show that our method outperforms general and legal LLMs, as well as existing agent-based approaches, while providing a transparent and interpretable legal reasoning process.

  </details>



- **A practical artificial intelligence framework for legal age estimation using clavicle computed tomography scans**  
  Javier Venema, Stefano De Luca, Pablo Mesejo, Óscar Ibáñez  
  _2026-03-18_ · https://arxiv.org/abs/2603.17926v1  
  <details><summary>Abstract</summary>

  Legal age estimation plays a critical role in forensic and medico-legal contexts, where decisions must be supported by accurate, robust, and reproducible methods with explicit uncertainty quantification. While prior artificial intelligence (AI)-based approaches have primarily focused on hand radiographs or dental imaging, clavicle computed tomography (CT) scans remain underexplored despite their documented effectiveness for legal age estimation. In this work, we present an interpretable, multi-stage pipeline for legal age estimation from clavicle CT scans. The proposed framework combines (i) a feature-based connected-component method for automatic clavicle detection that requires minimal manual annotation, (ii) an Integrated Gradients-guided slice selection strategy used to construct the input data for a multi-slice convolutional neural network that estimates legal age, and (iii) conformal prediction intervals to support uncertainty-aware decisions in accordance with established international protocols. The pipeline is evaluated on 1,158 full-body post-mortem CT scans from a public forensic dataset (the New Mexico Decedent Image Database). The final model achieves state-of-the-art performance with a mean absolute error (MAE) of 1.55 $\pm$ 0.16 years on a held-out test set, outperforming both human experts (MAE of approximately 1.90 years) and previous methods (MAEs above 1.75 years in our same dataset). Furthermore, conformal prediction enables configurable coverage levels aligned with forensic requirements. Attribution maps indicate that the model focuses on anatomically relevant regions of the medial clavicular epiphysis. The proposed method, which is currently being added as part of the Skeleton-ID software (https://skeleton-id.com/skeleton-id/), is intended as a decision-support component within multi-factorial forensic workflows.

  </details>



- **SegFly: A 2D-3D-2D Paradigm for Aerial RGB-Thermal Semantic Segmentation at Scale**  
  Markus Gross, Sai Bharadhwaj Matha, Rui Song, Viswanathan Muthuveerappan, Conrad Christoph, Julius Huber, Daniel Cremers  
  _2026-03-18_ · https://arxiv.org/abs/2603.17920v1  
  <details><summary>Abstract</summary>

  Semantic segmentation for uncrewed aerial vehicles (UAVs) is fundamental for aerial scene understanding, yet existing RGB and RGB-T datasets remain limited in scale, diversity, and annotation efficiency due to the high cost of manual labeling and the difficulties of accurate RGB-T alignment on off-the-shelf UAVs. To address these challenges, we propose a scalable geometry-driven 2D-3D-2D paradigm that leverages multi-view redundancy in high-overlap aerial imagery to automatically propagate labels from a small subset of manually annotated RGB images to both RGB and thermal modalities within a unified framework. By lifting less than 3% of RGB images into a semantic 3D point cloud and reprojecting it into all views, our approach enables dense pseudo ground-truth generation across large image collections, automatically producing 97% of RGB labels and 100% of thermal labels while achieving 91% and 88% annotation accuracy without any 2D manual refinement. We further extend this 2D-3D-2D paradigm to cross-modal image registration, using 3D geometry as an intermediate alignment space to obtain fully automatic, strong pixel-level RGB-T alignment with 87% registration accuracy and no hardware-level synchronization. Applying our framework to existing geo-referenced aerial imagery, we construct SegFly, a large-scale benchmark with over 20,000 high-resolution RGB images and more than 15,000 geometrically aligned RGB-T pairs spanning diverse urban, industrial, and rural environments across multiple altitudes and seasons. On SegFly, we establish the Firefly baseline for RGB and thermal semantic segmentation and show that both conventional architectures and vision foundation models benefit substantially from SegFly supervision, highlighting the potential of geometry-driven 2D-3D-2D pipelines for scalable multi-modal scene understanding. Data and Code available at https://github.com/markus-42/SegFly.

  </details>



- **Differential Attention-Augmented BiomedCLIP with Asymmetric Focal Optimization for Imbalanced Multi-Label Video Capsule Endoscopy Classification**  
  Podakanti Satyajith Chary, Nagarajan Ganapathy  
  _2026-03-18_ · https://arxiv.org/abs/2603.17879v1  
  <details><summary>Abstract</summary>

  This work presents a multi-label classification framework for video capsule endoscopy (VCE) that addresses the extreme class imbalance inherent in the Galar dataset through a combination of architectural and optimization-level strategies. Our approach modifies BiomedCLIP, a biomedical vision-language foundation model, by replacing its standard multi-head self-attention with a differential attention mechanism that computes the difference between two softmax attention maps to suppress attention noise. To counteract the skewed label distribution, where pathological findings constitute less than 0.1% of all annotated frames, a sqrt-frequency weighted sampler, asymmetric focal loss, mixup regularization, and per-class threshold optimization are employed. Temporal coherence is enforced through median-filter smoothing and gap merging prior to event-level JSON generation. On the held-out RARE-VISION test set comprising three NaviCam examinations (161,025 frames), the pipeline achieves an overall temporal mAP@0.5 of 0.2456 and mAP@0.95 of 0.2353, with total inference completed in approximately 8.6 minutes on a single GPU.

  </details>



- **Edit Spillover as a Probe: Do Image Editing Models Implicitly Understand World Relations?**  
  Guandong Li, Zhaobin Chu  
  _2026-03-18_ · https://arxiv.org/abs/2603.17876v1  
  <details><summary>Abstract</summary>

  Instruction-following image editing models are expected to modify only the specified region while keeping the rest of the image unchanged. However, in practice, we observe a pervasive phenomenon -- edit spillover: models alter semantically related but unspecified content outside the edit region. This raises a fundamental question -- does spillover reflect genuine implicit world understanding, or is it merely attention leakage? We propose EditSpilloverProbe, a systematic framework that repurposes edit spillover as a natural probe for world knowledge in image editing models. We introduce a spillover taxonomy (spatial, semantic, mixed, random), an automated detection-and-classification pipeline, and a benchmark dataset constructed from real-world Chinese text editing tasks, EditSpilloverBench. Systematic evaluation of 5 representative editing models reveals three core findings: (1) spillover rates vary dramatically across architectures, from 3.49% to 11.46%, with a 3.3x ratio; (2) absolute semantic spillover quantity reveals models' world understanding capability -- nano_banana produces the most semantic spillover (27.8 per image), while qwen_2511 has the most precise editing control but lower semantic spillover (16.3 per image), revealing a trade-off between editing control and world understanding; (3) spatial decay analysis shows spillover area density decays exponentially with distance, but the proportion of semantically relevant spillover remains constant (40%-58%), providing direct evidence that semantic spillover reflects genuine world understanding rather than spatial diffusion.

  </details>



- **DexViTac: Collecting Human Visuo-Tactile-Kinematic Demonstrations for Contact-Rich Dexterous Manipulation**  
  Xitong Chen, Yifeng Pan, Min Li, Xiaotian Ding  
  _2026-03-18_ · https://arxiv.org/abs/2603.17851v1  
  <details><summary>Abstract</summary>

  Large-scale, high-quality multimodal demonstrations are essential for robot learning of contact-rich dexterous manipulation. While human-centric data collection systems lower the barrier to scaling, they struggle to capture the tactile information during physical interactions. Motivated by this, we present DexViTac, a portable, human-centric data collection system tailored for contact-rich dexterous manipulation. The system enables the high-fidelity acquisition of first-person vision, high-density tactile sensing, end-effector poses, and hand kinematics within unstructured, in-the-wild environments. Building upon this hardware, we propose a kinematics-grounded tactile representation learning algorithm that effectively resolves semantic ambiguities within tactile signals. Leveraging the efficiency of DexViTac, we construct a multimodal dataset comprising over 2,400 visuo-tactile-kinematic demonstrations. Experiments demonstrate that DexViTac achieves a collection efficiency exceeding 248 demonstrations per hour and remains robust against complex visual occlusions. Real-world deployment confirms that policies trained with the proposed dataset and learning strategy achieve an average success rate exceeding 85% across four challenging tasks. This performance significantly outperforms baseline methods, thereby validating the substantial improvement the system provides for learning contact-rich dexterous manipulation. Project page: https://xitong-c.github.io/DexViTac/.

  </details>



- **ProbeFlow: Training-Free Adaptive Flow Matching for Vision-Language-Action Models**  
  Zhou Fang, Jiaqi Wang, Yi Zhou, Qiongfeng Shi  
  _2026-03-18_ · https://arxiv.org/abs/2603.17850v1  
  <details><summary>Abstract</summary>

  Recent Vision-Language-Action (VLA) models equipped with Flow Matching (FM) action heads achieve state-of-the-art performance in complex robot manipulation. However, the multi-step iterative ODE solving required by FM introduces inference latency that precludes responsive physical control. While current acceleration efforts optimize the Vision-Language Model (VLM) backbone, the action head bottleneck remains overlooked. To address this, we propose ProbeFlow, a training-free adaptive inference framework tai- lored for continuous robotic control. By evaluating geometric trajectory complexity via the cosine similarity between initial and lookahead velocity vectors, ProbeFlow dynamically sched- ules integration steps to prune redundant network evaluations. On the MetaWorld benchmark, it accelerates action decoding by 14.8x (reducing average steps from N = 50 to 2.6) and cuts end-to-end system latency by 2.8x without compromising the manipulation success rate. On the long-horizon LIBERO benchmark, the probe automatically allocates a denser schedule to navigate semantic bottlenecks, effectively resolving the flow solver delay. Real-world physical deployments confirm that ProbeFlow successfully mitigates action decoding latency while ensuring execution stability, offering a highly practical solution for low-latency continuous generative policies.

  </details>



- **M2P: Improving Visual Foundation Models with Mask-to-Point Weakly-Supervised Learning for Dense Point Tracking**  
  Qiangqiang Wu, Tianyu Yang, Bo Fang, Jia Wan, Matias Di Martino, Guillermo Sapiro, Antoni B. Chan  
  _2026-03-18_ · https://arxiv.org/abs/2603.17813v1  
  <details><summary>Abstract</summary>

  Tracking Any Point (TAP) has emerged as a fundamental tool for video understanding. Current approaches adapt Vision Foundation Models (VFMs) like DINOv2 via offline finetuning or test-time optimization. However, these VFMs rely on static image pre-training, which is inherently sub-optimal for capturing dense temporal correspondence in videos. To address this, we propose Mask-to-Point (M2P) learning, which leverages rich video object segmentation (VOS) mask annotations to improve VFMs for dense point tracking. Our M2P introduces three new mask-based constraints for weakly-supervised representation learning. First, we propose a local structure consistency loss, which leverages Procrustes analysis to model the cohesive motion of points lying within a local structure, achieving more reliable point-to-point matching learning. Second, we propose a mask label consistency (MLC) loss, which enforces that sampled foreground points strictly match foreground regions across frames. The proposed MLC loss can be regarded as a regularization, which stabilizes training and prevents convergence to trivial solutions. Finally, mask boundary constrain is applied to explicitly supervise boundary points. We show that our weaklysupervised M2P models significantly outperform baseline VFMs with efficient training by using only 3.6K VOS training videos. Notably, M2P achieves 12.8% and 14.6% performance gains over DINOv2-B/14 and DINOv3-B/16 on the TAP-Vid-DAVIS benchmark, respectively. Moreover, the proposed M2P models are used as pre-trained backbones for both test-time optimized and offline fine-tuned TAP tasks, demonstrating its potential to serve as general pre-trained models for point tracking. Code will be made publicly available upon acceptance.

  </details>



- **EVA: Aligning Video World Models with Executable Robot Actions via Inverse Dynamics Rewards**  
  Ruixiang Wang, Qingming Liu, Yueci Deng, Guiliang Liu, Zhen Liu, Kui Jia  
  _2026-03-18_ · https://arxiv.org/abs/2603.17808v1  
  <details><summary>Abstract</summary>

  Video generative models are increasingly used as world models for robotics, where a model generates a future visual rollout conditioned on the current observation and task instruction, and an inverse dynamics model (IDM) converts the generated frames into executable robot actions. However, current video world models lack explicit executability constraints. As a result, visually coherent rollouts may still violate rigid-body and kinematic consistency, producing unstable or infeasible control commands when decoded by an IDM. We refer to this mismatch between visual generation and physically executable control as the executability gap. While this gap can be mitigated at inference time using techniques such as rejection sampling, such approaches are inefficient due to the high cost of video generation. In this paper, we leverage the executability gap as a training signal and introduce Executable Video Alignment (EVA), a reinforcement-learning post-training framework for aligning video world models. EVA trains an inverse dynamics model on real robot trajectories and repurposes it as a reward model that evaluates generated videos through the action sequences they induce, encouraging smooth motions measured by velocity, acceleration, and jerk while penalizing actions that violate embodiment constraints. Importantly, the reward remains informative even when generated videos contain severe visual artifacts, since such artifacts typically translate into unstable or out-of-bound actions. Experiments on the RoboTwin benchmark and a real bimanual robot show that EVA reduces embodiment-specific artifacts in generated rollouts and improves downstream task execution success.

  </details>



- **Concept-to-Pixel: Prompt-Free Universal Medical Image Segmentation**  
  Haoyun Chen, Fenghe Tang, Wenxin Ma, Shaohua Kevin Zhou  
  _2026-03-18_ · https://arxiv.org/abs/2603.17746v1  
  <details><summary>Abstract</summary>

  Universal medical image segmentation seeks to use a single foundational model to handle diverse tasks across multiple imaging modalities. However, existing approaches often rely heavily on manual visual prompts or retrieved reference images, which limits their automation and robustness. In addition, naive joint training across modalities often fails to address large domain shifts. To address these limitations, we propose Concept-to-Pixel (C2P), a novel prompt-free universal segmentation framework. C2P explicitly separates anatomical knowledge into two components: Geometric and Semantic representations. It leverages Multimodal Large Language Models (MLLMs) to distill abstract, high-level medical concepts into learnable Semantic Tokens and introduces explicitly supervised Geometric Tokens to enforce universal physical and structural constraints. These disentangled tokens interact deeply with image features to generate input-specific dynamic kernels for precise mask prediction. Furthermore, we introduce a Geometry-Aware Inference Consensus mechanism, which utilizes the model's predicted geometric constraints to assess prediction reliability and suppress outliers. Extensive experiments and analysis on a unified benchmark comprising eight diverse datasets across seven modalities demonstrate the significant superiority of our jointly trained approach, compared to universe- or single-model approaches. Remarkably, our unified model demonstrates strong generalization, achieving impressive results not only on zero-shot tasks involving unseen cases but also in cross-modal transfers across similar tasks. Code is available at: https://github.com/Yundi218/Concept-to-Pixel

  </details>



- **VolumeDP: Modeling Volumetric Representation for Manipulation Policy Learning**  
  Tianxing Zhou, Feiyang Xue, Zhangchen Ye, Tianyuan Yuan, Hang Zhao, Tao Jiang  
  _2026-03-18_ · https://arxiv.org/abs/2603.17720v1  
  <details><summary>Abstract</summary>

  Imitation learning is a prominent paradigm for robotic manipulation. However, existing visual imitation methods map 2D image observations directly to 3D action outputs, imposing a 2D-3D mismatch that hinders spatial reasoning and degrades robustness. We present VolumeDP, a policy architecture that restores spatial alignment by explicitly reasoning in 3D. VolumeDP first lifts image features into a Volumetric Representation via cross-attention. It then selects task-relevant voxels with a learnable module and converts them into a compact set of spatial tokens, markedly reducing computation while preserving action-critical geometry. Finally, a multi-token decoder conditions on the entire token set to predict actions, thereby avoiding lossy aggregation that collapses multiple spatial tokens into a single descriptor. VolumeDP achieves a state-of-the-art average success rate of 88.8% on the LIBERO simulation benchmark, outperforming the strongest baseline by a substantial 14.8% improvement. It also delivers large performance gains over prior methods on the ManiSkill and LIBERO-Plus benchmarks. Real-world experiments further demonstrate higher success rates and robust generalization to novel spatial layouts, camera viewpoints, and environment backgrounds. Code will be released.

  </details>



- **Eye image segmentation using visual and concept prompts with Segment Anything Model 3 (SAM3)**  
  Diederick C. Niehorster, Marcus Nyström  
  _2026-03-18_ · https://arxiv.org/abs/2603.17715v1  
  <details><summary>Abstract</summary>

  Previous work has reported that vision foundation models show promising zero-shot performance in eye image segmentation. Here we examine whether the latest iteration of the Segment Anything Model, SAM3, offers better eye image segmentation performance than SAM2, and explore the performance of its new concept (text) prompting mode. Eye image segmentation performance was evaluated using diverse datasets encompassing both high-resolution high-quality videos from a lab environment and the TEyeD dataset consisting of challenging eye videos acquired in the wild. Results show that in most cases SAM3 with either visual or concept prompts did not perform better than SAM2, for both lab and in-the-wild datasets. Since SAM2 not only performed better but was also faster, we conclude that SAM2 remains the best option for eye image segmentation. We provide our adaptation of SAM3's codebase that allows processing videos of arbitrary duration.

  </details>



- **DancingBox: A Lightweight MoCap System for Character Animation from Physical Proxies**  
  Haocheng Yuan, Adrien Bousseau, Hao Pan, Lei Zhong, Changjian Li  
  _2026-03-18_ · https://arxiv.org/abs/2603.17704v1  
  <details><summary>Abstract</summary>

  Creating compelling 3D character animations typically requires either expert use of professional software or expensive motion capture systems operated by skilled actors. We present DancingBox, a lightweight, vision-based system that makes motion capture accessible to novices by reimagining the process as digital puppetry. Instead of tracking precise human motions, DancingBox captures the approximate movements of everyday objects manipulated by users with a single webcam. These coarse proxy motions are then refined into realistic character animations by conditioning a generative motion model on bounding-box representations, enriched with human motion priors learned from large-scale datasets. To overcome the lack of paired proxy-animation data, we synthesize training pairs by converting existing motion capture sequences into proxy representations. A user study demonstrates that DancingBox enables intuitive and creative character animation using diverse proxies, from plush toys to bananas, lowering the barrier to entry for novice animators.

  </details>



- **WeatherReasonSeg: A Benchmark for Weather-Aware Reasoning Segmentation in Visual Language Models**  
  Wanjun Du, Zifeng Yuan, Tingting Chen, Fucai Ke, Beibei Lin, Shunli Zhang  
  _2026-03-18_ · https://arxiv.org/abs/2603.17680v1  
  <details><summary>Abstract</summary>

  Existing vision-language models (VLMs) have demonstrated impressive performance in reasoning-based segmentation. However, current benchmarks are primarily constructed from high-quality images captured under idealized conditions. This raises a critical question: when visual cues are severely degraded by adverse weather conditions such as rain, snow, or fog, can VLMs sustain reliable reasoning segmentation capabilities? In response to this challenge, we introduce WeatherReasonSeg, a benchmark designed to evaluate VLM performance in reasoning-based segmentation under adverse weather conditions. It consists of two complementary components. First, we construct a controllable reasoning dataset by applying synthetic weather with varying severity levels to existing segmentation datasets, enabling fine-grained robustness analysis. Second, to capture real-world complexity, we curate a real-world adverse-weather reasoning segmentation dataset with semantically consistent queries generated via mask-guided LLM prompting. We further broaden the evaluation scope across five reasoning dimensions, including functionality, application scenarios, structural attributes, interactions, and requirement matching. Extensive experiments across diverse VLMs reveal two key findings: (1) VLM performance degrades monotonically with increasing weather severity, and (2) different weather types induce distinct vulnerability patterns. We hope WeatherReasonSeg will serve as a foundation for advancing robust, weather-aware reasoning.

  </details>



- **Illumination-Aware Contactless Fingerprint Spoof Detection via Paired Flash-Non-Flash Imaging**  
  Roja Sahoo, Anoop Namboodiri  
  _2026-03-18_ · https://arxiv.org/abs/2603.17679v1  
  <details><summary>Abstract</summary>

  Contactless fingerprint recognition enables hygienic and convenient biometric authentication but poses new challenges for spoof detection due to the absence of physical contact and traditional liveness cues. Most existing methods rely on single-image acquisition and appearance-based features, which often generalize poorly across devices, capture conditions, and spoof materials. In this work, we study paired flash-non-flash contactless fingerprint acquisition as a lightweight active sensing mechanism for spoof detection. Through a preliminary empirical analysis, we show that flash illumination accentuates material- and structure-dependent properties, including ridge visibility, subsurface scattering, micro-geometry, and surface oils, while non-flash images provide a baseline appearance context. We analyze lighting-induced differences using interpretable metrics such as inter-channel correlation, specular reflection characteristics, texture realism, and differential imaging. These complementary features help discriminate genuine fingerprints from printed, digital, and molded presentation attacks. We further examine the limitations of paired acquisition, including sensitivity to imaging settings, dataset scale, and emerging high-fidelity spoofs. Our findings demonstrate the potential of illumination-aware analysis to improve robustness and interpretability in contactless fingerprint presentation attack detection, motivating future work on paired acquisition and physics-informed feature design. Code is available in the repository.

  </details>



- **AgentVLN: Towards Agentic Vision-and-Language Navigation**  
  Zihao Xin, Wentong Li, Yixuan Jiang, Ziyuan Huang, Bin Wang, Piji Li, Jianke Zhu, Jie Qin, Shengjun Huang  
  _2026-03-18_ · https://arxiv.org/abs/2603.17670v1  
  <details><summary>Abstract</summary>

  Vision-and-Language Navigation (VLN) requires an embodied agent to ground complex natural-language instructions into long-horizon navigation in unseen environments. While Vision-Language Models (VLMs) offer strong 2D semantic understanding, current VLN systems remain constrained by limited spatial perception, 2D-3D representation mismatch, and monocular scale ambiguity. In this paper, we propose AgentVLN, a novel and efficient embodied navigation framework that can be deployed on edge computing platforms. We formulate VLN as a Partially Observable Semi-Markov Decision Process (POSMDP) and introduce a VLM-as-Brain paradigm that decouples high-level semantic reasoning from perception and planning via a plug-and-play skill library. To resolve multi-level representation inconsistency, we design a cross-space representation mapping that projects perception-layer 3D topological waypoints into the image plane, yielding pixel-aligned visual prompts for the VLM. Building on this bridge, we integrate a context-aware self-correction and active exploration strategy to recover from occlusions and suppress error accumulation over long trajectories. To further address the spatial ambiguity of instructions in unstructured environments, we propose a Query-Driven Perceptual Chain-of-Thought (QD-PCoT) scheme, enabling the agent with the metacognitive ability to actively seek geometric depth information. Finally, we construct AgentVLN-Instruct, a large-scale instruction-tuning dataset with dynamic stage routing conditioned on target visibility. Extensive experiments show that AgentVLN consistently outperforms prior state-of-the-art methods (SOTA) on long-horizon VLN benchmarks, offering a practical paradigm for lightweight deployment of next-generation embodied navigation models. Code: https://github.com/Allenxinn/AgentVLN.

  </details>



- **FINER: MLLMs Hallucinate under Fine-grained Negative Queries**  
  Rui Xiao, Sanghwan Kim, Yongqin Xian, Zeynep Akata, Stephan Alaniz  
  _2026-03-18_ · https://arxiv.org/abs/2603.17662v1  
  <details><summary>Abstract</summary>

  Multimodal large language models (MLLMs) struggle with hallucinations, particularly with fine-grained queries, a challenge underrepresented by existing benchmarks that focus on coarse image-related questions. We introduce FIne-grained NEgative queRies (FINER), alongside two benchmarks: FINER-CompreCap and FINER-DOCCI. Using FINER, we analyze hallucinations across four settings: multi-object, multi-attribute, multi-relation, and ``what'' questions. Our benchmarks reveal that MLLMs hallucinate when fine-grained mismatches co-occur with genuinely present elements in the image. To address this, we propose FINER-Tuning, leveraging Direct Preference Optimization (DPO) on FINER-inspired data. Finetuning four frontier MLLMs with FINER-Tuning yields up to 24.2\% gains (InternVL3.5-14B) on hallucinations from our benchmarks, while simultaneously improving performance on eight existing hallucination suites, and enhancing general multimodal capabilities across six benchmarks. Code, benchmark, and models are available at \href{https://explainableml.github.io/finer-project/}{https://explainableml.github.io/finer-project/}.

  </details>



- **Anchoring and Rescaling Attention for Semantically Coherent Inbetweening**  
  Tae Eun Choi, Sumin Shim, Junhyeok Kim, Seong Jae Hwang  
  _2026-03-18_ · https://arxiv.org/abs/2603.17651v1  
  <details><summary>Abstract</summary>

  Generative inbetweening (GI) seeks to synthesize realistic intermediate frames between the first and last keyframes beyond mere interpolation. As sequences become sparser and motions larger, previous GI models struggle with inconsistent frames with unstable pacing and semantic misalignment. Since GI involves fixed endpoints and numerous plausible paths, this task requires additional guidance gained from the keyframes and text to specify the intended path. Thus, we give semantic and temporal guidance from the keyframes and text onto each intermediate frame through Keyframe-anchored Attention Bias. We also better enforce frame consistency with Rescaled Temporal RoPE, which allows self-attention to attend to keyframes more faithfully. TGI-Bench, the first benchmark specifically designed for text-conditioned GI evaluation, enables challenge-targeted evaluation to analyze GI models. Without additional training, our method achieves state-of-the-art frame consistency, semantic fidelity, and pace stability for both short and long sequences across diverse challenges.

  </details>



- **Part-Aware Open-Vocabulary 3D Affordance Grounding via Prototypical Semantic and Geometric Alignment**  
  Dongqiang Gou, Xuming He  
  _2026-03-18_ · https://arxiv.org/abs/2603.17647v1  
  <details><summary>Abstract</summary>

  Grounding natural language questions to functionally relevant regions in 3D objects -- termed language-driven 3D affordance grounding -- is essential for embodied intelligence and human-AI interaction. Existing methods, while progressing from label-based to language-driven approaches, still face challenges in open-vocabulary generalization, fine-grained geometric alignment, and part-level semantic consistency. To address these issues, we propose a novel two-stage cross-modal framework that enhances both semantic and geometric representations for open-vocabulary 3D affordance grounding. In the first stage, large language models generate part-aware instructions to recover missing semantics, enabling the model to link semantically similar affordances. In the second stage, we introduce two key components: Affordance Prototype Aggregation (APA), which captures cross-object geometric consistency for each affordance, and Intra-Object Relational Modeling (IORM), which refines geometric differentiation within objects to support precise semantic alignment. We validate the effectiveness of our method through extensive experiments on a newly introduced benchmark, as well as two existing benchmarks, demonstrating superior performance in comparison with existing methods.

  </details>



- **Edit-As-Act: Goal-Regressive Planning for Open-Vocabulary 3D Indoor Scene Editing**  
  Seongrae Noh, SeungWon Seo, Gyeong-Moon Park, HyeongYeop Kang  
  _2026-03-18_ · https://arxiv.org/abs/2603.17583v1  
  <details><summary>Abstract</summary>

  Editing a 3D indoor scene from natural language is conceptually straightforward but technically challenging. Existing open-vocabulary systems often regenerate large portions of a scene or rely on image-space edits that disrupt spatial structure, resulting in unintended global changes or physically inconsistent layouts. These limitations stem from treating editing primarily as a generative task. We take a different view. A user instruction defines a desired world state, and editing should be the minimal sequence of actions that makes this state true while preserving everything else. This perspective motivates Edit-As-Act, a framework that performs open-vocabulary scene editing as goal-regressive planning in 3D space. Given a source scene and free-form instruction, Edit-As-Act predicts symbolic goal predicates and plans in EditLang, a PDDL-inspired action language that we design with explicit preconditions and effects encoding support, contact, collision, and other geometric relations. A language-driven planner proposes actions, and a validator enforces goal-directedness, monotonicity, and physical feasibility, producing interpretable and physically coherent transformations. By separating reasoning from low-level generation, Edit-As-Act achieves instruction fidelity, semantic consistency, and physical plausibility - three criteria that existing paradigms cannot satisfy together. On E2A-Bench, our benchmark of 63 editing tasks across 9 indoor environments, Edit-As-Act significantly outperforms prior approaches across all edit types and scene categories.

  </details>



- **PanoVGGT: Feed-Forward 3D Reconstruction from Panoramic Imagery**  
  Yijing Guo, Mengjun Chao, Luo Wang, Tianyang Zhao, Haizhao Dai, Yingliang Zhang, Jingyi Yu, Yujiao Shi  
  _2026-03-18_ · https://arxiv.org/abs/2603.17571v1  
  <details><summary>Abstract</summary>

  Panoramic imagery offers a full 360° field of view and is increasingly common in consumer devices. However, it introduces non-pinhole distortions that challenge joint pose estimation and 3D reconstruction. Existing feed-forward models, built for perspective cameras, generalize poorly to this setting. We propose PanoVGGT, a permutation-equivariant Transformer framework that jointly predicts camera poses, depth maps, and 3D point clouds from one or multiple panoramas in a single forward pass. The model incorporates spherical-aware positional embeddings and a panorama-specific three-axis SO(3) rotation augmentation, enabling effective geometric reasoning in the spherical domain. To resolve inherent global-frame ambiguity, we further introduce a stochastic anchoring strategy during training. In addition, we contribute PanoCity, a large-scale outdoor panoramic dataset with dense depth and 6-DoF pose annotations. Extensive experiments on PanoCity and standard benchmarks demonstrate that PanoVGGT achieves competitive accuracy, strong robustness, and improved cross-domain generalization. Code and dataset will be released.

  </details>



- **Face anonymization preserving facial expressions and photometric realism**  
  Luigi Celona, Simone Bianco, Raimondo Schettini  
  _2026-03-18_ · https://arxiv.org/abs/2603.17567v1  
  <details><summary>Abstract</summary>

  The widespread sharing of face images on social media platforms and in large-scale datasets raises pressing privacy concerns, as biometric identifiers can be exploited without consent. Face anonymization seeks to generate realistic facial images that irreversibly conceal the subject's identity while preserving their usefulness for downstream tasks. However, most existing generative approaches focus on identity removal and image realism, often neglecting facial expressions as well as photometric consistency -- specifically attributes such as illumination and skin tone -- that are critical for applications like relighting, color constancy, and medical or affective analysis. In this work, we propose a feature-preserving anonymization framework that extends DeepPrivacy by incorporating dense facial landmarks to better retain expressions, and by introducing lightweight post-processing modules that ensure consistency in lighting direction and skin color. We further establish evaluation metrics specifically designed to quantify expression fidelity, lighting consistency, and color preservation, complementing standard measures of image realism, pose accuracy, and re-identification resistance. Experiments on the CelebA-HQ dataset demonstrate that our method produces anonymized faces with improved realism and significantly higher fidelity in expression, illumination, and skin tone compared to state-of-the-art baselines. These results underscore the importance of feature-aware anonymization as a step toward more useful, fair, and trustworthy privacy-preserving facial data.

  </details>



- **FrescoDiffusion: 4K Image-to-Video with Prior-Regularized Tiled Diffusion**  
  Hugo Caselles-Dupré, Mathis Koroglu, Guillaume Jeanneret, Arnaud Dapogny, Matthieu Cord  
  _2026-03-18_ · https://arxiv.org/abs/2603.17555v1  
  <details><summary>Abstract</summary>

  Diffusion-based image-to-video (I2V) models are increasingly effective, yet they struggle to scale to ultra-high-resolution inputs (e.g., 4K). Generating videos at the model's native resolution often loses fine-grained structure, whereas high-resolution tiled denoising preserves local detail but breaks global layout consistency. This failure mode is particularly severe in the fresco animation setting: monumental artworks containing many distinct characters, objects, and semantically different sub-scenes that must remain spatially coherent over time. We introduce FrescoDiffusion, a training-free method for coherent large-format I2V generation from a single complex image. The key idea is to augment tiled denoising with a precomputed latent prior: we first generate a low-resolution video at the underlying model resolution and upsample its latent trajectory to obtain a global reference that captures long-range temporal and spatial structure. For 4K generation, we compute per-tile noise predictions and fuse them with this reference at every diffusion timestep by minimizing a single weighted least-squares objective in model-output space. The objective combines a standard tile-merging criterion with our regularization term, yielding a closed-form fusion update that strengthens global coherence while retaining fine detail. We additionally provide a spatial regularization variable that enables region-level control over where motion is allowed. Experiments on the VBench-I2V dataset and our proposed fresco I2V dataset show improved global consistency and fidelity over tiled baselines, while being computationally efficient. Our regularization enables explicit controllability of the trade-off between creativity and consistency.

  </details>



- **MM-OVSeg:Multimodal Optical-SAR Fusion for Open-Vocabulary Segmentation in Remote Sensing**  
  Yimin Wei, Aoran Xiao, Hongruixuan Chen, Junshi Xia, Naoto Yokoya  
  _2026-03-18_ · https://arxiv.org/abs/2603.17528v1  
  <details><summary>Abstract</summary>

  Open-vocabulary segmentation enables pixel-level recognition from an open set of textual categories, allowing generalization beyond fixed classes. Despite great potential in remote sensing, progress in this area remains largely limited to clear-sky optical data and struggles under cloudy or haze-contaminated conditions. We present MM-OVSeg, a multimodal Optical-SAR fusion framework for resilient open-vocabulary segmentation under adverse weather conditions. MM-OVSeg leverages the complementary strengths of the two modalities--optical imagery provides rich spectral semantics, while synthetic aperture radar (SAR) offers cloud-penetrating structural cues. To address the cross-modal domain gap and the limited dense prediction capability of current vision-language models, we propose two key designs: a cross-modal unification process for multi-sensor representation alignment, and a dual-encoder fusion module that integrates hierarchical features from multiple vision foundation models for text-aligned multimodal segmentation. Extensive experiments demonstrate that MM-OVSeg achieves superior robustness and generalization across diverse cloud conditions. The source dataset and code are available here.

  </details>



- **Omni-I2C: A Holistic Benchmark for High-Fidelity Image-to-Code Generation**  
  Jiawei Zhou, Chi Zhang, Xiang Feng, Qiming Zhang, Haibo Qiu, Lihuo He, Dengpan Ye, Xinbo Gao, Jing Zhang  
  _2026-03-18_ · https://arxiv.org/abs/2603.17508v1  
  <details><summary>Abstract</summary>

  We present Omni-I2C, a comprehensive benchmark designed to evaluate the capability of Large Multimodal Models (LMMs) in converting complex, structured digital graphics into executable code. We argue that this task represents a non-trivial challenge for the current generation of LMMs: it demands an unprecedented synergy between high-fidelity visual perception -- to parse intricate spatial hierarchies and symbolic details -- and precise generative expression -- to synthesize syntactically sound and logically consistent code. Unlike traditional descriptive tasks, Omni-I2C requires a holistic understanding where any minor perceptual hallucination or coding error leads to a complete failure in visual reconstruction. Omni-I2C features 1080 meticulously curated samples, defined by its breadth across subjects, image modalities, and programming languages. By incorporating authentic user-sourced cases, the benchmark spans a vast spectrum of digital content -- from scientific visualizations to complex symbolic notations -- each paired with executable reference code. To complement this diversity, our evaluation framework provides necessary depth; by decoupling performance into perceptual fidelity and symbolic precision, it transcends surface-level accuracy to expose the granular structural failures and reasoning bottlenecks of current LMMs. Our evaluation reveals a substantial performance gap among leading LMMs; even state-of-the-art models struggle to preserve structural integrity in complex scenarios, underscoring that multimodal code generation remains a formidable challenge. Data and code are available at https://github.com/MiliLab/Omni-I2C.

  </details>



- **UAV-CB: A Complex-Background RGB-T Dataset and Local Frequency Bridge Network for UAV Detection**  
  Shenghui Huang, Menghao Hu, Longkun Zou, Hongyu Chi, Zekai Li, Feng Gao, Fan Yang, Qingyao Wu, Ke Chen  
  _2026-03-18_ · https://arxiv.org/abs/2603.17492v1  
  <details><summary>Abstract</summary>

  Detecting Unmanned Aerial Vehicles (UAVs) in low-altitude environments is essential for perception and defense systems but remains highly challenging due to complex backgrounds, camouflage, and multimodal interference. In real-world scenarios, UAVs are frequently visually blended with surrounding structures such as buildings, vegetation, and power lines, resulting in low contrast, weak boundaries, and strong confusion with cluttered background textures. Existing UAV detection datasets, though diverse, are not specifically designed to capture these camouflage and complex-background challenges, which limits progress toward robust real-world perception. To fill this gap, we construct UAV-CB, a new RGB-T UAV detection dataset deliberately curated to emphasize complex low-altitude backgrounds and camouflage characteristics. Furthermore, we propose the Local Frequency Bridge Network (LFBNet), which models features in localized frequency space to bridge both the frequency-spatial fusion gap and the cross-modality discrepancy gap in RGB-T fusion. Extensive experiments on UAV-CB and public benchmarks demonstrate that LFBNet achieves state-of-the-art detection performance and strong robustness under camouflaged and cluttered conditions, offering a frequency-aware perspective on multimodal UAV perception in real-world applications.

  </details>



- **UniSAFE: A Comprehensive Benchmark for Safety Evaluation of Unified Multimodal Models**  
  Segyu Lee, Boryeong Cho, Hojung Jung, Seokhyun An, Juhyeong Kim, Jaehyun Kwak, Yongjin Yang, Sangwon Jang, Youngrok Park, Wonjun Chang, et al.  
  _2026-03-18_ · https://arxiv.org/abs/2603.17476v1  
  <details><summary>Abstract</summary>

  Unified Multimodal Models (UMMs) offer powerful cross-modality capabilities but introduce new safety risks not observed in single-task models. Despite their emergence, existing safety benchmarks remain fragmented across tasks and modalities, limiting the comprehensive evaluation of complex system-level vulnerabilities. To address this gap, we introduce UniSAFE, the first comprehensive benchmark for system-level safety evaluation of UMMs across 7 I/O modality combinations, spanning conventional tasks and novel multimodal-context image generation settings. UniSAFE is built with a shared-target design that projects common risk scenarios across task-specific I/O configurations, enabling controlled cross-task comparisons of safety failures. Comprising 6,802 curated instances, we use UniSAFE to evaluate 15 state-of-the-art UMMs, both proprietary and open-source. Our results reveal critical vulnerabilities across current UMMs, including elevated safety violations in multi-image composition and multi-turn settings, with image-output tasks consistently more vulnerable than text-output tasks. These findings highlight the need for stronger system-level safety alignment for UMMs. Our code and data are publicly available at https://github.com/segyulee/UniSAFE

  </details>



- **VirPro: Visual-referred Probabilistic Prompt Learning for Weakly-Supervised Monocular 3D Detection**  
  Chupeng Liu, Jiyong Rao, Shangquan Sun, Runkai Zhao, Weidong Cai  
  _2026-03-18_ · https://arxiv.org/abs/2603.17470v1  
  <details><summary>Abstract</summary>

  Monocular 3D object detection typically relies on pseudo-labeling techniques to reduce dependency on real-world annotations. Recent advances demonstrate that deterministic linguistic cues can serve as effective auxiliary weak supervision signals, providing complementary semantic context. However, hand-crafted textual descriptions struggle to capture the inherent visual diversity of individuals across scenes, limiting the model's ability to learn scene-aware representations. To address this challenge, we propose Visual-referred Probabilistic Prompt Learning (VirPro), an adaptive multi-modal pretraining paradigm that can be seamlessly integrated into diverse weakly supervised monocular 3D detection frameworks. Specifically, we generate a diverse set of learnable, instance-conditioned prompts across scenes and store them in an Adaptive Prompt Bank (APB). Subsequently, we introduce Multi-Gaussian Prompt Modeling (MGPM), which incorporates scene-based visual features into the corresponding textual embeddings, allowing the text prompts to express visual uncertainties. Then, from the fused vision-language embeddings, we decode a prompt-targeted Gaussian, from which we derive a unified object-level prompt embedding for each instance. RoI-level contrastive matching is employed to enforce modality alignment, bringing embeddings of co-occurring objects within the same scene closer in the latent space, thus enhancing semantic coherence. Extensive experiments on the KITTI benchmark demonstrate that integrating our pretraining paradigm consistently yields substantial performance gains, achieving up to a 4.8% average precision improvement than the baseline.

  </details>



- **AdaZoom-GUI: Adaptive Zoom-based GUI Grounding with Instruction Refinement**  
  Siqi Pei, Liang Tang, Tiaonan Duan, Long Chen, Shuxian Li, Kaer Huang, Yanzhe Jing, Yiqiang Yan, Bo Zhang, Chenghao Jiang, et al.  
  _2026-03-18_ · https://arxiv.org/abs/2603.17441v1  
  <details><summary>Abstract</summary>

  GUI grounding is a critical capability for vision-language models (VLMs) that enables automated interaction with graphical user interfaces by locating target elements from natural language instructions. However, grounding on GUI screenshots remains challenging due to high-resolution images, small UI elements, and ambiguous user instructions. In this work, we propose AdaZoom-GUI, an adaptive zoom-based GUI grounding framework that improves both localization accuracy and instruction understanding. Our approach introduces an instruction refinement module that rewrites natural language commands into explicit and detailed descriptions, allowing the grounding model to focus on precise element localization. In addition, we design a conditional zoom-in strategy that selectively performs a second-stage inference on predicted small elements, improving localization accuracy while avoiding unnecessary computation and context loss on simpler cases. To support this framework, we construct a high-quality GUI grounding dataset and train the grounding model using Group Relative Policy Optimization (GRPO), enabling the model to predict both click coordinates and element bounding boxes. Experiments on public benchmarks demonstrate that our method achieves state-of-the-art performance among models with comparable or even larger parameter sizes, highlighting its effectiveness for high-resolution GUI understanding and practical GUI agent deployment.

  </details>



- **FloorPlan-VLN: A New Paradigm for Floor Plan Guided Vision-Language Navigation**  
  Kehan Chen, Yan Huang, Dong An, Jiawei He, Yifei Su, Jing Liu, Nianfeng Liu, Liang Wang  
  _2026-03-18_ · https://arxiv.org/abs/2603.17437v1  
  <details><summary>Abstract</summary>

  Existing Vision-Language Navigation (VLN) task requires agents to follow verbose instructions, ignoring some potentially useful global spatial priors, limiting their capability to reason about spatial structures. Although human-readable spatial schematics (e.g., floor plans) are ubiquitous in real-world buildings, current agents lack the cognitive ability to comprehend and utilize them. To bridge this gap, we introduce \textbf{FloorPlan-VLN}, a new paradigm that leverages structured semantic floor plans as global spatial priors to enable navigation with only concise instructions. We first construct the FloorPlan-VLN dataset, which comprises over 10k episodes across 72 scenes. It pairs more than 100 semantically annotated floor plans with Matterport3D-based navigation trajectories and concise instructions that omit step-by-step guidance. Then, we propose a simple yet effective method \textbf{FP-Nav} that uses a dual-view, spatio-temporally aligned video sequence, and auxiliary reasoning tasks to align observations, floor plans, and instructions. When evaluated under this new benchmark, our method significantly outperforms adapted state-of-the-art VLN baselines, achieving more than a 60\% relative improvement in navigation success rate. Furthermore, comprehensive noise modeling and real-world deployments demonstrate the feasibility and robustness of FP-Nav to actuation drift and floor plan distortions. These results validate the effectiveness of floor plan guided navigation and highlight FloorPlan-VLN as a promising step toward more spatially intelligent navigation.

  </details>



- **Towards Motion-aware Referring Image Segmentation**  
  Chaeyun Kim, Seunghoon Yi, Yejin Kim, Yohan Jo, Joonseok Lee  
  _2026-03-18_ · https://arxiv.org/abs/2603.17413v1  
  <details><summary>Abstract</summary>

  Referring Image Segmentation (RIS) requires identifying objects from images based on textual descriptions. We observe that existing methods significantly underperform on motion-related queries compared to appearance-based ones. To address this, we first introduce an efficient data augmentation scheme that extracts motion-centric phrases from original captions, exposing models to more motion expressions without additional annotations. Second, since the same object can be described differently depending on the context, we propose Multimodal Radial Contrastive Learning (MRaCL), performed on fused image-text embeddings rather than unimodal representations. For comprehensive evaluation, we introduce a new test split focusing on motion-centric queries, and introduce a new benchmark called M-Bench, where objects are distinguished primarily by actions. Extensive experiments show our method substantially improves performance on motion-centric queries across multiple RIS models, maintaining competitive results on appearance-based descriptions. Codes are available at https://github.com/snuviplab/MRaCL

  </details>



- **Mutually Causal Semantic Distillation Network for Zero-Shot Learning**  
  Shiming Chen, Shuhuang Chen, Guo-Sen Xie, Xinge You  
  _2026-03-18_ · https://arxiv.org/abs/2603.17412v1  
  <details><summary>Abstract</summary>

  Zero-shot learning (ZSL) aims to recognize the unseen classes in the open-world guided by the side-information (e.g., attributes). Its key task is how to infer the latent semantic knowledge between visual and attribute features on seen classes, and thus conducting a desirable semantic knowledge transfer from seen classes to unseen ones. Prior works simply utilize unidirectional attention within a weakly-supervised manner to learn the spurious and limited latent semantic representations, which fail to effectively discover the intrinsic semantic knowledge (e.g., attribute semantic) between visual and attribute features. To solve the above challenges, we propose a mutually causal semantic distillation network (termed MSDN++) to distill the intrinsic and sufficient semantic representations for ZSL. MSDN++ consists of an attribute$\rightarrow$visual causal attention sub-net that learns attribute-based visual features, and a visual$\rightarrow$attribute causal attention sub-net that learns visual-based attribute features. The causal attentions encourages the two sub-nets to learn causal vision-attribute associations for representing reliable features with causal visual/attribute learning. With the guidance of semantic distillation loss, the two mutual attention sub-nets learn collaboratively and teach each other throughout the training process. Extensive experiments on three widely-used benchmark datasets (e.g., CUB, SUN, AWA2, and FLO) show that our MSDN++ yields significant improvements over the strong baselines, leading to new state-of-the-art performances.

  </details>



- **Harnessing the Power of Foundation Models for Accurate Material Classification**  
  Qingran Lin, Fengwei Yang, Chaolun Zhu  
  _2026-03-18_ · https://arxiv.org/abs/2603.17390v1  
  <details><summary>Abstract</summary>

  Material classification has emerged as a critical task in computer vision and graphics, supporting the assignment of accurate material properties to a wide range of digital and real-world applications. While traditionally framed as an image classification task, this domain faces significant challenges due to the scarcity of annotated data, limiting the accuracy and generalizability of trained models. Recent advances in vision-language foundation models (VLMs) offer promising avenues to address these issues, yet existing solutions leveraging these models still exhibit unsatisfying results in material recognition tasks. In this work, we propose a novel framework that effectively harnesses foundation models to overcome data limitations and enhance classification accuracy. Our method integrates two key innovations: (a) a robust image generation and auto-labeling pipeline that creates a diverse and high-quality training dataset with material-centric images, and automatically assigns labels by fusing object semantics and material attributes in text prompts; (b) a prior incorporation strategy to distill information from VLMs, combined with a joint fine-tuning method that optimizes a pre-trained vision foundation model alongside VLM-derived priors, preserving broad generalizability while adapting to material-specific features.Extensive experiments demonstrate significant improvements on multiple datasets. We show that our synthetic dataset effectively captures the characteristics of real world materials, and the integration of priors from vision-language models significantly enhances the final performance. The source code and dataset will be released.

  </details>



- **Shot-Aware Frame Sampling for Video Understanding**  
  Mengyu Zhao, Di Fu, Yongyu Xie, Jiaxing Zhang, Zhigang Yuan, Shirin Jalali, Yong Cao  
  _2026-03-18_ · https://arxiv.org/abs/2603.17374v1  
  <details><summary>Abstract</summary>

  Video frame sampling is essential for efficient long-video understanding with Vision-Language Models (VLMs), since dense inputs are costly and often exceed context limits. Yet when only a small number of frames can be retained, existing samplers often fail to balance broad video coverage with brief but critical events, which can lead to unreliable downstream predictions. To address this issue, we present InfoShot, a task-agnostic, shot-aware frame sampler for long-video understanding. InfoShot first partitions a video into semantically consistent shots, and then selects two complementary keyframes from each shot: one to represent the main content and one to capture unusual within-shot changes. This design is guided by an information-theoretic objective that encourages the sampled set to retain high information about both shot structure and sparse within-shot deviations. In this way, it improves the chance of preserving both overall video context and short decision-critical moments without requiring any retraining. To better evaluate such short-lived events, we further introduce SynFlash, a synthetic benchmark with controllable sub-second anomaly patterns and frame-level ground truth, and we also evaluate InfoShot on existing anomaly datasets and general video understanding tasks. Experiments show that InfoShot improves anomaly hit rate and downstream Video-QA accuracy under frame number constraints, while matching or outperforming strong baselines on standard video understanding benchmarks.

  </details>



- **Material Magic Wand: Material-Aware Grouping of 3D Parts in Untextured Meshes**  
  Umangi Jain, Vladimir Kim, Matheus Gadelha, Igor Gilitschenski, Zhiqin Chen  
  _2026-03-18_ · https://arxiv.org/abs/2603.17370v1  
  <details><summary>Abstract</summary>

  We introduce the problem of material-aware part grouping in untextured meshes. Many real-world shapes, such as scales of pinecones or windows of buildings, contain repeated structures that share the same material but exhibit geometric variations. When assigning materials to such meshes, these repeated parts often require piece-by-piece manual identification and selection, which is tedious and time-consuming. To address this, we propose Material Magic Wand, a tool that allows artists to select part groups based on their estimated material properties -- when one part is selected, our algorithm automatically retrieves all other parts likely to share the same material. The key component of our approach is a part encoder that generates a material-aware embedding for each 3D part, accounting for both local geometry and global context. We train our model with a supervised contrastive loss that brings embeddings of material-consistent parts closer while separating those of different materials; therefore, part grouping can be achieved by retrieving embeddings that are close to the embedding of the selected part. To benchmark this task, we introduce a curated dataset of 100 shapes with 241 part-level queries. We verify the effectiveness of our method through extensive experiments and demonstrate its practical value in an interactive material assignment application.

  </details>



- **OnlineHMR: Video-based Online World-Grounded Human Mesh Recovery**  
  Yiwen Zhao, Ce Zheng, Yufu Wang, Hsueh-Han Daniel Yang, Liting Wen, Laszlo A. Jeni  
  _2026-03-18_ · https://arxiv.org/abs/2603.17355v1  
  <details><summary>Abstract</summary>

  Human mesh recovery (HMR) models 3D human body from monocular videos, with recent works extending it to world-coordinate human trajectory and motion reconstruction. However, most existing methods remain offline, relying on future frames or global optimization, which limits their applicability in interactive feedback and perception-action loop scenarios such as AR/VR and telepresence. To address this, we propose OnlineHMR, a fully online framework that jointly satisfies four essential criteria of online processing, including system-level causality, faithfulness, temporal consistency, and efficiency. Built upon a two-branch architecture, OnlineHMR enables streaming inference via a causal key-value cache design and a curated sliding-window learning strategy. Meanwhile, a human-centric incremental SLAM provides online world-grounded alignment under physically plausible trajectory correction. Experimental results show that our method achieves performance comparable to existing chunk-based approaches on the standard EMDB benchmark and highly dynamic custom videos, while uniquely supporting online processing. Page and code are available at https://tsukasane.github.io/Video-OnlineHMR/.

  </details>



- **OmniVLN: Omnidirectional 3D Perception and Token-Efficient LLM Reasoning for Visual-Language Navigation across Air and Ground Platforms**  
  Zhongyuang Liu, Min He, Shaonan Yu, Xinhang Xu, Muqing Cao, Jianping Li, Jianfei Yang, Lihua Xie  
  _2026-03-18_ · https://arxiv.org/abs/2603.17351v1  
  <details><summary>Abstract</summary>

  Language-guided embodied navigation requires an agent to interpret object-referential instructions, search across multiple rooms, localize the referenced target, and execute reliable motion toward it. Existing systems remain limited in real indoor environments because narrow field-of-view sensing exposes only a partial local scene at each step, often forcing repeated rotations, delaying target discovery, and producing fragmented spatial understanding; meanwhile, directly prompting LLMs with dense 3D maps or exhaustive object lists quickly exceeds the context budget. We present OmniVLN, a zero-shot visual-language navigation framework that couples omnidirectional 3D perception with token-efficient hierarchical reasoning for both aerial and ground robots. OmniVLN fuses a rotating LiDAR and panoramic vision into a hardware-agnostic mapping stack, incrementally constructs a five-layer Dynamic Scene Graph (DSG) from mesh geometry to room- and building-level structure, and stabilizes high-level topology through persistent-homology-based room partitioning and hybrid geometric/VLM relation verification. For navigation, the global DSG is transformed into an agent-centric 3D octant representation with multi-resolution spatial attention prompting, enabling the LLM to progressively filter candidate rooms, infer egocentric orientation, localize target objects, and emit executable navigation primitives while preserving fine local detail and compact long-range memory. Experiments show that the proposed hierarchical interface improves spatial referring accuracy from 77.27\% to 93.18\%, reduces cumulative prompt tokens by up to 61.7\% in cluttered multi-room settings, and improves navigation success by up to 11.68\% over a flat-list baseline. We will release the code and an omnidirectional multimodal dataset to support reproducible research.

  </details>



- **FineViT: Progressively Unlocking Fine-Grained Perception with Dense Recaptions**  
  Peisen Zhao, Xiaopeng Zhang, Mingxing Xu, Ruoyu Sun, Zewei Du, Dunzheng Wang, Guanghao Zheng, Haohang Xu, Zhibo Zhang, Yuhang Zhang, et al.  
  _2026-03-18_ · https://arxiv.org/abs/2603.17326v1  
  <details><summary>Abstract</summary>

  While Multimodal Large Language Models (MLLMs) have experienced rapid advancements, their visual encoders frequently remain a performance bottleneck. Conventional CLIP-based encoders struggle with dense spatial tasks due to the loss of visual details caused by low-resolution pretraining and the reliance on noisy, coarse web-crawled image-text pairs. To overcome these limitations, we introduce FineViT, a novel vision encoder specifically designed to unlock fine-grained perception. By replacing coarse web data with dense recaptions, we systematically mitigate information loss through a progressive training paradigm.: first, the encoder is trained from scratch at a high native resolution on billions of global recaptioned image-text pairs, establishing a robust, detail rich semantic foundation. Subsequently, we further enhance its local perception through LLM alignment, utilizing our curated FineCap-450M dataset that comprises over $450$ million high quality local captions. Extensive experiments validate the effectiveness of the progressive strategy. FineViT achieves state-of-the-art zero-shot recognition and retrieval performance, especially in long-context retrieval, and consistently outperforms multimodal visual encoders such as SigLIP2 and Qwen-ViT when integrated into MLLMs. We hope FineViT could serve as a powerful new baseline for fine-grained visual perception.

  </details>



- **3D MRI-Based Alzheimer's Disease Classification Using Multi-Modal 3D CNN with Leakage-Aware Subject-Level Evaluation**  
  Md Sifat, Sania Akter, Akif Islam, Md. Ekramul Hamid, Abu Saleh Musa Miah, Najmul Hassan, Md Abdur Rahim, Jungpil Shin  
  _2026-03-18_ · https://arxiv.org/abs/2603.17304v1  
  <details><summary>Abstract</summary>

  Deep learning has become an important tool for Alzheimer's disease (AD) classification from structural MRI. Many existing studies analyze individual 2D slices extracted from MRI volumes, while clinical neuroimaging practice typically relies on the full three dimensional structure of the brain. From this perspective, volumetric analysis may better capture spatial relationships among brain regions that are relevant to disease progression. Motivated by this idea, this work proposes a multimodal 3D convolutional neural network for AD classification using raw OASIS 1 MRI volumes. The model combines structural T1 information with gray matter, white matter, and cerebrospinal fluid probability maps obtained through FSL FAST segmentation in order to capture complementary neuroanatomical information. The proposed approach is evaluated on the clinically labelled OASIS 1 cohort using 5 fold subject level cross validation, achieving a mean accuracy of 72.34% plus or minus 4.66% and a ROC AUC of 0.7781 plus or minus 0.0365. GradCAM visualizations further indicate that the model focuses on anatomically meaningful regions, including the medial temporal lobe and ventricular areas that are known to be associated with Alzheimer's related structural changes. To better understand how data representation and evaluation strategies may influence reported performance, additional diagnostic experiments were conducted on a slice based version of the dataset under both slice level and subject level protocols. These observations help provide context for the volumetric results. Overall, the proposed multimodal 3D framework establishes a reproducible subject level benchmark and highlights the potential benefits of volumetric MRI analysis for Alzheimer's disease classification.

  </details>



- **Directing the Narrative: A Finetuning Method for Controlling Coherence and Style in Story Generation**  
  Jianzhang Zhang, Yijing Tian, Jiwang Qu, Chuang Liu  
  _2026-03-18_ · https://arxiv.org/abs/2603.17295v1  
  <details><summary>Abstract</summary>

  Story visualization requires generating sequential imagery that aligns semantically with evolving narratives while maintaining rigorous consistency in character identity and visual style. However, existing methodologies often struggle with subject inconsistency and identity drift, particularly when depicting complex interactions or extended narrative arcs. To address these challenges, we propose a cohesive two-stage framework designed for robust and consistent story generation. First, we introduce Group-Shared Attention (GSA), a mechanism that fosters intrinsic consistency by enabling lossless cross-sample information flow within attention layers. This allows the model to structurally encode identity correspondence across frames without relying on external encoders. Second, we leverage Direct Preference Optimization (DPO) to align generated outputs with human aesthetic and narrative standards. Unlike conventional methods that rely on conflicting auxiliary losses, our approach simultaneously enhances visual fidelity and identity preservation by learning from holistic preference data. Extensive evaluations on the ViStoryBench benchmark demonstrate that our method establishes a new state-of-the-art, significantly outperforming strong baselines with gains of +10.0 in Character Identity (CIDS) and +18.7 in Style Consistency (CSD), all while preserving high-fidelity generation.

  </details>



- **ConfusionBench: An Expert-Validated Benchmark for Confusion Recognition and Localization in Educational Videos**  
  Lu Dong, Xiao Wang, Mark Frank, Srirangaraj Setlur, Venu Govindaraju, Ifeoma Nwogu  
  _2026-03-18_ · https://arxiv.org/abs/2603.17267v1  
  <details><summary>Abstract</summary>

  Recognizing and localizing student confusion from video is an important yet challenging problem in educational AI. Existing confusion datasets suffer from noisy labels, coarse temporal annotations, and limited expert validation, which hinder reliable fine-grained recognition and temporally grounded analysis. To address these limitations, we propose a practical multi-stage filtering pipeline that integrates two stages of model-assisted screening, researcher curation, and expert validation to build a higher-quality benchmark for confusion understanding. Based on this pipeline, we introduce ConfusionBench, a new benchmark for educational videos consisting of a balanced confusion recognition dataset and a video localization dataset. We further provide zero-shot baseline evaluations of a representative open-source model and a proprietary model on clip-level confusion recognition, long-video confusion localization tasks. Experimental results show that the proprietary model performs better overall but tends to over-predict transitional segments, while the open-source model is more conservative and more prone to missed detections. In addition, the proposed student confusion report visualization can support educational experts in making intervention decisions and adapting learning plans accordingly. All datasets and related materials will be made publicly available on our project page.

  </details>



- **LED: A Benchmark for Evaluating Layout Error Detection in Document Analysis**  
  Inbum Heo, Taewook Hwang, Jeesu Jung, Sangkeun Jung  
  _2026-03-18_ · https://arxiv.org/abs/2603.17265v1  
  <details><summary>Abstract</summary>

  Recent advances in Large Language Models (LLMs) and Large Multimodal Models (LMMs) have improved Document Layout Analysis (DLA), yet structural errors such as region merging, splitting, and omission remain persistent. Conventional overlap-based metrics (e.g., IoU, mAP) fail to capture such logical inconsistencies. To overcome this limitation, we propose Layout Error Detection (LED), a benchmark that evaluates structural reasoning in DLA predictions beyond surface-level accuracy. LED defines eight standardized error types (Missing, Hallucination, Size Error, Split, Merge, Overlap, Duplicate, and Misclassification) and provides quantitative rules and injection algorithms for realistic error simulation. Using these definitions, we construct LED-Dataset and design three evaluation tasks: document-level error detection, document-level error-type classification, and element-level error-type classification. Experiments with state-of-the-art multimodal models show that LED enables fine-grained and interpretable assessment of structural understanding, revealing clear weaknesses across modalities and architectures. Overall, LED establishes a unified and explainable benchmark for diagnosing the structural robustness and reasoning capability of document understanding models.

  </details>



- **GigaWorld-Policy: An Efficient Action-Centered World--Action Model**  
  Angen Ye, Boyuan Wang, Chaojun Ni, Guan Huang, Guosheng Zhao, Hao Li, Hengtao Li, Jie Li, Jindi Lv, Jingyu Liu, et al.  
  _2026-03-18_ · https://arxiv.org/abs/2603.17240v1  
  <details><summary>Abstract</summary>

  World-Action Models (WAM) initialized from pre-trained video generation backbones have demonstrated remarkable potential for robot policy learning. However, existing approaches face two critical bottlenecks that hinder performance and deployment. First, jointly reasoning over future visual dynamics and corresponding actions incurs substantial inference overhead. Second, joint modeling often entangles visual and motion representations, making motion prediction accuracy heavily dependent on the quality of future video forecasts. To address these issues, we introduce GigaWorld-Policy, an action-centered WAM that learns 2D pixel-action dynamics while enabling efficient action decoding, with optional video generation. Specifically, we formulate policy training into two coupled components: the model predicts future action sequences conditioned on the current observation, and simultaneously generates future videos conditioned on the predicted actions and the same observation. The policy is supervised by both action prediction and video generation, providing richer learning signals and encouraging physically plausible actions through visual-dynamics constraints. With a causal design that prevents future-video tokens from influencing action tokens, explicit future-video generation is optional at inference time, allowing faster action prediction during deployment. To support this paradigm, we curate a diverse, large-scale robot dataset to pre-train an action-centered video generation model, which is then adapted as the backbone for robot policy learning. Experimental results on real-world robotic platforms show that GigaWorld-Policy runs 9x faster than the leading WAM baseline, Motus, while improving task success rates by 7%. Moreover, compared with pi-0.5, GigaWorld-Policy improves performance by 95% on RoboTwin 2.0.

  </details>



- **Adaptive Anchor Policies for Efficient 4D Gaussian Streaming**  
  Ashim Dahal, Rabab Abdelfattah, Nick Rahimi  
  _2026-03-18_ · https://arxiv.org/abs/2603.17227v1  
  <details><summary>Abstract</summary>

  Dynamic scene reconstruction with Gaussian Splatting has enabled efficient streaming for real-time rendering and free-viewpoint video. However, most pipelines rely on fixed anchor selection such as Farthest Point Sampling (FPS), typically using 8,192 anchors regardless of scene complexity, which over-allocates computation under strict budgets. We propose Efficient Gaussian Streaming (EGS), a plug-in, budget-aware anchor sampler that replaces FPS with a reinforcement-learned policy while keeping the Gaussian streaming reconstruction backbone unchanged. The policy jointly selects an anchor budget and a subset of informative anchors under discrete constraints, balancing reconstruction quality and runtime using spatial features of the Gaussian representation. We evaluate EGS in two settings: fast rendering, which prioritizes runtime efficiency, and high-quality refinement, which enables additional optimization. Experiments on dynamic multi-view datasets show consistent improvements in the quality--efficiency trade-off over FPS sampling. On unseen data, in fast rendering at 256 anchors ($32\times$ fewer than 8,192), EGS improves PSNR by $+0.52$--$0.61$\,dB while running $1.29$--$1.35\times$ faster than IGS@8192 (N3DV and MeetingRoom). In high-quality refinement, EGS remains competitive with the full-anchor baseline at substantially lower anchor budgets. \emph{Code and pretrained checkpoints will be released upon acceptance.} \keywords{4D Gaussian Splatting \and 4D Gaussian Streaming \and Reinforcement Learning}

  </details>



- **FastLoop: Parallel Loop Closing with GPU-Acceleration in Visual SLAM**  
  Soudabeh Mohammadhashemi, Shishir Gopinath, Kimia Khabiri, Parsa Hosseininejad, Karthik Dantu, Steven Y. Ko  
  _2026-03-17_ · https://arxiv.org/abs/2603.17201v1  
  <details><summary>Abstract</summary>

  Visual SLAM systems combine visual tracking with global loop closure to maintain a consistent map and accurate localization. Loop closure is a computationally expensive process as we need to search across the whole map for matches. This paper presents FastLoop, a GPU-accelerated loop closing module to alleviate this computational complexity. We identify key performance bottlenecks in the loop closing pipeline of visual SLAM and address them through parallel optimizations on the GPU. Specifically, we use task-level and data-level parallelism and integrate a GPU-accelerated pose graph optimization. Our implementation is built on top of ORB-SLAM3 and leverages CUDA for GPU programming. Experimental results show that FastLoop achieves an average speedup of 1.4x and 1.3x on the EuRoC dataset and 3.0x and 2.4x on the TUM-VI dataset for the loop closing module on desktop and embedded platforms, respectively, while maintaining the accuracy of the original system.

  </details>



- **Visual Product Search Benchmark**  
  Karthik Sulthanpete Govindappa  
  _2026-03-17_ · https://arxiv.org/abs/2603.17186v1  
  <details><summary>Abstract</summary>

  Reliable product identification from images is a critical requirement in industrial and commercial applications, particularly in maintenance, procurement, and operational workflows where incorrect matches can lead to costly downstream failures. At the core of such systems lies the visual search component, which must retrieve and rank the exact object instance from large and continuously evolving catalogs under diverse imaging conditions. This report presents a structured benchmark of modern visual embedding models for instance-level image retrieval, with a focus on industrial applications. A curated set of open-source foundation embedding models, proprietary multi-modal embedding systems, and domain-specific vision-only models are evaluated under a unified image-to-image retrieval protocol. The benchmark includes curated datasets, which includes industrial datasets derived from production deployments in Manufacturing, Automotive, DIY, and Retail, as well as established public benchmarks. Evaluation is conducted without post-processing, isolating the retrieval capability of each model. The results provide insight into how well contemporary foundation and unified embedding models transfer to fine-grained instance retrieval tasks, and how they compare to models explicitly trained for industrial applications. By emphasizing realistic constraints, heterogeneous image conditions, and exact instance matching requirements, this benchmark aims to inform both practitioners and researchers about the strengths and limitations of current visual embedding approaches in production-level product identification systems. An interactive companion website presenting the benchmark results, evaluation details, and additional visualizations is available at https://benchmark.nyris.io.

  </details>



- **Generalist Multimodal LLMs Gain Biometric Expertise via Human Salience**  
  Jacob Piland, Byron Dowling, Christopher Sweet, Adam Czajka  
  _2026-03-17_ · https://arxiv.org/abs/2603.17173v1  
  <details><summary>Abstract</summary>

  Iris presentation attack detection (PAD) is critical for secure biometric deployments, yet developing specialized models faces significant practical barriers: collecting data representing future unknown attacks is impossible, and collecting diverse-enough data, yet still limited in terms of its predictive power, is expensive. Additionally, sharing biometric data raises privacy concerns. Due to rapid emergence of new attack vectors demanding adaptable solutions, we thus investigate in this paper whether general-purpose multimodal large language models (MLLMs) can perform iris PAD when augmented with human expert knowledge, operating under strict privacy constraints that prohibit sending biometric data to public cloud MLLM services. Through analysis of vision encoder embeddings applied to our dataset, we demonstrate that pre-trained vision transformers in MLLMs inherently cluster many iris attack types despite never being explicitly trained for this task. However, where clustering shows overlap between attack classes, we find that structured prompts incorporating human salience (verbal descriptions from subjects identifying attack indicators) enable these models to resolve ambiguities. Testing on an IRB-restricted dataset of 224 iris images spanning seven attack types, using only university-approved services (Gemini 2.5 Pro) or locally-hosted models (e.g., Llama 3.2-Vision), we show that Gemini with expert-informed prompts outperforms both a specialized convolutional neural networks (CNN)-based baseline and human examiners, while the locally-deployable Llama achieves near-human performance. Our results establish that MLLMs deployable within institutional privacy constraints offer a viable path for iris PAD.

  </details>



- **SLAM Adversarial Lab: An Extensible Framework for Visual SLAM Robustness Evaluation under Adverse Conditions**  
  Mohamed Hefny, Karthik Dantu, Steven Y. Ko  
  _2026-03-17_ · https://arxiv.org/abs/2603.17165v1  
  <details><summary>Abstract</summary>

  We present SAL (SLAM Adversarial Lab), a modular framework for evaluating visual SLAM systems under adversarial conditions such as fog and rain. SAL represents each adversarial condition as a perturbation that transforms an existing dataset into an adversarial dataset. When transforming a dataset, SAL supports severity levels using easily-interpretable real-world units such as meters for fog visibility. SAL's extensible architecture decouples datasets, perturbations, and SLAM algorithms through common interfaces, so users can add new components without rewriting integration code. Moreover, SAL includes a search procedure that finds the severity level of a perturbation at which a SLAM system fails. To showcase the capabilities of SAL, our evaluation integrates seven SLAM algorithms and evaluates them across three datasets under weather, camera, and video transport perturbations.

  </details>



- **GazeOnce360: Fisheye-Based 360° Multi-Person Gaze Estimation with Global-Local Feature Fusion**  
  Zhuojiang Cai, Zhenghui Sun, Feng Lu  
  _2026-03-17_ · https://arxiv.org/abs/2603.17161v1  
  <details><summary>Abstract</summary>

  We present GazeOnce360, a novel end-to-end model for multi-person gaze estimation from a single tabletop-mounted upward-facing fisheye camera. Unlike conventional approaches that rely on forward-facing cameras in constrained viewpoints, we address the underexplored setting of estimating the 3D gaze direction of multiple people distributed across a 360° scene from an upward fisheye perspective. To support research in this setting, we introduce MPSGaze360, a large-scale synthetic dataset rendered using Unreal Engine, featuring diverse multi-person configurations with accurate 3D gaze and eye landmark annotations. Our model tackles the severe distortion and perspective variation inherent in fisheye imagery by incorporating rotational convolutions and eye landmark supervision. To better capture fine-grained eye features crucial for gaze estimation, we propose a dual-resolution architecture that fuses global low-resolution context with high-resolution local eye regions. Experimental results demonstrate the effectiveness of each component in our model. This work highlights the feasibility and potential of fisheye-based 360° gaze estimation in practical multi-person scenarios. Project page: https://caizhuojiang.github.io/GazeOnce360/.

  </details>


