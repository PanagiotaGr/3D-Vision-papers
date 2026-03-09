# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-03-09 07:16 UTC_

Total papers shown: **50**


---

- **Multimodal Large Language Models as Image Classifiers**  
  Nikita Kisel, Illia Volkov, Klara Janouskova, Jiri Matas  
  _2026-03-06_ · https://arxiv.org/abs/2603.06578v1  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLM) classification performance depends critically on evaluation protocol and ground truth quality. Studies comparing MLLMs with supervised and vision-language models report conflicting conclusions, and we show these conflicts stem from protocols that either inflate or underestimate performance. Across the most common evaluation protocols, we identify and fix key issues: model outputs that fall outside the provided class list and are discarded, inflated results from weak multiple-choice distractors, and an open-world setting that underperforms only due to poor output mapping. We additionally quantify the impact of commonly overlooked design choices - batch size, image ordering, and text encoder selection - showing they substantially affect accuracy. Evaluating on ReGT, our multilabel reannotation of 625 ImageNet-1k classes, reveals that MLLMs benefit most from corrected labels (up to +10.8%), substantially narrowing the perceived gap with supervised models. Much of the reported MLLMs underperformance on classification is thus an artifact of noisy ground truth and flawed evaluation protocol rather than genuine model deficiency. Models less reliant on supervised training signals prove most sensitive to annotation quality. Finally, we show that MLLMs can assist human annotators: in a controlled case study, annotators confirmed or integrated MLLMs predictions in approximately 50% of difficult cases, demonstrating their potential for large-scale dataset curation.

  </details>



- **Fly360: Omnidirectional Obstacle Avoidance within Drone View**  
  Xiangkai Zhang, Dizhe Zhang, WenZhuo Cao, Zhaoliang Wan, Yingjie Niu, Lu Qi, Xu Yang, Zhiyong Liu  
  _2026-03-06_ · https://arxiv.org/abs/2603.06573v1  
  <details><summary>Abstract</summary>

  Obstacle avoidance in unmanned aerial vehicles (UAVs), as a fundamental capability, has gained increasing attention with the growing focus on spatial intelligence. However, current obstacle-avoidance methods mainly depend on limited field-of-view sensors and are ill-suited for UAV scenarios which require full-spatial awareness when the movement direction differs from the UAV's heading. This limitation motivates us to explore omnidirectional obstacle avoidance for panoramic drones with full-view perception. We first study an under explored problem setting in which a UAV must generate collision-free motion in environments with obstacles from arbitrary directions, and then construct a benchmark that consists of three representative flight tasks. Based on such settings, we propose Fly360, a two-stage perception-decision pipeline with a fixed random-yaw training strategy. At the perception stage, panoramic RGB observations are input and converted into depth maps as a robust intermediate representation. For the policy network, it is lightweight and used to output body-frame velocity commands from depth inputs. Extensive simulation and real-world experiments demonstrate that Fly360 achieves stable omnidirectional obstacle avoidance and outperforms forward-view baselines across all tasks. Our model is available at https://zxkai.github.io/fly360/

  </details>



- **SUREON: A Benchmark and Vision-Language-Model for Surgical Reasoning**  
  Alejandra Perez, Anita Rau, Lee White, Busisiwe Mlambo, Chinedu Nwoye, Muhammad Abdullah Jamal, Omid Mohareri  
  _2026-03-06_ · https://arxiv.org/abs/2603.06570v1  
  <details><summary>Abstract</summary>

  Surgeons don't just see -- they interpret. When an expert observes a surgical scene, they understand not only what instrument is being used, but why it was chosen, what risk it poses, and what comes next. Current surgical AI cannot answer such questions, largely because training data that explicitly encodes surgical reasoning is immensely difficult to annotate at scale. Yet surgical video lectures already contain exactly this -- explanations of intent, rationale, and anticipation, narrated by experts for the purpose of teaching. Though inherently noisy and unstructured, these narrations encode the reasoning that surgical AI currently lacks. We introduce SUREON, a large-scale video QA dataset that systematically harvests this training signal from surgical academic videos. SUREON defines 12 question categories covering safety assessment, decision rationale, and forecasting, and uses a multi-agent pipeline to extract and structure supervision at scale. Across 134.7K clips and 170 procedure types, SUREON yields 206.8k QA pairs and an expert-validated benchmark of 354 examples. To evaluate the extent to which this supervision translates to surgical reasoning ability, we introduce two models: SureonVLM, a vision-language model adapted through supervised fine-tuning, and SureonVLM-R1, a reasoning model trained with Group Relative Policy Optimization. Both models can answer complex questions about surgery and substantially outperform larger general-domain models, exceeding 84% accuracy on the SUREON benchmark while outperforming general-domain models on standard surgical perception tasks. Qualitative analysis of SureonVLM-R1 reveals explicit reasoning behavior, such as inferring operative intent from visual context.

  </details>



- **EgoReasoner: Learning Egocentric 4D Reasoning via Task-Adaptive Structured Thinking**  
  Fangrui Zhu, Yunfeng Xi, Jianmo Ni, Mu Cai, Boqing Gong, Long Zhao, Chen Qu, Ian Miao, Yi Li, Cheng Zhong, et al.  
  _2026-03-06_ · https://arxiv.org/abs/2603.06561v1  
  <details><summary>Abstract</summary>

  Egocentric video understanding is inherently complex due to the dynamic 4D nature of the environment, where camera motion and object displacements necessitate a continuous re-evaluation of spatial relations. In this work, we target a suite of under-explored egocentric 4D reasoning tasks, including fixture interaction counting, viewpoint-relative fixture location, object movement itinerary tracking, and stationary object localization, that require fundamentally different cognitive operations: spatial anchoring, temporal tracking, and duration reasoning. We observe that these structural differences make task-agnostic approaches insufficient: generic Chain-of-Thought methods lack task-appropriate reasoning primitives, and uniform reinforcement learning actively destabilizes performance on spatial tasks. To address this, we propose EgoReasoner, a two-stage framework that aligns both the reasoning scaffold and the reward signal to each task's cognitive structure. In the first stage, Task-Adaptive Thinking Templates guide the synthesis of structured CoT traces that teach the model to reason adaptively across task types via supervised fine-tuning. In the second stage, task-aware reward functions verify entity grounding, temporal alignment, and task-adaptive logical consistency, selectively strengthening each reasoning pathway via reinforcement fine-tuning with GRPO. Our 3B-parameter model, trained on only 16K samples, achieves 37.5% average accuracy on the challenging HD-EPIC benchmark, surpassing Qwen2.5-VL-7B (25.7%) by over 10 points.

  </details>



- **SurgFormer: Scalable Learning of Organ Deformation with Resection Support and Real-Time Inference**  
  Ashkan Shahbazi, Elaheh Akbari, Kyvia Pereira, Jon S. Heiselman, Annie C. Benson, Garrison L. H. Johnston, Jie Ying Wu, Nabil Simaan, Michael I. Miga, Soheil Kolouri  
  _2026-03-06_ · https://arxiv.org/abs/2603.06543v1  
  <details><summary>Abstract</summary>

  We introduce SurgFormer, a multiresolution gated transformer for data driven soft tissue simulation on volumetric meshes. High fidelity biomechanical solvers are often too costly for interactive use, so we train SurgFormer on solver generated data to predict nodewise displacement fields at near real time rates. SurgFormer builds a fixed mesh hierarchy and applies repeated multibranch blocks that combine local message passing, coarse global self attention, and pointwise feedforward updates, fused by learned per node, per channel gates to adaptively integrate local and long range information while remaining scalable on large meshes. For cut conditioned simulation, resection information is encoded as a learned cut embedding and provided as an additional input, enabling a unified model for both standard deformation prediction and topology altering cases. We also introduce two surgical simulation datasets generated under a unified protocol with XFEM based supervision: a cholecystectomy resection dataset and an appendectomy manipulation and resection dataset with cut and uncut cases. To our knowledge, this is the first learned volumetric surrogate setting to study XFEM supervised cut conditioned deformation within the same volumetric pipeline as standard deformation prediction. Across diverse baselines, SurgFormer achieves strong accuracy with favorable efficiency, making it a practical backbone for both tasks. {Code, data, and project page: \href{https://mint-vu.github.io/SurgFormer/}{available here}}

  </details>



- **NEGATE: Constrained Semantic Guidance for Linguistic Negation in Text-to-Video Diffusion**  
  Taewon Kang, Ming C. Lin  
  _2026-03-06_ · https://arxiv.org/abs/2603.06533v1  
  <details><summary>Abstract</summary>

  Negation is a fundamental linguistic operator, yet it remains inadequately modeled in diffusion-based generative systems. In this work, we present a formal treatment of linguistic negation in diffusion-based generative models by modeling it as a structured feasibility constraint on semantic guidance within diffusion dynamics. Rather than introducing heuristics or retraining model parameters, we reinterpret classifier-free guidance as defining a semantic update direction and enforce negation by projecting the update onto a convex constraint set derived from linguistic structure. This novel formulation provides a unified framework for handling diverse negation phenomena, including object absence, graded non-inversion semantics, multi-negation composition, and scope-sensitive disambiguation. Our approach is training-free, compatible with pretrained diffusion backbones, and naturally extends from image generation to temporally evolving video trajectories. In addition, we introduce a structured negation-centric benchmark suite that isolates distinct linguistic failure modes in generative systems, to further research in this area. Experiments demonstrate that our method achieves robust negation compliance while preserving visual fidelity and structural coherence, establishing the first unified formulation of linguistic negation in diffusion-based generative models beyond representation-level evaluation.

  </details>



- **AV-Unified: A Unified Framework for Audio-visual Scene Understanding**  
  Guangyao Li, Xin Wang, Wenwu Zhu  
  _2026-03-06_ · https://arxiv.org/abs/2603.06530v1  
  <details><summary>Abstract</summary>

  When humans perceive the world, they naturally integrate multiple audio-visual tasks within dynamic, real-world scenes. However, current works such as event localization, parsing, segmentation and question answering are mostly explored individually, making it challenging to comprehensively understand complex audio-visual scenes and explore inter-task relationships. Hence, we propose \textbf{AV-Unified}, a unified framework that enables joint learning across a wide range of audio-visual scene understanding tasks. AV-Unified standardizes the diverse input-output formats of each task and incorporates a multi-scale spatiotemporal perception network to effectively capture audio-visual associations. Specifically, we unify the inputs and outputs of all supported tasks by converting them into sequences of discrete tokens, establishing a shared representation that allows a single architecture to be trained jointly across heterogeneous varied datasets. Considering the varying temporal granularity of audio-visual events, a multi-scale temporal perception module is designed to capture key cues. Meanwhile, to overcome the lack of auditory supervision in the visual domain, we design a cross-modal guidance-based spatial perception module that models spatial audio-visual associations. Furthermore, task-specific text prompts are employed to enhance the model's adaptability and task-awareness. Extensive experiments on benchmark datasets (e.g., AVE, LLP, MUSIC-AVQA, VGG-SS and AVS) demonstrate the effectiveness of AV-Unified across temporal, spatial, and spatiotemporal tasks.

  </details>



- **SG-DOR: Learning Scene Graphs with Direction-Conditioned Occlusion Reasoning for Pepper Plants**  
  Rohit Menon, Niklas Mueller-Goldingen, Sicong Pan, Gokul Krishna Chenchani, Maren Bennewitz  
  _2026-03-06_ · https://arxiv.org/abs/2603.06512v1  
  <details><summary>Abstract</summary>

  Robotic harvesting in dense crop canopies requires effective interventions that depend not only on geometry, but also on explicit, direction-conditioned relations identifying which organs obstruct a target fruit. We present SG-DOR (Scene Graphs with Direction-Conditioned Occlusion Reasoning), a relational framework that, given instance-segmented organ point clouds, infers a scene graph encoding physical attachments and direction-conditioned occlusion. We introduce an occlusion ranking task for retrieving and ranking candidate leaves for a target fruit and approach direction, and propose a direction-aware graph neural architecture with per-fruit leaf-set attention and union-level aggregation. Experiments on a multi-plant synthetic pepper dataset show improved occlusion prediction (F1=0.73, NDCG@3=0.85) and attachment inference (edge F1=0.83) over strong ablations, yielding a structured relational signal for downstream intervention planning.

  </details>



- **CFEAR-Teach-and-Repeat: Fast and Accurate Radar-only Localization**  
  Maximilian Hilger, Daniel Adolfsson, Ralf Becker, Henrik Andreasson, Achim J. Lilienthal  
  _2026-03-06_ · https://arxiv.org/abs/2603.06501v1  
  <details><summary>Abstract</summary>

  Reliable localization in prior maps is essential for autonomous navigation, particularly under adverse weather, where optical sensors may fail. We present CFEAR-TR, a teach-and-repeat localization pipeline using a single spinning radar, which is designed for easily deployable, lightweight, and robust navigation in adverse conditions. Our method localizes by jointly aligning live scans to both stored scans from the teach mapping pass, and to a sliding window of recent live keyframes. This ensures accurate and robust pose estimation across different seasons and weather phenomena. Radar scans are represented using a sparse set of oriented surface points, computed from Doppler-compensated measurements. The map is stored in a pose graph that is traversed during localization. Experiments on the held-out test sequences from the Boreas dataset show that CFEAR-TR can localize with an accuracy as low as 0.117 m and 0.096°, corresponding to improvements of up to 63% over the previous state of the art, while running efficiently at 29 Hz. These results substantially narrow the gap to lidar-level localization, particularly in heading estimation. We make the C++ implementation of our work available to the community.

  </details>



- **Match4Annotate: Propagating Sparse Video Annotations via Implicit Neural Feature Matching**  
  Zhuorui Zhang, Roger Pallarès-López, Praneeth Namburi, Brian W. Anthony  
  _2026-03-06_ · https://arxiv.org/abs/2603.06471v1  
  <details><summary>Abstract</summary>

  Acquiring per-frame video annotations remains a primary bottleneck for deploying computer vision in specialized domains such as medical imaging, where expert labeling is slow and costly. Label propagation offers a natural solution, yet existing approaches face fundamental limitations. Video trackers and segmentation models can propagate labels within a single sequence but require per-video initialization and cannot generalize across videos. Classic correspondence pipelines operate on detector-chosen keypoints and struggle in low-texture scenes, while dense feature matching and one-shot segmentation methods enable cross-video propagation but lack spatiotemporal smoothness and unified support for both point and mask annotations. We present Match4Annotate, a lightweight framework for both intra-video and inter-video propagation of point and mask annotations. Our method fits a SIREN-based implicit neural representation to DINOv3 features at test time, producing a continuous, high-resolution spatiotemporal feature field, and learns a smooth implicit deformation field between frame pairs to guide correspondence matching. We evaluate on three challenging clinical ultrasound datasets. Match4Annotate achieves state-of-the-art inter-video propagation, outperforming feature matching and one-shot segmentation baselines, while remaining competitive with specialized trackers for intra-video propagation. Our results show that lightweight, test-time-optimized feature matching pipelines have the potential to offer an efficient and accessible solution for scalable annotation workflows.

  </details>



- **GreenRFM: Toward a resource-efficient radiology foundation model**  
  Yingtai Li, Shuai Ming, Mingyue Zhao, Haoran Lai, Rongsheng Wang, Rui Zhou, Rundong Wang, Yujia Li, Wei Wei, Shaohua Kevin Zhou  
  _2026-03-06_ · https://arxiv.org/abs/2603.06467v1  
  <details><summary>Abstract</summary>

  The development of radiology foundation models (RFMs) is hindered by a reliance on brute-force scaling. Existing approaches often directly translate methods for natural images, which prioritize scale over precision and hence lead to brittle and expensive models in clinical practice. To address this, we present a resource-efficient pre-training framework, GreenRFM, that achieves state-of-the-art performance. Our framework ensures robust generalization across diverse patient populations and imaging protocols, reducing computational requirements by orders of magnitude while surpassing complex, parameter-heavy models. These capabilities stem from principled supervision design that aims to maximally utilize supervisory signals via More distilled, Ubiquitous, Semantic-enforcing, and Task-aligning (MUST) supervision, rather than simply piling up the quantity of training data. We offer two GreenRFM configurations: (i) a performant model that establishes a new state-of-the-art using a single 24GB GPU within 24 hours, and (ii) a lightweight model that matches existing benchmarks with 6GB VRAM in 4 hours. We conduct extensive experiments using over 200,000 images from four institutions and of two modalities. GreenRFMs achieve superior performances on chest and abdominal CT datasets, regardless of public or private benchmark, surpassing a range of baseline models. In addition, the results on internal musculoskeletal MRI images show that the same supervision principles transfer between different modalities. Our performance and efficiency challenge the ``scale is all you need'' dogma and democratize the equitable development of state-of-the-art RFMs for clinicians even on a laptop.

  </details>



- **Training Flow Matching: The Role of Weighting and Parameterization**  
  Anne Gagneux, Ségolène Martin, Rémi Gribonval, Mathurin Massias  
  _2026-03-06_ · https://arxiv.org/abs/2603.06454v1  
  <details><summary>Abstract</summary>

  We study the training objectives of denoising-based generative models, with a particular focus on loss weighting and output parameterization, including noise-, clean image-, and velocity-based formulations. Through a systematic numerical study, we analyze how these training choices interact with the intrinsic dimensionality of the data manifold, model architecture, and dataset size. Our experiments span synthetic datasets with controlled geometry as well as image data, and compare training objectives using quantitative metrics for denoising accuracy (PSNR across noise levels) and generative quality (FID). Rather than proposing a new method, our goal is to disentangle the various factors that matter when training a flow matching model, in order to provide practical insights on design choices.

  </details>



- **Pinterest Canvas: Large-Scale Image Generation at Pinterest**  
  Yu Wang, Eric Tzeng, Raymond Shiau, Jie Yang, Dmitry Kislyuk, Charles Rosenberg  
  _2026-03-06_ · https://arxiv.org/abs/2603.06453v1  
  <details><summary>Abstract</summary>

  While recent image generation models demonstrate a remarkable ability to handle a wide variety of image generation tasks, this flexibility makes them hard to control via prompting or simple inference adaptation alone, rendering them unsuitable for use cases with strict product requirements. In this paper, we introduce Pinterest Canvas, our large-scale image generation system built to support image editing and enhancement use cases at Pinterest. Canvas is first trained on a diverse, multimodal dataset to produce a foundational diffusion model with broad image-editing capabilities. However, rather than relying on one generic model to handle every downstream task, we instead rapidly fine-tune variants of this base model on task-specific datasets, producing specialized models for individual use cases. We describe key components of Canvas and summarize our best practices for dataset curation, training, and inference. We also showcase task-specific variants through case studies on background enhancement and aspect-ratio outpainting, highlighting how we tackle their specific product requirements. Online A/B experiments demonstrate that our enhanced images receive a significant 18.0% and 12.5% engagement lift, respectively, and comparisons with human raters further validate that our models outperform third-party models on these tasks. Finally, we showcase other Canvas variants, including multi-image scene synthesis and image-to-video generation, demonstrating that our approach can generalize to a wide variety of potential downstream tasks.

  </details>



- **What if? Emulative Simulation with World Models for Situated Reasoning**  
  Ruiping Liu, Yufan Chen, Yuheng Zhang, Junwei Zheng, Kunyu Peng, Chengzhi Wu, Chenguang Huang, Di Wen, Jiaming Zhang, Kailun Yang, et al.  
  _2026-03-06_ · https://arxiv.org/abs/2603.06445v1  
  <details><summary>Abstract</summary>

  Situated reasoning often relies on active exploration, yet in many real-world scenarios such exploration is infeasible due to physical constraints of robots or safety concerns of visually impaired users. Given only a limited observation, can an agent mentally simulate a future trajectory toward a target situation and answer spatial what-if questions? We introduce WanderDream, the first large-scale dataset designed for the emulative simulation of mental exploration, enabling models to reason without active exploration. WanderDream-Gen comprises 15.8K panoramic videos across 1,088 real scenes from HM3D, ScanNet++, and real-world captures, depicting imagined trajectories from current viewpoints to target situations. WanderDream-QA contains 158K question-answer pairs, covering starting states, paths, and end states along each trajectory to comprehensively evaluate exploration-based reasoning. Extensive experiments with world models and MLLMs demonstrate (1) that mental exploration is essential for situated reasoning, (2) that world models achieve compelling performance on WanderDream-Gen, (3) that imagination substantially facilitates reasoning on WanderDream-QA, and (4) that WanderDream data exhibit remarkable transferability to real-world scenarios. The source code and all data will be released.

  </details>



- **CLoPA: Continual Low Parameter Adaptation of Interactive Segmentation for Medical Image Annotation**  
  Parhom Esmaeili, Chayanin Tangwiriyasakul, Eli Gibson, Sebastien Ourselin, M. Jorge Cardoso  
  _2026-03-06_ · https://arxiv.org/abs/2603.06426v1  
  <details><summary>Abstract</summary>

  Interactive segmentation enables clinicians to guide annotation, but existing zero-shot models like nnInteractive fail to consistently reach expert-level performance across diverse medical imaging tasks. Because annotation campaigns produce a growing stream of task-specific labelled data, online adaptation of the segmentation model is a natural complement to zero-shot inference. We propose CLoPA, a continual adaptation strategy that tunes a small fraction of nnInteractive's parameters on the annotation cache, triggered by lightweight episode scheduling. CLoPA requires no new parameters or changes to the inference pipeline, and operates entirely within the existing annotation workflow. Across eight Medical Segmentation Decathlon tasks spanning diverse anatomical targets and imaging characteristics, CLoPA rapidly elevates performance to expert-level, even for tasks where nnInteractive previously failed, with the majority of gains realised after a single training episode. We show that the benefits of tuning different parameter groups depends on task characteristics and data regimes. Also, that for targets with complex geometries (e.g., hepatic vessels), instance normalisation and low-level feature tuning saturates, suggesting a need for deeper feature-representation alignment in the most challenging scenarios.

  </details>



- **Non-invasive Growth Monitoring of Small Freshwater Fish in Home Aquariums via Stereo Vision**  
  Clemens Seibold, Anna Hilsmann, Peter Eisert  
  _2026-03-06_ · https://arxiv.org/abs/2603.06421v1  
  <details><summary>Abstract</summary>

  Monitoring fish growth behavior provides relevant information about fish health in aquaculture and home aquariums. Yet, monitoring fish sizes poses different challenges, as fish are small and subject to strong refractive distortions in aquarium environments. Image-based measurement offers a practical, non-invasive alternative that allows frequent monitoring without disturbing the fish. In this paper, we propose a non-invasive refraction-aware stereo vision method to estimate fish length in aquariums. Our approach uses a YOLOv11-Pose network to detect fish and predict anatomical keypoints on the fish in each stereo image. A refraction-aware epipolar constraint accounting for the air-glass-water interfaces enables robust matching, and unreliable detections are removed using a learned quality score. A subsequent refraction-aware 3D triangulation recovers 3D keypoints, from which fish length is measured. We validate our approach on a new stereo dataset of endangered Sulawesi ricefish captured under aquarium-like conditions and demonstrate that filtering low-quality detections is essential for accurate length estimation. The proposed system offers a simple and practical solution for non-invasive growth monitoring and can be easily applied in home aquariums.

  </details>



- **DiffInf: Influence-Guided Diffusion for Supervision Alignment in Facial Attribute Learning**  
  Basudha Pal, Rama Chellappa  
  _2026-03-06_ · https://arxiv.org/abs/2603.06399v1  
  <details><summary>Abstract</summary>

  Facial attribute classification relies on large-scale annotated datasets in which many traits, such as age and expression, are inherently ambiguous and continuous but are discretized into categorical labels. Annotation inconsistencies arise from subjectivity and visual confounders such as pose, illumination, expression, and demographic variation, creating mismatch between images and assigned labels. These inconsistencies introduce supervision errors that impair representation learning and degrade downstream prediction. We introduce DiffInf, a self-influence--guided diffusion framework for mitigating annotation inconsistencies in facial attribute learning. We first train a baseline classifier and compute sample-wise self-influence scores using a practical first-order approximation to identify training instances that disproportionately destabilize optimization. Instead of discarding these influential samples, we apply targeted generative correction via a latent diffusion autoencoder to better align visual content with assigned labels while preserving identity and realism. To enable differentiable guidance during correction, we train a lightweight predictor of high-influence membership and use it as a surrogate influence regularizer. The edited samples replace the originals, yielding an influence-refined dataset of unchanged size. Across multi-class facial attribute classification, DiffInf consistently improves generalization compared with standard noisy-label training, robust optimization baselines, and influence-based filtering. Our results demonstrate that repairing influential annotation inconsistencies at the image level enhances downstream facial attribute classification without sacrificing distributional coverage.

  </details>



- **Solving Jigsaw Puzzles in the Wild: Human-Guided Reconstruction of Cultural Heritage Fragments**  
  Omidreza Safaei, Sinem Aslan, Sebastiano Vascon, Luca Palmieri, Marina Khoroshiltseva, Marcello Pelillo  
  _2026-03-06_ · https://arxiv.org/abs/2603.06389v1  
  <details><summary>Abstract</summary>

  Reassembling real-world archaeological artifacts from fragmented pieces poses significant challenges due to erosion, missing regions, irregular shapes, and large-scale ambiguity. Traditional jigsaw puzzle solvers, often designed for clean synthetic scenarios, struggle under these conditions, especially when the number of fragments grows into the thousands, as in the RePAIR benchmark. In this paper, we propose a human-in-the-loop (HIL) puzzle solving framework designed to address the complexity and scale of real-world cultural heritage reconstruction. Our approach integrates an automatic relaxation-labeling solver with interactive human guidance, allowing users to iteratively lock verified placements, correct errors, and guide the system toward semantically and geometrically coherent assemblies. We introduce two complementary interaction strategies, Iterative Anchoring and Continuous Interactive Refinement, which support scalable reconstruction across varying levels of ambiguity and puzzle size. Experiments on several RePAIR groups demonstrate that our hybrid approach substantially outperforms both fully automatic and manual baselines in accuracy and efficiency, offering a practical solution for large-scale expert-in-the-loop artifact reassembly.

  </details>



- **REACT++: Efficient Cross-Attention for Real-Time Scene Graph Generation**  
  Maëlic Neau, Zoe Falomir  
  _2026-03-06_ · https://arxiv.org/abs/2603.06386v1  
  <details><summary>Abstract</summary>

  Scene Graph Generation (SGG) is a task that encodes visual relationships between objects in images as graph structures. SGG shows significant promise as a foundational component for downstream tasks, such as reasoning for embodied agents. To enable real-time applications, SGG must address the trade-off between performance and inference speed. However, current methods tend to focus on one of the following: (1) improving relation prediction accuracy, (2) enhancing object detection accuracy, or (3) reducing latency, without aiming to balance all three objectives simultaneously. To address this limitation, we build on the powerful Real-time Efficiency and Accuracy Compromise for Tradeoffs in Scene Graph Generation (REACT) architecture and propose REACT++, a new state-of-the-art model for real-time SGG. By leveraging efficient feature extraction and subject-to-object cross-attention within the prototype space, REACT++ balances latency and representational power. REACT++ achieves the highest inference speed among existing SGG models, improving relation prediction accuracy without sacrificing object detection performance. Compared to the previous REACT version, REACT++ is 20% faster with a gain of 10% in relation prediction accuracy on average. The code is available at https://github.com/Maelic/SGG-Benchmark.

  </details>



- **Prompt Group-Aware Training for Robust Text-Guided Nuclei Segmentation**  
  Yonghuang Wu, Zhenyang Liang, Wenwen Zeng, Xuan Xie, Jinhua Yu  
  _2026-03-06_ · https://arxiv.org/abs/2603.06384v1  
  <details><summary>Abstract</summary>

  Foundation models such as Segment Anything Model 3 (SAM3) enable flexible text-guided medical image segmentation, yet their predictions remain highly sensitive to prompt formulation. Even semantically equivalent descriptions can yield inconsistent masks, limiting reliability in clinical and pathology workflows. We reformulate prompt sensitivity as a group-wise consistency problem. Semantically related prompts are organized into \emph{prompt groups} sharing the same ground-truth mask, and a prompt group-aware training framework is introduced for robust text-guided nuclei segmentation. The approach combines (i) a quality-guided group regularization that leverages segmentation loss as an implicit ranking signal, and (ii) a logit-level consistency constraint with a stop-gradient strategy to align predictions within each group. The method requires no architectural modification and leaves inference unchanged. Extensive experiments on multi-dataset nuclei benchmarks show consistent gains under textual prompting and markedly reduced performance variance across prompt quality levels. On six zero-shot cross-dataset tasks, our method improves Dice by an average of 2.16 points. These results demonstrate improved robustness and generalization for vision-language segmentation in computational pathology.

  </details>



- **OralGPT-Plus: Learning to Use Visual Tools via Reinforcement Learning for Panoramic X-ray Analysis**  
  Yuxuan Fan, Jing Hao, Hong Chen, Jiahao Bao, Yihua Shao, Yuci Liang, Kuo Feng Hung, Hao Tang  
  _2026-03-06_ · https://arxiv.org/abs/2603.06366v1  
  <details><summary>Abstract</summary>

  Panoramic dental radiographs require fine-grained spatial reasoning, bilateral symmetry understanding, and multi-step diagnostic verification, yet existing vision-language models operate under a static single-pass paradigm that limits their clinical reliability. In this paper, we introduce OralGPT-Plus, an agentic vision-language model designed to perform iterative and symmetry-aware diagnostic reasoning for panoramic dental radiograph analysis. To support this paradigm, we construct DentalProbe, a five-thousand-image dataset with expert-curated diagnostic trajectories that provide structured supervision for localized inspection and contralateral comparison. We further develop a Reinspection-driven reinforcement learning framework that encourages clinically meaningful re-examination and stabilizes long-horizon reasoning with rubric-based reward and conditioned diagnostic-driven reward. In parallel, we present MMOral-X, the first benchmark for holistic panoramic diagnosis, containing 300 open-ended questions and region-level annotations across multiple difficulty levels. OralGPT-Plus demonstrates consistent and reliable improvements over strong baselines on MMOral-X and established panoramic benchmarks, indicating the effectiveness of interactive and symmetry-informed reasoning. Our work highlights the value of agentic modeling for dental imaging and provides a foundation for future research in clinically aligned panoramic radiograph analysis.

  </details>



- **Computer vision-based estimation of invertebrate biomass**  
  Mikko Impiö, Philipp M. Rehsen, Jarrett Blair, Cecilie Mielec, Arne J. Beermann, Florian Leese, Toke T. Høye, Jenni Raitoharju  
  _2026-03-06_ · https://arxiv.org/abs/2603.06362v1  
  <details><summary>Abstract</summary>

  The ability to estimate invertebrate biomass using only images could help scaling up quantitative biodiversity monitoring efforts. Computer vision-based methods have the potential to omit the manual, time-consuming, and destructive process of dry weighing specimens. We present two approaches for dry mass estimation that do not require additional manual effort apart from imaging the specimens: fitting a linear model with novel predictors, automatically calculated by an imaging device, and training a family of end-to-end deep neural networks for the task, using single-view, multi-view, and metadata-aware architectures. We propose using area and sinking speed as predictors. These can be calculated with BIODISCOVER, which is a dual-camera system that captures image sequences of specimens sinking in an ethanol column. For this study, we collected a large dataset of dry mass measurement and image sequence pairs to train and evaluate models. We show that our methods can estimate specimen dry mass even with complex and visually diverse specimen morphologies. Combined with automatic taxonomic classification, our approach is an accurate method for group-level dry mass estimation, with a median percentage error of 10-20% for individuals. We highlight the importance of choosing appropriate evaluation metrics, and encourage using both percentage errors and absolute errors as metrics, because they measure different properties. We also explore different optimization losses, data augmentation methods, and model architectures for training deep-learning models.

  </details>



- **P-SLCR: Unsupervised Point Cloud Semantic Segmentation via Prototypes Structure Learning and Consistent Reasoning**  
  Lixin Zhan, Jie Jiang, Tianjian Zhou, Yukun Du, Yan Zheng, Xuehu Duan  
  _2026-03-06_ · https://arxiv.org/abs/2603.06321v1  
  <details><summary>Abstract</summary>

  Current semantic segmentation approaches for point cloud scenes heavily rely on manual labeling, while research on unsupervised semantic segmentation methods specifically for raw point clouds is still in its early stages. Unsupervised point cloud learning poses significant challenges due to the absence of annotation information and the lack of pre-training. The development of effective strategies is crucial in this context. In this paper, we propose a novel prototype library-driven unsupervised point cloud semantic segmentation strategy that utilizes Structure Learning and Consistent Reasoning (P-SLCR). First, we propose a Consistent Structure Learning to establish structural feature learning between consistent points and the library of consistent prototypes by selecting high-quality features. Second, we propose a Semantic Relation Consistent Reasoning that constructs a prototype inter-relation matrix between consistent and ambiguous prototype libraries separately. This process ensures the preservation of semantic consistency by imposing constraints on consistent and ambiguous prototype libraries through the prototype inter-relation matrix. Finally, our method was extensively evaluated on the S3DIS, SemanticKITTI, and Scannet datasets, achieving the best performance compared to unsupervised methods. Specifically, the mIoU of 47.1% is achieved for Area-5 of the S3DIS dataset, surpassing the classical fully supervised method PointNet by 2.5%.

  </details>



- **Attribute Distribution Modeling and Semantic-Visual Alignment for Generative Zero-shot Learning**  
  Haojie Pu, Zhuoming Li, Yongbiao Gao, Yuheng Jia  
  _2026-03-06_ · https://arxiv.org/abs/2603.06281v1  
  <details><summary>Abstract</summary>

  Generative zero-shot learning (ZSL) synthesizes features for unseen classes, leveraging semantic conditions to transfer knowledge from seen classes. However, it also introduces two intrinsic challenges: (1) class-level attributes fails to capture instance-specific visual appearances due to substantial intra-class variability, thus causing the class-instance gap; (2) the substantial mismatch between semantic and visual feature distributions, manifested in inter-class correlations, gives rise to the semantic-visual domain gap. To address these challenges, we propose an Attribute Distribution Modeling and Semantic-Visual Alignment (ADiVA) approach, jointly modeling attribute distributions and performing explicit semantic-visual alignment. Specifically, our ADiVA consists of two modules: an Attribute Distribution Modeling (ADM) module that learns a transferable attribute distribution for each class and samples instance-level attributes for unseen classes, and a Visual-Guided Alignment (VGA) module that refines semantic representations to better reflect visual structures. Experiments on three widely used benchmark datasets demonstrate that ADiVA significantly outperforms state-of-the-art methods (e.g., achieving gains of 4.7% and 6.1% on AWA2 and SUN, respectively). Moreover, our approach can serve as a plugin to enhance existing generative ZSL methods.

  </details>



- **SuperSuit: An Isomorphic Bimodal Interface for Scalable Mobile Manipulation**  
  Tongqing Chen, Hang Wu, Jiasen Wang, Xiaotao Li, Zhu Jin, Lu Fang  
  _2026-03-06_ · https://arxiv.org/abs/2603.06280v1  
  <details><summary>Abstract</summary>

  High-quality, long-horizon demonstrations are essential for embodied AI, yet acquiring such data for tightly coupled wheeled mobile manipulators remains a fundamental bottleneck. Unlike fixed-base systems, mobile manipulators require continuous coordination between $SE(2)$ locomotion and precise manipulation, exposing limitations in existing teleoperation and wearable interfaces. We present \textbf{SuperSuit}, a bimodal data acquisition framework that supports both robot-in-the-loop teleoperation and active demonstration under a shared kinematic interface. Both modalities produce structurally identical joint-space trajectories, enabling direct data mixing without modifying downstream policies. For locomotion, SuperSuit maps natural human stepping to continuous planar base velocities, eliminating discrete command switches. For manipulation, it employs a strictly isomorphic wearable arm in both modes, while policy training is formulated in a shift-invariant delta-joint representation to mitigate calibration offsets and structural compliance without inverse kinematics. Real-world experiments on long-horizon mobile manipulation tasks show 2.6$\times$ higher demonstration throughput in active mode compared to a teleoperation baseline, comparable policy performance when substituting teleoperation data with active demonstrations at fixed dataset size, and monotonic performance improvement as active data volume increases. These results indicate that consistent kinematic representations across collection modalities enable scalable data acquisition for long-horizon mobile manipulation.

  </details>



- **Can we Trust Unreliable Voxels? Exploring 3D Semantic Occupancy Prediction under Label Noise**  
  Wenxin Li, Kunyu Peng, Di Wen, Junwei Zheng, Jiale Wei, Mengfei Duan, Yuheng Zhang, Rui Fan, Kailun Yang  
  _2026-03-06_ · https://arxiv.org/abs/2603.06279v1  
  <details><summary>Abstract</summary>

  3D semantic occupancy prediction is a cornerstone of robotic perception, yet real-world voxel annotations are inherently corrupted by structural artifacts and dynamic trailing effects. This raises a critical but underexplored question: can autonomous systems safely rely on such unreliable occupancy supervision? To systematically investigate this issue, we establish OccNL, the first benchmark dedicated to 3D occupancy under occupancy-asymmetric and dynamic trailing noise. Our analysis reveals a fundamental domain gap: state-of-the-art 2D label noise learning strategies collapse catastrophically in sparse 3D voxel spaces, exposing a critical vulnerability in existing paradigms. To address this challenge, we propose DPR-Occ, a principled label noise-robust framework that constructs reliable supervision through dual-source partial label reasoning. By synergizing temporal model memory with representation-level structural affinity, DPR-Occ dynamically expands and prunes candidate label sets to preserve true semantics while suppressing noise propagation. Extensive experiments on SemanticKITTI demonstrate that DPR-Occ prevents geometric and semantic collapse under extreme corruption. Notably, even at 90% label noise, our method achieves significant performance gains (up to 2.57% mIoU and 13.91% IoU) over existing label noise learning baselines adapted to the 3D occupancy prediction task. By bridging label noise learning and 3D perception, OccNL and DPR-Occ provide a reliable foundation for safety-critical robotic perception in dynamic environments. The benchmark and source code will be made publicly available at https://github.com/mylwx/OccNL.

  </details>



- **GazeMoE: Perception of Gaze Target with Mixture-of-Experts**  
  Zhuangzhuang Dai, Zhongxi Lu, Vincent G. Zakka, Luis J. Manso, Jose M Alcaraz Calero, Chen Li  
  _2026-03-06_ · https://arxiv.org/abs/2603.06256v1  
  <details><summary>Abstract</summary>

  Estimating human gaze target from visible images is a critical task for robots to understand human attention, yet the development of generalizable neural architectures and training paradigms remains challenging. While recent advances in pre-trained vision foundation models offer promising avenues for locating gaze targets, the integration of multi-modal cues -- including eyes, head poses, gestures, and contextual features -- demands adaptive and efficient decoding mechanisms. Inspired by Mixture-of-Experts (MoE) for adaptive domain expertise in large vision-language models, we propose GazeMoE, a novel end-to-end framework that selectively leverages gaze-target-related cues from a frozen foundation model through MoE modules. To address class imbalance in gaze target classification (in-frame vs. out-of-frame) and enhance robustness, GazeMoE incorporates a class-balancing auxiliary loss alongside strategic data augmentations, including region-specific cropping and photometric transformations. Extensive experiments on benchmark datasets demonstrate that our GazeMoE achieves state-of-the-art performance, outperforming existing methods on challenging gaze estimation tasks. The code and pre-trained models are released at https://huggingface.co/zdai257/GazeMoE

  </details>



- **NOVA: Next-step Open-Vocabulary Autoregression for 3D Multi-Object Tracking in Autonomous Driving**  
  Kai Luo, Xu Wang, Rui Fan, Kailun Yang  
  _2026-03-06_ · https://arxiv.org/abs/2603.06254v1  
  <details><summary>Abstract</summary>

  Generalizing across unknown targets is critical for open-world perception, yet existing 3D Multi-Object Tracking (3D MOT) pipelines remain limited by closed-set assumptions and ``semantic-blind'' heuristics. To address this, we propose Next-step Open-Vocabulary Autoregression (NOVA), an innovative paradigm that shifts 3D tracking from traditional fragmented distance-based matching toward generative spatio-temporal semantic modeling. NOVA reformulates 3D trajectories as structured spatio-temporal semantic sequences, enabling the simultaneous encoding of physical motion continuity and deep linguistic priors. By leveraging the autoregressive capabilities of Large Language Models (LLMs), we transform the tracking task into a principled process of next-step sequence completion. This mechanism allows the model to explicitly utilize the hierarchical structure of language space to resolve fine-grained semantic ambiguities and maintain identity consistency across complex long-range sequences through high-level commonsense reasoning. Extensive experiments on nuScenes, V2X-Seq-SPD, and KITTI demonstrate the superior performance of NOVA. Notably, on the nuScenes dataset, NOVA achieves an AMOTA of 22.41% for Novel categories, yielding a significant 20.21% absolute improvement over the baseline. These gains are realized through a compact 0.5B autoregressive model. Code will be available at https://github.com/xifen523/NOVA.

  </details>



- **Word-Anchored Temporal Forgery Localization**  
  Tianyi Wang, Xi Shao, Harry Cheng, Yinglong Wang, Mohan Kankanhalli  
  _2026-03-06_ · https://arxiv.org/abs/2603.06220v1  
  <details><summary>Abstract</summary>

  Current temporal forgery localization (TFL) approaches typically rely on temporal boundary regression or continuous frame-level anomaly detection paradigms to derive candidate forgery proposals. However, they suffer not only from feature granularity misalignment but also from costly computation. To address these issues, we propose word-anchored temporal forgery localization (WAFL), a novel paradigm that shifts the TFL task from temporal regression and continuous localization to discrete word-level binary classification. Specifically, we first analyze the essence of temporal forgeries and identify the minimum meaningful forgery units, word tokens, and then align data preprocessing with the natural linguistic boundaries of speech. To adapt powerful pre-trained foundation backbones for feature extraction, we introduce the forensic feature realignment (FFR) module, mapping representations from the pre-trained semantic space to a discriminative forensic manifold. This allows subsequent lightweight linear classifiers to efficiently perform binary classification and accomplish the TFL task. Furthermore, to overcome the extreme class imbalance inherent to forgery detection, we design the artifact-centric asymmetric (ACA) loss, which breaks the standard precision-recall trade-off by dynamically suppressing overwhelming authentic gradients while asymmetrically prioritizing subtle forensic artifacts. Extensive experiments demonstrate that WAFL significantly outperforms state-of-the-art approaches in localization performance under both in- and cross-dataset settings, while requiring substantially fewer learnable parameters and operating at high computational efficiency.

  </details>



- **Few-Shot Neural Differentiable Simulator: Real-to-Sim Rigid-Contact Modeling**  
  Zhenhao Huang, Siyuan Luo, Bingyang Zhou, Ziqiu Zeng, Jason Pho, Fan Shi  
  _2026-03-06_ · https://arxiv.org/abs/2603.06218v1  
  <details><summary>Abstract</summary>

  Accurate physics simulation is essential for robotic learning and control, yet analytical simulators often fail to capture complex contact dynamics, while learning-based simulators typically require large amounts of costly real-world data. To bridge this gap, we propose a few-shot real-to-sim approach that combines the physical consistency of analytical formulations with the representational capacity of graph neural network (GNN)-based models. Using only a small amount of real-world data, our method calibrates analytical simulators to generate large-scale synthetic datasets that capture diverse contact interactions. On this foundation, we introduce a mesh-based GNN that implicitly models rigid-body forward dynamics and derive surrogate gradients for collision detection, achieving full differentiability. Experimental results demonstrate that our approach enables learning-based simulators to outperform differentiable baselines in replicating real-world trajectories. In addition, the differentiable design supports gradient-based optimization, which we validate through simulation-based policy learning in multi-object interaction scenarios. Extensive experiments show that our framework not only improves simulation fidelity with minimal supervision but also increases the efficiency of policy learning. Taken together, these findings suggest that differentiable simulation with few-shot real-world grounding provides a powerful direction for advancing future robotic manipulation and control.

  </details>



- **EntON: Eigenentropy-Optimized Neighborhood Densification in 3D Gaussian Splatting**  
  Miriam Jäger, Boris Jutzi  
  _2026-03-06_ · https://arxiv.org/abs/2603.06216v1  
  <details><summary>Abstract</summary>

  We present a novel Eigenentropy-optimized neighboorhood densification strategy EntON in 3D Gaussian Splatting (3DGS) for geometrically accurate and high-quality rendered 3D reconstruction. While standard 3DGS produces Gaussians whose centers and surfaces are poorly aligned with the underlying object geometry, surface-focused reconstruction methods frequently sacrifice photometric accuracy. In contrast to the conventional densification strategy, which relies on the magnitude of the view-space position gradient, our approach introduces a geometry-aware strategy to guide adaptive splitting and pruning. Specifically, we compute the 3D shape feature Eigenentropy from the eigenvalues of the covariance matrix in the k-nearest neighborhood of each Gaussian center, which quantifies the local structural order. These Eigenentropy values are integrated into an alternating optimization framework: During the optimization process, the algorithm alternates between (i) standard gradient-based densification, which refines regions via view-space gradients, and (ii) Eigenentropy-aware densification, which preferentially densifies Gaussians in low-Eigenentropy (ordered, flat) neighborhoods to better capture fine geometric details on the object surface, and prunes those in high-Eigenentropy (disordered, spherical) regions. We provide quantitative and qualitative evaluations on two benchmark datasets: small-scale DTU dataset and large-scale TUM2TWIN dataset, covering man-made objects and urban scenes. Experiments demonstrate that our Eigenentropy-aware alternating densification strategy improves geometric accuracy by up to 33% and rendering quality by up to 7%, while reducing the number of Gaussians by up to 50% and training time by up to 23%. Overall, EnTON achieves a favorable balance between geometric accuracy, rendering quality and efficiency by avoiding unnecessary scene expansion.

  </details>



- **VG3S: Visual Geometry Grounded Gaussian Splatting for Semantic Occupancy Prediction**  
  Xiaoyang Yan, Muleilan Pei, Shaojie Shen  
  _2026-03-06_ · https://arxiv.org/abs/2603.06210v1  
  <details><summary>Abstract</summary>

  3D semantic occupancy prediction has become a crucial perception task for comprehensive scene understanding in autonomous driving. While recent advances have explored 3D Gaussian splatting for occupancy modeling to substantially reduce computational overhead, the generation of high-quality 3D Gaussians relies heavily on accurate geometric cues, which are often insufficient in purely vision-centric paradigms. To bridge this gap, we advocate for injecting the strong geometric grounding capability from Vision Foundation Models (VFMs) into occupancy prediction. In this regard, we introduce Visual Geometry Grounded Gaussian Splatting (VG3S), a novel framework that empowers Gaussian-based occupancy prediction with cross-view 3D geometric grounding. Specifically, to fully exploit the rich 3D geometric priors from a frozen VFM, we propose a plug-and-play hierarchical geometric feature adapter, which can effectively transform generic VFM tokens via feature aggregation, task-specific alignment, and multi-scale restructuring. Extensive experiments on the nuScenes occupancy benchmark demonstrate that VG3S achieves remarkable improvements of 12.6% in IoU and 7.5% in mIoU over the baseline. Furthermore, we show that VG3S generalizes seamlessly across diverse VFMs, consistently enhancing occupancy prediction accuracy and firmly underscoring the immense value of integrating priors derived from powerful, pre-trained geometry-grounded VFMs.

  </details>



- **Point-Supervised Skeleton-Based Human Action Segmentation**  
  Hongsong Wang, Yiqin Shen, Pengbo Yan, Jie Gui  
  _2026-03-06_ · https://arxiv.org/abs/2603.06201v1  
  <details><summary>Abstract</summary>

  Skeleton-based temporal action segmentation is a fundamental yet challenging task, playing a crucial role in enabling intelligent systems to perceive and respond to human activities. While fully-supervised methods achieve satisfactory performance, they require costly frame-level annotations and are sensitive to ambiguous action boundaries. To address these issues, we introduce a point-supervised framework for skeleton-based action segmentation, where only a single frame per action segment is labeled. We leverage multimodal skeleton data, including joint, bone, and motion information, encoded via a pretrained unified model to extract rich feature representations. To generate reliable pseudo-labels, we propose a novel prototype similarity method and integrate it with two existing methods: energy function and constrained K-Medoids clustering. Multimodal pseudo-label integration is proposed to enhance the reliability of the pseudo-label and guide the model training. We establish new benchmarks on PKU-MMD (X-Sub and X-View), MCFS-22, and MCFS-130, and implement baselines for point-supervised skeleton-based human action segmentation. Extensive experiments show that our method achieves competitive performance, even surpassing some fully-supervised methods while significantly reducing annotation effort.

  </details>



- **Adaptive Language-Aware Image Reflection Removal Network**  
  Siyan Fang, Yuntao Wang, Jinpu Zhang, Ziwen Li, Yuehuan Wang  
  _2026-03-06_ · https://arxiv.org/abs/2603.06200v1  
  <details><summary>Abstract</summary>

  Existing image reflection removal methods struggle to handle complex reflections. Accurate language descriptions can help the model understand the image content to remove complex reflections. However, due to blurred and distorted interferences in reflected images, machine-generated language descriptions of the image content are often inaccurate, which harms the performance of language-guided reflection removal. To address this, we propose the Adaptive Language-Aware Network (ALANet) to remove reflections even with inaccurate language inputs. Specifically, ALANet integrates both filtering and optimization strategies. The filtering strategy reduces the negative effects of language while preserving its benefits, whereas the optimization strategy enhances the alignment between language and visual features. ALANet also utilizes language cues to decouple specific layer content from feature maps, improving its ability to handle complex reflections. To evaluate the model's performance under complex reflections and varying levels of language accuracy, we introduce the Complex Reflection and Language Accuracy Variance (CRLAV) dataset. Experimental results demonstrate that ALANet surpasses state-of-the-art methods for image reflection removal. The code and dataset are available at https://github.com/fashyon/ALANet.

  </details>



- **SpaCRD: Multimodal Deep Fusion of Histology and Spatial Transcriptomics for Cancer Region Detection**  
  Shuailin Xue, Jun Wan, Lihua Zhang, Wenwen Min  
  _2026-03-06_ · https://arxiv.org/abs/2603.06186v1  
  <details><summary>Abstract</summary>

  Accurate detection of cancer tissue regions (CTR) enables deeper analysis of the tumor microenvironment and offers crucial insights into treatment response. Traditional CTR detection methods, which typically rely on the rich cellular morphology in histology images, are susceptible to a high rate of false positives due to morphological similarities across different tissue regions. The groundbreaking advances in spatial transcriptomics (ST) provide detailed cellular phenotypes and spatial localization information, offering new opportunities for more accurate cancer region detection. However, current methods are unable to effectively integrate histology images with ST data, especially in the context of cross-sample and cross-platform/batch settings for accomplishing the CTR detection. To address this challenge, we propose SpaCRD, a transfer learning-based method that deeply integrates histology images and ST data to enable reliable CTR detection across diverse samples, platforms, and batches. Once trained on source data, SpaCRD can be readily generalized to accurately detect cancerous regions across samples from different platforms and batches. The core of SpaCRD is a category-regularized variational reconstruction-guided bidirectional cross-attention fusion network, which enables the model to adaptively capture latent co-expression patterns between histological features and gene expression from multiple perspectives. Extensive benchmark analysis on 23 matched histology-ST datasets spanning various disease types, platforms, and batches demonstrates that SpaCRD consistently outperforms existing eight state-of-the-art methods in CTR detection.

  </details>



- **CRIMSON: A Clinically-Grounded LLM-Based Metric for Generative Radiology Report Evaluation**  
  Mohammed Baharoon, Thibault Heintz, Siavash Raissi, Mahmoud Alabbad, Mona Alhammad, Hassan AlOmaish, Sung Eun Kim, Oishi Banerjee, Pranav Rajpurkar  
  _2026-03-06_ · https://arxiv.org/abs/2603.06183v1  
  <details><summary>Abstract</summary>

  We introduce CRIMSON, a clinically grounded evaluation framework for chest X-ray report generation that assesses reports based on diagnostic correctness, contextual relevance, and patient safety. Unlike prior metrics, CRIMSON incorporates full clinical context, including patient age, indication, and guideline-based decision rules, and prevents normal or clinically insignificant findings from exerting disproportionate influence on the overall score. The framework categorizes errors into a comprehensive taxonomy covering false findings, missing findings, and eight attribute-level errors (e.g., location, severity, measurement, and diagnostic overinterpretation). Each finding is assigned a clinical significance level (urgent, actionable non-urgent, non-actionable, or expected/benign), based on a guideline developed in collaboration with attending cardiothoracic radiologists, enabling severity-aware weighting that prioritizes clinically consequential mistakes over benign discrepancies. CRIMSON is validated through strong alignment with clinically significant error counts annotated by six board-certified radiologists in ReXVal (Kendalls tau = 0.61-0.71; Pearsons r = 0.71-0.84), and through two additional benchmarks that we introduce. In RadJudge, a targeted suite of clinically challenging pass-fail scenarios, CRIMSON shows consistent agreement with expert judgment. In RadPref, a larger radiologist preference benchmark of over 100 pairwise cases with structured error categorization, severity modeling, and 1-5 overall quality ratings from three cardiothoracic radiologists, CRIMSON achieves the strongest alignment with radiologist preferences. We release the metric, the evaluation benchmarks, RadJudge and RadPref, and a fine-tuned MedGemma model to enable reproducible evaluation of report generation, all available at https://github.com/rajpurkarlab/CRIMSON.

  </details>



- **Towards Motion Turing Test: Evaluating Human-Likeness in Humanoid Robots**  
  Mingzhe Li, Mengyin Liu, Zekai Wu, Xincheng Lin, Junsheng Zhang, Ming Yan, Zengye Xie, Changwang Zhang, Chenglu Wen, Lan Xu, et al.  
  _2026-03-06_ · https://arxiv.org/abs/2603.06181v1  
  <details><summary>Abstract</summary>

  Humanoid robots have achieved significant progress in motion generation and control, exhibiting movements that appear increasingly natural and human-like. Inspired by the Turing Test, we propose the Motion Turing Test, a framework that evaluates whether human observers can discriminate between humanoid robot and human poses using only kinematic information. To facilitate this evaluation, we present the Human-Humanoid Motion (HHMotion) dataset, which consists of 1,000 motion sequences spanning 15 action categories, performed by 11 humanoid models and 10 human subjects. All motion sequences are converted into SMPL-X representations to eliminate the influence of visual appearance. We recruited 30 annotators to rate the human-likeness of each pose on a 0-5 scale, resulting in over 500 hours of annotation. Analysis of the collected data reveals that humanoid motions still exhibit noticeable deviations from human movements, particularly in dynamic actions such as jumping, boxing, and running. Building on HHMotion, we formulate a human-likeness evaluation task that aims to automatically predict human-likeness scores from motion data. Despite recent progress in multimodal large language models, we find that they remain inadequate for assessing motion human-likeness. To address this, we propose a simple baseline model and demonstrate that it outperforms several contemporary LLM-based methods. The dataset, code, and benchmark will be publicly released to support future research in the community.

  </details>



- **VLM-RobustBench: A Comprehensive Benchmark for Robustness of Vision-Language Models**  
  Rohit Saxena, Alessandro Suglia, Pasquale Minervini  
  _2026-03-06_ · https://arxiv.org/abs/2603.06148v1  
  <details><summary>Abstract</summary>

  Vision-language models (VLMs) achieve strong performance on standard, high-quality datasets, but we still do not fully understand how they perform under real-world image distortions. We present VLM-RobustBench, a benchmark spanning 49 augmentation types across noise, blur, weather, digital, and geometric perturbations, evaluated under graded severities (low/mid/high) and binary transforms, yielding 133 corrupted settings. We evaluate VLMs from four families (Qwen, InternVL, Molmo, Gemma) on two complementary benchmarks: MMBench (visually grounded) and MMMU-Pro (reasoning-oriented). Our results reveal that visual severity is a weak predictor of difficulty: low-severity spatial perturbations often degrade performance more than visually severe photometric corruptions. In particular, low-severity glass_blur reduces MMBench accuracy by about 8 pp on average across models, while the largest drops arise from resampling and geometric distortions (e.g., upsample, elastic_transform), reaching up to 34 pp. Overall, our findings suggest current VLMs are semantically strong but spatially fragile, motivating the definition of novel robustness evaluation protocols and training regimes that emphasize resampling and geometric invariances.

  </details>



- **Longitudinal NSCLC Treatment Progression via Multimodal Generative Models**  
  Massimiliano Mantegna, Elena Mulero Ayllón, Alice Natalina Caragliano, Francesco Di Feola, Claudia Tacconi, Michele Fiore, Edy Ippolito, Carlo Greco, Sara Ramella, Philippe C. Cattin, et al.  
  _2026-03-06_ · https://arxiv.org/abs/2603.06147v1  
  <details><summary>Abstract</summary>

  Predicting tumor evolution during radiotherapy is a clinically critical challenge, particularly when longitudinal changes are driven by both anatomy and treatment. In this work, we introduce a Virtual Treatment (VT) framework that formulates non-small cell lung cancer (NSCLC) progression as a dose-aware multimodal conditional image-to-image translation problem. Given a CT scan, baseline clinical variables, and a specified radiation dose increment, VT aims to synthesize plausible follow-up CT images reflecting treatment-induced anatomical changes. We evaluate the proposed framework on a longitudinal dataset of 222 stage III NSCLC patients, comprising 895 CT scans acquired during radiotherapy under irregular clinical schedules. The generative process is conditioned on delivered dose increments together with demographic and tumor-related clinical variables. Representative GAN-based and diffusion-based models are benchmarked across 2D and 2.5D configurations. Quantitative and qualitative results indicate that diffusion-based models benefit more consistently from multimodal, dose-aware conditioning and produce more stable and anatomically plausible tumor evolution trajectories than GAN-based baselines, supporting the potential of VT as a tool for in-silico treatment monitoring and adaptive radiotherapy research in NSCLC.

  </details>



- **Spatial Colour Mixing Illusions as a Perception Stress Test for Vision-Language Models**  
  Nicoleta-Nina Basoc, Adrian Cosma, Emilian Radoi  
  _2026-03-06_ · https://arxiv.org/abs/2603.06141v1  
  <details><summary>Abstract</summary>

  Vision-language models (VLMs) achieve strong benchmark results, yet can exhibit systematic perceptual weaknesses: structured, large changes to pixel values can cause confident yet nonsensical predictions, even when the underlying scene remains easily recognizable to humans. We study this gap using Spatial Colour Mixing, a programmatic family of colour distortions that overlays structured patterns (in both RGB and Ostwald colour systems) onto natural images. We introduce a framework of eight spatial colour mixing variants and evaluate nine VLMs across three model families on four datasets. Across models and datasets, accuracy degrades sharply with increasing distortion, and scaling the language model does not reliably mitigate the failure. In a human study with 61 participants on an animal recognition dataset, humans substantially outperform VLMs under the same distortions. Finally, we show that a simple human-inspired preprocessing step recovers a meaningful portion of performance for several distortion types, motivating perception-aware preprocessing and tool-use as practical strategies for improving VLM robustness.

  </details>



- **DeepSight: Bridging Depth Maps and Language with a Depth-Driven Multimodal Model**  
  Hao Yang, Hongbo Zhang, Yanyan Zhao, Bing Qin  
  _2026-03-06_ · https://arxiv.org/abs/2603.06090v1  
  <details><summary>Abstract</summary>

  Multimodal large language models (MLLMs) have achieved impressive performance across various tasks such as image captioning and visual question answer(VQA); however, they often struggle to accurately interpret depth information inherent in visual data. In this work, we introduce DeepSight, the first dedicated depth MLLM designed to enhance three-dimensional scene understanding. Unlike conventional methods that align RGB image encodings with text, our approach takes advantage of the unique characteristics of depth images: single-channel grayscale images where the pixel values directly reflect depth cues to improve spatial reasoning. To address challenges associated with limited depth data and the inadequacy of simple channel replication, we construct a novel depth image-text pair dataset and a depth instruction dataset. Depth maps are generated from visual images using the GLPN model, and GPT-4 is employed to curate corresponding depth instructions, an approach validated by LLaVA. Additionally, we modify the ViT encoder in CLIP to incorporate local object information, thereby capturing the subtle continuous variations of depth more effectively. To evaluate the performance of our model, we develop a comprehensive depth question answer benchmark based on existing depth image datasets, which rigorously assesses understanding in typical depth map scenarios. Experimental results demonstrate that DeepSight significantly enhances depth perception and downstream task performance, marking a substantial step forward in multimodal three-dimensional understanding.

  </details>



- **Multimodal Behavior Tree Generation: A Small Vision-Language Model for Robot Task Planning**  
  Cristiano Battistini, Riccardo Andrea Izzo, Gianluca Bardaro, Matteo Matteucci  
  _2026-03-06_ · https://arxiv.org/abs/2603.06084v1  
  <details><summary>Abstract</summary>

  Large and small language models have been widely used for robotic task planning. At the same time, vision-language models (VLMs) have successfully tackled problems such as image captioning, scene understanding, and visual question answering. In this work, we combine these two approaches by deploying a compact, open-source multimodal model to generate behavior trees for robotic task planning. The main obstacle to achieving this goal is the lack of an existing dataset that links visual observations and instructions to executable behavior trees. We propose a method to construct such a dataset starting from existing robotic episodes (i.e., Open X-Embodiment), in which a large model serves as a teacher in a multi-stage generation pipeline. We use this dataset to fine-tune VLMs ranging from 500M to 4B parameters via parameter-efficient fine-tuning (PEFT). The generated behavior trees, compatible with the BehaviorTree.CPP library, are evaluated both offline, using structural and lexical metrics, and online through the execution of household tasks in a state-of-the-art embodied simulator. Our results demonstrate that our fine-tuned 4B-parameter VLM approaches the performance of state-of-the-art closed-source models, achieving an 87\% success rate while requiring only a fraction of the computational resources.

  </details>



- **TempoSyncDiff: Distilled Temporally-Consistent Diffusion for Low-Latency Audio-Driven Talking Head Generation**  
  Soumya Mazumdar, Vineet Kumar Rakesh  
  _2026-03-06_ · https://arxiv.org/abs/2603.06057v1  
  <details><summary>Abstract</summary>

  Diffusion models have recently advanced photorealistic human synthesis, although practical talking-head generation (THG) remains constrained by high inference latency, temporal instability such as flicker and identity drift, and imperfect audio-visual alignment under challenging speech conditions. This paper introduces TempoSyncDiff, a reference-conditioned latent diffusion framework that explores few-step inference for efficient audio-driven talking-head generation. The approach adopts a teacher-student distillation formulation in which a diffusion teacher trained with a standard noise prediction objective guides a lightweight student denoiser capable of operating with significantly fewer inference steps to improve generation stability. The framework incorporates identity anchoring and temporal regularization designed to mitigate identity drift and frame-to-frame flicker during synthesis, while viseme-based audio conditioning provides coarse lip motion control. Experiments on the LRS3 dataset report denoising-stage component-level metrics relative to VAE reconstructions and preliminary latency characterization, including CPU-only and edge computing measurements and feasibility estimates for edge deployment. The results suggest that distilled diffusion models can retain much of the reconstruction behaviour of a stronger teacher while enabling substantially lower latency inference. The study is positioned as an initial step toward practical diffusion-based talking-head generation under constrained computational settings. GitHub: https://mazumdarsoumya.github.io/TempoSyncDiff

  </details>



- **Devil is in Narrow Policy: Unleashing Exploration in Driving VLA Models**  
  Canyu Chen, Yuguang Yang, Zhewen Tan, Yizhi Wang, Ruiyi Zhan, Haiyan Liu, Xuanyao Mao, Jason Bao, Xinyue Tang, Linlin Yang, et al.  
  _2026-03-06_ · https://arxiv.org/abs/2603.06049v1  
  <details><summary>Abstract</summary>

  We identify a fundamental Narrow Policy limitation undermining the performance of autonomous VLA models, where driving Imitation Learning (IL) tends to collapse exploration and limit the potential of subsequent Reinforcement Learning (RL) stages, which often saturate prematurely due to insufficient feedback diversity. Thereby, we propose Curious-VLA, a framework that alleviates the exploit-explore dilemma through a two-stage design. During IL, we introduce a Feasible Trajectory Expansion (FTE) strategy to generate multiple physically valid trajectories and a step-wise normalized trajectory representation to adapt this diverse data. In the RL stage, we present Adaptive Diversity-Aware Sampling (ADAS) that prioritizes high-diversity samples and introduce Spanning Driving Reward (SDR) with a focal style weighting to amplify reward's value span for improving sensitivity to driving quality. On the Navsim benchmark, Curious-VLA achieves SoTA results (PDMS 90.3, EPDMS 85.4) and a Best-of-N PDMS of 94.8, demonstrating its effectiveness in unlocking the exploratory potential of VLA models. Code: https://github.com/Mashiroln/curious_vla.git.

  </details>



- **FontUse: A Data-Centric Approach to Style- and Use-Case-Conditioned In-Image Typography**  
  Xia Xin, Yuki Endo, Yoshihiro Kanamori  
  _2026-03-06_ · https://arxiv.org/abs/2603.06038v1  
  <details><summary>Abstract</summary>

  Recent text-to-image models can generate high-quality images from natural-language prompts, yet controlling typography remains challenging: requested typographic appearance is often ignored or only weakly followed. We address this limitation with a data-centric approach that trains image generation models using targeted supervision derived from a structured annotation pipeline specialized for typography. Our pipeline constructs a large-scale typography-focused dataset, FontUse, consisting of about 70K images annotated with user-friendly prompts, text-region locations, and OCR-recognized strings. The annotations are automatically produced using segmentation models and multimodal large language models (MLLMs). The prompts explicitly combine font styles (e.g., serif, script, elegant) and use cases (e.g., wedding invitations, coffee-shop menus), enabling intuitive specification even for novice users. Fine-tuning existing generators with these annotations allows them to consistently interpret style and use-case conditions as textual prompts without architectural modification. For evaluation, we introduce a Long-CLIP-based metric that measures alignment between generated typography and requested attributes. Experiments across diverse prompts and layouts show that models trained with our pipeline produce text renderings more consistent with prompts than competitive baselines. The source code for our annotation pipeline is available at https://github.com/xiaxinz/FontUSE.

  </details>



- **Ensemble Learning with Sparse Hypercolumns**  
  Julia Dietlmeier, Vayangi Ganepola, Oluwabukola G. Adegboro, Mayug Maniparambil, Claudia Mazo, Noel E. O'Connor  
  _2026-03-06_ · https://arxiv.org/abs/2603.06036v1  
  <details><summary>Abstract</summary>

  Directly inspired by findings in biological vision, high-dimensional hypercolumns are feature vectors built by concatenating multi-scale activations of convolutional neural networks for a single image pixel location. Together with powerful classifiers, they can be used for image segmentation i.e. pixel classification. However, in practice, there are only very few works dedicated to the use of hypercolumns. One reason is the computational complexity of processing concatenated dense hypercolumns that grows linearly with the size $N$ of the training set. In this work, we address this challenge by applying stratified subsampling to the VGG16 based hypercolumns. Furthermore, we investigate the performance of ensemble learning on sparse hypercolumns. Our experiments on a brain tumor dataset show that stacking and voting ensembles deliver competitive performance, but in the extreme low-shot case of $N \leq 20$, a simple Logistic Regression classifier is the most effective method. For 10% stratified subsampling rate, our best average Dice score is 0.66 for $N=20$. This is a statistically significant improvement of 24.53% over the standard multi-scale UNet baseline ($p$-value = $[3.07e-11]$, Wilcoxon signed-rank test), which is less effective due to overfitting.

  </details>



- **MOSIV: Multi-Object System Identification from Videos**  
  Chunjiang Liu, Xiaoyuan Wang, Qingran Lin, Albert Xiao, Haoyu Chen, Shizheng Wen, Hao Zhang, Lu Qi, Ming-Hsuan Yang, Laszlo A. Jeni, et al.  
  _2026-03-06_ · https://arxiv.org/abs/2603.06022v1  
  <details><summary>Abstract</summary>

  We introduce the challenging problem of multi-object system identification from videos, for which prior methods are ill-suited due to their focus on single-object scenes or discrete material classification with a fixed set of material prototypes. To address this, we propose MOSIV, a new framework that directly optimizes for continuous, per-object material parameters using a differentiable simulator guided by geometric objectives derived from video. We also present a new synthetic benchmark with contact-rich, multi-object interactions to facilitate evaluation. On this benchmark, MOSIV substantially improves grounding accuracy and long-horizon simulation fidelity over adapted baselines, establishing it as a strong baseline for this new task. Our analysis shows that object-level fine-grained supervision and geometry-aligned objectives are critical for stable optimization in these complex, multi-object settings. The source code and dataset will be released.

  </details>



- **EffectMaker: Unifying Reasoning and Generation for Customized Visual Effect Creation**  
  Shiyuan Yang, Ruihuang Li, Jiale Tao, Shuai Shao, Qinglin Lu, Jing Liao  
  _2026-03-06_ · https://arxiv.org/abs/2603.06014v1  
  <details><summary>Abstract</summary>

  Visual effects (VFX) are essential for enhancing the expressiveness and creativity of video content, yet producing high-quality effects typically requires expert knowledge and costly production pipelines. Existing AIGC systems face significant challenges in VFX generation due to the scarcity of effect-specific data and the inherent difficulty of modeling supernatural or stylized effects. Moreover, these approaches often require per-effect fine-tuning, which severely limits their scalability and generalization to novel VFX. In this work, we present EffectMaker, a unified reasoning-generation framework that enables reference-based VFX customization. EffectMaker employs a multimodal large language model to interpret high-level effect semantics and reason about how they should adapt to a target subject, while a diffusion transformer leverages in-context learning to capture fine-grained visual cues from reference videos. These two components form a semantic-visual dual-path guidance mechanism that enables accurate, controllable, and effect-consistent synthesis without per-effect fine-tuning. Furthermore, we construct EffectData, the largest high-quality synthetic dataset containing 130k videos across 3k VFX categories, to improve generalization and scalability. Experiments show that EffectMaker achieves superior visual quality and effect consistency over state-of-the-art baselines, offering a scalable and flexible paradigm for customized VFX generation. Project page: https://effectmaker.github.io

  </details>



- **Restoring Linguistic Grounding in VLA Models via Train-Free Attention Recalibration**  
  Ninghao Zhang, Bin Zhu, Shijie Zhou, Jingjing Chen  
  _2026-03-06_ · https://arxiv.org/abs/2603.06001v1  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models enable robots to perform manipulation tasks directly from natural language instructions and are increasingly viewed as a foundation for generalist robotic policies. However, their reliability under Out-of-Distribution (OOD) instructions remains underexplored. In this paper, we reveal a critical failure mode in which VLA policies continue executing visually plausible actions even when the language instruction contradicts the scene. We refer to this phenomenon as linguistic blindness, where VLA policies prioritize visual priors over instruction semantics during action generation. To systematically analyze this issue, we introduce ICBench, a diagnostic benchmark constructed from the LIBERO dataset that probes language-action coupling by injecting controlled OOD instruction contradictions while keeping the visual environment unchanged. Evaluations on three representative VLA architectures, including Pi0, Pi0.5 and OpenVLA OFT, show that these models frequently succeed at tasks despite logically impossible instructions, revealing a strong visual bias in action generation. To mitigate this issue, we propose Instruction-Guided Attention Recalibration (IGAR), a train-free inference-time mechanism that rebalances attention distributions to restore the influence of language instructions. IGAR operates without retraining or architectural modification and can be directly applied to existing VLA models. Experiments across 30 LIBERO tasks demonstrate that IGAR substantially reduces erroneous execution under OOD contradictory instructions while preserving baseline task performance. We additionally validate the approach on a real Franka robotic arm, where IGAR effectively prevents manipulation triggered by inconsistent instructions.

  </details>



- **Moving Through Clutter: Scaling Data Collection and Benchmarking for 3D Scene-Aware Humanoid Locomotion via Virtual Reality**  
  Beichen Wang, Yuanjie Lu, Linji Wang, Liuchuan Yu, Xuesu Xiao  
  _2026-03-06_ · https://arxiv.org/abs/2603.05993v1  
  <details><summary>Abstract</summary>

  Recent advances in humanoid locomotion have enabled dynamic behaviors such as dancing, martial arts, and parkour, yet these capabilities are predominantly demonstrated in open, flat, and obstacle-free settings. In contrast, real-world environments such as homes, offices, and public spaces, are densely cluttered, three-dimensional, and geometrically constrained, requiring scene-aware whole-body coordination, precise balance control, and reasoning over spatial constraints imposed by furniture and household objects. However, humanoid locomotion in cluttered 3D environments remains underexplored, and no public dataset systematically couples full-body human locomotion with the scene geometry that shapes it. To address this gap, we present Moving Through Clutter (MTC), an opensource Virtual Reality (VR) based data collection and evaluation framework for scene-aware humanoid locomotion in cluttered environments. Our system procedurally generates scenes with controllable clutter levels and captures embodiment-consistent, whole-body human motion through immersive VR navigation, which is then automatically retargeted to a humanoid robot model. We further introduce benchmarks that quantify environment clutter level and locomotion performance, including stability and collision safety. Using this framework, we compile a dataset of 348 trajectories across 145 diverse 3D cluttered scenes. The dataset provides a foundation for studying geometry-induced adaptation in humanoid locomotion and developing scene-aware planning and control methods.

  </details>


