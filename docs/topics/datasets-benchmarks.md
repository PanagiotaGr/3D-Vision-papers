# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-01-19 06:56 UTC_

Total papers shown: **25**


---

- **ShapeR: Robust Conditional 3D Shape Generation from Casual Captures**  
  Yawar Siddiqui, Duncan Frost, Samir Aroudj, Armen Avetisyan, Henry Howard-Jenkins, Daniel DeTone, Pierre Moulon, Qirui Wu, Zhengqin Li, Julian Straub, et al.  
  _2026-01-16_ · https://arxiv.org/abs/2601.11514v1  
  <details><summary>Abstract</summary>

  Recent advances in 3D shape generation have achieved impressive results, but most existing methods rely on clean, unoccluded, and well-segmented inputs. Such conditions are rarely met in real-world scenarios. We present ShapeR, a novel approach for conditional 3D object shape generation from casually captured sequences. Given an image sequence, we leverage off-the-shelf visual-inertial SLAM, 3D detection algorithms, and vision-language models to extract, for each object, a set of sparse SLAM points, posed multi-view images, and machine-generated captions. A rectified flow transformer trained to effectively condition on these modalities then generates high-fidelity metric 3D shapes. To ensure robustness to the challenges of casually captured data, we employ a range of techniques including on-the-fly compositional augmentations, a curriculum training scheme spanning object- and scene-level datasets, and strategies to handle background clutter. Additionally, we introduce a new evaluation benchmark comprising 178 in-the-wild objects across 7 real-world scenes with geometry annotations. Experiments show that ShapeR significantly outperforms existing approaches in this challenging setting, achieving an improvement of 2.7x in Chamfer distance compared to state of the art.

  </details>



- **ReScene4D: Temporally Consistent Semantic Instance Segmentation of Evolving Indoor 3D Scenes**  
  Emily Steiner, Jianhao Zheng, Henry Howard-Jenkins, Chris Xie, Iro Armeni  
  _2026-01-16_ · https://arxiv.org/abs/2601.11508v1  
  <details><summary>Abstract</summary>

  Indoor environments evolve as objects move, appear, or disappear. Capturing these dynamics requires maintaining temporally consistent instance identities across intermittently captured 3D scans, even when changes are unobserved. We introduce and formalize the task of temporally sparse 4D indoor semantic instance segmentation (SIS), which jointly segments, identifies, and temporally associates object instances. This setting poses a challenge for existing 3DSIS methods, which require a discrete matching step due to their lack of temporal reasoning, and for 4D LiDAR approaches, which perform poorly due to their reliance on high-frequency temporal measurements that are uncommon in the longer-horizon evolution of indoor environments. We propose ReScene4D, a novel method that adapts 3DSIS architectures for 4DSIS without needing dense observations. It explores strategies to share information across observations, demonstrating that this shared context not only enables consistent instance tracking but also improves standard 3DSIS quality. To evaluate this task, we define a new metric, t-mAP, that extends mAP to reward temporal identity consistency. ReScene4D achieves state-of-the-art performance on the 3RScan dataset, establishing a new benchmark for understanding evolving indoor scenes.

  </details>



- **Map2Thought: Explicit 3D Spatial Reasoning via Metric Cognitive Maps**  
  Xiangjun Gao, Zhensong Zhang, Dave Zhenyu Chen, Songcen Xu, Long Quan, Eduardo Pérez-Pellitero, Youngkyoon Jang  
  _2026-01-16_ · https://arxiv.org/abs/2601.11442v1  
  <details><summary>Abstract</summary>

  We propose Map2Thought, a framework that enables explicit and interpretable spatial reasoning for 3D VLMs. The framework is grounded in two key components: Metric Cognitive Map (Metric-CogMap) and Cognitive Chain-of-Thought (Cog-CoT). Metric-CogMap provides a unified spatial representation by integrating a discrete grid for relational reasoning with a continuous, metric-scale representation for precise geometric understanding. Building upon the Metric-CogMap, Cog-CoT performs explicit geometric reasoning through deterministic operations, including vector operations, bounding-box distances, and occlusion-aware appearance order cues, producing interpretable inference traces grounded in 3D structure. Experimental results show that Map2Thought enables explainable 3D understanding, achieving 59.9% accuracy using only half the supervision, closely matching the 60.9% baseline trained with the full dataset. It consistently outperforms state-of-the-art methods by 5.3%, 4.8%, and 4.0% under 10%, 25%, and 50% training subsets, respectively, on the VSI-Bench.

  </details>



- **The Great March 100: 100 Detail-oriented Tasks for Evaluating Embodied AI Agents**  
  Ziyu Wang, Chenyuan Liu, Yushun Xiang, Runhao Zhang, Qingbo Hao, Hongliang Lu, Houyu Chen, Zhizhong Feng, Kaiyue Zheng, Dehao Ye, et al.  
  _2026-01-16_ · https://arxiv.org/abs/2601.11421v1  
  <details><summary>Abstract</summary>

  Recently, with the rapid development of robot learning and imitation learning, numerous datasets and methods have emerged. However, these datasets and their task designs often lack systematic consideration and principles. This raises important questions: Do the current datasets and task designs truly advance the capabilities of robotic agents? Do evaluations on a few common tasks accurately reflect the differentiated performance of various methods proposed by different teams and evaluated on different tasks? To address these issues, we introduce the Great March 100 (\textbf{GM-100}) as the first step towards a robot learning Olympics. GM-100 consists of 100 carefully designed tasks that cover a wide range of interactions and long-tail behaviors, aiming to provide a diverse and challenging set of tasks to comprehensively evaluate the capabilities of robotic agents and promote diversity and complexity in robot dataset task designs. These tasks are developed through systematic analysis and expansion of existing task designs, combined with insights from human-object interaction primitives and object affordances. We collect a large amount of trajectory data on different robotic platforms and evaluate several baseline models. Experimental results demonstrate that the GM-100 tasks are 1) feasible to execute and 2) sufficiently challenging to effectively differentiate the performance of current VLA models. Our data and code are available at https://rhos.ai/research/gm-100.

  </details>



- **SME-YOLO: A Real-Time Detector for Tiny Defect Detection on PCB Surfaces**  
  Meng Han  
  _2026-01-16_ · https://arxiv.org/abs/2601.11402v1  
  <details><summary>Abstract</summary>

  Surface defects on Printed Circuit Boards (PCBs) directly compromise product reliability and safety. However, achieving high-precision detection is challenging because PCB defects are typically characterized by tiny sizes, high texture similarity, and uneven scale distributions. To address these challenges, this paper proposes a novel framework based on YOLOv11n, named SME-YOLO (Small-target Multi-scale Enhanced YOLO). First, we employ the Normalized Wasserstein Distance Loss (NWDLoss). This metric effectively mitigates the sensitivity of Intersection over Union (IoU) to positional deviations in tiny objects. Second, the original upsampling module is replaced by the Efficient Upsampling Convolution Block (EUCB). By utilizing multi-scale convolutions, the EUCB gradually recovers spatial resolution and enhances the preservation of edge and texture details for tiny defects. Finally, this paper proposes the Multi-Scale Focused Attention (MSFA) module. Tailored to the specific spatial distribution of PCB defects, this module adaptively strengthens perception within key scale intervals, achieving efficient fusion of local fine-grained features and global context information. Experimental results on the PKU-PCB dataset demonstrate that SME-YOLO achieves state-of-the-art performance. Specifically, compared to the baseline YOLOv11n, SME-YOLO improves mAP by 2.2% and Precision by 4%, validating the effectiveness of the proposed method.

  </details>



- **Wetland mapping from sparse annotations with satellite image time series and temporal-aware segment anything model**  
  Shuai Yuan, Tianwu Lin, Shuang Chen, Yu Xia, Peng Qin, Xiangyu Liu, Xiaoqing Xu, Nan Xu, Hongsheng Zhang, Jie Wang, et al.  
  _2026-01-16_ · https://arxiv.org/abs/2601.11400v1  
  <details><summary>Abstract</summary>

  Accurate wetland mapping is essential for ecosystem monitoring, yet dense pixel-level annotation is prohibitively expensive and practical applications usually rely on sparse point labels, under which existing deep learning models perform poorly, while strong seasonal and inter-annual wetland dynamics further render single-date imagery inadequate and lead to significant mapping errors; although foundation models such as SAM show promising generalization from point prompts, they are inherently designed for static images and fail to model temporal information, resulting in fragmented masks in heterogeneous wetlands. To overcome these limitations, we propose WetSAM, a SAM-based framework that integrates satellite image time series for wetland mapping from sparse point supervision through a dual-branch design, where a temporally prompted branch extends SAM with hierarchical adapters and dynamic temporal aggregation to disentangle wetland characteristics from phenological variability, and a spatial branch employs a temporally constrained region-growing strategy to generate reliable dense pseudo-labels, while a bidirectional consistency regularization jointly optimizes both branches. Extensive experiments across eight global regions of approximately 5,000 km2 each demonstrate that WetSAM substantially outperforms state-of-the-art methods, achieving an average F1-score of 85.58%, and delivering accurate and structurally consistent wetland segmentation with minimal labeling effort, highlighting its strong generalization capability and potential for scalable, low-cost, high-resolution wetland mapping.

  </details>



- **SUG-Occ: An Explicit Semantics and Uncertainty Guided Sparse Learning Framework for Real-Time 3D Occupancy Prediction**  
  Hanlin Wu, Pengfei Lin, Ehsan Javanmardi, Nanren Bao, Bo Qian, Hao Si, Manabu Tsukada  
  _2026-01-16_ · https://arxiv.org/abs/2601.11396v1  
  <details><summary>Abstract</summary>

  As autonomous driving moves toward full scene understanding, 3D semantic occupancy prediction has emerged as a crucial perception task, offering voxel-level semantics beyond traditional detection and segmentation paradigms. However, such a refined representation for scene understanding incurs prohibitive computation and memory overhead, posing a major barrier to practical real-time deployment. To address this, we propose SUG-Occ, an explicit Semantics and Uncertainty Guided Sparse Learning Enabled 3D Occupancy Prediction Framework, which exploits the inherent sparsity of 3D scenes to reduce redundant computation while maintaining geometric and semantic completeness. Specifically, we first utilize semantic and uncertainty priors to suppress projections from free space during view transformation while employing an explicit unsigned distance encoding to enhance geometric consistency, producing a structurally consistent sparse 3D representation. Secondly, we design an cascade sparse completion module via hyper cross sparse convolution and generative upsampling to enable efficiently coarse-to-fine reasoning. Finally, we devise an object contextual representation (OCR) based mask decoder that aggregates global semantic context from sparse features and refines voxel-wise predictions via lightweight query-context interactions, avoiding expensive attention operations over volumetric features. Extensive experiments on SemanticKITTI benchmark demonstrate that the proposed approach outperforms the baselines, achieving a 7.34/% improvement in accuracy and a 57.8\% gain in efficiency.

  </details>



- **The Mini Wheelbot Dataset: High-Fidelity Data for Robot Learning**  
  Henrik Hose, Paul Brunzema, Devdutt Subhasish, Sebastian Trimpe  
  _2026-01-16_ · https://arxiv.org/abs/2601.11394v1  
  <details><summary>Abstract</summary>

  The development of robust learning-based control algorithms for unstable systems requires high-quality, real-world data, yet access to specialized robotic hardware remains a significant barrier for many researchers. This paper introduces a comprehensive dynamics dataset for the Mini Wheelbot, an open-source, quasi-symmetric balancing reaction wheel unicycle. The dataset provides 1 kHz synchronized data encompassing all onboard sensor readings, state estimates, ground-truth poses from a motion capture system, and third-person video logs. To ensure data diversity, we include experiments across multiple hardware instances and surfaces using various control paradigms, including pseudo-random binary excitation, nonlinear model predictive control, and reinforcement learning agents. We include several example applications in dynamics model learning, state estimation, and time-series classification to illustrate common robotics algorithms that can be benchmarked on our dataset.

  </details>



- **Context-Aware Semantic Segmentation via Stage-Wise Attention**  
  Antoine Carreaud, Elias Naha, Arthur Chansel, Nina Lahellec, Jan Skaloud, Adrien Gressin  
  _2026-01-16_ · https://arxiv.org/abs/2601.11310v1  
  <details><summary>Abstract</summary>

  Semantic ultra high resolution image (UHR) segmentation is essential in remote sensing applications such as aerial mapping and environmental monitoring. Transformer-based models struggle in this setting because memory grows quadratically with token count, constraining either the contextual scope or the spatial resolution. We introduce CASWiT (Context-Aware Stage-Wise Transformer), a dual-branch, Swin-based architecture that injects global cues into fine-grained UHR features. A context encoder processes a downsampled neighborhood to capture long-range dependencies, while a high resolution encoder extracts detailed features from UHR patches. A cross-scale fusion module, combining cross-attention and gated feature injection, enriches high-resolution tokens with context. Beyond architecture, we propose a SimMIM-style pretraining. We mask 75% of the high-resolution image tokens and the low-resolution center region that spatially corresponds to the UHR patch, then train the shared dual-encoder with small decoder to reconstruct the UHR initial image. Extensive experiments on the large-scale IGN FLAIR-HUB aerial dataset demonstrate the effectiveness of CASWiT. Our method achieves 65.83% mIoU, outperforming RGB baselines by 1.78 points. On URUR, CASWiT achieves 49.1% mIoU, surpassing the current SoTA by +0.9% under the official evaluation protocol. All codes are provided on: https://huggingface.co/collections/heig-vd-geo/caswit.

  </details>



- **SAMannot: A Memory-Efficient, Local, Open-source Framework for Interactive Video Instance Segmentation based on SAM2**  
  Gergely Dinya, András Gelencsér, Krisztina Kupán, Clemens Küpper, Kristóf Karacs, Anna Gelencsér-Horváth  
  _2026-01-16_ · https://arxiv.org/abs/2601.11301v1  
  <details><summary>Abstract</summary>

  Current research workflows for precise video segmentation are often forced into a compromise between labor-intensive manual curation, costly commercial platforms, and/or privacy-compromising cloud-based services. The demand for high-fidelity video instance segmentation in research is often hindered by the bottleneck of manual annotation and the privacy concerns of cloud-based tools. We present SAMannot, an open-source, local framework that integrates the Segment Anything Model 2 (SAM2) into a human-in-the-loop workflow. To address the high resource requirements of foundation models, we modified the SAM2 dependency and implemented a processing layer that minimizes computational overhead and maximizes throughput, ensuring a highly responsive user interface. Key features include persistent instance identity management, an automated ``lock-and-refine'' workflow with barrier frames, and a mask-skeletonization-based auto-prompting mechanism. SAMannot facilitates the generation of research-ready datasets in YOLO and PNG formats alongside structured interaction logs. Verified through animal behavior tracking use-cases and subsets of the LVOS and DAVIS benchmark datasets, the tool provides a scalable, private, and cost-effective alternative to commercial platforms for complex video annotation tasks.

  </details>



- **Efficient On-Board Processing of Oblique UAV Video for Rapid Flood Extent Mapping**  
  Vishisht Sharma, Sam Leroux, Lisa Landuyt, Nick Witvrouwen, Pieter Simoens  
  _2026-01-16_ · https://arxiv.org/abs/2601.11290v1  
  <details><summary>Abstract</summary>

  Effective disaster response relies on rapid disaster response, where oblique aerial video is the primary modality for initial scouting due to its ability to maximize spatial coverage and situational awareness in limited flight time. However, the on-board processing of high-resolution oblique streams is severely bottlenecked by the strict Size, Weight, and Power (SWaP) constraints of Unmanned Aerial Vehicles (UAVs). The computational density required to process these wide-field-of-view streams precludes low-latency inference on standard edge hardware. To address this, we propose Temporal Token Reuse (TTR), an adaptive inference framework capable of accelerating video segmentation on embedded devices. TTR exploits the intrinsic spatiotemporal redundancy of aerial video by formulating image patches as tokens; it utilizes a lightweight similarity metric to dynamically identify static regions and propagate their precomputed deep features, thereby bypassing redundant backbone computations. We validate the framework on standard benchmarks and a newly curated Oblique Floodwater Dataset designed for hydrological monitoring. Experimental results on edge-grade hardware demonstrate that TTR achieves a 30% reduction in inference latency with negligible degradation in segmentation accuracy (< 0.5% mIoU). These findings confirm that TTR effectively shifts the operational Pareto frontier, enabling high-fidelity, real-time oblique video understanding for time-critical remote sensing missions

  </details>



- **X-Distill: Cross-Architecture Vision Distillation for Visuomotor Learning**  
  Maanping Shao, Feihong Zhang, Gu Zhang, Baiye Cheng, Zhengrong Xue, Huazhe Xu  
  _2026-01-16_ · https://arxiv.org/abs/2601.11269v1  
  <details><summary>Abstract</summary>

  Visuomotor policies often leverage large pre-trained Vision Transformers (ViTs) for their powerful generalization capabilities. However, their significant data requirements present a major challenge in the data-scarce context of most robotic learning settings, where compact CNNs with strong inductive biases can be more easily optimized. To address this trade-off, we introduce X-Distill, a simple yet highly effective method that synergizes the strengths of both architectures. Our approach involves an offline, cross-architecture knowledge distillation, transferring the rich visual representations of a large, frozen DINOv2 teacher to a compact ResNet-18 student on the general-purpose ImageNet dataset. This distilled encoder, now endowed with powerful visual priors, is then jointly fine-tuned with a diffusion policy head on the target manipulation tasks. Extensive experiments on $34$ simulated benchmarks and $5$ challenging real-world tasks demonstrate that our method consistently outperforms policies equipped with from-scratch ResNet or fine-tuned DINOv2 encoders. Notably, X-Distill also surpasses 3D encoders that utilize privileged point cloud observations or much larger Vision-Language Models. Our work highlights the efficacy of a simple, well-founded distillation strategy for achieving state-of-the-art performance in data-efficient robotic manipulation.

  </details>



- **Skill-Aware Diffusion for Generalizable Robotic Manipulation**  
  Aoshen Huang, Jiaming Chen, Jiyu Cheng, Ran Song, Wei Pan, Wei Zhang  
  _2026-01-16_ · https://arxiv.org/abs/2601.11266v1  
  <details><summary>Abstract</summary>

  Robust generalization in robotic manipulation is crucial for robots to adapt flexibly to diverse environments. Existing methods usually improve generalization by scaling data and networks, but model tasks independently and overlook skill-level information. Observing that tasks within the same skill share similar motion patterns, we propose Skill-Aware Diffusion (SADiff), which explicitly incorporates skill-level information to improve generalization. SADiff learns skill-specific representations through a skill-aware encoding module with learnable skill tokens, and conditions a skill-constrained diffusion model to generate object-centric motion flow. A skill-retrieval transformation strategy further exploits skill-specific trajectory priors to refine the mapping from 2D motion flow to executable 3D actions. Furthermore, we introduce IsaacSkill, a high-fidelity dataset containing fundamental robotic skills for comprehensive evaluation and sim-to-real transfer. Experiments in simulation and real-world settings show that SADiff achieves good performance and generalization across various manipulation tasks. Code, data, and videos are available at https://sites.google.com/view/sa-diff.

  </details>



- **FTDMamba: Frequency-Assisted Temporal Dilation Mamba for Unmanned Aerial Vehicle Video Anomaly Detection**  
  Cheng-Zhuang Liu, Si-Bao Chen, Qing-Ling Shu, Chris Ding, Jin Tang, Bin Luo  
  _2026-01-16_ · https://arxiv.org/abs/2601.11254v1  
  <details><summary>Abstract</summary>

  Recent advances in video anomaly detection (VAD) mainly focus on ground-based surveillance or unmanned aerial vehicle (UAV) videos with static backgrounds, whereas research on UAV videos with dynamic backgrounds remains limited. Unlike static scenarios, dynamically captured UAV videos exhibit multi-source motion coupling, where the motion of objects and UAV-induced global motion are intricately intertwined. Consequently, existing methods may misclassify normal UAV movements as anomalies or fail to capture true anomalies concealed within dynamic backgrounds. Moreover, many approaches do not adequately address the joint modeling of inter-frame continuity and local spatial correlations across diverse temporal scales. To overcome these limitations, we propose the Frequency-Assisted Temporal Dilation Mamba (FTDMamba) network for UAV VAD, including two core components: (1) a Frequency Decoupled Spatiotemporal Correlation Module, which disentangles coupled motion patterns and models global spatiotemporal dependencies through frequency analysis; and (2) a Temporal Dilation Mamba Module, which leverages Mamba's sequence modeling capability to jointly learn fine-grained temporal dynamics and local spatial structures across multiple temporal receptive fields. Additionally, unlike existing UAV VAD datasets which focus on static backgrounds, we construct a large-scale Moving UAV VAD dataset (MUVAD), comprising 222,736 frames with 240 anomaly events across 12 anomaly types. Extensive experiments demonstrate that FTDMamba achieves state-of-the-art (SOTA) performance on two public static benchmarks and the new MUVAD dataset. The code and MUVAD dataset will be available at: https://github.com/uavano/FTDMamba.

  </details>



- **VLAgents: A Policy Server for Efficient VLA Inference**  
  Tobias Jülg, Khaled Gamal, Nisarga Nilavadi, Pierre Krack, Seongjin Bien, Michael Krawez, Florian Walter, Wolfram Burgard  
  _2026-01-16_ · https://arxiv.org/abs/2601.11250v1  
  <details><summary>Abstract</summary>

  The rapid emergence of Vision-Language-Action models (VLAs) has a significant impact on robotics. However, their deployment remains complex due to the fragmented interfaces and the inherent communication latency in distributed setups. To address this, we introduce VLAgents, a modular policy server that abstracts VLA inferencing behind a unified Gymnasium-style protocol. Crucially, its communication layer transparently adapts to the context by supporting both zero-copy shared memory for high-speed simulation and compressed streaming for remote hardware. In this work, we present the architecture of VLAgents and validate it by integrating seven policies -- including OpenVLA and Pi Zero. In a benchmark with both local and remote communication, we further demonstrate how it outperforms the default policy servers provided by OpenVLA, OpenPi, and LeRobot. VLAgents is available at https://github.com/RobotControlStack/vlagents

  </details>



- **Democratizing planetary-scale analysis: An ultra-lightweight Earth embedding database for accurate and flexible global land monitoring**  
  Shuang Chen, Jie Wang, Shuai Yuan, Jiayang Li, Yu Xia, Yuanhong Liao, Junbo Wei, Jincheng Yuan, Xiaoqing Xu, Xiaolin Zhu, et al.  
  _2026-01-16_ · https://arxiv.org/abs/2601.11183v1  
  <details><summary>Abstract</summary>

  The rapid evolution of satellite-borne Earth Observation (EO) systems has revolutionized terrestrial monitoring, yielding petabyte-scale archives. However, the immense computational and storage requirements for global-scale analysis often preclude widespread use, hindering planetary-scale studies. To address these barriers, we present Embedded Seamless Data (ESD), an ultra-lightweight, 30-m global Earth embedding database spanning the 25-year period from 2000 to 2024. By transforming high-dimensional, multi-sensor observations from the Landsat series (5, 7, 8, and 9) and MODIS Terra into information-dense, quantized latent vectors, ESD distills essential geophysical and semantic features into a unified latent space. Utilizing the ESDNet architecture and Finite Scalar Quantization (FSQ), the dataset achieves a transformative ~340-fold reduction in data volume compared to raw archives. This compression allows the entire global land surface for a single year to be encapsulated within approximately 2.4 TB, enabling decadal-scale global analysis on standard local workstations. Rigorous validation demonstrates high reconstructive fidelity (MAE: 0.0130; RMSE: 0.0179; CC: 0.8543). By condensing the annual phenological cycle into 12 temporal steps, the embeddings provide inherent denoising and a semantically organized space that outperforms raw reflectance in land-cover classification, achieving 79.74% accuracy (vs. 76.92% for raw fusion). With robust few-shot learning capabilities and longitudinal consistency, ESD provides a versatile foundation for democratizing planetary-scale research and advancing next-generation geospatial artificial intelligence.

  </details>



- **Vision-as-Inverse-Graphics Agent via Interleaved Multimodal Reasoning**  
  Shaofeng Yin, Jiaxin Ge, Zora Zhiruo Wang, Xiuyu Li, Michael J. Black, Trevor Darrell, Angjoo Kanazawa, Haiwen Feng  
  _2026-01-16_ · https://arxiv.org/abs/2601.11109v1  
  <details><summary>Abstract</summary>

  Vision-as-inverse-graphics, the concept of reconstructing an image as an editable graphics program is a long-standing goal of computer vision. Yet even strong VLMs aren't able to achieve this in one-shot as they lack fine-grained spatial and physical grounding capability. Our key insight is that closing this gap requires interleaved multimodal reasoning through iterative execution and verification. Stemming from this, we present VIGA (Vision-as-Inverse-Graphic Agent) that starts from an empty world and reconstructs or edits scenes through a closed-loop write-run-render-compare-revise procedure. To support long-horizon reasoning, VIGA combines (i) a skill library that alternates generator and verifier roles and (ii) an evolving context memory that contains plans, code diffs, and render history. VIGA is task-agnostic as it doesn't require auxiliary modules, covering a wide range of tasks such as 3D reconstruction, multi-step scene editing, 4D physical interaction, and 2D document editing, etc. Empirically, we found VIGA substantially improves one-shot baselines on BlenderGym (35.32%) and SlideBench (117.17%). Moreover, VIGA is also model-agnostic as it doesn't require finetuning, enabling a unified protocol to evaluate heterogeneous foundation VLMs. To better support this protocol, we introduce BlenderBench, a challenging benchmark that stress-tests interleaved multimodal reasoning with graphics engine, where VIGA improves by 124.70%.

  </details>



- **Simple Models, Rich Representations: Visual Decoding from Primate Intracortical Neural Signals**  
  Matteo Ciferri, Matteo Ferrante, Nicola Toschi  
  _2026-01-16_ · https://arxiv.org/abs/2601.11108v1  
  <details><summary>Abstract</summary>

  Understanding how neural activity gives rise to perception is a central challenge in neuroscience. We address the problem of decoding visual information from high-density intracortical recordings in primates, using the THINGS Ventral Stream Spiking Dataset. We systematically evaluate the effects of model architecture, training objectives, and data scaling on decoding performance. Results show that decoding accuracy is mainly driven by modeling temporal dynamics in neural signals, rather than architectural complexity. A simple model combining temporal attention with a shallow MLP achieves up to 70% top-1 image retrieval accuracy, outperforming linear baselines as well as recurrent and convolutional approaches. Scaling analyses reveal predictable diminishing returns with increasing input dimensionality and dataset size. Building on these findings, we design a modular generative decoding pipeline that combines low-resolution latent reconstruction with semantically conditioned diffusion, generating plausible images from 200 ms of brain activity. This framework provides principles for brain-computer interfaces and semantic neural decoding.

  </details>



- **Graph Smoothing for Enhanced Local Geometry Learning in Point Cloud Analysis**  
  Shangbo Yuan, Jie Xu, Ping Hu, Xiaofeng Zhu, Na Zhao  
  _2026-01-16_ · https://arxiv.org/abs/2601.11102v1  
  <details><summary>Abstract</summary>

  Graph-based methods have proven to be effective in capturing relationships among points for 3D point cloud analysis. However, these methods often suffer from suboptimal graph structures, particularly due to sparse connections at boundary points and noisy connections in junction areas. To address these challenges, we propose a novel method that integrates a graph smoothing module with an enhanced local geometry learning module. Specifically, we identify the limitations of conventional graph structures, particularly in handling boundary points and junction areas. In response, we introduce a graph smoothing module designed to optimize the graph structure and minimize the negative impact of unreliable sparse and noisy connections. Based on the optimized graph structure, we improve the feature extract function with local geometry information. These include shape features derived from adaptive geometric descriptors based on eigenvectors and distribution features obtained through cylindrical coordinate transformation. Experimental results on real-world datasets validate the effectiveness of our method in various point cloud learning tasks, i.e., classification, part segmentation, and semantic segmentation.

  </details>



- **PhysRVG: Physics-Aware Unified Reinforcement Learning for Video Generative Models**  
  Qiyuan Zhang, Biao Gong, Shuai Tan, Zheng Zhang, Yujun Shen, Xing Zhu, Yuyuan Li, Kelu Yao, Chunhua Shen, Changqing Zou  
  _2026-01-16_ · https://arxiv.org/abs/2601.11087v1  
  <details><summary>Abstract</summary>

  Physical principles are fundamental to realistic visual simulation, but remain a significant oversight in transformer-based video generation. This gap highlights a critical limitation in rendering rigid body motion, a core tenet of classical mechanics. While computer graphics and physics-based simulators can easily model such collisions using Newton formulas, modern pretrain-finetune paradigms discard the concept of object rigidity during pixel-level global denoising. Even perfectly correct mathematical constraints are treated as suboptimal solutions (i.e., conditions) during model optimization in post-training, fundamentally limiting the physical realism of generated videos. Motivated by these considerations, we introduce, for the first time, a physics-aware reinforcement learning paradigm for video generation models that enforces physical collision rules directly in high-dimensional spaces, ensuring the physics knowledge is strictly applied rather than treated as conditions. Subsequently, we extend this paradigm to a unified framework, termed Mimicry-Discovery Cycle (MDcycle), which allows substantial fine-tuning while fully preserving the model's ability to leverage physics-grounded feedback. To validate our approach, we construct new benchmark PhysRVGBench and perform extensive qualitative and quantitative experiments to thoroughly assess its effectiveness.

  </details>



- **Generation of Chest CT pulmonary Nodule Images by Latent Diffusion Models using the LIDC-IDRI Dataset**  
  Kaito Urata, Maiko Nagao, Atsushi Teramoto, Kazuyoshi Imaizumi, Masashi Kondo, Hiroshi Fujita  
  _2026-01-16_ · https://arxiv.org/abs/2601.11085v1  
  <details><summary>Abstract</summary>

  Recently, computer-aided diagnosis systems have been developed to support diagnosis, but their performance depends heavily on the quality and quantity of training data. However, in clinical practice, it is difficult to collect the large amount of CT images for specific cases, such as small cell carcinoma with low epidemiological incidence or benign tumors that are difficult to distinguish from malignant ones. This leads to the challenge of data imbalance. In this study, to address this issue, we proposed a method to automatically generate chest CT nodule images that capture target features using latent diffusion models (LDM) and verified its effectiveness. Using the LIDC-IDRI dataset, we created pairs of nodule images and finding-based text prompts based on physician evaluations. For the image generation models, we used Stable Diffusion version 1.5 (SDv1) and 2.0 (SDv2), which are types of LDM. Each model was fine-tuned using the created dataset. During the generation process, we adjusted the guidance scale (GS), which indicates the fidelity to the input text. Both quantitative and subjective evaluations showed that SDv2 (GS = 5) achieved the best performance in terms of image quality, diversity, and text consistency. In the subjective evaluation, no statistically significant differences were observed between the generated images and real images, confirming that the quality was equivalent to real clinical images. We proposed a method for generating chest CT nodule images based on input text using LDM. Evaluation results demonstrated that the proposed method could generate high-quality images that successfully capture specific medical features.

  </details>



- **Visual Marker Search for Autonomous Drone Landing in Diverse Urban Environments**  
  Jiaohong Yao, Linfeng Liang, Yao Deng, Xi Zheng, Richard Han, Yuankai Qi  
  _2026-01-16_ · https://arxiv.org/abs/2601.11078v1  
  <details><summary>Abstract</summary>

  Marker-based landing is widely used in drone delivery and return-to-base systems for its simplicity and reliability. However, most approaches assume idealized landing site visibility and sensor performance, limiting robustness in complex urban settings. We present a simulation-based evaluation suite on the AirSim platform with systematically varied urban layouts, lighting, and weather to replicate realistic operational diversity. Using onboard camera sensors (RGB for marker detection and depth for obstacle avoidance), we benchmark two heuristic coverage patterns and a reinforcement learning-based agent, analyzing how exploration strategy and scene complexity affect success rate, path efficiency, and robustness. Results underscore the need to evaluate marker-based autonomous landing under diverse, sensor-relevant conditions to guide the development of reliable aerial navigation systems.

  </details>



- **Visual question answering-based image-finding generation for pulmonary nodules on chest CT from structured annotations**  
  Maiko Nagao, Kaito Urata, Atsushi Teramoto, Kazuyoshi Imaizumi, Masashi Kondo, Hiroshi Fujita  
  _2026-01-16_ · https://arxiv.org/abs/2601.11075v1  
  <details><summary>Abstract</summary>

  Interpretation of imaging findings based on morphological characteristics is important for diagnosing pulmonary nodules on chest computed tomography (CT) images. In this study, we constructed a visual question answering (VQA) dataset from structured data in an open dataset and investigated an image-finding generation method for chest CT images, with the aim of enabling interactive diagnostic support that presents findings based on questions that reflect physicians' interests rather than fixed descriptions. In this study, chest CT images included in the Lung Image Database Consortium and Image Database Resource Initiative (LIDC-IDRI) datasets were used. Regions of interest surrounding the pulmonary nodules were extracted from these images, and image findings and questions were defined based on morphological characteristics recorded in the database. A dataset comprising pairs of cropped images, corresponding questions, and image findings was constructed, and the VQA model was fine-tuned on it. Language evaluation metrics such as BLEU were used to evaluate the generated image findings. The VQA dataset constructed using the proposed method contained image findings with natural expressions as radiological descriptions. In addition, the generated image findings showed a high CIDEr score of 3.896, and a high agreement with the reference findings was obtained through evaluation based on morphological characteristics. We constructed a VQA dataset for chest CT images using structured information on the morphological characteristics from the LIDC-IDRI dataset. Methods for generating image findings in response to these questions have also been investigated. Based on the generated results and evaluation metric scores, the proposed method was effective as an interactive diagnostic support system that can present image findings according to physicians' interests.

  </details>



- **H-AIM: Orchestrating LLMs, PDDL, and Behavior Trees for Hierarchical Multi-Robot Planning**  
  Haishan Zeng, Peng Li  
  _2026-01-16_ · https://arxiv.org/abs/2601.11063v1  
  <details><summary>Abstract</summary>

  In embodied artificial intelligence, enabling heterogeneous robot teams to execute long-horizon tasks from high-level instructions remains a critical challenge. While large language models (LLMs) show promise in instruction parsing and preliminary planning, they exhibit limitations in long-term reasoning and dynamic multi-robot coordination. We propose Hierarchical Autonomous Intelligent Multi-Robot Planning(H-AIM), a novel embodied multi-robot task planning framework that addresses these issues through a three-stage cascaded architecture: 1) It leverages an LLM to parse instructions and generate Planning Domain Definition Language (PDDL) problem descriptions, thereby transforming commands into formal planning problems; 2) It combines the semantic reasoning of LLMs with the search capabilities of a classical planner to produce optimized action sequences; 3) It compiles the resulting plan into behavior trees for reactive control. The framework supports dynamically sized heterogeneous robot teams via a shared blackboard mechanism for communication and state synchronization. To validate our approach, we introduce the MACE-THOR benchmark dataset, comprising 42 complex tasks across 8 distinct household layouts. Experimental results demonstrate that H-AIM achieves a remarkable performance improvement, elevating the task success rate from 12% to 55% and boosting the goal condition recall from 32% to 72% against the strongest baseline, LaMMA-P.

  </details>



- **Your One-Stop Solution for AI-Generated Video Detection**  
  Long Ma, Zihao Xue, Yan Wang, Zhiyuan Yan, Jin Xu, Xiaorui Jiang, Haiyang Yu, Yong Liao, Zhen Bi  
  _2026-01-16_ · https://arxiv.org/abs/2601.11035v1  
  <details><summary>Abstract</summary>

  Recent advances in generative modeling can create remarkably realistic synthetic videos, making it increasingly difficult for humans to distinguish them from real ones and necessitating reliable detection methods. However, two key limitations hinder the development of this field. \textbf{From the dataset perspective}, existing datasets are often limited in scale and constructed using outdated or narrowly scoped generative models, making it difficult to capture the diversity and rapid evolution of modern generative techniques. Moreover, the dataset construction process frequently prioritizes quantity over quality, neglecting essential aspects such as semantic diversity, scenario coverage, and technological representativeness. \textbf{From the benchmark perspective}, current benchmarks largely remain at the stage of dataset creation, leaving many fundamental issues and in-depth analysis yet to be systematically explored. Addressing this gap, we propose AIGVDBench, a benchmark designed to be comprehensive and representative, covering \textbf{31} state-of-the-art generation models and over \textbf{440,000} videos. By executing more than \textbf{1,500} evaluations on \textbf{33} existing detectors belonging to four distinct categories. This work presents \textbf{8 in-depth analyses} from multiple perspectives and identifies \textbf{4 novel findings} that offer valuable insights for future research. We hope this work provides a solid foundation for advancing the field of AI-generated video detection. Our benchmark is open-sourced at https://github.com/LongMa-2025/AIGVDBench.

  </details>


