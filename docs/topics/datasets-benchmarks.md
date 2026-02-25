# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-02-25 07:15 UTC_

Total papers shown: **50**


---

- **Learning from Trials and Errors: Reflective Test-Time Planning for Embodied LLMs**  
  Yining Hong, Huang Huang, Manling Li, Li Fei-Fei, Jiajun Wu, Yejin Choi  
  _2026-02-24_ · https://arxiv.org/abs/2602.21198v1  
  <details><summary>Abstract</summary>

  Embodied LLMs endow robots with high-level task reasoning, but they cannot reflect on what went wrong or why, turning deployment into a sequence of independent trials where mistakes repeat rather than accumulate into experience. Drawing upon human reflective practitioners, we introduce Reflective Test-Time Planning, which integrates two modes of reflection: \textit{reflection-in-action}, where the agent uses test-time scaling to generate and score multiple candidate actions using internal reflections before execution; and \textit{reflection-on-action}, which uses test-time training to update both its internal reflection model and its action policy based on external reflections after execution. We also include retrospective reflection, allowing the agent to re-evaluate earlier decisions and perform model updates with hindsight for proper long-horizon credit assignment. Experiments on our newly-designed Long-Horizon Household benchmark and MuJoCo Cupboard Fitting benchmark show significant gains over baseline models, with ablative studies validating the complementary roles of reflection-in-action and reflection-on-action. Qualitative analyses, including real-robot trials, highlight behavioral correction through reflection.

  </details>



- **NoRD: A Data-Efficient Vision-Language-Action Model that Drives without Reasoning**  
  Ishaan Rawal, Shubh Gupta, Yihan Hu, Wei Zhan  
  _2026-02-24_ · https://arxiv.org/abs/2602.21172v1  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models are advancing autonomous driving by replacing modular pipelines with unified end-to-end architectures. However, current VLAs face two expensive requirements: (1) massive dataset collection, and (2) dense reasoning annotations. In this work, we address both challenges with \modelname (\textbf{No} \textbf{R}easoning for \textbf{D}riving). Compared to existing VLAs, \modelname achieves competitive performance while being fine-tuned on $<$60\% of the data and no reasoning annotations, resulting in 3$\times$ fewer tokens. We identify that standard Group Relative Policy Optimization (GRPO) fails to yield significant improvements when applied to policies trained on such small, reasoning-free datasets. We show that this limitation stems from difficulty bias, which disproportionately penalizes reward signals from scenarios that produce high-variance rollouts within GRPO. \modelname overcomes this by incorporating Dr.~GRPO, a recent algorithm designed to mitigate difficulty bias in LLMs. As a result, \modelname achieves competitive performance on Waymo and NAVSIM with a fraction of the training data and no reasoning overhead, enabling more efficient autonomous systems.

  </details>



- **HALO: A Unified Vision-Language-Action Model for Embodied Multimodal Chain-of-Thought Reasoning**  
  Quanxin Shou, Fangqi Zhu, Shawn Chen, Puxin Yan, Zhengyang Yan, Yikun Miao, Xiaoyi Pang, Zicong Hong, Ruikai Shi, Hao Huang, et al.  
  _2026-02-24_ · https://arxiv.org/abs/2602.21157v1  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models have shown strong performance in robotic manipulation, but often struggle in long-horizon or out-of-distribution scenarios due to the lack of explicit mechanisms for multimodal reasoning and anticipating how the world will evolve under action. Recent works introduce textual chain-of-thought or visual subgoal prediction within VLA models to reason, but still fail to offer a unified human-like reasoning framework for joint textual reasoning, visual foresight, and action prediction. To this end, we propose HALO, a unified VLA model that enables embodied multimodal chain-of-thought (EM-CoT) reasoning through a sequential process of textual task reasoning, visual subgoal prediction for fine-grained guidance, and EM-CoT-augmented action prediction. We instantiate HALO with a Mixture-of-Transformers (MoT) architecture that decouples semantic reasoning, visual foresight, and action prediction into specialized experts while allowing seamless cross-expert collaboration. To enable HALO learning at scale, we introduce an automated pipeline to synthesize EM-CoT training data along with a carefully crafted training recipe. Extensive experiments demonstrate that: (1) HALO achieves superior performance in both simulated and real-world environments, surpassing baseline policy pi_0 by 34.1% on RoboTwin benchmark; (2) all proposed components of the training recipe and EM-CoT design help improve task success rate; and (3) HALO exhibits strong generalization capabilities under aggressive unseen environmental randomization with our proposed EM-CoT reasoning.

  </details>



- **LUMEN: Longitudinal Multi-Modal Radiology Model for Prognosis and Diagnosis**  
  Zhifan Jiang, Dong Yang, Vishwesh Nath, Abhijeet Parida, Nishad P. Kulkarni, Ziyue Xu, Daguang Xu, Syed Muhammad Anwar, Holger R. Roth, Marius George Linguraru  
  _2026-02-24_ · https://arxiv.org/abs/2602.21142v1  
  <details><summary>Abstract</summary>

  Large vision-language models (VLMs) have evolved from general-purpose applications to specialized use cases such as in the clinical domain, demonstrating potential for decision support in radiology. One promising application is assisting radiologists in decision-making by the analysis of radiology imaging data such as chest X-rays (CXR) via a visual and natural language question-answering (VQA) interface. When longitudinal imaging is available, radiologists analyze temporal changes, which are essential for accurate diagnosis and prognosis. The manual longitudinal analysis is a time-consuming process, motivating the development of a training framework that can provide prognostic capabilities. We introduce a novel training framework LUMEN, that is optimized for longitudinal CXR interpretation, leveraging multi-image and multi-task instruction fine-tuning to enhance prognostic and diagnostic performance. We conduct experiments on the publicly available MIMIC-CXR and its associated Medical-Diff-VQA datasets. We further formulate and construct a novel instruction-following dataset incorporating longitudinal studies, enabling the development of a prognostic VQA task. Our method demonstrates significant improvements over baseline models in diagnostic VQA tasks, and more importantly, shows promising potential for prognostic capabilities. These results underscore the value of well-designed, instruction-tuned VLMs in enabling more accurate and clinically meaningful radiological interpretation of longitudinal radiological imaging data.

  </details>



- **SynthRender and IRIS: Open-Source Framework and Dataset for Bidirectional Sim-Real Transfer in Industrial Object Perception**  
  Jose Moises Araya-Martinez, Thushar Tom, Adrián Sanchis Reig, Pablo Rey Valiente, Jens Lambrecht, Jörg Krüger  
  _2026-02-24_ · https://arxiv.org/abs/2602.21141v1  
  <details><summary>Abstract</summary>

  Object perception is fundamental for tasks such as robotic material handling and quality inspection. However, modern supervised deep-learning perception models require large datasets for robust automation under semi-uncontrolled conditions. The cost of acquiring and annotating such data for proprietary parts is a major barrier for widespread deployment. In this context, we release SynthRender, an open source framework for synthetic image generation with Guided Domain Randomization capabilities. Furthermore, we benchmark recent Reality-to-Simulation techniques for 3D asset creation from 2D images of real parts. Combined with Domain Randomization, these synthetic assets provide low-overhead, transferable data even for parts lacking 3D files. We also introduce IRIS, the Industrial Real-Sim Imagery Set, containing 32 categories with diverse textures, intra-class variation, strong inter-class similarities and about 20,000 labels. Ablations on multiple benchmarks outline guidelines for efficient data generation with SynthRender. Our method surpasses existing approaches, achieving 99.1% mAP@50 on a public robotics dataset, 98.3% mAP@50 on an automotive benchmark, and 95.3% mAP@50 on IRIS.

  </details>



- **UDVideoQA: A Traffic Video Question Answering Dataset for Multi-Object Spatio-Temporal Reasoning in Urban Dynamics**  
  Joseph Raj Vishal, Nagasiri Poluri, Katha Naik, Rutuja Patil, Kashyap Hegde Kota, Krishna Vinod, Prithvi Jai Ramesh, Mohammad Farhadi, Yezhou Yang, Bharatesh Chakravarthi  
  _2026-02-24_ · https://arxiv.org/abs/2602.21137v1  
  <details><summary>Abstract</summary>

  Understanding the complex, multi-agent dynamics of urban traffic remains a fundamental challenge for video language models. This paper introduces Urban Dynamics VideoQA, a benchmark dataset that captures the unscripted real-world behavior of dynamic urban scenes. UDVideoQA is curated from 16 hours of traffic footage recorded at multiple city intersections under diverse traffic, weather, and lighting conditions. It employs an event-driven dynamic blur technique to ensure privacy preservation without compromising scene fidelity. Using a unified annotation pipeline, the dataset contains 28K question-answer pairs generated across 8 hours of densely annotated video, averaging one question per second. Its taxonomy follows a hierarchical reasoning level, spanning basic understanding and attribution to event reasoning, reverse reasoning, and counterfactual inference, enabling systematic evaluation of both visual grounding and causal reasoning. Comprehensive experiments benchmark 10 SOTA VideoLMs on UDVideoQA and 8 models on a complementary video question generation benchmark. Results reveal a persistent perception-reasoning gap, showing models that excel in abstract inference often fail with fundamental visual grounding. While models like Gemini Pro achieve the highest zero-shot accuracy, fine-tuning the smaller Qwen2.5-VL 7B model on UDVideoQA bridges this gap, achieving performance comparable to proprietary systems. In VideoQGen, Gemini 2.5 Pro, and Qwen3 Max generate the most relevant and complex questions, though all models exhibit limited linguistic diversity, underscoring the need for human-centric evaluation. The UDVideoQA suite, including the dataset, annotation tools, and benchmarks for both VideoQA and VideoQGen, provides a foundation for advancing robust, privacy-aware, and real-world multimodal reasoning. UDVideoQA is available at https://ud-videoqa.github.io/UD-VideoQA/UD-VideoQA/.

  </details>



- **OCR-Agent: Agentic OCR with Capability and Memory Reflection**  
  Shimin Wen, Zeyu Zhang, Xingdou Bian, Hongjie Zhu, Lulu He, Layi Shama, Daji Ergu, Ying Cai  
  _2026-02-24_ · https://arxiv.org/abs/2602.21053v1  
  <details><summary>Abstract</summary>

  Large Vision-Language Models (VLMs) have demonstrated significant potential on complex visual understanding tasks through iterative optimization methods.However, these models generally lack effective self-correction mechanisms, making it difficult for them to independently rectify cognitive biases. Consequently, during multi-turn revisions, they often fall into repetitive and ineffective attempts, failing to achieve stable improvements in answer quality.To address this issue, we propose a novel iterative self-correction framework that endows models with two key capabilities: Capability Reflection and Memory Reflection. This framework guides the model to first diagnose errors and generate a correction plan via Capability Reflection, then leverage Memory Reflection to review past attempts to avoid repetition and explore new solutions, and finally, optimize the answer through rigorous re-reasoning. Experiments on the challenging OCRBench v2 benchmark show that OCR-Agent outperforms the current open-source SOTA model InternVL3-8B by +2.0 on English and +1.2 on Chinese subsets, while achieving state-of-the-art results in Visual Understanding (79.9) and Reasoning (66.5) - surpassing even larger fine-tuned models. Our method demonstrates that structured, self-aware reflection can significantly enhance VLMs' reasoning robustness without additional training. Code: https://github.com/AIGeeksGroup/OCR-Agent.

  </details>



- **MIP Candy: A Modular PyTorch Framework for Medical Image Processing**  
  Tianhao Fu, Yucheng Chen  
  _2026-02-24_ · https://arxiv.org/abs/2602.21033v1  
  <details><summary>Abstract</summary>

  Medical image processing demands specialized software that handles high-dimensional volumetric data, heterogeneous file formats, and domain-specific training procedures. Existing frameworks either provide low-level components that require substantial integration effort or impose rigid, monolithic pipelines that resist modification. We present MIP Candy (MIPCandy), a freely available, PyTorch-based framework designed specifically for medical image processing. MIPCandy provides a complete, modular pipeline spanning data loading, training, inference, and evaluation, allowing researchers to obtain a fully functional process workflow by implementing a single method, $\texttt{build_network}$, while retaining fine-grained control over every component. Central to the design is $\texttt{LayerT}$, a deferred configuration mechanism that enables runtime substitution of convolution, normalization, and activation modules without subclassing. The framework further offers built-in $k$-fold cross-validation, dataset inspection with automatic region-of-interest detection, deep supervision, exponential moving average, multi-frontend experiment tracking (Weights & Biases, Notion, MLflow), training state recovery, and validation score prediction via quotient regression. An extensible bundle ecosystem provides pre-built model implementations that follow a consistent trainer--predictor pattern and integrate with the core framework without modification. MIPCandy is open-source under the Apache-2.0 license and requires Python~3.12 or later. Source code and documentation are available at https://github.com/ProjectNeura/MIPCandy.

  </details>



- **From Perception to Action: An Interactive Benchmark for Vision Reasoning**  
  Yuhao Wu, Maojia Song, Yihuai Lan, Lei Wang, Zhiqiang Hu, Yao Xiao, Heng Zhou, Weihua Zheng, Dylan Raharja, Soujanya Poria, et al.  
  _2026-02-24_ · https://arxiv.org/abs/2602.21015v1  
  <details><summary>Abstract</summary>

  Understanding the physical structure is essential for real-world applications such as embodied agents, interactive design, and long-horizon manipulation. Yet, prevailing Vision-Language Model (VLM) evaluations still center on structure-agnostic, single-turn setups (e.g., VQA), which fail to assess agents' ability to reason about how geometry, contact, and support relations jointly constrain what actions are possible in a dynamic environment. To address this gap, we introduce the Causal Hierarchy of Actions and Interactions (CHAIN) benchmark, an interactive 3D, physics-driven testbed designed to evaluate whether models can understand, plan, and execute structured action sequences grounded in physical constraints. CHAIN shifts evaluation from passive perception to active problem solving, spanning tasks such as interlocking mechanical puzzles and 3D stacking and packing. We conduct a comprehensive study of state-of-the-art VLMs and diffusion-based models under unified interactive settings. Our results show that top-performing models still struggle to internalize physical structure and causal constraints, often failing to produce reliable long-horizon plans and cannot robustly translate perceived structure into effective actions. The project is available at https://social-ai-studio.github.io/CHAIN/.

  </details>



- **Multimodal MRI Report Findings Supervised Brain Lesion Segmentation with Substructures**  
  Yubin Ge, Yongsong Huang, Xiaofeng Liu  
  _2026-02-24_ · https://arxiv.org/abs/2602.20994v1  
  <details><summary>Abstract</summary>

  Report-supervised (RSuper) learning seeks to alleviate the need for dense tumor voxel labels with constraints derived from radiology reports (e.g., volumes, counts, sizes, locations). In MRI studies of brain tumors, however, we often involve multi-parametric scans and substructures. Here, fine-grained modality/parameter-wise reports are usually provided along with global findings and are correlated with different substructures. Moreover, the reports often describe only the largest lesion and provide qualitative or uncertain cues (``mild,'' ``possible''). Classical RSuper losses (e.g., sum volume consistency) can over-constrain or hallucinate unreported findings under such incompleteness, and are unable to utilize these hierarchical findings or exploit the priors of varied lesion types in a merged dataset. We explicitly parse the global quantitative and modality-wise qualitative findings and introduce a unified, one-sided, uncertainty-aware formulation (MS-RSuper) that: (i) aligns modality-specific qualitative cues (e.g., T1c enhancement, FLAIR edema) with their corresponding substructures using existence and absence losses; (ii) enforces one-sided lower-bounds for partial quantitative cues (e.g., largest lesion size, minimal multiplicity); and (iii) adds extra- vs. intra-axial anatomical priors to respect cohort differences. Certainty tokens scale penalties; missing cues are down-weighted. On 1238 report-labeled BraTS-MET/MEN scans, our MS-RSuper largely outperforms both a sparsely-supervised baseline and a naive RSuper method.

  </details>



- **Are Multimodal Large Language Models Good Annotators for Image Tagging?**  
  Ming-Kun Xie, Jia-Hao Xiao, Zhiqiang Kou, Zhongnian Li, Gang Niu, Masashi Sugiyama  
  _2026-02-24_ · https://arxiv.org/abs/2602.20972v1  
  <details><summary>Abstract</summary>

  Image tagging, a fundamental vision task, traditionally relies on human-annotated datasets to train multi-label classifiers, which incurs significant labor and costs. While Multimodal Large Language Models (MLLMs) offer promising potential to automate annotation, their capability to replace human annotators remains underexplored. This paper aims to analyze the gap between MLLM-generated and human annotations and to propose an effective solution that enables MLLM-based annotation to replace manual labeling. Our analysis of MLLM annotations reveals that, under a conservative estimate, MLLMs can reduce annotation cost to as low as one-thousandth of the human cost, mainly accounting for GPU usage, which is nearly negligible compared to manual efforts. Their annotation quality reaches about 50\% to 80\% of human performance, while achieving over 90\% performance on downstream training tasks.Motivated by these findings, we propose TagLLM, a novel framework for image tagging, which aims to narrow the gap between MLLM-generated and human annotations. TagLLM comprises two components: Candidates generation, which employs structured group-wise prompting to efficiently produce a compact candidate set that covers as many true labels as possible while reducing subsequent annotation workload; and label disambiguation, which interactively calibrates the semantic concept of categories in the prompts and effectively refines the candidate labels. Extensive experiments show that TagLLM substantially narrows the gap between MLLM-generated and human annotations, especially in downstream training performance, where it closes about 60\% to 80\% of the difference.

  </details>



- **UFO: Unifying Feed-Forward and Optimization-based Methods for Large Driving Scene Modeling**  
  Kaiyuan Tan, Yingying Shen, Mingfei Tu, Haohui Zhu, Bing Wang, Guang Chen, Hangjun Ye, Haiyang Sun  
  _2026-02-24_ · https://arxiv.org/abs/2602.20943v1  
  <details><summary>Abstract</summary>

  Dynamic driving scene reconstruction is critical for autonomous driving simulation and closed-loop learning. While recent feed-forward methods have shown promise for 3D reconstruction, they struggle with long-range driving sequences due to quadratic complexity in sequence length and challenges in modeling dynamic objects over extended durations. We propose UFO, a novel recurrent paradigm that combines the benefits of optimization-based and feed-forward methods for efficient long-range 4D reconstruction. Our approach maintains a 4D scene representation that is iteratively refined as new observations arrive, using a visibility-based filtering mechanism to select informative scene tokens and enable efficient processing of long sequences. For dynamic objects, we introduce an object pose-guided modeling approach that supports accurate long-range motion capture. Experiments on the Waymo Open Dataset demonstrate that our method significantly outperforms both per-scene optimization and existing feed-forward methods across various sequence lengths. Notably, our approach can reconstruct 16-second driving logs within 0.5 second while maintaining superior visual quality and geometric accuracy.

  </details>



- **Computing a Characteristic Orientation for Rotation-Independent Image Analysis**  
  Cristian Valero-Abundio, Emilio Sansano-Sansano, Raúl Montoliu, Marina Martínez García  
  _2026-02-24_ · https://arxiv.org/abs/2602.20930v1  
  <details><summary>Abstract</summary>

  Handling geometric transformations, particularly rotations, remains a challenge in deep learning for computer vision. Standard neural networks lack inherent rotation invariance and typically rely on data augmentation or architectural modifications to improve robustness. Although effective, these approaches increase computational demands, require specialised implementations, or alter network structures, limiting their applicability. This paper introduces General Intensity Direction (GID), a preprocessing method that improves rotation robustness without modifying the network architecture. The method estimates a global orientation for each image and aligns it to a canonical reference frame, allowing standard models to process inputs more consistently across different rotations. Unlike moment-based approaches that extract invariant descriptors, this method directly transforms the image while preserving spatial structure, making it compatible with convolutional networks. Experimental evaluation on the rotated MNIST dataset shows that the proposed method achieves higher accuracy than state-of-the-art rotation-invariant architectures. Additional experiments on the CIFAR-10 dataset, confirm that the method remains effective under more complex conditions.

  </details>



- **ParkDiffusion++: Ego Intention Conditioned Joint Multi-Agent Trajectory Prediction for Automated Parking using Diffusion Models**  
  Jiarong Wei, Anna Rehr, Christian Feist, Abhinav Valada  
  _2026-02-24_ · https://arxiv.org/abs/2602.20923v1  
  <details><summary>Abstract</summary>

  Automated parking is a challenging operational domain for advanced driver assistance systems, requiring robust scene understanding and interaction reasoning. The key challenge is twofold: (i) predict multiple plausible ego intentions according to context and (ii) for each intention, predict the joint responses of surrounding agents, enabling effective what-if decision-making. However, existing methods often fall short, typically treating these interdependent problems in isolation. We propose ParkDiffusion++, which jointly learns a multi-modal ego intention predictor and an ego-conditioned multi-agent joint trajectory predictor for automated parking. Our approach makes several key contributions. First, we introduce an ego intention tokenizer that predicts a small set of discrete endpoint intentions from agent histories and vectorized map polylines. Second, we perform ego-intention-conditioned joint prediction, yielding socially consistent predictions of the surrounding agents for each possible ego intention. Third, we employ a lightweight safety-guided denoiser with different constraints to refine joint scenes during training, thus improving accuracy and safety. Fourth, we propose counterfactual knowledge distillation, where an EMA teacher refined by a frozen safety-guided denoiser provides pseudo-targets that capture how agents react to alternative ego intentions. Extensive evaluations demonstrate that ParkDiffusion++ achieves state-of-the-art performance on the Dragon Lake Parking (DLP) dataset and the Intersections Drone (inD) dataset. Importantly, qualitative what-if visualizations show that other agents react appropriately to different ego intentions.

  </details>



- **Task-oriented grasping for dexterous robots using postural synergies and reinforcement learning**  
  Dimitrios Dimou, José Santos-Victor, Plinio Moreno  
  _2026-02-24_ · https://arxiv.org/abs/2602.20915v1  
  <details><summary>Abstract</summary>

  In this paper, we address the problem of task-oriented grasping for humanoid robots, emphasizing the need to align with human social norms and task-specific objectives. Existing methods, employ a variety of open-loop and closed-loop approaches but lack an end-to-end solution that can grasp several objects while taking into account the downstream task's constraints. Our proposed approach employs reinforcement learning to enhance task-oriented grasping, prioritizing the post-grasp intention of the agent. We extract human grasp preferences from the ContactPose dataset, and train a hand synergy model based on the Variational Autoencoder (VAE) to imitate the participant's grasping actions. Based on this data, we train an agent able to grasp multiple objects while taking into account distinct post-grasp intentions that are task-specific. By combining data-driven insights from human grasping behavior with learning by exploration provided by reinforcement learning, we can develop humanoid robots capable of context-aware manipulation actions, facilitating collaboration in human-centered environments.

  </details>



- **From Isolation to Integration: Building an Adaptive Expert Forest for Pre-Trained Model-based Class-Incremental Learning**  
  Ruiqi Liu, Boyu Diao, Hangda Liu, Zhulin An, Fei Wang, Yongjun Xu  
  _2026-02-24_ · https://arxiv.org/abs/2602.20911v1  
  <details><summary>Abstract</summary>

  Class-Incremental Learning (CIL) requires models to learn new classes without forgetting old ones. A common method is to freeze a pre-trained model and train a new, lightweight adapter for each task. While this prevents forgetting, it treats the learned knowledge as a simple, unstructured collection and fails to use the relationships between tasks. To this end, we propose the Semantic-guided Adaptive Expert Forest (SAEF), a new method that organizes adapters into a structured hierarchy for better knowledge sharing. SAEF first groups tasks into conceptual clusters based on their semantic relationships. Then, within each cluster, it builds a balanced expert tree by creating new adapters from merging the adapters of similar tasks. At inference time, SAEF finds and activates a set of relevant experts from the forest for any given input. The final prediction is made by combining the outputs of these activated experts, weighted by how confident each expert is. Experiments on several benchmark datasets show that SAEF achieves SOTA performance.

  </details>



- **TextPecker: Rewarding Structural Anomaly Quantification for Enhancing Visual Text Rendering**  
  Hanshen Zhu, Yuliang Liu, Xuecheng Wu, An-Lan Wang, Hao Feng, Dingkang Yang, Chao Feng, Can Huang, Jingqun Tang, Xiang Bai  
  _2026-02-24_ · https://arxiv.org/abs/2602.20903v1  
  <details><summary>Abstract</summary>

  Visual Text Rendering (VTR) remains a critical challenge in text-to-image generation, where even advanced models frequently produce text with structural anomalies such as distortion, blurriness, and misalignment. However, we find that leading MLLMs and specialist OCR models largely fail to perceive these structural anomalies, creating a critical bottleneck for both VTR evaluation and RL-based optimization. As a result, even state-of-the-art generators (e.g., SeedDream4.0, Qwen-Image) still struggle to render structurally faithful text. To address this, we propose TextPecker, a plug-and-play structural anomaly perceptive RL strategy that mitigates noisy reward signals and works with any textto-image generator. To enable this capability, we construct a recognition dataset with character-level structural-anomaly annotations and develop a stroke-editing synthesis engine to expand structural-error coverage. Experiments show that TextPecker consistently improves diverse text-to-image models; even on the well-optimized Qwen-Image, it significantly yields average gains of 4% in structural fidelity and 8.7% in semantic alignment for Chinese text rendering, establishing a new state-of-the-art in high-fidelity VTR. Our work fills a gap in VTR optimization, providing a foundational step towards reliable and structural faithful visual text generation.

  </details>



- **SpatiaLQA: A Benchmark for Evaluating Spatial Logical Reasoning in Vision-Language Models**  
  Yuechen Xie, Xiaoyan Zhang, Yicheng Shan, Hao Zhu, Rui Tang, Rong Wei, Mingli Song, Yuanyu Wan, Jie Song  
  _2026-02-24_ · https://arxiv.org/abs/2602.20901v1  
  <details><summary>Abstract</summary>

  Vision-Language Models (VLMs) have been increasingly applied in real-world scenarios due to their outstanding understanding and reasoning capabilities. Although VLMs have already demonstrated impressive capabilities in common visual question answering and logical reasoning, they still lack the ability to make reasonable decisions in complex real-world environments. We define this ability as spatial logical reasoning, which not only requires understanding the spatial relationships among objects in complex scenes, but also the logical dependencies between steps in multi-step tasks. To bridge this gap, we introduce Spatial Logical Question Answering (SpatiaLQA), a benchmark designed to evaluate the spatial logical reasoning capabilities of VLMs. SpatiaLQA consists of 9,605 question answer pairs derived from 241 real-world indoor scenes. We conduct extensive experiments on 41 mainstream VLMs, and the results show that even the most advanced models still struggle with spatial logical reasoning. To address this issue, we propose a method called recursive scene graph assisted reasoning, which leverages visual foundation models to progressively decompose complex scenes into task-relevant scene graphs, thereby enhancing the spatial logical reasoning ability of VLMs, outperforming all previous methods. Code and dataset are available at https://github.com/xieyc99/SpatiaLQA.

  </details>



- **MUSE: Harnessing Precise and Diverse Semantics for Few-Shot Whole Slide Image Classification**  
  Jiahao Xu, Sheng Huang, Xin Zhang, Zhixiong Nan, Jiajun Dong, Nankun Mu  
  _2026-02-24_ · https://arxiv.org/abs/2602.20873v1  
  <details><summary>Abstract</summary>

  In computational pathology, few-shot whole slide image classification is primarily driven by the extreme scarcity of expert-labeled slides. Recent vision-language methods incorporate textual semantics generated by large language models, but treat these descriptions as static class-level priors that are shared across all samples and lack sample-wise refinement. This limits both the diversity and precision of visual-semantic alignment, hindering generalization under limited supervision. To overcome this, we propose the stochastic MUlti-view Semantic Enhancement (MUSE), a framework that first refines semantic precision via sample-wise adaptation and then enhances semantic richness through retrieval-augmented multi-view generation. Specifically, MUSE introduces Sample-wise Fine-grained Semantic Enhancement (SFSE), which yields a fine-grained semantic prior for each sample through MoE-based adaptive visual-semantic interaction. Guided by this prior, Stochastic Multi-view Model Optimization (SMMO) constructs an LLM-generated knowledge base of diverse pathological descriptions per class, then retrieves and stochastically integrates multiple matched textual views during training. These dynamically selected texts serve as enriched semantic supervisions to stochastically optimize the vision-language model, promoting robustness and mitigating overfitting. Experiments on three benchmark WSI datasets show that MUSE consistently outperforms existing vision-language baselines in few-shot settings, demonstrating that effective few-shot pathology learning requires not only richer semantic sources but also their active and sample-aware semantic optimization. Our code is available at: https://github.com/JiahaoXu-god/CVPR2026_MUSE.

  </details>



- **FLIM Networks with Bag of Feature Points**  
  João Deltregia Martinelli, Marcelo Luis Rodrigues Filho, Felipe Crispim da Rocha Salvagnini, Gilson Junior Soares, Jefersson A. dos Santos, Alexandre X. Falcão  
  _2026-02-24_ · https://arxiv.org/abs/2602.20845v1  
  <details><summary>Abstract</summary>

  Convolutional networks require extensive image annotation, which can be costly and time-consuming. Feature Learning from Image Markers (FLIM) tackles this challenge by estimating encoder filters (i.e., kernel weights) from user-drawn markers on discriminative regions of a few representative images without traditional optimization. Such an encoder combined with an adaptive decoder comprises a FLIM network fully trained without backpropagation. Prior research has demonstrated their effectiveness in Salient Object Detection (SOD), being significantly lighter than existing lightweight models. This study revisits FLIM SOD and introduces FLIM-Bag of Feature Points (FLIM-BoFP), a considerably faster filter estimation method. The previous approach, FLIM-Cluster, derives filters through patch clustering at each encoder's block, leading to computational overhead and reduced control over filter locations. FLIM-BoFP streamlines this process by performing a single clustering at the input block, creating a bag of feature points, and defining filters directly from mapped feature points across all blocks. The paper evaluates the benefits in efficiency, effectiveness, and generalization of FLIM-BoFP compared to FLIM-Cluster and other state-of-the-art baselines for parasite detection in optical microscopy images.

  </details>



- **GatedCLIP: Gated Multimodal Fusion for Hateful Memes Detection**  
  Yingying Guo, Ke Zhang, Zirong Zeng  
  _2026-02-24_ · https://arxiv.org/abs/2602.20818v1  
  <details><summary>Abstract</summary>

  Detecting hateful content in multimodal memes presents unique challenges, as harmful messages often emerge from the complex interplay between benign images and text. We propose GatedCLIP, a Vision-Language model that enhances CLIP's multimodal capabilities with specialized architectural improvements for hateful memes detection. Our approach introduces learned projection heads that map CLIP embeddings to a task-optimized semantic space, a dynamic gated fusion mechanism that adaptively weights visual and textual features, and a contrastive learning objective that maintains cross-modal semantic alignment. Experiments on the Hateful Memes dataset demonstrate that GatedCLIP achieves an AUROC of 0.66, substantially outperforming the CLIP baseline (AUROC 0.49) while maintaining computational efficiency with only 350K trainable parameters.

  </details>



- **VGGDrive: Empowering Vision-Language Models with Cross-View Geometric Grounding for Autonomous Driving**  
  Jie Wang, Guang Li, Zhijian Huang, Chenxu Dang, Hangjun Ye, Yahong Han, Long Chen  
  _2026-02-24_ · https://arxiv.org/abs/2602.20794v1  
  <details><summary>Abstract</summary>

  The significance of cross-view 3D geometric modeling capabilities for autonomous driving is self-evident, yet existing Vision-Language Models (VLMs) inherently lack this capability, resulting in their mediocre performance. While some promising approaches attempt to mitigate this by constructing Q&A data for auxiliary training, they still fail to fundamentally equip VLMs with the ability to comprehensively handle diverse evaluation protocols. We thus chart a new course, advocating for the infusion of VLMs with the cross-view geometric grounding of mature 3D foundation models, closing this critical capability gap in autonomous driving. In this spirit, we propose a novel architecture, VGGDrive, which empowers Vision-language models with cross-view Geometric Grounding for autonomous Driving. Concretely, to bridge the cross-view 3D geometric features from the frozen visual 3D model with the VLM's 2D visual features, we introduce a plug-and-play Cross-View 3D Geometric Enabler (CVGE). The CVGE decouples the base VLM architecture and effectively empowers the VLM with 3D features through a hierarchical adaptive injection mechanism. Extensive experiments show that VGGDrive enhances base VLM performance across five autonomous driving benchmarks, including tasks like cross-view risk perception, motion prediction, and trajectory planning. It's our belief that mature 3D foundation models can empower autonomous driving tasks through effective integration, and we hope our initial exploration demonstrates the potential of this paradigm to the autonomous driving community.

  </details>



- **SIMSPINE: A Biomechanics-Aware Simulation Framework for 3D Spine Motion Annotation and Benchmarking**  
  Muhammad Saif Ullah Khan, Didier Stricker  
  _2026-02-24_ · https://arxiv.org/abs/2602.20792v1  
  <details><summary>Abstract</summary>

  Modeling spinal motion is fundamental to understanding human biomechanics, yet remains underexplored in computer vision due to the spine's complex multi-joint kinematics and the lack of large-scale 3D annotations. We present a biomechanics-aware keypoint simulation framework that augments existing human pose datasets with anatomically consistent 3D spinal keypoints derived from musculoskeletal modeling. Using this framework, we create the first open dataset, named SIMSPINE, which provides sparse vertebra-level 3D spinal annotations for natural full-body motions in indoor multi-camera capture without external restraints. With 2.14 million frames, this enables data-driven learning of vertebral kinematics from subtle posture variations and bridges the gap between musculoskeletal simulation and computer vision. In addition, we release pretrained baselines covering fine-tuned 2D detectors, monocular 3D pose lifting models, and multi-view reconstruction pipelines, establishing a unified benchmark for biomechanically valid spine motion estimation. Specifically, our 2D spine baselines improve the state-of-the-art from 0.63 to 0.80 AUC in controlled environments, and from 0.91 to 0.93 AP for in-the-wild spine tracking. Together, the simulation framework and SIMSPINE dataset advance research in vision-based biomechanics, motion analysis, and digital human modeling by enabling reproducible, anatomically grounded 3D spine estimation under natural conditions.

  </details>



- **Onboard-Targeted Segmentation of Straylight in Space Camera Sensors**  
  Riccardo Gallon, Fabian Schiemenz, Alessandra Menicucci, Eberhard Gill  
  _2026-02-24_ · https://arxiv.org/abs/2602.20709v1  
  <details><summary>Abstract</summary>

  This study details an artificial intelligence (AI)-based methodology for the semantic segmentation of space camera faults. Specifically, we address the segmentation of straylight effects induced by solar presence around the camera's Field of View (FoV). Anomalous images are sourced from our published dataset. Our approach emphasizes generalization across diverse flare textures, leveraging pre-training on a public dataset (Flare7k++) including flares in various non-space contexts to mitigate the scarcity of realistic space-specific data. A DeepLabV3 model with MobileNetV3 backbone performs the segmentation task. The model design targets deployment in spacecraft resource-constrained hardware. Finally, based on a proposed interface between our model and the onboard navigation pipeline, we develop custom metrics to assess the model's performance in the system-level context.

  </details>



- **NGL-Prompter: Training-Free Sewing Pattern Estimation from a Single Image**  
  Anna Badalyan, Pratheba Selvaraju, Giorgio Becherini, Omid Taheri, Victoria Fernandez Abrevaya, Michael Black  
  _2026-02-24_ · https://arxiv.org/abs/2602.20700v1  
  <details><summary>Abstract</summary>

  Estimating sewing patterns from images is a practical approach for creating high-quality 3D garments. Due to the lack of real-world pattern-image paired data, prior approaches fine-tune large vision language models (VLMs) on synthetic garment datasets generated by randomly sampling from a parametric garment model GarmentCode. However, these methods often struggle to generalize to in-the-wild images, fail to capture real-world correlations between garment parts, and are typically restricted to single-layer outfits. In contrast, we observe that VLMs are effective at describing garments in natural language, yet perform poorly when asked to directly regress GarmentCode parameters from images. To bridge this gap, we propose NGL (Natural Garment Language), a novel intermediate language that restructures GarmentCode into a representation more understandable to language models. Leveraging this language, we introduce NGL-Prompter, a training-free pipeline that queries large VLMs to extract structured garment parameters, which are then deterministically mapped to valid GarmentCode. We evaluate our method on the Dress4D, CloSe and a newly collected dataset of approximately 5,000 in-the-wild fashion images. Our approach achieves state-of-the-art performance on standard geometry metrics and is strongly preferred in both human and GPT-based perceptual evaluations compared to existing baselines. Furthermore, NGL-prompter can recover multi-layer outfits whereas competing methods focus mostly on single-layer garments, highlighting its strong generalization to real-world images even with occluded parts. These results demonstrate that accurate sewing pattern reconstruction is possible without costly model training. Our code and data will be released for research use.

  </details>



- **AnimeAgent: Is the Multi-Agent via Image-to-Video models a Good Disney Storytelling Artist?**  
  Hailong Yan, Shice Liu, Tao Wang, Xiangtao Zhang, Yijie Zhong, Jinwei Chen, Le Zhang, Bo Li  
  _2026-02-24_ · https://arxiv.org/abs/2602.20664v1  
  <details><summary>Abstract</summary>

  Custom Storyboard Generation (CSG) aims to produce high-quality, multi-character consistent storytelling. Current approaches based on static diffusion models, whether used in a one-shot manner or within multi-agent frameworks, face three key limitations: (1) Static models lack dynamic expressiveness and often resort to "copy-paste" pattern. (2) One-shot inference cannot iteratively correct missing attributes or poor prompt adherence. (3) Multi-agents rely on non-robust evaluators, ill-suited for assessing stylized, non-realistic animation. To address these, we propose AnimeAgent, the first Image-to-Video (I2V)-based multi-agent framework for CSG. Inspired by Disney's "Combination of Straight Ahead and Pose to Pose" workflow, AnimeAgent leverages I2V's implicit motion prior to enhance consistency and expressiveness, while a mixed subjective-objective reviewer enables reliable iterative refinement. We also collect a human-annotated CSG benchmark with ground-truth. Experiments show AnimeAgent achieves SOTA performance in consistency, prompt fidelity, and stylization.

  </details>



- **SD4R: Sparse-to-Dense Learning for 3D Object Detection with 4D Radar**  
  Xiaokai Bai, Jiahao Cheng, Songkai Wang, Yixuan Luo, Lianqing Zheng, Xiaohan Zhang, Si-Yuan Cao, Hui-Liang Shen  
  _2026-02-24_ · https://arxiv.org/abs/2602.20653v1  
  <details><summary>Abstract</summary>

  4D radar measurements offer an affordable and weather-robust solution for 3D perception. However, the inherent sparsity and noise of radar point clouds present significant challenges for accurate 3D object detection, underscoring the need for effective and robust point clouds densification. Despite recent progress, existing densification methods often fail to address the extreme sparsity of 4D radar point clouds and exhibit limited robustness when processing scenes with a small number of points. In this paper, we propose SD4R, a novel framework that transforms sparse radar point clouds into dense representations. SD4R begins by utilizing a foreground point generator (FPG) to mitigate noise propagation and produce densified point clouds. Subsequently, a logit-query encoder (LQE) enhances conventional pillarization, resulting in robust feature representations. Through these innovations, our SD4R demonstrates strong capability in both noise reduction and foreground point densification. Extensive experiments conducted on the publicly available View-of-Delft dataset demonstrate that SD4R achieves state-of-the-art performance. Source code is available at https://github.com/lancelot0805/SD4R.

  </details>



- **Dataset Color Quantization: A Training-Oriented Framework for Dataset-Level Compression**  
  Chenyue Yu, Lingao Xiao, Jinhong Deng, Ivor W. Tsang, Yang He  
  _2026-02-24_ · https://arxiv.org/abs/2602.20650v1  
  <details><summary>Abstract</summary>

  Large-scale image datasets are fundamental to deep learning, but their high storage demands pose challenges for deployment in resource-constrained environments. While existing approaches reduce dataset size by discarding samples, they often ignore the significant redundancy within each image -- particularly in the color space. To address this, we propose Dataset Color Quantization (DCQ), a unified framework that compresses visual datasets by reducing color-space redundancy while preserving information crucial for model training. DCQ achieves this by enforcing consistent palette representations across similar images, selectively retaining semantically important colors guided by model perception, and maintaining structural details necessary for effective feature learning. Extensive experiments across CIFAR-10, CIFAR-100, Tiny-ImageNet, and ImageNet-1K show that DCQ significantly improves training performance under aggressive compression, offering a scalable and robust solution for dataset-level storage reduction. Code is available at \href{https://github.com/he-y/Dataset-Color-Quantization}{https://github.com/he-y/Dataset-Color-Quantization}.

  </details>



- **SurgAtt-Tracker: Online Surgical Attention Tracking via Temporal Proposal Reranking and Motion-Aware Refinement**  
  Rulin Zhou, Guankun Wang, An Wang, Yujie Ma, Lixin Ouyang, Bolin Cui, Junyan Li, Chaowei Zhu, Mingyang Li, Ming Chen, et al.  
  _2026-02-24_ · https://arxiv.org/abs/2602.20636v1  
  <details><summary>Abstract</summary>

  Accurate and stable field-of-view (FoV) guidance is critical for safe and efficient minimally invasive surgery, yet existing approaches often conflate visual attention estimation with downstream camera control or rely on direct object-centric assumptions. In this work, we formulate surgical attention tracking as a spatio-temporal learning problem and model surgeon focus as a dense attention heatmap, enabling continuous and interpretable frame-wise FoV guidance. We propose SurgAtt-Tracker, a holistic framework that robustly tracks surgical attention by exploiting temporal coherence through proposal-level reranking and motion-aware refinement, rather than direct regression. To support systematic training and evaluation, we introduce SurgAtt-1.16M, a large-scale benchmark with a clinically grounded annotation protocol that enables comprehensive heatmap-based attention analysis across procedures and institutions. Extensive experiments on multiple surgical datasets demonstrate that SurgAtt-Tracker consistently achieves state-of-the-art performance and strong robustness under occlusion, multi-instrument interference, and cross-domain settings. Beyond attention tracking, our approach provides a frame-wise FoV guidance signal that can directly support downstream robotic FoV planning and automatic camera control.

  </details>



- **Object-Scene-Camera Decomposition and Recomposition for Data-Efficient Monocular 3D Object Detection**  
  Zhaonian Kuang, Rui Ding, Meng Yang, Xinhu Zheng, Gang Hua  
  _2026-02-24_ · https://arxiv.org/abs/2602.20627v1  
  <details><summary>Abstract</summary>

  Monocular 3D object detection (M3OD) is intrinsically ill-posed, hence training a high-performance deep learning based M3OD model requires a humongous amount of labeled data with complicated visual variation from diverse scenes, variety of objects and camera poses.However, we observe that, due to strong human bias, the three independent entities, i.e., object, scene, and camera pose, are always tightly entangled when an image is captured to construct training data. More specifically, specific 3D objects are always captured in particular scenes with fixed camera poses, and hence lacks necessary diversity. Such tight entanglement induces the challenging issues of insufficient utilization and overfitting to uniform training data. To mitigate this, we propose an online object-scene-camera decomposition and recomposition data manipulation scheme to more efficiently exploit the training data. We first fully decompose training images into textured 3D object point models and background scenes in an efficient computation and storage manner. We then continuously recompose new training images in each epoch by inserting the 3D objects into the freespace of the background scenes, and rendering them with perturbed camera poses from textured 3D point representation. In this way, the refreshed training data in all epochs can cover the full spectrum of independent object, scene, and camera pose combinations. This scheme can serve as a plug-and-play component to boost M3OD models, working flexibly with both fully and sparsely supervised settings. In the sparsely-supervised setting, objects closest to the ego-camera for all instances are sparsely annotated. We then can flexibly increase the annotated objects to control annotation cost. For validation, our method is widely applied to five representative M3OD models and evaluated on both the KITTI and the more complicated Waymo datasets.

  </details>



- **VAGNet: Grounding 3D Affordance from Human-Object Interactions in Videos**  
  Aihua Mao, Kaihang Huang, Yong-Jin Liu, Chee Seng Chan, Ying He  
  _2026-02-24_ · https://arxiv.org/abs/2602.20608v1  
  <details><summary>Abstract</summary>

  3D object affordance grounding aims to identify regions on 3D objects that support human-object interaction (HOI), a capability essential to embodied visual reasoning. However, most existing approaches rely on static visual or textual cues, neglecting that affordances are inherently defined by dynamic actions. As a result, they often struggle to localize the true contact regions involved in real interactions. We take a different perspective. Humans learn how to use objects by observing and imitating actions, not just by examining shapes. Motivated by this intuition, we introduce video-guided 3D affordance grounding, which leverages dynamic interaction sequences to provide functional supervision. To achieve this, we propose VAGNet, a framework that aligns video-derived interaction cues with 3D structure to resolve ambiguities that static cues cannot address. To support this new setting, we introduce PVAD, the first HOI video-3D pairing affordance dataset, providing functional supervision unavailable in prior works. Extensive experiments on PVAD show that VAGNet achieves state-of-the-art performance, significantly outperforming static-based baselines. The code and dataset will be open publicly.

  </details>



- **An interactive enhanced driving dataset for autonomous driving**  
  Haojie Feng, Peizhi Zhang, Mengjie Tian, Xinrui Zhang, Zhuoren Li, Junpeng Huang, Xiurong Wang, Junfan Zhu, Jianzhou Wang, Dongxiao Yin, et al.  
  _2026-02-24_ · https://arxiv.org/abs/2602.20575v1  
  <details><summary>Abstract</summary>

  The evolution of autonomous driving towards full automation demands robust interactive capabilities; however, the development of Vision-Language-Action (VLA) models is constrained by the sparsity of interactive scenarios and inadequate multimodal alignment in existing data. To this end, this paper proposes the Interactive Enhanced Driving Dataset (IEDD). We develop a scalable pipeline to mine million-level interactive segments from naturalistic driving data based on interactive trajectories, and design metrics to quantify the interaction processes. Furthermore, the IEDD-VQA dataset is constructed by generating synthetic Bird's Eye View (BEV) videos where semantic actions are strictly aligned with structured language. Benchmark results evaluating ten mainstream Vision Language Models (VLMs) are provided to demonstrate the dataset's reuse value in assessing and fine-tuning the reasoning capabilities of autonomous driving models.

  </details>



- **AIForge-Doc: A Benchmark for Detecting AI-Forged Tampering in Financial and Form Documents**  
  Jiaqi Wu, Yuchen Zhou, Muduo Xu, Zisheng Liang, Simiao Ren, Jiayu Xue, Meige Yang, Siying Chen, Jingheng Huan  
  _2026-02-24_ · https://arxiv.org/abs/2602.20569v1  
  <details><summary>Abstract</summary>

  We present AIForge-Doc, the first dedicated benchmark targeting exclusively diffusion-model-based inpainting in financial and form documents with pixel-level annotation. Existing document forgery datasets rely on traditional digital editing tools (e.g., Adobe Photoshop, GIMP), creating a critical gap: state-of-the-art detectors are blind to the rapidly growing threat of AI-forged document fraud. AIForge-Doc addresses this gap by systematically forging numeric fields in real-world receipt and form images using two AI inpainting APIs -- Gemini 2.5 Flash Image and Ideogram v2 Edit -- yielding 4,061 forged images from four public document datasets (CORD, WildReceipt, SROIE, XFUND) across nine languages, annotated with pixel-precise tampered-region masks in DocTamper-compatible format. We benchmark three representative detectors -- TruFor, DocTamper, and a zero-shot GPT-4o judge -- and find that all existing methods degrade substantially: TruFor achieves AUC=0.751 (zero-shot, out-of-distribution) vs. AUC=0.96 on NIST16; DocTamper achieves AUC=0.563 vs. AUC=0.98 in-distribution, with pixel-level IoU=0.020; GPT-4o achieves only 0.509 -- essentially at chance -- confirming that AI-forged values are indistinguishable to automated detectors and VLMs. These results demonstrate that AIForge-Doc represents a qualitatively new and unsolved challenge for document forensics.

  </details>



- **BFA++: Hierarchical Best-Feature-Aware Token Prune for Multi-View Vision Language Action Model**  
  Haosheng Li, Weixin Mao, Zihan Lan, Hongwei Xiong, Hongan Wang, Chenyang Si, Ziwei Liu, Xiaoming Deng, Hua Chen  
  _2026-02-24_ · https://arxiv.org/abs/2602.20566v1  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models have achieved significant breakthroughs by leveraging Large Vision Language Models (VLMs) to jointly interpret instructions and visual inputs. However, the substantial increase in visual tokens, particularly from multi-view inputs, poses serious challenges to real-time robotic manipulation. Existing acceleration techniques for VLMs, such as token pruning, often result in degraded performance when directly applied to VLA models, as they overlook the relationships between different views and fail to account for the dynamic and task-specific characteristics of robotic operation. To address this, we propose BFA++, a dynamic token pruning framework designed specifically for VLA models. BFA++ introduces a hierarchical pruning strategy guided by two-level importance predictors: an intra-view predictor highlights task-relevant regions within each image to suppress spatial noise, while an inter-view predictor identifies critical camera views throughout different manipulation phases to reduce cross-view redundancy. This design enables efficient token selection while preserving essential visual cues, resulting in improved computational efficiency and higher manipulation success rates. Evaluations on the RoboTwin benchmark and real-world robotic tasks demonstrate that BFA++ consistently outperforms existing methods. BFA++ improves the success rate by about 10% on both the π0 and RDT models, achieving speedup of 1.8X and 1.5X, respectively. Our results highlight that context-sensitive and task-aware token pruning serves as a more effective strategy than full visual processing, enabling faster inference and improved manipulation accuracy in real-world robotic systems.

  </details>



- **WildGHand: Learning Anti-Perturbation Gaussian Hand Avatars from Monocular In-the-Wild Videos**  
  Hanhui Li, Xuan Huang, Wanquan Liu, Yuhao Cheng, Long Chen, Yiqiang Yan, Xiaodan Liang, Chenqiang Gao  
  _2026-02-24_ · https://arxiv.org/abs/2602.20556v1  
  <details><summary>Abstract</summary>

  Despite recent progress in 3D hand reconstruction from monocular videos, most existing methods rely on data captured in well-controlled environments and therefore degrade in real-world settings with severe perturbations, such as hand-object interactions, extreme poses, illumination changes, and motion blur. To tackle these issues, we introduce WildGHand, an optimization-based framework that enables self-adaptive 3D Gaussian splatting on in-the-wild videos and produces high-fidelity hand avatars. WildGHand incorporates two key components: (i) a dynamic perturbation disentanglement module that explicitly represents perturbations as time-varying biases on 3D Gaussian attributes during optimization, and (ii) a perturbation-aware optimization strategy that generates per-frame anisotropic weighted masks to guide optimization. Together, these components allow the framework to identify and suppress perturbations across both spatial and temporal dimensions. We further curate a dataset of monocular hand videos captured under diverse perturbations to benchmark in-the-wild hand avatar reconstruction. Extensive experiments on this dataset and two public datasets demonstrate that WildGHand achieves state-of-the-art performance and substantially improves over its base model across multiple metrics (e.g., up to a $15.8\%$ relative gain in PSNR and a $23.1\%$ relative reduction in LPIPS). Our implementation and dataset are available at https://github.com/XuanHuang0/WildGHand.

  </details>



- **Beyond Human Performance: A Vision-Language Multi-Agent Approach for Quality Control in Pharmaceutical Manufacturing**  
  Subhra Jyoti Mandal, Lara Rachidi, Puneet Jain, Matthieu Duvinage, Sander W. Timmer  
  _2026-02-24_ · https://arxiv.org/abs/2602.20543v1  
  <details><summary>Abstract</summary>

  Colony-forming unit (CFU) detection is critical in pharmaceutical manufacturing, serving as a key component of Environmental Monitoring programs and ensuring compliance with stringent quality standards. Manual counting is labor-intensive and error-prone, while deep learning (DL) approaches, though accurate, remain vulnerable to sample quality variations and artifacts. Building on our earlier CNN-based framework (Beznik et al., 2020), we evaluated YOLOv5, YOLOv7, and YOLOv8 for CFU detection; however, these achieved only 97.08 percent accuracy, insufficient for pharmaceutical-grade requirements. A custom Detectron2 model trained on GSK's dataset of over 50,000 Petri dish images achieved 99 percent detection rate with 2 percent false positives and 0.6 percent false negatives. Despite high validation accuracy, Detectron2 performance degrades on outlier cases including contaminated plates, plastic artifacts, or poor optical clarity. To address this, we developed a multi-agent framework combining DL with vision-language models (VLMs). The VLM agent first classifies plates as valid or invalid. For valid samples, both DL and VLM agents independently estimate colony counts. When predictions align within 5 percent, results are automatically recorded in Postgres and SAP; otherwise, samples are routed for expert review. Expert feedback enables continuous retraining and self-improvement. Initial DL-based automation reduced human verification by 50 percent across vaccine manufacturing sites. With VLM integration, this increased to 85 percent, delivering significant operational savings. The proposed system provides a scalable, auditable, and regulation-ready solution for microbiological quality control, advancing automation in biopharmaceutical production.

  </details>



- **Strategy-Supervised Autonomous Laparoscopic Camera Control via Event-Driven Graph Mining**  
  Keyu Zhou, Peisen Xu, Yahao Wu, Jiming Chen, Gaofeng Li, Shunlei Li  
  _2026-02-24_ · https://arxiv.org/abs/2602.20500v1  
  <details><summary>Abstract</summary>

  Autonomous laparoscopic camera control must maintain a stable and safe surgical view under rapid tool-tissue interactions while remaining interpretable to surgeons. We present a strategy-grounded framework that couples high-level vision-language inference with low-level closed-loop control. Offline, raw surgical videos are parsed into camera-relevant temporal events (e.g., interaction, working-distance deviation, and view-quality degradation) and structured as attributed event graphs. Mining these graphs yields a compact set of reusable camera-handling strategy primitives, which provide structured supervision for learning. Online, a fine-tuned Vision-Language Model (VLM) processes the live laparoscopic view to predict the dominant strategy and discrete image-based motion commands, executed by an IBVS-RCM controller under strict safety constraints; optional speech input enables intuitive human-in-the-loop conditioning. On a surgeon-annotated dataset, event parsing achieves reliable temporal localization (F1-score 0.86), and the mined strategies show strong semantic alignment with expert interpretation (cluster purity 0.81). Extensive ex vivo experiments on silicone phantoms and porcine tissues demonstrate that the proposed system outperforms junior surgeons in standardized camera-handling evaluations, reducing field-of-view centering error by 35.26% and image shaking by 62.33%, while preserving smooth motion and stable working-distance regulation.

  </details>



- **SceMoS: Scene-Aware 3D Human Motion Synthesis by Planning with Geometry-Grounded Tokens**  
  Anindita Ghosh, Vladislav Golyanik, Taku Komura, Philipp Slusallek, Christian Theobalt, Rishabh Dabral  
  _2026-02-24_ · https://arxiv.org/abs/2602.20476v1  
  <details><summary>Abstract</summary>

  Synthesizing text-driven 3D human motion within realistic scenes requires learning both semantic intent ("walk to the couch") and physical feasibility (e.g., avoiding collisions). Current methods use generative frameworks that simultaneously learn high-level planning and low-level contact reasoning, and rely on computationally expensive 3D scene data such as point clouds or voxel occupancy grids. We propose SceMoS, a scene-aware motion synthesis framework that shows that structured 2D scene representations can serve as a powerful alternative to full 3D supervision in physically grounded motion synthesis. SceMoS disentangles global planning from local execution using lightweight 2D cues and relying on (1) a text-conditioned autoregressive global motion planner that operates on a bird's-eye-view (BEV) image rendered from an elevated corner of the scene, encoded with DINOv2 features, as the scene representation, and (2) a geometry-grounded motion tokenizer trained via a conditional VQ-VAE, that uses 2D local scene heightmap, thus embedding surface physics directly into a discrete vocabulary. This 2D factorization reaches an efficiency-fidelity trade-off: BEV semantics capture spatial layout and affordance for global reasoning, while local heightmaps enforce fine-grained physical adherence without full 3D volumetric reasoning. SceMoS achieves state-of-the-art motion realism and contact accuracy on the TRUMANS benchmark, reducing the number of trainable parameters for scene encoding by over 50%, showing that 2D scene cues can effectively ground 3D human-scene interaction.

  </details>



- **gQIR: Generative Quanta Image Reconstruction**  
  Aryan Garg, Sizhuo Ma, Mohit Gupta  
  _2026-02-23_ · https://arxiv.org/abs/2602.20417v1  
  <details><summary>Abstract</summary>

  Capturing high-quality images from only a few detected photons is a fundamental challenge in computational imaging. Single-photon avalanche diode (SPAD) sensors promise high-quality imaging in regimes where conventional cameras fail, but raw \emph{quanta frames} contain only sparse, noisy, binary photon detections. Recovering a coherent image from a burst of such frames requires handling alignment, denoising, and demosaicing (for color) under noise statistics far outside those assumed by standard restoration pipelines or modern generative models. We present an approach that adapts large text-to-image latent diffusion models to the photon-limited domain of quanta burst imaging. Our method leverages the structural and semantic priors of internet-scale diffusion models while introducing mechanisms to handle Bernoulli photon statistics. By integrating latent-space restoration with burst-level spatio-temporal reasoning, our approach produces reconstructions that are both photometrically faithful and perceptually pleasing, even under high-speed motion. We evaluate the method on synthetic benchmarks and new real-world datasets, including the first color SPAD burst dataset and a challenging \textit{Deforming (XD)} video benchmark. Across all settings, the approach substantially improves perceptual quality over classical and modern learning-based baselines, demonstrating the promise of adapting large generative priors to extreme photon-limited sensing. Code at \href{https://github.com/Aryan-Garg/gQIR}{https://github.com/Aryan-Garg/gQIR}.

  </details>



- **SimLBR: Learning to Detect Fake Images by Learning to Detect Real Images**  
  Aayush Dhakal, Subash Khanal, Srikumar Sastry, Jacob Arndt, Philipe Ambrozio Dias, Dalton Lunga, Nathan Jacobs  
  _2026-02-23_ · https://arxiv.org/abs/2602.20412v1  
  <details><summary>Abstract</summary>

  The rapid advancement of generative models has made the detection of AI-generated images a critical challenge for both research and society. Recent works have shown that most state-of-the-art fake image detection methods overfit to their training data and catastrophically fail when evaluated on curated hard test sets with strong distribution shifts. In this work, we argue that it is more principled to learn a tight decision boundary around the real image distribution and treat the fake category as a sink class. To this end, we propose SimLBR, a simple and efficient framework for fake image detection using Latent Blending Regularization (LBR). Our method significantly improves cross-generator generalization, achieving up to +24.85\% accuracy and +69.62\% recall on the challenging Chameleon benchmark. SimLBR is also highly efficient, training orders of magnitude faster than existing approaches. Furthermore, we emphasize the need for reliability-oriented evaluation in fake image detection, introducing risk-adjusted metrics and worst-case estimates to better assess model robustness. All code and models will be released on HuggingFace and GitHub.

  </details>



- **Generalizing from References using a Multi-Task Reference and Goal-Driven RL Framework**  
  Jiashun Wang, M. Eva Mungai, He Li, Jean Pierre Sleiman, Jessica Hodgins, Farbod Farshidian  
  _2026-02-23_ · https://arxiv.org/abs/2602.20375v1  
  <details><summary>Abstract</summary>

  Learning agile humanoid behaviors from human motion offers a powerful route to natural, coordinated control, but existing approaches face a persistent trade-off: reference-tracking policies are often brittle outside the demonstration dataset, while purely task-driven Reinforcement Learning (RL) can achieve adaptability at the cost of motion quality. We introduce a unified multi-task RL framework that bridges this gap by treating reference motion as a prior for behavioral shaping rather than a deployment-time constraint. A single goal-conditioned policy is trained jointly on two tasks that share the same observation and action spaces, but differ in their initialization schemes, command spaces, and reward structures: (i) a reference-guided imitation task in which reference trajectories define dense imitation rewards but are not provided as policy inputs, and (ii) a goal-conditioned generalization task in which goals are sampled independently of any reference and where rewards reflect only task success. By co-optimizing these objectives within a shared formulation, the policy acquires structured, human-like motor skills from dense reference supervision while learning to adapt these skills to novel goals and initial conditions. This is achieved without adversarial objectives, explicit trajectory tracking, phase variables, or reference-dependent inference. We evaluate the method on a challenging box-based parkour playground that demands diverse athletic behaviors (e.g., jumping and climbing), and show that the learned controller transfers beyond the reference distribution while preserving motion naturalness. Finally, we demonstrate long-horizon behavior generation by composing multiple learned skills, illustrating the flexibility of the learned polices in complex scenarios.

  </details>



- **Energy-Based Injury Protection Database: Including Shearing Contact Thresholds for Hand and Finger Using Porcine Surrogates**  
  Robin Jeanne Kirschner, Anna Huber, Carina M. Micheler, Dirk Müller, Nader Rajaei, Rainer Burgkart, Sami Haddadin  
  _2026-02-23_ · https://arxiv.org/abs/2602.20362v1  
  <details><summary>Abstract</summary>

  While robotics research continues to propose strategies for collision avoidance in human-robot interaction, the reality of constrained environments and future humanoid systems makes contact inevitable. To mitigate injury risks, energy-constraining control approaches are commonly used, often relying on safety thresholds derived from blunt impact data in EN ISO 10218-2:2025. However, this dataset does not extend to edged or pointed collisions. Without scalable, clinically grounded datasets covering diverse contact scenarios, safety validation remains limited. Previous studies have laid the groundwork by assessing surrogate-based velocity and mass limits across various geometries, focusing on perpendicular impacts. This study expands those datasets by including shearing contact scenarios in unconstrained collisions, revealing that collision angle significantly affects injury outcomes. Notably, unconstrained shearing contacts result in fewer injuries than perpendicular ones. By reevaluating all prior porcine surrogate data, we establish energy thresholds across geometries and contact types, forming the first energy-based Injury Protection Database. This enables the development of meaningful energy-limiting controllers that ensure safety across a wide range of realistic collision events.

  </details>



- **3DSPA: A 3D Semantic Point Autoencoder for Evaluating Video Realism**  
  Bhavik Chandna, Kelsey R. Allen  
  _2026-02-23_ · https://arxiv.org/abs/2602.20354v1  
  <details><summary>Abstract</summary>

  AI video generation is evolving rapidly. For video generators to be useful for applications ranging from robotics to film-making, they must consistently produce realistic videos. However, evaluating the realism of generated videos remains a largely manual process -- requiring human annotation or bespoke evaluation datasets which have restricted scope. Here we develop an automated evaluation framework for video realism which captures both semantics and coherent 3D structure and which does not require access to a reference video. Our method, 3DSPA, is a 3D spatiotemporal point autoencoder which integrates 3D point trajectories, depth cues, and DINO semantic features into a unified representation for video evaluation. 3DSPA models how objects move and what is happening in the scene, enabling robust assessments of realism, temporal consistency, and physical plausibility. Experiments show that 3DSPA reliably identifies videos which violate physical laws, is more sensitive to motion artifacts, and aligns more closely with human judgments of video quality and realism across multiple datasets. Our results demonstrate that enriching trajectory-based representations with 3D semantics offers a stronger foundation for benchmarking generative video models, and implicitly captures physical rule violations. The code and pretrained model weights will be available at https://github.com/TheProParadox/3dspa_code.

  </details>



- **De-rendering, Reasoning, and Repairing Charts with Vision-Language Models**  
  Valentin Bonas, Martin Sinnona, Viviana Siless, Emmanuel Iarussi  
  _2026-02-23_ · https://arxiv.org/abs/2602.20291v1  
  <details><summary>Abstract</summary>

  Data visualizations are central to scientific communication, journalism, and everyday decision-making, yet they are frequently prone to errors that can distort interpretation or mislead audiences. Rule-based visualization linters can flag violations, but they miss context and do not suggest meaningful design changes. Directly querying general-purpose LLMs about visualization quality is unreliable: lacking training to follow visualization design principles, they often produce inconsistent or incorrect feedback. In this work, we introduce a framework that combines chart de-rendering, automated analysis, and iterative improvement to deliver actionable, interpretable feedback on visualization design. Our system reconstructs the structure of a chart from an image, identifies design flaws using vision-language reasoning, and proposes concrete modifications supported by established principles in visualization research. Users can selectively apply these improvements and re-render updated figures, creating a feedback loop that promotes both higher-quality visualizations and the development of visualization literacy. In our evaluation on 1,000 charts from the Chart2Code benchmark, the system generated 10,452 design recommendations, which clustered into 10 coherent categories (e.g., axis formatting, color accessibility, legend consistency). These results highlight the promise of LLM-driven recommendation systems for delivering structured, principle-based feedback on visualization design, opening the door to more intelligent and accessible authoring tools.

  </details>



- **A Very Big Video Reasoning Suite**  
  Maijunxian Wang, Ruisi Wang, Juyi Lin, Ran Ji, Thaddäus Wiedemer, Qingying Gao, Dezhi Luo, Yaoyao Qian, Lianyu Huang, Zelong Hong, et al.  
  _2026-02-23_ · https://arxiv.org/abs/2602.20159v2  
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


