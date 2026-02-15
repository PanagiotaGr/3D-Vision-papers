# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-02-15 07:05 UTC_

Total papers shown: **25**


---

- **Scaling Verification Can Be More Effective than Scaling Policy Learning for Vision-Language-Action Alignment**  
  Jacky Kwok, Xilun Zhang, Mengdi Xu, Yuejiang Liu, Azalia Mirhoseini, Chelsea Finn, Marco Pavone  
  _2026-02-12_ · https://arxiv.org/abs/2602.12281v1  
  <details><summary>Abstract</summary>

  The long-standing vision of general-purpose robots hinges on their ability to understand and act upon natural language instructions. Vision-Language-Action (VLA) models have made remarkable progress toward this goal, yet their generated actions can still misalign with the given instructions. In this paper, we investigate test-time verification as a means to shrink the "intention-action gap.'' We first characterize the test-time scaling law for embodied instruction following and demonstrate that jointly scaling the number of rephrased instructions and generated actions greatly increases test-time sample diversity, often recovering correct actions more efficiently than scaling each dimension independently. To capitalize on these scaling laws, we present CoVer, a contrastive verifier for vision-language-action alignment, and show that our architecture scales gracefully with additional computational resources and data. We then introduce "boot-time compute" and a hierarchical verification inference pipeline for VLAs. At deployment, our framework precomputes a diverse set of rephrased instructions from a Vision-Language-Model (VLM), repeatedly generates action candidates for each instruction, and then uses a verifier to select the optimal high-level prompt and low-level action chunks. Compared to scaling policy pre-training on the same data, our verification approach yields 22% gains in-distribution and 13% out-of-distribution on the SIMPLER benchmark, with a further 45% improvement in real-world experiments. On the PolaRiS benchmark, CoVer achieves 14% gains in task progress and 9% in success rate.

  </details>



- **Energy-Aware Spike Budgeting for Continual Learning in Spiking Neural Networks for Neuromorphic Vision**  
  Anika Tabassum Meem, Muntasir Hossain Nadid, Md Zesun Ahmed Mia  
  _2026-02-12_ · https://arxiv.org/abs/2602.12236v1  
  <details><summary>Abstract</summary>

  Neuromorphic vision systems based on spiking neural networks (SNNs) offer ultra-low-power perception for event-based and frame-based cameras, yet catastrophic forgetting remains a critical barrier to deployment in continually evolving environments. Existing continual learning methods, developed primarily for artificial neural networks, seldom jointly optimize accuracy and energy efficiency, with particularly limited exploration on event-based datasets. We propose an energy-aware spike budgeting framework for continual SNN learning that integrates experience replay, learnable leaky integrate-and-fire neuron parameters, and an adaptive spike scheduler to enforce dataset-specific energy constraints during training. Our approach exhibits modality-dependent behavior: on frame-based datasets (MNIST, CIFAR-10), spike budgeting acts as a sparsity-inducing regularizer, improving accuracy while reducing spike rates by up to 47\%; on event-based datasets (DVS-Gesture, N-MNIST, CIFAR-10-DVS), controlled budget relaxation enables accuracy gains up to 17.45 percentage points with minimal computational overhead. Across five benchmarks spanning both modalities, our method demonstrates consistent performance improvements while minimizing dynamic power consumption, advancing the practical viability of continual learning in neuromorphic vision systems.

  </details>



- **LDA-1B: Scaling Latent Dynamics Action Model via Universal Embodied Data Ingestion**  
  Jiangran Lyu, Kai Liu, Xuheng Zhang, Haoran Liao, Yusen Feng, Wenxuan Zhu, Tingrui Shen, Jiayi Chen, Jiazhao Zhang, Yifei Dong, et al.  
  _2026-02-12_ · https://arxiv.org/abs/2602.12215v1  
  <details><summary>Abstract</summary>

  Recent robot foundation models largely rely on large-scale behavior cloning, which imitates expert actions but discards transferable dynamics knowledge embedded in heterogeneous embodied data. While the Unified World Model (UWM) formulation has the potential to leverage such diverse data, existing instantiations struggle to scale to foundation-level due to coarse data usage and fragmented datasets. We introduce LDA-1B, a robot foundation model that scales through universal embodied data ingestion by jointly learning dynamics, policy, and visual forecasting, assigning distinct roles to data of varying quality. To support this regime at scale, we assemble and standardize EI-30k, an embodied interaction dataset comprising over 30k hours of human and robot trajectories in a unified format. Scalable dynamics learning over such heterogeneous data is enabled by prediction in a structured DINO latent space, which avoids redundant pixel-space appearance modeling. Complementing this representation, LDA-1B employs a multi-modal diffusion transformer to handle asynchronous vision and action streams, enabling stable training at the 1B-parameter scale. Experiments in simulation and the real world show LDA-1B outperforms prior methods (e.g., $π_{0.5}$) by up to 21\%, 48\%, and 23\% on contact-rich, dexterous, and long-horizon tasks, respectively. Notably, LDA-1B enables data-efficient fine-tuning, gaining 10\% by leveraging 30\% low-quality trajectories typically harmful and discarded.

  </details>



- **EO-VAE: Towards A Multi-sensor Tokenizer for Earth Observation Data**  
  Nils Lehmann, Yi Wang, Zhitong Xiong, Xiaoxiang Zhu  
  _2026-02-12_ · https://arxiv.org/abs/2602.12177v1  
  <details><summary>Abstract</summary>

  State-of-the-art generative image and video models rely heavily on tokenizers that compress high-dimensional inputs into more efficient latent representations. While this paradigm has revolutionized RGB generation, Earth observation (EO) data presents unique challenges due to diverse sensor specifications and variable spectral channels. We propose EO-VAE, a multi-sensor variational autoencoder designed to serve as a foundational tokenizer for the EO domain. Unlike prior approaches that train separate tokenizers for each modality, EO-VAE utilizes a single model to encode and reconstruct flexible channel combinations via dynamic hypernetworks. Our experiments on the TerraMesh dataset demonstrate that EO-VAE achieves superior reconstruction fidelity compared to the TerraMind tokenizers, establishing a robust baseline for latent generative modeling in remote sensing.

  </details>



- **PosterOmni: Generalized Artistic Poster Creation via Task Distillation and Unified Reward Feedback**  
  Sixiang Chen, Jianyu Lai, Jialin Gao, Hengyu Shi, Zhongying Liu, Tian Ye, Junfeng Luo, Xiaoming Wei, Lei Zhu  
  _2026-02-12_ · https://arxiv.org/abs/2602.12127v1  
  <details><summary>Abstract</summary>

  Image-to-poster generation is a high-demand task requiring not only local adjustments but also high-level design understanding. Models must generate text, layout, style, and visual elements while preserving semantic fidelity and aesthetic coherence. The process spans two regimes: local editing, where ID-driven generation, rescaling, filling, and extending must preserve concrete visual entities; and global creation, where layout- and style-driven tasks rely on understanding abstract design concepts. These intertwined demands make image-to-poster a multi-dimensional process coupling entity-preserving editing with concept-driven creation under image-prompt control. To address these challenges, we propose PosterOmni, a generalized artistic poster creation framework that unlocks the potential of a base edit model for multi-task image-to-poster generation. PosterOmni integrates the two regimes, namely local editing and global creation, within a single system through an efficient data-distillation-reward pipeline: (i) constructing multi-scenario image-to-poster datasets covering six task types across entity-based and concept-based creation; (ii) distilling knowledge between local and global experts for supervised fine-tuning; and (iii) applying unified PosterOmni Reward Feedback to jointly align visual entity-preserving and aesthetic preference across all tasks. Additionally, we establish PosterOmni-Bench, a unified benchmark for evaluating both local editing and global creation. Extensive experiments show that PosterOmni significantly enhances reference adherence, global composition quality, and aesthetic harmony, outperforming all open-source baselines and even surpassing several proprietary systems.

  </details>



- **GigaBrain-0.5M*: a VLA That Learns From World Model-Based Reinforcement Learning**  
  GigaBrain Team, Boyuan Wang, Chaojun Ni, Guan Huang, Guosheng Zhao, Hao Li, Jie Li, Jindi Lv, Jingyu Liu, Lv Feng, et al.  
  _2026-02-12_ · https://arxiv.org/abs/2602.12099v1  
  <details><summary>Abstract</summary>

  Vision-language-action (VLA) models that directly predict multi-step action chunks from current observations face inherent limitations due to constrained scene understanding and weak future anticipation capabilities. In contrast, video world models pre-trained on web-scale video corpora exhibit robust spatiotemporal reasoning and accurate future prediction, making them a natural foundation for enhancing VLA learning. Therefore, we propose \textit{GigaBrain-0.5M*}, a VLA model trained via world model-based reinforcement learning. Built upon \textit{GigaBrain-0.5}, which is pre-trained on over 10,000 hours of robotic manipulation data, whose intermediate version currently ranks first on the international RoboChallenge benchmark. \textit{GigaBrain-0.5M*} further integrates world model-based reinforcement learning via \textit{RAMP} (Reinforcement leArning via world Model-conditioned Policy) to enable robust cross-task adaptation. Empirical results demonstrate that \textit{RAMP} achieves substantial performance gains over the RECAP baseline, yielding improvements of approximately 30\% on challenging tasks including \texttt{Laundry Folding}, \texttt{Box Packing}, and \texttt{Espresso Preparation}. Critically, \textit{GigaBrain-0.5M$^*$} exhibits reliable long-horizon execution, consistently accomplishing complex manipulation tasks without failure as validated by real-world deployment videos on our \href{https://gigabrain05m.github.io}{project page}.

  </details>



- **Can Local Vision-Language Models improve Activity Recognition over Vision Transformers? -- Case Study on Newborn Resuscitation**  
  Enrico Guerriero, Kjersti Engan, Øyvind Meinich-Bache  
  _2026-02-12_ · https://arxiv.org/abs/2602.12002v1  
  <details><summary>Abstract</summary>

  Accurate documentation of newborn resuscitation is essential for quality improvement and adherence to clinical guidelines, yet remains underutilized in practice. Previous work using 3D-CNNs and Vision Transformers (ViT) has shown promising results in detecting key activities from newborn resuscitation videos, but also highlighted the challenges in recognizing such fine-grained activities. This work investigates the potential of generative AI (GenAI) methods to improve activity recognition from such videos. Specifically, we explore the use of local vision-language models (VLMs), combined with large language models (LLMs), and compare them to a supervised TimeSFormer baseline. Using a simulated dataset comprising 13.26 hours of newborn resuscitation videos, we evaluate several zero-shot VLM-based strategies and fine-tuned VLMs with classification heads, including Low-Rank Adaptation (LoRA). Our results suggest that small (local) VLMs struggle with hallucinations, but when fine-tuned with LoRA, the results reach F1 score at 0.91, surpassing the TimeSformer results of 0.70.

  </details>



- **Benchmarking Vision-Language Models for French PDF-to-Markdown Conversion**  
  Bruno Rigal, Victor Dupriez, Alexis Mignon, Ronan Le Hy, Nicolas Mery  
  _2026-02-12_ · https://arxiv.org/abs/2602.11960v1  
  <details><summary>Abstract</summary>

  This report evaluates PDF-to-Markdown conversion using recent Vision-Language Models (VLMs) on challenging French documents. Document parsing is a critical step for Retrieval-Augmented Generation (RAG) pipelines, where transcription and layout errors propagate to downstream retrieval and grounding. Existing benchmarks often emphasize English or Chinese and can over-penalize benign formatting and linearization choices (e.g., line breaks, list segmentation, alternative table renderings) that are largely irrelevant for downstream use. We introduce a French-focused benchmark of difficult pages selected via model-disagreement sampling from a corpus of 60{,}000 documents, covering handwritten forms, complex layouts, dense tables, and graphics-rich pages. Evaluation is performed with unit-test-style checks that target concrete failure modes (text presence, reading order, and local table constraints) combined with category-specific normalization designed to discount presentation-only variance. Across 15 models, we observe substantially higher robustness for the strongest proprietary models on handwriting and forms, while several open-weights systems remain competitive on standard printed layouts.

  </details>



- **Synthesis of Late Gadolinium Enhancement Images via Implicit Neural Representations for Cardiac Scar Segmentation**  
  Soufiane Ben Haddou, Laura Alvarez-Florez, Erik J. Bekkers, Fleur V. Y. Tjong, Ahmad S. Amin, Connie R. Bezzina, Ivana Išgum  
  _2026-02-12_ · https://arxiv.org/abs/2602.11942v1  
  <details><summary>Abstract</summary>

  Late gadolinium enhancement (LGE) imaging is the clinical standard for myocardial scar assessment, but limited annotated datasets hinder the development of automated segmentation methods. We propose a novel framework that synthesises both LGE images and their corresponding segmentation masks using implicit neural representations (INRs) combined with denoising diffusion models. Our approach first trains INRs to capture continuous spatial representations of LGE data and associated myocardium and fibrosis masks. These INRs are then compressed into compact latent embeddings, preserving essential anatomical information. A diffusion model operates on this latent space to generate new representations, which are decoded into synthetic LGE images with anatomically consistent segmentation masks. Experiments on 133 cardiac MRI scans suggest that augmenting training data with 200 synthetic volumes contributes to improved fibrosis segmentation performance, with the Dice score showing an increase from 0.509 to 0.524. Our approach provides an annotation-free method to help mitigate data scarcity.The code for this research is publicly available.

  </details>



- **Robot-DIFT: Distilling Diffusion Features for Geometrically Consistent Visuomotor Control**  
  Yu Deng, Yufeng Jin, Xiaogang Jia, Jiahong Xue, Gerhard Neumann, Georgia Chalvatzaki  
  _2026-02-12_ · https://arxiv.org/abs/2602.11934v1  
  <details><summary>Abstract</summary>

  We hypothesize that a key bottleneck in generalizable robot manipulation is not solely data scale or policy capacity, but a structural mismatch between current visual backbones and the physical requirements of closed-loop control. While state-of-the-art vision encoders (including those used in VLAs) optimize for semantic invariance to stabilize classification, manipulation typically demands geometric sensitivity the ability to map millimeter-level pose shifts to predictable feature changes. Their discriminative objective creates a "blind spot" for fine-grained control, whereas generative diffusion models inherently encode geometric dependencies within their latent manifolds, encouraging the preservation of dense multi-scale spatial structure. However, directly deploying stochastic diffusion features for control is hindered by stochastic instability, inference latency, and representation drift during fine-tuning. To bridge this gap, we propose Robot-DIFT, a framework that decouples the source of geometric information from the process of inference via Manifold Distillation. By distilling a frozen diffusion teacher into a deterministic Spatial-Semantic Feature Pyramid Network (S2-FPN), we retain the rich geometric priors of the generative model while ensuring temporal stability, real-time execution, and robustness against drift. Pretrained on the large-scale DROID dataset, Robot-DIFT demonstrates superior geometric consistency and control performance compared to leading discriminative baselines, supporting the view that how a model learns to see dictates how well it can learn to act.

  </details>



- **DynaHOI: Benchmarking Hand-Object Interaction for Dynamic Target**  
  BoCheng Hu, Zhonghan Zhao, Kaiyue Zhou, Hongwei Wang, Gaoang Wang  
  _2026-02-12_ · https://arxiv.org/abs/2602.11919v1  
  <details><summary>Abstract</summary>

  Most existing hand motion generation benchmarks for hand-object interaction (HOI) focus on static objects, leaving dynamic scenarios with moving targets and time-critical coordination largely untested. To address this gap, we introduce the DynaHOI-Gym, a unified online closed-loop platform with parameterized motion generators and rollout-based metrics for dynamic capture evaluation. Built on DynaHOI-Gym, we release DynaHOI-10M, a large-scale benchmark with 10M frames and 180K hand capture trajectories, whose target motions are organized into 8 major categories and 22 fine-grained subcategories. We also provide a simple observe-before-act baseline (ObAct) that integrates short-term observations with the current frame via spatiotemporal attention to predict actions, achieving an 8.1% improvement in location success rate.

  </details>



- **Learning to Manipulate Anything: Revealing Data Scaling Laws in Bounding-Box Guided Policies**  
  Yihao Wu, Jinming Ma, Junbo Tan, Yanzhao Yu, Shoujie Li, Mingliang Zhou, Diyun Xiang, Xueqian Wang  
  _2026-02-12_ · https://arxiv.org/abs/2602.11885v1  
  <details><summary>Abstract</summary>

  Diffusion-based policies show limited generalization in semantic manipulation, posing a key obstacle to the deployment of real-world robots. This limitation arises because relying solely on text instructions is inadequate to direct the policy's attention toward the target object in complex and dynamic environments. To solve this problem, we propose leveraging bounding-box instruction to directly specify target object, and further investigate whether data scaling laws exist in semantic manipulation tasks. Specifically, we design a handheld segmentation device with an automated annotation pipeline, Label-UMI, which enables the efficient collection of demonstration data with semantic labels. We further propose a semantic-motion-decoupled framework that integrates object detection and bounding-box guided diffusion policy to improve generalization and adaptability in semantic manipulation. Throughout extensive real-world experiments on large-scale datasets, we validate the effectiveness of the approach, and reveal a power-law relationship between generalization performance and the number of bounding-box objects. Finally, we summarize an effective data collection strategy for semantic manipulation, which can achieve 85\% success rates across four tasks on both seen and unseen objects. All datasets and code will be released to the community.

  </details>



- **Zooming without Zooming: Region-to-Image Distillation for Fine-Grained Multimodal Perception**  
  Lai Wei, Liangbo He, Jun Lan, Lingzhong Dong, Yutong Cai, Siyuan Li, Huijia Zhu, Weiqiang Wang, Linghe Kong, Yue Wang, et al.  
  _2026-02-12_ · https://arxiv.org/abs/2602.11858v1  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLMs) excel at broad visual understanding but still struggle with fine-grained perception, where decisive evidence is small and easily overwhelmed by global context. Recent "Thinking-with-Images" methods alleviate this by iteratively zooming in and out regions of interest during inference, but incur high latency due to repeated tool calls and visual re-encoding. To address this, we propose Region-to-Image Distillation, which transforms zooming from an inference-time tool into a training-time primitive, thereby internalizing the benefits of agentic zooming into a single forward pass of an MLLM. In particular, we first zoom in to micro-cropped regions to let strong teacher models generate high-quality VQA data, and then distill this region-grounded supervision back to the full image. After training on such data, the smaller student model improves "single-glance" fine-grained perception without tool use. To rigorously evaluate this capability, we further present ZoomBench, a hybrid-annotated benchmark of 845 VQA data spanning six fine-grained perceptual dimensions, together with a dual-view protocol that quantifies the global--regional "zooming gap". Experiments show that our models achieve leading performance across multiple fine-grained perception benchmarks, and also improve general multimodal cognition on benchmarks such as visual reasoning and GUI agents. We further discuss when "Thinking-with-Images" is necessary versus when its gains can be distilled into a single forward pass. Our code is available at https://github.com/inclusionAI/Zooming-without-Zooming.

  </details>



- **How to Sample High Quality 3D Fractals for Action Recognition Pre-Training?**  
  Marko Putak, Thomas B. Moeslund, Joakim Bruslund Haurum  
  _2026-02-12_ · https://arxiv.org/abs/2602.11810v1  
  <details><summary>Abstract</summary>

  Synthetic datasets are being recognized in the deep learning realm as a valuable alternative to exhaustively labeled real data. One such synthetic data generation method is Formula Driven Supervised Learning (FDSL), which can provide an infinite number of perfectly labeled data through a formula driven approach, such as fractals or contours. FDSL does not have common drawbacks like manual labor, privacy and other ethical concerns. In this work we generate 3D fractals using 3D Iterated Function Systems (IFS) for pre-training an action recognition model. The fractals are temporally transformed to form a video that is used as a pre-training dataset for downstream task of action recognition. We find that standard methods of generating fractals are slow and produce degenerate 3D fractals. Therefore, we systematically explore alternative ways of generating fractals and finds that overly-restrictive approaches, while generating aesthetically pleasing fractals, are detrimental for downstream task performance. We propose a novel method, Targeted Smart Filtering, to address both the generation speed and fractal diversity issue. The method reports roughly 100 times faster sampling speed and achieves superior downstream performance against other 3D fractal filtering methods.

  </details>



- **Code2Worlds: Empowering Coding LLMs for 4D World Generation**  
  Yi Zhang, Yunshuang Wang, Zeyu Zhang, Hao Tang  
  _2026-02-12_ · https://arxiv.org/abs/2602.11757v1  
  <details><summary>Abstract</summary>

  Achieving spatial intelligence requires moving beyond visual plausibility to build world simulators grounded in physical laws. While coding LLMs have advanced static 3D scene generation, extending this paradigm to 4D dynamics remains a critical frontier. This task presents two fundamental challenges: multi-scale context entanglement, where monolithic generation fails to balance local object structures with global environmental layouts; and a semantic-physical execution gap, where open-loop code generation leads to physical hallucinations lacking dynamic fidelity. We introduce Code2Worlds, a framework that formulates 4D generation as language-to-simulation code generation. First, we propose a dual-stream architecture that disentangles retrieval-augmented object generation from hierarchical environmental orchestration. Second, to ensure dynamic fidelity, we establish a physics-aware closed-loop mechanism in which a PostProcess Agent scripts dynamics, coupled with a VLM-Motion Critic that performs self-reflection to iteratively refine simulation code. Evaluations on the Code4D benchmark show Code2Worlds outperforms baselines with a 41% SGS gain and 49% higher Richness, while uniquely generating physics-aware dynamics absent in prior static methods. Code: https://github.com/AIGeeksGroup/Code2Worlds. Website: https://aigeeksgroup.github.io/Code2Worlds.

  </details>



- **STVG-R1: Incentivizing Instance-Level Reasoning and Grounding in Videos via Reinforcement Learning**  
  Xiaowen Zhang, Zhi Gao, Licheng Jiao, Lingling Li, Qing Li  
  _2026-02-12_ · https://arxiv.org/abs/2602.11730v1  
  <details><summary>Abstract</summary>

  In vision-language models (VLMs), misalignment between textual descriptions and visual coordinates often induces hallucinations. This issue becomes particularly severe in dense prediction tasks such as spatial-temporal video grounding (STVG). Prior approaches typically focus on enhancing visual-textual alignment or attaching auxiliary decoders. However, these strategies inevitably introduce additional trainable modules, leading to significant annotation costs and computational overhead. In this work, we propose a novel visual prompting paradigm that avoids the difficult problem of aligning coordinates across modalities. Specifically, we reformulate per-frame coordinate prediction as a compact instance-level identification problem by assigning each object a unique, temporally consistent ID. These IDs are embedded into the video as visual prompts, providing explicit and interpretable inputs to the VLMs. Furthermore, we introduce STVG-R1, the first reinforcement learning framework for STVG, which employs a task-driven reward to jointly optimize temporal accuracy, spatial consistency, and structural format regularization. Extensive experiments on six benchmarks demonstrate the effectiveness of our approach. STVG-R1 surpasses the baseline Qwen2.5-VL-7B by a remarkable margin of 20.9% on m_IoU on the HCSTVG-v2 benchmark, establishing a new state of the art (SOTA). Surprisingly, STVG-R1 also exhibits strong zero-shot generalization to multi-object referring video object segmentation tasks, achieving a SOTA 47.3% J&F on MeViS.

  </details>



- **TG-Field: Geometry-Aware Radiative Gaussian Fields for Tomographic Reconstruction**  
  Yuxiang Zhong, Jun Wei, Chaoqi Chen, Senyou An, Hui Huang  
  _2026-02-12_ · https://arxiv.org/abs/2602.11705v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has revolutionized 3D scene representation with superior efficiency and quality. While recent adaptations for computed tomography (CT) show promise, they struggle with severe artifacts under highly sparse-view projections and dynamic motions. To address these challenges, we propose Tomographic Geometry Field (TG-Field), a geometry-aware Gaussian deformation framework tailored for both static and dynamic CT reconstruction. A multi-resolution hash encoder is employed to capture local spatial priors, regularizing primitive parameters under ultra-sparse settings. We further extend the framework to dynamic reconstruction by introducing time-conditioned representations and a spatiotemporal attention block to adaptively aggregate features, thereby resolving spatiotemporal ambiguities and enforcing temporal coherence. In addition, a motion-flow network models fine-grained respiratory motion to track local anatomical deformations. Extensive experiments on synthetic and real-world datasets demonstrate that TG-Field consistently outperforms existing methods, achieving state-of-the-art reconstruction accuracy under highly sparse-view conditions.

  </details>



- **Semantically Conditioned Diffusion Models for Cerebral DSA Synthesis**  
  Qiwen Xu, David Rügamer, Holger Wenz, Johann Fontana, Nora Meggyeshazi, Andreas Bender, Máté E. Maros  
  _2026-02-12_ · https://arxiv.org/abs/2602.11703v1  
  <details><summary>Abstract</summary>

  Digital subtraction angiography (DSA) plays a central role in the diagnosis and treatment of cerebrovascular disease, yet its invasive nature and high acquisition cost severely limit large-scale data collection and public data sharing. Therefore, we developed a semantically conditioned latent diffusion model (LDM) that synthesizes arterial-phase cerebral DSA frames under explicit control of anatomical circulation (anterior vs.\ posterior) and canonical C-arm positions. We curated a large single-centre DSA dataset of 99,349 frames and trained a conditional LDM using text embeddings that encoded anatomy and acquisition geometry. To assess clinical realism, four medical experts, including two neuroradiologists, one neurosurgeon, and one internal medicine expert, systematically rated 400 synthetic DSA images using a 5-grade Likert scale for evaluating proximal large, medium, and small peripheral vessels. The generated images achieved image-wise overall Likert scores ranging from 3.1 to 3.3, with high inter-rater reliability (ICC(2,k) = 0.80--0.87). Distributional similarity to real DSA frames was supported by a low median Fréchet inception distance (FID) of 15.27. Our results indicate that semantically controlled LDMs can produce realistic synthetic DSAs suitable for downstream algorithm development, research, and training.

  </details>



- **Beyond Pixels: Vector-to-Graph Transformation for Reliable Schematic Auditing**  
  Chengwei Ma, Zhen Tian, Zhou Zhou, Zhixian Xu, Xiaowei Zhu, Xia Hua, Si Shi, F. Richard Yu  
  _2026-02-12_ · https://arxiv.org/abs/2602.11678v1  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLMs) have shown remarkable progress in visual understanding, yet they suffer from a critical limitation: structural blindness. Even state-of-the-art models fail to capture topology and symbolic logic in engineering schematics, as their pixel-driven paradigm discards the explicit vector-defined relations needed for reasoning. To overcome this, we propose a Vector-to-Graph (V2G) pipeline that converts CAD diagrams into property graphs where nodes represent components and edges encode connectivity, making structural dependencies explicit and machine-auditable. On a diagnostic benchmark of electrical compliance checks, V2G yields large accuracy gains across all error categories, while leading MLLMs remain near chance level. These results highlight the systemic inadequacy of pixel-based methods and demonstrate that structure-aware representations provide a reliable path toward practical deployment of multimodal AI in engineering domains. To facilitate further research, we release our benchmark and implementation at https://github.com/gm-embodied/V2G-Audit.

  </details>



- **RI-Mamba: Rotation-Invariant Mamba for Robust Text-to-Shape Retrieval**  
  Khanh Nguyen, Dasith de Silva Edirimuni, Ghulam Mubashar Hassan, Ajmal Mian  
  _2026-02-12_ · https://arxiv.org/abs/2602.11673v1  
  <details><summary>Abstract</summary>

  3D assets have rapidly expanded in quantity and diversity due to the growing popularity of virtual reality and gaming. As a result, text-to-shape retrieval has become essential in facilitating intuitive search within large repositories. However, existing methods require canonical poses and support few object categories, limiting their real-world applicability where objects can belong to diverse classes and appear in random orientations. To address this challenge, we propose RI-Mamba, the first rotation-invariant state-space model for point clouds. RI-Mamba defines global and local reference frames to disentangle pose from geometry and uses Hilbert sorting to construct token sequences with meaningful geometric structure while maintaining rotation invariance. We further introduce a novel strategy to compute orientational embeddings and reintegrate them via feature-wise linear modulation, effectively recovering spatial context and enhancing model expressiveness. Our strategy is inherently compatible with state-space models and operates in linear time. To scale up retrieval, we adopt cross-modal contrastive learning with automated triplet generation, allowing training on diverse datasets without manual annotation. Extensive experiments demonstrate RI-Mamba's superior representational capacity and robustness, achieving state-of-the-art performance on the OmniObject3D benchmark across more than 200 object categories under arbitrary orientations. Our code will be made available at https://github.com/ndkhanh360/RI-Mamba.git.

  </details>



- **U-Net with Hadamard Transform and DCT Latent Spaces for Next-day Wildfire Spread Prediction**  
  Yingyi Luo, Shuaiang Rong, Adam Watts, Ahmet Enis Cetin  
  _2026-02-12_ · https://arxiv.org/abs/2602.11672v1  
  <details><summary>Abstract</summary>

  We developed a lightweight and computationally efficient tool for next-day wildfire spread prediction using multimodal satellite data as input. The deep learning model, which we call Transform Domain Fusion UNet (TD-FusionUNet), incorporates trainable Hadamard Transform and Discrete Cosine Transform layers that apply two-dimensional transforms, enabling the network to capture essential "frequency" components in orthogonalized latent spaces. Additionally, we introduce custom preprocessing techniques, including random margin cropping and a Gaussian mixture model, to enrich the representation of the sparse pre-fire masks and enhance the model's generalization capability. The TD-FusionUNet is evaluated on two datasets which are the Next-Day Wildfire Spread dataset released by Google Research in 2023, and WildfireSpreadTS dataset. Our proposed TD-FusionUNet achieves an F1 score of 0.591 with 370k parameters, outperforming the UNet baseline using ResNet18 as the encoder reported in the WildfireSpreadTS dataset while using substantially fewer parameters. These results show that the proposed latent space fusion model balances accuracy and efficiency under a lightweight setting, making it suitable for real time wildfire prediction applications in resource limited environments.

  </details>



- **Egocentric Gaze Estimation via Neck-Mounted Camera**  
  Haoyu Huang, Yoichi Sato  
  _2026-02-12_ · https://arxiv.org/abs/2602.11669v1  
  <details><summary>Abstract</summary>

  This paper introduces neck-mounted view gaze estimation, a new task that estimates user gaze from the neck-mounted camera perspective. Prior work on egocentric gaze estimation, which predicts device wearer's gaze location within the camera's field of view, mainly focuses on head-mounted cameras while alternative viewpoints remain underexplored. To bridge this gap, we collect the first dataset for this task, consisting of approximately 4 hours of video collected from 8 participants during everyday activities. We evaluate a transformer-based gaze estimation model, GLC, on the new dataset and propose two extensions: an auxiliary gaze out-of-bound classification task and a multi-view co-learning approach that jointly trains head-view and neck-view models using a geometry-aware auxiliary loss. Experimental results show that incorporating gaze out-of-bound classification improves performance over standard fine-tuning, while the co-learning approach does not yield gains. We further analyze these results and discuss implications for neck-mounted gaze estimation.

  </details>



- **Clutt3R-Seg: Sparse-view 3D Instance Segmentation for Language-grounded Grasping in Cluttered Scenes**  
  Jeongho Noh, Tai Hyoung Rhee, Eunho Lee, Jeongyun Kim, Sunwoo Lee, Ayoung Kim  
  _2026-02-12_ · https://arxiv.org/abs/2602.11660v1  
  <details><summary>Abstract</summary>

  Reliable 3D instance segmentation is fundamental to language-grounded robotic manipulation. Its critical application lies in cluttered environments, where occlusions, limited viewpoints, and noisy masks degrade perception. To address these challenges, we present Clutt3R-Seg, a zero-shot pipeline for robust 3D instance segmentation for language-grounded grasping in cluttered scenes. Our key idea is to introduce a hierarchical instance tree of semantic cues. Unlike prior approaches that attempt to refine noisy masks, our method leverages them as informative cues: through cross-view grouping and conditional substitution, the tree suppresses over- and under-segmentation, yielding view-consistent masks and robust 3D instances. Each instance is enriched with open-vocabulary semantic embeddings, enabling accurate target selection from natural language instructions. To handle scene changes during multi-stage tasks, we further introduce a consistency-aware update that preserves instance correspondences from only a single post-interaction image, allowing efficient adaptation without rescanning. Clutt3R-Seg is evaluated on both synthetic and real-world datasets, and validated on a real robot. Across all settings, it consistently outperforms state-of-the-art baselines in cluttered and sparse-view scenarios. Even on the most challenging heavy-clutter sequences, Clutt3R-Seg achieves an AP@25 of 61.66, over 2.2x higher than baselines, and with only four input views it surpasses MaskClustering with eight views by more than 2x. The code is available at: https://github.com/jeonghonoh/clutt3r-seg.

  </details>



- **SToRM: Supervised Token Reduction for Multi-modal LLMs toward efficient end-to-end autonomous driving**  
  Seo Hyun Kim, Jin Bok Park, Do Yeon Koo, Ho Gun Park, Il Yong Chun  
  _2026-02-12_ · https://arxiv.org/abs/2602.11656v1  
  <details><summary>Abstract</summary>

  In autonomous driving, end-to-end (E2E) driving systems that predict control commands directly from sensor data have achieved significant advancements. For safe driving in unexpected scenarios, these systems may additionally rely on human interventions such as natural language instructions. Using a multi-modal large language model (MLLM) facilitates human-vehicle interaction and can improve performance in such scenarios. However, this approach requires substantial computational resources due to its reliance on an LLM and numerous visual tokens from sensor inputs, which are limited in autonomous vehicles. Many MLLM studies have explored reducing visual tokens, but often suffer end-task performance degradation compared to using all tokens. To enable efficient E2E driving while maintaining performance comparable to using all tokens, this paper proposes the first Supervised Token Reduction framework for multi-modal LLMs (SToRM). The proposed framework consists of three key elements. First, a lightweight importance predictor with short-term sliding windows estimates token importance scores. Second, a supervised training approach uses an auxiliary path to obtain pseudo-supervision signals from an all-token LLM pass. Third, an anchor-context merging module partitions tokens into anchors and context tokens, and merges context tokens into relevant anchors to reduce redundancy while minimizing information loss. Experiments on the LangAuto benchmark show that SToRM outperforms state-of-the-art E2E driving MLLMs under the same reduced-token budget, maintaining all-token performance while reducing computational cost by up to 30x.

  </details>



- **GR-Diffusion: 3D Gaussian Representation Meets Diffusion in Whole-Body PET Reconstruction**  
  Mengxiao Geng, Zijie Chen, Ran Hong, Bingxuan Li, Qiegen Liu  
  _2026-02-12_ · https://arxiv.org/abs/2602.11653v1  
  <details><summary>Abstract</summary>

  Positron emission tomography (PET) reconstruction is a critical challenge in molecular imaging, often hampered by noise amplification, structural blurring, and detail loss due to sparse sampling and the ill-posed nature of inverse problems. The three-dimensional discrete Gaussian representation (GR), which efficiently encodes 3D scenes using parameterized discrete Gaussian distributions, has shown promise in computer vision. In this work, we pro-pose a novel GR-Diffusion framework that synergistically integrates the geometric priors of GR with the generative power of diffusion models for 3D low-dose whole-body PET reconstruction. GR-Diffusion employs GR to generate a reference 3D PET image from projection data, establishing a physically grounded and structurally explicit benchmark that overcomes the low-pass limitations of conventional point-based or voxel-based methods. This reference image serves as a dual guide during the diffusion process, ensuring both global consistency and local accuracy. Specifically, we employ a hierarchical guidance mechanism based on the GR reference. Fine-grained guidance leverages differences to refine local details, while coarse-grained guidance uses multi-scale difference maps to correct deviations. This strategy allows the diffusion model to sequentially integrate the strong geometric prior from GR and recover sub-voxel information. Experimental results on the UDPET and Clinical datasets with varying dose levels show that GR-Diffusion outperforms state-of-the-art methods in enhancing 3D whole-body PET image quality and preserving physiological details.

  </details>


