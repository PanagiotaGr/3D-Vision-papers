# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-02-13 07:13 UTC_

Total papers shown: **50**


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



- **ScalSelect: Scalable Training-Free Multimodal Data Selection for Efficient Visual Instruction Tuning**  
  Changti Wu, Jiahuai Mao, Yuzhuo Miao, Shijie Lian, Bin Yu, Xiaopeng Lin, Cong Huang, Lei Zhang, Kai Chen  
  _2026-02-12_ · https://arxiv.org/abs/2602.11636v1  
  <details><summary>Abstract</summary>

  Large-scale Visual Instruction Tuning (VIT) has become a key paradigm for advancing the performance of vision-language models (VLMs) across various multimodal tasks. However, training on the large-scale datasets is computationally expensive and inefficient due to redundancy in the data, which motivates the need for multimodal data selection to improve training efficiency. Existing data selection methods for VIT either require costly training or gradient computation. Training-free alternatives often depend on proxy models or datasets, instruction-agnostic representations, and pairwise similarity with quadratic complexity, limiting scalability and representation fidelity. In this work, we propose ScalSelect, a scalable training-free multimodal data selection method with linear-time complexity with respect to the number of samples, eliminating the need for external models or auxiliary datasets. ScalSelect first constructs sample representations by extracting visual features most attended by instruction tokens in the target VLM, capturing instruction-relevant information. It then identifies samples whose representations best approximate the dominant subspace of the full dataset representations, enabling scalable importance scoring without pairwise comparisons. Extensive experiments across multiple VLMs, datasets, and selection budgets demonstrate that ScalSelect achieves over 97.5% of the performance of training on the full dataset using only 16% of the data, and even outperforms full-data training in some settings. The code is available at \href{https://github.com/ChangtiWu/ScalSelect}{ScalSelect}.

  </details>



- **PLESS: Pseudo-Label Enhancement with Spreading Scribbles for Weakly Supervised Segmentation**  
  Yeva Gabrielyan, Varduhi Yeghiazaryan, Irina Voiculescu  
  _2026-02-12_ · https://arxiv.org/abs/2602.11628v1  
  <details><summary>Abstract</summary>

  Weakly supervised learning with scribble annotations uses sparse user-drawn strokes to indicate segmentation labels on a small subset of pixels. This annotation reduces the cost of dense pixel-wise labeling, but suffers inherently from noisy and incomplete supervision. Recent scribble-based approaches in medical image segmentation address this limitation using pseudo-label-based training; however, the quality of the pseudo-labels remains a key performance limit. We propose PLESS, a generic pseudo-label enhancement strategy which improves reliability and spatial consistency. It builds on a hierarchical partitioning of the image into a hierarchy of spatially coherent regions. PLESS propagates scribble information to refine pseudo-labels within semantically coherent regions. The framework is model-agnostic and easily integrates into existing pseudo-label methods. Experiments on two public cardiac MRI datasets (ACDC and MSCMRseg) across four scribble-supervised algorithms show consistent improvements in segmentation accuracy. Code will be made available on GitHub upon acceptance.

  </details>



- **ReaDy-Go: Real-to-Sim Dynamic 3D Gaussian Splatting Simulation for Environment-Specific Visual Navigation with Moving Obstacles**  
  Seungyeon Yoo, Youngseok Jang, Dabin Kim, Youngsoo Han, Seungwoo Jung, H. Jin Kim  
  _2026-02-12_ · https://arxiv.org/abs/2602.11575v1  
  <details><summary>Abstract</summary>

  Visual navigation models often struggle in real-world dynamic environments due to limited robustness to the sim-to-real gap and the difficulty of training policies tailored to target deployment environments (e.g., households, restaurants, and factories). Although real-to-sim navigation simulation using 3D Gaussian Splatting (GS) can mitigate this gap, prior works have assumed only static scenes or unrealistic dynamic obstacles, despite the importance of safe navigation in dynamic environments. To address these issues, we propose ReaDy-Go, a novel real-to-sim simulation pipeline that synthesizes photorealistic dynamic scenarios for target environments. ReaDy-Go generates photorealistic navigation datasets for dynamic environments by combining a reconstructed static GS scene with dynamic human GS obstacles, and trains policies robust to both the sim-to-real gap and moving obstacles. The pipeline consists of three components: (1) a dynamic GS simulator that integrates scene GS with a human animation module, enabling the insertion of animatable human GS avatars and the synthesis of plausible human motions from 2D trajectories, (2) navigation dataset generation for dynamic environments that leverages the simulator, a robot expert planner designed for dynamic GS representations, and a human planner, and (3) policy learning using the generated datasets. ReaDy-Go outperforms baselines across target environments in both simulation and real-world experiments, demonstrating improved navigation performance even after sim-to-real transfer and in the presence of moving obstacles. Moreover, zero-shot sim-to-real deployment in an unseen environment indicates its generalization potential. Project page: https://syeon-yoo.github.io/ready-go-site/.

  </details>



- **Vascular anatomy-aware self-supervised pre-training for X-ray angiogram analysis**  
  De-Xing Huang, Chaohui Yu, Xiao-Hu Zhou, Tian-Yu Xiang, Qin-Yi Zhang, Mei-Jiang Gui, Rui-Ze Ma, Chen-Yu Wang, Nu-Fang Xiao, Fan Wang, et al.  
  _2026-02-12_ · https://arxiv.org/abs/2602.11536v1  
  <details><summary>Abstract</summary>

  X-ray angiography is the gold standard imaging modality for cardiovascular diseases. However, current deep learning approaches for X-ray angiogram analysis are severely constrained by the scarcity of annotated data. While large-scale self-supervised learning (SSL) has emerged as a promising solution, its potential in this domain remains largely unexplored, primarily due to the lack of effective SSL frameworks and large-scale datasets. To bridge this gap, we introduce a vascular anatomy-aware masked image modeling (VasoMIM) framework that explicitly integrates domain-specific anatomical knowledge. Specifically, VasoMIM comprises two key designs: an anatomy-guided masking strategy and an anatomical consistency loss. The former strategically masks vessel-containing patches to compel the model to learn robust vascular semantics, while the latter preserves structural consistency of vessels between original and reconstructed images, enhancing the discriminability of the learned representations. In conjunction with VasoMIM, we curate XA-170K, the largest X-ray angiogram pre-training dataset to date. We validate VasoMIM on four downstream tasks across six datasets, where it demonstrates superior transferability and achieves state-of-the-art performance compared to existing methods. These findings highlight the significant potential of VasoMIM as a foundation model for advancing a wide range of X-ray angiogram analysis tasks. VasoMIM and XA-170K will be available at https://github.com/Dxhuang-CASIA/XA-SSL.

  </details>



- **How Smart Is Your GUI Agent? A Framework for the Future of Software Interaction**  
  Sidong Feng, Chunyang Chen  
  _2026-02-12_ · https://arxiv.org/abs/2602.11514v1  
  <details><summary>Abstract</summary>

  GUI agents are rapidly becoming a new interaction to software, allowing people to navigate web, desktop and mobile rather than execute them click by click. Yet ``agent'' is described with radically different degrees of autonomy, obscuring capability, responsibility and risk. We call for conceptual clarity through GUI Agent Autonomy Levels (GAL), a six-level framework that makes autonomy explicit and helps benchmark progress toward trustworthy software interaction.

  </details>



- **Multimodal Fact-Level Attribution for Verifiable Reasoning**  
  David Wan, Han Wang, Ziyang Wang, Elias Stengel-Eskin, Hyunji Lee, Mohit Bansal  
  _2026-02-12_ · https://arxiv.org/abs/2602.11509v1  
  <details><summary>Abstract</summary>

  Multimodal large language models (MLLMs) are increasingly used for real-world tasks involving multi-step reasoning and long-form generation, where reliability requires grounding model outputs in heterogeneous input sources and verifying individual factual claims. However, existing multimodal grounding benchmarks and evaluation methods focus on simplified, observation-based scenarios or limited modalities and fail to assess attribution in complex multimodal reasoning. We introduce MuRGAt (Multimodal Reasoning with Grounded Attribution), a benchmark for evaluating fact-level multimodal attribution in settings that require reasoning beyond direct observation. Given inputs spanning video, audio, and other modalities, MuRGAt requires models to generate answers with explicit reasoning and precise citations, where each citation specifies both modality and temporal segments. To enable reliable assessment, we introduce an automatic evaluation framework that strongly correlates with human judgments. Benchmarking with human and automated scores reveals that even strong MLLMs frequently hallucinate citations despite correct reasoning. Moreover, we observe a key trade-off: increasing reasoning depth or enforcing structured grounding often degrades accuracy, highlighting a significant gap between internal reasoning and verifiable attribution.

  </details>



- **Hierarchical Concept Embedding & Pursuit for Interpretable Image Classification**  
  Nghia Nguyen, Tianjiao Ding, René Vidal  
  _2026-02-11_ · https://arxiv.org/abs/2602.11448v1  
  <details><summary>Abstract</summary>

  Interpretable-by-design models are gaining traction in computer vision because they provide faithful explanations for their predictions. In image classification, these models typically recover human-interpretable concepts from an image and use them for classification. Sparse concept recovery methods leverage the latent space of vision-language models to represent image embeddings as a sparse combination of concept embeddings. However, because such methods ignore the hierarchical structure of concepts, they can produce correct predictions with explanations that are inconsistent with the hierarchy. In this work, we propose Hierarchical Concept Embedding \& Pursuit (HCEP), a framework that induces a hierarchy of concept embeddings in the latent space and uses hierarchical sparse coding to recover the concepts present in an image. Given a hierarchy of semantic concepts, we construct a corresponding hierarchy of concept embeddings and, assuming the correct concepts for an image form a rooted path in the hierarchy, derive desirable conditions for identifying them in the embedded space. We show that hierarchical sparse coding reliably recovers hierarchical concept embeddings, whereas vanilla sparse coding fails. Our experiments on real-world datasets demonstrate that HCEP outperforms baselines in concept precision and recall while maintaining competitive classification accuracy. Moreover, when the number of samples is limited, HCEP achieves superior classification accuracy and concept recovery. These results show that incorporating hierarchical structures into sparse coding yields more reliable and interpretable image classification models.

  </details>



- **Ctrl&Shift: High-Quality Geometry-Aware Object Manipulation in Visual Generation**  
  Penghui Ruan, Bojia Zi, Xianbiao Qi, Youze Huang, Rong Xiao, Pichao Wang, Jiannong Cao, Yuhui Shi  
  _2026-02-11_ · https://arxiv.org/abs/2602.11440v1  
  <details><summary>Abstract</summary>

  Object-level manipulation, relocating or reorienting objects in images or videos while preserving scene realism, is central to film post-production, AR, and creative editing. Yet existing methods struggle to jointly achieve three core goals: background preservation, geometric consistency under viewpoint shifts, and user-controllable transformations. Geometry-based approaches offer precise control but require explicit 3D reconstruction and generalize poorly; diffusion-based methods generalize better but lack fine-grained geometric control. We present Ctrl&Shift, an end-to-end diffusion framework to achieve geometry-consistent object manipulation without explicit 3D representations. Our key insight is to decompose manipulation into two stages, object removal and reference-guided inpainting under explicit camera pose control, and encode both within a unified diffusion process. To enable precise, disentangled control, we design a multi-task, multi-stage training strategy that separates background, identity, and pose signals across tasks. To improve generalization, we introduce a scalable real-world dataset construction pipeline that generates paired image and video samples with estimated relative camera poses. Extensive experiments demonstrate that Ctrl&Shift achieves state-of-the-art results in fidelity, viewpoint consistency, and controllability. To our knowledge, this is the first framework to unify fine-grained geometric control and real-world generalization for object manipulation, without relying on any explicit 3D modeling.

  </details>



- **Exploring Real-Time Super-Resolution: Benchmarking and Fine-Tuning for Streaming Content**  
  Evgeney Bogatyrev, Khaled Abud, Ivan Molodetskikh, Nikita Alutis, Dmitry Vatolin  
  _2026-02-11_ · https://arxiv.org/abs/2602.11339v1  
  <details><summary>Abstract</summary>

  Recent advancements in real-time super-resolution have enabled higher-quality video streaming, yet existing methods struggle with the unique challenges of compressed video content. Commonly used datasets do not accurately reflect the characteristics of streaming media, limiting the relevance of current benchmarks. To address this gap, we introduce a comprehensive dataset - StreamSR - sourced from YouTube, covering a wide range of video genres and resolutions representative of real-world streaming scenarios. We benchmark 11 state-of-the-art real-time super-resolution models to evaluate their performance for the streaming use-case. Furthermore, we propose EfRLFN, an efficient real-time model that integrates Efficient Channel Attention and a hyperbolic tangent activation function - a novel design choice in the context of real-time super-resolution. We extensively optimized the architecture to maximize efficiency and designed a composite loss function that improves training convergence. EfRLFN combines the strengths of existing architectures while improving both visual quality and runtime performance. Finally, we show that fine-tuning other models on our dataset results in significant performance gains that generalize well across various standard benchmarks. We made the dataset, the code, and the benchmark available at https://github.com/EvgeneyBogatyrev/EfRLFN.

  </details>



- **MolmoSpaces: A Large-Scale Open Ecosystem for Robot Navigation and Manipulation**  
  Yejin Kim, Wilbert Pumacay, Omar Rayyan, Max Argus, Winson Han, Eli VanderBilt, Jordi Salvador, Abhay Deshpande, Rose Hendrix, Snehal Jauhri, et al.  
  _2026-02-11_ · https://arxiv.org/abs/2602.11337v1  
  <details><summary>Abstract</summary>

  Deploying robots at scale demands robustness to the long tail of everyday situations. The countless variations in scene layout, object geometry, and task specifications that characterize real environments are vast and underrepresented in existing robot benchmarks. Measuring this level of generalization requires infrastructure at a scale and diversity that physical evaluation alone cannot provide. We introduce MolmoSpaces, a fully open ecosystem to support large-scale benchmarking of robot policies. MolmoSpaces consists of over 230k diverse indoor environments, ranging from handcrafted household scenes to procedurally generated multiroom houses, populated with 130k richly annotated object assets, including 48k manipulable objects with 42M stable grasps. Crucially, these environments are simulator-agnostic, supporting popular options such as MuJoCo, Isaac, and ManiSkill. The ecosystem supports the full spectrum of embodied tasks: static and mobile manipulation, navigation, and multiroom long-horizon tasks requiring coordinated perception, planning, and interaction across entire indoor environments. We also design MolmoSpaces-Bench, a benchmark suite of 8 tasks in which robots interact with our diverse scenes and richly annotated objects. Our experiments show MolmoSpaces-Bench exhibits strong sim-to-real correlation (R = 0.96, \r{ho} = 0.98), confirm newer and stronger zero-shot policies outperform earlier versions in our benchmarks, and identify key sensitivities to prompt phrasing, initial joint positions, and camera occlusion. Through MolmoSpaces and its open-source assets and tooling, we provide a foundation for scalable data generation, policy training, and benchmark creation for robot learning research.

  </details>



- **H-WM: Robotic Task and Motion Planning Guided by Hierarchical World Model**  
  Wenyuan Chen, Jinbang Huang, Oscar Pang, Zhiyuan Li, Xiao Hu, Lingfeng Zhang, Zhanguang Zhang, Mark Coates, Tongtong Cao, Xingyue Quan, et al.  
  _2026-02-11_ · https://arxiv.org/abs/2602.11291v1  
  <details><summary>Abstract</summary>

  World models are becoming central to robotic planning and control, as they enable prediction of future state transitions. Existing approaches often emphasize video generation or natural language prediction, which are difficult to directly ground in robot actions and suffer from compounding errors over long horizons. Traditional task and motion planning relies on symbolic logic world models, such as planning domains, that are robot-executable and robust for long-horizon reasoning. However, these methods typically operate independently of visual perception, preventing synchronized symbolic and perceptual state prediction. We propose a Hierarchical World Model (H-WM) that jointly predicts logical and visual state transitions within a unified bilevel framework. H-WM combines a high-level logical world model with a low-level visual world model, integrating the robot-executable, long-horizon robustness of symbolic reasoning with perceptual grounding from visual observations. The hierarchical outputs provide stable and consistent intermediate guidance for long-horizon tasks, mitigating error accumulation and enabling robust execution across extended task sequences. To train H-WM, we introduce a robotic dataset that aligns robot motion with symbolic states, actions, and visual observations. Experiments across vision-language-action (VLA) control policies demonstrate the effectiveness and generality of the approach.

  </details>



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



- **Stress Tests REVEAL Fragile Temporal and Visual Grounding in Video-Language Models**  
  Sethuraman T, Savya Khosla, Aditi Tiwari, Vidya Ganesh, Rakshana Jayaprakash, Aditya Jain, Vignesh Srinivasakumar, Onkar Kishor Susladkar, Srinidhi Sunkara, Aditya Shanmugham, et al.  
  _2026-02-11_ · https://arxiv.org/abs/2602.11244v1  
  <details><summary>Abstract</summary>

  This work investigates a fundamental question: Do Video-Language Models (VidLMs) robustly account for video content, temporal sequence, and motion? Our investigation shows that, surprisingly, they often do not. We introduce REVEAL{}, a diagnostic benchmark that probes fundamental weaknesses of contemporary VidLMs through five controlled stress tests; assessing temporal expectation bias, reliance on language-only shortcuts, video sycophancy, camera motion sensitivity, and robustness to spatiotemporal occlusion. We test leading open- and closed-source VidLMs and find that these models confidently describe reversed scenes as forward, answer questions while neglecting video content, agree with false claims, struggle with basic camera motion, and fail to aggregate temporal information amidst simple spatiotemporal masking. Humans, on the other hand, succeed at these tasks with ease. Alongside our benchmark, we provide a data pipeline that automatically generates diagnostic examples for our stress tests, enabling broader and more scalable evaluation. We will release our benchmark and code to support future research.

  </details>



- **Toward Reliable Tea Leaf Disease Diagnosis Using Deep Learning Model: Enhancing Robustness With Explainable AI and Adversarial Training**  
  Samanta Ghosh, Jannatul Adan Mahi, Shayan Abrar, Md Parvez Mia, Asaduzzaman Rayhan, Abdul Awal Yasir, Asaduzzaman Hridoy  
  _2026-02-11_ · https://arxiv.org/abs/2602.11239v1  
  <details><summary>Abstract</summary>

  Tea is a valuable asset for the economy of Bangladesh. So, tea cultivation plays an important role to boost the economy. These valuable plants are vulnerable to various kinds of leaf infections which may cause less production and low quality. It is not so easy to detect these diseases manually. It may take time and there could be some errors in the detection.Therefore, the purpose of the study is to develop an automated deep learning model for tea leaf disease classification based on the teaLeafBD dataset so that anyone can detect the diseases more easily and efficiently. There are 5,278 high-resolution images in this dataset. The images are classified into seven categories. Six of them represents various diseases and the rest one represents healthy leaves. The proposed pipeline contains data preprocessing, data splitting, adversarial training, augmentation, model training, evaluation, and comprehension made possible with Explainable AI strategies. DenseNet201 and EfficientNetB3 were employed to perform the classification task. To prepare the model more robustly, we applied adversarial training so it can operate effectively even with noisy or disturbed inputs. In addition, Grad-CAM visualization was executed to analyze the model's predictions by identifying the most influential regions of each image. Our experimental outcomes revealed that EfficientNetB3 achieved the highest classification accuracy of 93%, while DenseNet201 reached 91%. The outcomes prove that the effectiveness of the proposed approach can accurately detect tea leaf diseases and provide a practical solution for advanced agricultural management.

  </details>



- **Chain-of-Look Spatial Reasoning for Dense Surgical Instrument Counting**  
  Rishikesh Bhyri, Brian R Quaranto, Philip J Seger, Kaity Tung, Brendan Fox, Gene Yang, Steven D. Schwaitzberg, Junsong Yuan, Nan Xi, Peter C W Kim  
  _2026-02-11_ · https://arxiv.org/abs/2602.11024v1  
  <details><summary>Abstract</summary>

  Accurate counting of surgical instruments in Operating Rooms (OR) is a critical prerequisite for ensuring patient safety during surgery. Despite recent progress of large visual-language models and agentic AI, accurately counting such instruments remains highly challenging, particularly in dense scenarios where instruments are tightly clustered. To address this problem, we introduce Chain-of-Look, a novel visual reasoning framework that mimics the sequential human counting process by enforcing a structured visual chain, rather than relying on classic object detection which is unordered. This visual chain guides the model to count along a coherent spatial trajectory, improving accuracy in complex scenes. To further enforce the physical plausibility of the visual chain, we introduce the neighboring loss function, which explicitly models the spatial constraints inherent to densely packed surgical instruments. We also present SurgCount-HD, a new dataset comprising 1,464 high-density surgical instrument images. Extensive experiments demonstrate that our method outperforms state-of-the-art approaches for counting (e.g., CountGD, REC) as well as Multimodality Large Language Models (e.g., Qwen, ChatGPT) in the challenging task of dense surgical instrument counting.

  </details>



- **ABot-M0: VLA Foundation Model for Robotic Manipulation with Action Manifold Learning**  
  Yandan Yang, Shuang Zeng, Tong Lin, Xinyuan Chang, Dekang Qi, Junjin Xiao, Haoyun Liu, Ronghan Chen, Yuzhi Chen, Dongjie Huo, et al.  
  _2026-02-11_ · https://arxiv.org/abs/2602.11236v1  
  <details><summary>Abstract</summary>

  Building general-purpose embodied agents across diverse hardware remains a central challenge in robotics, often framed as the ''one-brain, many-forms'' paradigm. Progress is hindered by fragmented data, inconsistent representations, and misaligned training objectives. We present ABot-M0, a framework that builds a systematic data curation pipeline while jointly optimizing model architecture and training strategies, enabling end-to-end transformation of heterogeneous raw data into unified, efficient representations. From six public datasets, we clean, standardize, and balance samples to construct UniACT-dataset, a large-scale dataset with over 6 million trajectories and 9,500 hours of data, covering diverse robot morphologies and task scenarios. Unified pre-training improves knowledge transfer and generalization across platforms and tasks, supporting general-purpose embodied intelligence. To improve action prediction efficiency and stability, we propose the Action Manifold Hypothesis: effective robot actions lie not in the full high-dimensional space but on a low-dimensional, smooth manifold governed by physical laws and task constraints. Based on this, we introduce Action Manifold Learning (AML), which uses a DiT backbone to predict clean, continuous action sequences directly. This shifts learning from denoising to projection onto feasible manifolds, improving decoding speed and policy stability. ABot-M0 supports modular perception via a dual-stream mechanism that integrates VLM semantics with geometric priors and multi-view inputs from plug-and-play 3D modules such as VGGT and Qwen-Image-Edit, enhancing spatial understanding without modifying the backbone and mitigating standard VLM limitations in 3D reasoning. Experiments show components operate independently with additive benefits. We will release all code and pipelines for reproducibility and future research.

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


