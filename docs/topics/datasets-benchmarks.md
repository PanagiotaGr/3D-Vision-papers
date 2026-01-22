# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-01-22 06:52 UTC_

Total papers shown: **50**


---

- **LuxRemix: Lighting Decomposition and Remixing for Indoor Scenes**  
  Ruofan Liang, Norman Müller, Ethan Weber, Duncan Zauss, Nandita Vijaykumar, Peter Kontschieder, Christian Richardt  
  _2026-01-21_ · https://arxiv.org/abs/2601.15283v1  
  <details><summary>Abstract</summary>

  We present a novel approach for interactive light editing in indoor scenes from a single multi-view scene capture. Our method leverages a generative image-based light decomposition model that factorizes complex indoor scene illumination into its constituent light sources. This factorization enables independent manipulation of individual light sources, specifically allowing control over their state (on/off), chromaticity, and intensity. We further introduce multi-view lighting harmonization to ensure consistent propagation of the lighting decomposition across all scene views. This is integrated into a relightable 3D Gaussian splatting representation, providing real-time interactive control over the individual light sources. Our results demonstrate highly photorealistic lighting decomposition and relighting outcomes across diverse indoor scenes. We evaluate our method on both synthetic and real-world datasets and provide a quantitative and qualitative comparison to state-of-the-art techniques. For video results and interactive demos, see https://luxremix.github.io.

  </details>



- **Rethinking Video Generation Model for the Embodied World**  
  Yufan Deng, Zilin Pan, Hongyu Zhang, Xiaojie Li, Ruoqing Hu, Yufei Ding, Yiming Zou, Yan Zeng, Daquan Zhou  
  _2026-01-21_ · https://arxiv.org/abs/2601.15282v1  
  <details><summary>Abstract</summary>

  Video generation models have significantly advanced embodied intelligence, unlocking new possibilities for generating diverse robot data that capture perception, reasoning, and action in the physical world. However, synthesizing high-quality videos that accurately reflect real-world robotic interactions remains challenging, and the lack of a standardized benchmark limits fair comparisons and progress. To address this gap, we introduce a comprehensive robotics benchmark, RBench, designed to evaluate robot-oriented video generation across five task domains and four distinct embodiments. It assesses both task-level correctness and visual fidelity through reproducible sub-metrics, including structural consistency, physical plausibility, and action completeness. Evaluation of 25 representative models highlights significant deficiencies in generating physically realistic robot behaviors. Furthermore, the benchmark achieves a Spearman correlation coefficient of 0.96 with human evaluations, validating its effectiveness. While RBench provides the necessary lens to identify these deficiencies, achieving physical realism requires moving beyond evaluation to address the critical shortage of high-quality training data. Driven by these insights, we introduce a refined four-stage data pipeline, resulting in RoVid-X, the largest open-source robotic dataset for video generation with 4 million annotated video clips, covering thousands of tasks and enriched with comprehensive physical property annotations. Collectively, this synergistic ecosystem of evaluation and data establishes a robust foundation for rigorous assessment and scalable training of video models, accelerating the evolution of embodied AI toward general intelligence.

  </details>



- **DrivIng: A Large-Scale Multimodal Driving Dataset with Full Digital Twin Integration**  
  Dominik Rößle, Xujun Xie, Adithya Mohan, Venkatesh Thirugnana Sambandham, Daniel Cremers, Torsten Schön  
  _2026-01-21_ · https://arxiv.org/abs/2601.15260v1  
  <details><summary>Abstract</summary>

  Perception is a cornerstone of autonomous driving, enabling vehicles to understand their surroundings and make safe, reliable decisions. Developing robust perception algorithms requires large-scale, high-quality datasets that cover diverse driving conditions and support thorough evaluation. Existing datasets often lack a high-fidelity digital twin, limiting systematic testing, edge-case simulation, sensor modification, and sim-to-real evaluations. To address this gap, we present DrivIng, a large-scale multimodal dataset with a complete geo-referenced digital twin of a ~18 km route spanning urban, suburban, and highway segments. Our dataset provides continuous recordings from six RGB cameras, one LiDAR, and high-precision ADMA-based localization, captured across day, dusk, and night. All sequences are annotated at 10 Hz with 3D bounding boxes and track IDs across 12 classes, yielding ~1.2 million annotated instances. Alongside the benefits of a digital twin, DrivIng enables a 1-to-1 transfer of real traffic into simulation, preserving agent interactions while enabling realistic and flexible scenario testing. To support reproducible research and robust validation, we benchmark DrivIng with state-of-the-art perception models and publicly release the dataset, digital twin, HD map, and codebase.

  </details>



- **PROGRESSLM: Towards Progress Reasoning in Vision-Language Models**  
  Jianshu Zhang, Chengxuan Qian, Haosen Sun, Haoran Lu, Dingcheng Wang, Letian Xue, Han Liu  
  _2026-01-21_ · https://arxiv.org/abs/2601.15224v1  
  <details><summary>Abstract</summary>

  Estimating task progress requires reasoning over long-horizon dynamics rather than recognizing static visual content. While modern Vision-Language Models (VLMs) excel at describing what is visible, it remains unclear whether they can infer how far a task has progressed from partial observations. To this end, we introduce Progress-Bench, a benchmark for systematically evaluating progress reasoning in VLMs. Beyond benchmarking, we further explore a human-inspired two-stage progress reasoning paradigm through both training-free prompting and training-based approach based on curated dataset ProgressLM-45K. Experiments on 14 VLMs show that most models are not yet ready for task progress estimation, exhibiting sensitivity to demonstration modality and viewpoint changes, as well as poor handling of unanswerable cases. While training-free prompting that enforces structured progress reasoning yields limited and model-dependent gains, the training-based ProgressLM-3B achieves consistent improvements even at a small model scale, despite being trained on a task set fully disjoint from the evaluation tasks. Further analyses reveal characteristic error patterns and clarify when and why progress reasoning succeeds or fails.

  </details>



- **ScenDi: 3D-to-2D Scene Diffusion Cascades for Urban Generation**  
  Hanlei Guo, Jiahao Shao, Xinya Chen, Xiyang Tan, Sheng Miao, Yujun Shen, Yiyi Liao  
  _2026-01-21_ · https://arxiv.org/abs/2601.15221v1  
  <details><summary>Abstract</summary>

  Recent advancements in 3D object generation using diffusion models have achieved remarkable success, but generating realistic 3D urban scenes remains challenging. Existing methods relying solely on 3D diffusion models tend to suffer a degradation in appearance details, while those utilizing only 2D diffusion models typically compromise camera controllability. To overcome this limitation, we propose ScenDi, a method for urban scene generation that integrates both 3D and 2D diffusion models. We first train a 3D latent diffusion model to generate 3D Gaussians, enabling the rendering of images at a relatively low resolution. To enable controllable synthesis, this 3DGS generation process can be optionally conditioned by specifying inputs such as 3d bounding boxes, road maps, or text prompts. Then, we train a 2D video diffusion model to enhance appearance details conditioned on rendered images from the 3D Gaussians. By leveraging the coarse 3D scene as guidance for 2D video diffusion, ScenDi generates desired scenes based on input conditions and successfully adheres to accurate camera trajectories. Experiments on two challenging real-world datasets, Waymo and KITTI-360, demonstrate the effectiveness of our approach.

  </details>



- **BBoxMaskPose v2: Expanding Mutual Conditioning to 3D**  
  Miroslav Purkrabek, Constantin Kolomiiets, Jiri Matas  
  _2026-01-21_ · https://arxiv.org/abs/2601.15200v1  
  <details><summary>Abstract</summary>

  Most 2D human pose estimation benchmarks are nearly saturated, with the exception of crowded scenes. We introduce PMPose, a top-down 2D pose estimator that incorporates the probabilistic formulation and the mask-conditioning. PMPose improves crowded pose estimation without sacrificing performance on standard scenes. Building on this, we present BBoxMaskPose v2 (BMPv2) integrating PMPose and an enhanced SAM-based mask refinement module. BMPv2 surpasses state-of-the-art by 1.5 average precision (AP) points on COCO and 6 AP points on OCHuman, becoming the first method to exceed 50 AP on OCHuman. We demonstrate that BMP's 2D prompting of 3D model improves 3D pose estimation in crowded scenes and that advances in 2D pose quality directly benefit 3D estimation. Results on the new OCHuman-Pose dataset show that multi-person performance is more affected by pose prediction accuracy than by detection. The code, models, and data are available on https://MiraPurkrabek.github.io/BBox-Mask-Pose/.

  </details>



- **BayesianVLA: Bayesian Decomposition of Vision Language Action Models via Latent Action Queries**  
  Shijie Lian, Bin Yu, Xiaopeng Lin, Laurence T. Yang, Zhaolong Shen, Changti Wu, Yuzhuo Miao, Cong Huang, Kai Chen  
  _2026-01-21_ · https://arxiv.org/abs/2601.15197v1  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models have shown promise in robot manipulation but often struggle to generalize to new instructions or complex multi-task scenarios. We identify a critical pathology in current training paradigms where goal-driven data collection creates a dataset bias. In such datasets, language instructions are highly predictable from visual observations alone, causing the conditional mutual information between instructions and actions to vanish, a phenomenon we term Information Collapse. Consequently, models degenerate into vision-only policies that ignore language constraints and fail in out-of-distribution (OOD) settings. To address this, we propose BayesianVLA, a novel framework that enforces instruction following via Bayesian decomposition. By introducing learnable Latent Action Queries, we construct a dual-branch architecture to estimate both a vision-only prior $p(a \mid v)$ and a language-conditioned posterior $π(a \mid v, \ell)$. We then optimize the policy to maximize the conditional Pointwise Mutual Information (PMI) between actions and instructions. This objective effectively penalizes the vision shortcut and rewards actions that explicitly explain the language command. Without requiring new data, BayesianVLA significantly improves generalization. Extensive experiments across on SimplerEnv and RoboCasa demonstrate substantial gains, including an 11.3% improvement on the challenging OOD SimplerEnv benchmark, validating the ability of our approach to robustly ground language in action.

  </details>



- **Large-Scale Multidimensional Knowledge Profiling of Scientific Literature**  
  Zhucun Xue, Jiangning Zhang, Juntao Jiang, Jinzhuo Liu, Haoyang He, Teng Hu, Xiaobin Hu, Guangming Yao, Yi Yuan, Yong Liu  
  _2026-01-21_ · https://arxiv.org/abs/2601.15170v1  
  <details><summary>Abstract</summary>

  The rapid expansion of research across machine learning, vision, and language has produced a volume of publications that is increasingly difficult to synthesize. Traditional bibliometric tools rely mainly on metadata and offer limited visibility into the semantic content of papers, making it hard to track how research themes evolve over time or how different areas influence one another. To obtain a clearer picture of recent developments, we compile a unified corpus of more than 100,000 papers from 22 major conferences between 2020 and 2025 and construct a multidimensional profiling pipeline to organize and analyze their textual content. By combining topic clustering, LLM-assisted parsing, and structured retrieval, we derive a comprehensive representation of research activity that supports the study of topic lifecycles, methodological transitions, dataset and model usage patterns, and institutional research directions. Our analysis highlights several notable shifts, including the growth of safety, multimodal reasoning, and agent-oriented studies, as well as the gradual stabilization of areas such as neural machine translation and graph-based methods. These findings provide an evidence-based view of how AI research is evolving and offer a resource for understanding broader trends and identifying emerging directions. Code and dataset: https://github.com/xzc-zju/Profiling_Scientific_Literature

  </details>



- **BREPS: Bounding-Box Robustness Evaluation of Promptable Segmentation**  
  Andrey Moskalenko, Danil Kuznetsov, Irina Dudko, Anastasiia Iasakova, Nikita Boldyrev, Denis Shepelev, Andrei Spiridonov, Andrey Kuznetsov, Vlad Shakhuro  
  _2026-01-21_ · https://arxiv.org/abs/2601.15123v1  
  <details><summary>Abstract</summary>

  Promptable segmentation models such as SAM have established a powerful paradigm, enabling strong generalization to unseen objects and domains with minimal user input, including points, bounding boxes, and text prompts. Among these, bounding boxes stand out as particularly effective, often outperforming points while significantly reducing annotation costs. However, current training and evaluation protocols typically rely on synthetic prompts generated through simple heuristics, offering limited insight into real-world robustness. In this paper, we investigate the robustness of promptable segmentation models to natural variations in bounding box prompts. First, we conduct a controlled user study and collect thousands of real bounding box annotations. Our analysis reveals substantial variability in segmentation quality across users for the same model and instance, indicating that SAM-like models are highly sensitive to natural prompt noise. Then, since exhaustive testing of all possible user inputs is computationally prohibitive, we reformulate robustness evaluation as a white-box optimization problem over the bounding box prompt space. We introduce BREPS, a method for generating adversarial bounding boxes that minimize or maximize segmentation error while adhering to naturalness constraints. Finally, we benchmark state-of-the-art models across 10 datasets, spanning everyday scenes to medical imaging. Code - https://github.com/emb-ai/BREPS.

  </details>



- **Training-Free and Interpretable Hateful Video Detection via Multi-stage Adversarial Reasoning**  
  Shuonan Yang, Yuchen Zhang, Zeyu Fu  
  _2026-01-21_ · https://arxiv.org/abs/2601.15115v1  
  <details><summary>Abstract</summary>

  Hateful videos pose serious risks by amplifying discrimination, inciting violence, and undermining online safety. Existing training-based hateful video detection methods are constrained by limited training data and lack of interpretability, while directly prompting large vision-language models often struggle to deliver reliable hate detection. To address these challenges, this paper introduces MARS, a training-free Multi-stage Adversarial ReaSoning framework that enables reliable and interpretable hateful content detection. MARS begins with the objective description of video content, establishing a neutral foundation for subsequent analysis. Building on this, it develops evidence-based reasoning that supports potential hateful interpretations, while in parallel incorporating counter-evidence reasoning to capture plausible non-hateful perspectives. Finally, these perspectives are synthesized into a conclusive and explainable decision. Extensive evaluation on two real-world datasets shows that MARS achieves up to 10% improvement under certain backbones and settings compared to other training-free approaches and outperforms state-of-the-art training-based methods on one dataset. In addition, MARS produces human-understandable justifications, thereby supporting compliance oversight and enhancing the transparency of content moderation workflows. The code is available at https://github.com/Multimodal-Intelligence-Lab-MIL/MARS.

  </details>



- **Three-dimensional visualization of X-ray micro-CT with large-scale datasets: Efficiency and accuracy for real-time interaction**  
  Yipeng Yin, Rao Yao, Qingying Li, Dazhong Wang, Hong Zhou, Zhijun Fang, Jianing Chen, Longjie Qian, Mingyue Wu  
  _2026-01-21_ · https://arxiv.org/abs/2601.15098v1  
  <details><summary>Abstract</summary>

  As Micro-CT technology continues to refine its characterization of material microstructures, industrial CT ultra-precision inspection is generating increasingly large datasets, necessitating solutions to the trade-off between accuracy and efficiency in the 3D characterization of defects during ultra-precise detection. This article provides a unique perspective on recent advances in accurate and efficient 3D visualization using Micro-CT, tracing its evolution from medical imaging to industrial non-destructive testing (NDT). Among the numerous CT reconstruction and volume rendering methods, this article selectively reviews and analyzes approaches that balance accuracy and efficiency, offering a comprehensive analysis to help researchers quickly grasp highly efficient and accurate 3D reconstruction methods for microscopic features. By comparing the principles of computed tomography with advancements in microstructural technology, this article examines the evolution of CT reconstruction algorithms from analytical methods to deep learning techniques, as well as improvements in volume rendering algorithms, acceleration, and data reduction. Additionally, it explores advanced lighting models for high-accuracy, photorealistic, and efficient volume rendering. Furthermore, this article envisions potential directions in CT reconstruction and volume rendering. It aims to guide future research in quickly selecting efficient and precise methods and developing new ideas and approaches for real-time online monitoring of internal material defects through virtual-physical interaction, for applying digital twin model to structural health monitoring (SHM).

  </details>



- **The Pictorial Cortex: Zero-Shot Cross-Subject fMRI-to-Image Reconstruction via Compositional Latent Modeling**  
  Jingyang Huo, Yikai Wang, Yanwei Fu, Jianfeng Feng  
  _2026-01-21_ · https://arxiv.org/abs/2601.15071v1  
  <details><summary>Abstract</summary>

  Decoding visual experiences from human brain activity remains a central challenge at the intersection of neuroscience, neuroimaging, and artificial intelligence. A critical obstacle is the inherent variability of cortical responses: neural activity elicited by the same visual stimulus differs across individuals and trials due to anatomical, functional, cognitive, and experimental factors, making fMRI-to-image reconstruction non-injective. In this paper, we tackle a challenging yet practically meaningful problem: zero-shot cross-subject fMRI-to-image reconstruction, where the visual experience of a previously unseen individual must be reconstructed without subject-specific training. To enable principled evaluation, we present a unified cortical-surface dataset -- UniCortex-fMRI, assembled from multiple visual-stimulus fMRI datasets to provide broad coverage of subjects and stimuli. Our UniCortex-fMRI is particularly processed by standardized data formats to make it possible to explore this possibility in the zero-shot scenario of cross-subject fMRI-to-image reconstruction. To tackle the modeling challenge, we propose PictorialCortex, which models fMRI activity using a compositional latent formulation that structures stimulus-driven representations under subject-, dataset-, and trial-related variability. PictorialCortex operates in a universal cortical latent space and implements this formulation through a latent factorization-composition module, reinforced by paired factorization and re-factorizing consistency regularization. During inference, surrogate latents synthesized under multiple seen-subject conditions are aggregated to guide diffusion-based image synthesis for unseen subjects. Extensive experiments show that PictorialCortex improves zero-shot cross-subject visual reconstruction, highlighting the benefits of compositional latent modeling and multi-dataset training.

  </details>



- **SpooFL: Spoofing Federated Learning**  
  Isaac Baglin, Xiatian Zhu, Simon Hadfield  
  _2026-01-21_ · https://arxiv.org/abs/2601.15055v1  
  <details><summary>Abstract</summary>

  Traditional defenses against Deep Leakage (DL) attacks in Federated Learning (FL) primarily focus on obfuscation, introducing noise, transformations or encryption to degrade an attacker's ability to reconstruct private data. While effective to some extent, these methods often still leak high-level information such as class distributions or feature representations, and are frequently broken by increasingly powerful denoising attacks. We propose a fundamentally different perspective on FL defense: framing it as a spoofing problem.We introduce SpooFL (Figure 1), a spoofing-based defense that deceives attackers into believing they have recovered the true training data, while actually providing convincing but entirely synthetic samples from an unrelated task. Unlike prior synthetic-data defenses that share classes or distributions with the private data and thus still leak semantic information, SpooFL uses a state-of-the-art generative model trained on an external dataset with no class overlap. As a result, attackers are misled into recovering plausible yet completely irrelevant samples, preventing meaningful data leakage while preserving FL training integrity. We implement the first example of such a spoofing defense, and evaluate our method against state-of-the-art DL defenses and demonstrate that it successfully misdirects attackers without compromising model performance significantly.

  </details>



- **Federated Transformer-GNN for Privacy-Preserving Brain Tumor Localization with Modality-Level Explainability**  
  Andrea Protani, Riccardo Taiello, Marc Molina Van Den Bosch, Luigi Serio  
  _2026-01-21_ · https://arxiv.org/abs/2601.15042v1  
  <details><summary>Abstract</summary>

  Deep learning models for brain tumor analysis require large and diverse datasets that are often siloed across healthcare institutions due to privacy regulations. We present a federated learning framework for brain tumor localization that enables multi-institutional collaboration without sharing sensitive patient data. Our method extends a hybrid Transformer-Graph Neural Network architecture derived from prior decoder-free supervoxel GNNs and is deployed within CAFEIN\textsuperscript{\textregistered}, CERN's federated learning platform designed for healthcare environments. We provide an explainability analysis through Transformer attention mechanisms that reveals which MRI modalities drive the model predictions. Experiments on the BraTS dataset demonstrate a key finding: while isolated training on individual client data triggers early stopping well before reaching full training capacity, federated learning enables continued model improvement by leveraging distributed data, ultimately matching centralized performance. This result provides strong justification for federated learning when dealing with complex tasks and high-dimensional input data, as aggregating knowledge from multiple institutions significantly benefits the learning process. Our explainability analysis, validated through rigorous statistical testing on the full test set (paired t-tests with Bonferroni correction), reveals that deeper network layers significantly increase attention to T2 and FLAIR modalities ($p<0.001$, Cohen's $d$=1.50), aligning with clinical practice.

  </details>



- **Mixture-of-Experts Models in Vision: Routing, Optimization, and Generalization**  
  Adam Rokah, Daniel Veress, Caleb Caulk, Sourav Sharan  
  _2026-01-21_ · https://arxiv.org/abs/2601.15021v1  
  <details><summary>Abstract</summary>

  Mixture-of-Experts (MoE) architectures enable conditional computation by routing inputs to multiple expert subnetworks and are often motivated as a mechanism for scaling large language models. In this project, we instead study MoE behavior in an image classification setting, focusing on predictive performance, expert utilization, and generalization. We compare dense, SoftMoE, and SparseMoE classifier heads on the CIFAR10 dataset under comparable model capacity. Both MoE variants achieve slightly higher validation accuracy than the dense baseline while maintaining balanced expert utilization through regularization, avoiding expert collapse. To analyze generalization, we compute Hessian-based sharpness metrics at convergence, including the largest eigenvalue and trace of the loss Hessian, evaluated on both training and test data. We find that SoftMoE exhibits higher sharpness by these metrics, while Dense and SparseMoE lie in a similar curvature regime, despite all models achieving comparable generalization performance. Complementary loss surface perturbation analyses reveal qualitative differences in non-local behavior under finite parameter perturbations between dense and MoE models, which help contextualize curvature-based measurements without directly explaining validation accuracy. We further evaluate empirical inference efficiency and show that naively implemented conditional routing does not yield inference speedups on modern hardware at this scale, highlighting the gap between theoretical and realized efficiency in sparse MoE models.

  </details>



- **SpatialV2A: Visual-Guided High-fidelity Spatial Audio Generation**  
  Yanan Wang, Linjie Ren, Zihao Li, Junyi Wang, Tian Gan  
  _2026-01-21_ · https://arxiv.org/abs/2601.15017v1  
  <details><summary>Abstract</summary>

  While video-to-audio generation has achieved remarkable progress in semantic and temporal alignment, most existing studies focus solely on these aspects, paying limited attention to the spatial perception and immersive quality of the synthesized audio. This limitation stems largely from current models' reliance on mono audio datasets, which lack the binaural spatial information needed to learn visual-to-spatial audio mappings. To address this gap, we introduce two key contributions: we construct BinauralVGGSound, the first large-scale video-binaural audio dataset designed to support spatially aware video-to-audio generation; and we propose a end-to-end spatial audio generation framework guided by visual cues, which explicitly models spatial features. Our framework incorporates a visual-guided audio spatialization module that ensures the generated audio exhibits realistic spatial attributes and layered spatial depth while maintaining semantic and temporal alignment. Experiments show that our approach substantially outperforms state-of-the-art models in spatial fidelity and delivers a more immersive auditory experience, without sacrificing temporal or semantic consistency. All datasets, code, and model checkpoints will be publicly released to facilitate future research.

  </details>



- **LiViBench: An Omnimodal Benchmark for Interactive Livestream Video Understanding**  
  Xiaodong Wang, Langling Huang, Zhirong Wu, Xu Zhao, Teng Xu, Xuhong Xia, Peixi Peng  
  _2026-01-21_ · https://arxiv.org/abs/2601.15016v1  
  <details><summary>Abstract</summary>

  The development of multimodal large language models (MLLMs) has advanced general video understanding. However, existing video evaluation benchmarks primarily focus on non-interactive videos, such as movies and recordings. To fill this gap, this paper proposes the first omnimodal benchmark for interactive livestream videos, LiViBench. It features a diverse set of 24 tasks, highlighting the perceptual, reasoning, and livestream-specific challenges. To efficiently construct the dataset, we design a standardized semi-automatic annotation workflow that incorporates the human-in-the-loop at multiple stages. The workflow leverages multiple MLLMs to form a multi-agent system for comprehensive video description and uses a seed-question-driven method to construct high-quality annotations. All interactive videos in the benchmark include audio, speech, and real-time comments modalities. To enhance models' understanding of interactive videos, we design tailored two-stage instruction-tuning and propose a Video-to-Comment Retrieval (VCR) module to improve the model's ability to utilize real-time comments. Based on these advancements, we develop LiVi-LLM-7B, an MLLM with enhanced knowledge of interactive livestreams. Experiments show that our model outperforms larger open-source models with up to 72B parameters, narrows the gap with leading proprietary models on LiViBench, and achieves enhanced performance on general video benchmarks, including VideoMME, LongVideoBench, MLVU, and VideoEval-Pro.

  </details>



- **Unified Multi-Dataset Training for TBPS**  
  Nilanjana Chatterjee, Sidharatha Garg, A V Subramanyam, Brejesh Lall  
  _2026-01-21_ · https://arxiv.org/abs/2601.14978v1  
  <details><summary>Abstract</summary>

  Text-Based Person Search (TBPS) has seen significant progress with vision-language models (VLMs), yet it remains constrained by limited training data and the fact that VLMs are not inherently pre-trained for pedestrian-centric recognition. Existing TBPS methods therefore rely on dataset-centric fine-tuning to handle distribution shift, resulting in multiple independently trained models for different datasets. While synthetic data can increase the scale needed to fine-tune VLMs, it does not eliminate dataset-specific adaptation. This motivates a fundamental question: can we train a single unified TBPS model across multiple datasets? We show that naive joint training over all datasets remains sub-optimal because current training paradigms do not scale to a large number of unique person identities and are vulnerable to noisy image-text pairs. To address these challenges, we propose Scale-TBPS with two contributions: (i) a noise-aware unified dataset curation strategy that cohesively merges diverse TBPS datasets; and (ii) a scalable discriminative identity learning framework that remains effective under a large number of unique identities. Extensive experiments on CUHK-PEDES, ICFG-PEDES, RSTPReid, IIITD-20K, and UFine6926 demonstrate that a single Scale-TBPS model outperforms dataset-centric optimized models and naive joint training.

  </details>



- **From Observation to Prediction: LSTM for Vehicle Lane Change Forecasting on Highway On/Off-Ramps**  
  Mohamed Abouras, Catherine M. Elias  
  _2026-01-21_ · https://arxiv.org/abs/2601.14848v1  
  <details><summary>Abstract</summary>

  On and off-ramps are understudied road sections even though they introduce a higher level of variation in highway interactions. Predicting vehicles' behavior in these areas can decrease the impact of uncertainty and increase road safety. In this paper, the difference between this Area of Interest (AoI) and a straight highway section is studied. Multi-layered LSTM architecture to train the AoI model with ExiD drone dataset is utilized. In the process, different prediction horizons and different models' workflow are tested. The results show great promise on horizons up to 4 seconds with prediction accuracy starting from about 76% for the AoI and 94% for the general highway scenarios on the maximum horizon.

  </details>



- **Implementing Knowledge Representation and Reasoning with Object Oriented Design**  
  Abdelrhman Bassiouny, Tom Schierenbeck, Sorin Arion, Benjamin Alt, Naren Vasantakumaar, Giang Nguyen, Michael Beetz  
  _2026-01-21_ · https://arxiv.org/abs/2601.14840v1  
  <details><summary>Abstract</summary>

  This paper introduces KRROOD, a framework designed to bridge the integration gap between modern software engineering and Knowledge Representation & Reasoning (KR&R) systems. While Object-Oriented Programming (OOP) is the standard for developing complex applications, existing KR&R frameworks often rely on external ontologies and specialized languages that are difficult to integrate with imperative code. KRROOD addresses this by treating knowledge as a first-class programming abstraction using native class structures, bridging the gap between the logic programming and OOP paradigms. We evaluate the system on the OWL2Bench benchmark and a human-robot task learning scenario. Experimental results show that KRROOD achieves strong performance while supporting the expressive reasoning required for real-world autonomous systems.

  </details>



- **Multimodal system for skin cancer detection**  
  Volodymyr Sydorskyi, Igor Krashenyi, Oleksii Yakubenko  
  _2026-01-21_ · https://arxiv.org/abs/2601.14822v1  
  <details><summary>Abstract</summary>

  Melanoma detection is vital for early diagnosis and effective treatment. While deep learning models on dermoscopic images have shown promise, they require specialized equipment, limiting their use in broader clinical settings. This study introduces a multi-modal melanoma detection system using conventional photo images, making it more accessible and versatile. Our system integrates image data with tabular metadata, such as patient demographics and lesion characteristics, to improve detection accuracy. It employs a multi-modal neural network combining image and metadata processing and supports a two-step model for cases with or without metadata. A three-stage pipeline further refines predictions by boosting algorithms and enhancing performance. To address the challenges of a highly imbalanced dataset, specific techniques were implemented to ensure robust training. An ablation study evaluated recent vision architectures, boosting algorithms, and loss functions, achieving a peak Partial ROC AUC of 0.18068 (0.2 maximum) and top-15 retrieval sensitivity of 0.78371. Results demonstrate that integrating photo images with metadata in a structured, multi-stage pipeline yields significant performance improvements. This system advances melanoma detection by providing a scalable, equipment-independent solution suitable for diverse healthcare environments, bridging the gap between specialized and general clinical practices.

  </details>



- **Synthetic Data Augmentation for Multi-Task Chinese Porcelain Classification: A Stable Diffusion Approach**  
  Ziyao Ling, Silvia Mirri, Paola Salomoni, Giovanni Delnevo  
  _2026-01-21_ · https://arxiv.org/abs/2601.14791v1  
  <details><summary>Abstract</summary>

  The scarcity of training data presents a fundamental challenge in applying deep learning to archaeological artifact classification, particularly for the rare types of Chinese porcelain. This study investigates whether synthetic images generated through Stable Diffusion with Low-Rank Adaptation (LoRA) can effectively augment limited real datasets for multi-task CNN-based porcelain classification. Using MobileNetV3 with transfer learning, we conducted controlled experiments comparing models trained on pure real data against those trained on mixed real-synthetic datasets (95:5 and 90:10 ratios) across four classification tasks: dynasty, glaze, kiln and type identification. Results demonstrate task-specific benefits: type classification showed the most substantial improvement (5.5\% F1-macro increase with 90:10 ratio), while dynasty and kiln tasks exhibited modest gains (3-4\%), suggesting that synthetic augmentation effectiveness depends on the alignment between generated features and task-relevant visual signatures. Our work contributes practical guidelines for deploying generative AI in archaeological research, demonstrating both the potential and limitations of synthetic data when archaeological authenticity must be balanced with data diversity.

  </details>



- **FunCineForge: A Unified Dataset Toolkit and Model for Zero-Shot Movie Dubbing in Diverse Cinematic Scenes**  
  Jiaxuan Liu, Yang Xiang, Han Zhao, Xiangang Li, Zhenhua Ling  
  _2026-01-21_ · https://arxiv.org/abs/2601.14777v1  
  <details><summary>Abstract</summary>

  Movie dubbing is the task of synthesizing speech from scripts conditioned on video scenes, requiring accurate lip sync, faithful timbre transfer, and proper modeling of character identity and emotion. However, existing methods face two major limitations: (1) high-quality multimodal dubbing datasets are limited in scale, suffer from high word error rates, contain sparse annotations, rely on costly manual labeling, and are restricted to monologue scenes, all of which hinder effective model training; (2) existing dubbing models rely solely on the lip region to learn audio-visual alignment, which limits their applicability to complex live-action cinematic scenes, and exhibit suboptimal performance in lip sync, speech quality, and emotional expressiveness. To address these issues, we propose FunCineForge, which comprises an end-to-end production pipeline for large-scale dubbing datasets and an MLLM-based dubbing model designed for diverse cinematic scenes. Using the pipeline, we construct the first Chinese television dubbing dataset with rich annotations, and demonstrate the high quality of these data. Experiments across monologue, narration, dialogue, and multi-speaker scenes show that our dubbing model consistently outperforms SOTA methods in audio quality, lip sync, timbre transfer, and instruction following. Code and demos are available at https://anonymous.4open.science/w/FunCineForge.

  </details>



- **Using Multi-Instance Learning to Identify Unique Polyps in Colon Capsule Endoscopy Images**  
  Puneet Sharma, Kristian Dalsbø Hindberg, Eibe Frank, Benedicte Schelde-Olesen, Ulrik Deding  
  _2026-01-21_ · https://arxiv.org/abs/2601.14771v1  
  <details><summary>Abstract</summary>

  Identifying unique polyps in colon capsule endoscopy (CCE) images is a critical yet challenging task for medical personnel due to the large volume of images, the cognitive load it creates for clinicians, and the ambiguity in labeling specific frames. This paper formulates this problem as a multi-instance learning (MIL) task, where a query polyp image is compared with a target bag of images to determine uniqueness. We employ a multi-instance verification (MIV) framework that incorporates attention mechanisms, such as variance-excited multi-head attention (VEMA) and distance-based attention (DBA), to enhance the model's ability to extract meaningful representations. Additionally, we investigate the impact of self-supervised learning using SimCLR to generate robust embeddings. Experimental results on a dataset of 1912 polyps from 754 patients demonstrate that attention mechanisms significantly improve performance, with DBA L1 achieving the highest test accuracy of 86.26\% and a test AUC of 0.928 using a ConvNeXt backbone with SimCLR pretraining. This study underscores the potential of MIL and self-supervised learning in advancing automated analysis of Colon Capsule Endoscopy images, with implications for broader medical imaging applications.

  </details>



- **ReinPath: A Multimodal Reinforcement Learning Approach for Pathology**  
  Kangcheng Zhou, Jun Jiang, Qing Zhang, Shuang Zheng, Qingli Li, Shugong Xu  
  _2026-01-21_ · https://arxiv.org/abs/2601.14757v1  
  <details><summary>Abstract</summary>

  Interpretability is significant in computational pathology, leading to the development of multimodal information integration from histopathological image and corresponding text data.However, existing multimodal methods have limited interpretability due to the lack of high-quality dataset that support explicit reasoning and inference and simple reasoning process.To address the above problems, we introduce a novel multimodal pathology large language model with strong reasoning capabilities.To improve the generation of accurate and contextually relevant textual descriptions, we design a semantic reward strategy integrated with group relative policy optimization.We construct a high-quality pathology visual question answering (VQA) dataset, specifically designed to support complex reasoning tasks.Comprehensive experiments conducted on this dataset demonstrate that our method outperforms state-of-the-art methods, even when trained with only 20% of the data.Our method also achieves comparable performance on downstream zero-shot image classification task compared with CLIP.

  </details>



- **SimD3: A Synthetic drone Dataset with Payload and Bird Distractor Modeling for Robust Detection**  
  Ami Pandat, Kanyala Muvva, Punna Rajasekhar, Gopika Vinod, Rohit Shukla  
  _2026-01-21_ · https://arxiv.org/abs/2601.14742v1  
  <details><summary>Abstract</summary>

  Reliable drone detection is challenging due to limited annotated real-world data, large appearance variability, and the presence of visually similar distractors such as birds. To address these challenges, this paper introduces SimD3, a large-scale high-fidelity synthetic dataset designed for robust drone detection in complex aerial environments. Unlike existing synthetic drone datasets, SimD3 explicitly models drones with heterogeneous payloads, incorporates multiple bird species as realistic distractors, and leverages diverse Unreal Engine 5 environments with controlled weather, lighting, and flight trajectories captured using a 360 six-camera rig. Using SimD3, we conduct an extensive experimental evaluation within the YOLOv5 detection framework, including an attention-enhanced variant termed Yolov5m+C3b, where standard bottleneck-based C3 blocks are replaced with C3b modules. Models are evaluated on synthetic data, combined synthetic and real data, and multiple unseen real-world benchmarks to assess robustness and generalization. Experimental results show that SimD3 provides effective supervision for small-object drone detection and that Yolov5m+C3b consistently outperforms the baseline across in-domain and cross-dataset evaluations. These findings highlight the utility of SimD3 for training and benchmarking robust drone detection models under diverse and challenging conditions.

  </details>



- **LookBench: A Live and Holistic Open Benchmark for Fashion Image Retrieval**  
  Chao Gao, Siqiao Xue, Yimin Peng, Jiwen Fu, Tingyi Gu, Shanshan Li, Fan Zhou  
  _2026-01-21_ · https://arxiv.org/abs/2601.14706v1  
  <details><summary>Abstract</summary>

  In this paper, we present LookBench (We use the term "look" to reflect retrieval that mirrors how people shop -- finding the exact item, a close substitute, or a visually consistent alternative.), a live, holistic and challenging benchmark for fashion image retrieval in real e-commerce settings. LookBench includes both recent product images sourced from live websites and AI-generated fashion images, reflecting contemporary trends and use cases. Each test sample is time-stamped and we intend to update the benchmark periodically, enabling contamination-aware evaluation aligned with declared training cutoffs. Grounded in our fine-grained attribute taxonomy, LookBench covers single-item and outfit-level retrieval across. Our experiments reveal that LookBench poses a significant challenge on strong baselines, with many models achieving below $60\%$ Recall@1. Our proprietary model achieves the best performance on LookBench, and we release an open-source counterpart that ranks second, with both models attaining state-of-the-art results on legacy Fashion200K evaluations. LookBench is designed to be updated semi-annually with new test samples and progressively harder task variants, providing a durable measure of progress. We publicly release our leaderboard, dataset, evaluation code, and trained models.

  </details>



- **RegFreeNet: A Registration-Free Network for CBCT-based 3D Dental Implant Planning**  
  Xinquan Yang, Xuguang Li, Mianjie Zheng, Xuefen Liu, Kun Tang, Kian Ming Lim, He Meng, Jianfeng Ren, Linlin Shen  
  _2026-01-21_ · https://arxiv.org/abs/2601.14703v1  
  <details><summary>Abstract</summary>

  As the commercial surgical guide design software usually does not support the export of implant position for pre-implantation data, existing methods have to scan the post-implantation data and map the implant to pre-implantation space to get the label of implant position for training. Such a process is time-consuming and heavily relies on the accuracy of registration algorithm. Moreover, not all hospitals have paired CBCT data, limitting the construction of multi-center dataset. Inspired by the way dentists determine the implant position based on the neighboring tooth texture, we found that even if the implant area is masked, it will not affect the determination of the implant position. Therefore, we propose to mask the implants in the post-implantation data so that any CBCT containing the implants can be used as training data. This paradigm enables us to discard the registration process and makes it possible to construct a large-scale multi-center implant dataset. On this basis, we proposes ImplantFairy, a comprehensive, publicly accessible dental implant dataset with voxel-level 3D annotations of 1622 CBCT data. Furthermore, according to the area variation characteristics of the tooth's spatial structure and the slope information of the implant, we designed a slope-aware implant position prediction network. Specifically, a neighboring distance perception (NDP) module is designed to adaptively extract tooth area variation features, and an implant slope prediction branch assists the network in learning more robust features through additional implant supervision information. Extensive experiments conducted on ImplantFairy and two public dataset demonstrate that the proposed RegFreeNet achieves the state-of-the-art performance.

  </details>



- **AutoDriDM: An Explainable Benchmark for Decision-Making of Vision-Language Models in Autonomous Driving**  
  Zecong Tang, Zixu Wang, Yifei Wang, Weitong Lian, Tianjian Gao, Haoran Li, Tengju Ru, Lingyi Meng, Zhejun Cui, Yichen Zhu, et al.  
  _2026-01-21_ · https://arxiv.org/abs/2601.14702v1  
  <details><summary>Abstract</summary>

  Autonomous driving is a highly challenging domain that requires reliable perception and safe decision-making in complex scenarios. Recent vision-language models (VLMs) demonstrate reasoning and generalization abilities, opening new possibilities for autonomous driving; however, existing benchmarks and metrics overemphasize perceptual competence and fail to adequately assess decision-making processes. In this work, we present AutoDriDM, a decision-centric, progressive benchmark with 6,650 questions across three dimensions - Object, Scene, and Decision. We evaluate mainstream VLMs to delineate the perception-to-decision capability boundary in autonomous driving, and our correlation analysis reveals weak alignment between perception and decision-making performance. We further conduct explainability analyses of models' reasoning processes, identifying key failure modes such as logical reasoning errors, and introduce an analyzer model to automate large-scale annotation. AutoDriDM bridges the gap between perception-centered and decision-centered evaluation, providing guidance toward safer and more reliable VLMs for real-world autonomous driving.

  </details>



- **FeedbackSTS-Det: Sparse Frames-Based Spatio-Temporal Semantic Feedback Network for Infrared Small Target Detection**  
  Yian Huang, Qing Qin, Aji Mao, Xiangyu Qiu, Liang Xu, Xian Zhang, Zhenming Peng  
  _2026-01-21_ · https://arxiv.org/abs/2601.14690v1  
  <details><summary>Abstract</summary>

  Infrared small target detection (ISTD) under complex backgrounds remains a critical yet challenging task, primarily due to the extremely low signal-to-clutter ratio, persistent dynamic interference, and the lack of distinct target features. While multi-frame detection methods leverages temporal cues to improve upon single-frame approaches, existing methods still struggle with inefficient long-range dependency modeling and insufficient robustness. To overcome these issues, we propose a novel scheme for ISTD, realized through a sparse frames-based spatio-temporal semantic feedback network named FeedbackSTS-Det. The core of our approach is a novel spatio-temporal semantic feedback strategy with a closed-loop semantic association mechanism, which consists of paired forward and backward refinement modules that work cooperatively across the encoder and decoder. Moreover, both modules incorporate an embedded sparse semantic module (SSM), which performs structured sparse temporal modeling to capture long-range dependencies with low computational cost. This integrated design facilitates robust implicit inter-frame registration and continuous semantic refinement, effectively suppressing false alarms. Furthermore, our overall procedure maintains a consistent training-inference pipeline, which ensures reliable performance transfer and increases model robustness. Extensive experiments on multiple benchmark datasets confirm the effectiveness of FeedbackSTS-Det. Code and models are available at: https://github.com/IDIP-Lab/FeedbackSTS-Det.

  </details>



- **Mirai: Autoregressive Visual Generation Needs Foresight**  
  Yonghao Yu, Lang Huang, Zerun Wang, Runyi Li, Toshihiko Yamasaki  
  _2026-01-21_ · https://arxiv.org/abs/2601.14671v1  
  <details><summary>Abstract</summary>

  Autoregressive (AR) visual generators model images as sequences of discrete tokens and are trained with next token likelihood. This strict causality supervision optimizes each step only by its immediate next token, which diminishes global coherence and slows convergence. We ask whether foresight, training signals that originate from later tokens, can help AR visual generation. We conduct a series of controlled diagnostics along the injection level, foresight layout, and foresight source axes, unveiling a key insight: aligning foresight to AR models' internal representation on the 2D image grids improves causality modeling. We formulate this insight with Mirai (meaning "future" in Japanese), a general framework that injects future information into AR training with no architecture change and no extra inference overhead: Mirai-E uses explicit foresight from multiple future positions of unidirectional representations, whereas Mirai-I leverages implicit foresight from matched bidirectional representations. Extensive experiments show that Mirai significantly accelerates convergence and improves generation quality. For instance, Mirai can speed up LlamaGen-B's convergence by up to 10$\times$ and reduce the generation FID from 5.34 to 4.34 on the ImageNet class-condition image generation benchmark. Our study highlights that visual autoregressive models need foresight.

  </details>



- **Forest-Chat: Adapting Vision-Language Agents for Interactive Forest Change Analysis**  
  James Brock, Ce Zhang, Nantheera Anantrasirichai  
  _2026-01-21_ · https://arxiv.org/abs/2601.14637v1  
  <details><summary>Abstract</summary>

  The increasing availability of high-resolution satellite imagery, together with advances in deep learning, creates new opportunities for enhancing forest monitoring workflows. Two central challenges in this domain are pixel-level change detection and semantic change interpretation, particularly for complex forest dynamics. While large language models (LLMs) are increasingly adopted for data exploration, their integration with vision-language models (VLMs) for remote sensing image change interpretation (RSICI) remains underexplored, especially beyond urban environments. We introduce Forest-Chat, an LLM-driven agent designed for integrated forest change analysis. The proposed framework enables natural language querying and supports multiple RSICI tasks, including change detection, change captioning, object counting, deforestation percentage estimation, and change reasoning. Forest-Chat builds upon a multi-level change interpretation (MCI) vision-language backbone with LLM-based orchestration, and incorporates zero-shot change detection via a foundation change detection model together with an interactive point-prompt interface to support fine-grained user guidance. To facilitate adaptation and evaluation in forest environments, we introduce the Forest-Change dataset, comprising bi-temporal satellite imagery, pixel-level change masks, and multi-granularity semantic change captions generated through a combination of human annotation and rule-based methods. Experimental results demonstrate that Forest-Chat achieves strong performance on Forest-Change and on LEVIR-MCI-Trees, a tree-focused subset of LEVIR-MCI, for joint change detection and captioning, highlighting the potential of interactive, LLM-driven RSICI systems to improve accessibility, interpretability, and analytical efficiency in forest change analysis.

  </details>



- **Probing Prompt Design for Socially Compliant Robot Navigation with Vision Language Models**  
  Ling Xiao, Toshihiko Yamasaki  
  _2026-01-21_ · https://arxiv.org/abs/2601.14622v1  
  <details><summary>Abstract</summary>

  Language models are increasingly used for social robot navigation, yet existing benchmarks largely overlook principled prompt design for socially compliant behavior. This limitation is particularly relevant in practice, as many systems rely on small vision language models (VLMs) for efficiency. Compared to large language models, small VLMs exhibit weaker decision-making capabilities, making effective prompt design critical for accurate navigation. Inspired by cognitive theories of human learning and motivation, we study prompt design along two dimensions: system guidance (action-focused, reasoning-oriented, and perception-reasoning prompts) and motivational framing, where models compete against humans, other AI systems, or their past selves. Experiments on two socially compliant navigation datasets reveal three key findings. First, for non-finetuned GPT-4o, competition against humans achieves the best performance, while competition against other AI systems performs worst. For finetuned models, competition against the model's past self yields the strongest results, followed by competition against humans, with performance further influenced by coupling effects among prompt design, model choice, and dataset characteristics. Second, inappropriate system prompt design can significantly degrade performance, even compared to direct finetuning. Third, while direct finetuning substantially improves semantic-level metrics such as perception, prediction, and reasoning, it yields limited gains in action accuracy. In contrast, our system prompts produce a disproportionately larger improvement in action accuracy, indicating that the proposed prompt design primarily acts as a decision-level constraint rather than a representational enhancement.

  </details>



- **Learning Consistent Taxonomic Classification through Hierarchical Reasoning**  
  Zhenghong Li, Kecheng Zheng, Haibin Ling  
  _2026-01-21_ · https://arxiv.org/abs/2601.14610v1  
  <details><summary>Abstract</summary>

  While Vision-Language Models (VLMs) excel at visual understanding, they often fail to grasp hierarchical knowledge. This leads to common errors where VLMs misclassify coarser taxonomic levels even when correctly identifying the most specific level (leaf level). Existing approaches largely overlook this issue by failing to model hierarchical reasoning. To address this gap, we propose VL-Taxon, a two-stage, hierarchy-based reasoning framework designed to improve both leaf-level accuracy and hierarchical consistency in taxonomic classification. The first stage employs a top-down process to enhance leaf-level classification accuracy. The second stage then leverages this accurate leaf-level output to ensure consistency throughout the entire taxonomic hierarchy. Each stage is initially trained with supervised fine-tuning to instill taxonomy knowledge, followed by reinforcement learning to refine the model's reasoning and generalization capabilities. Extensive experiments reveal a remarkable result: our VL-Taxon framework, implemented on the Qwen2.5-VL-7B model, outperforms its original 72B counterpart by over 10% in both leaf-level and hierarchical consistency accuracy on average on the iNaturalist-2021 dataset. Notably, this significant gain was achieved by fine-tuning on just a small subset of data, without relying on any examples generated by other VLMs.

  </details>



- **U-Harmony: Enhancing Joint Training for Segmentation Models with Universal Harmonization**  
  Weiwei Ma, Xiaobing Yu, Peijie Qiu, Jin Yang, Pan Xiao, Xiaoqi Zhao, Xiaofeng Liu, Tomo Miyazaki, Shinichiro Omachi, Yongsong Huang  
  _2026-01-21_ · https://arxiv.org/abs/2601.14605v1  
  <details><summary>Abstract</summary>

  In clinical practice, medical segmentation datasets are often limited and heterogeneous, with variations in modalities, protocols, and anatomical targets across institutions. Existing deep learning models struggle to jointly learn from such diverse data, often sacrificing either generalization or domain-specific knowledge. To overcome these challenges, we propose a joint training method called Universal Harmonization (U-Harmony), which can be integrated into deep learning-based architectures with a domain-gated head, enabling a single segmentation model to learn from heterogeneous datasets simultaneously. By integrating U-Harmony, our approach sequentially normalizes and then denormalizes feature distributions to mitigate domain-specific variations while preserving original dataset-specific knowledge. More appealingly, our framework also supports universal modality adaptation, allowing the seamless learning of new imaging modalities and anatomical classes. Extensive experiments on cross-institutional brain lesion datasets demonstrate the effectiveness of our approach, establishing a new benchmark for robust and adaptable 3D medical image segmentation models in real-world clinical settings.

  </details>



- **LFS: Learnable Frame Selector for Event-Aware and Temporally Diverse Video Captioning**  
  Lianying Chao, Linfeng Yin, Peiyu Ren, Yifan Jiang, Qiaoyu Ren, Dingcheng Shan, Jing-cheng Pang, Sijie Wu, Xubin Li, Kai Zhang  
  _2026-01-21_ · https://arxiv.org/abs/2601.14594v1  
  <details><summary>Abstract</summary>

  Video captioning models convert frames into visual tokens and generate descriptions with large language models (LLMs). Since encoding all frames is prohibitively expensive, uniform sampling is the default choice, but it enforces equal temporal coverage while ignoring the uneven events distribution. This motivates a Learnable Frame Selector (LFS) that selects temporally diverse and event-relevant frames. LFS explicitly models temporal importance to balance temporal diversity and event relevance, and employs a stratified strategy to ensure temporal coverage while avoiding clustering. Crucially, LFS leverages caption feedback from frozen video-LLMs to learn frame selection that directly optimizes downstream caption quality. Additionally, we identify the gap between existing benchmark and human's cognition. Thus, we introduce ICH-CC built from carefully designed questions by annotators that reflect human-consistent understanding of video. Experiments indicate that LFS consistently improves detailed video captioning across two representative community benchmarks and ICH-CC, achieving up to 2.0% gains on VDC and over 4% gains on ICH-CC. Moreover, we observe that enhanced captions with LFS leads to improved performance on video question answering. Overall, LFS provides an effective and easy-to-integrate solution for detailed video captioning.

  </details>



- **From Volumes to Slices: Computationally Efficient Contrastive Learning for Sequential Abdominal CT Analysis**  
  Po-Kai Chiu, Hung-Hsuan Chen  
  _2026-01-21_ · https://arxiv.org/abs/2601.14593v1  
  <details><summary>Abstract</summary>

  The requirement for expert annotations limits the effectiveness of deep learning for medical image analysis. Although 3D self-supervised methods like volume contrast learning (VoCo) are powerful and partially address the labeling scarcity issue, their high computational cost and memory consumption are barriers. We propose 2D-VoCo, an efficient adaptation of the VoCo framework for slice-level self-supervised pre-training that learns spatial-semantic features from unlabeled 2D CT slices via contrastive learning. The pre-trained CNN backbone is then integrated into a CNN-LSTM architecture to classify multi-organ injuries. In the RSNA 2023 Abdominal Trauma dataset, 2D-VoCo pre-training significantly improves mAP, precision, recall, and RSNA score over training from scratch. Our framework provides a practical method to reduce the dependency on labeled data and enhance model performance in clinical CT analysis. We release the code for reproducibility. https://github.com/tkz05/2D-VoCo-CT-Classifier

  </details>



- **Scribble-Supervised Medical Image Segmentation with Dynamic Teacher Switching and Hierarchical Consistency**  
  Thanh-Huy Nguyen, Hoang-Loc Cao, Dat T. Chung, Mai-Anh Vu, Thanh-Minh Nguyen, Minh Le, Phat K. Huynh, Ulas Bagci  
  _2026-01-21_ · https://arxiv.org/abs/2601.14563v1  
  <details><summary>Abstract</summary>

  Scribble-supervised methods have emerged to mitigate the prohibitive annotation burden in medical image segmentation. However, the inherent sparsity of these annotations introduces significant ambiguity, which results in noisy pseudo-label propagation and hinders the learning of robust anatomical boundaries. To address this challenge, we propose SDT-Net, a novel dual-teacher, single-student framework designed to maximize supervision quality from these weak signals. Our method features a Dynamic Teacher Switching (DTS) module to adaptively select the most reliable teacher. This selected teacher then guides the student via two synergistic mechanisms: high-confidence pseudo-labels, refined by a Pick Reliable Pixels (PRP) mechanism, and multi-level feature alignment, enforced by a Hierarchical Consistency (HiCo) module. Extensive experiments on the ACDC and MSCMRseg datasets demonstrate that SDT-Net achieves state-of-the-art performance, producing more accurate and anatomically plausible segmentation.

  </details>



- **GutenOCR: A Grounded Vision-Language Front-End for Documents**  
  Hunter Heidenreich, Ben Elliott, Olivia Dinica, Yosheb Getachew  
  _2026-01-20_ · https://arxiv.org/abs/2601.14490v1  
  <details><summary>Abstract</summary>

  GutenOCR is a family of grounded OCR front-ends obtained by fine-tuning Qwen2.5-VL-3B and Qwen2.5-VL-7B. The resulting single-checkpoint vision-language models expose reading, detection, and grounding through a unified, prompt-based interface. Trained on business documents, scientific articles, and synthetic grounding data, the models support full-page and localized reading with line- and paragraph-level bounding boxes and conditional ``where is x?'' queries. We introduce a grounded OCR evaluation protocol and show that GutenOCR-7B more than doubles the composite grounded OCR score of its Qwen2.5-VL-7B backbone on 10.5K held-out business and scientific pages (0.40 to 0.82). On Fox and OmniDocBench v1.5, our approach substantially improves region- and line-level OCR as well as text-detection recall, but reveals trade-offs in page-level linearization, color-guided OCR, and formula-heavy layouts.

  </details>



- **XD-MAP: Cross-Modal Domain Adaptation using Semantic Parametric Mapping**  
  Frank Bieder, Hendrik Königshof, Haohao Hu, Fabian Immel, Yinzhe Shen, Jan-Hendrik Pauls, Christoph Stiller  
  _2026-01-20_ · https://arxiv.org/abs/2601.14477v1  
  <details><summary>Abstract</summary>

  Until open-world foundation models match the performance of specialized approaches, the effectiveness of deep learning models remains heavily dependent on dataset availability. Training data must align not only with the target object categories but also with the sensor characteristics and modalities. To bridge the gap between available datasets and deployment domains, domain adaptation strategies are widely used. In this work, we propose a novel approach to transferring sensor-specific knowledge from an image dataset to LiDAR, an entirely different sensing domain. Our method XD-MAP leverages detections from a neural network on camera images to create a semantic parametric map. The map elements are modeled to produce pseudo labels in the target domain without any manual annotation effort. Unlike previous domain transfer approaches, our method does not require direct overlap between sensors and enables extending the angular perception range from a front-view camera to a full 360 view. On our large-scale road feature dataset, XD-MAP outperforms single shot baseline approaches by +19.5 mIoU for 2D semantic segmentation, +19.5 PQth for 2D panoptic segmentation, and +32.3 mIoU in 3D semantic segmentation. The results demonstrate the effectiveness of our approach achieving strong performance on LiDAR data without any manual labeling.

  </details>



- **Real-Time Wildfire Localization on the NASA Autonomous Modular Sensor using Deep Learning**  
  Yajvan Ravan, Aref Malek, Chester Dolph, Nikhil Behari  
  _2026-01-20_ · https://arxiv.org/abs/2601.14475v1  
  <details><summary>Abstract</summary>

  High-altitude, multi-spectral, aerial imagery is scarce and expensive to acquire, yet it is necessary for algorithmic advances and application of machine learning models to high-impact problems such as wildfire detection. We introduce a human-annotated dataset from the NASA Autonomous Modular Sensor (AMS) using 12-channel, medium to high altitude (3 - 50 km) aerial wildfire images similar to those used in current US wildfire missions. Our dataset combines spectral data from 12 different channels, including infrared (IR), short-wave IR (SWIR), and thermal. We take imagery from 20 wildfire missions and randomly sample small patches to generate over 4000 images with high variability, including occlusions by smoke/clouds, easily-confused false positives, and nighttime imagery. We demonstrate results from a deep-learning model to automate the human-intensive process of fire perimeter determination. We train two deep neural networks, one for image classification and the other for pixel-level segmentation. The networks are combined into a unique real-time segmentation model to efficiently localize active wildfire on an incoming image feed. Our model achieves 96% classification accuracy, 74% Intersection-over-Union(IoU), and 84% recall surpassing past methods, including models trained on satellite data and classical color-rule algorithms. By leveraging a multi-spectral dataset, our model is able to detect active wildfire at nighttime and behind clouds, while distinguishing between false positives. We find that data from the SWIR, IR, and thermal bands is the most important to distinguish fire perimeters. Our code and dataset can be found here: https://github.com/nasa/Autonomous-Modular-Sensor-Wildfire-Segmentation/tree/main and https://drive.google.com/drive/folders/1-u4vs9rqwkwgdeeeoUhftCxrfe_4QPTn?=usp=drive_link

  </details>



- **Vision-Based Natural Language Scene Understanding for Autonomous Driving: An Extended Dataset and a New Model for Traffic Scene Description Generation**  
  Danial Sadrian Zadeh, Otman A. Basir, Behzad Moshiri  
  _2026-01-20_ · https://arxiv.org/abs/2601.14438v1  
  <details><summary>Abstract</summary>

  Traffic scene understanding is essential for enabling autonomous vehicles to accurately perceive and interpret their environment, thereby ensuring safe navigation. This paper presents a novel framework that transforms a single frontal-view camera image into a concise natural language description, effectively capturing spatial layouts, semantic relationships, and driving-relevant cues. The proposed model leverages a hybrid attention mechanism to enhance spatial and semantic feature extraction and integrates these features to generate contextually rich and detailed scene descriptions. To address the limited availability of specialized datasets in this domain, a new dataset derived from the BDD100K dataset has been developed, with comprehensive guidelines provided for its construction. Furthermore, the study offers an in-depth discussion of relevant evaluation metrics, identifying the most appropriate measures for this task. Extensive quantitative evaluations using metrics such as CIDEr and SPICE, complemented by human judgment assessments, demonstrate that the proposed model achieves strong performance and effectively fulfills its intended objectives on the newly developed dataset.

  </details>



- **Large-Scale Label Quality Assessment for Medical Segmentation via a Vision-Language Judge and Synthetic Data**  
  Yixiong Chen, Zongwei Zhou, Wenxuan Li, Alan Yuille  
  _2026-01-20_ · https://arxiv.org/abs/2601.14406v1  
  <details><summary>Abstract</summary>

  Large-scale medical segmentation datasets often combine manual and pseudo-labels of uneven quality, which can compromise training and evaluation. Low-quality labels may hamper performance and make the model training less robust. To address this issue, we propose SegAE (Segmentation Assessment Engine), a lightweight vision-language model (VLM) that automatically predicts label quality across 142 anatomical structures. Trained on over four million image-label pairs with quality scores, SegAE achieves a high correlation coefficient of 0.902 with ground-truth Dice similarity and evaluates a 3D mask in 0.06s. SegAE shows several practical benefits: (I) Our analysis reveals widespread low-quality labeling across public datasets; (II) SegAE improves data efficiency and training performance in active and semi-supervised learning, reducing dataset annotation cost by one-third and quality-checking time by 70% per label. This tool provides a simple and effective solution for quality control in large-scale medical segmentation datasets. The dataset, model weights, and codes are released at https://github.com/Schuture/SegAE.

  </details>



- **VideoMaMa: Mask-Guided Video Matting via Generative Prior**  
  Sangbeom Lim, Seoung Wug Oh, Jiahui Huang, Heeji Yoon, Seungryong Kim, Joon-Young Lee  
  _2026-01-20_ · https://arxiv.org/abs/2601.14255v1  
  <details><summary>Abstract</summary>

  Generalizing video matting models to real-world videos remains a significant challenge due to the scarcity of labeled data. To address this, we present Video Mask-to-Matte Model (VideoMaMa) that converts coarse segmentation masks into pixel accurate alpha mattes, by leveraging pretrained video diffusion models. VideoMaMa demonstrates strong zero-shot generalization to real-world footage, even though it is trained solely on synthetic data. Building on this capability, we develop a scalable pseudo-labeling pipeline for large-scale video matting and construct the Matting Anything in Video (MA-V) dataset, which offers high-quality matting annotations for more than 50K real-world videos spanning diverse scenes and motions. To validate the effectiveness of this dataset, we fine-tune the SAM2 model on MA-V to obtain SAM2-Matte, which outperforms the same model trained on existing matting datasets in terms of robustness on in-the-wild videos. These findings emphasize the importance of large-scale pseudo-labeled video matting and showcase how generative priors and accessible segmentation cues can drive scalable progress in video matting research.

  </details>



- **Motion 3-to-4: 3D Motion Reconstruction for 4D Synthesis**  
  Hongyuan Chen, Xingyu Chen, Youjia Zhang, Zexiang Xu, Anpei Chen  
  _2026-01-20_ · https://arxiv.org/abs/2601.14253v1  
  <details><summary>Abstract</summary>

  We present Motion 3-to-4, a feed-forward framework for synthesising high-quality 4D dynamic objects from a single monocular video and an optional 3D reference mesh. While recent advances have significantly improved 2D, video, and 3D content generation, 4D synthesis remains difficult due to limited training data and the inherent ambiguity of recovering geometry and motion from a monocular viewpoint. Motion 3-to-4 addresses these challenges by decomposing 4D synthesis into static 3D shape generation and motion reconstruction. Using a canonical reference mesh, our model learns a compact motion latent representation and predicts per-frame vertex trajectories to recover complete, temporally coherent geometry. A scalable frame-wise transformer further enables robustness to varying sequence lengths. Evaluations on both standard benchmarks and a new dataset with accurate ground-truth geometry show that Motion 3-to-4 delivers superior fidelity and spatial consistency compared to prior work. Project page is available at https://motion3-to-4.github.io/.

  </details>



- **LightOnOCR: A 1B End-to-End Multilingual Vision-Language Model for State-of-the-Art OCR**  
  Said Taghadouini, Adrien Cavaillès, Baptiste Aubertin  
  _2026-01-20_ · https://arxiv.org/abs/2601.14251v1  
  <details><summary>Abstract</summary>

  We present \textbf{LightOnOCR-2-1B}, a 1B-parameter end-to-end multilingual vision--language model that converts document images (e.g., PDFs) into clean, naturally ordered text without brittle OCR pipelines. Trained on a large-scale, high-quality distillation mix with strong coverage of scans, French documents, and scientific PDFs, LightOnOCR-2 achieves state-of-the-art results on OlmOCR-Bench while being 9$\times$ smaller and substantially faster than prior best-performing models. We further extend the output format to predict normalized bounding boxes for embedded images, introducing localization during pretraining via a resume strategy and refining it with RLVR using IoU-based rewards. Finally, we improve robustness with checkpoint averaging and task-arithmetic merging. We release model checkpoints under Apache 2.0, and publicly release the dataset and \textbf{LightOnOCR-bbox-bench} evaluation under their respective licenses.

  </details>



- **KAGE-Bench: Fast Known-Axis Visual Generalization Evaluation for Reinforcement Learning**  
  Egor Cherepanov, Daniil Zelezetsky, Alexey K. Kovalev, Aleksandr I. Panov  
  _2026-01-20_ · https://arxiv.org/abs/2601.14232v1  
  <details><summary>Abstract</summary>

  Pixel-based reinforcement learning agents often fail under purely visual distribution shift even when latent dynamics and rewards are unchanged, but existing benchmarks entangle multiple sources of shift and hinder systematic analysis. We introduce KAGE-Env, a JAX-native 2D platformer that factorizes the observation process into independently controllable visual axes while keeping the underlying control problem fixed. By construction, varying a visual axis affects performance only through the induced state-conditional action distribution of a pixel policy, providing a clean abstraction for visual generalization. Building on this environment, we define KAGE-Bench, a benchmark of six known-axis suites comprising 34 train-evaluation configuration pairs that isolate individual visual shifts. Using a standard PPO-CNN baseline, we observe strong axis-dependent failures, with background and photometric shifts often collapsing success, while agent-appearance shifts are comparatively benign. Several shifts preserve forward motion while breaking task completion, showing that return alone can obscure generalization failures. Finally, the fully vectorized JAX implementation enables up to 33M environment steps per second on a single GPU, enabling fast and reproducible sweeps over visual factors. Code: https://avanturist322.github.io/KAGEBench/.

  </details>



- **Copy-Trasform-Paste: Zero-Shot Object-Object Alignment Guided by Vision-Language and Geometric Constraints**  
  Rotem Gatenyo, Ohad Fried  
  _2026-01-20_ · https://arxiv.org/abs/2601.14207v1  
  <details><summary>Abstract</summary>

  We study zero-shot 3D alignment of two given meshes, using a text prompt describing their spatial relation -- an essential capability for content creation and scene assembly. Earlier approaches primarily rely on geometric alignment procedures, while recent work leverages pretrained 2D diffusion models to model language-conditioned object-object spatial relationships. In contrast, we directly optimize the relative pose at test time, updating translation, rotation, and isotropic scale with CLIP-driven gradients via a differentiable renderer, without training a new model. Our framework augments language supervision with geometry-aware objectives: a variant of soft-Iterative Closest Point (ICP) term to encourage surface attachment and a penetration loss to discourage interpenetration. A phased schedule strengthens contact constraints over time, and camera control concentrates the optimization on the interaction region. To enable evaluation, we curate a benchmark containing diverse categories and relations, and compare against baselines. Our method outperforms all alternatives, yielding semantically faithful and physically plausible alignments.

  </details>



- **IIR-VLM: In-Context Instance-level Recognition for Large Vision-Language Models**  
  Liang Shi, Wei Li, Kevin M Beussman, Lin Chen, Yun Fu  
  _2026-01-20_ · https://arxiv.org/abs/2601.14188v1  
  <details><summary>Abstract</summary>

  Instance-level recognition (ILR) concerns distinguishing individual instances from one another, with person re-identification as a prominent example. Despite the impressive visual perception capabilities of modern VLMs, we find their performance on ILR unsatisfactory, often dramatically underperforming domain-specific ILR models. This limitation hinders many practical application of VLMs, e.g. where recognizing familiar people and objects is crucial for effective visual understanding. Existing solutions typically learn to recognize instances one at a time using instance-specific datasets, which not only incur substantial data collection and training costs but also struggle with fine-grained discrimination. In this work, we propose IIR-VLM, a VLM enhanced for In-context Instance-level Recognition. We integrate pre-trained ILR expert models as auxiliary visual encoders to provide specialized features for learning diverse instances, which enables VLMs to learn new instances in-context in a one-shot manner. Further, IIR-VLM leverages this knowledge for instance-aware visual understanding. We validate IIR-VLM's efficacy on existing instance personalization benchmarks. Finally, we demonstrate its superior ILR performance on a challenging new benchmark, which assesses ILR capabilities across varying difficulty and diverse categories, with person, face, pet and general objects as the instances at task.

  </details>



- **Progressive self-supervised blind-spot denoising method for LDCT denoising**  
  Yichao Liu, Yueyang Teng, Junwen Guo  
  _2026-01-20_ · https://arxiv.org/abs/2601.14180v1  
  <details><summary>Abstract</summary>

  Self-supervised learning is increasingly investigated for low-dose computed tomography (LDCT) image denoising, as it alleviates the dependence on paired normal-dose CT (NDCT) data, which are often difficult to acquire in clinical practice. In this paper, we propose a novel self-supervised training strategy that relies exclusively on LDCT images. We introduce a step-wise blind-spot denoising mechanism that enforces conditional independence in a progressive manner, enabling more fine-grained denoising learning. In addition, we add Gaussian noise to LDCT images, which acts as a regularization and mitigates overfitting. Extensive experiments on the Mayo LDCT dataset demonstrate that the proposed method consistently outperforms existing self-supervised approaches and achieves performance comparable to, or better than, several representative supervised denoising methods.

  </details>


