# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-01-21 06:54 UTC_

Total papers shown: **50**


---

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



- **LLM Augmented Intervenable Multimodal Adaptor for Post-operative Complication Prediction in Lung Cancer Surgery**  
  Shubham Pandey, Bhavin Jawade, Srirangaraj Setlur, Venu Govindaraju, Kenneth Seastedt  
  _2026-01-20_ · https://arxiv.org/abs/2601.14154v1  
  <details><summary>Abstract</summary>

  Postoperative complications remain a critical concern in clinical practice, adversely affecting patient outcomes and contributing to rising healthcare costs. We present MIRACLE, a deep learning architecture for prediction of risk of postoperative complications in lung cancer surgery by integrating preoperative clinical and radiological data. MIRACLE employs a hyperspherical embedding space fusion of heterogeneous inputs, enabling the extraction of robust, discriminative features from both structured clinical records and high-dimensional radiological images. To enhance transparency of prediction and clinical utility, we incorporate an interventional deep learning module in MIRACLE, that not only refines predictions but also provides interpretable and actionable insights, allowing domain experts to interactively adjust recommendations based on clinical expertise. We validate our approach on POC-L, a real-world dataset comprising 3,094 lung cancer patients who underwent surgery at Roswell Park Comprehensive Cancer Center. Our results demonstrate that MIRACLE outperforms various traditional machine learning models and contemporary large language models (LLM) variants alone, for personalized and explainable postoperative risk management.

  </details>



- **GIC-DLC: Differentiable Logic Circuits for Hardware-Friendly Grayscale Image Compression**  
  Till Aczel, David F. Jenny, Simon Bührer, Andreas Plesner, Antonio Di Maio, Roger Wattenhofer  
  _2026-01-20_ · https://arxiv.org/abs/2601.14130v1  
  <details><summary>Abstract</summary>

  Neural image codecs achieve higher compression ratios than traditional hand-crafted methods such as PNG or JPEG-XL, but often incur substantial computational overhead, limiting their deployment on energy-constrained devices such as smartphones, cameras, and drones. We propose Grayscale Image Compression with Differentiable Logic Circuits (GIC-DLC), a hardware-aware codec where we train lookup tables to combine the flexibility of neural networks with the efficiency of Boolean operations. Experiments on grayscale benchmark datasets show that GIC-DLC outperforms traditional codecs in compression efficiency while allowing substantial reductions in energy consumption and latency. These results demonstrate that learned compression can be hardware-friendly, offering a promising direction for low-power image compression on edge devices.

  </details>



- **The Side Effects of Being Smart: Safety Risks in MLLMs' Multi-Image Reasoning**  
  Renmiao Chen, Yida Lu, Shiyao Cui, Xuan Ouyang, Victor Shea-Jay Huang, Shumin Zhang, Chengwei Pan, Han Qiu, Minlie Huang  
  _2026-01-20_ · https://arxiv.org/abs/2601.14127v1  
  <details><summary>Abstract</summary>

  As Multimodal Large Language Models (MLLMs) acquire stronger reasoning capabilities to handle complex, multi-image instructions, this advancement may pose new safety risks. We study this problem by introducing MIR-SafetyBench, the first benchmark focused on multi-image reasoning safety, which consists of 2,676 instances across a taxonomy of 9 multi-image relations. Our extensive evaluations on 19 MLLMs reveal a troubling trend: models with more advanced multi-image reasoning can be more vulnerable on MIR-SafetyBench. Beyond attack success rates, we find that many responses labeled as safe are superficial, often driven by misunderstanding or evasive, non-committal replies. We further observe that unsafe generations exhibit lower attention entropy than safe ones on average. This internal signature suggests a possible risk that models may over-focus on task solving while neglecting safety constraints. Our code and data are available at https://github.com/thu-coai/MIR-SafetyBench.

  </details>



- **Interp3D: Correspondence-aware Interpolation for Generative Textured 3D Morphing**  
  Xiaolu Liu, Yicong Li, Qiyuan He, Jiayin Zhu, Wei Ji, Angela Yao, Jianke Zhu  
  _2026-01-20_ · https://arxiv.org/abs/2601.14103v1  
  <details><summary>Abstract</summary>

  Textured 3D morphing seeks to generate smooth and plausible transitions between two 3D assets, preserving both structural coherence and fine-grained appearance. This ability is crucial not only for advancing 3D generation research but also for practical applications in animation, editing, and digital content creation. Existing approaches either operate directly on geometry, limiting them to shape-only morphing while neglecting textures, or extend 2D interpolation strategies into 3D, which often causes semantic ambiguity, structural misalignment, and texture blurring. These challenges underscore the necessity to jointly preserve geometric consistency, texture alignment, and robustness throughout the transition process. To address this, we propose Interp3D, a novel training-free framework for textured 3D morphing. It harnesses generative priors and adopts a progressive alignment principle to ensure both geometric fidelity and texture coherence. Starting from semantically aligned interpolation in condition space, Interp3D enforces structural consistency via SLAT (Structured Latent)-guided structure interpolation, and finally transfers appearance details through fine-grained texture fusion. For comprehensive evaluations, we construct a dedicated dataset, Interp3DData, with graded difficulty levels and assess generation results from fidelity, transition smoothness, and plausibility. Both quantitative metrics and human studies demonstrate the significant advantages of our proposed approach over previous methods. Source code is available at https://github.com/xiaolul2/Interp3D.

  </details>



- **Curriculum-Based Strategies for Efficient Cross-Domain Action Recognition**  
  Emily Kim, Allen Wu, Jessica Hodgins  
  _2026-01-20_ · https://arxiv.org/abs/2601.14101v1  
  <details><summary>Abstract</summary>

  Despite significant progress in human action recognition, generalizing to diverse viewpoints remains a challenge. Most existing datasets are captured from ground-level perspectives, and models trained on them often struggle to transfer to drastically different domains such as aerial views. This paper examines how curriculum-based training strategies can improve generalization to unseen real aerial-view data without using any real aerial data during training. We explore curriculum learning for cross-view action recognition using two out-of-domain sources: synthetic aerial-view data and real ground-view data. Our results on the evaluation on order of training (fine-tuning on synthetic aerial data vs. real ground data) shows that fine-tuning on real ground data but differ in how they transition from synthetic to real. The first uses a two-stage curriculum with direct fine-tuning, while the second applies a progressive curriculum that expands the dataset in multiple stages before fine-tuning. We evaluate both methods on the REMAG dataset using SlowFast (CNN-based) and MViTv2 (Transformer-based) architectures. Results show that combining the two out-of-domain datasets clearly outperforms training on a single domain, whether real ground-view or synthetic aerial-view. Both curriculum strategies match the top-1 accuracy of simple dataset combination while offering efficiency gains. With the two-step fine-tuning method, SlowFast achieves up to a 37% reduction in iterations and MViTv2 up to a 30% reduction compared to simple combination. The multi-step progressive approach further reduces iterations, by up to 9% for SlowFast and 30% for MViTv2, relative to the two-step method. These findings demonstrate that curriculum-based training can maintain comparable performance (top-1 accuracy within 3% range) while improving training efficiency in cross-view action recognition.

  </details>



- **DermaBench: A Clinician-Annotated Benchmark Dataset for Dermatology Visual Question Answering and Reasoning**  
  Abdurrahim Yilmaz, Ozan Erdem, Ece Gokyayla, Ayda Acar, Burc Bugra Dagtas, Dilara Ilhan Erdil, Gulsum Gencoglan, Burak Temelkuran  
  _2026-01-20_ · https://arxiv.org/abs/2601.14084v1  
  <details><summary>Abstract</summary>

  Vision-language models (VLMs) are increasingly important in medical applications; however, their evaluation in dermatology remains limited by datasets that focus primarily on image-level classification tasks such as lesion recognition. While valuable for recognition, such datasets cannot assess the full visual understanding, language grounding, and clinical reasoning capabilities of multimodal models. Visual question answering (VQA) benchmarks are required to evaluate how models interpret dermatological images, reason over fine-grained morphology, and generate clinically meaningful descriptions. We introduce DermaBench, a clinician-annotated dermatology VQA benchmark built on the Diverse Dermatology Images (DDI) dataset. DermaBench comprises 656 clinical images from 570 unique patients spanning Fitzpatrick skin types I-VI. Using a hierarchical annotation schema with 22 main questions (single-choice, multi-choice, and open-ended), expert dermatologists annotated each image for diagnosis, anatomic site, lesion morphology, distribution, surface features, color, and image quality, together with open-ended narrative descriptions and summaries, yielding approximately 14.474 VQA-style annotations. DermaBench is released as a metadata-only dataset to respect upstream licensing and is publicly available at Harvard Dataverse.

  </details>



- **Unsupervised Video Class-Incremental Learning via Deep Embedded Clustering Management**  
  Nattapong Kurpukdee, Adrian G. Bors  
  _2026-01-20_ · https://arxiv.org/abs/2601.14069v1  
  <details><summary>Abstract</summary>

  Unsupervised video class incremental learning (uVCIL) represents an important learning paradigm for learning video information without forgetting, and without considering any data labels. Prior approaches have focused on supervised class-incremental learning, relying on using the knowledge of labels and task boundaries, which is costly, requires human annotation, or is simply not a realistic option. In this paper, we propose a simple yet effective approach to address the uVCIL. We first consider a deep feature extractor network, providing a set of representative video features during each task without assuming any class or task information. We then progressively build a series of deep clusters from the extracted features. During the successive task learning, the model updated from the previous task is used as an initial state in order to transfer knowledge to the current learning task. We perform in-depth evaluations on three standard video action recognition datasets, including UCF101, HMDB51, and Something-to-Something V2, by ignoring the labels from the supervised setting. Our approach significantly outperforms other baselines on all datasets.

  </details>



- **Decoder-Free Supervoxel GNN for Accurate Brain-Tumor Localization in Multi-Modal MRI**  
  Andrea Protani, Marc Molina Van Den Bosch, Lorenzo Giusti, Heloisa Barbosa Da Silva, Paolo Cacace, Albert Sund Aillet, Miguel Angel Gonzalez Ballester, Friedhelm Hummel, Luigi Serio  
  _2026-01-20_ · https://arxiv.org/abs/2601.14055v1  
  <details><summary>Abstract</summary>

  Modern vision backbones for 3D medical imaging typically process dense voxel grids through parameter-heavy encoder-decoder structures, a design that allocates a significant portion of its parameters to spatial reconstruction rather than feature learning. Our approach introduces SVGFormer, a decoder-free pipeline built upon a content-aware grouping stage that partitions the volume into a semantic graph of supervoxels. Its hierarchical encoder learns rich node representations by combining a patch-level Transformer with a supervoxel-level Graph Attention Network, jointly modeling fine-grained intra-region features and broader inter-regional dependencies. This design concentrates all learnable capacity on feature encoding and provides inherent, dual-scale explainability from the patch to the region level. To validate the framework's flexibility, we trained two specialized models on the BraTS dataset: one for node-level classification and one for tumor proportion regression. Both models achieved strong performance, with the classification model achieving a F1-score of 0.875 and the regression model a MAE of 0.028, confirming the encoder's ability to learn discriminative and localized features. Our results establish that a graph-based, encoder-only paradigm offers an accurate and inherently interpretable alternative for 3D medical image representation.

  </details>



- **Weather-R1: Logically Consistent Reinforcement Fine-Tuning for Multimodal Reasoning in Meteorology**  
  Kaiyu Wu, Pucheng Han, Hualong Zhang, Naigeng Wu, Keze Wang  
  _2026-01-20_ · https://arxiv.org/abs/2601.14044v1  
  <details><summary>Abstract</summary>

  While Vision Language Models (VLMs) show advancing reasoning capabilities, their application in meteorology is constrained by a domain gap and a reasoning faithfulness gap. Specifically, mainstream Reinforcement Fine-Tuning (RFT) can induce Self-Contradictory Reasoning (Self-Contra), where the model's reasoning contradicts its final answer, which is unacceptable in such a high-stakes domain. To address these challenges, we construct WeatherQA, a novel multimodal reasoning benchmark in meteorology. We also propose Logically Consistent Reinforcement Fine-Tuning (LoCo-RFT), which resolves Self-Contra by introducing a logical consistency reward. Furthermore, we introduce Weather-R1, the first reasoning VLM with logical faithfulness in meteorology, to the best of our knowledge. Experiments demonstrate that Weather-R1 improves performance on WeatherQA by 9.8 percentage points over the baseline, outperforming Supervised Fine-Tuning and RFT, and even surpassing the original Qwen2.5-VL-32B. These results highlight the effectiveness of our LoCo-RFT and the superiority of Weather-R1. Our benchmark and code are available at https://github.com/Marcowky/Weather-R1.

  </details>



- **Generalizing Abstention for Noise-Robust Learning in Medical Image Segmentation**  
  Wesam Moustafa, Hossam Elsafty, Helen Schneider, Lorenz Sparrenberg, Rafet Sifa  
  _2026-01-20_ · https://arxiv.org/abs/2601.14039v1  
  <details><summary>Abstract</summary>

  Label noise is a critical problem in medical image segmentation, often arising from the inherent difficulty of manual annotation. Models trained on noisy data are prone to overfitting, which degrades their generalization performance. While a number of methods and strategies have been proposed to mitigate noisy labels in the segmentation domain, this area remains largely under-explored. The abstention mechanism has proven effective in classification tasks by enhancing the capabilities of Cross Entropy, yet its potential in segmentation remains unverified. In this paper, we address this gap by introducing a universal and modular abstention framework capable of enhancing the noise-robustness of a diverse range of loss functions. Our framework improves upon prior work with two key components: an informed regularization term to guide abstention behaviour, and a more flexible power-law-based auto-tuning algorithm for the abstention penalty. We demonstrate the framework's versatility by systematically integrating it with three distinct loss functions to create three novel, noise-robust variants: GAC, SAC, and ADS. Experiments on the CaDIS and DSAD medical datasets show our methods consistently and significantly outperform their non-abstaining baselines, especially under high noise levels. This work establishes that enabling models to selectively ignore corrupted samples is a powerful and generalizable strategy for building more reliable segmentation models. Our code is publicly available at https://github.com/wemous/abstention-for-segmentation.

  </details>



- **Correcting and Quantifying Systematic Errors in 3D Box Annotations for Autonomous Driving**  
  Alexandre Justo Miro, Ludvig af Klinteberg, Bogdan Timus, Aron Asefaw, Ajinkya Khoche, Thomas Gustafsson, Sina Sharif Mansouri, Masoud Daneshtalab  
  _2026-01-20_ · https://arxiv.org/abs/2601.14038v1  
  <details><summary>Abstract</summary>

  Accurate ground truth annotations are critical to supervised learning and evaluating the performance of autonomous vehicle systems. These vehicles are typically equipped with active sensors, such as LiDAR, which scan the environment in predefined patterns. 3D box annotation based on data from such sensors is challenging in dynamic scenarios, where objects are observed at different timestamps, hence different positions. Without proper handling of this phenomenon, systematic errors are prone to being introduced in the box annotations. Our work is the first to discover such annotation errors in widely used, publicly available datasets. Through our novel offline estimation method, we correct the annotations so that they follow physically feasible trajectories and achieve spatial and temporal consistency with the sensor data. For the first time, we define metrics for this problem; and we evaluate our method on the Argoverse 2, MAN TruckScenes, and our proprietary datasets. Our approach increases the quality of box annotations by more than 17% in these datasets. Furthermore, we quantify the annotation errors in them and find that the original annotations are misplaced by up to 2.5 m, with highly dynamic objects being the most affected. Finally, we test the impact of the errors in benchmarking and find that the impact is larger than the improvements that state-of-the-art methods typically achieve with respect to the previous state-of-the-art methods; showing that accurate annotations are essential for correct interpretation of performance. Our code is available at https://github.com/alexandre-justo-miro/annotation-correction-3D-boxes.

  </details>



- **STEC: A Reference-Free Spatio-Temporal Entropy Coverage Metric for Evaluating Sampled Video Frames**  
  Shih-Yao Lin  
  _2026-01-20_ · https://arxiv.org/abs/2601.13974v1  
  <details><summary>Abstract</summary>

  Frame sampling is a fundamental component in video understanding and video--language model pipelines, yet evaluating the quality of sampled frames remains challenging. Existing evaluation metrics primarily focus on perceptual quality or reconstruction fidelity, and are not designed to assess whether a set of sampled frames adequately captures informative and representative video content. We propose Spatio-Temporal Entropy Coverage (STEC), a simple and non-reference metric for evaluating the effectiveness of video frame sampling. STEC builds upon Spatio-Temporal Frame Entropy (STFE), which measures per-frame spatial information via entropy-based structural complexity, and evaluates sampled frames based on their temporal coverage and redundancy. By jointly modeling spatial information strength, temporal dispersion, and non-redundancy, STEC provides a principled and lightweight measure of sampling quality. Experiments on the MSR-VTT test-1k benchmark demonstrate that STEC clearly differentiates common sampling strategies, including random, uniform, and content-aware methods. We further show that STEC reveals robustness patterns across individual videos that are not captured by average performance alone, highlighting its practical value as a general-purpose evaluation tool for efficient video understanding. We emphasize that STEC is not designed to predict downstream task accuracy, but to provide a task-agnostic diagnostic signal for analyzing frame sampling behavior under constrained budgets.

  </details>



- **DExTeR: Weakly Semi-Supervised Object Detection with Class and Instance Experts for Medical Imaging**  
  Adrien Meyer, Didier Mutter, Nicolas Padoy  
  _2026-01-20_ · https://arxiv.org/abs/2601.13954v1  
  <details><summary>Abstract</summary>

  Detecting anatomical landmarks in medical imaging is essential for diagnosis and intervention guidance. However, object detection models rely on costly bounding box annotations, limiting scalability. Weakly Semi-Supervised Object Detection (WSSOD) with point annotations proposes annotating each instance with a single point, minimizing annotation time while preserving localization signals. A Point-to-Box teacher model, trained on a small box-labeled subset, converts these point annotations into pseudo-box labels to train a student detector. Yet, medical imagery presents unique challenges, including overlapping anatomy, variable object sizes, and elusive structures, which hinder accurate bounding box inference. To overcome these challenges, we introduce DExTeR (DETR with Experts), a transformer-based Point-to-Box regressor tailored for medical imaging. Built upon Point-DETR, DExTeR encodes single-point annotations as object queries, refining feature extraction with the proposed class-guided deformable attention, which guides attention sampling using point coordinates and class labels to capture class-specific characteristics. To improve discrimination in complex structures, it introduces CLICK-MoE (CLass, Instance, and Common Knowledge Mixture of Experts), decoupling class and instance representations to reduce confusion among adjacent or overlapping instances. Finally, we implement a multi-point training strategy which promotes prediction consistency across different point placements, improving robustness to annotation variability. DExTeR achieves state-of-the-art performance across three datasets spanning different medical domains (endoscopy, chest X-rays, and endoscopic ultrasound) highlighting its potential to reduce annotation costs while maintaining high detection accuracy.

  </details>



- **VTONGuard: Automatic Detection and Authentication of AI-Generated Virtual Try-On Content**  
  Shengyi Wu, Yan Hong, Shengyao Chen, Zheng Wang, Xianbing Sun, Jiahui Zhan, Jun Lan, Jianfu Zhang  
  _2026-01-20_ · https://arxiv.org/abs/2601.13951v1  
  <details><summary>Abstract</summary>

  With the rapid advancement of generative AI, virtual try-on (VTON) systems are becoming increasingly common in e-commerce and digital entertainment. However, the growing realism of AI-generated try-on content raises pressing concerns about authenticity and responsible use. To address this, we present VTONGuard, a large-scale benchmark dataset containing over 775,000 real and synthetic try-on images. The dataset covers diverse real-world conditions, including variations in pose, background, and garment styles, and provides both authentic and manipulated examples. Based on this benchmark, we conduct a systematic evaluation of multiple detection paradigms under unified training and testing protocols. Our results reveal each method's strengths and weaknesses and highlight the persistent challenge of cross-paradigm generalization. To further advance detection, we design a multi-task framework that integrates auxiliary segmentation to enhance boundary-aware feature learning, achieving the best overall performance on VTONGuard. We expect this benchmark to enable fair comparisons, facilitate the development of more robust detection models, and promote the safe and responsible deployment of VTON technologies in practice.

  </details>



- **TrackletGPT: A Language-like GPT Framework for White Matter Tract Segmentation**  
  Anoushkrit Goel, Simroop Singh, Ankita Joshi, Ranjeet Ranjan Jha, Chirag Ahuja, Aditya Nigam, Arnav Bhavsar  
  _2026-01-20_ · https://arxiv.org/abs/2601.13935v1  
  <details><summary>Abstract</summary>

  White Matter Tract Segmentation is imperative for studying brain structural connectivity, neurological disorders and neurosurgery. This task remains complex, as tracts differ among themselves, across subjects and conditions, yet have similar 3D structure across hemispheres and subjects. To address these challenges, we propose TrackletGPT, a language-like GPT framework which reintroduces sequential information in tokens using tracklets. TrackletGPT generalises seamlessly across datasets, is fully automatic, and encodes granular sub-streamline segments, Tracklets, scaling and refining GPT models in Tractography Segmentation. Based on our experiments, TrackletGPT outperforms state-of-the-art methods on average DICE, Overlap and Overreach scores on TractoInferno and HCP datasets, even on inter-dataset experiments.

  </details>



- **OCCAM: Class-Agnostic, Training-Free, Prior-Free and Multi-Class Object Counting**  
  Michail Spanakis, Iason Oikonomidis, Antonis Argyros  
  _2026-01-20_ · https://arxiv.org/abs/2601.13871v1  
  <details><summary>Abstract</summary>

  Class-Agnostic object Counting (CAC) involves counting instances of objects from arbitrary classes within an image. Due to its practical importance, CAC has received increasing attention in recent years. Most existing methods assume a single object class per image, rely on extensive training of large deep learning models and address the problem by incorporating additional information, such as visual exemplars or text prompts. In this paper, we present OCCAM, the first training-free approach to CAC that operates without the need of any supplementary information. Moreover, our approach addresses the multi-class variant of the problem, as it is capable of counting the object instances in each and every class among arbitrary object classes within an image. We leverage Segment Anything Model 2 (SAM2), a foundation model, and a custom threshold-based variant of the First Integer Neighbor Clustering Hierarchy (FINCH) algorithm to achieve competitive performance on widely used benchmark datasets, FSC-147 and CARPK. We propose a synthetic multi-class dataset and F1 score as a more suitable evaluation metric. The code for our method and the proposed synthetic dataset will be made publicly available at https://mikespanak.github.io/OCCAM_counter.

  </details>



- **DisasterVQA: A Visual Question Answering Benchmark Dataset for Disaster Scenes**  
  Aisha Al-Mohannadi, Ayisha Firoz, Yin Yang, Muhammad Imran, Ferda Ofli  
  _2026-01-20_ · https://arxiv.org/abs/2601.13839v1  
  <details><summary>Abstract</summary>

  Social media imagery provides a low-latency source of situational information during natural and human-induced disasters, enabling rapid damage assessment and response. While Visual Question Answering (VQA) has shown strong performance in general-purpose domains, its suitability for the complex and safety-critical reasoning required in disaster response remains unclear. We introduce DisasterVQA, a benchmark dataset designed for perception and reasoning in crisis contexts. DisasterVQA consists of 1,395 real-world images and 4,405 expert-curated question-answer pairs spanning diverse events such as floods, wildfires, and earthquakes. Grounded in humanitarian frameworks including FEMA ESF and OCHA MIRA, the dataset includes binary, multiple-choice, and open-ended questions covering situational awareness and operational decision-making tasks. We benchmark seven state-of-the-art vision-language models and find performance variability across question types, disaster categories, regions, and humanitarian tasks. Although models achieve high accuracy on binary questions, they struggle with fine-grained quantitative reasoning, object counting, and context-sensitive interpretation, particularly for underrepresented disaster scenarios. DisasterVQA provides a challenging and practical benchmark to guide the development of more robust and operationally meaningful vision-language models for disaster response. The dataset is publicly available at https://zenodo.org/records/18267770.

  </details>



- **FutureOmni: Evaluating Future Forecasting from Omni-Modal Context for Multimodal LLMs**  
  Qian Chen, Jinlan Fu, Changsong Li, See-Kiong Ng, Xipeng Qiu  
  _2026-01-20_ · https://arxiv.org/abs/2601.13836v1  
  <details><summary>Abstract</summary>

  Although Multimodal Large Language Models (MLLMs) demonstrate strong omni-modal perception, their ability to forecast future events from audio-visual cues remains largely unexplored, as existing benchmarks focus mainly on retrospective understanding. To bridge this gap, we introduce FutureOmni, the first benchmark designed to evaluate omni-modal future forecasting from audio-visual environments. The evaluated models are required to perform cross-modal causal and temporal reasoning, as well as effectively leverage internal knowledge to predict future events. FutureOmni is constructed via a scalable LLM-assisted, human-in-the-loop pipeline and contains 919 videos and 1,034 multiple-choice QA pairs across 8 primary domains. Evaluations on 13 omni-modal and 7 video-only models show that current systems struggle with audio-visual future prediction, particularly in speech-heavy scenarios, with the best accuracy of 64.8% achieved by Gemini 3 Flash. To mitigate this limitation, we curate a 7K-sample instruction-tuning dataset and propose an Omni-Modal Future Forecasting (OFF) training strategy. Evaluations on FutureOmni and popular audio-visual and video-only benchmarks demonstrate that OFF enhances future forecasting and generalization. We publicly release all code (https://github.com/OpenMOSS/FutureOmni) and datasets (https://huggingface.co/datasets/OpenMOSS-Team/FutureOmni).

  </details>



- **Insight: Interpretable Semantic Hierarchies in Vision-Language Encoders**  
  Kai Wittenmayer, Sukrut Rao, Amin Parchami-Araghi, Bernt Schiele, Jonas Fischer  
  _2026-01-20_ · https://arxiv.org/abs/2601.13798v1  
  <details><summary>Abstract</summary>

  Language-aligned vision foundation models perform strongly across diverse downstream tasks. Yet, their learned representations remain opaque, making interpreting their decision-making hard. Recent works decompose these representations into human-interpretable concepts, but provide poor spatial grounding and are limited to image classification tasks. In this work, we propose Insight, a language-aligned concept foundation model that provides fine-grained concepts, which are human-interpretable and spatially grounded in the input image. We leverage a hierarchical sparse autoencoder and a foundation model with strong semantic representations to automatically extract concepts at various granularities. Examining local co-occurrence dependencies of concepts allows us to define concept relationships. Through these relations we further improve concept naming and obtain richer explanations. On benchmark data, we show that Insight provides performance on classification and segmentation that is competitive with opaque foundation models while providing fine-grained, high quality concept-based explanations. Code is available at https://github.com/kawi19/Insight.

  </details>



- **HiT: History-Injection Transformers for Onboard Continuous Flood Change Detection**  
  Daniel Kyselica, Jonáš Herec, Oliver Kutis, Rado Pitoňák  
  _2026-01-20_ · https://arxiv.org/abs/2601.13751v1  
  <details><summary>Abstract</summary>

  Natural disaster monitoring through continuous satellite observation requires processing multi-temporal data under strict operational constraints. This paper addresses flood detection, a critical application for hazard management, by developing an onboard change detection system that operates within the memory and computational limits of small satellites. We propose History Injection mechanism for Transformer models (HiT), that maintains historical context from previous observations while reducing data storage by over 99\% of original image size. Moreover, testing on the STTORM-CD flood dataset confirms that the HiT mechanism within the Prithvi-tiny foundation model maintains detection accuracy compared to the bitemporal baseline. The proposed HiT-Prithvi model achieved 43 FPS on Jetson Orin Nano, a representative onboard hardware used in nanosats. This work establishes a practical framework for satellite-based continuous monitoring of natural disasters, supporting real-time hazard assessment without dependency on ground-based processing infrastructure. Architecture as well as model checkpoints is available at https://github.com/zaitra/HiT-change-detection

  </details>



- **Facial Spatiotemporal Graphs: Leveraging the 3D Facial Surface for Remote Physiological Measurement**  
  Sam Cantrill, David Ahmedt-Aristizabal, Lars Petersson, Hanna Suominen, Mohammad Ali Armin  
  _2026-01-20_ · https://arxiv.org/abs/2601.13724v1  
  <details><summary>Abstract</summary>

  Facial remote photoplethysmography (rPPG) methods estimate physiological signals by modeling subtle color changes on the 3D facial surface over time. However, existing methods fail to explicitly align their receptive fields with the 3D facial surface-the spatial support of the rPPG signal. To address this, we propose the Facial Spatiotemporal Graph (STGraph), a novel representation that encodes facial color and structure using 3D facial mesh sequences-enabling surface-aligned spatiotemporal processing. We introduce MeshPhys, a lightweight spatiotemporal graph convolutional network that operates on the STGraph to estimate physiological signals. Across four benchmark datasets, MeshPhys achieves state-of-the-art or competitive performance in both intra- and cross-dataset settings. Ablation studies show that constraining the model's receptive field to the facial surface acts as a strong structural prior, and that surface-aligned, 3D-aware node features are critical for robustly encoding facial surface color. Together, the STGraph and MeshPhys constitute a novel, principled modeling paradigm for facial rPPG, enabling robust, interpretable, and generalizable estimation. Code is available at https://samcantrill.github.io/facial-stgraph-rppg/ .

  </details>



- **MVGD-Net: A Novel Motion-aware Video Glass Surface Detection Network**  
  Yiwei Lu, Hao Huang, Tao Yan  
  _2026-01-20_ · https://arxiv.org/abs/2601.13715v1  
  <details><summary>Abstract</summary>

  Glass surface ubiquitous in both daily life and professional environments presents a potential threat to vision-based systems, such as robot and drone navigation. To solve this challenge, most recent studies have shown significant interest in Video Glass Surface Detection (VGSD). We observe that objects in the reflection (or transmission) layer appear farther from the glass surfaces. Consequently, in video motion scenarios, the notable reflected (or transmitted) objects on the glass surface move slower than objects in non-glass regions within the same spatial plane, and this motion inconsistency can effectively reveal the presence of glass surfaces. Based on this observation, we propose a novel network, named MVGD-Net, for detecting glass surfaces in videos by leveraging motion inconsistency cues. Our MVGD-Net features three novel modules: the Cross-scale Multimodal Fusion Module (CMFM) that integrates extracted spatial features and estimated optical flow maps, the History Guided Attention Module (HGAM) and Temporal Cross Attention Module (TCAM), both of which further enhances temporal features. A Temporal-Spatial Decoder (TSD) is also introduced to fuse the spatial and temporal features for generating the glass region mask. Furthermore, for learning our network, we also propose a large-scale dataset, which comprises 312 diverse glass scenarios with a total of 19,268 frames. Extensive experiments demonstrate that our MVGD-Net outperforms relevant state-of-the-art methods.

  </details>



- **ParkingTwin: Training-Free Streaming 3D Reconstruction for Parking-Lot Digital Twins**  
  Xinhao Liu, Yu Wang, Xiansheng Guo, Gordon Owusu Boateng, Yu Cao, Haonan Si, Xingchen Guo, Nirwan Ansari  
  _2026-01-20_ · https://arxiv.org/abs/2601.13706v1  
  <details><summary>Abstract</summary>

  High-fidelity parking-lot digital twins provide essential priors for path planning, collision checking, and perception validation in Automated Valet Parking (AVP). Yet robot-oriented reconstruction faces a trilemma: sparse forward-facing views cause weak parallax and ill-posed geometry; dynamic occlusions and extreme lighting hinder stable texture fusion; and neural rendering typically needs expensive offline optimization, violating edge-side streaming constraints. We propose ParkingTwin, a training-free, lightweight system for online streaming 3D reconstruction. First, OSM-prior-driven geometric construction uses OpenStreetMap semantic topology to directly generate a metric-consistent TSDF, replacing blind geometric search with deterministic mapping and avoiding costly optimization. Second, geometry-aware dynamic filtering employs a quad-modal constraint field (normal/height/depth consistency) to reject moving vehicles and transient occlusions in real time. Third, illumination-robust fusion in CIELAB decouples luminance and chromaticity via adaptive L-channel weighting and depth-gradient suppression, reducing seams under abrupt lighting changes. ParkingTwin runs at 30+ FPS on an entry-level GTX 1660. On a 68,000 m^2 real-world dataset, it achieves SSIM 0.87 (+16.0%), delivers about 15x end-to-end speedup, and reduces GPU memory by 83.3% compared with state-of-the-art 3D Gaussian Splatting (3DGS) that typically requires high-end GPUs (RTX 4090D). The system outputs explicit triangle meshes compatible with Unity/Unreal digital-twin pipelines. Project page: https://mihoutao-liu.github.io/ParkingTwin/

  </details>



- **Finally Outshining the Random Baseline: A Simple and Effective Solution for Active Learning in 3D Biomedical Imaging**  
  Carsten T. Lüth, Jeremias Traub, Kim-Celine Kahl, Till J. Bungert, Lukas Klein, Lars Krämer, Paul F. Jäger, Klaus Maier-Hein, Fabian Isensee  
  _2026-01-20_ · https://arxiv.org/abs/2601.13677v1  
  <details><summary>Abstract</summary>

  Active learning (AL) has the potential to drastically reduce annotation costs in 3D biomedical image segmentation, where expert labeling of volumetric data is both time-consuming and expensive. Yet, existing AL methods are unable to consistently outperform improved random sampling baselines adapted to 3D data, leaving the field without a reliable solution. We introduce Class-stratified Scheduled Power Predictive Entropy (ClaSP PE), a simple and effective query strategy that addresses two key limitations of standard uncertainty-based AL methods: class imbalance and redundancy in early selections. ClaSP PE combines class-stratified querying to ensure coverage of underrepresented structures and log-scale power noising with a decaying schedule to enforce query diversity in early-stage AL and encourage exploitation later. In our evaluation on 24 experimental settings using four 3D biomedical datasets within the comprehensive nnActive benchmark, ClaSP PE is the only method that generally outperforms improved random baselines in terms of both segmentation quality with statistically significant gains, whilst remaining annotation efficient. Furthermore, we explicitly simulate the real-world application by testing our method on four previously unseen datasets without manual adaptation, where all experiment parameters are set according to predefined guidelines. The results confirm that ClaSP PE robustly generalizes to novel tasks without requiring dataset-specific tuning. Within the nnActive framework, we present compelling evidence that an AL method can consistently outperform random baselines adapted to 3D segmentation, in terms of both performance and annotation efficiency in a realistic, close-to-production scenario. Our open-source implementation and clear deployment guidelines make it readily applicable in practice. Code is at https://github.com/MIC-DKFZ/nnActive.

  </details>



- **Transformer based Multi-task Fusion Network for Food Spoilage Detection and Shelf life Forecasting**  
  Mounika Kanulla, Rajasree Dadigi, Sailaja Thota, Vivek Yelleti  
  _2026-01-20_ · https://arxiv.org/abs/2601.13665v1  
  <details><summary>Abstract</summary>

  Food wastage is one of the critical challenges in the agricultural supply chain, and accurate and effective spoilage detection can help to reduce it. Further, it is highly important to forecast the spoilage information. This aids the longevity of the supply chain management in the agriculture field. This motivated us to propose fusion based architectures by combining CNN with LSTM and DeiT transformer for the following multi-tasks simultaneously: (i) vegetable classification, (ii) food spoilage detection, and (iii) shelf life forecasting. We developed a dataset by capturing images of vegetables from their fresh state until they were completely spoiled. From the experimental analysis it is concluded that the proposed fusion architectures CNN+CNN-LSTM and CNN+DeiT Transformer outperformed several deep learning models such as CNN, VGG16, ResNet50, Capsule Networks, and DeiT Transformers. Overall, CNN + DeiT Transformer yielded F1-score of 0.98 and 0.61 in vegetable classification and spoilage detection respectively and mean squared error (MSE) and symmetric mean absolute percentage error (SMAPE) of 3.58, and 41.66% respectively in spoilage forecasting. Further, the reliability of the fusion models was validated on noisy images and integrated with LIME to visualize the model decisions.

  </details>



- **Scaling Test-time Inference for Visual Grounding**  
  Guanqi Zhan, Changye Li, Zhijian Liu, Yao Lu, Yi Wu, Song Han, Ligeng Zhu  
  _2026-01-20_ · https://arxiv.org/abs/2601.13633v1  
  <details><summary>Abstract</summary>

  Visual grounding is an essential capability of Visual Language Models (VLMs) to understand the real physical world. Previous state-of-the-art grounding visual language models usually have large model sizes, making them heavy for deployment and slow for inference. However, we notice that the sizes of visual encoders are nearly the same for small and large VLMs and the major difference is the sizes of the language models. Small VLMs fall behind larger VLMs in grounding because of the difference in language understanding capability rather than visual information handling. To mitigate the gap, we introduce 'Efficient visual Grounding language Models' (EGM): a method to scale the test-time computation (#generated tokens). Scaling the test-time computation of a small model is deployment-friendly, and yields better end-to-end latency as the cost of each token is much cheaper compared to directly running a large model. On the RefCOCO benchmark, our EGM-Qwen3-VL-8B demonstrates 91.4 IoU with an average of 737ms (5.9x faster) latency while Qwen3-VL-235B demands 4,320ms to achieve 90.5 IoU. To validate our approach's generality, we further set up a new amodal grounding setting that requires the model to predict both the visible and occluded parts of the objects. Experiments show our method can consistently and significantly improve the vanilla grounding and amodal grounding capabilities of small models to be on par with or outperform the larger models, thereby improving the efficiency for visual grounding.

  </details>



- **LogicEnvGen: Task-Logic Driven Generation of Diverse Simulated Environments for Embodied AI**  
  Jianan Wang, Siyang Zhang, Bin Li, Juan Chen, Jingtao Qi, Zhuo Zhang, Chen Qian  
  _2026-01-20_ · https://arxiv.org/abs/2601.13556v1  
  <details><summary>Abstract</summary>

  Simulated environments play an essential role in embodied AI, functionally analogous to test cases in software engineering. However, existing environment generation methods often emphasize visual realism (e.g., object diversity and layout coherence), overlooking a crucial aspect: logical diversity from the testing perspective. This limits the comprehensive evaluation of agent adaptability and planning robustness in distinct simulated environments. To bridge this gap, we propose LogicEnvGen, a novel method driven by Large Language Models (LLMs) that adopts a top-down paradigm to generate logically diverse simulated environments as test cases for agents. Given an agent task, LogicEnvGen first analyzes its execution logic to construct decision-tree-structured behavior plans and then synthesizes a set of logical trajectories. Subsequently, it adopts a heuristic algorithm to refine the trajectory set, reducing redundant simulation. For each logical trajectory, which represents a potential task situation, LogicEnvGen correspondingly instantiates a concrete environment. Notably, it employs constraint solving for physical plausibility. Furthermore, we introduce LogicEnvEval, a novel benchmark comprising four quantitative metrics for environment evaluation. Experimental results verify the lack of logical diversity in baselines and demonstrate that LogicEnvGen achieves 1.04-2.61x greater diversity, significantly improving the performance in revealing agent faults by 4.00%-68.00%.

  </details>



- **DiffFace-Edit: A Diffusion-Based Facial Dataset for Forgery-Semantic Driven Deepfake Detection Analysis**  
  Feng Ding, Wenhui Yi, Xinan He, Mengyao Xiao, Jianfeng Xu, Jianqiang Du  
  _2026-01-20_ · https://arxiv.org/abs/2601.13551v1  
  <details><summary>Abstract</summary>

  Generative models now produce imperceptible, fine-grained manipulated faces, posing significant privacy risks. However, existing AI-generated face datasets generally lack focus on samples with fine-grained regional manipulations. Furthermore, no researchers have yet studied the real impact of splice attacks, which occur between real and manipulated samples, on detectors. We refer to these as detector-evasive samples. Based on this, we introduce the DiffFace-Edit dataset, which has the following advantages: 1) It contains over two million AI-generated fake images. 2) It features edits across eight facial regions (e.g., eyes, nose) and includes a richer variety of editing combinations, such as single-region and multi-region edits. Additionally, we specifically analyze the impact of detector-evasive samples on detection models. We conduct a comprehensive analysis of the dataset and propose a cross-domain evaluation that combines IMDL methods. Dataset will be available at https://github.com/ywh1093/DiffFace-Edit.

  </details>



- **GO-MLVTON: Garment Occlusion-Aware Multi-Layer Virtual Try-On with Diffusion Models**  
  Yang Yu, Yunze Deng, Yige Zhang, Yanjie Xiao, Youkun Ou, Wenhao Hu, Mingchao Li, Bin Feng, Wenyu Liu, Dandan Zheng, et al.  
  _2026-01-20_ · https://arxiv.org/abs/2601.13524v1  
  <details><summary>Abstract</summary>

  Existing Image-based virtual try-on (VTON) methods primarily focus on single-layer or multi-garment VTON, neglecting multi-layer VTON (ML-VTON), which involves dressing multiple layers of garments onto the human body with realistic deformation and layering to generate visually plausible outcomes. The main challenge lies in accurately modeling occlusion relationships between inner and outer garments to reduce interference from redundant inner garment features. To address this, we propose GO-MLVTON, the first multi-layer VTON method, introducing the Garment Occlusion Learning module to learn occlusion relationships and the StableDiffusion-based Garment Morphing & Fitting module to deform and fit garments onto the human body, producing high-quality multi-layer try-on results. Additionally, we present the MLG dataset for this task and propose a new metric named Layered Appearance Coherence Difference (LACD) for evaluation. Extensive experiments demonstrate the state-of-the-art performance of GO-MLVTON. Project page: https://upyuyang.github.io/go-mlvton/.

  </details>



- **Using deep learning for predicting cleansing quality of colon capsule endoscopy images**  
  Puneet Sharma, Kristian Dalsbø Hindberg, Benedicte Schelde-Olesen, Ulrik Deding, Esmaeil S. Nadimi, Jan-Matthias Braun  
  _2026-01-19_ · https://arxiv.org/abs/2601.13412v1  
  <details><summary>Abstract</summary>

  In this study, we explore the application of deep learning techniques for predicting cleansing quality in colon capsule endoscopy (CCE) images. Using a dataset of 500 images labeled by 14 clinicians on the Leighton-Rex scale (Poor, Fair, Good, and Excellent), a ResNet-18 model was trained for classification, leveraging stratified K-fold cross-validation to ensure robust performance. To optimize the model, structured pruning techniques were applied iteratively, achieving significant sparsity while maintaining high accuracy. Explainability of the pruned model was evaluated using Grad-CAM, Grad-CAM++, Eigen-CAM, Ablation-CAM, and Random-CAM, with the ROAD method employed for consistent evaluation. Our results indicate that for a pruned model, we can achieve a cross-validation accuracy of 88% with 79% sparsity, demonstrating the effectiveness of pruning in improving efficiency from 84% without compromising performance. We also highlight the challenges of evaluating cleansing quality of CCE images, emphasize the importance of explainability in clinical applications, and discuss the challenges associated with using the ROAD method for our task. Finally, we employ a variant of adaptive temperature scaling to calibrate the pruned models for an external dataset.

  </details>



- **Reasoning with Pixel-level Precision: QVLM Architecture and SQuID Dataset for Quantitative Geospatial Analytics**  
  Peter A. Massih, Eric Cosatto  
  _2026-01-19_ · https://arxiv.org/abs/2601.13401v1  
  <details><summary>Abstract</summary>

  Current Vision-Language Models (VLMs) fail at quantitative spatial reasoning because their architectures destroy pixel-level information required for counting and measurements. Vision encoders compress images through patch embeddings, reducing spatial indexing and losing the precise pixel-level tracking required for accurate counting. We present two contributions to address this fundamental limitation. First, we introduce SQuID (Satellite Quantitative Intelligence Dataset), a benchmark of 2,000 satellite image Question-Answer pairs with both numerical range and categorical answers, designed to evaluate quantitative spatial reasoning. The dataset spans three difficulty tiers with annotations automatically generated from human labels and their learned variability. Second, we propose QVLM (Quantitative Vision-Language Model), a code-generation architecture that maintains pixel precision by decoupling language understanding from visual analysis. Instead of encoding images into embeddings, QVLM generates executable code that first calls a segmentation model to obtain pixel-level masks, then operates directly on these masks, preserving spatial indexing throughout the reasoning process. Our experiments show that QVLM using GPT-5 as coder achieves 42.0% accuracy on SQuID compared to 28.1% for a VLM prompted with image-question pairs. Our work reveals that, for quantitative spatial reasoning, architectural decoupling enables better accuracy on quantitative tasks.

  </details>



- **Deep Image Prior with L0 Gradient Regularizer for Image Smoothing**  
  Nhat Thanh Tran, Kevin Bui, Jack Xin  
  _2026-01-19_ · https://arxiv.org/abs/2601.13400v1  
  <details><summary>Abstract</summary>

  Image smoothing is a fundamental image processing operation that preserves the underlying structure, such as strong edges and contours, and removes minor details and textures in an image. Many image smoothing algorithms rely on computing local window statistics or solving an optimization problem. Recent state-of-the-art methods leverage deep learning, but they require a carefully curated training dataset. Because constructing a proper training dataset for image smoothing is challenging, we propose DIP-$\ell_0$, a deep image prior framework that incorporates the $\ell_0$ gradient regularizer. This framework can perform high-quality image smoothing without any training data. To properly minimize the associated loss function that has the nonconvex, nonsmooth $\ell_0$ ``norm", we develop an alternating direction method of multipliers algorithm that utilizes an off-the-shelf $\ell_0$ gradient minimization solver. Numerical experiments demonstrate that the proposed DIP-$\ell_0$ outperforms many image smoothing algorithms in edge-preserving image smoothing and JPEG artifact removal.

  </details>



- **Organ-Aware Attention Improves CT Triage and Classification**  
  Lavsen Dahal, Yubraj Bhandari, Geoffrey D. Rubin, Joseph Y. Lo  
  _2026-01-19_ · https://arxiv.org/abs/2601.13385v1  
  <details><summary>Abstract</summary>

  There is an urgent need for triage and classification of high-volume medical imaging modalities such as computed tomography (CT), which can improve patient care and mitigate radiologist burnout. Study-level CT triage requires calibrated predictions with localized evidence; however, off-the-shelf Vision Language Models (VLM) struggle with 3D anatomy, protocol shifts, and noisy report supervision. This study used the two largest publicly available chest CT datasets: CT-RATE and RADCHEST-CT (held-out external test set). Our carefully tuned supervised baseline (instantiated as a simple Global Average Pooling head) establishes a new supervised state of the art, surpassing all reported linear-probe VLMs. Building on this baseline, we present ORACLE-CT, an encoder-agnostic, organ-aware head that pairs Organ-Masked Attention (mask-restricted, per-organ pooling that yields spatial evidence) with Organ-Scalar Fusion (lightweight fusion of normalized volume and mean-HU cues). In the chest setting, ORACLE-CT masked attention model achieves AUROC 0.86 on CT-RATE; in the abdomen setting, on MERLIN (30 findings), our supervised baseline exceeds a reproduced zero-shot VLM baseline obtained by running publicly released weights through our pipeline, and adding masked attention plus scalar fusion further improves performance to AUROC 0.85. Together, these results deliver state-of-the-art supervised classification performance across both chest and abdomen CT under a unified evaluation protocol. The source code is available at https://github.com/lavsendahal/oracle-ct.

  </details>



- **Practical Insights into Semi-Supervised Object Detection Approaches**  
  Chaoxin Wang, Bharaneeshwar Balasubramaniyam, Anurag Sangem, Nicolais Guevara, Doina Caragea  
  _2026-01-19_ · https://arxiv.org/abs/2601.13380v1  
  <details><summary>Abstract</summary>

  Learning in data-scarce settings has recently gained significant attention in the research community. Semi-supervised object detection(SSOD) aims to improve detection performance by leveraging a large number of unlabeled images alongside a limited number of labeled images(a.k.a.,few-shot learning). In this paper, we present a comprehensive comparison of three state-of-the-art SSOD approaches, including MixPL, Semi-DETR and Consistent-Teacher, with the goal of understanding how performance varies with the number of labeled images. We conduct experiments using the MS-COCO and Pascal VOC datasets, two popular object detection benchmarks which allow for standardized evaluation. In addition, we evaluate the SSOD approaches on a custom Beetle dataset which enables us to gain insights into their performance on specialized datasets with a smaller number of object categories. Our findings highlight the trade-offs between accuracy, model size, and latency, providing insights into which methods are best suited for low-data regimes.

  </details>



- **Real-Time 4D Radar Perception for Robust Human Detection in Harsh Enclosed Environments**  
  Zhenan Liu, Yaodong Cui, Amir Khajepour, George Shaker  
  _2026-01-19_ · https://arxiv.org/abs/2601.13364v1  
  <details><summary>Abstract</summary>

  This paper introduces a novel methodology for generating controlled, multi-level dust concentrations in a highly cluttered environment representative of harsh, enclosed environments, such as underground mines, road tunnels, or collapsed buildings, enabling repeatable mm-wave propagation studies under severe electromagnetic constraints. We also present a new 4D mmWave radar dataset, augmented by camera and LiDAR, illustrating how dust particles and reflective surfaces jointly impact the sensing functionality. To address these challenges, we develop a threshold-based noise filtering framework leveraging key radar parameters (RCS, velocity, azimuth, elevation) to suppress ghost targets and mitigate strong multipath reflections at the raw data level. Building on the filtered point clouds, a cluster-level, rule-based classification pipeline exploits radar semantics-velocity, RCS, and volumetric spread-to achieve reliable, real-time pedestrian detection without extensive domainspecific training. Experimental results confirm that this integrated approach significantly enhances clutter mitigation, detection robustness, and overall system resilience in dust-laden mining environments.

  </details>



- **CausalSpatial: A Benchmark for Object-Centric Causal Spatial Reasoning**  
  Wenxin Ma, Chenlong Wang, Ruisheng Yuan, Hao Chen, Nanru Dai, S. Kevin Zhou, Yijun Yang, Alan Yuille, Jieneng Chen  
  _2026-01-19_ · https://arxiv.org/abs/2601.13304v1  
  <details><summary>Abstract</summary>

  Humans can look at a static scene and instantly predict what happens next -- will moving this object cause a collision? We call this ability Causal Spatial Reasoning. However, current multimodal large language models (MLLMs) cannot do this, as they remain largely restricted to static spatial perception, struggling to answer "what-if" questions in a 3D scene. We introduce CausalSpatial, a diagnostic benchmark evaluating whether models can anticipate consequences of object motions across four tasks: Collision, Compatibility, Occlusion, and Trajectory. Results expose a severe gap: humans score 84% while GPT-5 achieves only 54%. Why do MLLMs fail? Our analysis uncovers a fundamental deficiency: models over-rely on textual chain-of-thought reasoning that drifts from visual evidence, producing fluent but spatially ungrounded hallucinations. To address this, we propose the Causal Object World model (COW), a framework that externalizes the simulation process by generating videos of hypothetical dynamics. With explicit visual cues of causality, COW enables models to ground their reasoning in physical reality rather than linguistic priors. We make the dataset and code publicly available here: https://github.com/CausalSpatial/CausalSpatial

  </details>



- **Enginuity: Building an Open Multi-Domain Dataset of Complex Engineering Diagrams**  
  Ethan Seefried, Prahitha Movva, Naga Harshita Marupaka, Tilak Kasturi, Tirthankar Ghosal  
  _2026-01-19_ · https://arxiv.org/abs/2601.13299v1  
  <details><summary>Abstract</summary>

  We propose Enginuity - the first open, large-scale, multi-domain engineering diagram dataset with comprehensive structural annotations designed for automated diagram parsing. By capturing hierarchical component relationships, connections, and semantic elements across diverse engineering domains, our proposed dataset would enable multimodal large language models to address critical downstream tasks including structured diagram parsing, cross-modal information retrieval, and AI-assisted engineering simulation. Enginuity would be transformative for AI for Scientific Discovery by enabling artificial intelligence systems to comprehend and manipulate the visual-structural knowledge embedded in engineering diagrams, breaking down a fundamental barrier that currently prevents AI from fully participating in scientific workflows where diagram interpretation, technical drawing analysis, and visual reasoning are essential for hypothesis generation, experimental design, and discovery.

  </details>



- **ConvMambaNet: A Hybrid CNN-Mamba State Space Architecture for Accurate and Real-Time EEG Seizure Detection**  
  Md. Nishan Khan, Kazi Shahriar Sanjid, Md. Tanzim Hossain, Asib Mostakim Fony, Istiak Ahmed, M. Monir Uddin  
  _2026-01-19_ · https://arxiv.org/abs/2601.13234v1  
  <details><summary>Abstract</summary>

  Epilepsy is a chronic neurological disorder marked by recurrent seizures that can severely impact quality of life. Electroencephalography (EEG) remains the primary tool for monitoring neural activity and detecting seizures, yet automated analysis remains challenging due to the temporal complexity of EEG signals. This study introduces ConvMambaNet, a hybrid deep learning model that integrates Convolutional Neural Networks (CNNs) with the Mamba Structured State Space Model (SSM) to enhance temporal feature extraction. By embedding the Mamba-SSM block within a CNN framework, the model effectively captures both spatial and long-range temporal dynamics. Evaluated on the CHB-MIT Scalp EEG dataset, ConvMambaNet achieved a 99% accuracy and demonstrated robust performance under severe class imbalance. These results underscore the model's potential for precise and efficient seizure detection, offering a viable path toward real-time, automated epilepsy monitoring in clinical environments.

  </details>



- **Not all Blends are Equal: The BLEMORE Dataset of Blended Emotion Expressions with Relative Salience Annotations**  
  Tim Lachmann, Alexandra Israelsson, Christina Tornberg, Teimuraz Saghinadze, Michal Balazia, Philipp Müller, Petri Laukka  
  _2026-01-19_ · https://arxiv.org/abs/2601.13225v1  
  <details><summary>Abstract</summary>

  Humans often experience not just a single basic emotion at a time, but rather a blend of several emotions with varying salience. Despite the importance of such blended emotions, most video-based emotion recognition approaches are designed to recognize single emotions only. The few approaches that have attempted to recognize blended emotions typically cannot assess the relative salience of the emotions within a blend. This limitation largely stems from the lack of datasets containing a substantial number of blended emotion samples annotated with relative salience. To address this shortcoming, we introduce BLEMORE, a novel dataset for multimodal (video, audio) blended emotion recognition that includes information on the relative salience of each emotion within a blend. BLEMORE comprises over 3,000 clips from 58 actors, performing 6 basic emotions and 10 distinct blends, where each blend has 3 different salience configurations (50/50, 70/30, and 30/70). Using this dataset, we conduct extensive evaluations of state-of-the-art video classification approaches on two blended emotion prediction tasks: (1) predicting the presence of emotions in a given sample, and (2) predicting the relative salience of emotions in a blend. Our results show that unimodal classifiers achieve up to 29% presence accuracy and 13% salience accuracy on the validation set, while multimodal methods yield clear improvements, with ImageBind + WavLM reaching 35% presence accuracy and HiCMAE 18% salience accuracy. On the held-out test set, the best models achieve 33% presence accuracy (VideoMAEv2 + HuBERT) and 18% salience accuracy (HiCMAE). In sum, the BLEMORE dataset provides a valuable resource to advancing research on emotion recognition systems that account for the complexity and significance of blended emotion expressions.

  </details>



- **ObjectVisA-120: Object-based Visual Attention Prediction in Interactive Street-crossing Environments**  
  Igor Vozniak, Philipp Mueller, Nils Lipp, Janis Sprenger, Konstantin Poddubnyy, Davit Hovhannisyan, Christian Mueller, Andreas Bulling, Philipp Slusallek  
  _2026-01-19_ · https://arxiv.org/abs/2601.13218v1  
  <details><summary>Abstract</summary>

  The object-based nature of human visual attention is well-known in cognitive science, but has only played a minor role in computational visual attention models so far. This is mainly due to a lack of suitable datasets and evaluation metrics for object-based attention. To address these limitations, we present \dataset~ -- a novel 120-participant dataset of spatial street-crossing navigation in virtual reality specifically geared to object-based attention evaluations. The uniqueness of the presented dataset lies in the ethical and safety affiliated challenges that make collecting comparable data in real-world environments highly difficult. \dataset~ not only features accurate gaze data and a complete state-space representation of objects in the virtual environment, but it also offers variable scenario complexities and rich annotations, including panoptic segmentation, depth information, and vehicle keypoints. We further propose object-based similarity (oSIM) as a novel metric to evaluate the performance of object-based visual attention models, a previously unexplored performance characteristic. Our evaluations show that explicitly optimising for object-based attention not only improves oSIM performance but also leads to an improved model performance on common metrics. In addition, we present SUMGraph, a Mamba U-Net-based model, which explicitly encodes critical scene objects (vehicles) in a graph representation, leading to further performance improvements over several state-of-the-art visual attention prediction methods. The dataset, code and models will be publicly released.

  </details>



- **Rethinking Skip Connections: Additive U-Net for Robust and Interpretable Denoising**  
  Vikram R Lakkavalli  
  _2026-01-19_ · https://arxiv.org/abs/2601.13208v1  
  <details><summary>Abstract</summary>

  Skip connections are central to U-Net architectures for image denoising, but standard concatenation doubles channel dimensionality and obscures information flow, allowing uncontrolled noise transfer. We propose the Additive U-Net, which replaces concatenative skips with gated additive connections. Each skip pathway is scaled by a learnable non-negative scalar, offering explicit and interpretable control over encoder contributions while avoiding channel inflation. Evaluations on the Kodak-17 denoising benchmark show that Additive U-Net achieves competitive PSNR/SSIM at noise levels σ = 15, 25, 50, with robustness across kernel schedules and depths. Notably, effective denoising is achieved even without explicit down/up-sampling or forced hierarchies, as the model naturally learns a progression from high-frequency to band-pass to low-frequency features. These results position additive skips as a lightweight and interpretable alternative to concatenation, enabling both efficient design and a clearer understanding of multi-scale information transfer in reconstruction networks.

  </details>



- **GTPred: Benchmarking MLLMs for Interpretable Geo-localization and Time-of-capture Prediction**  
  Jinnao Li, Zijian Chen, Tingzhu Chen, Changbo Wang  
  _2026-01-19_ · https://arxiv.org/abs/2601.13207v1  
  <details><summary>Abstract</summary>

  Geo-localization aims to infer the geographic location where an image was captured using observable visual evidence. Traditional methods achieve impressive results through large-scale training on massive image corpora. With the emergence of multi-modal large language models (MLLMs), recent studies have explored their applications in geo-localization, benefiting from improved accuracy and interpretability. However, existing benchmarks largely ignore the temporal information inherent in images, which can further constrain the location. To bridge this gap, we introduce GTPred, a novel benchmark for geo-temporal prediction. GTPred comprises 370 globally distributed images spanning over 120 years. We evaluate MLLM predictions by jointly considering year and hierarchical location sequence matching, and further assess intermediate reasoning chains using meticulously annotated ground-truth reasoning processes. Experiments on 8 proprietary and 7 open-source MLLMs show that, despite strong visual perception, current models remain limited in world knowledge and geo-temporal reasoning. Results also demonstrate that incorporating temporal information significantly enhances location inference performance.

  </details>



- **A Streamlined Attention-Based Network for Descriptor Extraction**  
  Mattia D'Urso, Emanuele Santellani, Christian Sormann, Mattia Rossi, Andreas Kuhn, Friedrich Fraundorfer  
  _2026-01-19_ · https://arxiv.org/abs/2601.13126v1  
  <details><summary>Abstract</summary>

  We introduce SANDesc, a Streamlined Attention-Based Network for Descriptor extraction that aims to improve on existing architectures for keypoint description. Our descriptor network learns to compute descriptors that improve matching without modifying the underlying keypoint detector. We employ a revised U-Net-like architecture enhanced with Convolutional Block Attention Modules and residual paths, enabling effective local representation while maintaining computational efficiency. We refer to the building blocks of our model as Residual U-Net Blocks with Attention. The model is trained using a modified triplet loss in combination with a curriculum learning-inspired hard negative mining strategy, which improves training stability. Extensive experiments on HPatches, MegaDepth-1500, and the Image Matching Challenge 2021 show that training SANDesc on top of existing keypoint detectors leads to improved results on multiple matching tasks compared to the original keypoint descriptors. At the same time, SANDesc has a model complexity of just 2.4 million parameters. As a further contribution, we introduce a new urban dataset featuring 4K images and pre-calibrated intrinsics, designed to evaluate feature extractors. On this benchmark, SANDesc achieves substantial performance gains over the existing descriptors while operating with limited computational resources.

  </details>


