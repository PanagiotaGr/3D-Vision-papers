# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-03-12 07:11 UTC_

Total papers shown: **9**


---

- **Are Video Reasoning Models Ready to Go Outside?**  
  Yangfan He, Changgyu Boo, Jaehong Yoon  
  _2026-03-11_ · https://arxiv.org/abs/2603.10652v1  
  <details><summary>Abstract</summary>

  In real-world deployment, vision-language models often encounter disturbances such as weather, occlusion, and camera motion. Under such conditions, their understanding and reasoning degrade substantially, revealing a gap between clean, controlled (i.e., unperturbed) evaluation settings and real-world robustness. To address this limitation, we propose ROVA, a novel training framework that improves robustness by modeling a robustness-aware consistency reward under spatio-temporal corruptions. ROVA introduces a difficulty-aware online training strategy that prioritizes informative samples based on the model's evolving capability. Specifically, it continuously re-estimates sample difficulty via self-reflective evaluation, enabling adaptive training with a robustness-aware consistency reward. We also introduce PVRBench, a new benchmark that injects real-world perturbations into embodied video datasets to assess both accuracy and reasoning quality under realistic disturbances. We evaluate ROVA and baselines on PVRBench, UrbanVideo, and VisBench, where open-source and proprietary models suffer up to 35% and 28% drops in accuracy and reasoning under realistic perturbations. ROVA effectively mitigates performance degradation, boosting relative accuracy by at least 24% and reasoning by over 9% compared with baseline models (QWen2.5/3-VL, InternVL2.5, Embodied-R). These gains transfer to clean standard benchmarks, yielding consistent improvements.

  </details>



- **EmoStory: Emotion-Aware Story Generation**  
  Jingyuan Yang, Rucong Chen, Hui Huang  
  _2026-03-11_ · https://arxiv.org/abs/2603.10349v1  
  <details><summary>Abstract</summary>

  Story generation aims to produce image sequences that depict coherent narratives while maintaining subject consistency across frames. Although existing methods have excelled in producing coherent and expressive stories, they remain largely emotion-neutral, focusing on what subject appears in a story while overlooking how emotions shape narrative interpretation and visual presentation. As stories are intended to engage audiences emotionally, we introduce emotion-aware story generation, a new task that aims to generate subject-consistent visual stories with explicit emotional directions. This task is challenging due to the abstract nature of emotions, which must be grounded in concrete visual elements and consistently expressed across a narrative through visual composition. To address these challenges, we propose EmoStory, a two-stage framework that integrates agent-based story planning and region-aware story generation. The planning stage transforms target emotions into coherent story prompts with emotion agent and writer agent, while the generation stage preserves subject consistency and injects emotion-related elements through region-aware composition. We evaluate EmoStory on a newly constructed dataset covering 25 subjects and 600 emotional stories. Extensive quantitative and qualitative results, along with user studies, show that EmoStory outperforms state-of-the-art story generation methods in emotion accuracy, prompt alignment, and subject consistency.

  </details>



- **Robotic Ultrasound Makes CBCT Alive**  
  Feng Li, Ziyuan Li, Zhongliang Jiang, Nassir Navab, Yuan Bi  
  _2026-03-10_ · https://arxiv.org/abs/2603.10220v1  
  <details><summary>Abstract</summary>

  Intraoperative Cone Beam Computed Tomography (CBCT) provides a reliable 3D anatomical context essential for interventional planning. However, its static nature fails to provide continuous monitoring of soft-tissue deformations induced by respiration, probe pressure, and surgical manipulation, leading to navigation discrepancies. We propose a deformation-aware CBCT updating framework that leverages robotic ultrasound as a dynamic proxy to infer tissue motion and update static CBCT slices in real time. Starting from calibration-initialized alignment with linear correlation of linear combination (LC2)-based rigid refinement, our method establishes accurate multimodal correspondence. To capture intraoperative dynamics, we introduce the ultrasound correlation UNet (USCorUNet), a lightweight network trained with optical flow-guided supervision to learn deformation-aware correlation representations, enabling accurate, real-time dense deformation field estimation from ultrasound streams. The inferred deformation is spatially regularized and transferred to the CBCT reference to produce deformation-consistent visualizations without repeated radiation exposure. We validate the proposed approach through deformation estimation and ultrasound-guided CBCT updating experiments. Results demonstrate real-time end-to-end CBCT slice updating and physically plausible deformation estimation, enabling dynamic refinement of static CBCT guidance during robotic ultrasound-assisted interventions. The source code is publicly available at https://github.com/anonymous-codebase/us-cbct-demo.

  </details>



- **4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video**  
  Jin Lyu, Liang An, Pujin Cheng, Yebin Liu, Xiaoying Tang  
  _2026-03-10_ · https://arxiv.org/abs/2603.10125v1  
  <details><summary>Abstract</summary>

  4D reconstruction of equine family (e.g. horses) from monocular video is important for animal welfare. Previous mainstream 4D animal reconstruction methods require joint optimization of motion and appearance over a whole video, which is time-consuming and sensitive to incomplete observation. In this work, we propose a novel framework called 4DEquine by disentangling the 4D reconstruction problem into two sub-problems: dynamic motion reconstruction and static appearance reconstruction. For motion, we introduce a simple yet effective spatio-temporal transformer with a post-optimization stage to regress smooth and pixel-aligned pose and shape sequences from video. For appearance, we design a novel feed-forward network that reconstructs a high-fidelity, animatable 3D Gaussian avatar from as few as a single image. To assist training, we create a large-scale synthetic motion dataset, VarenPoser, which features high-quality surface motions and diverse camera trajectories, as well as a synthetic appearance dataset, VarenTex, comprising realistic multi-view images generated through multi-view diffusion. While training only on synthetic datasets, 4DEquine achieves state-of-the-art performance on real-world APT36K and AiM datasets, demonstrating the superiority of 4DEquine and our new datasets for both geometry and appearance reconstruction. Comprehensive ablation studies validate the effectiveness of both the motion and appearance reconstruction network. Project page: https://luoxue-star.github.io/4DEquine_Project_Page/.

  </details>



- **BrainSTR: Spatio-Temporal Contrastive Learning for Interpretable Dynamic Brain Network Modeling**  
  Guiliang Guo, Guangqi Wen, Lingwen Liu, Ruoxian Song, Peng Cao, Jinzhu Yang, Fei Wang, Xiaoli Liu, Osmar R. Zaiane  
  _2026-03-10_ · https://arxiv.org/abs/2603.09825v1  
  <details><summary>Abstract</summary>

  Dynamic functional connectivity captures time-varying brain states for better neuropsychiatric diagnosis and spatio-temporal interpretability, i.e., identifying when discriminative disease signatures emerge and where they reside in the connectivity topology. Reliable interpretability faces major challenges: diagnostic signals are often subtle and sparsely distributed across both time and topology, while nuisance fluctuations and non-diagnostic connectivities are pervasive. To address these issues, we propose BrainSTR, a spatio-temporal contrastive learning framework for interpretable dynamic brain network modeling. BrainSTR learns state-consistent phase boundaries via a data-driven Adaptive Phase Partition module, identifies diagnostically critical phases with attention, and extracts disease-related connectivity within each phase using an Incremental Graph Structure Generator regularized by binarization, temporal smoothness, and sparsity. Then, we introduce a spatio-temporal supervised contrastive learning approach that leverages diagnosis-relevant spatio-temporal patterns to refine the similarity metric between samples and capture more discriminative spatio-temporal features, thereby constructing a well-structured semantic space for coherent and interpretable representations. Experiments on ASD, BD, and MDD validate the effectiveness of BrainSTR, and the discovered critical phases and subnetworks provide interpretable evidence consistent with prior neuroimaging findings. Our code: https://anonymous.4open.science/r/BrainSTR1.

  </details>



- **FrameDiT: Diffusion Transformer with Frame-Level Matrix Attention for Efficient Video Generation**  
  Minh Khoa Le, Kien Do, Duc Thanh Nguyen, Truyen Tran  
  _2026-03-10_ · https://arxiv.org/abs/2603.09721v1  
  <details><summary>Abstract</summary>

  High-fidelity video generation remains challenging for diffusion models due to the difficulty of modeling complex spatio-temporal dynamics efficiently. Recent video diffusion methods typically represent a video as a sequence of spatio-temporal tokens which can be modeled using Diffusion Transformers (DiTs). However, this approach faces a trade-off between the strong but expensive Full 3D Attention and the efficient but temporally limited Local Factorized Attention. To resolve this trade-off, we propose Matrix Attention, a frame-level temporal attention mechanism that processes an entire frame as a matrix and generates query, key, and value matrices via matrix-native operations. By attending across frames rather than tokens, Matrix Attention effectively preserves global spatio-temporal structure and adapts to significant motion. We build FrameDiT-G, a DiT architecture based on MatrixAttention, and further introduce FrameDiT-H, which integrates Matrix Attention with Local Factorized Attention to capture both large and small motion. Extensive experiments show that FrameDiT-H achieves state-of-the-art results across multiple video generation benchmarks, offering improved temporal coherence and video quality while maintaining efficiency comparable to Local Factorized Attention.

  </details>



- **DiffWind: Physics-Informed Differentiable Modeling of Wind-Driven Object Dynamics**  
  Yuanhang Lei, Boming Zhao, Zesong Yang, Xingxuan Li, Tao Cheng, Haocheng Peng, Ru Zhang, Yang Yang, Siyuan Huang, Yujun Shen, et al.  
  _2026-03-10_ · https://arxiv.org/abs/2603.09668v1  
  <details><summary>Abstract</summary>

  Modeling wind-driven object dynamics from video observations is highly challenging due to the invisibility and spatio-temporal variability of wind, as well as the complex deformations of objects. We present DiffWind, a physics-informed differentiable framework that unifies wind-object interaction modeling, video-based reconstruction, and forward simulation. Specifically, we represent wind as a grid-based physical field and objects as particle systems derived from 3D Gaussian Splatting, with their interaction modeled by the Material Point Method (MPM). To recover wind-driven object dynamics, we introduce a reconstruction framework that jointly optimizes the spatio-temporal wind force field and object motion through differentiable rendering and simulation. To ensure physical validity, we incorporate the Lattice Boltzmann Method (LBM) as a physics-informed constraint, enforcing compliance with fluid dynamics laws. Beyond reconstruction, our method naturally supports forward simulation under novel wind conditions and enables new applications such as wind retargeting. We further introduce WD-Objects, a dataset of synthetic and real-world wind-driven scenes. Extensive experiments demonstrate that our method significantly outperforms prior dynamic scene modeling approaches in both reconstruction accuracy and simulation fidelity, opening a new avenue for video-based wind-object interaction modeling.

  </details>



- **EventVGGT: Exploring Cross-Modal Distillation for Consistent Event-based Depth Estimation**  
  Yinrui Ren, Jinjing Zhu, Kanghao Chen, Zhuoxiao Li, Jing Ou, Zidong Cao, Tongyan Hua, Peilun Shi, Yingchun Fu, Wufan Zhao, et al.  
  _2026-03-10_ · https://arxiv.org/abs/2603.09385v1  
  <details><summary>Abstract</summary>

  Event cameras offer superior sensitivity to high-speed motion and extreme lighting, making event-based monocular depth estimation a promising approach for robust 3D perception in challenging conditions. However, progress is severely hindered by the scarcity of dense depth annotations. While recent annotation-free approaches mitigate this by distilling knowledge from Vision Foundation Models (VFMs), a critical limitation persists: they process event streams as independent frames. By neglecting the inherent temporal continuity of event data, these methods fail to leverage the rich temporal priors encoded in VFMs, ultimately yielding temporally inconsistent and less accurate depth predictions. To address this, we introduce EventVGGT, a novel framework that explicitly models the event stream as a coherent video sequence. To the best of our knowledge, we are the first to distill spatio-temporal and multi-view geometric priors from the Visual Geometry Grounded Transformer (VGGT) into the event domain. We achieve this via a comprehensive tri-level distillation strategy: (i) Cross-Modal Feature Mixture (CMFM) bridges the modality gap at the output level by fusing RGB and event features to generate auxiliary depth predictions; (ii) Spatio-Temporal Feature Distillation (STFD) distills VGGT's powerful spatio-temporal representations at the feature level; and (iii) Temporal Consistency Distillation (TCD) enforces cross-frame coherence at the temporal level by aligning inter-frame depth changes. Extensive experiments demonstrate that EventVGGT consistently outperforms existing methods -- reducing the absolute mean depth error at 30m by over 53\% on EventScape (from 2.30 to 1.06) -- while exhibiting robust zero-shot generalization on the unseen DENSE and MVSEC datasets.

  </details>



- **Retrieval-Augmented Gaussian Avatars: Improving Expression Generalization**  
  Matan Levy, Gavriel Habib, Issar Tzachor, Dvir Samuel, Rami Ben-Ari, Nir Darshan, Or Litany, Dani Lischinski  
  _2026-03-09_ · https://arxiv.org/abs/2603.08645v1  
  <details><summary>Abstract</summary>

  Template-free animatable head avatars can achieve high visual fidelity by learning expression-dependent facial deformation directly from a subject's capture, avoiding parametric face templates and hand-designed blendshape spaces. However, since learned deformation is supervised only by the expressions observed for a single identity, these models suffer from limited expression coverage and often struggle when driven by motions that deviate from the training distribution. We introduce RAF (Retrieval-Augmented Faces), a simple training-time augmentation designed for template-free head avatars that learn deformation from data. RAF constructs a large unlabeled expression bank and, during training, replaces a subset of the subject's expression features with nearest-neighbor expressions retrieved from this bank while still reconstructing the subject's original frames. This exposes the deformation field to a broader range of expression conditions, encouraging stronger identity-expression decoupling and improving robustness to expression distribution shift without requiring paired cross-identity data, additional annotations, or architectural changes. We further analyze how retrieval augmentation increases expression diversity and validate retrieval quality with a user study showing that retrieved neighbors are perceptually closer in expression and pose. Experiments on the NeRSemble benchmark demonstrate that RAF consistently improves expression fidelity over the baseline, in both self-driving and cross-driving scenarios.

  </details>


