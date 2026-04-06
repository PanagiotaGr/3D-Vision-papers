# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-04-06 08:00 UTC_

Total papers shown: **7**


---

- **Multi-View Video Diffusion Policy: A 3D Spatio-Temporal-Aware Video Action Model**  
  Peiyan Li, Yixiang Chen, Yuan Xu, Jiabing Yang, Xiangnan Wu, Jun Guo, Nan Sun, Long Qian, Xinghang Li, Xin Xiao, et al.  
  _2026-04-03_ · https://arxiv.org/abs/2604.03181v1  
  <details><summary>Abstract</summary>

  Robotic manipulation requires understanding both the 3D spatial structure of the environment and its temporal evolution, yet most existing policies overlook one or both. They typically rely on 2D visual observations and backbones pretrained on static image--text pairs, resulting in high data requirements and limited understanding of environment dynamics. To address this, we introduce MV-VDP, a multi-view video diffusion policy that jointly models the 3D spatio-temporal state of the environment. The core idea is to simultaneously predict multi-view heatmap videos and RGB videos, which 1) align the representation format of video pretraining with action finetuning, and 2) specify not only what actions the robot should take, but also how the environment is expected to evolve in response to those actions. Extensive experiments show that MV-VDP enables data-efficient, robust, generalizable, and interpretable manipulation. With only ten demonstration trajectories and without additional pretraining, MV-VDP successfully performs complex real-world tasks, demonstrates strong robustness across a range of model hyperparameters, generalizes to out-of-distribution settings, and predicts realistic future videos. Experiments on Meta-World and real-world robotic platforms demonstrate that MV-VDP consistently outperforms video-prediction--based, 3D-based, and vision--language--action models, establishing a new state of the art in data-efficient multi-task manipulation.

  </details>



- **Explicit Time-Frequency Dynamics for Skeleton-Based Gait Recognition**  
  Seoyeon Ko, Yeojin Song, Egene Chung, Luca Quagliato, Taeyong Lee, Junhyug Noh  
  _2026-04-03_ · https://arxiv.org/abs/2604.03002v1  
  <details><summary>Abstract</summary>

  Skeleton-based gait recognizers excel at modeling spatial configurations but often underuse explicit motion dynamics that are crucial under appearance changes. We introduce a plug-and-play Wavelet Feature Stream that augments any skeleton backbone with time-frequency dynamics of joint velocities. Concretely, per-joint velocity sequences are transformed by the continuous wavelet transform (CWT) into multi-scale scalograms, from which a lightweight multi-scale CNN learns discriminative dynamic cues. The resulting descriptor is fused with the backbone representation for classification, requiring no changes to the backbone architecture or additional supervision. Across CASIA-B, the proposed stream delivers consistent gains on strong skeleton backbones (e.g., GaitMixer, GaitFormer, GaitGraph) and establishes a new skeleton-based state of the art when attached to GaitMixer. The improvements are especially pronounced under covariate shifts such as carrying bags (BG) and wearing coats (CL), highlighting the complementarity of explicit time-frequency modeling and standard spatio-temporal encoders.

  </details>



- **Rendering Multi-Human and Multi-Object with 3D Gaussian Splatting**  
  Weiquan Wang, Jun Xiao, Feifei Shao, Yi Yang, Yueting Zhuang, Long Chen  
  _2026-04-03_ · https://arxiv.org/abs/2604.02996v1  
  <details><summary>Abstract</summary>

  Reconstructing dynamic scenes with multiple interacting humans and objects from sparse-view inputs is a critical yet challenging task, essential for creating high-fidelity digital twins for robotics and VR/AR. This problem, which we term Multi-Human Multi-Object (MHMO) rendering, presents two significant obstacles: achieving view-consistent representations for individual instances under severe mutual occlusion, and explicitly modeling the complex and combinatorial dependencies that arise from their interactions. To overcome these challenges, we propose MM-GS, a novel hierarchical framework built upon 3D Gaussian Splatting. Our method first employs a Per-Instance Multi-View Fusion module to establish a robust and consistent representation for each instance by aggregating visual information across all available views. Subsequently, a Scene-Level Instance Interaction module operates on a global scene graph to reason about relationships between all participants, refining their attributes to capture subtle interaction effects. Extensive experiments on challenging datasets demonstrate that our method significantly outperforms strong baselines, producing state-of-the-art results with high-fidelity details and plausible inter-instance contacts.

  </details>



- **Learning from Synthetic Data via Provenance-Based Input Gradient Guidance**  
  Koshiro Nagano, Ryo Fujii, Ryo Hachiuma, Fumiaki Sato, Taiki Sekii, Hideo Saito  
  _2026-04-03_ · https://arxiv.org/abs/2604.02946v1  
  <details><summary>Abstract</summary>

  Learning methods using synthetic data have attracted attention as an effective approach for increasing the diversity of training data while reducing collection costs, thereby improving the robustness of model discrimination. However, many existing methods improve robustness only indirectly through the diversification of training samples and do not explicitly teach the model which regions in the input space truly contribute to discrimination; consequently, the model may learn spurious correlations caused by synthesis biases and artifacts. Motivated by this limitation, this paper proposes a learning framework that uses provenance information obtained during the training data synthesis process, indicating whether each region in the input space originates from the target object, as an auxiliary supervisory signal to promote the acquisition of representations focused on target regions. Specifically, input gradients are decomposed based on information about target and non-target regions during synthesis, and input gradient guidance is introduced to suppress gradients over non-target regions. This suppresses the model's reliance on non-target regions and directly promotes the learning of discriminative representations for target regions. Experiments demonstrate the effectiveness and generality of the proposed method across multiple tasks and modalities, including weakly supervised object localization, spatio-temporal action localization, and image classification.

  </details>



- **MMTalker: Multiresolution 3D Talking Head Synthesis with Multimodal Feature Fusion**  
  Bin Liu, Zhixiang Xiong, Zhifen He, Bo Li  
  _2026-04-03_ · https://arxiv.org/abs/2604.02941v1  
  <details><summary>Abstract</summary>

  Speech-driven three-dimensional (3D) facial animation synthesis aims to build a mapping from one-dimensional (1D) speech signals to time-varying 3D facial motion signals. Current methods still face challenges in maintaining lip-sync accuracy and producing realistic facial expressions, primarily due to the highly ill-posed nature of this cross-modal mapping. In this paper, we introduce a novel 3D audio-driven facial animation synthesis method through multi-resolution representation and multi-modal feature fusion, called MMTalker which can accurately reconstruct the rich details of 3D facial motion. We first achieve the continuous representation of 3D face with details by mesh parameterization and non-uniform differentiable sampling. The mesh parameterization technique establishes the correspondence between UV plane and 3D facial mesh and is used to offer ground truth for the continuous learning. Differentiable non-uniform sampling enables precise facial detail acquisition by setting learnable sampling probability in each triangular face. Next, we employ residual graph convolutional network and dual cross-attention mechanism to extract discriminative facial motion feature from multiple input modalities. This proposed multimodal fusion strategy takes full use of the hierarchical features of speech and the explicit spatiotemporal geometric features of facial mesh. Finally, a lightweight regression network predicts the vertex-wise geometric displacements of the synthesized talking face by jointly processing the sampled points in the canonical UV space and the encoded facial motion features. Comprehensive experiments demonstrate that significant improvements are achieved over state-of-the-art methods, especially in the synchronization accuracy of lip and eye movements.

  </details>



- **BEVPredFormer: Spatio-temporal Attention for BEV Instance Prediction in Autonomous Driving**  
  Miguel Antunes-García, Santiago Montiel-Marín, Fabio Sánchez-García, Rodrigo Gutiérrez-Moreno, Rafael Barea, Luis M. Bergasa  
  _2026-04-03_ · https://arxiv.org/abs/2604.02930v1  
  <details><summary>Abstract</summary>

  A robust awareness of how dynamic scenes evolve is essential for Autonomous Driving systems, as they must accurately detect, track, and predict the behaviour of surrounding obstacles. Traditional perception pipelines that rely on modular architectures tend to suffer from cumulative errors and latency. Instance Prediction models provide a unified solution, performing Bird's-Eye-View segmentation and motion estimation across current and future frames using information directly obtained from different sensors. However, a key challenge in these models lies in the effective processing of the dense spatial and temporal information inherent in dynamic driving environments. This level of complexity demands architectures capable of capturing fine-grained motion patterns and long-range dependencies without compromising real-time performance. We introduce BEVPredFormer, a novel camera-only architecture for BEV instance prediction that uses attention-based temporal processing to improve temporal and spatial comprehension of the scene and relies on an attention-based 3D projection of the camera information. BEVPredFormer employs a recurrent-free design that incorporates gated transformer layers, divided spatio-temporal attention mechanisms, and multi-scale head tasks. Additionally, we incorporate a difference-guided feature extraction module that enhances temporal representations. Extensive ablation studies validate the effectiveness of each architectural component. When evaluated on the nuScenes dataset, BEVPredFormer was on par or surpassed State-Of-The-Art methods, highlighting its potential for robust and efficient Autonomous Driving perception.

  </details>



- **GP-4DGS: Probabilistic 4D Gaussian Splatting from Monocular Video via Variational Gaussian Processes**  
  Mijeong Kim, Jungtaek Kim, Bohyung Han  
  _2026-04-03_ · https://arxiv.org/abs/2604.02915v1  
  <details><summary>Abstract</summary>

  We present GP-4DGS, a novel framework that integrates Gaussian Processes (GPs) into 4D Gaussian Splatting (4DGS) for principled probabilistic modeling of dynamic scenes. While existing 4DGS methods focus on deterministic reconstruction, they are inherently limited in capturing motion ambiguity and lack mechanisms to assess prediction reliability. By leveraging the kernel-based probabilistic nature of GPs, our approach introduces three key capabilities: (i) uncertainty quantification for motion predictions, (ii) motion estimation for unobserved or sparsely sampled regions, and (iii) temporal extrapolation beyond observed training frames. To scale GPs to the large number of Gaussian primitives in 4DGS, we design spatio-temporal kernels that capture the correlation structure of deformation fields and adopt variational Gaussian Processes with inducing points for tractable inference. Our experiments show that GP-4DGS enhances reconstruction quality while providing reliable uncertainty estimates that effectively identify regions of high motion ambiguity. By addressing these challenges, our work takes a meaningful step toward bridging probabilistic modeling and neural graphics.

  </details>


