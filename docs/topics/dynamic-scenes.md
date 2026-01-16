# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-01-16 13:08 UTC_

Total papers shown: **8**


---

- **Beyond Inpainting: Unleash 3D Understanding for Precise Camera-Controlled Video Generation**  
  Dong-Yu Chen, Yixin Guo, Shuojin Yang, Tai-Jiang Mu, Shi-Min Hu  
  _2026-01-15_ · https://arxiv.org/abs/2601.10214v1  
  <details><summary>Abstract</summary>

  Camera control has been extensively studied in conditioned video generation; however, performing precisely altering the camera trajectories while faithfully preserving the video content remains a challenging task. The mainstream approach to achieving precise camera control is warping a 3D representation according to the target trajectory. However, such methods fail to fully leverage the 3D priors of video diffusion models (VDMs) and often fall into the Inpainting Trap, resulting in subject inconsistency and degraded generation quality. To address this problem, we propose DepthDirector, a video re-rendering framework with precise camera controllability. By leveraging the depth video from explicit 3D representation as camera-control guidance, our method can faithfully reproduce the dynamic scene of an input video under novel camera trajectories. Specifically, we design a View-Content Dual-Stream Condition mechanism that injects both the source video and the warped depth sequence rendered under the target viewpoint into the pretrained video generation model. This geometric guidance signal enables VDMs to comprehend camera movements and leverage their 3D understanding capabilities, thereby facilitating precise camera control and consistent content generation. Next, we introduce a lightweight LoRA-based video diffusion adapter to train our framework, fully preserving the knowledge priors of VDMs. Additionally, we construct a large-scale multi-camera synchronized dataset named MultiCam-WarpData using Unreal Engine 5, containing 8K videos across 1K dynamic scenes. Extensive experiments show that DepthDirector outperforms existing methods in both camera controllability and visual quality. Our code and dataset will be publicly available.

  </details>



- **CogRail: Benchmarking VLMs in Cognitive Intrusion Perception for Intelligent Railway Transportation Systems**  
  Yonglin Tian, Qiyao Zhang, Wei Xu, Yutong Wang, Yihao Wu, Xinyi Li, Xingyuan Dai, Hui Zhang, Zhiyong Cui, Baoqing Guo, et al.  
  _2026-01-14_ · https://arxiv.org/abs/2601.09613v1  
  <details><summary>Abstract</summary>

  Accurate and early perception of potential intrusion targets is essential for ensuring the safety of railway transportation systems. However, most existing systems focus narrowly on object classification within fixed visual scopes and apply rule-based heuristics to determine intrusion status, often overlooking targets that pose latent intrusion risks. Anticipating such risks requires the cognition of spatial context and temporal dynamics for the object of interest (OOI), which presents challenges for conventional visual models. To facilitate deep intrusion perception, we introduce a novel benchmark, CogRail, which integrates curated open-source datasets with cognitively driven question-answer annotations to support spatio-temporal reasoning and prediction. Building upon this benchmark, we conduct a systematic evaluation of state-of-the-art visual-language models (VLMs) using multimodal prompts to identify their strengths and limitations in this domain. Furthermore, we fine-tune VLMs for better performance and propose a joint fine-tuning framework that integrates three core tasks, position perception, movement prediction, and threat analysis, facilitating effective adaptation of general-purpose foundation models into specialized models tailored for cognitive intrusion perception. Extensive experiments reveal that current large-scale multimodal models struggle with the complex spatial-temporal reasoning required by the cognitive intrusion perception task, underscoring the limitations of existing foundation models in this safety-critical domain. In contrast, our proposed joint fine-tuning framework significantly enhances model performance by enabling targeted adaptation to domain-specific reasoning demands, highlighting the advantages of structured multi-task learning in improving both accuracy and interpretability. Code will be available at https://github.com/Hub-Tian/CogRail.

  </details>



- **Trustworthy Longitudinal Brain MRI Completion: A Deformation-Based Approach with KAN-Enhanced Diffusion Model**  
  Tianli Tao, Ziyang Wang, Delong Yang, Han Zhang, Le Zhang  
  _2026-01-14_ · https://arxiv.org/abs/2601.09572v1  
  <details><summary>Abstract</summary>

  Longitudinal brain MRI is essential for lifespan study, yet high attrition rates often lead to missing data, complicating analysis. Deep generative models have been explored, but most rely solely on image intensity, leading to two key limitations: 1) the fidelity or trustworthiness of the generated brain images are limited, making downstream studies questionable; 2) the usage flexibility is restricted due to fixed guidance rooted in the model structure, restricting full ability to versatile application scenarios. To address these challenges, we introduce DF-DiffCom, a Kolmogorov-Arnold Networks (KAN)-enhanced diffusion model that smartly leverages deformation fields for trustworthy longitudinal brain image completion. Trained on OASIS-3, DF-DiffCom outperforms state-of-the-art methods, improving PSNR by 5.6% and SSIM by 0.12. More importantly, its modality-agnostic nature allows smooth extension to varied MRI modalities, even to attribute maps such as brain tissue segmentation results.

  </details>



- **Learning Whole-Body Human-Humanoid Interaction from Human-Human Demonstrations**  
  Wei-Jin Huang, Yue-Yi Zhang, Yi-Lin Wei, Zhi-Wei Xia, Juantao Tan, Yuan-Ming Li, Zhilin Zhao, Wei-Shi Zheng  
  _2026-01-14_ · https://arxiv.org/abs/2601.09518v1  
  <details><summary>Abstract</summary>

  Enabling humanoid robots to physically interact with humans is a critical frontier, but progress is hindered by the scarcity of high-quality Human-Humanoid Interaction (HHoI) data. While leveraging abundant Human-Human Interaction (HHI) data presents a scalable alternative, we first demonstrate that standard retargeting fails by breaking the essential contacts. We address this with PAIR (Physics-Aware Interaction Retargeting), a contact-centric, two-stage pipeline that preserves contact semantics across morphology differences to generate physically consistent HHoI data. This high-quality data, however, exposes a second failure: conventional imitation learning policies merely mimic trajectories and lack interactive understanding. We therefore introduce D-STAR (Decoupled Spatio-Temporal Action Reasoner), a hierarchical policy that disentangles when to act from where to act. In D-STAR, Phase Attention (when) and a Multi-Scale Spatial module (where) are fused by the diffusion head to produce synchronized whole-body behaviors beyond mimicry. By decoupling these reasoning streams, our model learns robust temporal phases without being distracted by spatial noise, leading to responsive, synchronized collaboration. We validate our framework through extensive and rigorous simulations, demonstrating significant performance gains over baseline approaches and a complete, effective pipeline for learning complex whole-body interactions from HHI data.

  </details>



- **V-DPM: 4D Video Reconstruction with Dynamic Point Maps**  
  Edgar Sucar, Eldar Insafutdinov, Zihang Lai, Andrea Vedaldi  
  _2026-01-14_ · https://arxiv.org/abs/2601.09499v1  
  <details><summary>Abstract</summary>

  Powerful 3D representations such as DUSt3R invariant point maps, which encode 3D shape and camera parameters, have significantly advanced feed forward 3D reconstruction. While point maps assume static scenes, Dynamic Point Maps (DPMs) extend this concept to dynamic 3D content by additionally representing scene motion. However, existing DPMs are limited to image pairs and, like DUSt3R, require post processing via optimization when more than two views are involved. We argue that DPMs are more useful when applied to videos and introduce V-DPM to demonstrate this. First, we show how to formulate DPMs for video input in a way that maximizes representational power, facilitates neural prediction, and enables reuse of pretrained models. Second, we implement these ideas on top of VGGT, a recent and powerful 3D reconstructor. Although VGGT was trained on static scenes, we show that a modest amount of synthetic data is sufficient to adapt it into an effective V-DPM predictor. Our approach achieves state of the art performance in 3D and 4D reconstruction for dynamic scenes. In particular, unlike recent dynamic extensions of VGGT such as P3, DPMs recover not only dynamic depth but also the full 3D motion of every point in the scene.

  </details>



- **GaussianFluent: Gaussian Simulation for Dynamic Scenes with Mixed Materials**  
  Bei Huang, Yixin Chen, Ruijie Lu, Gang Zeng, Hongbin Zha, Yuru Pei, Siyuan Huang  
  _2026-01-14_ · https://arxiv.org/abs/2601.09265v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has emerged as a prominent 3D representation for high-fidelity and real-time rendering. Prior work has coupled physics simulation with Gaussians, but predominantly targets soft, deformable materials, leaving brittle fracture largely unresolved. This stems from two key obstacles: the lack of volumetric interiors with coherent textures in GS representation, and the absence of fracture-aware simulation methods for Gaussians. To address these challenges, we introduce GaussianFluent, a unified framework for realistic simulation and rendering of dynamic object states. First, it synthesizes photorealistic interiors by densifying internal Gaussians guided by generative models. Second, it integrates an optimized Continuum Damage Material Point Method (CD-MPM) to enable brittle fracture simulation at remarkably high speed. Our approach handles complex scenarios including mixed-material objects and multi-stage fracture propagation, achieving results infeasible with previous methods. Experiments clearly demonstrate GaussianFluent's capability for photo-realistic, real-time rendering with structurally consistent interiors, highlighting its potential for downstream application, such as VR and Robotics.

  </details>



- **Point Tracking as a Temporal Cue for Robust Myocardial Segmentation in Echocardiography Videos**  
  Bahar Khodabakhshian, Nima Hashemi, Armin Saadat, Zahra Gholami, In-Chang Hwang, Samira Sojoudi, Christina Luong, Purang Abolmaesumi, Teresa Tsang  
  _2026-01-14_ · https://arxiv.org/abs/2601.09207v1  
  <details><summary>Abstract</summary>

  Purpose: Myocardium segmentation in echocardiography videos is a challenging task due to low contrast, noise, and anatomical variability. Traditional deep learning models either process frames independently, ignoring temporal information, or rely on memory-based feature propagation, which accumulates error over time. Methods: We propose Point-Seg, a transformer-based segmentation framework that integrates point tracking as a temporal cue to ensure stable and consistent segmentation of myocardium across frames. Our method leverages a point-tracking module trained on a synthetic echocardiography dataset to track key anatomical landmarks across video sequences. These tracked trajectories provide an explicit motion-aware signal that guides segmentation, reducing drift and eliminating the need for memory-based feature accumulation. Additionally, we incorporate a temporal smoothing loss to further enhance temporal consistency across frames. Results: We evaluate our approach on both public and private echocardiography datasets. Experimental results demonstrate that Point-Seg has statistically similar accuracy in terms of Dice to state-of-the-art segmentation models in high quality echo data, while it achieves better segmentation accuracy in lower quality echo with improved temporal stability. Furthermore, Point-Seg has the key advantage of pixel-level myocardium motion information as opposed to other segmentation methods. Such information is essential in the computation of other downstream tasks such as myocardial strain measurement and regional wall motion abnormality detection. Conclusion: Point-Seg demonstrates that point tracking can serve as an effective temporal cue for consistent video segmentation, offering a reliable and generalizable approach for myocardium segmentation in echocardiography videos. The code is available at https://github.com/DeepRCL/PointSeg.

  </details>



- **TRACE: Reconstruction-Based Anomaly Detection in Ensemble and Time-Dependent Simulations**  
  Hamid Gadirov, Martijn Westra, Steffen Frey  
  _2026-01-13_ · https://arxiv.org/abs/2601.08659v1  
  <details><summary>Abstract</summary>

  Detecting anomalies in high-dimensional, time-dependent simulation data is challenging due to complex spatial and temporal dynamics. We study reconstruction-based anomaly detection for ensemble data from parameterized Kármán vortex street simulations using convolutional autoencoders. We compare a 2D autoencoder operating on individual frames with a 3D autoencoder that processes short temporal stacks. The 2D model identifies localized spatial irregularities in single time steps, while the 3D model exploits spatio-temporal context to detect anomalous motion patterns and reduces redundant detections across time. We further evaluate volumetric time-dependent data and find that reconstruction errors are strongly influenced by the spatial distribution of mass, with highly concentrated regions yielding larger errors than dispersed configurations. Our results highlight the importance of temporal context for robust anomaly detection in dynamic simulations.

  </details>


