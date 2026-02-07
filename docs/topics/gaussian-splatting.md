# Gaussian Splatting & 3DGS

_Updated: 2026-02-07 06:59 UTC_

Total papers shown: **9**


---

- **Splat and Distill: Augmenting Teachers with Feed-Forward 3D Reconstruction For 3D-Aware Distillation**  
  David Shavin, Sagie Benaim  
  _2026-02-05_ · https://arxiv.org/abs/2602.06032v1  
  <details><summary>Abstract</summary>

  Vision Foundation Models (VFMs) have achieved remarkable success when applied to various downstream 2D tasks. Despite their effectiveness, they often exhibit a critical lack of 3D awareness. To this end, we introduce Splat and Distill, a framework that instills robust 3D awareness into 2D VFMs by augmenting the teacher model with a fast, feed-forward 3D reconstruction pipeline. Given 2D features produced by a teacher model, our method first lifts these features into an explicit 3D Gaussian representation, in a feedforward manner. These 3D features are then ``splatted" onto novel viewpoints, producing a set of novel 2D feature maps used to supervise the student model, ``distilling" geometrically grounded knowledge. By replacing slow per-scene optimization of prior work with our feed-forward lifting approach, our framework avoids feature-averaging artifacts, creating a dynamic learning process where the teacher's consistency improves alongside that of the student. We conduct a comprehensive evaluation on a suite of downstream tasks, including monocular depth estimation, surface normal estimation, multi-view correspondence, and semantic segmentation. Our method significantly outperforms prior works, not only achieving substantial gains in 3D awareness but also enhancing the underlying semantic richness of 2D features. Project page is available at https://davidshavin4.github.io/Splat-and-Distill/

  </details>



- **NVS-HO: A Benchmark for Novel View Synthesis of Handheld Objects**  
  Musawar Ali, Manuel Carranza-García, Nicola Fioraio, Samuele Salti, Luigi Di Stefano  
  _2026-02-05_ · https://arxiv.org/abs/2602.05822v1  
  <details><summary>Abstract</summary>

  We propose NVS-HO, the first benchmark designed for novel view synthesis of handheld objects in real-world environments using only RGB inputs. Each object is recorded in two complementary RGB sequences: (1) a handheld sequence, where the object is manipulated in front of a static camera, and (2) a board sequence, where the object is fixed on a ChArUco board to provide accurate camera poses via marker detection. The goal of NVS-HO is to learn a NVS model that captures the full appearance of an object from (1), whereas (2) provides the ground-truth images used for evaluation. To establish baselines, we consider both a classical SfM pipeline and a state-of-the-art pre-trained feed-forward neural network (VGGT) as pose estimators, and train NVS models based on NeRF and Gaussian Splatting. Our experiments reveal significant performance gaps in current methods under unconstrained handheld conditions, highlighting the need for more robust approaches. NVS-HO thus offers a challenging real-world benchmark to drive progress in RGB-based novel view synthesis of handheld objects.

  </details>



- **Unified Sensor Simulation for Autonomous Driving**  
  Nikolay Patakin, Arsenii Shirokov, Anton Konushin, Dmitry Senushkin  
  _2026-02-05_ · https://arxiv.org/abs/2602.05617v1  
  <details><summary>Abstract</summary>

  In this work, we introduce \textbf{XSIM}, a sensor simulation framework for autonomous driving. XSIM extends 3DGUT splatting with a generalized rolling-shutter modeling tailored for autonomous driving applications. Our framework provides a unified and flexible formulation for appearance and geometric sensor modeling, enabling rendering of complex sensor distortions in dynamic environments. We identify spherical cameras, such as LiDARs, as a critical edge case for existing 3DGUT splatting due to cyclic projection and time discontinuities at azimuth boundaries leading to incorrect particle projection. To address this issue, we propose a phase modeling mechanism that explicitly accounts temporal and shape discontinuities of Gaussians projected by the Unscented Transform at azimuth borders. In addition, we introduce an extended 3D Gaussian representation that incorporates two distinct opacity parameters to resolve mismatches between geometry and color distributions. As a result, our framework provides enhanced scene representations with improved geometric consistency and photorealistic appearance. We evaluate our framework extensively on multiple autonomous driving datasets, including Waymo Open Dataset, Argoverse 2, and PandaSet. Our framework consistently outperforms strong recent baselines and achieves state-of-the-art performance across all datasets. The source code is publicly available at \href{https://github.com/whesense/XSIM}{https://github.com/whesense/XSIM}.

  </details>



- **PoseGaussian: Pose-Driven Novel View Synthesis for Robust 3D Human Reconstruction**  
  Ju Shen, Chen Chen, Tam V. Nguyen, Vijayan K. Asari  
  _2026-02-05_ · https://arxiv.org/abs/2602.05190v1  
  <details><summary>Abstract</summary>

  We propose PoseGaussian, a pose-guided Gaussian Splatting framework for high-fidelity human novel view synthesis. Human body pose serves a dual purpose in our design: as a structural prior, it is fused with a color encoder to refine depth estimation; as a temporal cue, it is processed by a dedicated pose encoder to enhance temporal consistency across frames. These components are integrated into a fully differentiable, end-to-end trainable pipeline. Unlike prior works that use pose only as a condition or for warping, PoseGaussian embeds pose signals into both geometric and temporal stages to improve robustness and generalization. It is specifically designed to address challenges inherent in dynamic human scenes, such as articulated motion and severe self-occlusion. Notably, our framework achieves real-time rendering at 100 FPS, maintaining the efficiency of standard Gaussian Splatting pipelines. We validate our approach on ZJU-MoCap, THuman2.0, and in-house datasets, demonstrating state-of-the-art performance in perceptual quality and structural accuracy (PSNR 30.86, SSIM 0.979, LPIPS 0.028).

  </details>



- **QuantumGS: Quantum Encoding Framework for Gaussian Splatting**  
  Grzegorz Wilczyński, Rafał Tobiasz, Paweł Gora, Marcin Mazur, Przemysław Spurek  
  _2026-02-04_ · https://arxiv.org/abs/2602.05047v1  
  <details><summary>Abstract</summary>

  Recent advances in neural rendering, particularly 3D Gaussian Splatting (3DGS), have enabled real-time rendering of complex scenes. However, standard 3DGS relies on spherical harmonics, which often struggle to accurately capture high-frequency view-dependent effects such as sharp reflections and transparency. While hybrid approaches like Viewing Direction Gaussian Splatting (VDGS) mitigate this limitation using classical Multi-Layer Perceptrons (MLPs), they remain limited by the expressivity of classical networks in low-parameter regimes. In this paper, we introduce QuantumGS, a novel hybrid framework that integrates Variational Quantum Circuits (VQC) into the Gaussian Splatting pipeline. We propose a unique encoding strategy that maps the viewing direction directly onto the Bloch sphere, leveraging the natural geometry of qubits to represent 3D directional data. By replacing classical color-modulating networks with quantum circuits generated via a hypernetwork or conditioning mechanism, we achieve higher expressivity and better generalization. Source code is available in the supplementary material. Code is available at https://github.com/gwilczynski95/QuantumGS

  </details>



- **Nix and Fix: Targeting 1000x Compression of 3D Gaussian Splatting with Diffusion Models**  
  Cem Eteke, Enzo Tartaglione  
  _2026-02-04_ · https://arxiv.org/abs/2602.04549v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) revolutionized novel view rendering. Instead of inferring from dense spatial points, as implicit representations do, 3DGS uses sparse Gaussians. This enables real-time performance but increases space requirements, hindering applications such as immersive communication. 3DGS compression emerged as a field aimed at alleviating this issue. While impressive progress has been made, at low rates, compression introduces artifacts that degrade visual quality significantly. We introduce NiFi, a method for extreme 3DGS compression through restoration via artifact-aware, diffusion-based one-step distillation. We show that our method achieves state-of-the-art perceptual quality at extremely low rates, down to 0.1 MB, and towards 1000x rate improvement over 3DGS at comparable perceptual performance. The code will be open-sourced upon acceptance.

  </details>



- **VecSet-Edit: Unleashing Pre-trained LRM for Mesh Editing from Single Image**  
  Teng-Fang Hsiao, Bo-Kai Ruan, Yu-Lun Liu, Hong-Han Shuai  
  _2026-02-04_ · https://arxiv.org/abs/2602.04349v1  
  <details><summary>Abstract</summary>

  3D editing has emerged as a critical research area to provide users with flexible control over 3D assets. While current editing approaches predominantly focus on 3D Gaussian Splatting or multi-view images, the direct editing of 3D meshes remains underexplored. Prior attempts, such as VoxHammer, rely on voxel-based representations that suffer from limited resolution and necessitate labor-intensive 3D mask. To address these limitations, we propose \textbf{VecSet-Edit}, the first pipeline that leverages the high-fidelity VecSet Large Reconstruction Model (LRM) as a backbone for mesh editing. Our approach is grounded on a analysis of the spatial properties in VecSet tokens, revealing that token subsets govern distinct geometric regions. Based on this insight, we introduce Mask-guided Token Seeding and Attention-aligned Token Gating strategies to precisely localize target regions using only 2D image conditions. Also, considering the difference between VecSet diffusion process versus voxel we design a Drift-aware Token Pruning to reject geometric outliers during the denoising process. Finally, our Detail-preserving Texture Baking module ensures that we not only preserve the geometric details of original mesh but also the textural information. More details can be found in our project page: https://github.com/BlueDyee/VecSet-Edit/tree/main

  </details>



- **JOintGS: Joint Optimization of Cameras, Bodies and 3D Gaussians for In-the-Wild Monocular Reconstruction**  
  Zihan Lou, Jinlong Fan, Sihan Ma, Yuxiang Yang, Jing Zhang  
  _2026-02-04_ · https://arxiv.org/abs/2602.04317v1  
  <details><summary>Abstract</summary>

  Reconstructing high-fidelity animatable 3D human avatars from monocular RGB videos remains challenging, particularly in unconstrained in-the-wild scenarios where camera parameters and human poses from off-the-shelf methods (e.g., COLMAP, HMR2.0) are often inaccurate. Splatting (3DGS) advances demonstrate impressive rendering quality and real-time performance, they critically depend on precise camera calibration and pose annotations, limiting their applicability in real-world settings. We present JOintGS, a unified framework that jointly optimizes camera extrinsics, human poses, and 3D Gaussian representations from coarse initialization through a synergistic refinement mechanism. Our key insight is that explicit foreground-background disentanglement enables mutual reinforcement: static background Gaussians anchor camera estimation via multi-view consistency; refined cameras improve human body alignment through accurate temporal correspondence; optimized human poses enhance scene reconstruction by removing dynamic artifacts from static constraints. We further introduce a temporal dynamics module to capture fine-grained pose-dependent deformations and a residual color field to model illumination variations. Extensive experiments on NeuMan and EMDB datasets demonstrate that JOintGS achieves superior reconstruction quality, with 2.1~dB PSNR improvement over state-of-the-art methods on NeuMan dataset, while maintaining real-time rendering. Notably, our method shows significantly enhanced robustness to noisy initialization compared to the baseline.Our source code is available at https://github.com/MiliLab/JOintGS.

  </details>



- **SkeletonGaussian: Editable 4D Generation through Gaussian Skeletonization**  
  Lifan Wu, Ruijie Zhu, Yubo Ai, Tianzhu Zhang  
  _2026-02-04_ · https://arxiv.org/abs/2602.04271v1  
  <details><summary>Abstract</summary>

  4D generation has made remarkable progress in synthesizing dynamic 3D objects from input text, images, or videos. However, existing methods often represent motion as an implicit deformation field, which limits direct control and editability. To address this issue, we propose SkeletonGaussian, a novel framework for generating editable dynamic 3D Gaussians from monocular video input. Our approach introduces a hierarchical articulated representation that decomposes motion into sparse rigid motion explicitly driven by a skeleton and fine-grained non-rigid motion. Concretely, we extract a robust skeleton and drive rigid motion via linear blend skinning, followed by a hexplane-based refinement for non-rigid deformations, enhancing interpretability and editability. Experimental results demonstrate that SkeletonGaussian surpasses existing methods in generation quality while enabling intuitive motion editing, establishing a new paradigm for editable 4D generation. Project page: https://wusar.github.io/projects/skeletongaussian/

  </details>


