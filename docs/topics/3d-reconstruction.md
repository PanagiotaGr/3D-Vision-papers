# 3D Reconstruction

_Updated: 2026-03-12 07:11 UTC_

Total papers shown: **21**


---

- **Neural Field Thermal Tomography: A Differentiable Physics Framework for Non-Destructive Evaluation**  
  Tao Zhong, Yixun Hu, Dongzhe Zheng, Aditya Sood, Christine Allen-Blanchette  
  _2026-03-11_ · https://arxiv.org/abs/2603.11045v1  
  <details><summary>Abstract</summary>

  We propose Neural Field Thermal Tomography (NeFTY), a differentiable physics framework for the quantitative 3D reconstruction of material properties from transient surface temperature measurements. While traditional thermography relies on pixel-wise 1D approximations that neglect lateral diffusion, and soft-constrained Physics-Informed Neural Networks (PINNs) often fail in transient diffusion scenarios due to gradient stiffness, NeFTY parameterizes the 3D diffusivity field as a continuous neural field optimized through a rigorous numerical solver. By leveraging a differentiable physics solver, our approach enforces thermodynamic laws as hard constraints while maintaining the memory efficiency required for high-resolution 3D tomography. Our discretize-then-optimize paradigm effectively mitigates the spectral bias and ill-posedness inherent in inverse heat conduction, enabling the recovery of subsurface defects at arbitrary scales. Experimental validation on synthetic data demonstrates that NeFTY significantly improves the accuracy of subsurface defect localization over baselines. Additional details at https://cab-lab-princeton.github.io/nefty/

  </details>



- **TreeON: Reconstructing 3D Tree Point Clouds from Orthophotos and Heightmaps**  
  Angeliki Grammatikaki, Johannes Eschner, Pedro Hermosilla, Oscar Argudo, Manuela Waldner  
  _2026-03-11_ · https://arxiv.org/abs/2603.10996v1  
  <details><summary>Abstract</summary>

  We present TreeON, a novel neural-based framework for reconstructing detailed 3D tree point clouds from sparse top-down geodata, using only a single orthophoto and its corresponding Digital Surface Model (DSM). Our method introduces a new training supervision strategy that combines both geometric supervision and differentiable shadow and silhouette losses to learn point cloud representations of trees without requiring species labels, procedural rules, terrestrial reconstruction data, or ground laser scans. To address the lack of ground truth data, we generate a synthetic dataset of point clouds from procedurally modeled trees and train our network on it. Quantitative and qualitative experiments demonstrate better reconstruction quality and coverage compared to existing methods, as well as strong generalization to real-world data, producing visually appealing and structurally plausible tree point cloud representations suitable for integration into interactive digital 3D maps. The codebase, synthetic dataset, and pretrained model are publicly available at https://angelikigram.github.io/treeON/.

  </details>



- **Pointy - A Lightweight Transformer for Point Cloud Foundation Models**  
  Konrad Szafer, Marek Kraft, Dominik Belter  
  _2026-03-11_ · https://arxiv.org/abs/2603.10963v1  
  <details><summary>Abstract</summary>

  Foundation models for point cloud data have recently grown in capability, often leveraging extensive representation learning from language or vision. In this work, we take a more controlled approach by introducing a lightweight transformer-based point cloud architecture. In contrast to the heavy reliance on cross-modal supervision, our model is trained only on 39k point clouds - yet it outperforms several larger foundation models trained on over 200k training samples. Interestingly, our method approaches state-of-the-art results from models that have seen over a million point clouds, images, and text samples, demonstrating the value of a carefully curated training setup and architecture. To ensure rigorous evaluation, we conduct a comprehensive replication study that standardizes the training regime and benchmarks across multiple point cloud architectures. This unified experimental framework isolates the impact of architectural choices, allowing for transparent comparisons and highlighting the benefits of our design and other tokenizer-free architectures. Our results show that simple backbones can deliver competitive results to more complex or data-rich strategies. The implementation, including code, pre-trained models, and training protocols, is available at https://github.com/KonradSzafer/Pointy.

  </details>



- **S2D: Sparse to Dense Lifting for 3D Reconstruction with Minimal Inputs**  
  Yuzhou Ji, Qijian Tian, He Zhu, Xiaoqi Jiang, Guangzhi Cao, Lizhuang Ma, Yuan Xie, Xin Tan  
  _2026-03-11_ · https://arxiv.org/abs/2603.10893v1  
  <details><summary>Abstract</summary>

  Explicit 3D representations have already become an essential medium for 3D simulation and understanding. However, the most commonly used point cloud and 3D Gaussian Splatting (3DGS) each suffer from non-photorealistic rendering and significant degradation under sparse inputs. In this paper, we introduce Sparse to Dense lifting (S2D), a novel pipeline that bridges the two representations and achieves high-quality 3DGS reconstruction with minimal inputs. Specifically, the S2D lifting is two-fold. We first present an efficient one-step diffusion model that lifts sparse point cloud for high-fidelity image artifact fixing. Meanwhile, to reconstruct 3D consistent scenes, we also design a corresponding reconstruction strategy with random sample drop and weighted gradient for robust model fitting from sparse input views to dense novel views. Extensive experiments show that S2D achieves the best consistency in generating novel view guidance and first-tier sparse view reconstruction quality under different input sparsity. By reconstructing stable scenes with the least possible captures among existing methods, S2D enables minimal input requirements for 3DGS applications.

  </details>



- **PolGS++: Physically-Guided Polarimetric Gaussian Splatting for Fast Reflective Surface Reconstruction**  
  Yufei Han, Chu Zhou, Youwei Lyu, Qi Chen, Si Li, Boxin Shi, Yunpeng Jia, Heng Guo, Zhanyu Ma  
  _2026-03-11_ · https://arxiv.org/abs/2603.10801v1  
  <details><summary>Abstract</summary>

  Accurate reconstruction of reflective surfaces remains a fundamental challenge in computer vision, with broad applications in real-time virtual reality and digital content creation. Although 3D Gaussian Splatting (3DGS) enables efficient novel-view rendering with explicit representations, its performance on reflective surfaces still lags behind implicit neural methods, especially in recovering fine geometry and surface normals. To address this gap, we propose PolGS++, a physically-guided polarimetric Gaussian Splatting framework for fast reflective surface reconstruction. Specifically, we integrate a polarized BRDF (pBRDF) model into 3DGS to explicitly decouple diffuse and specular components, providing physically grounded reflectance modeling and stronger geometric cues for reflective surface recovery. Furthermore, we introduce a depth-guided visibility mask acquisition mechanism that enables angle-of-polarization (AoP)-based tangent-space consistency constraints in Gaussian Splatting without costly ray-tracing intersections. This physically guided design improves reconstruction quality and efficiency, requiring only about 10 minutes of training. Extensive experiments on both synthetic and real-world datasets validate the effectiveness of our method.

  </details>



- **WalkGPT: Grounded Vision-Language Conversation with Depth-Aware Segmentation for Pedestrian Navigation**  
  Rafi Ibn Sultan, Hui Zhu, Xiangyu Zhou, Chengyin Li, Prashant Khanduri, Marco Brocanelli, Dongxiao Zhu  
  _2026-03-11_ · https://arxiv.org/abs/2603.10703v1  
  <details><summary>Abstract</summary>

  Ensuring accessible pedestrian navigation requires reasoning about both semantic and spatial aspects of complex urban scenes, a challenge that existing Large Vision-Language Models (LVLMs) struggle to meet. Although these models can describe visual content, their lack of explicit grounding leads to object hallucinations and unreliable depth reasoning, limiting their usefulness for accessibility guidance. We introduce WalkGPT, a pixel-grounded LVLM for the new task of Grounded Navigation Guide, unifying language reasoning and segmentation within a single architecture for depth-aware accessibility guidance. Given a pedestrian-view image and a navigation query, WalkGPT generates a conversational response with segmentation masks that delineate accessible and harmful features, along with relative depth estimation. The model incorporates a Multi-Scale Query Projector (MSQP) that shapes the final image tokens by aggregating them along text tokens across spatial hierarchies, and a Calibrated Text Projector (CTP), guided by a proposed Region Alignment Loss, that maps language embeddings into segmentation-aware representations. These components enable fine-grained grounding and depth inference without user-provided cues or anchor points, allowing the model to generate complete and realistic navigation guidance. We also introduce PAVE, a large-scale benchmark of 41k pedestrian-view images paired with accessibility-aware questions and depth-grounded answers. Experiments show that WalkGPT achieves strong grounded reasoning and segmentation performance. The source code and dataset are available on the \href{https://sites.google.com/view/walkgpt-26/home}{project website}.

  </details>



- **TopGen: Learning Structural Layouts and Cross-Fields for Quadrilateral Mesh Generation**  
  Yuguang Chen, Xinhai Liu, Xiangyu Zhu, Yiling Zhu, Zhuo Chen, Dongyu Zhang, Chunchao Guo  
  _2026-03-11_ · https://arxiv.org/abs/2603.10606v1  
  <details><summary>Abstract</summary>

  High-quality quadrilateral mesh generation is a fundamental challenge in computer graphics. Traditional optimization-based methods are often constrained by the topological quality of input meshes and suffer from severe efficiency bottlenecks, frequently becoming computationally prohibitive when handling high-resolution models. While emerging learning-based approaches offer greater flexibility, they primarily focus on cross-field prediction, often resulting in the loss of critical structural layouts and a lack of editability. In this paper, we propose TopGen, a robust and efficient learning-based framework that mimics professional manual modeling workflows by simultaneously predicting structural layouts and cross-fields. By processing input triangular meshes through point cloud sampling and a shape encoder, TopGen is inherently robust to non-manifold geometries and low-quality initial topologies. We introduce a dual-query decoder using edge-based and face-based sampling points as queries to perform structural line classification and cross-field regression in parallel. This integrated approach explicitly extracts the geometric skeleton while concurrently capturing orientation fields. Such synergy ensures the preservation of geometric integrity and provides an intuitive, editable foundation for subsequent quadrilateral remeshing. To support this framework, we also introduce a large-scale quadrilateral mesh dataset, TopGen-220K, featuring high-quality paired data comprising raw triangular meshes, structural layouts, cross-fields, and their corresponding quad meshes. Experimental results demonstrate that TopGen significantly outperforms existing state-of-the-art methods in both geometric fidelity and topological edge flow rationality.

  </details>



- **LCAMV: High-Accuracy 3D Reconstruction of Color-Varying Objects Using LCA Correction and Minimum-Variance Fusion in Structured Light**  
  Wonbeen Oh, Jae-Sang Hyun  
  _2026-03-11_ · https://arxiv.org/abs/2603.10456v1  
  <details><summary>Abstract</summary>

  Accurate 3D reconstruction of colored objects with structured light (SL) is hindered by lateral chromatic aberration (LCA) in optical components and uneven noise characteristics across RGB channels. This paper introduces lateral chromatic aberration correction and minimum-variance fusion (LCAMV), a robust 3D reconstruction method that operates with a single projector-camera pair without additional hardware or acquisition constraints. LCAMV analytically models and pixel-wise compensates LCA in both the projector and camera, then adaptively fuses multi-channel phase data using a Poisson-Gaussian noise model and minimum-variance estimation. Unlike existing methods that require extra hardware or multiple exposures, LCAMV enables fast acquisition. Experiments on planar and non-planar colored surfaces show that LCAMV outperforms grayscale conversion and conventional channel-weighting, reducing depth error by up to 43.6\%. These results establish LCAMV as an effective solution for high-precision 3D reconstruction of nonuniformly colored objects.

  </details>



- **AsyncMDE: Real-Time Monocular Depth Estimation via Asynchronous Spatial Memory**  
  Lianjie Ma, Yuquan Li, Bingzheng Jiang, Ziming Zhong, Han Ding, Lijun Zhu  
  _2026-03-11_ · https://arxiv.org/abs/2603.10438v1  
  <details><summary>Abstract</summary>

  Foundation-model-based monocular depth estimation offers a viable alternative to active sensors for robot perception, yet its computational cost often prohibits deployment on edge platforms. Existing methods perform independent per-frame inference, wasting the substantial computational redundancy between adjacent viewpoints in continuous robot operation. This paper presents AsyncMDE, an asynchronous depth perception system consisting of a foundation model and a lightweight model that amortizes the foundation model's computational cost over time. The foundation model produces high-quality spatial features in the background, while the lightweight model runs asynchronously in the foreground, fusing cached memory with current observations through complementary fusion, outputting depth estimates, and autoregressively updating the memory. This enables cross-frame feature reuse with bounded accuracy degradation. At a mere 3.83M parameters, it operates at 237 FPS on an RTX 4090, recovering 77% of the accuracy gap to the foundation model while achieving a 25X parameter reduction. Validated across indoor static, dynamic, and synthetic extreme-motion benchmarks, AsyncMDE degrades gracefully between refreshes and achieves 161FPS on a Jetson AGX Orin with TensorRT, clearly demonstrating its feasibility for real-time edge deployment.

  </details>



- **On the Structural Failure of Chamfer Distance in 3D Shape Optimization**  
  Chang-Yong Song, David Hyde  
  _2026-03-10_ · https://arxiv.org/abs/2603.09925v1  
  <details><summary>Abstract</summary>

  Chamfer distance is the standard training loss for point cloud reconstruction, completion, and generation, yet directly optimizing it can produce worse Chamfer values than not optimizing it at all. We show that this paradoxical failure is gradient-structural. The per-point Chamfer gradient creates a many-to-one collapse that is the unique attractor of the forward term and cannot be resolved by any local regularizer, including repulsion, smoothness, and density-aware re-weighting. We derive a necessary condition for collapse suppression: coupling must propagate beyond local neighborhoods. In a controlled 2D setting, shared-basis deformation suppresses collapse by providing global coupling; in 3D shape morphing, a differentiable MPM prior instantiates the same principle, consistently reducing the Chamfer gap across 20 directed pairs with a 2.5$\times$ improvement on the topologically complex dragon. The presence or absence of non-local coupling determines whether Chamfer optimization succeeds or collapses. This provides a practical design criterion for any pipeline that optimizes point-level distance metrics.

  </details>



- **VLM-Loc: Localization in Point Cloud Maps via Vision-Language Models**  
  Shuhao Kang, Youqi Liao, Peijie Wang, Wenlong Liao, Qilin Zhang, Benjamin Busam, Xieyuanli Chen, Yun Liu  
  _2026-03-10_ · https://arxiv.org/abs/2603.09826v1  
  <details><summary>Abstract</summary>

  Text-to-point-cloud (T2P) localization aims to infer precise spatial positions within 3D point cloud maps from natural language descriptions, reflecting how humans perceive and communicate spatial layouts through language. However, existing methods largely rely on shallow text-point cloud correspondence without effective spatial reasoning, limiting their accuracy in complex environments. To address this limitation, we propose VLM-Loc, a framework that leverages the spatial reasoning capability of large vision-language models (VLMs) for T2P localization. Specifically, we transform point clouds into bird's-eye-view (BEV) images and scene graphs that jointly encode geometric and semantic context, providing structured inputs for the VLM to learn cross-modal representations bridging linguistic and spatial semantics. On top of these representations, we introduce a partial node assignment mechanism that explicitly associates textual cues with scene graph nodes, enabling interpretable spatial reasoning for accurate localization. To facilitate systematic evaluation across diverse scenes, we present CityLoc, a benchmark built from multi-source point clouds for fine-grained T2P localization. Experiments on CityLoc demonstrate VLM-Loc achieves superior accuracy and robustness compared to state-of-the-art methods. Our code, model, and dataset are available at \href{https://github.com/MCG-NKU/nku-3d-vision}{repository}.

  </details>



- **ConfCtrl: Enabling Precise Camera Control in Video Diffusion via Confidence-Aware Interpolation**  
  Liudi Yang, George Eskandar, Fengyi Shen, Mohammad Altillawi, Yang Bai, Chi Zhang, Ziyuan Liu, Abhinav Valada  
  _2026-03-10_ · https://arxiv.org/abs/2603.09819v1  
  <details><summary>Abstract</summary>

  We address the challenge of novel view synthesis from only two input images under large viewpoint changes. Existing regression-based methods lack the capacity to reconstruct unseen regions, while camera-guided diffusion models often deviate from intended trajectories due to noisy point cloud projections or insufficient conditioning from camera poses. To address these issues, we propose ConfCtrl, a confidence-aware video interpolation framework that enables diffusion models to follow prescribed camera poses while completing unseen regions. ConfCtrl initializes the diffusion process by combining a confidence-weighted projected point cloud latent with noise as the conditioning input. It then applies a Kalman-inspired predict-update mechanism, treating the projected point cloud as a noisy measurement and using learned residual corrections to balance pose-driven predictions with noisy geometric observations. This allows the model to rely on reliable projections while down-weighting uncertain regions, yielding stable, geometry-aware generation. Experiments on multiple datasets show that ConfCtrl produces geometrically consistent and visually plausible novel views, effectively reconstructing occluded regions under large viewpoint changes.

  </details>



- **DRIFT: Dual-Representation Inter-Fusion Transformer for Automated Driving Perception with 4D Radar Point Clouds**  
  Siqi Pei, Andras Palffy, Dariu M. Gavrila  
  _2026-03-10_ · https://arxiv.org/abs/2603.09695v1  
  <details><summary>Abstract</summary>

  4D radars, which provide 3D point cloud data along with Doppler velocity, are attractive components of modern automated driving systems due to their low cost and robustness under adverse weather conditions. However, they provide a significantly lower point cloud density than LiDAR sensors. This makes it important to exploit not only local but also global contextual scene information. This paper proposes DRIFT, a model that effectively captures and fuses both local and global contexts through a dual-path architecture. The model incorporates a point path to aggregate fine-grained local features and a pillar path to encode coarse-grained global features. These two parallel paths are intertwined via novel feature-sharing layers at multiple stages, enabling full utilization of both representations. DRIFT is evaluated on the widely used View-of-Delft (VoD) dataset and a proprietary internal dataset. It outperforms the baselines on the tasks of object detection and/or free road estimation. For example, DRIFT achieves a mean average precision (mAP) of 52.6\% (compared to, say, 45.4\% of CenterPoint) on the VoD dataset.

  </details>



- **SurgFed: Language-guided Multi-Task Federated Learning for Surgical Video Understanding**  
  Zheng Fang, Ziwei Niu, Ziyue Wang, Zhu Zhuo, Haofeng Liu, Shuyang Qian, Jun Xia, Yueming Jin  
  _2026-03-10_ · https://arxiv.org/abs/2603.09496v1  
  <details><summary>Abstract</summary>

  Surgical scene Multi-Task Federated Learning (MTFL) is essential for robot-assisted minimally invasive surgery (RAS) but remains underexplored in surgical video understanding due to two key challenges: (1) Tissue Diversity: Local models struggle to adapt to site-specific tissue features, limiting their effectiveness in heterogeneous clinical environments and leading to poor local predictions. (2) Task Diversity: Server-side aggregation, relying solely on gradient-based clustering, often produces suboptimal or incorrect parameter updates due to inter-site task heterogeneity, resulting in inaccurate localization. In light of these two issues, we propose SurgFed, a multi-task federated learning framework, enabling federated learning for surgical scene segmentation and depth estimation across diverse surgical types. SurgFed is powered by two appealing designs, i.e., Language-guided Channel Selection (LCS) and Language-guided Hyper Aggregation (LHA), to address the challenge of fully exploration on corss-site and cross-task. Technically, the LCS is first designed a lightweight personalized channel selection network that enhances site-specific adaptation using pre-defined text inputs, which optimally the local model learn the specific embeddings. We further introduce the LHA that employs a layer-wise cross-attention mechanism with pre-defined text inputs to model task interactions across sites and guide a hypernetwork for personalized parameter updates. Extensive empirical evidence shows that SurgFed yields improvements over the state-of-the-art methods in five public datasets across four surgical types. The code is available at https://anonymous.4open.science/r/SurgFed-070E/.

  </details>



- **EventVGGT: Exploring Cross-Modal Distillation for Consistent Event-based Depth Estimation**  
  Yinrui Ren, Jinjing Zhu, Kanghao Chen, Zhuoxiao Li, Jing Ou, Zidong Cao, Tongyan Hua, Peilun Shi, Yingchun Fu, Wufan Zhao, et al.  
  _2026-03-10_ · https://arxiv.org/abs/2603.09385v1  
  <details><summary>Abstract</summary>

  Event cameras offer superior sensitivity to high-speed motion and extreme lighting, making event-based monocular depth estimation a promising approach for robust 3D perception in challenging conditions. However, progress is severely hindered by the scarcity of dense depth annotations. While recent annotation-free approaches mitigate this by distilling knowledge from Vision Foundation Models (VFMs), a critical limitation persists: they process event streams as independent frames. By neglecting the inherent temporal continuity of event data, these methods fail to leverage the rich temporal priors encoded in VFMs, ultimately yielding temporally inconsistent and less accurate depth predictions. To address this, we introduce EventVGGT, a novel framework that explicitly models the event stream as a coherent video sequence. To the best of our knowledge, we are the first to distill spatio-temporal and multi-view geometric priors from the Visual Geometry Grounded Transformer (VGGT) into the event domain. We achieve this via a comprehensive tri-level distillation strategy: (i) Cross-Modal Feature Mixture (CMFM) bridges the modality gap at the output level by fusing RGB and event features to generate auxiliary depth predictions; (ii) Spatio-Temporal Feature Distillation (STFD) distills VGGT's powerful spatio-temporal representations at the feature level; and (iii) Temporal Consistency Distillation (TCD) enforces cross-frame coherence at the temporal level by aligning inter-frame depth changes. Extensive experiments demonstrate that EventVGGT consistently outperforms existing methods -- reducing the absolute mean depth error at 30m by over 53\% on EventScape (from 2.30 to 1.06) -- while exhibiting robust zero-shot generalization on the unseen DENSE and MVSEC datasets.

  </details>



- **SpaceSense-Bench: A Large-Scale Multi-Modal Benchmark for Spacecraft Perception and Pose Estimation**  
  Aodi Wu, Jianhong Zuo, Zeyuan Zhao, Xubo Luo, Ruisuo Wang, Xue Wan  
  _2026-03-10_ · https://arxiv.org/abs/2603.09320v1  
  <details><summary>Abstract</summary>

  Autonomous space operations such as on-orbit servicing and active debris removal demand robust part-level semantic understanding and precise relative navigation of target spacecraft, yet collecting large-scale real data in orbit remains impractical due to cost and access constraints. Existing synthetic datasets, moreover, suffer from limited target diversity, single-modality sensing, and incomplete ground-truth annotations. We present \textbf{SpaceSense-Bench}, a large-scale multi-modal benchmark for spacecraft perception encompassing 136~satellite models with approximately 70~GB of data. Each frame provides time-synchronized 1024$\times$1024 RGB images, millimeter-precision depth maps, and 256-beam LiDAR point clouds, together with dense 7-class part-level semantic labels at both the pixel and point level as well as accurate 6-DoF pose ground truth. The dataset is generated through a high-fidelity space simulation built in Unreal Engine~5 and a fully automated pipeline covering data acquisition, multi-stage quality control, and conversion to mainstream formats. We benchmark five representative tasks (object detection, 2D semantic segmentation, RGB--LiDAR fusion-based 3D point cloud segmentation, monocular depth estimation, and orientation estimation) and identify two key findings: (i)~perceiving small-scale components (\emph{e.g.}, thrusters and omni-antennas) and generalizing to entirely unseen spacecraft in a zero-shot setting remain critical bottlenecks for current methods, and (ii)~scaling up the number of training satellites yields substantial performance gains on novel targets, underscoring the value of large-scale, diverse datasets for space perception research. The dataset, code, and toolkit are publicly available at https://github.com/wuaodi/SpaceSense-Bench.

  </details>



- **NLiPsCalib: An Efficient Calibration Framework for High-Fidelity 3D Reconstruction of Curved Visuotactile Sensors**  
  Xuhao Qin, Feiyu Zhao, Yatao Leng, Runze Hu, Chenxi Xiao  
  _2026-03-10_ · https://arxiv.org/abs/2603.09319v1  
  <details><summary>Abstract</summary>

  Recent advances in visuotactile sensors increasingly employ biomimetic curved surfaces to enhance sensorimotor capabilities. Although such curved visuotactile sensors enable more conformal object contact, their perceptual quality is often degraded by non-uniform illumination, which reduces reconstruction accuracy and typically necessitates calibration. Existing calibration methods commonly rely on customized indenters and specialized devices to collect large-scale photometric data, but these processes are expensive and labor-intensive. To overcome these calibration challenges, we present NLiPsCalib, a physics-consistent and efficient calibration framework for curved visuotactile sensors. NLiPsCalib integrates controllable near-field light sources and leverages Near-Light Photometric Stereo (NLiPs) to estimate contact geometry, simplifying calibration to just a few simple contacts with everyday objects. We further introduce NLiPsTac, a controllable-light-source tactile sensor developed to validate our framework. Experimental results demonstrate that our approach enables high-fidelity 3D reconstruction across diverse curved form factors with a simple calibration procedure. We emphasize that our approach lowers the barrier to developing customized visuotactile sensors of diverse geometries, thereby making visuotactile sensing more accessible to the broader community.

  </details>



- **Implicit Geometry Representations for Vision-and-Language Navigation from Web Videos**  
  Mingfei Han, Haihong Hao, Liang Ma, Kamila Zhumakhanova, Ekaterina Radionova, Jingyi Zhang, Xiaojun Chang, Xiaodan Liang, Ivan Laptev  
  _2026-03-10_ · https://arxiv.org/abs/2603.09259v1  
  <details><summary>Abstract</summary>

  Vision-and-Language Navigation (VLN) has long been constrained by the limited diversity and scalability of simulator-curated datasets, which fail to capture the complexity of real-world environments. To overcome this limitation, we introduce a large-scale video-instruction framework derived from web-based room tour videos, enabling agents to learn from natural human walking demonstrations in diverse, realistic indoor settings. Unlike existing datasets, our framework integrates both open-ended description-enriched trajectories and action-enriched trajectories reconstructed in 3D, providing richer spatial and semantic supervision. A key extension in this work is the incorporation of implicit geometry representations, which extract spatial cues directly from RGB frames without requiring fragile 3D reconstruction. This approach substantially improves data utilization, alleviates reconstruction failures, and unlocks large portions of previously unusable video data. Comprehensive experiments across multiple VLN benchmarks (CVDN, SOON, R2R, and REVERIE) demonstrate that our method not only sets new state-of-the-art performance but also enables the development of robust zero-shot navigation agents. By bridging large-scale web videos with implicit spatial reasoning, this work advances embodied navigation towards more scalable, generalizable, and real-world applicable solutions.

  </details>



- **Point Cloud as a Foreign Language for Multi-modal Large Language Model**  
  Sneha Paul, Zachary Patterson, Nizar Bouguila  
  _2026-03-10_ · https://arxiv.org/abs/2603.09173v1  
  <details><summary>Abstract</summary>

  Multi-modal large language models (MLLMs) have shown remarkable progress in integrating visual and linguistic understanding. Recent efforts have extended these capabilities to 3D understanding through encoder-based architectures that rely on pre-trained 3D encoders to extract geometric features. However, such approaches suffer from semantic misalignment between geometric and linguistic spaces, resolution sensitivity, and substantial computational overhead. In this work, we present SAGE, the first end-to-end 3D MLLM that directly processes raw point clouds without relying on a pre-trained 3D encoder. Our approach introduces a lightweight 3D tokenizer that combines geometric sampling and neighbourhood aggregation with vector quantization to convert point clouds into discrete tokens--treating 3D data as a foreign language that naturally extends the LLM's vocabulary. Furthermore, to enhance the model's reasoning capability on complex 3D tasks, we propose a preference optimization training strategy with a semantic alignment-based reward, specifically designed for open-ended 3D question answering where responses are descriptive. Extensive experiments across diverse 3D understanding benchmarks demonstrate that our end-to-end approach outperforms existing encoder-based methods while offering significant advantages in computational efficiency, generalization across LLM backbones, and robustness to input resolution variations. Code is available at: github.com/snehaputul/SAGE3D.

  </details>



- **mmGAT: Pose Estimation by Graph Attention with Mutual Features from mmWave Radar Point Cloud**  
  Abdullah Al Masud, Shi Xintong, Mondher Bouazizi, Ohtsuki Tomoaki  
  _2026-03-09_ · https://arxiv.org/abs/2603.08551v1  
  <details><summary>Abstract</summary>

  Pose estimation and human action recognition (HAR) are pivotal technologies spanning various domains. While the image-based pose estimation and HAR are widely admired for their superior performance, they lack in privacy protection and suboptimal performance in low-light and dark environments. This paper exploits the capabilities of millimeter-wave (mmWave) radar technology for human pose estimation by processing radar data with Graph Neural Network (GNN) architecture, coupled with the attention mechanism. Our goal is to capture the finer details of the radar point cloud to improve the pose estimation performance. To this end, we present a unique feature extraction technique that exploits the full potential of the GNN processing method for pose estimation. Our model mmGAT demonstrates remarkable performance on two publicly available benchmark mmWave datasets and establishes new state of the art results in most scenarios in terms of human pose estimation. Our approach achieves a noteworthy reduction of pose estimation mean per joint position error (MPJPE) by 35.6% and PA-MPJPE by 14.1% from the current state of the art benchmark within this domain.

  </details>



- **PCFEx: Point Cloud Feature Extraction for Graph Neural Networks**  
  Abdullah Al Masud, Shi Xintong, Mondher Bouazizi, Ohtsuki Tomoaki  
  _2026-03-09_ · https://arxiv.org/abs/2603.08540v1  
  <details><summary>Abstract</summary>

  Graph neural networks (GNNs) have gained significant attention for their effectiveness across various domains. This study focuses on applying GNN to process 3D point cloud data for human pose estimation (HPE) and human activity recognition (HAR). We propose novel point cloud feature extraction (PCFEx) techniques to capture meaningful information at the point, edge, and graph levels of the point cloud by considering point cloud as a graph. Moreover, we introduce a GNN architecture designed to efficiently process these features. Our approach is evaluated on four most popular publicly available millimeter wave radar datasets, three for HPE and one for HAR. The results show substantial improvements, with significantly reduced errors in all three HPE benchmarks, and an overall accuracy of 98.8% in mmWave-based HAR, outperforming the existing state of the art models. This work demonstrates the great potential of feature extraction incorporated with GNN modeling approach to enhance the precision of point cloud processing.

  </details>


