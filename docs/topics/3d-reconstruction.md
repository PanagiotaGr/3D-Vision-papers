# 3D Reconstruction

_Updated: 2026-03-02 07:14 UTC_

Total papers shown: **9**


---

- **FocusTrack: One-Stage Focus-and-Suppress Framework for 3D Point Cloud Object Tracking**  
  Sifan Zhou, Jiahao Nie, Ziyu Zhao, Yichao Cao, Xiaobo Lu  
  _2026-02-27_ · https://arxiv.org/abs/2602.24133v1  
  <details><summary>Abstract</summary>

  In 3D point cloud object tracking, the motion-centric methods have emerged as a promising avenue due to its superior performance in modeling inter-frame motion. However, existing two-stage motion-based approaches suffer from fundamental limitations: (1) error accumulation due to decoupled optimization caused by explicit foreground segmentation prior to motion estimation, and (2) computational bottlenecks from sequential processing. To address these challenges, we propose FocusTrack, a novel one-stage paradigms tracking framework that unifies motion-semantics co-modeling through two core innovations: Inter-frame Motion Modeling (IMM) and Focus-and-Suppress Attention. The IMM module employs a temp-oral-difference siamese encoder to capture global motion patterns between adjacent frames. The Focus-and-Suppress attention that enhance the foreground semantics via motion-salient feature gating and suppress the background noise based on the temporal-aware motion context from IMM without explicit segmentation. Based on above two designs, FocusTrack enables end-to-end training with compact one-stage pipeline. Extensive experiments on prominent 3D tracking benchmarks, such as KITTI, nuScenes, and Waymo, demonstrate that the FocusTrack achieves new SOTA performance while running at a high speed with 105 FPS.

  </details>



- **DiffusionHarmonizer: Bridging Neural Reconstruction and Photorealistic Simulation with Online Diffusion Enhancer**  
  Yuxuan Zhang, Katarína Tóthová, Zian Wang, Kangxue Yin, Haithem Turki, Riccardo de Lutio, Yen-Yu Chang, Or Litany, Sanja Fidler, Zan Gojcic  
  _2026-02-27_ · https://arxiv.org/abs/2602.24096v1  
  <details><summary>Abstract</summary>

  Simulation is essential to the development and evaluation of autonomous robots such as self-driving vehicles. Neural reconstruction is emerging as a promising solution as it enables simulating a wide variety of scenarios from real-world data alone in an automated and scalable way. However, while methods such as NeRF and 3D Gaussian Splatting can produce visually compelling results, they often exhibit artifacts particularly when rendering novel views, and fail to realistically integrate inserted dynamic objects, especially when they were captured from different scenes. To overcome these limitations, we introduce DiffusionHarmonizer, an online generative enhancement framework that transforms renderings from such imperfect scenes into temporally consistent outputs while improving their realism. At its core is a single-step temporally-conditioned enhancer that is converted from a pretrained multi-step image diffusion model, capable of running in online simulators on a single GPU. The key to training it effectively is a custom data curation pipeline that constructs synthetic-real pairs emphasizing appearance harmonization, artifact correction, and lighting realism. The result is a scalable system that significantly elevates simulation fidelity in both research and production environments.

  </details>



- **EvalMVX: A Unified Benchmarking for Neural 3D Reconstruction under Diverse Multiview Setups**  
  Zaiyan Yang, Jieji Ren, Xiangyi Wang, zonglin li, Xu Cao, Heng Guo, Zhanyu Ma, Boxin Shi  
  _2026-02-27_ · https://arxiv.org/abs/2602.24065v1  
  <details><summary>Abstract</summary>

  Recent advancements in neural surface reconstruction have significantly enhanced 3D reconstruction. However, current real world datasets mainly focus on benchmarking multiview stereo (MVS) based on RGB inputs. Multiview photometric stereo (MVPS) and multiview shape from polarization (MVSfP), though indispensable on high-fidelity surface reconstruction and sparse inputs, have not been quantitatively assessed together with MVS. To determine the working range of different MVX (MVS, MVSfP, and MVPS) techniques, we propose EvalMVX, a real-world dataset containing $25$ objects, each captured with a polarized camera under $20$ varying views and $17$ light conditions including OLAT and natural illumination, leading to $8,500$ images. Each object includes aligned ground-truth 3D mesh, facilitating quantitative benchmarking of MVX methods simultaneously. Based on our EvalMVX, we evaluate $13$ MVX methods published in recent years, record the best-performing methods, and identify open problems under diverse geometric details and reflectance types. We hope EvalMVX and the benchmarking results can inspire future research on multiview 3D reconstruction.

  </details>



- **SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting**  
  Xiang Feng, Xiangbo Wang, Tieshi Zhong, Chengkai Wang, Yiting Zhao, Tianxiang Xu, Zhenzhong Kuang, Feiwei Qin, Xuefei Yin, Yanming Zhu  
  _2026-02-27_ · https://arxiv.org/abs/2602.24020v1  
  <details><summary>Abstract</summary>

  3D super-resolution (3DSR) aims to reconstruct high-resolution (HR) 3D scenes from low-resolution (LR) multi-view images. Existing methods rely on dense LR inputs and per-scene optimization, which restricts the high-frequency priors for constructing HR 3D Gaussian Splatting (3DGS) to those inherited from pretrained 2D super-resolution (2DSR) models. This severely limits reconstruction fidelity, cross-scene generalization, and real-time usability. We propose to reformulate 3DSR as a direct feed-forward mapping from sparse LR views to HR 3DGS representations, enabling the model to autonomously learn 3D-specific high-frequency geometry and appearance from large-scale, multi-scene data. This fundamentally changes how 3DSR acquires high-frequency knowledge and enables robust generalization to unseen scenes. Specifically, we introduce SR3R, a feed-forward framework that directly predicts HR 3DGS representations from sparse LR views via the learned mapping network. To further enhance reconstruction fidelity, we introduce Gaussian offset learning and feature refinement, which stabilize reconstruction and sharpen high-frequency details. SR3R is plug-and-play and can be paired with any feed-forward 3DGS reconstruction backbone: the backbone provides an LR 3DGS scaffold, and SR3R upscales it to an HR 3DGS. Extensive experiments across three 3D benchmarks demonstrate that SR3R surpasses state-of-the-art (SOTA) 3DSR methods and achieves strong zero-shot generalization, even outperforming SOTA per-scene optimization methods on unseen scenes.

  </details>



- **AHAP: Reconstructing Arbitrary Humans from Arbitrary Perspectives with Geometric Priors**  
  Xiaozhen Qiao, Wenjia Wang, Zhiyuan Zhao, Jiacheng Sun, Ping Luo, Hongyuan Zhang, Xuelong Li  
  _2026-02-27_ · https://arxiv.org/abs/2602.23951v1  
  <details><summary>Abstract</summary>

  Reconstructing 3D humans from images captured at multiple perspectives typically requires pre-calibration, like using checkerboards or MVS algorithms, which limits scalability and applicability in diverse real-world scenarios. In this work, we present \textbf{AHAP} (Reconstructing \textbf{A}rbitrary \textbf{H}umans from \textbf{A}rbitrary \textbf{P}erspectives), a feed-forward framework for reconstructing arbitrary humans from arbitrary camera perspectives without requiring camera calibration. Our core lies in the effective fusion of multi-view geometry to assist human association, reconstruction and localization. Specifically, we use a Cross-View Identity Association module through learnable person queries and soft assignment, supervised by contrastive learning to resolve cross-view human identity association. A Human Head fuses cross-view features and scene context for SMPL prediction, guided by cross-view reprojection losses to enforce body pose consistency. Additionally, multi-view geometry eliminates the depth ambiguity inherent in monocular methods, providing more precise 3D human localization through multi-view triangulation. Experiments on EgoHumans and EgoExo4D demonstrate that AHAP achieves competitive performance on both world-space human reconstruction and camera pose estimation, while being 180$\times$ faster than optimization-based approaches.

  </details>



- **PointCoT: A Multi-modal Benchmark for Explicit 3D Geometric Reasoning**  
  Dongxu Zhang, Yiding Sun, Pengcheng Li, Yumou Liu, Hongqiang Lin, Haoran Xu, Xiaoxuan Mu, Liang Lin, Wenbiao Yan, Ning Yang, et al.  
  _2026-02-27_ · https://arxiv.org/abs/2602.23945v1  
  <details><summary>Abstract</summary>

  While Multimodal Large Language Models (MLLMs) demonstrate proficiency in 2D scenes, extending their perceptual intelligence to 3D point cloud understanding remains a significant challenge. Current approaches focus primarily on aligning 3D features with pre-trained models. However, they typically treat geometric reasoning as an implicit mapping process. These methods bypass intermediate logical steps and consequently suffer from geometric hallucinations. They confidently generate plausible responses that fail to ground in precise structural details. To bridge this gap, we present PointCoT, a novel framework that empowers MLLMs with explicit Chain-of-Thought (CoT) reasoning for 3D data. We advocate for a \textit{Look, Think, then Answer} paradigm. In this approach, the model is supervised to generate geometry-grounded rationales before predicting final answers. To facilitate this, we construct Point-Reason-Instruct, a large-scale benchmark comprising $\sim$86k instruction-tuning samples with hierarchical CoT annotations. By leveraging a dual-stream multi-modal architecture, our method synergizes semantic appearance with geometric truth. Extensive experiments demonstrate that PointCoT achieves state-of-the-art performance on complex reasoning tasks.

  </details>



- **Leveraging Geometric Prior Uncertainty and Complementary Constraints for High-Fidelity Neural Indoor Surface Reconstruction**  
  Qiyu Feng, Jiwei Shan, Shing Shin Cheng, Hesheng Wang  
  _2026-02-27_ · https://arxiv.org/abs/2602.23926v1  
  <details><summary>Abstract</summary>

  Neural implicit surface reconstruction with signed distance function has made significant progress, but recovering fine details such as thin structures and complex geometries remains challenging due to unreliable or noisy geometric priors. Existing approaches rely on implicit uncertainty that arises during optimization to filter these priors, which is indirect and inefficient, and masking supervision in high-uncertainty regions further leads to under-constrained optimization. To address these issues, we propose GPU-SDF, a neural implicit framework for indoor surface reconstruction that leverages geometric prior uncertainty and complementary constraints. We introduce a self-supervised module that explicitly estimates prior uncertainty without auxiliary networks. Based on this estimation, we design an uncertainty-guided loss that modulates prior influence rather than discarding it, thereby retaining weak but informative cues. To address regions with high prior uncertainty, GPU-SDF further incorporates two complementary constraints: an edge distance field that strengthens boundary supervision and a multi-view consistency regularization that enforces geometric coherence. Extensive experiments confirm that GPU-SDF improves the reconstruction of fine details and serves as a plug-and-play enhancement for existing frameworks. Source code will be available at https://github.com/IRMVLab/GPU-SDF

  </details>



- **Altitude-Aware Visual Place Recognition in Top-Down View**  
  Xingyu Shao, Mengfan He, Chunyu Li, Liangzheng Sun, Ziyang Meng  
  _2026-02-27_ · https://arxiv.org/abs/2602.23872v1  
  <details><summary>Abstract</summary>

  To address the challenge of aerial visual place recognition (VPR) problem under significant altitude variations, this study proposes an altitude-adaptive VPR approach that integrates ground feature density analysis with image classification techniques. The proposed method estimates airborne platforms' relative altitude by analyzing the density of ground features in images, then applies relative altitude-based cropping to generate canonical query images, which are subsequently used in a classification-based VPR strategy for localization. Extensive experiments across diverse terrains and altitude conditions demonstrate that the proposed approach achieves high accuracy and robustness in both altitude estimation and VPR under significant altitude changes. Compared to conventional methods relying on barometric altimeters or Time-of-Flight (ToF) sensors, this solution requires no additional hardware and offers a plug-and-play solution for downstream applications, {making it suitable for small- and medium-sized airborne platforms operating in diverse environments, including rural and urban areas.} Under significant altitude variations, incorporating our relative altitude estimation module into the VPR retrieval pipeline boosts average R@1 and R@5 by 29.85\% and 60.20\%, respectively, compared with applying VPR retrieval alone. Furthermore, compared to traditional {Monocular Metric Depth Estimation (MMDE) methods}, the proposed method reduces the mean error by 202.1 m, yielding average additional improvements of 31.4\% in R@1 and 44\% in R@5. These results demonstrate that our method establishes a robust, vision-only framework for three-dimensional visual place recognition, offering a practical and scalable solution for accurate airborne platforms localization under large altitude variations and limited sensor availability.

  </details>



- **Action-Geometry Prediction with 3D Geometric Prior for Bimanual Manipulation**  
  Chongyang Xu, Haipeng Li, Shen Cheng, Jingyu Hu, Haoqiang Fan, Ziliang Feng, Shuaicheng Liu  
  _2026-02-27_ · https://arxiv.org/abs/2602.23814v1  
  <details><summary>Abstract</summary>

  Bimanual manipulation requires policies that can reason about 3D geometry, anticipate how it evolves under action, and generate smooth, coordinated motions. However, existing methods typically rely on 2D features with limited spatial awareness, or require explicit point clouds that are difficult to obtain reliably in real-world settings. At the same time, recent 3D geometric foundation models show that accurate and diverse 3D structure can be reconstructed directly from RGB images in a fast and robust manner. We leverage this opportunity and propose a framework that builds bimanual manipulation directly on a pre-trained 3D geometric foundation model. Our policy fuses geometry-aware latents, 2D semantic features, and proprioception into a unified state representation, and uses diffusion model to jointly predict a future action chunk and a future 3D latent that decodes into a dense pointmap. By explicitly predicting how the 3D scene will evolve together with the action sequence, the policy gains strong spatial understanding and predictive capability using only RGB observations. We evaluate our method both in simulation on the RoboTwin benchmark and in real-world robot executions. Our approach consistently outperforms 2D-based and point-cloud-based baselines, achieving state-of-the-art performance in manipulation success, inter-arm coordination, and 3D spatial prediction accuracy. Code is available at https://github.com/Chongyang-99/GAP.git.

  </details>


