# NeRF & Neural Radiance Fields

_Updated: 2026-01-23 06:52 UTC_

Total papers shown: **10**


---

- **EVolSplat4D: Efficient Volume-based Gaussian Splatting for 4D Urban Scene Synthesis**  
  Sheng Miao, Sijin Li, Pan Wang, Dongfeng Bai, Bingbing Liu, Yue Wang, Andreas Geiger, Yiyi Liao  
  _2026-01-22_ · https://arxiv.org/abs/2601.15951v1  
  <details><summary>Abstract</summary>

  Novel view synthesis (NVS) of static and dynamic urban scenes is essential for autonomous driving simulation, yet existing methods often struggle to balance reconstruction time with quality. While state-of-the-art neural radiance fields and 3D Gaussian Splatting approaches achieve photorealism, they often rely on time-consuming per-scene optimization. Conversely, emerging feed-forward methods frequently adopt per-pixel Gaussian representations, which lead to 3D inconsistencies when aggregating multi-view predictions in complex, dynamic environments. We propose EvolSplat4D, a feed-forward framework that moves beyond existing per-pixel paradigms by unifying volume-based and pixel-based Gaussian prediction across three specialized branches. For close-range static regions, we predict consistent geometry of 3D Gaussians over multiple frames directly from a 3D feature volume, complemented by a semantically-enhanced image-based rendering module for predicting their appearance. For dynamic actors, we utilize object-centric canonical spaces and a motion-adjusted rendering module to aggregate temporal features, ensuring stable 4D reconstruction despite noisy motion priors. Far-Field scenery is handled by an efficient per-pixel Gaussian branch to ensure full-scene coverage. Experimental results on the KITTI-360, KITTI, Waymo, and PandaSet datasets show that EvolSplat4D reconstructs both static and dynamic environments with superior accuracy and consistency, outperforming both per-scene optimization and state-of-the-art feed-forward baselines.

  </details>



- **Seeing through Light and Darkness: Sensor-Physics Grounded Deblurring HDR NeRF from Single-Exposure Images and Events**  
  Yunshan Qi, Lin Zhu, Nan Bao, Yifan Zhao, Jia Li  
  _2026-01-21_ · https://arxiv.org/abs/2601.15475v1  
  <details><summary>Abstract</summary>

  Novel view synthesis from low dynamic range (LDR) blurry images, which are common in the wild, struggles to recover high dynamic range (HDR) and sharp 3D representations in extreme lighting conditions. Although existing methods employ event data to address this issue, they ignore the sensor-physics mismatches between the camera output and physical world radiance, resulting in suboptimal HDR and deblurring results. To cope with this problem, we propose a unified sensor-physics grounded NeRF framework for sharp HDR novel view synthesis from single-exposure blurry LDR images and corresponding events. We employ NeRF to directly represent the actual radiance of the 3D scene in the HDR domain and model raw HDR scene rays hitting the sensor pixels as in the physical world. A pixel-wise RGB mapping field is introduced to align the above rendered pixel values with the sensor-recorded LDR pixel values of the input images. A novel event mapping field is also designed to bridge the physical scene dynamics and actual event sensor output. The two mapping fields are jointly optimized with the NeRF network, leveraging the spatial and temporal dynamic information in events to enhance the sharp HDR 3D representation learning. Experiments on the collected and public datasets demonstrate that our method can achieve state-of-the-art deblurring HDR novel view synthesis results with single-exposure blurry LDR images and corresponding events.

  </details>



- **SplatBus: A Gaussian Splatting Viewer Framework via GPU Interprocess Communication**  
  Yinghan Xu, Théo Morales, John Dingliana  
  _2026-01-21_ · https://arxiv.org/abs/2601.15431v1  
  <details><summary>Abstract</summary>

  Radiance field-based rendering methods have attracted significant interest from the computer vision and computer graphics communities. They enable high-fidelity rendering with complex real-world lighting effects, but at the cost of high rendering time. 3D Gaussian Splatting solves this issue with a rasterisation-based approach for real-time rendering, enabling applications such as autonomous driving, robotics, virtual reality, and extended reality. However, current 3DGS implementations are difficult to integrate into traditional mesh-based rendering pipelines, which is a common use case for interactive applications and artistic exploration. To address this limitation, this software solution uses Nvidia's interprocess communication (IPC) APIs to easily integrate into implementations and allow the results to be viewed in external clients such as Unity, Blender, Unreal Engine, and OpenGL viewers. The code is available at https://github.com/RockyXu66/splatbus.

  </details>



- **RayRoPE: Projective Ray Positional Encoding for Multi-view Attention**  
  Yu Wu, Minsik Jeon, Jen-Hao Rick Chang, Oncel Tuzel, Shubham Tulsiani  
  _2026-01-21_ · https://arxiv.org/abs/2601.15275v1  
  <details><summary>Abstract</summary>

  We study positional encodings for multi-view transformers that process tokens from a set of posed input images, and seek a mechanism that encodes patches uniquely, allows SE(3)-invariant attention with multi-frequency similarity, and can be adaptive to the geometry of the underlying scene. We find that prior (absolute or relative) encoding schemes for multi-view attention do not meet the above desiderata, and present RayRoPE to address this gap. RayRoPE represents patch positions based on associated rays but leverages a predicted point along the ray instead of the direction for a geometry-aware encoding. To achieve SE(3) invariance, RayRoPE computes query-frame projective coordinates for computing multi-frequency similarity. Lastly, as the 'predicted' 3D point along a ray may not be precise, RayRoPE presents a mechanism to analytically compute the expected position encoding under uncertainty. We validate RayRoPE on the tasks of novel-view synthesis and stereo depth estimation and show that it consistently improves over alternate position encoding schemes (e.g. 15% relative improvement on LPIPS in CO3D). We also show that RayRoPE can seamlessly incorporate RGB-D input, resulting in even larger gains over alternatives that cannot positionally encode this information.

  </details>



- **Three-dimensional visualization of X-ray micro-CT with large-scale datasets: Efficiency and accuracy for real-time interaction**  
  Yipeng Yin, Rao Yao, Qingying Li, Dazhong Wang, Hong Zhou, Zhijun Fang, Jianing Chen, Longjie Qian, Mingyue Wu  
  _2026-01-21_ · https://arxiv.org/abs/2601.15098v1  
  <details><summary>Abstract</summary>

  As Micro-CT technology continues to refine its characterization of material microstructures, industrial CT ultra-precision inspection is generating increasingly large datasets, necessitating solutions to the trade-off between accuracy and efficiency in the 3D characterization of defects during ultra-precise detection. This article provides a unique perspective on recent advances in accurate and efficient 3D visualization using Micro-CT, tracing its evolution from medical imaging to industrial non-destructive testing (NDT). Among the numerous CT reconstruction and volume rendering methods, this article selectively reviews and analyzes approaches that balance accuracy and efficiency, offering a comprehensive analysis to help researchers quickly grasp highly efficient and accurate 3D reconstruction methods for microscopic features. By comparing the principles of computed tomography with advancements in microstructural technology, this article examines the evolution of CT reconstruction algorithms from analytical methods to deep learning techniques, as well as improvements in volume rendering algorithms, acceleration, and data reduction. Additionally, it explores advanced lighting models for high-accuracy, photorealistic, and efficient volume rendering. Furthermore, this article envisions potential directions in CT reconstruction and volume rendering. It aims to guide future research in quickly selecting efficient and precise methods and developing new ideas and approaches for real-time online monitoring of internal material defects through virtual-physical interaction, for applying digital twin model to structural health monitoring (SHM).

  </details>



- **GAT-NeRF: Geometry-Aware-Transformer Enhanced Neural Radiance Fields for High-Fidelity 4D Facial Avatars**  
  Zhe Chang, Haodong Jin, Ying Sun, Yan Song, Hui Yu  
  _2026-01-21_ · https://arxiv.org/abs/2601.14875v1  
  <details><summary>Abstract</summary>

  High-fidelity 4D dynamic facial avatar reconstruction from monocular video is a critical yet challenging task, driven by increasing demands for immersive virtual human applications. While Neural Radiance Fields (NeRF) have advanced scene representation, their capacity to capture high-frequency facial details, such as dynamic wrinkles and subtle textures from information-constrained monocular streams, requires significant enhancement. To tackle this challenge, we propose a novel hybrid neural radiance field framework, called Geometry-Aware-Transformer Enhanced NeRF (GAT-NeRF) for high-fidelity and controllable 4D facial avatar reconstruction, which integrates the Transformer mechanism into the NeRF pipeline. GAT-NeRF synergistically combines a coordinate-aligned Multilayer Perceptron (MLP) with a lightweight Transformer module, termed as Geometry-Aware-Transformer (GAT) due to its processing of multi-modal inputs containing explicit geometric priors. The GAT module is enabled by fusing multi-modal input features, including 3D spatial coordinates, 3D Morphable Model (3DMM) expression parameters, and learnable latent codes to effectively learn and enhance feature representations pertinent to fine-grained geometry. The Transformer's effective feature learning capabilities are leveraged to significantly augment the modeling of complex local facial patterns like dynamic wrinkles and acne scars. Comprehensive experiments unequivocally demonstrate GAT-NeRF's state-of-the-art performance in visual fidelity and high-frequency detail recovery, forging new pathways for creating realistic dynamic digital humans for multimedia applications.

  </details>



- **POTR: Post-Training 3DGS Compression**  
  Bert Ramlot, Martijn Courteaux, Peter Lambert, Glenn Van Wallendael  
  _2026-01-21_ · https://arxiv.org/abs/2601.14821v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has recently emerged as a promising contender to Neural Radiance Fields (NeRF) in 3D scene reconstruction and real-time novel view synthesis. 3DGS outperforms NeRF in training and inference speed but has substantially higher storage requirements. To remedy this downside, we propose POTR, a post-training 3DGS codec built on two novel techniques. First, POTR introduces a novel pruning approach that uses a modified 3DGS rasterizer to efficiently calculate every splat's individual removal effect simultaneously. This technique results in 2-4x fewer splats than other post-training pruning techniques and as a result also significantly accelerates inference with experiments demonstrating 1.5-2x faster inference than other compressed models. Second, we propose a novel method to recompute lighting coefficients, significantly reducing their entropy without using any form of training. Our fast and highly parallel approach especially increases AC lighting coefficient sparsity, with experiments demonstrating increases from 70% to 97%, with minimal loss in quality. Finally, we extend POTR with a simple fine-tuning scheme to further enhance pruning, inference, and rate-distortion performance. Experiments demonstrate that POTR, even without fine-tuning, consistently outperforms all other post-training compression techniques in both rate-distortion performance and inference speed.

  </details>



- **High-Fidelity 3D Tooth Reconstruction by Fusing Intraoral Scans and CBCT Data via a Deep Implicit Representation**  
  Yi Zhu, Razmig Kechichian, Raphaël Richert, Satoshi Ikehata, Sébastien Valette  
  _2026-01-21_ · https://arxiv.org/abs/2601.15358v1  
  <details><summary>Abstract</summary>

  High-fidelity 3D tooth models are essential for digital dentistry, but must capture both the detailed crown and the complete root. Clinical imaging modalities are limited: Cone-Beam Computed Tomography (CBCT) captures the root but has a noisy, low-resolution crown, while Intraoral Scanners (IOS) provide a high-fidelity crown but no root information. A naive fusion of these sources results in unnatural seams and artifacts. We propose a novel, fully-automated pipeline that fuses CBCT and IOS data using a deep implicit representation. Our method first segments and robustly registers the tooth instances, then creates a hybrid proxy mesh combining the IOS crown and the CBCT root. The core of our approach is to use this noisy proxy to guide a class-specific DeepSDF network. This optimization process projects the input onto a learned manifold of ideal tooth shapes, generating a seamless, watertight, and anatomically coherent model. Qualitative and quantitative evaluations show our method uniquely preserves both the high-fidelity crown from IOS and the patient-specific root morphology from CBCT, overcoming the limitations of each modality and naive stitching.

  </details>



- **One-Shot Refiner: Boosting Feed-forward Novel View Synthesis via One-Step Diffusion**  
  Yitong Dong, Qi Zhang, Minchao Jiang, Zhiqiang Wu, Qingnan Fan, Ying Feng, Huaqi Zhang, Hujun Bao, Guofeng Zhang  
  _2026-01-20_ · https://arxiv.org/abs/2601.14161v1  
  <details><summary>Abstract</summary>

  We present a novel framework for high-fidelity novel view synthesis (NVS) from sparse images, addressing key limitations in recent feed-forward 3D Gaussian Splatting (3DGS) methods built on Vision Transformer (ViT) backbones. While ViT-based pipelines offer strong geometric priors, they are often constrained by low-resolution inputs due to computational costs. Moreover, existing generative enhancement methods tend to be 3D-agnostic, resulting in inconsistent structures across views, especially in unseen regions. To overcome these challenges, we design a Dual-Domain Detail Perception Module, which enables handling high-resolution images without being limited by the ViT backbone, and endows Gaussians with additional features to store high-frequency details. We develop a feature-guided diffusion network, which can preserve high-frequency details during the restoration process. We introduce a unified training strategy that enables joint optimization of the ViT-based geometric backbone and the diffusion-based refinement module. Experiments demonstrate that our method can maintain superior generation quality across multiple datasets.

  </details>



- **VENI: Variational Encoder for Natural Illumination**  
  Paul Walker, James A. D. Gardner, Andreea Ardelean, William A. P. Smith, Bernhard Egger  
  _2026-01-20_ · https://arxiv.org/abs/2601.14079v1  
  <details><summary>Abstract</summary>

  Inverse rendering is an ill-posed problem, but priors like illumination priors, can simplify it. Existing work either disregards the spherical and rotation-equivariant nature of illumination environments or does not provide a well-behaved latent space. We propose a rotation-equivariant variational autoencoder that models natural illumination on the sphere without relying on 2D projections. To preserve the SO(2)-equivariance of environment maps, we use a novel Vector Neuron Vision Transformer (VN-ViT) as encoder and a rotation-equivariant conditional neural field as decoder. In the encoder, we reduce the equivariance from SO(3) to SO(2) using a novel SO(2)-equivariant fully connected layer, an extension of Vector Neurons. We show that our SO(2)-equivariant fully connected layer outperforms standard Vector Neurons when used in our SO(2)-equivariant model. Compared to previous methods, our variational autoencoder enables smoother interpolation in latent space and offers a more well-behaved latent space.

  </details>


