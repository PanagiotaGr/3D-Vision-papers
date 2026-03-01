# 3D Reconstruction

_Updated: 2026-03-01 07:02 UTC_

Total papers shown: **6**


---

- **VGG-T$^3$: Offline Feed-Forward 3D Reconstruction at Scale**  
  Sven Elflein, Ruilong Li, Sérgio Agostinho, Zan Gojcic, Laura Leal-Taixé, Qunjie Zhou, Aljosa Osep  
  _2026-02-26_ · https://arxiv.org/abs/2602.23361v1  
  <details><summary>Abstract</summary>

  We present a scalable 3D reconstruction model that addresses a critical limitation in offline feed-forward methods: their computational and memory requirements grow quadratically w.r.t. the number of input images. Our approach is built on the key insight that this bottleneck stems from the varying-length Key-Value (KV) space representation of scene geometry, which we distill into a fixed-size Multi-Layer Perceptron (MLP) via test-time training. VGG-T$^3$ (Visual Geometry Grounded Test Time Training) scales linearly w.r.t. the number of input views, similar to online models, and reconstructs a $1k$ image collection in just $54$ seconds, achieving a $11.6\times$ speed-up over baselines that rely on softmax attention. Since our method retains global scene aggregation capability, our point map reconstruction error outperforming other linear-time methods by large margins. Finally, we demonstrate visual localization capabilities of our model by querying the scene representation with unseen images.

  </details>



- **UniScale: Unified Scale-Aware 3D Reconstruction for Multi-View Understanding via Prior Injection for Robotic Perception**  
  Mohammad Mahdavian, Gordon Tan, Binbin Xu, Yuan Ren, Dongfeng Bai, Bingbing Liu  
  _2026-02-26_ · https://arxiv.org/abs/2602.23224v1  
  <details><summary>Abstract</summary>

  We present UniScale, a unified, scale-aware multi-view 3D reconstruction framework for robotic applications that flexibly integrates geometric priors through a modular, semantically informed design. In vision-based robotic navigation, the accurate extraction of environmental structure from raw image sequences is critical for downstream tasks. UniScale addresses this challenge with a single feed-forward network that jointly estimates camera intrinsics and extrinsics, scale-invariant depth and point maps, and the metric scale of a scene from multi-view images, while optionally incorporating auxiliary geometric priors when available. By combining global contextual reasoning with camera-aware feature representations, UniScale is able to recover the metric-scale of the scene. In robotic settings where camera intrinsics are known, they can be easily incorporated to improve performance, with additional gains obtained when camera poses are also available. This co-design enables robust, metric-aware 3D reconstruction within a single unified model. Importantly, UniScale does not require training from scratch, and leverages world priors exhibited in pre-existing models without geometric encoding strategies, making it particularly suitable for resource-constrained robotic teams. We evaluate UniScale on multiple benchmarks, demonstrating strong generalization and consistent performance across diverse environments. We will release our implementation upon acceptance.

  </details>



- **Efficient Encoder-Free Fourier-based 3D Large Multimodal Model**  
  Guofeng Mei, Wei Lin, Luigi Riz, Yujiao Wu, Yiming Wang, Fabio Poiesi  
  _2026-02-26_ · https://arxiv.org/abs/2602.23153v1  
  <details><summary>Abstract</summary>

  Large Multimodal Models (LMMs) that process 3D data typically rely on heavy, pre-trained visual encoders to extract geometric features. While recent 2D LMMs have begun to eliminate such encoders for efficiency and scalability, extending this paradigm to 3D remains challenging due to the unordered and large-scale nature of point clouds. This leaves a critical unanswered question: How can we design an LMM that tokenizes unordered 3D data effectively and efficiently without a cumbersome encoder? We propose Fase3D, the first efficient encoder-free Fourier-based 3D scene LMM. Fase3D tackles the challenges of scalability and permutation invariance with a novel tokenizer that combines point cloud serialization and the Fast Fourier Transform (FFT) to approximate self-attention. This design enables an effective and computationally minimal architecture, built upon three key innovations: First, we represent large scenes compactly via structured superpoints. Second, our space-filling curve serialization followed by an FFT enables efficient global context modeling and graph-based token merging. Lastly, our Fourier-augmented LoRA adapters inject global frequency-aware interactions into the LLMs at a negligible cost. Fase3D achieves performance comparable to encoder-based 3D LMMs while being significantly more efficient in computation and parameters. Project website: https://tev-fbk.github.io/Fase3D.

  </details>



- **Align then Adapt: Rethinking Parameter-Efficient Transfer Learning in 4D Perception**  
  Yiding Sun, Jihua Zhu, Haozhe Cheng, Chaoyi Lu, Zhichuan Yang, Lin Chen, Yaonan Wang  
  _2026-02-26_ · https://arxiv.org/abs/2602.23069v1  
  <details><summary>Abstract</summary>

  Point cloud video understanding is critical for robotics as it accurately encodes motion and scene interaction. We recognize that 4D datasets are far scarcer than 3D ones, which hampers the scalability of self-supervised 4D models. A promising alternative is to transfer 3D pre-trained models to 4D perception tasks. However, rigorous empirical analysis reveals two critical limitations that impede transfer capability: overfitting and the modality gap. To overcome these challenges, we develop a novel "Align then Adapt" (PointATA) paradigm that decomposes parameter-efficient transfer learning into two sequential stages. Optimal-transport theory is employed to quantify the distributional discrepancy between 3D and 4D datasets, enabling our proposed point align embedder to be trained in Stage 1 to alleviate the underlying modality gap. To mitigate overfitting, an efficient point-video adapter and a spatial-context encoder are integrated into the frozen 3D backbone to enhance temporal modeling capacity in Stage 2. Notably, with the above engineering-oriented designs, PointATA enables a pre-trained 3D model without temporal knowledge to reason about dynamic video content at a smaller parameter cost compared to previous work. Extensive experiments show that PointATA can match or even outperform strong full fine-tuning models, whilst enjoying the advantage of parameter efficiency, e.g. 97.21 \% accuracy on 3D action recognition, $+8.7 \%$ on 4 D action segmentation, and 84.06\% on 4D semantic segmentation.

  </details>



- **UCM: Unifying Camera Control and Memory with Time-aware Positional Encoding Warping for World Models**  
  Tianxing Xu, Zixuan Wang, Guangyuan Wang, Li Hu, Zhongyi Zhang, Peng Zhang, Bang Zhang, Song-Hai Zhang  
  _2026-02-26_ · https://arxiv.org/abs/2602.22960v1  
  <details><summary>Abstract</summary>

  World models based on video generation demonstrate remarkable potential for simulating interactive environments but face persistent difficulties in two key areas: maintaining long-term content consistency when scenes are revisited and enabling precise camera control from user-provided inputs. Existing methods based on explicit 3D reconstruction often compromise flexibility in unbounded scenarios and fine-grained structures. Alternative methods rely directly on previously generated frames without establishing explicit spatial correspondence, thereby constraining controllability and consistency. To address these limitations, we present UCM, a novel framework that unifies long-term memory and precise camera control via a time-aware positional encoding warping mechanism. To reduce computational overhead, we design an efficient dual-stream diffusion transformer for high-fidelity generation. Moreover, we introduce a scalable data curation strategy utilizing point-cloud-based rendering to simulate scene revisiting, facilitating training on over 500K monocular videos. Extensive experiments on real-world and synthetic benchmarks demonstrate that UCM significantly outperforms state-of-the-art methods in long-term scene consistency, while also achieving precise camera controllability in high-fidelity video generation.

  </details>



- **Sapling-NeRF: Geo-Localised Sapling Reconstruction in Forests for Ecological Monitoring**  
  Miguel Ángel Muñoz-Bañón, Nived Chebrolu, Sruthi M. Krishna Moorthy, Yifu Tao, Fernando Torres, Roberto Salguero-Gómez, Maurice Fallon  
  _2026-02-26_ · https://arxiv.org/abs/2602.22731v1  
  <details><summary>Abstract</summary>

  Saplings are key indicators of forest regeneration and overall forest health. However, their fine-scale architectural traits are difficult to capture with existing 3D sensing methods, which make quantitative evaluation difficult. Terrestrial Laser Scanners (TLS), Mobile Laser Scanners (MLS), or traditional photogrammetry approaches poorly reconstruct thin branches, dense foliage, and lack the scale consistency needed for long-term monitoring. Implicit 3D reconstruction methods such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) are promising alternatives, but cannot recover the true scale of a scene and lack any means to be accurately geo-localised. In this paper, we present a pipeline which fuses NeRF, LiDAR SLAM, and GNSS to enable repeatable, geo-localised ecological monitoring of saplings. Our system proposes a three-level representation: (i) coarse Earth-frame localisation using GNSS, (ii) LiDAR-based SLAM for centimetre-accurate localisation and reconstruction, and (iii) NeRF-derived object-centric dense reconstruction of individual saplings. This approach enables repeatable quantitative evaluation and long-term monitoring of sapling traits. Our experiments in forest plots in Wytham Woods (Oxford, UK) and Evo (Finland) show that stem height, branching patterns, and leaf-to-wood ratios can be captured with increased accuracy as compared to TLS. We demonstrate that accurate stem skeletons and leaf distributions can be measured for saplings with heights between 0.5m and 2m in situ, giving ecologists access to richer structural and quantitative data for analysing forest dynamics.

  </details>


