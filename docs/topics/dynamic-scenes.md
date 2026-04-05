# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-04-05 07:21 UTC_

Total papers shown: **4**


---

- **GroundVTS: Visual Token Sampling in Multimodal Large Language Models for Video Temporal Grounding**  
  Rong Fan, Kaiyan Xiao, Minghao Zhu, Liuyi Wang, Kai Dai, Zhao Yang  
  _2026-04-02_ · https://arxiv.org/abs/2604.02093v1  
  <details><summary>Abstract</summary>

  Video temporal grounding (VTG) is a critical task in video understanding and a key capability for extending video large language models (Vid-LLMs) to broader applications. However, existing Vid-LLMs rely on uniform frame sampling to extract video information, resulting in a sparse distribution of key frames and the loss of crucial temporal cues. To address this limitation, we propose Grounded Visual Token Sampling (GroundVTS), a Vid-LLM architecture that focuses on the most informative temporal segments. GroundVTS employs a fine-grained, query-guided mechanism to filter visual tokens before feeding them into the LLM, thereby preserving essential spatio-temporal information and maintaining temporal coherence. Futhermore, we introduce a progressive optimization strategy that enables the LLM to effectively adapt to the non-uniform distribution of visual features, enhancing its ability to model temporal dependencies and achieve precise video localization. We comprehensively evaluate GroundVTS on three standard VTG benchmarks, where it outperforms existing methods, achieving a 7.7-point improvement in mIoU for moment retrieval and 12.0-point improvement in mAP for highlight detection. Code is available at https://github.com/Florence365/GroundVTS.

  </details>



- **MAVFusion: Efficient Infrared and Visible Video Fusion via Motion-Aware Sparse Interaction**  
  Xilai Li, Weijun Jiang, Xiaosong Li, Yang Liu, Hongbin Wang, Tao Ye, Huafeng Li, Haishu Tan  
  _2026-04-02_ · https://arxiv.org/abs/2604.01958v1  
  <details><summary>Abstract</summary>

  Infrared and visible video fusion combines the object saliency from infrared images with the texture details from visible images to produce semantically rich fusion results. However, most existing methods are designed for static image fusion and cannot effectively handle frame-to-frame motion in videos. Current video fusion methods improve temporal consistency by introducing interactions across frames, but they often require high computational cost. To mitigate these challenges, we propose MAVFusion, an end-to-end video fusion framework featuring a motion-aware sparse interaction mechanism that enhances efficiency while maintaining superior fusion quality. Specifically, we leverage optical flow to identify dynamic regions in multi-modal sequences, adaptively allocating computationally intensive cross-modal attention to these sparse areas to capture salient transitions and facilitate inter-modal information exchange. For static background regions, a lightweight weak interaction module is employed to maintain structural and appearance integrity. By decoupling the processing of dynamic and static regions, MAVFusion simultaneously preserves temporal consistency and fine-grained details while significantly accelerating inference. Extensive experiments demonstrate that MAVFusion achieves state-of-the-art performance on multiple infrared and visible video benchmarks, achieving a speed of 14.16\,FPS at $640 \times 480$ resolution. The source code will be available at https://github.com/ixilai/MAVFusion.

  </details>



- **FTPFusion: Frequency-Aware Infrared and Visible Video Fusion with Temporal Perturbation**  
  Xilai Li, Chusheng Fang, Xiaosong Li  
  _2026-04-02_ · https://arxiv.org/abs/2604.01900v1  
  <details><summary>Abstract</summary>

  Infrared and visible video fusion plays a critical role in intelligent surveillance and low-light monitoring. However, maintaining temporal stability while preserving spatial detail remains a fundamental challenge. Existing methods either focus on frame-wise enhancement with limited temporal modeling or rely on heavy spatio-temporal aggregation that often sacrifices high-frequency details. In this paper, we propose FTPFusion, a frequency-aware infrared and visible video fusion method based on temporal perturbation and sparse cross-modal interaction. Specifically, FTPFusion decomposes the feature representations into high-frequency and low-frequency components for collaborative modeling. The high-frequency branch performs sparse cross-modal spatio-temporal interaction to capture motion-related context and complementary details. The low-frequency branch introduces a temporal perturbation strategy to enhance robustness against complex video variations, such as flickering, jitter, and local misalignment. Furthermore, we design an offset-aware temporal consistency constraint to explicitly stabilize cross-frame representations under temporal disturbances. Extensive experiments on multiple public benchmarks demonstrate that FTPFusion consistently outperforms state-of-the-art methods across multiple metrics in both spatial fidelity and temporal consistency. The source code will be available at https://github.com/ixilai/FTPFusion.

  </details>



- **DriveDreamer-Policy: A Geometry-Grounded World-Action Model for Unified Generation and Planning**  
  Yang Zhou, Xiaofeng Wang, Hao Shao, Letian Wang, Guosheng Zhao, Jiangnan Shao, Jiagang Zhu, Tingdong Yu, Zheng Zhu, Guan Huang, et al.  
  _2026-04-02_ · https://arxiv.org/abs/2604.01765v1  
  <details><summary>Abstract</summary>

  Recently, world-action models (WAM) have emerged to bridge vision-language-action (VLA) models and world models, unifying their reasoning and instruction-following capabilities and spatio-temporal world modeling. However, existing WAM approaches often focus on modeling 2D appearance or latent representations, with limited geometric grounding-an essential element for embodied systems operating in the physical world. We present DriveDreamer-Policy, a unified driving world-action model that integrates depth generation, future video generation, and motion planning within a single modular architecture. The model employs a large language model to process language instructions, multi-view images, and actions, followed by three lightweight generators that produce depth, future video, and actions. By learning a geometry-aware world representation and using it to guide both future prediction and planning within a unified framework, the proposed model produces more coherent imagined futures and more informed driving actions, while maintaining modularity and controllable latency. Experiments on the Navsim v1 and v2 benchmarks demonstrate that DriveDreamer-Policy achieves strong performance on both closed-loop planning and world generation tasks. In particular, our model reaches 89.2 PDMS on Navsim v1 and 88.7 EPDMS on Navsim v2, outperforming existing world-model-based approaches while producing higher-quality future video and depth predictions. Ablation studies further show that explicit depth learning provides complementary benefits to video imagination and improves planning robustness.

  </details>


