# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-03-22 07:06 UTC_

Total papers shown: **3**


---

- **MonoArt: Progressive Structural Reasoning for Monocular Articulated 3D Reconstruction**  
  Haitian Li, Haozhe Xie, Junxiang Xu, Beichen Wen, Fangzhou Hong, Ziwei Liu  
  _2026-03-19_ · https://arxiv.org/abs/2603.19231v1  
  <details><summary>Abstract</summary>

  Reconstructing articulated 3D objects from a single image requires jointly inferring object geometry, part structure, and motion parameters from limited visual evidence. A key difficulty lies in the entanglement between motion cues and object structure, which makes direct articulation regression unstable. Existing methods address this challenge through multi-view supervision, retrieval-based assembly, or auxiliary video generation, often sacrificing scalability or efficiency. We present MonoArt, a unified framework grounded in progressive structural reasoning. Rather than predicting articulation directly from image features, MonoArt progressively transforms visual observations into canonical geometry, structured part representations, and motion-aware embeddings within a single architecture. This structured reasoning process enables stable and interpretable articulation inference without external motion templates or multi-stage pipelines. Extensive experiments on PartNet-Mobility demonstrate that OM achieves state-of-the-art performance in both reconstruction accuracy and inference speed. The framework further generalizes to robotic manipulation and articulated scene reconstruction.

  </details>



- **Motion-o: Trajectory-Grounded Video Reasoning**  
  Bishoy Galoaa, Shayda Moezzi, Xiangyu Bai, Sarah Ostadabbas  
  _2026-03-19_ · https://arxiv.org/abs/2603.18856v1  
  <details><summary>Abstract</summary>

  Recent research has made substantial progress on video reasoning, with many models leveraging spatio-temporal evidence chains to strengthen their inference capabilities. At the same time, a growing set of datasets and benchmarks now provides structured annotations designed to support and evaluate such reasoning. However, little attention has been paid to reasoning about \emph{how} objects move between observations: no prior work has articulated the motion patterns by connecting successive observations, leaving trajectory understanding implicit and difficult to verify. We formalize this missing capability as Spatial-Temporal-Trajectory (STT) reasoning and introduce \textbf{Motion-o}, a motion-centric video understanding extension to visual language models that makes trajectories explicit and verifiable. To enable motion reasoning, we also introduce a trajectory-grounding dataset artifact that expands sparse keyframe supervision via augmentation to yield denser bounding box tracks and a stronger trajectory-level training signal. Finally, we introduce Motion Chain of Thought (MCoT), a structured reasoning pathway that makes object trajectories through discrete \texttt{<motion/>} tag summarizing per-object direction, speed, and scale (of velocity) change to explicitly connect grounded observations into trajectories. To train Motion-o, we design a reward function that compels the model to reason directly over visual evidence, all while requiring no architectural modifications. Empirical results demonstrate that Motion-o improves spatial-temporal grounding and trajectory prediction while remaining fully compatible with existing frameworks, establishing motion reasoning as a critical extension for evidence-based video understanding. Code is available at https://github.com/ostadabbas/Motion-o.

  </details>



- **PhysVideo: Physically Plausible Video Generation with Cross-View Geometry Guidance**  
  Cong Wang, Hanxin Zhu, Xiao Tang, Jiayi Luo, Xin Jin, Long Chen, Fei-Yue Wang, Zhibo Chen  
  _2026-03-19_ · https://arxiv.org/abs/2603.18639v1  
  <details><summary>Abstract</summary>

  Recent progress in video generation has led to substantial improvements in visual fidelity, yet ensuring physically consistent motion remains a fundamental challenge. Intuitively, this limitation can be attributed to the fact that real-world object motion unfolds in three-dimensional space, while video observations provide only partial, view-dependent projections of such dynamics. To address these issues, we propose PhysVideo, a two-stage framework that first generates physics-aware orthogonal foreground videos and then synthesizes full videos with background. In the first stage, Phys4View leverages physics-aware attention to capture the influence of physical attributes on motion dynamics, and enhances spatio-temporal consistency by incorporating geometry-enhanced cross-view attention and temporal attention. In the second stage, VideoSyn uses the generated foreground videos as guidance and learns the interactions between foreground dynamics and background context for controllable video synthesis. To support training, we construct PhysMV, a dataset containing 40K scenes, each consisting of four orthogonal viewpoints, resulting in a total of 160K video sequences. Extensive experiments demonstrate that PhysVideo significantly improves physical realism and spatial-temporal coherence over existing video generation methods. Home page: https://anonymous.4open.science/w/Phys4D/.

  </details>


