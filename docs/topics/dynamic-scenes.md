# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-03-23 07:38 UTC_

Total papers shown: **4**


---

- **NEC-Diff: Noise-Robust Event-RAW Complementary Diffusion for Seeing Motion in Extreme Darkness**  
  Haoyue Liu, Jinghan Xu, Luxin Feng, Hanyu Zhou, Haozhi Zhao, Yi Chang, Luxin Yan  
  _2026-03-20_ · https://arxiv.org/abs/2603.20005v1  
  <details><summary>Abstract</summary>

  High-quality imaging of dynamic scenes in extremely low-light conditions is highly challenging. Photon scarcity induces severe noise and texture loss, causing significant image degradation. Event cameras, featuring a high dynamic range (120 dB) and high sensitivity to motion, serve as powerful complements to conventional cameras by offering crucial cues for preserving subtle textures. However, most existing approaches emphasize texture recovery from events, while paying little attention to image noise or the intrinsic noise of events themselves, which ultimately hinders accurate pixel reconstruction under photon-starved conditions. In this work, we propose NEC-Diff, a novel diffusion-based event-RAW hybrid imaging framework that extracts reliable information from heavily noisy signals to reconstruct fine scene structures. The framework is driven by two key insights: (1) combining the linear light-response property of RAW images with the brightness-change nature of events to establish a physics-driven constraint for robust dual-modal denoising; and (2) dynamically estimating the SNR of both modalities based on denoising results to guide adaptive feature fusion, thereby injecting reliable cues into the diffusion process for high-fidelity visual reconstruction. Furthermore, we construct the REAL (Raw and Event Acquired in Low-light) dataset which provides 47,800 pixel-aligned low-light RAW images, events, and high-quality references under 0.001-0.8 lux illumination. Extensive experiments demonstrate the superiority of NEC-Diff under extreme darkness. The project are available at: https://github.com/jinghan-xu/NEC-Diff.

  </details>



- **RAM: Recover Any 3D Human Motion in-the-Wild**  
  Sen Jia, Ning Zhu, Jinqin Zhong, Jiale Zhou, Huaping Zhang, Jenq-Neng Hwang, Lei Li  
  _2026-03-20_ · https://arxiv.org/abs/2603.19929v1  
  <details><summary>Abstract</summary>

  RAM incorporates a motion-aware semantic tracker with adaptive Kalman filtering to achieve robust identity association under severe occlusions and dynamic interactions. A memory-augmented Temporal HMR module further enhances human motion reconstruction by injecting spatio-temporal priors for consistent and smooth motion estimation. Moreover, a lightweight Predictor module forecasts future poses to maintain reconstruction continuity, while a gated combiner adaptively fuses reconstructed and predicted features to ensure coherence and robustness. Experiments on in-the-wild multi-person benchmarks such as PoseTrack and 3DPW, demonstrate that RAM substantially outperforms previous state-of-the-art in both Zero-shot tracking stability and 3D accuracy, offering a generalizable paradigm for markerless 3D human motion capture in-the-wild.

  </details>



- **PCSTracker: Long-Term Scene Flow Estimation for Point Cloud Sequences**  
  Min Lin, Gangwei Xu, Xianqi Wang, Yuyi Peng, Xin Yang  
  _2026-03-20_ · https://arxiv.org/abs/2603.19762v1  
  <details><summary>Abstract</summary>

  Point cloud scene flow estimation is fundamental to long-term and fine-grained 3D motion analysis. However, existing methods are typically limited to pairwise settings and struggle to maintain temporal consistency over long sequences as geometry evolves, occlusions emerge, and errors accumulate. In this work, we propose PCSTracker, the first end-to-end framework specifically designed for consistent scene flow estimation in point cloud sequences. Specifically, we introduce an iterative geometry motion joint optimization module (IGMO) that explicitly models the temporal evolution of point features to alleviate correspondence inconsistencies caused by dynamic geometric changes. In addition, a spatio-temporal point trajectory update module (STTU) is proposed to leverage broad temporal context to infer plausible positions for occluded points, ensuring coherent motion estimation. To further handle long sequences, we employ an overlapping sliding-window inference strategy that alternates cross-window propagation and in-window refinement, effectively suppressing error accumulation and maintaining stable long-term motion consistency. Extensive experiments on the synthetic PointOdyssey3D and real-world ADT3D datasets show that PCSTracker achieves the best accuracy in long-term scene flow estimation and maintains real-time performance at 32.5 FPS, while demonstrating superior 3D motion understanding compared to RGB-D-based approaches.

  </details>



- **PhysNeXt: Next-Generation Dual-Branch Structured Attention Fusion Network for Remote Photoplethysmography Measurement**  
  Junzhe Cao, Bo Zhao, Zhiyi Niu, Dan Guo, Yue Sun, Haochen Liang, Yong Xu, Zitong YU  
  _2026-03-20_ · https://arxiv.org/abs/2603.19752v1  
  <details><summary>Abstract</summary>

  Remote photoplethysmography (rPPG) enables contactless measurement of heart rate and other vital signs by analyzing subtle color variations in facial skin induced by cardiac pulsation. Current rPPG methods are mainly based on either end-to-end modeling from raw videos or intermediate spatial-temporal map (STMap) representations. The former preserves complete spatiotemporal information and can capture subtle heartbeat-related signals, but it also introduces substantial noise from motion artifacts and illumination variations. The latter stacks the temporal color changes of multiple facial regions of interest into compact two-dimensional representations, significantly reducing data volume and computational complexity, although some high-frequency details may be lost. To effectively integrate the mutual strengths, we propose PhysNeXt, a dual-input deep learning framework that jointly exploits video frames and STMap representations. By incorporating a spatio-temporal difference modeling unit, a cross-modal interaction module, and a structured attention-based decoder, PhysNeXt collaboratively enhances the robustness of pulse signal extraction. Experimental results demonstrate that PhysNeXt achieves more stable and fine-grained rPPG signal recovery under challenging conditions, validating the effectiveness of joint modeling of video and STMap representations. The codes will be released.

  </details>


