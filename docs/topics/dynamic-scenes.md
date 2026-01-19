# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-01-19 06:56 UTC_

Total papers shown: **2**


---

- **VidLeaks: Membership Inference Attacks Against Text-to-Video Models**  
  Li Wang, Wenyu Chen, Ning Yu, Zheng Li, Shanqing Guo  
  _2026-01-16_ · https://arxiv.org/abs/2601.11210v1  
  <details><summary>Abstract</summary>

  The proliferation of powerful Text-to-Video (T2V) models, trained on massive web-scale datasets, raises urgent concerns about copyright and privacy violations. Membership inference attacks (MIAs) provide a principled tool for auditing such risks, yet existing techniques - designed for static data like images or text - fail to capture the spatio-temporal complexities of video generation. In particular, they overlook the sparsity of memorization signals in keyframes and the instability introduced by stochastic temporal dynamics. In this paper, we conduct the first systematic study of MIAs against T2V models and introduce a novel framework VidLeaks, which probes sparse-temporal memorization through two complementary signals: 1) Spatial Reconstruction Fidelity (SRF), using a Top-K similarity to amplify spatial memorization signals from sparsely memorized keyframes, and 2) Temporal Generative Stability (TGS), which measures semantic consistency across multiple queries to capture temporal leakage. We evaluate VidLeaks under three progressively restrictive black-box settings - supervised, reference-based, and query-only. Experiments on three representative T2V models reveal severe vulnerabilities: VidLeaks achieves AUC of 82.92% on AnimateDiff and 97.01% on InstructVideo even in the strict query-only setting, posing a realistic and exploitable privacy risk. Our work provides the first concrete evidence that T2V models leak substantial membership information through both sparse and temporal memorization, establishing a foundation for auditing video generation systems and motivating the development of new defenses. Code is available at: https://zenodo.org/records/17972831.

  </details>



- **Convolutions Need Registers Too: HVS-Inspired Dynamic Attention for Video Quality Assessment**  
  Mayesha Maliha R. Mithila, Mylene C. Q. Farias  
  _2026-01-16_ · https://arxiv.org/abs/2601.11045v1  
  <details><summary>Abstract</summary>

  No-reference video quality assessment (NR-VQA) estimates perceptual quality without a reference video, which is often challenging. While recent techniques leverage saliency or transformer attention, they merely address global context of the video signal by using static maps as auxiliary inputs rather than embedding context fundamentally within feature extraction of the video sequence. We present Dynamic Attention with Global Registers for Video Quality Assessment (DAGR-VQA), the first framework integrating register-token directly into a convolutional backbone for spatio-temporal, dynamic saliency prediction. By embedding learnable register tokens as global context carriers, our model enables dynamic, HVS-inspired attention, producing temporally adaptive saliency maps that track salient regions over time without explicit motion estimation. Our model integrates dynamic saliency maps with RGB inputs, capturing spatial data and analyzing it through a temporal transformer to deliver a perceptually consistent video quality assessment. Comprehensive tests conducted on the LSVQ, KonVid-1k, LIVE-VQC, and YouTube-UGC datasets show that the performance is highly competitive, surpassing the majority of top baselines. Research on ablation studies demonstrates that the integration of register tokens promotes the development of stable and temporally consistent attention mechanisms. Achieving an efficiency of 387.7 FPS at 1080p, DAGR-VQA demonstrates computational performance suitable for real-time applications like multimedia streaming systems.

  </details>


