# NeRF

_Updated: 2026-01-15 07:20 UTC_

Total papers shown: **4**


---

- **HERE: Hierarchical Active Exploration of Radiance Field with Epistemic Uncertainty Minimization**  
  Taekbeom Lee, Dabin Kim, Youngseok Jang, H. Jin Kim  
  _2026-01-12_ · https://arxiv.org/abs/2601.07242v1  
  <details><summary>Abstract</summary>

  We present HERE, an active 3D scene reconstruction framework based on neural radiance fields, enabling high-fidelity implicit mapping. Our approach centers around an active learning strategy for camera trajectory generation, driven by accurate identification of unseen regions, which supports efficient data acquisition and precise scene reconstruction. The key to our approach is epistemic uncertainty quantification based on evidential deep learning, which directly captures data insufficiency and exhibits a strong correlation with reconstruction errors. This allows our framework to more reliably identify unexplored or poorly reconstructed regions compared to existing methods, leading to more informed and targeted exploration. Additionally, we design a hierarchical exploration strategy that leverages learned epistemic uncertainty, where local planning extracts target viewpoints from high-uncertainty voxels based on visibility for trajectory generation, and global planning uses uncertainty to guide large-scale coverage for efficient and comprehensive reconstruction. The effectiveness of the proposed method in active 3D reconstruction is demonstrated by achieving higher reconstruction completeness compared to previous approaches on photorealistic simulated scenes across varying scales, while a hardware demonstration further validates its real-world applicability.

  </details>



- **HOSC: A Periodic Activation with Saturation Control for High-Fidelity Implicit Neural Representations**  
  Michal Jan Wlodarczyk, Danzel Serrano, Przemyslaw Musialski  
  _2026-01-10_ · https://arxiv.org/abs/2601.07870v1  
  <details><summary>Abstract</summary>

  Periodic activations such as sine preserve high-frequency information in implicit neural representations (INRs) through their oscillatory structure, but often suffer from gradient instability and limited control over multi-scale behavior. We introduce the Hyperbolic Oscillator with Saturation Control (HOSC) activation, $\text{HOSC}(x) = \tanh\bigl(β\sin(ω_0 x)\bigr)$, which exposes an explicit parameter $β$ that controls the Lipschitz bound of the activation by $βω_0$. This provides a direct mechanism to tune gradient magnitudes while retaining a periodic carrier. We provide a mathematical analysis and conduct a comprehensive empirical study across images, audio, video, NeRFs, and SDFs using standardized training protocols. Comparative analysis against SIREN, FINER, and related methods shows where HOSC provides substantial benefits and where it achieves competitive parity. Results establish HOSC as a practical periodic activation for INR applications, with domain-specific guidance on hyperparameter selection. For code visit the project page https://hosc-nn.github.io/ .

  </details>



- **QNeRF: Neural Radiance Fields on a Simulated Gate-Based Quantum Computer**  
  Daniele Lizzio Bosco, Shuteng Wang, Giuseppe Serra, Vladislav Golyanik  
  _2026-01-08_ · https://arxiv.org/abs/2601.05250v1  
  <details><summary>Abstract</summary>

  Recently, Quantum Visual Fields (QVFs) have shown promising improvements in model compactness and convergence speed for learning the provided 2D or 3D signals. Meanwhile, novel-view synthesis has seen major advances with Neural Radiance Fields (NeRFs), where models learn a compact representation from 2D images to render 3D scenes, albeit at the cost of larger models and intensive training. In this work, we extend the approach of QVFs by introducing QNeRF, the first hybrid quantum-classical model designed for novel-view synthesis from 2D images. QNeRF leverages parameterised quantum circuits to encode spatial and view-dependent information via quantum superposition and entanglement, resulting in more compact models compared to the classical counterpart. We present two architectural variants. Full QNeRF maximally exploits all quantum amplitudes to enhance representational capabilities. In contrast, Dual-Branch QNeRF introduces a task-informed inductive bias by branching spatial and view-dependent quantum state preparations, drastically reducing the complexity of this operation and ensuring scalability and potential hardware compatibility. Our experiments demonstrate that -- when trained on images of moderate resolution -- QNeRF matches or outperforms classical NeRF baselines while using less than half the number of parameters. These results suggest that quantum machine learning can serve as a competitive alternative for continuous signal representation in mid-level tasks in computer vision, such as 3D representation learning from 2D observations.

  </details>



- **DivAS: Interactive 3D Segmentation of NeRFs via Depth-Weighted Voxel Aggregation**  
  Ayush Pande  
  _2026-01-08_ · https://arxiv.org/abs/2601.04860v1  
  <details><summary>Abstract</summary>

  Existing methods for segmenting Neural Radiance Fields (NeRFs) are often optimization-based, requiring slow per-scene training that sacrifices the zero-shot capabilities of 2D foundation models. We introduce DivAS (Depth-interactive Voxel Aggregation Segmentation), an optimization-free, fully interactive framework that addresses these limitations. Our method operates via a fast GUI-based workflow where 2D SAM masks, generated from user point prompts, are refined using NeRF-derived depth priors to improve geometric accuracy and foreground-background separation. The core of our contribution is a custom CUDA kernel that aggregates these refined multi-view masks into a unified 3D voxel grid in under 200ms, enabling real-time visual feedback. This optimization-free design eliminates the need for per-scene training. Experiments on Mip-NeRF 360° and LLFF show that DivAS achieves segmentation quality comparable to optimization-based methods, while being 2-2.5x faster end-to-end, and up to an order of magnitude faster when excluding user prompting time.

  </details>


