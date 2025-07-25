# Multi-Head Neural Operator (MHNO) for Modelling Interfacial Dynamics

This repository contains the code and data for the paper:

**Multi-Head Neural Operator for Modelling Interfacial Dynamics**  
*Mohammad Sadegh Eshaghi, Navid Valizadeh, Cosmin Anitescu, Yizheng Wang, Xiaoying Zhuang, Timon Rabczuk*  
📄 [arXiv:2507.17763](https://doi.org/10.48550/arXiv.2507.17763)

---

## 🧠 Overview

Interfacial dynamics play a central role in many physical systems, such as phase transitions, microstructure evolution, and thin-film growth. These problems are modeled by stiff, nonlinear, time-dependent PDEs, which are challenging to solve, especially over long time horizons.

We propose the **Multi-Head Neural Operator (MHNO)**, a neural operator architecture designed to capture **PDEs with long temporal dynamics**. MHNO combines time-step-specific projection operators and explicit temporal message passing, enabling the accurate and efficient prediction of full-time evolutions in a single forward pass.

We validate MHNO on a wide range of phase field models, including:
- Antiphase boundary motion
- Spinodal decomposition
- Pattern formation
- Atomic-scale modeling
- Molecular Beam Epitaxy (MBE)

---

## 📁 Repository Structure
- **AC2D/**, **CH2D/**, **SH2D/**, **PFC2D/**, **MBE2D/**, **AC2D/** – Problem-specific modules and datasets  
- **configs/** – Experiment configuration files  
- **data/** – Raw and preprocessed simulation data  
- **MatlabCode/** – MATLAB scripts used for data creation 
- **Result/** – Generated results, figures, and logs  
- **comparision.py** – Script for benchmarking MHNO against other methods  
- **hyperparameter\_*.py** – Utilities for sweeping, tuning, and reporting hyperparameters  
- **main.py** – Entry point for training and evaluation  
- **networks.py** – Neural network and operator definitions  
- **training.py** – Training loop and checkpointing  
- **post\_processing.py** – Metrics calculation and visualization  
- **utilities.py** – Helper functions 


---

## 📢 Availability

🔒 **The code and data will be made publicly available after the journal review process is complete.**  
Please ⭐ star this repository to get notified when the release is published.

---

## 📫 Contact

For questions or collaboration requests, feel free to reach out to:  
- Mohammad Sadegh Eshaghi: eshaghi.khanghah@iop.uni-hannover.de
- Navid Valizadeh: valizadeh@iop.uni-hannover.de
- Xiaoying Zhuang: zhuang@iop.uni-hannover.de

---

## 📘 Citation

If you use this work in your research, please cite:

```bibtex
@article{eshaghi2025mhno,
  title={Multi-Head Neural Operator for Modelling Interfacial Dynamics},
  author={Eshaghi, Mohammad Sadegh and Valizadeh, Navid and Anitescu, Cosmin and Wang, Yizheng and Zhuang, Xiaoying and Rabczuk, Timon},
  journal={arXiv preprint},
  volume={arXiv:2507.17763},
  year={2025},
  doi={10.48550/arXiv.2507.17763}
}
