# Jose Sanchez Gonzalez

I build robot perception, simulation, and the software to evaluate them.
Seeking ML/AI engineering and robotics roles.
M.S. Artificial Intelligence and B.S. Computer Science, Oregon State University.

[Portfolio](https://jose-sanchez-portfolio-com.vercel.app) · [Résumé](https://jose-sanchez-portfolio-com.vercel.app/resume.pdf) · [LinkedIn](https://linkedin.com/in/jose-j-sanchez-gonzalez-84a800257/) · [Email](mailto:josejsanchez20172@gmail.com)

[![A simulated pruning arm approaches a branch, performs a gated surrogate release, and returns home](https://raw.githubusercontent.com/joses2017smjh/isaac-sim-pruning-workflow/develop/docs/demo/isaac_two_trees_vision_sequence.gif)](https://jose-sanchez-portfolio-com.vercel.app/projects/isaac-pruning-workflow/)

*One controlled Isaac Sim episode. Classical RGB-D tracking, simulator depth and rigid-piece release; this is not a learned pruning policy or a wood-fracture model.*

## Featured work

| Project | What I built | Evidence and limits |
|---|---|---|
| [Vision-guided pruning](https://github.com/joses2017smjh/isaac-sim-pruning-workflow) | RGB-D control, dual-ToF release checks, and an independent recording grader. | Published success and stopped-closure recordings; one selected episode passes 17/17 sequence checks. |
| [SPUR metric depth](https://github.com/joses2017smjh/spur-depth-service) | Depth inference API, multi-view refinement and split ONNX export. | Synthetic validation: 0.0445 ± 0.0057 m RMSE for the published three-pair refiner. Real-orchard accuracy remains unverified. |
| [Humanoid Robustness Ladder](https://github.com/joses2017smjh/bhl-robustness-ladder) | Isaac Lab curricula, MuJoCo evaluation and controls that expose task failures. | 379/384 final-stage first episodes on a fixed, oracle-guided Isaac route; separate MuJoCo demos and negative controls. Hardware walking is not claimed. |
| [MetaNaviT](https://github.com/joses2017smjh/MetaNavT) | Hybrid retrieval, source tracing and reviewable file operations. | Recall@50 0.938 on 136 queries / 61 fixture files, using hash embeddings and overlap reranking. |

## Technical focus

Python, PyTorch, OpenCV, Isaac Sim / Isaac Lab, MuJoCo, ONNX Runtime,
FastAPI and Slurm. I connect model outputs to measurable behavior, document
failure cases, and keep demos linked to the code and evaluation that support them.

The [portfolio](https://jose-sanchez-portfolio-com.vercel.app/#projects) provides
short visual case studies; the repositories contain setup, architecture,
results and current limitations.
