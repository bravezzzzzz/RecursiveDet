# RecursiveDet
This is the official implementation of the paper "[RecursiveDet: End-to-End Region-based Recursive Object Detection](https://arxiv.org/abs/2209.10391)"

## Methods
![](readme/pipeline.png)
This paper investigates the region-based object detectors. 
1. Recursive structure for decoder:

   a) We share the decoder parameters among different stages, which significantly reduces the model size.
   
   b) A short recusion loop is made to increase the depth of model.
3. Positional Encoding:

   a) We design bounding box PE into region-based detectos.

   b) Centerness-based PE is proposed to distinguish the RoI feature element and dynamic kernels at different positions within the bounding box.

### The codes will be release soon!

## Citing

If you use this code for your research, please cite

```BibTeX

@article{zhao2023recursivedet,
  title={RecursiveDet: End-to-End Region-based Recursive Object Detection},
  author={Zhao, Jing and Sun, Li and Li, Qingli},
  journal={arXiv preprint arXiv:2307.13619},
  year={2023}
}

```
