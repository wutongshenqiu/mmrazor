# AutoSlim
> [ResRep: Lossless CNN Pruning via Decoupling Remembering and Forgetting](https://arxiv.org/abs/2007.03260)

<!-- [ALGORITHM] -->

## Abstract

We propose ResRep, a novel method for lossless channel pruning (a.k.a.  filter pruning), which slims down a CNN by reducing the width (number of output channels) of convolutional layers. Inspired by the neurobiology  research about the independence of remembering and forgetting, we  propose to re-parameterize a CNN into the remembering parts and  forgetting parts, where the former learn to maintain the performance and the latter learn to prune. Via training with regular SGD on the former  but a novel update rule with penalty gradients on the latter, we realize structured sparsity. Then we equivalently merge the remembering and  forgetting parts into the original architecture with narrower layers. In this sense, ResRep can be viewed as a successful application of  Structural Re-parameterization. Such a methodology distinguishes ResRep  from the traditional learning-based pruning paradigm that applies a  penalty on parameters to produce sparsity, which may suppress the  parameters essential for the remembering. ResRep slims down a standard  ResNet-50 with 76.15% accuracy on ImageNet to a narrower one with only  45% FLOPs and no accuracy drop, which is the first to achieve lossless  pruning with such a high compression ratio.

![pipeline](/docs/en/imgs/model_zoo/resrep/pipeline.png)


## Introduction
### Train
<pre>
python tools/mmcls/train_mmcls.py \
  configs/pruning/resrep/resrep_resnet50_supernet_in1k-paper.py \
  --work-dir <em>your_work_dir</em>
</pre>

### Convert Compactor
<pre>
python tools/model_converters/convert_compactor.py \
  configs/pruning/resrep/resrep_resnet50_supernet_in1k-paper.py
  <em>your_checkpoint_path</em> \
  --output-dir <em>your_output_dir</em>
</pre>

### Test
<pre>
python tools/mmcls/test_mmcls.py \
  configs/pruning/resrep/resrep_resnet50_supernet_in1k-paper-deploy.py \
  <em>your_pruned_checkpoint_path</em> --metrics accuracy
</pre>


## Citation

```latex
@inproceedings{ding2021resrep,
    title={ResRep: Lossless CNN Pruning via Decoupling Remembering and Forgetting},
    author={Ding, Xiaohan and Hao, Tianxiang and Tan, Jianchao and Liu, Ji and Han, Jungong and Guo, Yuchen and Ding, Guiguang},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages={4510--4520},
    year={2021}
}
```
