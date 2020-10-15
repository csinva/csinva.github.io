# Unsupervised Natural Visual Perception

**stella yu, uc berkeley**



- intuition
  - high-level categories automatically emerge from low-level labels
  - good representations should learn good similarities between images
- Paper: [Unsupervised feature learning via non-parametric instance discrimination](http://openaccess.thecvf.com/content_cvpr_2018/html/Wu_Unsupervised_Feature_Learning_CVPR_2018_paper.html)
  - instance discrimination - maintain ability to differentiate all the instances
  - like having $n$ classes
  - this is a way to learn good unsupervised features
  - classifier is now non-parameteric = metric learning
  - classify using most similar instance
    - instead of $n$-dimensional output, now learn some feature vector
  - can learn this with MLE sped up by noise-contrastive estimation NCE

[SegSort: Segmentation by Discriminative Sorting of Segments](https://openaccess.thecvf.com/content_ICCV_2019/html/Hwang_SegSort_Segmentation_by_Discriminative_Sorting_of_Segments_ICCV_2019_paper.html) - perform segmentation fully unsupervised by clustering segments

- pixel-to-segment contrastive learning - get an oversegmentation and then learn similarities between the segments