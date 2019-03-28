- berardino 17 eigendistortions
  - **Fisher info matrix** under certain assumptions = $Jacob^TJacob$ (pixels x pixels) where *Jacob* is the Jacobian matrix for the function f action on the pixels x
  - most and least noticeable distortion directions corresponding to the eigenvectors of the Fisher info matrix
- gao_19_v1_repr
  - don't learn from images - v1 repr should come from motion like it does in the real world
  - repr
    - vector of local content
    - matrix of local displacement
  - why is this repr nice?
    - separate reps of static image content and change due to motion
    - disentangled rotations
  - learning
    - predict next image given current image + displacement field
    - predict next image vector given current frame vectors + displacement
- kietzmann_18_dnn_in_neuro_rvw
- friston_10_free_energy
  - ![friston_free_energy](data_ovw/friston_free_energy.png)



# deeptune-style

- ponce_19_evolving_stimuli
- bashivan_18_ann_synthesis