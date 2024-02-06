---
layout: notes
title: A gentle introduction to tensor product representations (TPRs)
category: blog
---

Tensor product representations (TPRs), [introduced by Paul Smolensky in 1990](https://www.sciencedirect.com/science/article/abs/pii/000437029090007M), are a popular and powerful model for variable binding and symbolic structures. Here, we give a brief mathematical descriptions of TPRs. The key components of tensor product representations are the concepts of "roles" and "fillers."

- **Roles:** These represent the positions or slots within a structure that can be filled with different entities or values. For example, in a sentence, roles could be subject, verb, and object.
- **Fillers:** These are the specific entities or values that occupy the roles in the structure. Using the sentence example, fillers could be "cat" (subject), "chased" (verb), and "mouse" (object).

The tensor product representation combines roles and fillers to encode structured information in a way that preserves both the elements and their structural relationships.

The tensor product representation involves creating a *composite representation* by taking the sum of the tensor product (i.e. outer product) of vectors representing roles and fillers. Suppose we have vectors for roles $R_i$ and fillers $F_j$, the tensor product of a role and a filler is a matrix (or higher-order tensor) that represents the combined information of that role-filler pair.

# Example: "Cat chases mouse"

<div style="text-align: center;">
    <img src="{{ site.baseurl }}/blog/research/assets/cat_chases_mouse.jpeg" class="noninverted medium_image"/><br/>
    Let's consider a simple sentence: "Cat chases mouse."
</div>
<br/>

- **Roles:** Subject (S), Verb (V), Object (O)
- **Fillers:** Cat, Chases, Mouse

Assume we represent each role and filler as vectors in some $n$-dimensional space. For simplicity, let's say $n=3$.
Orthogonal role vectors can be straightforwardly defined as the standard basis vectors in $\mathbb{R}^3$:

- **Role Vectors:**
  - $R_S = [1, 0, 0]$ for Subject
  - $R_V = [0, 1, 0]$ for Verb
  - $R_O = [0, 0, 1]$ for Object

Let's also define some filler vectors in a 3-dimensional space for consistency:

- **Filler Vectors:**
  - $F_{Cat} = [2, 3, 4]$ for Cat
  - $F_{Chases} = [5, 6, 7]$ for Chases
  - $F_{Mouse} = [8, 9, 10]$ for Mouse


To represent the sentence "Cat chases mouse," we compute the sum of the tensor products of the role and filler vectors:

- $S = R_S \otimes F_{Cat} + R_V \otimes F_{Chases} + R_O \otimes F_{Mouse}$

Each tensor product results in a matrix for each pair, representing a 2D plane in the 3D vector space:

- For $R_S$ and $F_{Cat}$: $$R_S \otimes F_{Cat} = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix} \otimes \begin{bmatrix} 2 \\ 3 \\ 4 \end{bmatrix} = \begin{bmatrix} 2 & 3 & 4 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}$$
- For $R_V$ and $F_{Chases}$: $$R_V \otimes F_{Chases} = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix} \otimes \begin{bmatrix} 5 \\ 6 \\ 7 \end{bmatrix} = \begin{bmatrix} 0 & 0 & 0 \\ 5 & 6 & 7 \\ 0 & 0 & 0 \end{bmatrix}$$

- For $R_O$ and $F_{Mouse}$: $$R_O \otimes F_{Mouse} = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix} \otimes \begin{bmatrix} 8 \\ 9 \\ 10 \end{bmatrix} = \begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \\ 8 & 9 & 10 \end{bmatrix}$$

The composite tensor for the sentence "Cat chases mouse" is the sum of these individual tensor products.
Since the roles are orthogonal, it's easy to see that the unique contribution of each role-filler pair is preserved without interference (in different rows).

This example simplifies many aspects for clarity. In practice, the dimensions for roles and fillers might be much larger to capture more nuanced semantic features, and the mathematical operations might involve more sophisticated mechanisms to encode, manipulate, and decode the structured representations effectively.

**Notes**
- Learning in TPRs involves optimizing the filler and role vectors to optimize the reconstruction of input structures from their TPRs, achievable through gradient descent or other techniques
- The simple outer product forms a strong foundation for symbolic learning and an ind
- TPRs continue to be a major part of ongoing research, e.g. see [this paper](https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/18599) for a forward-looking perspective or the [TP-Transformer](https://arxiv.org/abs/1910.06611) that enhances transformers with role vectors

