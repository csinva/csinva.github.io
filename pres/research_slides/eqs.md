## equations

- $\beta$ = relevant, $\gamma$ = irrelevant, $i$ = layer index
- linear/conv: $$\begin{align} \beta_i &= W\beta_{i-1} + \frac{|W\beta_{i-1}|}{|W\beta_{i-1}| + |W\gamma_{i-1}|} \cdot b \\  \gamma_i &= W\gamma_{i-1} + \frac{|W\gamma_{i-1}|}{|W\beta_{i-1}| + |W\gamma_{i-1}|} \cdot b\end{align}$$
- maxpool: $\begin{align} 
      max\_idxs &= \underset{idxs}{\text{argmax}} \: \left[ \text{maxpool}(\beta_{i-1} + \gamma_{i-1}; idxs) \right] \\
      \beta_i &=  \beta_{i-1}[max\_idxs] \\ 
      \gamma_i &=  \gamma_{i-1}[max\_idxs]
  \end{align}$
- relu: $\begin{align} 
      \beta_{i} &=  \text{ReLU}(\beta_{i-1}) \\ 
      \gamma_{i} &=  \text{ReLU}(\beta_{i-1} + \gamma_{i-1}) - \text{ReLU}(\beta_{i-1})
  \end{align}$





## properties

- translation invariance:  $ scat\left( f \left( x \right)  \right) =  scat  \left( f \left( x-c \right)  \right)   \forall f \in L^{2} \left( \mathbb{R}^{d} \right) , c \in \mathbb{R}^{d} $ 

- Lipschitz continuity:  $  \forall f, h  \vert  \vert  scat  \left( f \right) - scat  \left( h \right)  \vert  \vert  \leq  \vert  \vert f-h \vert  \vert  $
