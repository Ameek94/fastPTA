# Fisher analysis to assess detectability of intensity and circular polarization anisotropy in the PTA data.

The signal spectrum is taken to be a power law.

Throughout, we work at a single frequency bin $f_k$. The cross-spectral density matrix between $N_p$ pulsars is

$$\mathbf{S}_k = \underbrace{\mathbf{R}_k}_{\text{real, symmetric}} + i\underbrace{\mathbf{A}_k}_{\text{real, antisymmetric}}$$

where

$$R_{k,ab} = S(f_k)\sum_{LM} c^I_{LM}\,\Gamma^I_{LM,ab} + N_{ab}(f_k), \qquad A_{k,ab} = S(f_k)\sum_{LM} c^V_{LM}\,\tilde{\Gamma}^V_{LM,ab}$$

and $\Gamma^V_{LM,ab} = i\tilde{\Gamma}^V_{LM,ab}$ with $\tilde{\Gamma}^V$ real and antisymmetric ($\tilde{\Gamma}^V_{ab} = -\tilde{\Gamma}^V_{ba}$). The parameter vector is $\boldsymbol{\theta} = \{c^I_{LM}, c^V_{LM}\}$.

We use the notation $\mathbf{G}^I_{LM} \equiv S(f_k)\Gamma^I_{LM}$ (real symmetric) and $\mathbf{G}^V_{LM} \equiv S(f_k)\tilde{\Gamma}^V_{LM}$ (real antisymmetric) for the signal-weighted ORF matrices, so that $\partial\mathbf{S}_k/\partial c^I_{LM} = \mathbf{G}^I_{LM}$ and $\partial\mathbf{S}_k/\partial c^V_{LM} = i\mathbf{G}^V_{LM}$.

Note $c^I_{00}$ is the monopole intensity coefficient should not be included in the Fisher since it is completely degenerate with the overall signal amplitude. $c^V_{00}$ should also not to be included since the monopole of circular polarization is not observable. We therefore restrict to $L\geq 1$ for both intensity and circular polarization..

The folders 'example_paper_anisotropies' and 'examples_paper_cosmic_variance' contain several notebooks that perform intensity anisotropy analysis. In the folder 'examples_circ_pol', we would like to perform a similar analysis for both intensity and circular polarization anisotropies, and to compare the results. 

In order of priority, the notebooks we would like to replicate from the intensity only analysis in the 'example_paper_anisotropies' and 'examples_paper_cosmic_variance' folders are:
linear_basis.ipynb: Perform C_l^I and C_l^V upper limit analysis using the linear basis. This is the most important notebook to have, and should be done first. For circular polarization there are no existing upper limits but we we can compare against prior only upper limits for C_l^V.

plot_all_results.ipynb: extend again to intensity and circular polarization anisotropies, and compare the results.

Why the Fisher is block-diagonal at fiducial $V=0$
At fiducial $V=0$, the V-mode signal $A=0$, so the doubled covariance is

$$\mathbf{M} = \tfrac{1}{2}\begin{pmatrix}R & 0 \\ 0 & R\end{pmatrix}, \qquad \mathbf{M}^{-1} = 2\begin{pmatrix}R^{-1} & 0 \\ 0 & R^{-1}\end{pmatrix}$$

The parameter derivatives split into two structural types. For any I-sector parameter (both spectral parameters and $c^I_{LM}$, since $\partial A/\partial\theta_\text{spec}=0$ when all $c^V_{LM}=0$):

$$\frac{\partial\mathbf{M}}{\partial\theta_I} = \tfrac{1}{2}\begin{pmatrix}X & 0 \\ 0 & X\end{pmatrix} \quad\Rightarrow\quad \mathbf{M}^{-1}\frac{\partial\mathbf{M}}{\partial\theta_I} = \begin{pmatrix}R^{-1}X & 0 \\ 0 & R^{-1}X\end{pmatrix}$$

For any V-sector parameter ($c^V_{LM}$):

$$\frac{\partial\mathbf{M}}{\partial\theta_V} = \tfrac{1}{2}\begin{pmatrix}0 & -G^V \\ G^V & 0\end{pmatrix} \quad\Rightarrow\quad \mathbf{M}^{-1}\frac{\partial\mathbf{M}}{\partial\theta_V} = \begin{pmatrix}0 & -R^{-1}G^V \\ R^{-1}G^V & 0\end{pmatrix}$$

The cross-block Fisher element is then

$$F_{\theta_I,\,\theta_V} = \tfrac{1}{2}\operatorname{Tr}\!\left[\begin{pmatrix}R^{-1}X & 0 \\ 0 & R^{-1}X\end{pmatrix}\begin{pmatrix}0 & -R^{-1}G^V \\ R^{-1}G^V & 0\end{pmatrix}\right] = \tfrac{1}{2}\operatorname{Tr}\begin{pmatrix}0 & \cdots \\ \cdots & 0\end{pmatrix} = 0$$

The product has zero diagonal blocks, so the trace vanishes exactly. This holds for all I-sector parameters — spectral and anisotropy alike — because $\partial A/\partial\theta_\text{spec}=0$ at fiducial $V=0$.

