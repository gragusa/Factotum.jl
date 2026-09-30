```@contents
Pages = ["mathematics.md"]
Depth = 3
```

# Mathematical introduction

## Data

Throughout this introduction, we use a quarterly US macroeconomic panel based on the data used by Stock and Watson. The series have been transformed for
  stationarity, adjusted for outliers, and locally demeaned. 

```@example intro
using Factotum
using CSV, DataFrames
using LinearAlgebra, Statistics
using Plots

datafile = joinpath(pkgdir(Factotum), "test", "data", "macrodata.csv")
df = CSV.read(datafile, DataFrame)

dates = df.DATE
varnames = names(df, Not(:DATE))
X = Matrix{Float64}(df[:, Not(:DATE)]);
# Series with no missing observations
complete_columns = vec(.!any(isnan, X; dims=1))
Xc = X[:, complete_columns]
```

The matrix `X` contains the full, unbalanced panel, including series with missing observations. The matrix `Xc` contains only series observed throughout the sample and is therefore a balanced panel.

```@example intro
size(X), size(Xc)
```

We begin with `Xc` to introduce principal-component estimation for complete data. We then return to the full panel `X` when discussing factor-model estimation with missing observations.

## The static approximate factor model

Let ``Z \in \mathbb{R}^{T\times N}`` contain ``T`` observations of ``N``
variables. `Factotum.jl` works with

```math
X_{ti} = \frac{Z_{ti}-\mu_i}{s_i},
```

where ``s_i=1`` if `scale=false` or ``s_i`` is the standard deviation of the ``i`` column of ``X``.  

The static factor model is:
```math
X = F\Lambda' + E, \qquad
F\in\mathbb{R}^{T\times r},\quad
\Lambda\in\mathbb{R}^{N\times r}.
```

Row ``t`` of ``F`` contains the latent factors at observation ``t``; row ``i``
of ``\Lambda`` contains the loadings of variable ``i``. Thus

```math
X_{ti}=f_t'\lambda_i+e_{ti}.
```

### PCA estimation

With complete data, estimation solves the best rank-``r`` approximation problem

```math
(\widehat F,\widehat\Lambda)
=\arg\min_{F,\Lambda}\|X-F\Lambda'\|_F^2,
\qquad \Lambda'\Lambda=I_r.
```

Let ``v_1,\ldots,v_r`` be the leading eigenvectors of ``X'X``. A solution to the above problem is

```math
\widehat\Lambda=[v_1,\ldots,v_r], \quad \widehat F=X\widehat\Lambda.
```
```@example intro
fm = FactorModel(Xc, 7; demean=true, scale=true)
Lambda = loadings(fm)
```

```@example intro
factors(fm)
```

### Eigenvalues

Use `eigvals` to extract the eigenvalues:

```@example intro
eigvals(fm)
```

!!! note "Scaling"
      The model eigenvalues are normalized by the number of observations, ``T``.
      Equivalently, they are the eigenvalues of ``X'X/T``.

```@example intro
μ = mean(Xc; dims=1); σ = std(Xc; dims=1, corrected=false)
X_std = (Xc .- μ) ./ σ
υ = eigen(Symmetric(X_std' * X_std)).values
sort(υ; rev=true)[1:7] ./ size(Xc, 1)
```

### Factor correlation

Since ``\widehat F=\widetilde X \widehat \Lambda``, using orthonormality of the eigenvector, we have

```math
\operatorname{Cov}(\widehat F) = \frac{\widehat F'\widehat F}{T}
=\widehat \Lambda'\frac{\widetilde X'\widetilde X}{T}\widehat \Lambda
=\widehat \Lambda'\left( \widehat \Lambda D \widehat \Lambda' \right)\widehat \Lambda = D,
```

where ``D`` is the diagonal matrix with entries equal to the eigenvalues of ``X'X/T``. 

The estimated factors are thus mutually uncorrelated and have variance equal to the eigenvalues. 

```@example intro
cov(factors(fm); corrected=false)
```

### Scree plot and explained variance

The total sample variance of the panel is

```math
\operatorname{tr}\!\left(\frac{X'X}{T}\right)
=\sum_{j=1}^N \nu_j,
```

the proportion of total variance explained by factor ``j`` is

```math
p_j=\frac{\nu_j}{\sum_{\ell=1}^N\nu_\ell}.
```

The cumulative proportion explained by the first ``k`` factors is
``P_k=\sum_{j=1}^k p_j``. Factotum computes ``p_j`` with
[`explained_variance`](@ref):

```@example intro
explained_variance(fm)
```

The total variance of the panel can be calculated by

```@example intro
total_variance(fm)
```

A scree plot displays the ordered eigenvalues. A sharp bend suggests that later
components contribute relatively little additional variation, although the
information criteria discussed below provide a more formal selection rule.

```@example intro
plot(
      1:numfactors(fm),
      explained_variance(fm);
      marker = :circle,
      xlabel = "Number of Factor",
      ylabel = "Variance explained",
      yformatter = y -> "$(round(100y; digits=1))%",
      label = false,
      xticks = 1:numfactors(fm),
  )
```

```@example intro
plot(
      1:numfactors(fm),
      cumsum(explained_variance(fm));
      marker = :circle,
      xlabel = "Number of factors",
      ylabel = "Cumulative variance explained",
      yformatter = y -> "$(round(100y; digits=1))%",
      ylims = (0, 1),
      label = false,
      xticks = 1:numfactors(fm),
  )
```


### Describe

```@example intro
Factotum.describe(fm)
```

## Missing observations

!!! note "Missing values"
    Missing values are represented by `NaN`. 

### EM

With `method=:em`, initially every missing observation with the mean of the available observations for that series. If ``\mathcal O_i`` is the set of observed dates for series ``i``, the initial completed panel is

```math
Z_{ti}^{(0)}=
\begin{cases}
Z_{ti}, & t\in\mathcal O_i,\\
|\mathcal O_i|^{-1}\displaystyle\sum_{s\in\mathcal O_i}Z_{si},
& t\notin\mathcal O_i.
\end{cases}
```

!!! note "Initilization"
    The `init` keyword controls this initial fill. Its default is `nanmean`; for example, `init=nanmedian` uses the observed column median instead. A completely unobserved series contains no information from which to estimate a loading and should normally be removed.


The panel is then centered and (when `scale=true`) scaleed ``Z^{(0)}`` and an initial PCA estimate is computed. Then the following iterations are performed:

1. replace the originally missing entries by the current fitted common
   component ``\widehat f_t'\widehat\lambda_i``, transformed back to the
   original units;
2. re-center and, when `scale=true`, re-standardize the completed panel;
3. recompute the factors and loadings by PCA.

Iterations 1-3 stop when the relative squared change in the common component,

```math
\frac{\|\widehat C^{(m)}-\widehat C^{(m-1)}\|_F^2}
     {\|\widehat C^{(m-1)}\|_F^2},
\qquad \widehat C^{(m)}=\widehat F^{(m)}\widehat\Lambda^{(m)\prime} < tol.
```

```@example intro
fm_em = FactorModel(X, 7; scale=true, method=:em)
Factotum.describe(fm_em)
```

### Least-squares

With `method=:ls`, the package instead alternates observed-data
least-squares regressions:

```math
\widehat\lambda_i
=\arg\min_{\lambda_i}\sum_{t\in\mathcal O_i}
(X_{ti}-f_t'\lambda_i)^2,
\qquad
\widehat f_t
=\arg\min_{f_t}\sum_{i\in\mathcal O_t}
(X_{ti}-f_t'\lambda_i)^2.
```

where ``\mathcal O_i`` and ``\mathcal O_t`` contain the observed entries for series $i$ and observation $t$. 

```@example intro
fm_ls = FactorModel(X, 7; scale=true, method=:ls)
Factotum.describe(fm_ls)
```

!!! note "Default method"
    EM is selected automatically for an incomplete panel when no loading
    constraints are supplied. Use `method=:ls` to request observed-data least
    squares explicitly.

!!! note "Comparing EM and least squares"
    EM and LS need not return identical factor columns. EM repeatedly completes the panel and applies PCA, whereas LS alternates regressions using only observed entries. The `tol` and `maxiter` keywords control convergence for both estimators. The stopping rules are different. EM stops when the relative squared change in the fitted common component is below `tol`. LS stops when the absolute change in the observed-data residual sum of squares is below `tol*T*N`. Consequently, the same numerical tolerance does not impose precisely the same convergence test. Reducing `tol` checks whether a discrepancy is numerical, but it does not force the two algorithms to have the same finite-sample solution. Thus tightening `tol` is a useful convergence check, but it should not be expected to make the reported EM and LS explained-variance vectors coincide.

### Normalization: can least squares reproduce the PCA convention?

Least squares, like PCA, identifies the model only up to an invertible
rotation. Only the fitted common component

```math
C=F\Lambda',\qquad C_{ti}=f_t'\lambda_i,
```

is pinned down by the data (together with any loading constraints). For any
nonsingular ``r\times r`` matrix ``H``,

```math
F^{*}=FH,\qquad \Lambda^{*}=\Lambda H^{-\prime},
```

leave ``C`` unchanged, since ``F^{*}\Lambda^{*\prime}=F\Lambda'``. A
*normalization* is a rule that picks one representative ``(F,\Lambda)`` from
this equivalence class.

The **PCA convention** imposes two conditions at once:

```math
\Lambda'\Lambda=I_r
\quad\text{(orthonormal loadings)},\qquad
\frac{F'F}{T}=D=\operatorname{diag}(\nu_1\ge\cdots\ge\nu_r)
\quad\text{(orthogonal factors, ordered by variance)}.
```

With complete data both hold automatically: ``\widehat F=\widetilde X\widehat\Lambda``
with ``\widehat\Lambda`` the leading eigenvectors of ``\widetilde X'\widetilde X``,
so ``\widehat\Lambda'\widehat\Lambda=I_r`` and
``\operatorname{Cov}(\widehat F)=D``. The EM estimator inherits both,
because every EM iteration ends with a PCA step.

**The answer is yes, but the default LS output only does half of it.** For an
unconstrained panel the default LS solution orthonormalizes the loadings, so
``\widehat\Lambda_{LS}'\widehat\Lambda_{LS}=I_r`` holds exactly, but it does
*not* orthogonalize the factors — ``\operatorname{Cov}(\widehat F_{LS})`` is
generally a full, non-diagonal matrix:

```@example intro
Λ_ls = loadings(fm_ls)
(
    loadings_orthonormal = round.(Λ_ls' * Λ_ls, digits = 8) ≈ I(7),
    factor_cov_offdiag_max = maximum(abs,
        cov(factors(fm_ls); corrected = false) -
        Diagonal(cov(factors(fm_ls); corrected = false))),
)
```

To obtain the *full* PCA convention, rotate the identified common component into
its own principal axes. Take the (thin) SVD of the LS common component,

```math
C_{LS}=F_{LS}\Lambda_{LS}'=USV',
```

and set

```math
\widehat\Lambda=V_{:,1:r},\qquad
\widehat F=U_{:,1:r}\,S_{1:r}.
```

Then ``\widehat\Lambda'\widehat\Lambda=I_r`` and
``\widehat F'\widehat F/T=\operatorname{diag}(s_j^2/T)`` is diagonal and ordered:
this is exactly the PCA convention, with eigenvalues ``\nu_j=s_j^2/T``. Because
``C_{LS}`` is rotation-invariant the SVD is unique up to column signs. It is the
same construction PCA performs on complete data, where the common component is
the rank-``r`` truncation of ``\widetilde X`` and its SVD returns
``\widehat F=\widetilde X\widehat\Lambda``.

```@example intro
C_ls = factors(fm_ls) * loadings(fm_ls)'
U, S, V = svd(C_ls)
F_pca = U[:, 1:7] .* S[1:7]'
Λ_pca = V[:, 1:7]

(
    common_component_unchanged = maximum(abs, F_pca * Λ_pca' - C_ls) < 1e-8,
    loadings_orthonormal = round.(Λ_pca' * Λ_pca, digits = 8) ≈ I(7),
    factors_orthogonal = round.(F_pca' * F_pca / size(C_ls, 1); digits = 6),
    eigenvalues = S[1:7] .^ 2 ./ size(C_ls, 1),
)
```

!!! note "Matching the *convention* is not matching the *estimates*"
    Placing LS in the PCA convention does **not** make the LS factors equal to
    the EM (or complete-data PCA) factors. EM and LS fit the incomplete panel
    differently, so their common components — and hence their eigenvalues
    ``\nu_j`` — differ. Normalizing only makes the second-moment conventions
    comparable; it does not reconcile the estimates. A rotation-invariant
    comparison is provided by the canonical correlations below.

!!! note "Constraints override the PCA rotation"
    The rotation to principal axes destroys loading restrictions
    ``R_i\lambda_i=q_i``. When constraints are supplied the package therefore
    keeps the raw LS solution and disables post-estimation orthonormalization;
    the identity-block normalization of the next section supplies the
    identifying restriction instead.

### Canonical correlation

A rotation-invariant comparison uses the canonical correlations between the two
estimated factor spaces. [`canonical_correlation`](@ref) computes these from
the two factor matrices:

```@example intro
canonical_correlation(factors(fm_em), factors(fm_ls))
```

Values close to one indicate that the methods estimate similar factor spaces,
even when their individual factor columns differ in sign, scale, or rotation.


## Restricted loadings and identification

For a series ``i``, Factotum accepts linear restrictions

```math
R_i\lambda_i=q_i.
```

They are imposed during the loading step of alternating least squares. A single
loading can be fixed or set to zero, a whole row can be fixed, or ``r`` named
series can be given an identity loading block. The last choice imposes ``r^2``
restrictions and resolves rotational indeterminacy:

```math
\Lambda_{(i_1,\ldots,i_r),:}=I_r.
```

When restrictions are present, post-estimation orthonormalization is disabled,
because a rotation would generally destroy them.

### Identity loading normalization

For example, we can identify a seven-factor model for the balanced panel `Xc`
by selecting seven series and fixing their loading rows to the identity matrix.
The selected series name the factors: the first has loading one on factor 1 and
zero on the others, and similarly for the second and third series.

```@example intro
r_restricted = 7
named_series = [1, 2, 3, 4, 5, 6, 7]  # column indices in Xc
constraints = identity_loading(named_series)

fm_restricted = FactorModel(
    Xc,
    r_restricted;
    demean = true,
    scale = true,
    method = :ls,
    constraints = constraints,
)

# Constraints are specified in the units of the input data. Because scale=true,
# request loadings in those original units when checking the restriction.
identity_block = loadings(fm_restricted; original_units=true)[named_series, :]
(
    series = varnames[complete_columns][named_series],
    loadings = identity_block,
    equals_identity = identity_block ≈ Matrix{Float64}(I, r_restricted, r_restricted),
)
```

This normalization fixes the otherwise arbitrary rotation of the factors. It
does not impose ``\Lambda'\Lambda=I_r`` on the complete loading matrix: under
constrained LS, the identity block is preserved instead of applying the usual
post-estimation orthonormalization. With `scale=true`, `loadings(fm_restricted)`
returns loadings for the standardized working data, so the diagonal entries of
the restricted block are ``1/\sigma_i`` rather than one. Use
`loadings(fm_restricted; original_units=true)` as above to recover and verify
the restrictions expressed in the units of `Xc`. Alternatively, setting
`scale=false` makes the stored and original-unit loadings coincide.

```@example intro
loadings(fm_restricted; original_units=true)
```


## Fit, variance, and diagnostics

### The quantity being fitted

The estimated model reproduces each entry of the panel by its **common
component**

```math
\widehat C_{ti}=\widehat f_t'\widehat\lambda_i
=\sum_{j=1}^r \widehat F_{tj}\,\widehat\lambda_{ij},
\qquad
e_{ti}=X_{ti}-\widehat C_{ti},
```

the sum of ``r`` rank-one product terms ``\widehat F_{tj}\widehat\lambda_{ij}``,
one per factor. All fit diagnostics are functions of the residuals ``e_{ti}``
measured on the working (standardized) scale. The panel totals are

```math
\operatorname{TSS}=\sum_{t,i}X_{ti}^2,\qquad
\operatorname{SSR}=\sum_{t,i}e_{ti}^2,
```

where, for an incomplete panel fitted by EM or LS, both sums run only over the
observed cells. The scalars `tss`, `ssr`, and `nobs` summarize the fit over the
whole panel, their ratio gives the overall ``R^2=1-\operatorname{SSR}/\operatorname{TSS}``,
and `residuals` returns the ``T\times N`` matrix ``[e_{ti}]``:

```@example intro
(
    TSS = tss(fm),
    SSR = ssr(fm),
    overall_R2 = 1 - ssr(fm) / tss(fm),
    observations = nobs(fm),
    residual_size = size(residuals(fm)),
)
```

### Series-level fit: `r2` and `total_r2`

For series ``i``, the fit of **all ``r`` factors jointly** is

```math
\operatorname{TSS}_i=\sum_t X_{ti}^2,\qquad
\operatorname{SSR}_i=\sum_t\bigl(X_{ti}-\widehat f_t'\widehat\lambda_i\bigr)^2,
\qquad
R_i^2=1-\frac{\operatorname{SSR}_i}{\operatorname{TSS}_i}.
```

`r2(fm)` returns the vector ``(R_1^2,\ldots,R_N^2)``. The `total_r2` wrapper
attaches names and implements the Tables.jl interface, making it convenient to
sort or convert the results to a `DataFrame`:

```@example intro
complete_varnames = varnames[complete_columns]
series_fit = DataFrame(total_r2(fm; varnames=complete_varnames))
sort!(series_fit, :R2; rev=true)
first(series_fit, 10)
```

### Per-factor fit: `byfactor_r2`

`byfactor_r2` isolates one factor at a time. For series ``i`` and factor ``j``
it reports the fit of the **single product term** ``\widehat F_{tj}\widehat\lambda_{ij}``,

```math
R_{ij}^2=1-\frac{\sum_t\bigl(X_{ti}-\widehat F_{tj}\widehat\lambda_{ij}\bigr)^2}
                {\sum_t X_{ti}^2}.
```

This reuses the *joint-fit* loading ``\widehat\lambda_{ij}`` and factor column
``\widehat F_{tj}``; it is not, in general, a re-estimated univariate
regression. These columns help identify which variables load most strongly on a
given factor:

```@example intro
factor_fit = DataFrame(byfactor_r2(fm; varnames=complete_varnames))
first(factor_fit, 10)
```

### How the two are related — and why normalization matters

Expanding the joint residual sum of squares for series ``i`` in terms of the
single-term contributions ``a_j=\widehat F_j\widehat\lambda_{ij}`` gives

```math
\operatorname{SSR}_i
=\sum_t X_{ti}^2
-2\sum_{j} \widehat\lambda_{ij}\,(X_i'\widehat F_j)
+\sum_{j,k}\widehat\lambda_{ij}\widehat\lambda_{ik}\,(\widehat F_j'\widehat F_k).
```

The cross terms ``\widehat F_j'\widehat F_k`` (``j\ne k``) are what couple the
factors, and they decide whether the per-factor ``R_{ij}^2`` add up.

**Under the PCA normalization** the factors are orthogonal,
``\widehat F_j'\widehat F_k=0`` for ``j\ne k``, so the cross terms vanish and the
per-factor fits decompose the total fit *exactly*:

```math
\sum_{j=1}^r R_{ij}^2=R_i^2.
```

Moreover the residual is then orthogonal to the factor space, so
``\widehat\lambda_{ij}`` equals the OLS coefficient of series ``i`` on factor
``j``, and ``R_{ij}^2`` is precisely the univariate ``R^2`` of regressing series
``i`` on factor ``j`` alone — hence ``0\le R_{ij}^2\le R_i^2``. The complete-data
PCA fit and the EM fit both satisfy this:

```@example intro
bf = Matrix(DataFrame(byfactor_r2(fm))[:, Not(:Variable)])
(
    max_additivity_gap = maximum(abs, vec(sum(bf; dims = 2)) - r2(fm)),
    any_negative = any(bf .< 0),
)
```

**Without the PCA normalization** — for instance the default LS fit, whose
factors are correlated, or any constrained fit — the cross terms
``\widehat F_j'\widehat F_k\ne0`` do not drop out. The ``R_{ij}^2`` then neither
sum to ``R_i^2`` nor need be non-negative, and they depend on the chosen
rotation:

```@example intro
bf_ls = Matrix(DataFrame(byfactor_r2(fm_ls))[:, Not(:Variable)])
(
    additivity_gap_series1 = sum(bf_ls[1, :]) - r2(fm_ls)[1],
    any_negative = any(bf_ls .< 0),
)
```

At the panel level, `explained_variance` gives the aggregate PCA decomposition:
under the PCA normalization factor ``j`` accounts for the fraction
``\nu_j/\sum_{\ell}\nu_\ell`` of ``\operatorname{tr}(X'X/T)``, where ``\nu_j`` is
the ``j``-th eigenvalue of ``X'X/T``. Use `total_r2` or `r2` for the joint fit
of all retained factors, `byfactor_r2` (in the PCA normalization) to attribute
that fit across factors, and `explained_variance` for the panel-level variance
decomposition.

## Selecting the number of factors

For each candidate ``k=0,\ldots,k_{\max}``, let

```math
V(k)=\frac{1}{NT}\|X-\widehat F_k\widehat\Lambda_k'\|_F^2.
```

The Bai–Ng IC family minimizes ``\log V(k)+k\,g(N,T)``; PCp-style criteria use
``V(k)+k\,\widehat\sigma^2g(N,T)``. Factotum also supplies AIC- and BIC-type
penalties. Since criteria encode different penalties, it is often useful to
compare several rather than treating one selection mechanically.

For incomplete data, `FactorModel(Z, kmax; ic=IC2)` can select ``r`` inside the
EM iterations. Otherwise fit ``kmax`` factors once and apply any criterion to
that fit.

For example, the following evaluates the second Bai--Ng information criterion
for zero through seven factors. `criterion` returns the complete objective
path, whereas `numfactors` returns the number of factors at its minimum:

```@example intro
ic2 = IC2(fm, 7)
ic2_path = DataFrame(
    Number_of_factors = 0:7,
    Criterion = criterion(ic2),
)
(
    values = ic2_path,
    selected_factors = numfactors(ic2),
    minimum = findmin(ic2),
)
```

Several penalties can be compared from the same fitted model with
`informationcriteria`:

```@example intro
criterion_types = (IC1, IC2, IC3, PCp1, PCp2, PCp3, AIC1, BIC1)
criteria = informationcriteria(criterion_types, fm, 7)
DataFrame(
    Criterion = string.(criteria),
    Selected_factors = numfactors.(criteria),
)
```

Here `fm` must contain at least `kmax` factors because each criterion evaluates
the nested fits obtained from its first ``0,\ldots,k_{\max}`` components. The
criteria need not select the same rank: each represents a different penalty for
model complexity.
