# macrodata.csv - Data Description

This file documents the data processing pipeline used to generate `macrodata.csv` from the raw macroeconomic data in `hom_fac_1.xlsx`.

## Source Data

The raw data comes from `hom_fac_1.xlsx` which contains:
- **Sheet 1**: Monthly macroeconomic series with 8 header rows
- **Sheet 2**: Quarterly macroeconomic series with 7 header rows

### Header Row Structure (Monthly Data)

| Row | Content |
|-----|---------|
| 1 | Long variable labels |
| 2 | Short variable labels |
| 3 | Aggregation codes |
| 4 | Transformation codes (tcode) |
| 5 | Deflation codes |
| 6 | Outlier codes |
| 7 | Include in factor estimation |
| 8 | Category codes (ordering) |

### Header Row Structure (Quarterly Data)

| Row | Content |
|-----|---------|
| 1 | Long variable labels |
| 2 | Short variable labels |
| 3 | Transformation codes (tcode) |
| 4 | Deflation codes |
| 5 | Outlier codes |
| 6 | Include in factor estimation |
| 7 | Category codes (ordering) |

## Processing Pipeline

The data undergoes the following transformations in order:

### 1. Price Deflation

Nominal series are deflated to obtain real values based on deflation codes:

| Code | Deflator (Monthly) | Deflator (Quarterly) |
|------|-------------------|---------------------|
| 1 | PCEPI | PCECTPI |
| 2 | PCEPILFE | JCXFE |
| 3 | - | GDPCTPI |

Series with deflation code $d$ are divided by the corresponding deflator:
$$x_t^{real} = \frac{x_t^{nominal}}{P_t}$$

### 2. Temporal Aggregation (Monthly → Quarterly)

Monthly series are converted to quarterly frequency using the **quarterly average** of the three monthly observations within each quarter.

### 3. Stationarity Transformations

Each series is transformed according to its transformation code (tcode):

| Code | Transformation | Formula |
|------|----------------|---------|
| 1 | Level (no transformation) | $y_t = x_t$ |
| 2 | First difference | $y_t = x_t - x_{t-1}$ |
| 3 | Second difference | $y_t = x_t - 2x_{t-1} + x_{t-2}$ |
| 4 | Logarithm | $y_t = \log(x_t)$ |
| 5 | First difference of log | $y_t = \log(x_t) - \log(x_{t-1}) \approx \% \Delta x_t$ |
| 6 | Second difference of log | $y_t = \Delta^2 \log(x_t)$ |
| 7 | First difference of percent change | $y_t = \Delta\left(\frac{x_t - x_{t-1}}{x_{t-1}}\right)$ |

### 4. Outlier Treatment

Outliers are detected and treated using an IQR-based method.

#### Detection
An observation $y_t$ is flagged as an outlier if:
$$|y_t - \text{median}(y)| > \tau \cdot \text{IQR}(y)$$

where $\text{IQR} = Q_{75} - Q_{25}$ and the threshold $\tau$ depends on the outlier code:
- Outlier code 1: $\tau = 4.5$
- Outlier code 2: $\tau = 3.0$

#### Treatment Method
Outliers are replaced using **local median** with a window of ±5 observations (method 4):
$$y_t^{adj} = \text{median}(y_{t-5}, \ldots, y_{t+5})$$

### 5. Local Demeaning (Biweight Filter)

To remove slow-moving trends while preserving higher-frequency variation, each series is locally demeaned using a biweight kernel filter with bandwidth $h = 100$ quarters.

The local mean at time $t$ is:
$$\bar{x}_t = \sum_{s=1}^{T} w_{ts} x_s$$

where the biweight kernel weights are:
$$w_{ts} = \frac{K\left(\frac{s-t}{h}\right)}{\sum_{j=1}^{T} K\left(\frac{j-t}{h}\right)}$$

and the biweight kernel is:
$$K(u) = \frac{15}{16}(1 - u^2)^2 \cdot \mathbf{1}(|u| < 1)$$

The demeaned series is:
$$\tilde{x}_t = x_t - \bar{x}_t$$

### 6. Merge and Filter

- Monthly-origin and quarterly-origin series are merged by date
- Data is filtered to observations after January 1, 1959

## Output Format

The resulting `macrodata.csv` file contains:
- **First column**: DATE (quarterly dates)
- **Remaining columns**: Transformed macroeconomic series

### Missing Values

Missing values are preserved as empty cells (CSV) or `NaN`. These arise from:
- Original missing data in the source
- Initial observations lost due to differencing (tcode 2, 5: 1 obs; tcode 3, 6, 7: 2 obs)
- Values that could not be computed (e.g., log of non-positive numbers)

## Global Parameters

The following parameters control the processing (defined in `factor_model_simp.jl`):

```julia
bw_demean = true    # Apply local demeaning
bw = 100            # Bandwidth for biweight filter (quarters)
outliers = true     # Enable outlier treatment
omethod = 4         # Outlier method (4 = local median, window=5)
threshold1 = 4.5    # IQR multiplier for outlier code 1
threshold2 = 3.0    # IQR multiplier for outlier code 2
```

## Reference

The data processing follows the methodology in:

Lazarus, E., Lewis, D. J., Stock, J. H., & Watson, M. W. (2018). HAR Inference: Recommendations for Practice. *Journal of Business & Economic Statistics*, 36(4), 541-559.
