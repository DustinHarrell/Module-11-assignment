# Additive Models and Singular Design Matrices in R

ISwR exercises 12.1 and 12.3, by Dustin Harrell, LIS4273.001F26, November 3, 2026

This post works through two exercises from chapter 12 of *Introductory Statistics with R*. The first fits an additive model to a crossover study and compares it with the t tests we already know. The second looks at how R builds model matrices for factors, and at what happens when the design matrix is rank-deficient. All output below was produced in R with the ISwR package; for 12.3 I used `set.seed(1)` so the random response is reproducible. The full code is in [`ch12_1.R`](ch12_1.R) and [`ch12_3.R`](ch12_3.R).

## Exercise 12.1: An additive model for the ashina data

The `ashina` data come from a crossover study of 16 migraine patients. Each patient was measured on an active treatment and on a placebo, and `grp` records the order: group 1 (10 patients) received the active treatment first, group 2 (6 patients) received placebo first. The model therefore has three additive effects: subject, period, and treatment.

### Setting up the data

The data are stored one row per patient, so I first stack them into one row per measurement. The period for the active measurement is `grp`, and for the placebo measurement it is `3 - grp`, because patients in group 1 had placebo in period 2 and vice versa.

```r
library(ISwR)
ashina$subject <- factor(1:16)
attach(ashina)
act  <- data.frame(vas=vas.active, subject, treat=1, period=grp)
plac <- data.frame(vas=vas.plac,   subject, treat=0, period=3-grp)
ashina.long <- rbind(act, plac)
detach(ashina)
```

### Fitting the additive model

```r
fit <- lm(vas ~ subject + period + treat, data=ashina.long)
summary(fit)
anova(fit)
```

The key estimates are:

| Term | Estimate | Std. error | t | p |
|---|---|---|---|---|
| period (2 vs 1) | 14.17 | 13.69 | 1.03 | 0.318 |
| treat (active vs placebo) | -39.33 | 13.69 | -2.87 | 0.012 |

The 95% confidence interval for the treatment effect is (-68.7, -10.0). The subject term has F = 2.43 on 15 and 14 df (p = 0.053), so there is some, but not conclusive, evidence of differences between patients.

### Comparison with t tests

```r
t.test(ashina$vas.active, ashina$vas.plac, paired=TRUE)

ashina$diff <- ashina$vas.active - ashina$vas.plac
t.test(diff ~ grp, data=ashina, var.equal=TRUE)
```

| Question | Additive model | t test |
|---|---|---|
| Treatment effect | -39.33, p = 0.012 | Paired t test: -42.88, t = -3.23, df = 15, p = 0.0056, CI (-71.2, -14.6) |
| Period effect | 14.17, t = 1.03, p = 0.318 | Two-sample t test on the within-patient differences by group: t = -1.03, p = 0.318 |

Three things stand out.

1. **The qualitative conclusions agree.** Active treatment lowers the VAS score, and there is no significant period effect.
2. **The period test is identical.** Comparing the within-patient differences between the two sequence groups is exactly a test for a period effect, and it reproduces the model's t value up to sign.
3. **The treatment estimates differ.** The paired t test averages all 16 differences, so the 10 patients in group 1 carry more weight than the 6 in group 2. The additive model instead gives the unweighted average of the two group means (-53.5 and -25.17), which is -39.33. It also adjusts for period and has 14 residual degrees of freedom rather than 15, which is why its p-value is somewhat larger. With an unbalanced crossover the model's estimate is the cleaner one, since it is not confounded with the period effect.

One practical note: in `anova(fit)` the treatment F statistic (8.26) equals the square of its t value because treatment is entered last, but the period F is not adjusted for treatment. For adjusted tests, read the t values from `summary()`.

## Exercise 12.3: Model matrices and singularities

The exercise defines two factors and two numeric covariates. (The text has two typos: `g1` should be `gl`, and `x <-- 1:8` should be `x <- 1:8`.)

```r
set.seed(1)
a <- gl(2, 2, 8)
b <- gl(2, 4, 8)
x <- 1:8
y <- c(1:4, 8:5)
z <- rnorm(8)
table(a, b)
```

The factors are fully crossed and balanced, with two observations in each of the four `a` by `b` cells. The covariate `x` is just 1 to 8, and `y` equals `x` for the first four observations and runs backwards for the last four. The correlation between them is 0.76.

### Model matrix for z ~ a*b

```r
model.matrix(~ a*b)
```

```
  (Intercept) a2 b2 a2:b2
1           1  0  0     0
2           1  0  0     0
3           1  1  0     0
4           1  1  0     0
5           1  0  1     0
6           1  0  1     0
7           1  1  1     1
8           1  1  1     1
```

This has an intercept, one dummy variable for each main effect, and one interaction column. Four columns for four cells, so the matrix has full rank and `lm(z ~ a*b)` fits without trouble.

### Model matrix for z ~ a:b

```r
model.matrix(~ a:b)
```

```
  (Intercept) a1:b1 a2:b1 a1:b2 a2:b2
1           1     1     0     0     0
2           1     1     0     0     0
3           1     0     1     0     0
4           1     0     1     0     0
5           1     0     0     1     0
6           1     0     0     1     0
7           1     0     0     0     1
8           1     0     0     0     1
```

Here there are five columns. Because the main effects are absent, R codes the interaction with a dummy variable for every cell, and those four columns add up to the intercept column. The matrix therefore has five columns but only rank 4, and it is singular.

```r
summary(lm(z ~ a:b))
```

```
Coefficients: (1 not defined because of singularities)
            Estimate Std. Error t value Pr(>|t|)
(Intercept)   0.6129     0.7049   0.869    0.434
a1:b1        -0.8343     0.9969  -0.837    0.450
a2:b1        -0.2331     0.9969  -0.234    0.827
a1:b2        -0.8584     0.9969  -0.861    0.438
a2:b2             NA         NA      NA       NA
```

R notices the redundancy while computing the fit and sets the last coefficient, `a2:b2`, to `NA`. The remaining coefficients are measured relative to that dropped cell: the intercept (0.6129) is the mean of cell (2,2), and each other coefficient is a cell mean minus that value.

### Which models are singular?

I fitted a range of models. Only one of them is rank-deficient.

| Model | Columns | Rank | Singular? |
|---|---|---|---|
| `z ~ a*b` | 4 | 4 | no |
| `z ~ a + b` | 3 | 3 | no |
| `z ~ a + a:b`, `z ~ b + a:b` | 4 | 4 | no |
| **`z ~ a:b`** | **5** | **4** | **yes (`a2:b2` is NA)** |
| `z ~ a:b - 1` | 4 | 4 | no |
| `z ~ a*x`, `z ~ a + a:x` | 4 | 4 | no |
| `z ~ a:x`, `z ~ a:x - 1` | 3 and 2 | 3 and 2 | no |
| `z ~ a*y`, `z ~ b*x`, `z ~ b*y`, `z ~ x + y + b` | 4 | 4 | no |
| `z ~ x + y` | 3 | 3 | no |

The model `z ~ a*b*x` has 8 parameters for 8 observations. It is not singular, but it fits the data perfectly and leaves no residual degrees of freedom, so it is useless for inference.

### Implications

- **The fit is unaffected; the coefficients are not.** `z ~ a:b` and `z ~ a*b` have identical residuals, R-squared (0.2198) and F test (p = 0.776). They describe the same column space with different parameters. The coefficients in the singular model are only one of infinitely many equivalent solutions, which is why R picks one by dropping a column.
- **Degrees of freedom follow the rank, not the number of columns.** Both models use 4 parameters and have 4 residual degrees of freedom.
- **Singularity is a property of the design, not of the data.** The response `z` is random noise, yet the problem appears every time, because it comes entirely from the structure of `a` and `b`.
- **The fix is to reparametrize.** Dropping the intercept with `z ~ a:b - 1` gives four columns, one per cell, and the coefficients are exactly the four cell means (-0.221, 0.380, -0.245, 0.613). Using `a*b` or `a + a:b` gives the more usual main-effect-plus-interaction parameterization.
- **Watch for the warning sign.** The line "(1 not defined because of singularities)" in `summary()` output, or an `NA` coefficient, means the design has redundant columns. `alias()` can help show which columns are linear combinations of the others.
- **Correlated is not the same as singular.** `x` and `y` are correlated (r = 0.76), but no model involving them was singular, because a perfect linear dependence is needed.

## Takeaways

The additive model for the crossover data gives the same conclusions as the t tests while also separating period from treatment effects, and it handles the unbalanced groups more sensibly than a simple paired test. The model-matrix exercise shows that a rank-deficient design is a matter of parameterization rather than a failure of the fit: R still produces the right fitted values, but the individual coefficients are not uniquely defined unless the redundancy is removed.
