(group_fairness_metrics)=
# Group Fairness Metrics

```{warning}
This section is still under construction.
```

<!-- 
## Association-based Metrics
Predictions should be independent of sensitive group membership.

### Demographic parity 

### Conditional Demographic Parity
In EU and US law, indirect discrimination in employment may not be unlawful if it is justified by a so-called "legitimate aim". Examples of legal justifications for discrimination are genuine occupational requirement and business necessity. For example, a film producer is allowed to hire only male actors to play a male role, as this is considered a genuine occupational requirement. Apart from employment, there may be characteristics that, either from a legal or ethical perspective, legitimize differences between groups. In particular, EU discrimination law is highly contextual - requiring normative or political evaluations on a case-by-case basis {footcite:p}`wachter2021fairness`. Loosely inspired by these legal imperatives, {footcite:t}`Kamiran2013` put forward a notion of fairness that we will refer to as {term}`conditional group fairness`. This is a variant of group fairness that allows for differences between groups, if these differences are explained by a legitimate feature that can be justified by ethics and/or law.  put forward a similar definition inspired by European Union law.

Conditional group fairness is best illustrated by an example. Imagine a scenario in which women have a lower income, on average, than men. This may imply that women are discriminated against. However, in our scenario many women work fewer hours than men. The observed disparity can therefore be at least partly explained by the lower number of working hours. Consequently, equalizing income between men and women would mean that women are paid more per hour than men. If we believe unequal hourly wages to be unfair, we can instead equalize income only between women and men who work similar hours. In other words, we minimize the difference that is still present after conditioning on working hours. Conditional group fairness is particularly relevant considering Simpson's paradox. This paradox states that if a correlation occurs in several different groups, it may disappear or even reverse when the groups are aggregated.

```{admonition} *Example:* Simpson's Paradox: Berkeley University Admissions
When considering all programs together, women were accepted less often than men, implying a gender bias. However, it turned out that women at Berkeley often apply for competitive programs with a relatively low acceptance rate. As a result, the overall acceptance rate of women in the aggregated data was lower -- even though the acceptance rate of women *within* each program was higher than the acceptance rate of men. Hence, if the admission's office would have tried to equalize the overall acceptance rate between men and women, men would have received an even lower acceptance rate.
```

## Error-based Metrics

### Equalized Odds

### Equal Opportunity

## Calibration

### Equal calibration 
-->

## References
```{footbibliography}
```