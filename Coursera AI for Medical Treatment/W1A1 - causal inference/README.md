# Causal Inference
https://www.coursera.org/learn/ai-for-medical-treatment/ - W1

$p_{treatment}=\frac{n_{treatment}}{n}$

$p_{event} = \frac{n_{event}}{n}$

$p_{event, treatment} = \frac{n_{event, treatment}}{n_{treatment}}$

$p_{event, baseline} = \frac{n_{event, \sim treatment}}{n_{\sim treatment}}$

$odds_{event}=\frac{p_{event}}{1-p_{event}}$

$odds_{event,treatment} = \frac{p_{event,treatment}}{1-p_{event,treatment}}$

$odds_{event,baseline} = \frac{p_{event,baseline}}{1-p_{event,baseline}}$

$oddsRatio_{event}=\frac{odds_{event,treatment}}{odds_{event,baseline}}$

modelling treatment effect - probability of event:

given $x$ - features, and $y$ - binary indicating whether event happened,

we fit a logistic regression model to approximate the probability of an event:

$\hat{p}_{event}=\sigma(\theta^T x) = \frac{1}{1 + exp(-\theta^T x)}$

once fitted we can get it via

$\hat{p}_{event}=lr\_model.predict\_proba(x)['event']$

this is actually a linear model for the logit/log odds of an event:

$logit(p_{event}) \triangleq \log \left(\frac{p_{event}}{1-p_{event}} \right)=\log(odds_{event}) \approx \theta^T x = \theta_{treatment} \times x_{treatment} + \theta_{other} \times x_{other} + \cdots$

similarly:

$\log(odds_{event,treatment}) \approx \theta_{treatment} \times 1 + \theta_{other} \times x_{other} + \cdots$

$\log(odds_{event,baseline}) \approx \theta_{treatment} \times 0 + \theta_{other} \times x_{other} + \cdots$

therefore:

$oddsRatio_{event}\approx e^{\theta_{treatment}}$

ARR - absolute risk reduction - how much the treatment reduces the risk

$ARR = p_{event,baseline}-p_{event,treatment}$

but this assumes all patients are the same - in fact, a patient’s actual ARR varies depending on their baseline risk - the probability of an event had they not been given treatment.

using the model, we can estimate this by

$\hat{p}_{event,baseline}=lr\_model.predict\_proba(x_{treatment}=False,x_{other})['event']$

$\hat{p}_{event,treatment}=lr\_model.predict\_proba(x_{treatment}=True,x_{other})['event']$

$predicted\_ARR = \hat{p}_{event,baseline}-\hat{p}_{event,treatment}$

the c index (concordance index) evaluates the model's ability to predict overall patient risk (not how well the model predicts benefit from treatment).

we can also estimate for each patient their baseline risk, then group patients by the baseline risk (risk group), and then compute for each group the actual probability of event for those who have been treated and those who haven’t.

we can use the model to predict ARR given the baseline risk:

$\theta_{treatment} = lr\_model.coef\_['treatment']$

$oddsRatio_{event}\approx e^{\theta_{treatment}}$

$odds_{event,baseline} = \frac{p_{event,baseline}}{1-p_{event,baseline}}$

$odds_{event,treatment} = oddsRatio_{event}*odds_{event,baseline}$

$p_{event, treatment} = \frac{odds_{event,treatment}}{1+odds_{event,treatment}}$

$ARR = p_{event,baseline}-p_{event,treatment}$

and this too can be computed for different estimated baseline risks.

now we want to evaluate the model’s discriminative power for predicting ARR (concordance index):

$P(A \text{ has higher predicted ARR than } B| A \text{ experienced a greater risk reduction than } B)$but we never actually know a patient’s true ARR because we can only observe one outcome.

so instead we’ll find 2 pairs of similar patients, one of which received treatment and one hasn’t.

then we can compute the c-statistics-for-benefit:

$P(\text{$P_1$ has a predicted ARR greater than $P_2$} | \text{$P_1$ experiences greater risk reduction than $P_2$})$

to do this, given a list of pairs:

- we compute for each pair the observed benefit - difference between outcomes within pair (we’d like this to be -1 or 1, not 0)
- we compute for each pair the predicted benefit - average predicted ARR within pair
- we iterate over all distinct pairs of pairs, and count the number of times both observed and predicted benefits of one pair was higher than those of the second - this is the number of concordant distinct  pairs of pairs
- we divide this by the number of distinct pairs of pairs - this is the c statistic for the list of pairs

now we have to generate the list of pairs:

- sort treated patients and untreated patients by their predicted ARR
- match between lists
- if there are more in one list than second, subsample

we will evaluate the model on the test set - compute the c statistic on the test set. the meaning of this is how good the model is in predicting a patient’s risk reduction.

to improve the predictive ability, we’ll use a t learner approach - build 2 models, one for treatment risk and one for control risk.
