# Causal Inference

https://www.coursera.org/learn/crash-course-in-causality/

- module 1 - causal effects
    - potential outcomes and counterfactuals
        
        treatments - A
        
        potential outcomes - Y^a, a\in A - outcome that would be observed if the treatment was a
        
        observed outcome - factual outcome
        
        counterfactual outcome - what would happen if the treatment would not have been given
        
    - hypothetical interventions
        - cleanest to think about causal effects of interventions or actions - variables that can be manipulated
        - we assume there’s only one version of treatment - no hidden versions
        - we assume the treatments are not immutable (can’t change a person’s race or gender, but we might try to control the name/sex on a form) - so we’ll consider manipulable interventions
        - we assume treatments that can be randomized and manipulable
    - causal effects
        
        average causal effect - E[Y^1-Y^0] ~= E[Y|A=1]-E[Y|A=0]
        
        E[Y|A=1] is restricting to the subpopulation of people who actually had A=1 - which might differ from the actual population
        
        E[Y|A=1]-E[Y|A=0] - not causal effect - this compares 2 different populations of people
        
        E[Y^1-Y^0] - causal effect
        
        E[Y^1/Y^0] - causal relative risk
        
        E[Y^1-Y^0|A=1] - causal effect of treatment on the treated
        
    - causal assumptions
        - SUTVA (stable unit treatment value assumption)
            - no interference between treatments on different units
            - consistency - potential outcome is equal to observed outcome E[Y|A=a,X=x]=E[Y^a|A=a,X=x]
        - ignorability - assignment is independent on potential outcomes - treatment is randomly assigned among people with same value of pre-treatment covariates X (we can drop conditioning on A) E[Y^a|A=a,X=x]=E[Y^a|X=x]
        - positivity assumption - treatment assignment was not deterministic given X P(A=a|X=x)>0
        - E[Y|A=a,X=x]=E[Y^a|A=a,X=x]=E[Y^a|X=x]
    - stratification
    - incident user design - only new initiators
    - active comparator design - comparing 2 active interventions - less confounding
- module 2 - confounding and DAGs
    - confounders - variables that affect treatment and the outcome
    - we want to identify a set of variables X that will make the ignorability assumption hold

https://www.coursera.org/learn/ai-for-medical-treatment/

W1A1

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