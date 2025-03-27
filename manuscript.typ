#set par.line(numbering: "1")
#set page(numbering: "1 of 1")
#set text(font: "STIX Two Text")
#show math.equation: set text(font: "STIX Two Math")
#set math.equation(numbering: "(1)")
#show table.cell.where(y: 0): strong

*Abstract*: Providing accurate estimates of uncertainty is key for the analysis, adoption, and interpretation of species distribution models. In this manuscript, through the analysis of data from an emblematic North American cryptid, I illustrate how Conformal Prediction allows fast and informative uncertainty quantification. I discuss how the conformal predictions can be used to gain more knowledge about the importance of variables in driving presences and absences, and how they help assess the importance of climatic novelty when projecting the models under future climate change scenarios.

= Introduction

The ability to predict where species may be found is a cornerstone of biogeography and macroecology #cite(<Elith2019>). Techniques from the field of applied machine learning (ML hereafter) are now routinely used alongside ecological approaches to train generalizable species distribution models (SDMs hereafter) #cite(<Beery2021>). SDMs generate a binary response (corresponding to the prediction that the species is likely present/absent under given environmental conditions) or a quantitative score most often as a probability of presence or habitat suitability, indicating how strongly we believe that the species may be present at the location.

Proper communication of the uncertainty associated to the prediction of a SDM is important, since we usually seek to apply these models to look both forward and backwards in time #cite(<Franklin2023>). This projection if the model to different times is usually called "transfer" #cite(<Zurell2012>), whereby a model trained under historical (baseline) conditions is applied to past/future projections of the same predictors. The projection of SDMs can also happen in space #cite(<Petitpierre2016>), to predict where species may invade or be naturalized. Even when predictions are not projected, spatial knowledge of the uncertainty is valuable information: it can be used to identify areas where the model predictions are trustworthy. Current checklists on the reproductibility of SDMs emphasize the consequences of data uncertainty #cite(<Feng2019>). Yet, predictions also have inherent uncertainty, which is usually not adequately communicated. This can be, for example, because of genuine uncertainty about (or inability to capture through the model) the actual response of the species to combination of predictors #cite(<Parker2024>).

A common way to capture information about the variability of SDMs is to rely on non-parametric bootstrapping #cite(<Valavi2021>), wherein models trained on random subsets of the data are compared to estimate the distribution of the response under incomplete sampling. This approach captures more than one type of variability #cite(<Thuiller2019>), and provide valuable information about the range of performances that can be expected from a model. Other methods are built into the predictor itself, as is the case for _e.g._ BARTs #cite(<Carlson2020>), which estimate their own uncertainty. But either situation comes with drawbacks. Bootstrapping requires to train and evaluate the model hundreds of times, and on partial datasets, which is computationally inefficient. Using built-in methods limits one to the classifier for which these methods are available, which prevents for example the use of a new algorithm with the same estimation of uncertainty.

In this manuscript, I illustrate how the ML technique of conformal prediction (CP) allows to identify instances (combinations of environmental variables) for which a trained and calibrated model cannot confidently make predictions #cite(<Gammerman1998>). A brief introduction to CP is provided in this manuscript, but the topic is covered in more depth by #cite(<Shafer2007a>, form: "prose") for the mathematical foundations, by #cite(<Fontana2020>, form: "prose") for a historical perspective, and by #cite(<Angelopoulos2023a>, form: "prose") for concrete recommendations. By way of contrast to _e.g._ bootstrapping, CP does not necessarily involve retraining the same model many times over, but instead wraps the model into an additional prediction step, and returns estimates of credibility based on the distribution of past model predictions compared to ground-truthed data. This is an important difference, as the variability measured through conformal prediction is inherent to the model, and is not a measure of variability coming through the distribution of data #cite(<Lei2013>). Conformal prediction provides what is essentially (for classification problems) a confidence interval around the presence or absence of a species in a given location. This is a particularly important feature, in that CP achieves this in a way that creates several analogues between ML prediction and fundamental concepts in frequentist statistics #cite(<Neyman1937>).

One of the reasons why CP is particularly promising for uncertainty quantification in SDMs is that it is a distribution-free method: it requires neither assumptions about the model nor prior knowledge of the outcome distribution to provide confidence intervals that are as small as possible while being _guaranteed_ to contain the true value under a set risk level #cite(<Vovk2018>). This is particularly important when transferring a SDM to novel environments #cite(<Zurell2012>), where we expect covariate shift (the joint distributions of predictors are different when training and predicting), a prediction context that CP is robust to #cite(<Fannjiang2022>) #cite(<Tibshirani2019>).

Using occurrence data about an emblematic North American cryptid, I provide a template for the adoption of CP as a natural way to quantify uncertainty of species distribution models. In particular, I show how predictions under CP (i) identify areas where the species range is uncertain, (ii) estimate uncertainty differently from bootstraping methods, (iii) can be explained using Shapley values analysis, and (iv) quantify the accumulated uncertainty when transferring the SDM to future conditions. I conclude by highlighting ways in which using CP can both simplify the process of training SDMs, and provide information that make their discussion and analysis more informative.

= Methods

== Data

=== Occurrence data

The occurrence data used in this article are geo-referenced observations of the Sasquatch #cite(<Lozier2009>). Although these observations are likely to be mis-categorized American black bears #cite(<Foxon2024>), they nevertheless share many features of the data that are used to train SDMs: high auto-correlation, uneven sampling effort, and clear association with several bioclimatic variables that is robust enough to train a predictive model. The recorded locations, as well a background points, are presented in #ref(<occurrences>).

=== Pseudo-absences generation

The dataset of observations is composed only of presences. In order to establish a baseline of absences to train a binary classifier, there is a need to generate a number of pseudo-absences, which simulates locations at which the species, if not absent, has not been observed. In order to do so, the presence data were first spatially thinned to be limited to one for each cell, at a 5.0 minutes of arc resolution. Cells that had no observation were potential candidates for a pseudo-absence, and were further selected by drawing a number of them, without replacement, where the probability of inclusion in the sample was proportional to $h _ "min" ^(-1)$, where $h _ "min"$ is the Haversine (great arc) distance to the nearest cell with an observation, measured in kilometers. In other words, cells that were close to an observation were unlikely to be included, and cells that were further away were more likely to be so. To avoid sampling pseudo-absences too close to presences, the pixels less than 10 kilometers away from known observations were excluded from the background data.

The number of pseudo-absences was arbitrarily set to two times the number of presences. Although #cite(<Barbet-Massin2012>, form: "prose") recommend to use the same number of presences and pseudo-absences for classifiers, using an imbalanced dataset is not a problem: stratified k-folds cross-validation is perfectly able to handle the moderate class imbalance we introduce #cite(<Szeghalmy2023>), and the model performance (as will be established in a later section) is sufficient. Moreover, most real-world applications of classification will have to deal with problems with class imbalance (this is particularly likely to be true of SDM application from sampling data, where presences may be the minority of outcomes); it is therefore important to ensure that we do not establish a testing scenario that is too optimistic about the prevalence of presences. In all cases, class imbalances is a feature of data that must be dealt with in order to get the more predictive models #cite(<Benkendorf2023>).

#figure(
  image("figures/occurrences.png", width: 100%),
  caption: [Overview of the occurrence data (green circles) and the pseudo-absences (grey points) for the states of, clockwise from the bottom, California, Oregon, Washington, Idaho, and Nevada (A). The underlying predictor data are at a resolution of 2.5 minutes of arc, and represented in the World Geodetic System 1984 CRS (EPSG 4326). The panels on the right column show the ROC curve (B) and PR curve (C), with the random classifier indicated by a dotted line. The area under the ROC curve is $approx 96%$.],
  placement: auto
) <occurrences>

=== Bioclimatic data

The model was trained, validated, and applied on the 19 WorldClim2 BIOCLIM variables #cite(<Fick2017>), at a spatial resolution of 2.5 minutes of arc. Preliminary analyses using 0.5, 2.5, 5, and 10 minutes of arc show that the qualitative results presented hold. For the projection of the model under climate change, I only report the future data under the SSP370 scenario ("business as usual"), for the MRI ESM2-0 GCM, over the period 2081-2100.

== Species distribution model

All analyses are conducted using the `SpeciesDistributionToolkit` package #cite(<Poisot2025>) for _Julia_ 1.11.

=== Model structure

The model used here is a logistic regression, with interactions terms up to a maximum degree of two (preliminary analyses with random forests, naive Bayes classifiers, and rotation forests gave similar results). When trained on a vector of features $bold(x)_i$ (with null means and unit variances), the model will return a probability $p_+$, which correspond to the probability of these environmental conditions being associated to the presenceof the species. This probability is turned into a presence/absence decision by comparing it to a threshold, as explained in a later section. Because this logistic regression is a deterministic classifier, the prediction $p_i+$ statisfies $0 <= p_i+ <= 1$, and we use $p_- = 1 - p_+$ as the probability that the species is absent from the location.

=== Tuning

We tune this model by (i) iteratively forward selecting the best set of predictor variables, and (ii) optimizing the threshold $tau$ above which a site with a probability for the positive class $p_+$ is considered to be positive (turning the prediction of presence into $p_+ >= tau$). In both cases, the cross-validation strategy is the same: the dataset is split in 10 random folds, 9 of which are used for training and one for validation. All folds are used for evaluation, providing exhaustive cross-validation. The folds are stratified so that the relative number of present cases in the training set is similar to that of the entire dataset. The performance on each set, for the purpose of defining the set of variables to include of the threshold to use, is measured as the average of the Matthews Correlation Coefficient (MCC) across each of the ten folds. The MCC is the most accurate representation of a binary classifier performance #cite(<Chicco2023>), and avoids the pitfalls of several other validation measures.

For all steps of model training and validation, the identity of instances composing the different folds remains fixed. This ensure that the changes in MCC are only due to the addition of the variable, and not to the random sampling of a training/validation set with different properties. Although some authors encourage the use of spatially-stratified cross-validation #cite(<Soley-Guardia2024>), this is not a desirable strategy for this use-case. The area in which the predictions will be made is entirely delimited by the bounding box of observed presences, and there is therefore no risk of covariate shift when shifting from validation to prediction (outside of the situation of temporal transfer of the SDM).

The predictors included in the model have been decided through the use of forward selection. This is an important step in order to perform dimensionality reduction (which generally increases the predictive accuracy), but also to ensure that the set of retained variables is reduced enough that it can be interpreted. Variables were retained as part of the final set of predictors if adding them increased the MCC for the model once retrained with this new variable.

One of the most efficient ways to increase the performance of binary classifiers is to change the decision rule leading to a positive (here, presence) prediction, so that presences are assigned when $p_+ >= tau$ – a process known as moving threshold classification #cite(<Liu2016>) #cite(<Liu2013>). The value of $tau$ is an hyper-parameter of the model, which is chosen to maximize the value of a measure of model performance (here the MCC) when evaluated over many different values. In this instance, we optimized the value of $tau$ by picking the value out of 200 linearly spaced value between the smallest and largest prediction made on the training set. The value of $tau$ that maximizes the MCC during cross-validation was selected as the optimal threshold for the classifier. Note that even though our decision rule for the presence of the species is $p_+ >= tau$, we will keep the information about $p_-$ as is it required for conformal prediction.

=== Bootstrap variability

Bagging (bootstrap aggregating) is often used as a measure of uncertainty to the underlying data when training SDMs #cite(<Beale2012>). When performing bagging, the model is trained on samples drawn with replacement from the training set (which leaves out approx. 37% of the dataset). Models are then evaluated on samples that were not used as part of their training, usually using cross-validation #cite(<Bylander2002>) or measures of the out-of-bag error #cite(<Janitza2018>). Although ensemble models _can_ result in a better predictive performance compared to single models #cite(<Drake2014>), this is not a guarantee (and depends on the structure of the bias/variance trade-off for the specific model and its training set). The many models trained on the bagging dataset form an homogeneous ensemble, which is to say a set of models that share the same algorithm and hyper-parameters, and only make different predictions as the result of having been trained on different subsets of the full training set.

Measures of whether the different models composing the homogeneous ensemble agree can provide a measure of the effect of data and parameter uncertainty #cite(<Petropoulos2018>), or what #cite(<Davies2023>, form: "prose") termed the "SDM uncertainty". The best model identified after thresholding was evaluated on a hundred bootstrap samples, yielding an homogeneous ensemble model from which we estimate bootstrap variability #cite(<Chen2019>). Because the model is kept constant in this analysis, the measure of variability we will derive from the ensemble model is an estimate of how sensitive the estimation of the model parameters is to small perturbations (specifically: spatially homogeneous under-sampling) to the training data.

= An introduction to conformal prediction

Conformal prediction differs from regular prediction in that, rather than a single point prediction, it returns sets corresponding to the ensemble of _credible_ outcomes given an input $bold(x)$ representing environmental conditions at which we seek to make the prediction. Given the observed quantiles of the model output on validation data, these sets are obtained through a simple calibration step. Therefore, CP requires an already trained model, and is agnostic to the process through which this model is trained. In this section, I highlight two important features of CP: the notion of _credible sets_ (and how they are obtained), and the notion of _coverage_ , which is a measure of tolerance to error.

== Understanding conformal predictions

By contrast to the non-conformal SDM, the conformal classifier returns, for an input of environmental predictors $bold(x)$, a set $C$ containing the "credible outcomes" for this prediction. This set is termed the _credible set_, and under a binary classification task (the species is either present or absent), there are four possible combinations for the content of credible sets: $C = {+}$, $C = {-}$, $C = {+, -}$, and $C = emptyset$.

The first two cases are simple: if the credible set contains a single output, the model can confidently make a prediction that excludes the other class. In the case of $C = {+}$, for example, the point prediction for the presence score $p_+$ is high enough that the outcome of absence can be ruled out given the known predictions on training examples. In some cases, the credible set may contain both classes, as in $C = {+, -}$. Although they may not be _equally likely_ (there is no guarantee that $p_+ approx p_-$), the scores are close enough to not confidently exclude one of the outcomes from the model prediction. In the specific cases of SDMs, these correspond to areas of true uncertainty, where the known training examples credibly support both the presence or absence of the species. The final situation, $C = emptyset$, corresponds to pathological cases where _neither_ outcome can be credibly supported. Given the training data (and the distribution of presences and absences), the model is not able to make a prediction for this input. The increased frequency of such predictions is most likely a strong sign that the risk level is too high (the confidence interval is too broad) for the training data given to the conformal model.

These situations correspond to four different outcomes in terms of the SDM certainty about the distribution of the species. The most intuitive situation is $C = {+}$ or $C = {-}$, in which case the conformal model predicts that the absence (resp. presence) of the species is _not_ a credible outcome for the environmental conditions given as an input. Throghout this manuscript, I will refer to these predictions as "sure presences" and "sure absences", as they convey the information that there is no reason to expect that the prediction is uncertain. The second situation, $C = {+, -}$, corresponds to inputs for which the presence and the absence of the species are credible, and I will refer to them as "unsure". The rare cases where $C = emptyset$ will be "undetermined" predictions.

== Obtaining conformal predictions

There are several ways to decide whether a point prediction from the model results in which credible set. A core assumption of CP is that the data used for training should be exchangeable, or in other words, their joint probability distribution should be (close to) invariant under finite permutations #cite(<Aldous1985>). This will almost never be the case for data with a spatial structure; nevertheless, this does not rule out the use of CP for species distribution modeling, as #cite(<Oliveira2024>, form: "prose") show that CP is acceptably robust to lack of exchangeability.

The central idea of CP is to associate a conformal score to a point prediction. This can be achieved by applying the $text("softmax")$ function to the values for $p_+$ and $p_-$, giving

$
s_+ = (exp p_+)/(exp p_+ + exp (1- p_+)), s_- = (exp (1 - p_+))/(exp p_+ + exp (1- p_+))
$ <equation-softmax>

The conformal score associated to a prediction is $1 - s_dot$, where $dot$ is the prediction ($+$ or $-$) made by the model. We call the distribution of conformal scores $cal(S)$. Note that this can be done without using the $text("softmax")$ function, but it is included here as it is best practice for classification.

The next step is to identify a critical value $accent(q, hat)$ above which a conformal score indicates that the prediction it describes is credible. This critical value is picked by examining the empirical quantile distribution of the conformal scores calculated over $n$ training examples, and an acceptable level of risk $alpha$ (explained in depth in the next sub-section), and specifically by identifying the $q_i$-th quantile, where 

$
  q_i = ceil((n+1)(1-alpha))/n 
$ <equation-quantile>

The corresponding value of $S$ below which a proportion $q_i$ of values lies is $accent(q, hat)$. In other, more intuitive words, the value $q_i$ indicates what proportion of wrong classification events we must accept before we have accumulated enough evidence to be confident about a prediction. When performing the prediction, we calculate the score of a new prediction according to #ref(<equation-softmax>). For every possible class $x$, if $s_x >= (1 - accent(q, hat))$, this class is retained as part of the credible set.

The value of $accent(q, hat)$ can be obtained either through using a holdout set for training (Split Conformal Prediction), by retraining the model in a way aking to Leave-One-Out cross-validation (Full Conformal Prediction), through the use of quantile regression #cite(<Romano2019>), or through taking the median of several estimates of $accent(q, hat)$ after cross-validation #cite(<Vovk2018>). In this manuscript, I employ the later method, as it provides a rapid and statistically acceptable estimate of $accent(q, hat)$, without requiring too much computing time.

To summarize, the output of the conformal classifier is, in a sense, a point estimate of the credible outcomes of a model, using the value estimated for $p_+$ as well as knowledge about which of these were associated to the correct label in the training data. A location is defined as included in the range is the positive outcome is included within the credible set returned by the conformal classifier, and as excluded from the range when it is not. Because the conformal classifier can identify that both outcomes are credible based on the training data (while giving them different weights), predictions in which both the positive and negative outcomes are included in the credible set can be seen as "uncertain" at this given risk level.

How frequently a specific prediction is uncertain is termed the inefficiency of the classifier, which is defined as the average cardinality of all credible sets. The inefficiency is bounded upwards by the number of classes (two for binary classification); when the inefficiency is $approx 1$, the conformal classifier behaves (essentially) like deterministic classifier, by returning a single class for each instance. An inefficiency close to unity is not desirable: smaller sets can hide our actual uncertainty #cite(<Sadinle2018>). Because the conformal models wraps the logisitc regression model, we can further divide the "unsure" predictions as a function of whether they would be within the range as predicted by the SDM, which I will call "unsure presences"; the other unsure predictions are referred to as "unsure absences".

== The coverage level

CP allows users to set a desired error rate, $alpha$, which appeared in #ref(<equation-quantile>). Intuitively, what CP does, is inform the user on whether the credible set contains the true value with probability $1-alpha$, which allows to directly interpret this value as a true confidence interval. This error rate is usually referred to as the _marginal coverage_, in that it captures the probability of success marginalized over the known validation points. Because the estimate of uncertainty involves the original model, it is important to apply CP on a model with adequate performance.

Chaning the risk level $alpha$ leads to different estimates of how commonly multiple classes will be accepted as a credible outcome. Using a low level of risk ($alpha approx 0$) yields usually leads to all outcomes being credible ($accent(q, hat) approx 1$), at the cost of a very high uncertainty. When values of $alpha$ get too large ($accent(q, hat) approx 0$), no class can be confidently predicted, and the model will eventually always return $C = emptyset$. Although this later situation is more difficult to make sense of intuitively, a value of inefficiency that gets smaller than unity should be interpreted as a model that accumulates more uncertainty (at a given risk level) than the data can support #cite(<Romano2020>). Conformal prediction can therefore inform us on the acceptable risk levels we can operate under given a trained predictive model.

In the rest of this analysis, I will set $alpha = 0.05$. As noted by #cite(<Angelopoulos2023a>, form: "prose"), this corresponds to estimating whether a specific prediction falls within, or outside of, the 95% confidence interval across all predictions, which is a convenient callback to frequentist statistics' usual risk tolerance. Recall that the CP credible sets are estimated based on the model output, and therefore even when aiming for full coverage, there may be non-ambiguous combinations of environmental predictors.

= Results

== Performance of the baseline model

#figure(
table(
  columns: 4,
  [Measure], [Validation], [Training], [Ensemble],
  [MCC], [0.75], [0.76], [0.76],
  [NPV], [0.93], [0.93], [0.94],
  [PPV], [0.82], [0.83], [0.82],
  [$kappa$], [0.75], [0.76], [0.76],
  [TSS], [0.74], [0.75], [0.76],
  [Accuracy], [0.91], [0.91], [0.91],
),
caption: [Overview of measures of model performance for the validation and training sets of the SDM, as well as the same measures for the ensemble model (measured on the out-of-bag models only). The values of $kappa$ and the true-skill statistic are generally comparable to the MCC, but are included as they are commonly reported in the SDM litterature #cite(<Allouche2006>). The high values of the negative and positive predictive values indicate that the model is suitable to detect both presences and absences.],
placement: auto,
) <table-performance>

In panels B and C of #ref(<occurrences>), we report the ROC and PR curves for the model. As evidenced by both these diagnostic tools, the model achieves a very high predictive accuracy. In #ref(<table-performance>), we report additional measures of performance for the training and validation set of the model (so as to ensure that the model is not performing better on training data), as well as a measure of the performance of the ensemble, to show that it can make valid predictions in addition to quantifying variability. These results confirm that the model is able to identify areas that are suitable to the species, and can be used for CP.

Before applying CP, it is useful to examine the output of the SDM in space. The predictions of the model for the entire region are given in #ref(<predictions>), alongside information about the model variability. Areas of lowest variability (according to the IQR based on non-parametric boostrap results from the ensemble) seem to be associated with the absence of the species, with the variability mostly increasing within the predicted range.

#figure(
  image("figures/prediction.png", width: 100%),
  caption: [Overview of the probability $p_+$ returned by the model (A), and the inter-quantile range of the non-parameteric bootstrap model predictions (B). The range, _i.e._ the limit of cells for which $p_+ >= tau$, is indicated by a solid red line; I maintain this convention for all subsequent figures. Note that the scale of the variability is logarithmic, as the model shows good performance and therefore has low variability overall.],
  placement: auto
) <predictions>

== Conformal prediction of the species range

Before discussing the spatial output of running the conformal model, it is worth considering why the thresholding step as visualized in #ref(<predictions>) is not really providing us with a set of certain presences and absences. When optimizing the threshold $tau$ above which a prediction $p_+$ from the non-conformal model is determined to be a presence, we inherently establish a sort of certain presences and certain absences, specifically by ignoring the possibility that there can be uncertain predictions. Indeed, the space covered by positive predictions is usually interpreted as the (potential) distribution of the species. But this prediction conveys a false sense of certainty, that has to do with the very nature of the threshold we optimize. By definition, the threshold is the value that finds the best balance between the false/true positive/negative cases on the validation data; this is in fact why the optimal threshold is the point closest to the corners of the ROC and PR curves indicating a perfect classifier #cite(<Balayla2020>). When a prediction $p_+$ gets closer to the threshold, a small perturbation to the environmental conditions locally could bring it on the other side of the threshold, and therefore flip the predicted class using the non-conformal classifier. Around the threshold is where we expect uncertainty to be the greatest.

To bring these considerations into a spatial context: we expect the areas where the score for the present class are closer to the threshold (the limits of the predicted range of the species) to be the most uncertain. Importantly, this is true _both_ for areas that are inside the range (for which $p_+$ is just above the threshold) and for areas that are outside of it (for which $p_+$ is just below the threshold). CP is perfectly suited to solving this issue, by identifying the areas where one class is predicted, but the other class is also credible. In this section, we will project the areas with uncertain predictions, and compare the uncertainty quantified by the conformal model to the uncertainty derived from the ensemble model.

#figure(
  image("figures/uncertainty.png", width: 100%),
  caption: [Overview of areas where the presence of the species is certain according to the CP model under a risk level $alpha = 0.05$ (A). The certain areas are in dark green, and the uncertain areas, wherein both presence and absence are credible, are in dark grey. (B) Surface covered by the sure absence and total range (including the superficy of the unsure area) for different risk levels. Note that for $alpha approx 0.1$, the total predicted range starts being lower than the range predicted by the SDM, and the uncertain range collapses. (C) Distribution of variability from #ref(<predictions>)B by type of CP model outcome.],
  placement: auto
) <uncertainty>

In #ref(<uncertainty>), we show that this prediction indeed stands: the range as predicted by the SDM (#ref(<uncertainty>, supplement: "fig.")A) falls within the range of unsure predictions. We also see that lowering the risk level $alpha$ leads to a contraction of the area (in $"km"^2$) considered to be credibly associated to only the presence of the species ($C = {+}$), while the range that is ambiguous ($C = {+, -}$) increases (#ref(<uncertainty>)B). As far as ecologists are concerned, the areas in which the credible set only has a score for the absence of the species are the easiest to make sense of: they correspond to regions where the model is certain (under the specified risk level) that the species is absent. All other areas (assuming that there are no predictions for which the credible set is empty, which I discuss in the next section) are _potentially_ part of the range of the species: some certainly, some uncertainly. Depending on the purpose for which the SDM is produced, the uncertain areas can be treated differently. As #cite(<Prescott2025>, form: "prose") argue, when dealing with invasive species, it may be more reasonable to interpret SDMs by erring on the side of caution, which here would mean considering that unsure presence area should be considered part of the species's range. On the other hand, when SDMs are meant to guide conservation actions that are costly or should be focused on areas of high certainty of suitability for the target species #cite(<Peknicova2016>), it may make sense to ignore the unsure presences.

== Relationship between variability and uncertainty

Note that the relationship between the certainty associated to CP, and the variability under the ensemble model presented in #ref(<predictions>)B is nuanced: in #ref(<uncertainty>, supplement: "fig.")C, it appears that although areas identified as unsure using CP tend to have higher variability, there is considerable overlap between the categories. Intriguingly, the overlap between areas that are uncertain according to the conformal classifier, and areas that are uncertain according to the bootstrap model, is imperfect. There are a number of points classified as sure presences for which the IQR is very high, *i.e.* points whose certainty is not affected by undersampling the training data. Notably, the results in #ref(<uncertainty>, supplement: "fig.")C show that it is not possible to find a cutoff in the measure of bootstrap variability that would identify areas of model uncertainty. This suggests that the classification of predictions as certain/uncertain according to the conformal prediction is in part reflecting genuine uncertainty in the underlying data, but also contributing novel information about the fact that some instances are more difficult to call.

These results can be better understood by contrasting what "uncertain" means in the context of CP, and how it differs from the uncertainty in the ensemble model. The uncertainty derived from the ensemble model represents whether many models trained on small perturbations of the full training dataset would agree on a specific prediction task, represented by an array of environmental predictors. Therefore, the uncertainty from the ensemble originates in the estimation of the parameters, and its sensitivity to being able to access the full information within the training data. Uncertainty in the conformal classifier is coming from comparing the prediction to all other predictions under an estimation of the distributions for the conditions leading to the prediction of the presence (or absence) outcome. Therefore, the uncertainty from the conformal predictors accounts for all the predictions the model can make, and accounts for the variability _across_ predictions within a fully known dataset.

== Identification of undetermined areas

In #ref(<uncertainty>)B, we see that there is a risk level above which the total predicted range starts to get lower than the range predicted bu the SDM. We can explain this behavior through the lens of the number of undetermined predictions, _i.e._ the number of inputs for which the CP model returns $C = emptyset$.

In #ref(<undetrange>, supplement: "fig.")A, we see that above $alpha approx 0.1$, the inefficiency of the classifier starts to fall under 1 - this indicates that _on average_, the model is returning fewer than one output for each prediction. In a sense, this creates an upper limit to the risk we can accept: the model trained on this dataset does not support conformal prediction for larger risk levels. In #ref(<undetrange>, supplement: "fig.")B, we see that this change of behavior in the model is indeed resulting in an increase in the range for which the model makes no prediction, which gets larger when the risk level is too high. The spatial distribution of undetermined areas is shows in #ref(<undetrange>, supplement: "fig.")C for $alpha = 0.2$: these areas are concentrated around the range limit as identified by the SDM. This suggests that using a risk level that it too high would result to no conformal predictions being made for the areas where our need to accurately quantify uncertainty are the most important, and calls for a cautious investigation of the appropriate risk level.

#figure(
  image("figures/undetrange.png", width: 100%),
  caption: [Overview of the occurrence data (green circles) and the pseudo-absences (grey points) for the states of, clockwise from the bottom, California, Oregon, Washington, Idaho, and Nevada. The underlying predictor data are at a resolution of 2.5 minutes of arc, and represented in the World Geodetic System 1984 CRS (EPSG 4326).],
  placement: auto
) <undetrange>

== Model explanation

In this section, I perform an analysis of Shapley values of the conformal predictor, in order to (i) assess the importance of variables and (ii) provide explainable results about the relationships between predictors and response. I rely on the common Monte-Carlo approximation of Shapley values #cite(<Roth1988>) #cite(<Touati2021>). Monte-Carlo Shapley values represent, for each prediction, how much the $i$th variable contributed to moving the prediction away from the average prediction. The Shapley value associated to variable $i$ is $phi_i in [-1,1]$, which measures how much this variable modified the _average_ prediction for this class. Shapley values have a number of desirable properties regarding the explanation of prediction of responses for environmental studies #cite(<Wadoux2023>), including their additivity: for any given prediction, $p = accent(p, hat) + sum_i^"variables" phi_i$. Because of this additive property, the importance of variables across many predictions is usually measured as the average of $| phi |$, where both positive (the class is more likely) and negative (the class is less likely) are counted. This measure of variable importance represents the relative impact that each variable had on the process of moving all predictions away from the average prediction and towards its actual value. Because Shapley values are both additive and independent, they can be measured and aggregated for any arbitrary stratification of the data (which allows reporting them conditional on the uncertainty status of the prediction).

As the predictions of the conformal model can be split by whether they are certain or uncertain, they offer a unique opportunity to delve into the mechanisms that _generate_ this uncertainty. Namely, if the relative importance of variables is different across these classes of predictions, this is strongly suggestive of the fact that there are certain environmental conditions (represented by combination of values for each variables) that create or reduce uncertainty. Furthermore, because we can split the certain predictions into a presence and absence class, this is a unique opportunity to investigate whether the factors leading to a species being present or absent are the same. An example of the spatial contribution of a variable is given in #ref(<shapley>)A.

#figure(
  image("figures/shapley.png", width: 100%),
  caption: [Overview of the occurrence data (green circles) and the pseudo-absences (grey points) for the states of, clockwise from the bottom, California, Oregon, Washington, Idaho, and Nevada. The underlying predictor data are at a resolution of 2.5 minutes of arc, and represented in the World Geodetic System 1984 CRS (EPSG 4326).],
  placement: auto
) <shapley>

We find that, for the most important variable (_i.e._ the one with the largest $sum |phi|$), the contribution of this variable tracks the status of the prediction: it tends to be negative when the absence is certain, positive when the presence is certain, and around zero when the prediction is unsure (#ref(<shapley>, supplement: "fig.")B). This is a fairly remarkable result, in that it ties Shapley values (a tool to help with ML models interpretation) to CP (a technique to accurately convey uncertainty). In #ref(<shapley>)C, I present the relative contribution of all selected variables split by the status of the prediction; this reveals that the Shapley values for sure presences and unsure areas are distributed in different ways. Notably, BIO15 is far more important in areas of high model uncertainty than in areas of either sure presences or absences. This suggests that the division of the prediction according to CP status can provide information about which sets of environmental conditions are driving the uncertainty, thereby providing useful information to guide future sampling or model interpretation.

#figure(
  image("figures/gainloss.png", width: 100%),
  caption: [Overview of the occurrence data (green circles) and the pseudo-absences (grey points) for the states of, clockwise from the bottom, California, Oregon, Washington, Idaho, and Nevada. The underlying predictor data are at a resolution of 2.5 minutes of arc, and represented in the World Geodetic System 1984 CRS (EPSG 4326).],
  placement: auto
) <gainloss>

== Model projection

#cite(<Zurell2012>) highlight the importance of uncertainty when transferring the model to novel climate data: there is a chance that the future climate condition will not have occurred in the training dataset, and therefore our confidence in the model outcome may be lowered. This covariate shift is well documented to decrease the performance of models #cite(<Mesgaran2014>), and CP offers an opportunity to quantify this phenomenon.

Using the data from the CanESM5 model #cite(<Swart2019>) under the SSP370 scenario for the year 2090, it is possible to split the landscape as a function of (i) climatic novelty defined as values of the bioclimatic variables not observed in the training data and (ii) status of the range for the species. These results are presented in the table below:

| Climatic novelty | Sure absence | Unsure | Sure presence |
|------------------|--------------|--------|---------------|
| Yes              | 50.46%       | 48.36% | 1.16%         |
| No               | 54.54%       | 37.27% | 8.17%         |
| *(difference)*   | 4.07%        | 11.09% | 7.01%         |

These results show that *on average*, the areas with climatic novelty had more uncertain outcomes, which is in line with ecological expectations.

#figure(
  image("figures/novelty.png", width: 100%),
  caption: [Overview of the occurrence data (green circles) and the pseudo-absences (grey points) for the states of, clockwise from the bottom, California, Oregon, Washington, Idaho, and Nevada. The underlying predictor data are at a resolution of 2.5 minutes of arc, and represented in the World Geodetic System 1984 CRS (EPSG 4326).],
  placement: auto
) <novelty>

= Conclusion

Conformal prediction, like most SDM methods, is not quite delivering a true estimate of the probability of presence #cite(<Phillips2013>). Nevertheless, it brings valuable information, in the form of a quantified measure of whether a prediction comes with uncertainty (are both presence and absence in the credible set?) in a way that is directly comparable with the non-conformal prediction. "Class overlap", where both presences and absences are observed under the same values of the predictions, decreases the predictive performance of models [#cite(<Valavi2021>, form: "prose")a] – CP is naturally suited at handling this, by assigning the area where overlap occurs to uncertain predictions.

Transparent communication of uncertainty, meaning, it is both spatially explicit, quantified, and expressed under a risk set by the user, is important: we do not expect a fully trained model to always be certain, as some areas are genuinely more difficult to predict. For example, small organisms are more inherently stochastic #cite(<Soininen2013>); any form of stochastic event will drive species distribution in the general case #cite(<Mohd2016>); these stochastic events can appear even in areas that are close to the species' environmental optimum #cite(<Dallas2020>).

CP contributes to dispel what #cite(<Messeri2024>, form: "prose") called the "illusion of understanding", which is often associated with ML models: it generates an understanding of the uncertainty from observations of a pre-trained model, and expresses this uncertainty both in absolute (is the "presence" event in the credible set?) and relative (is the conformal score for presence larger than for absence?) terms. Because this technique is computationally efficient and works on pre-trained models, it opens up the opportunity for more systematic uncertainty quantification #cite(<Zurell2020>) in SDMs. CP, in short, can deliver the "maps of ignorance" that #cite(<Rocchini2011>, form: "prose") argued for: how difficult is it to make a prediction for the range at a given risk level is, in and of itself, an important information to frame the reliability of the results. Finally, CP can provide guidance on the feedback loop between SDM training and field validation #cite(<Johnson2023>) – areas where the range is certain are a much lower priority for sampling. Looking back at *TK*, the uncertain areas are much smaller than the certain ones, which provides actionable guidance for field-based validation.

= References

#bibliography("references.bib", style: "annual-reviews-author-date")