#set text(font: "Libertinus Sans")
#show math.equation: set text(font: "Libertinus Math")

#let response(ed: false, body) = block(fill: color.hsl(195deg, 100%, 98%), inset: 1em, radius: 0.15em, width: 100%, stroke: 0.1mm + color.hsv(195deg, 100%, 45%))[
  #if ed == true {
    text(weight: "bold")[Comments for the editor]
    linebreak()
    linebreak()
  } else {
    text(weight: "bold")[Response:]
  }
  #body
]

#let add(body) = text(fill: rgb(0, 100, 0), weight: "regular")[#body]
#let change(body) = text(fill: rgb(0, 100, 100))[#underline(body, stroke: rgb(0, 90, 90))]
#let cut(body) = text(fill: rgb(150, 150, 150))[#strike(body, stroke: rgb(100, 0, 0))]
#show "TK": text(weight: "bold", fill: rgb("#e08619"))[TK]

#response(ed: true)[
  We have ...

  The changes are also presented in a track changed manuscript uploaded as a document for reviewers, which indicates #add[additions], #change[changes], and #cut[deletions] to the text following comments made by both reviewers.
]

= Reviewer 1

I think the manuscript is an interesting basis for an introduction to Conformal Prediction for SDM, and that such introduction could help the SDM community to appropriate these methods.

#response[I appreciate the comment by the reviewer. The purpose of this manuscript is indeed to provide a conceptual overview of what CP can achieve when applied to SDMs, and to flag some conceptual issues that remain to be adressed. In this light, the reviewer made several highly relevant comments that helped strengthen the manuscript, for which I am grateful.]

Yet, I acknowledge that it's not the first paper using this method for SDM (e.g. Davis et al., 2024, PS: I'm not even author of it). The study doesn't present a new method, nor scientific results in terms of methods or ecological knowledge. Besides, there's a lot of issues regarding the interpretations of CP. 

#response[I appreciate the reviewer for pointing out this manuscript, that I missed during my literature review. I Want to point out that the description of CP in Davis et al. is minimal, and that our manuscripts therefore serve two very distinct purposes. As I dedicate a full manuscript to providing an overview of CP, and identify issues that remain, I do not see the existence of the Davis et al. manuscript as incompatible with this one.]

So, even though the manuscript is useful, i'm not sure it fits as a research article with the current content, but I let the editor decide about that. There's at least a need for major revisions and added content.

#response[I have addressed the responses to the reviewer and hope that the editors will see the value of this manuscript in its revised format. The manuscript in its revised form presents the conformal prediction framework in a lot more depth, and provides clear recommendations for both its extension and adoption for species distribution models.]

== General

Even though conformal prediction is yet scarcely used for SDMs (but see e.g. this published SDM workflow).

Davis, A. J., Groom, Q., ... & Strubbe, D. (2024). Reproducible WiSDM: a workflow for reproducible invasive alien species risk maps under climate change scenarios using standardized open data. Frontiers in Ecology and Evolution, 12, 1148895.

#response[*TODO CITE*]

What do CP predicted sets mean under climatic novelty? If my understanding is correct, exchangeability, i.e. the main assumption of CP, is violated when we predict under shifted covariates compared to training conditions. It means that CP loose its desired marginal coverage guarantee for predictions under future climate scenarios, so this method shouldn't be used or interpreted in this context.

#response[The reviewer is absolutely correct that exchangeability is a core assumption of CP. That it would be violated under climate change scenarios is possible, but not necessarily always so, and this is why in this manuscript I quantify this through the proxy of bioclimatic novelty. Furthermore, recent results suggest that in-sample calibration makes CP more robust during training, and also establish additional techniques to minimize the effect of possible loss of exchangeability during the model application. These results are now cited in the main text (the section on climate change has been re-organized), and are also given a sub-section in the discussion.]

Not sure it's a relevant choice to use PO + pseudo-absence data to illustrate CP because it is misleading to then speak about probability of presence, while you could just use real presence-absence data. Obviously, it doesn't exist for the Sasquatch, which is not fun, but take comfort in the fact it can be difficult to understand what is meant by the uncertainty of predictions of observations of a fictional creature...

#response[the reviewer is correct about the fact that absences in this dataset are pseudo-absences. I maintain that is is relevant here, for three reasons.

First, presence-only data are the overwhelming majority of cases for which SDMs are applied. Biodiversity is a data-defficient field when it comes to species presence, but is orders of magnitude more data defficient for species absences. Testing a method on the scenario where it will most lilkely be applied makes the most sense.

Second, the assumption that pseudo-absences are equivalent to absences is inherent to all forms of SDMs, even when for example using MaxEnt. That this assumption is most often unstated does not make it less true. It is established practice in the litterature to simplify the writing by discussing the pseudo-absences as absences.

Finally, as the reviewer correctly identifies, the species is fictional. The manuscript would work exactly in the same way with, for example, virtual species, for which full knowledge of presences and absences would be available. But the data used in the manuscript are realistic-looking, and work well enough to discuss the method.]

There appears to be errors in the mathematical definition of the conformity score which doesn't make sense (cf my comment below). The text must be fixed, and the associated results and Figures corrected accordingly.

#response[as I will explain in the detailed comments, the reviewer is factually incorrect about this statement. After a careful re-reading of this section, and verification of the implementation, there is no need to modify either the text or the figures, as these were produced in the correct way initially.]

A variety of score functions are used, even for binary classification, so I think an introduction to CP for SDM should assess the pros and cons of these.

#response[the reviewer is correct in that a nunmber of CP techniques are used -- these are now more explicitely mentioned in the manuscript, and a section has been added to pinpoint two techniques with particular relevance to SDMs.]

Various abuse of terms, including "species" (Is it?), "presence", "absence" -> This is misleading as you work with PO with many "false absences". This leads to the even more misleading term "sure absence" when it comes to CP credible sets

#response[the reviewer seems to disagree with the choice of study system and nomenclature. I take note of this disagreement, but as I have laid out my arguments about why these are immaterial to the description of the method, I have made no further changes to the manuscript.]

I demand a consistent change of terms, as for instance "presence" should rather be "observation" and "absence" should be "non observation".

#response[this change has not been made. I have added a paragraph about the choice of study system and terminology. I am hopeful that it will help readers figure out what the model does, which is the purpose of this article. Indeed, producing a species distribution model of the Sasquatch is not a relevant scholarly exercise, not because it is uninteresting, but because it has already been done by multiple colleagues (whose contributions are cited in the main text).]

It would be interesting to developp and clarify the comparison between bootstrap and CP, or even better evaluate the alignment empirically.

#response[the reviewer made several specific comments about this point here and later on, and I have adressed them in the revision. My specific comments and a list of changes made are presented in response to each of these specific comments.]

Fig.3.C suggests that the uncertainty quantification of bootstrap aligns with the one of CP which makes sense, but i'm also not surprised that the link between IQR of ensemble predictions and CP categories is not strong because you're probably not comparing the right quantities. Indeed, uncertainty quantification with bootstrap ensemble should also account for the mean predicted value, not only for the variability/IQR.

#response[the use of bootstrap variability to evaluate uncertainty, and of the original model to report the prediction, is common in SDMs. I have clarified this information in the text, alongside additional references. For this reason, I maintain that the current approach is valid. I have also added a clarification about how the bootstrap variability _v._ uncertainty coming from the CP prediction can be used to suggest different types of sampling, namely more data for model training and more data for model validation.]

A key difference between CP predictive sets vs any ensemble based predictive set, is that, unless the ensemble model prediction is perfectly calibrated to the real probability p(Y=1|X) (unrealistic), the latter don't provide the minimum coverage guarantee of CP, i.e. they don't inform about (absolute) predictive error. 

#response[this is correct. As no claim to that effect is made in the manuscript, and that the reviewer does not seem to direct this comment at a specific claim made in the manuscript, I have made no changes.]

Besides, ensemble based uncertainty doesn't account for model bias/misspecification. For instance, ensemble CI will shrink a lot when the ensemble is trained on big data even when the component models are too simplistic and biased (e.g. linear regression)-> illusion of confidence.

#response[this is correct, but I again fail to extract an actionable comment from this sentence. Surely the reviewer is not suggesting that regression is a bad approach for SDMs? Although I do not disagree with this point, for the same reason as the above comment, I have made no changes to the manuscript.]

== Precise comments

=== Intro

30-32: Prediction uncertainty can also arise from uncertainty about the predictors themselves (e.g. errors in climate variables, which are important ), but those not accounted for in conformal -> point of discussion?

#response[this is correct - this point has been added to the conclusion section.]

40-41: yes, but bootstrap uses all the data for the training and prediction, while conformal is deprived from the calibration part of data, a problem to discuss for rare species?

#response[TK this is only partially correct. Methods like cross-conformal prediction apply, essentially, cross-validation to the identification of the threshold. This was already cited in the manuscript. I think that it is too early to introduce this nuance here, and as the point is covered later on, I have made no changes to the manuscript at this specific place.]

41: "built-in methods" what do you mean?

#response[this has been changed to "methods that are specific to a particular classification algorithm". I thank the reviewer for pointing out the confusing formulation.]

54: "is not a measure of variability coming through the distribution of data" -> I don't understand this statement, could you developp? In my understanding, conformal sets implicitely capture the probabilistic distribution of the data conditionally to predictors, which is needed to guarantee coverage. This is in line with your next sentence, so I guess it's just a misleading formulation.

#response[]

=== Methods

84: No reference for this method to generate pseudo-absences. I understand the simple logic, but it doesn't account for undervisited areas which are equally selected as very visited ones for a given distance to presence observation (among other sampling biases). Classical Pseudo-Absence generation methods could be used (see e.g. Wisz, M. S., & Guisan, A. (2009). Do pseudo-absence selection strategies influence species distribution models and their predictions?)

#response[*TODO THIS IS CLASSICAL*]

I get that what means "presence" or "absence" is not central in this article, but if you're goal is really to ease the uptake of this kind of method, you should rather use a relevant use-case.

#response[this point has already been adressed. As the reviewer correctly figured out, the definition of presence and absence is not central to this manuscript.]

To avoid this kind of discussion, i would suggest to use a presence-absence dataset.

#response[this point has already been adresed.]

110-116: What does it mean that results hold qualitatively? Lack of transparency about this internal result, why not shown as appendice?

#response[]

117-118: Why?

#response[]

119: "v." -> "vs"?

#response[_v._, _vs_, and _vs._ are all accepted abbreviations of the Latin _versus_.]

128-129: What does "similar" mean? This statement is not factual and exhibits a lack of scientific transparency. Given that it's not central anyways, you could just remove it.

#response[before answering the question, I need to voice my displeasure at the phrasing of this comment. To qualify a statement as "not factual" and "lacking transparency" is a rather serious accusation, and one I wish the reviewer would not be so eager to throw around.

With this stated: this sentence means that, when using different classifiers to produce the model on which CP is then applied, the predicted distributions do not change overmuch. This has been clarified as "resulted in similar predicted ranges and cross-validation performance, which suggest that the problem can be handled well by multiple algorithms".

This statement is, in my opinion, factual, in that it describes the fact that model selection does not have an outsized impact on the results. It is, still in my opinion, contributing to transparency in that it informs that this type of checks have been performed before discussing the results. I do not think this model comparison is important enough, and it is, regardless, certaintly not central enough, to warrant a supplementary material.

Should _the editor_ disagree, I would be willing to reconsider.]

131: This is misleading. Such model can't capture presence probability because it's based on opportunistic presence-only data and pseudo-absences, i would change :
"associated to the presenceof the species" -> "associated to the observation of the species"
and stick to this terminology for all along

#response[this point has already been adressed.]

134: Similarly "the probability that the species is absent from the location." -> "the probability that the species is not observed in the location."

#response[this point has already been adressed.]

=== Conformal Prediction

215-217: I don't understand this sentence, Could you clarify? specifically which confidence interval? The one on the probability of presence?

#response[]

258-231: Does it happen that the predicted set is empty? Would you then interpret it differently as the set with both outcomes {0,1}?

#response[]

234-247: There appears to be several errors around the definition of the conformity score l.234/235. Where does this score comes from? Why predicted probabilities are put in the exponential?

#response[]

For a site i, the classic softmax score would simply be p_{-}^i when the ground truth (true label) is absence, or p_{+}^i when it's presence, i.e. the score for site i is: s^i = p_{+}^i 1_{Y_i=1} + p_{-}^i 1_{Y_i=0}

#response[the reviewer is factually incorrect. The softmax function has a single definition:

$
sigma(bold(x))_i = "exp"(x_i)/(sum_j "exp"(x_j))  
$

For binary classification, where the sum of predictions for the positive and negative classes must be one by definition, the reviewer will be able to verify that the formulation in the main text is indeed correct. The formulation that the reviewer suggests is also, in fact, mentioned in the main text:  "[n]ote that this can be done without using the $text("softmax")$ function #add[(_i.e._ $s_+ = p_+$, $s_- = 1 - p_+$)]".

]

You don't mention that, in calibration, the true label determines which probability is used and that the quantile is computed over $s^1$, $s^2$, ...

#response[this has been fixed in the revision.]

Furthermore, You don't mention that there exist different commonly used score functions, even in this simple case of binary classification, which are alternatives to the softmax that you intend to use, e.g. the Adaptive Prediction Sets (APS) score, see: Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: A gentle introduction. Foundations and Trends® in Machine Learning, 16(4), 494-591.

#response[more than a different score function, APS is a different way to handle the scores of all classes (but in the reference that the reviewer cites, softmax is still used to generate these scores). I have added a clarification about APS, as well as additional clarifications and mentions of CP techniques. Note that the manuscript already discussed full, split, inductive, and cross-conformal approaches.]

263-266: But if the empty set is predicted quite often, it's possible to have an inefficiency of 1 and yet to have predictions with both outcomes, right?

#response[the reviewer is correct, in that a model that would return $emptyset$ half the time, and ${+, -}$ half the time, would achieve an inefficiency of one, but this is an almost situation to observe empirically. The next sentence in this paragraph estalishes that an inefficiency close to unity is not desirable anyway, as the purpose of CP is to identify uncertain predictions, _i.e._ it should be applied in a way that results in "enough" (as defined by the $alpha$ value) predictions having $C = {+, -}$.]

282-284: Sentence not clear to me.

#response[]

287-289: Sentence not clear to me.

#response[]

Table 1: what means NPV and PPV?

#response[this information has been added to Table 1. I apologize for the omission in the original submission.]

=== Results

332-333: And this threshold depends on how the chosen evaluation metric "weight" TP, TN, FP and FN, e.g. TSS would give different threshold vs Jaccard or F1 , but the interpretation behind the metric choice is quite implicit

#response[this is correct, and I have added "for a given measure of model optimality" to the text here. The model used in this manuscript (MCC) was clearly identified in the section on tuning, together with a citation that establishes that it is the current state of the art for the evaluation of binary classifiers.]

387-389: "Uncertainty in the conformal classifier is coming from comparing the prediction to all other predictions under an estimation of the distributions for the conditions leading to the prediction of the presence (or absence) outcome" -> Sentence too abstract and not clear to me, and the next sentence doesn't make it more concrete.

#response[]

= Reviewer 2

Disclaimer: although an ecological modeller by training and doing mostly statistical modelling, I am not doing active research on SDMs (except those that separate detection from presence, i.e., not the ones considered in the submitted manuscript). I was previously to my reading of the manuscript unfamiliar with conformal prediction. My review should therefore be taken as that of somebody reasonably familiar with ecological statistics at large, but not that of a specialist in either classical species distribution modelling or machine learning. 

#response[]

== General overview

The manuscript introduces how conformal prediction, a machine learning technique well-grounded in statistical theory, can help make predictions of SDMs that better incorporate uncertainty. It is illustrated with a (somewhat humorous) empirical example, including under novel climate conditions. Better communicating the uncertainty of SDM predictions is a major endeavour for ecological modelling and this manuscript will greatly help advancing in that direction. 

#response[]

The manuscript is pedagogical and very well written. Most SDM modelling choices have sounded very standard or well-justified to me, and I only have minor comments. I was able to get most of the theory by reading together the present manuscript together with a bit of Angelopoulos & Bates 2023, I suggest to put the arXiv link to this publication in the bibliography list to help readers. 

#response[]

== Line-by-line comments

l. 131 presenceof → presence of

#response[Fixed --- apologies for the typo.]

l. 133 p_{i}+ Does the author mean p_{i+}?

#response[indeed, the notation was intended to be $p_(i+)$; I have fixed this in the sentence, and added a parenthesis to clarify exactly what it means.]

l. 197 Here or slightly later, I would recommend to mention that Angelopoulos & Bates call credible sets «prediction sets» (if I got that correctly) and that coverage is a well-defined, classical property of confidence intervals in statistics, to help the reader connect with the existing literature. As I understand it coverage means exactly the same thing here as in classical statistics.

#response[I have replaced "credible sets" by "prediction sets" throughout, which is indeed the terminology used in Angelopoulos and Bates. I have also added the clarification about coverage, and thank the reviewer for pointing it out.]

l. 232 isn’t this robustness only true for split CP? Or is it more general?

#response[]

l. 243 Why uppercase S? (as opposed to lowercase)

#response[I used $cal(S)$ to note the distribution of the scores, from which the critical value for the inclusion of a class is derived. This has now been clarified in the text, and this notation is now re-used in a few additional places when intoducing variants of conformal prediction.]

Figure 3C I wasn’t sure if it was defined for alpha=0.05 or a variable alpha level

#response[]

l. 369-391 Perhaps related to the above, I found this section more difficult to grasp.

#response[]

l. 394 bu the SDM→ by the SDM

#response[]

l. 451 Regarding BIO15, perhaps link to a table of results since these is not presented here (possible to do that in a supplementary or code/data repository?)

#response[]

l. 463 Isn’t it fig 3A as fig 2A does not incorporate uncertainty?

#response[]

l. 469 Figure 6 legend. Does the author mean equivalent to Figure 3A (rather than 2A)?

#response[]

l. 504-505 It is indeed very true that CP is directly comparable to classical SDM prediction. Perhaps it would be useful to illustrate this with some numbers (or another figure). 

#response[]

Fig. 6 shows how the CP approach allows to disentangle predictions by levels of uncertainty – which is really great (nicely done). As I understand it, Fig 6B is built through matching Fig 6A to Fig 3A. In my opinion, it would also be interesting to compare Fig 6B to its equivalent built through matching Fig 2A to a future climate variant (naïve climate projection approach). How wrong can the naïve approach be, compared to an approach that incorporates uncertainty?

#response[]

l. 563 Specifics of that Elith 2019 ref are missing. 

#response[]