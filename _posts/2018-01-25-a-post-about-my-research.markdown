---
layout: post
title:  "My Research: Latent Variable Measure of Interstate Hostility"
date:   2017-11-25 
---

This blog post presents my Master thesis (available [here]({{ site.url }}/assets/ZTerechshenko_MA.pdf)), where I created a new measure of interstate hostility by applying Bayesian ordinal item-response theory (O-IRT) model to a conflict events dataset, which I have created using data from Militarized Interstate Disputes (MID), Integrated Crisis Early Warning System (ICEWS), and International Crisis Behavior (ICB) datasets. 

The model I constructed assumes that hostility is a unidimensional trait that can be measured only using observed outcomes. I employed an item-response theory (IRT) model, which is a type of latent variable model used to generate estimates of a latent trait of interest (hostility) by combining information from observable items or manifest variables (conflict events). 

The IRT model can be written as:

<img src="/assets/IRT-1.png" alt="my caption" style="width: 350px;"/>

where F(.) denotes the logistic cumulative distribution function.

The likelihood function for $\beta$, $\alpha$ and $\theta$ given the data is presented below:

<img src="/assets/IRT-2.png" alt="my caption" style="height: 100px;"/>

Since *θ*<sub>*it*</sub> cannot be fully observed, all of the parameters of interests, namely *θ*, *β*, and *α* must be estimated simultaneously. The use of Bayesian estimation, in which the model identification is achieved through the assignment of prior distributions is one of the most common approaches to solve this issue.

I set the prior for the latent trait  to *θ*<sub>*it*</sub> ∼ *N*(0,1). This constraint reflects an assumption that the population of dyads is roughly normally distributed across the spectrum of hostility.  Slightly informative prior *β*<sub>*j*</sub> ∼ *Γ*(4,3)  restricts the value of the item discrimination parameter to be positive and reflects an assumption that all indicators contribute significantly (and in the same direction) to the latent variable. Under this model specification, increases on the values of each indicator 
*y*<sub>*j*</sub> correspond to the higher values of the latent trait. The item difficulty parameters *α* were given *N*(0,4) priors with *α*<sub>*j1*</sub> > *α*<sub>*j2*</sub> for all <sub>*j*</sub>. 

The model was estimated using Stan, a C++ program, which performs Bayesian inference  using a No-U-Turn sampler.

Here is the example of posterior estimates for two dyads: United States-Russia and Russia-Sweden.  The first dyad has consistently higher levels of hostility across years of observation than the second dyad. Furthermore, it has narrower confidence intervals, indicating higher levels of certainty about the measure.


![jpeg](/assets/PostEstim.jpeg)


In order to assess the fit of the model I used posterior predictive checks. 
In particular, I simulated replicated data and then compared these data to the observed dataset. 
If the model fits, then replicated data generated under the O-IRT model should look similar to observed data from the combined conflict data set. Using 1,000 draws from the posterior distribution, I predicted each of the *j* items *y*<sub>*itj*</sub> for every dyad-year observation, for each *y*<sub>*itj*</sub> observed. 
I then computed a sum of squared differences of observed *y*<sub>*itj*</sub> and *d* posterior predicted values *ŷ*<sub>*itjd*</sub> to measure the accuracy of each set of predictions. This calculation is expressed as:

<img src="/assets/PostPred.png" alt="my caption" style="width: 300px;"/>

The figures below display the results of the predictive checks. Blue lines correspond to the inferred means with 95% confidence intervals. Orange dots correspond to the actual sample means. The fact that real values fall within the confidence intervals of simulated values provides an evidence that the model fits data well.


<img src="/assets/postpredcheks.png" alt="my caption" width = "800" height = "600" >









