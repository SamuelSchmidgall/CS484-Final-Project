# Selective Hearing

Hearing loss, Schizophrenia and ADHD are all irreversible handicaps that make it difficult for a surprising
large population of affected individuals to listen and focus in on a single voice in a noisy environment.
What if it were possible to cut out the noise out of the environment so that a given individual could
solely hear the speaker? The goal of this work was to use Data Mining techniques with the goal of
removing extraneous noise from an environment where an individual is speaking. In this project,
podcast data is taken from the MIT podcast dataset, and sinusoid waves of varying frequencies are
combined with the original dataset to serve as noise files. The dataset for this project contained 51,033
wav files which totaled to 41.94 GBs. Using the noise files as the input data and the frequency of the
sinusoid wave as the expected value, three statistical models were trained with the goal of predicting
which frequency of noise was added to the data element in question; a Support Vector Machine, an
Artificial Neural Network, and a Convolutional Neural Network. The Artificial Neural Network showed
the best results with an average error of 21 hz on the training set and 28 hz on the test set.

## Getting Started

Definitely read the report if you're interested in this project.

## Authors

* **Samuel Schmidgall **

## Acknowledgments

* Dr. Rangwala -- my professor
