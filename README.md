# Part-of-Speech Tagging using the Viterbi Algorithm

## Overview

In this assignment, I implemented a part-of-speech (POS) tagger using the Viterbi algorithm and the Hidden Markov Model (HMM) approach. The main components include reading and processing the training data, calculating prior probabilities and likelihoods, implementing the Viterbi algorithm, and evaluating performance on development and test sets.

## Data Processing

- **Training Data:**  
  The training data (`WSJ_02-21.pos`) was read and processed to extract sentences, where each sentence is a list of (word, tag) tuples.
- **Probability Calculations:**  
  - **Prior Probabilities:** Calculated based on bigram occurrences of tags.  
  - **Likelihoods:** Computed from the occurrences of word-tag pairs.

## Handling Out-of-Vocabulary (OOV) Words

OOV words (words not present in the training corpus) were managed by:
- Assigning a constant likelihood for all OOV items.
- Relying on the transition probabilities for tag assignments.

## Viterbi Algorithm

The Viterbi algorithm was implemented to find the most likely sequence of tags for a given sentence, using the calculated prior probabilities and likelihoods.

## Evaluation

- **Development Set:**  
  Evaluated on `WSJ_24.words` using the provided scoring program (`score.py`), comparing the output to the gold standard tags in `WSJ_24.pos`.
- **Final Submission:**  
  The tagger was retrained on a combined corpus of the training and development sets, then evaluated on the test set (`WSJ_23.words`). The final output was saved in `submission.pos`.

## Results

- **Development Set Accuracy:** 95.84237%
- **Test Set Accuracy:** 98.96131% (as per the score grader)

## Conclusion

The implementation of the Viterbi algorithm for POS tagging using the HMM approach proved highly effective, achieving high accuracy on both the development and test sets. This project provided valuable insights into POS tagging and the application of the Viterbi algorithm in natural language processing tasks.
