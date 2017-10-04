Automated Extraction of Digital Biomarkers using A Hierarchy of Convolutional Recurrent Attentive Neural Networks
-----------------------------------------------------------------------------------------------------------------

Machine-learning approach to classify user's diagnostic state using smartphone sensor data.

Author(s): Patrick Schwab, ETH Zurich <patrick.schwab@hest.ethz.ch>
License: GPLv3, see LICENSE.txt

USAGE:
------
- Runable scripts are in the dream_parkinsons/apps/ subdirectory.
- Command line parameters are described in dream_parkinsons/apps/parameters.py
- To use our scripts to train/evaluate you first need to download all synapse data into a SQLite database using the provided load_db.py scripts.
  Note: This is to speed up training / evaluation.
- As a final step, you may use apply_pca.py to generate a PCA transformation for the extracted feature vectors based on
  the samples in the validation set.
