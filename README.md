## Automated Extraction of Digital Biomarkers for Parkinson's Disease

Machine-learning approach to classify user's diagnostic state using smartphone sensor data using observational data from [the mPower study](https://www.synapse.org/#!Synapse:syn8717496). You can find documentation of our approach [on Synapse](https://www.synapse.org/#!Synapse:syn10922704/wiki/471154).

Author(s): Patrick Schwab, ETH Zurich <patrick.schwab@hest.ethz.ch>
License: GPLv3, see LICENSE.txt

![overview](https://s3.amazonaws.com/proddata.sagebase.org/3357581/47271573-d357-476d-8240-7b8be3f64e25/h-carnn.png?response-content-disposition=attachment%3B%20filename%3Dh-carnn.png&response-content-type=image%2Fpng&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20180313T094849Z&X-Amz-SignedHeaders=host&X-Amz-Expires=30&X-Amz-Credential=AKIAIV5XCDRXPWB67YRQ%2F20180313%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=89ef127cc42a42b838b0fba1819219a733d50de8da4c1672cad4219384e6224b)

### USAGE:

- Runable scripts are in the dream_parkinsons/apps/ subdirectory.
- Command line parameters are described in dream_parkinsons/apps/parameters.py
- To use our scripts to train/evaluate you first need to download all synapse data into a SQLite database using the provided load_db.py scripts.
  Note: This is to speed up training / evaluation.
- (optional) As a final step, you may use apply_pca.py to generate a PCA transformation for the extracted feature vectors based on
  the samples in the validation set.


### Citation

If you reference our methodology, code or results in your work, please consider citing:

    @article{20.500.11850/224487,
      author = {Schwab, Patrick and Khashkhashi Moghaddam, Mohammad A. and Karlen, Walter},
      year = {2017},
      title = {Automated Extraction of Digital Biomarkers for Parkinson's Disease using A Hierarchy of Convolutional Recurrent Attentive Neural Networks},
      journal = {10th Annual RECOMB/ISCB Conference on Regulatory & Systems Genomics with DREAM Challenges; Conference Location: New York, NY, USA}
    }
