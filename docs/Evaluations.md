## Evaluation strategy

While several evaluation methods exist to see the performance of the model, we will first 
evaluate the models, silver truth or otherwise on the basis of the following metrics:
- Jaccard

We currently have two places containing results that will help us see if we are evaluating the models correctly.

1. Results from 2020.07.31 excel sheet, seg training page
2. SEG_log files in the competitors' folders

Utilizing synchronized data and dataset parquet from preprocessing step, we can now call evaluate competitor from evaluations.py

This method outputs results of a selected competitor or all of them from a given dataset.

These results when compared to SEG_log files and excel sheet are similar, but there are slight differences.
To see these differences, we utilize the finding_problem_labels.ipynb in notebooks folder. This shows us that the 
synchronization algorithm we use is different to the one used in the competition. Slight differences in low 
percentage of labels skew the results a little bit. Here I'm going to make the decision to use our new synchronization,
and therefore also it's results as our baseline.