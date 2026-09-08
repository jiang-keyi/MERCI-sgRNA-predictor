# DeepMEns: an ensemble model for predicting of sgRNA on-target activity based on multiple features
## Environment
The algorithm mainly uses the following packages:
- python=3.7
- tensorFlow==2.4.1
- keras==2.10.0
- numpy==1.19.5
- pandas==1.3.5
- sklearn== 1.0.2
- matplotlib==3.5.3
## Useage
- 5_Wt/HF1/esp_DeepMEns.py: Python package, storing DeepMEns related codes in WT-SpCas9, eSpCas9(1.1), and SpCas9-HF1 datasets.
- HF1_dataset, WT_dataset, esp_dataset, independent_test_datasets: Contains all datasets and features for experiments.
- Wt/HF1/esp_DeepMEns_0.h5, Wt/HF1/esp_DeepMEns_1.h5, Wt/HF1/esp_DeepMEns_2.h5, Wt/HF1/esp_DeepMEns_3.h5, Wt/HF1/esp_DeepMEns_4.h5: They are models trained by training sets in code that can be used directly in WT-SpCas9, 
eSpCas9(1.1), and SpCas9-HF1 datasets.
- hand-crafted-feature.py: Code that contains the biological characteristics used in the article.
