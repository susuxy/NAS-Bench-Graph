# Machine Learning models to predict accuracy based on AUTOGL parameters

Run the following codes:
```
python main.py --train_mode normal
```

Users can specify --train_mode be ['normal', 'data_transfer']. 

If set it to be normal, it means use one dataset 80% as training set and the remaining 20% is testing set. Users can set --data to specify which datasets to use. 

If set it to be data_transfer, it means training on some of the datasets and testing on other datasets. Users can set --train_data and --test_data to specify the datasets.
