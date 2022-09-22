Run the train_transfer or train_basic to train a cross-domain diagnosis or basic diagnosis.
Run the test_offline to test trained model, the args could be changed.
1. Choose the dataset:
   1. two parameters:
      1. "data_dir": dataset file path including filename.
      2. "dataset": the dataset class (not file but class).
   2. choose the working conditions. It is given through the parameter of dataset.prepare(wc)
2. About the model:
   1. transfer: There is a corresponding transfernet for each type data (2D and 1D).
   2. basic: There are several basic networks to finish basic diagnosis and be the backbone of transfernet.
