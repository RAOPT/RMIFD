# Introduction
### 基于深度迁移学习的旋转机械核心部件的智能故障诊断
1. 代码采用OOP思想，基于现有的开源代码框架和迁移学习代码框架整合而成。

2. 将智能故障诊断流程中的操作整合为了：训练器（分普通诊断、迁移诊断）、测试器、数据加载器、优化器、模型、分类器等；

3. 代码操作简单，实现逻辑清晰；

4. 可扩展性比较强；

5. 自定义程度高：主要体现在可以适应不同类型的数据集、不同类型的基本模型作为特征提取器；

6. 自定义dataset需要提供统一的接口，以保证后续代码可以不用更改调用操作；

7. 提供了一些常用分析方法的代码，比如绘制数据分布的散点图（将数据转换为ndarray后再操作）；

8. 支持分阶段训练（保存最近、最优模型参数、以及周期、学习率等状态）;
# Method
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
# Nerd Tree
1. CNN_Datasets: convert raw data to datasets the CNN could handle.
2. datasets_tools: classes for deal with different datasets.
3. Log: records of every training process. Named by different models\ working conditions \datasets \data types.
4. networks_basic: normal diagnosis tasks models.
5. networks_transfer: transfer diagnosis tasks models; the backbone come from networks_basic.
6. Shell_yaml: config yaml file and shell script.
7. x_tools: support the train_util classes that including detail training code.