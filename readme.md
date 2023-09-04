# 挖呀挖呀挖

## 1. 代码结构
- src: 代码文件夹
  - train.py: 训练示范代码，包含了数据的载入，模型训练，模型保存等功能
    - WorldModelDecisionTransformerModel： 模型网络结构定义
    - DynamicsModel： 环境转移模型类
    - IterDataset： 数据集类
  - utils.py: 一些辅助函数
  - local_eval.py 本地模型测试文件
  - predict.py: predict示例文件
  - env.py: 环境类，在本示例中主要用于数据处理
- dataset: 数据集文件夹
- commit: 提交文件夹
  - predict.py: 模型评估示范代码，组织方根据该文件进行模型评估
    - state_predict： 模型预测函数，**参赛者提交的代码中必须实现此函数，组织方调用此函数进行模型评估**
- requirements.txt: 依赖包列表
- test_environment.txt: 评测环境依赖包列表

## 2. 环境配置
- 操作系统：Ubuntu 20.04
- GPU驱动和CUDA安装：如果需要用GPU训练模型，需要安装NVIDIA驱动，CUDA和cuDNN，具体安装方法请参考[PyTorch官网](https://pytorch.org/get-started/locally/)
- python版本：python3.8
- 依赖包安装：pip install -r requirements.txt
- torch安装：pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
- accelerate安装：pip install accelerate==0.19.0 setuptools==59.5.0
- transformers安装：pip install transformers==4.29.0

## 3. 数据集
- 数据集下载：[数据下载链接](https://yeying-gateway.apps-fp.danlu.netease.com/xiyin/release_data.zip?Signature=9oUjukwmkN%2F8qkBt9df1KoYFZFg%3D&Expires=4845006516&AWSAccessKeyId=HU4J73EY50QM95RMYOGM)
- 数据集说明：数据集包含了训练数据和验证数据
  - 训练数据：包含了训练数据和数据标签，训练数据用于模型训练
  - 验证数据：包含了验证数据和数据标签，验证数据用于模型调参，**也可以用于模型训练**
- 数据集结构
  - dataset: 数据集文件夹
    - train：训练数据文件夹
      - XXX1：子文件夹
        - XXX1.csv: 具体训练数据文件1
        - XXX2.csv: 具体训练数据文件2
        - ...
      - XXX2：子文件夹
        - XXX1.csv: 具体训练数据文件1
        - XXX2.csv: 具体训练数据文件2
        - ...
      - ...
    - test：验证数据文件夹
      - XXX1：子文件夹
        - XXX1.csv: 具体验证数据文件1
        - XXX2.csv: 具体验证数据文件2
        - ...
      - XXX2：子文件夹
        - XXX1.csv: 具体验证数据文件1
        - XXX2.csv: 具体验证数据文件2
        - ...
      - ...
- 数据载入: 参考train.py中的IterDataset类，该类实现了数据的载入，参赛者可以根据自己的需要进行修改
- CSV文件内各列含义：
  - XXX_pos: XXX关节角度
  - XXX_vel: XXX关节角速度
  - XXX_pwm: XXX关节PWM，液压驱动信号，用于控制挖机关节动作
  - XXX_next_pos: XXX关节下一时刻角度角度（类似于label）
  - XXX_next_vel: XXX关节下一时刻角速度（类似于label）
  - state_time: 收到pos，vel等信息后多久发出了PWM信号（单位：秒）
  - action_time: 发出PWM信号后多久收到了pos，vel等信息（单位：秒） 
## 4. 提交说明
- 日常评测提交代码说明
  - 代码提交：参赛者需要维护一个私有github仓库，并将组织方的github账号（superdi424@gmail.com）作为[协作者添加到仓库中](https://docs.github.com/zh/account-and-profile/setting-up-and-managing-your-personal-account-on-github/managing-access-to-your-personal-repositories/inviting-collaborators-to-a-personal-repository)，该仓库用于存放参赛者的代码，组织方会在参赛者每次在赛事网站上进行提交操作后，自动从参赛者的git仓库中拉取代码，进行模型评估
  - 评测代码：参赛者需要将和模型评测相关的文件统一放在commit文件下，其中需要包括predict.py文件，在predcit.py文件中需要实现state_predict函数，评测时会直接调用该函数做模型评估，函数的输入输出需要满足[规范](./commit/predict.py)，[predict.py](./src/predict.py)为官方提供的参考实现示例，参赛者可以根据自己的需要进行修改；选手提交的代码中所导入的包在评测[支持的包列表](./test_environment.txt)内，否则可能会导致评测失败。
  - 模型权重文件：参赛者需要提交一个模型权重文件，该文件用于模型评估，模型文件放在github仓库commit文件夹下，第一阶段模型大小限制为100M以内，评测时会进行检查，不通过直接判定提交失败
  - 依赖包列表：如果有本文档中说明的依赖包以外的依赖包，参赛者需要提交一个requirements.txt文件，该文件用于安装参赛者代码所需的依赖包，原则上不建议使用额外的依赖包
  - 代码说明【推荐】：参赛者需要提交一个README.md文件，该文件用于说明参赛者的代码，包括但不限于模型结构、模型训练方法、模型评估方法等
- 复核阶段评测提交代码说明（具体开始和截止日期以及提交方式请关注赛事网站公告）
  - 训练代码：参赛者需要提交一个train.py文件，该文件内需要实现模型训练的代码，参赛者可以根据自己的需要进行修改
  - 依赖包列表：如果有本文档中说明的依赖包以外的依赖包，参赛者需要提交一个requirements.txt文件，该文件用于安装参赛者代码所需的依赖包，原则上不建议使用额外的依赖包
  - 代码说明【推荐】：参赛者需要提交一个README.md文件，该文件用于说明参赛者的代码，包括但不限于模型结构、模型训练方法、模型评估方法等
  

## 5. 运行与启动
- 如果需要使用accelerate,请在运行前安装accelerate
  - 部分推荐的accelerate配置：
```shell
- Accelerate version: 0.19.0
- Platform: Linux-5.10.0-21-cloud-amd64-x86_64-with-glibc2.29
- Python version: 3.8.10
- Numpy version: 1.18.5
- PyTorch version: 1.9.1+cu111 (True)
- Accelerate default config:
        - compute_environment: LOCAL_MACHINE
        - distributed_type: MULTI_GPU
        - mixed_precision: fp16 
        - use_cpu: False
        - gpu_ids: all
        - rdzv_backend: static
        - same_network: True
        - main_training_function: main
        - downcast_bf16: no
        - tpu_use_cluster: False
        - tpu_use_sudo: False
```
- 训练启动命令
1. 使用accelerate运行
```shell 
cd src
accelerate launch train.py --命令行参数1 --命令行参数2 ...
```

2. 不使用accelerate运行
```shell
cd src
python3 train.py --命令行参数1 --命令行参数2 ...
```

- 本地测试启动命令（建议参数选手提交代码前先运行本地测试文件，本地测试无误后，再将评测相关代码和模型放在commit文件夹下）
```shell
cd src
python local_eval.py
```

在Jupiter服务器上配置test_environment.txt环境，

- world_model_9995.0_00488k.pth，99.525415
- world_model_30426.0_01485k.pth，99.641523
- world_model_30527.0_01490k.pth,  99.64179

A为第一项，B为第二项，AS(A_state)，AP(A_position)，BS，BP

loss修改为e^(A)+e^(-B):

- 8.29/world_model_2749.0_00134k.pth: 99.061751

loss修改为AP+10*AP/BP:

- 8.29v2/world_model_5398.0_00263k.pth: 

妤妃修改的模型：

- world_model_4676.0_00228k.pth：提交时间： 2023-08-31 12:49:04， 99.396133
- world_model_12403.0_00605k.pth：提交时间：2023-09-01 13:47:33，，99.534424
- world_model_20993.0_01025k.pth，提交时间：2023-09-02 22:00:28，99.519424，对比上个模型，末端误差更小，但是末端误差比例更大
- world_model_27897.0_01362k.pth，提交时间：2023-09-03 22:33:04，99.569192，末端误差更高，比例更小了
- world_model_32935.0_01608k.pth，提交时间：2023-09-04 18:45:32，99.64718
- world_model_34862.0_01702k.pth，提交时间：

未修改代码的模型：

- world_model_32835.0_01603k.pth， 提交时间：2023-09-02 22:55:48，99.570749
- world_model_44295.0_02162k.pth，
- world_model_44335.0_02164k.pth，提交时间：
