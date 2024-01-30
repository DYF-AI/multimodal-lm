
### 代码
- run_train_v1.py
    - dataset使用preprocess函数处理
    - 不支持prompt训练
- run_train_v2.py


### 数据处理
1）图片来源：网上爬虫+书籍pdf   
2）数据转换：将书籍pdf转为图片   
3）图片ocr：将数据数据调用ppcor转成文本+box(目前为了进行数据校验,生成为ppocrlabel格式)    
4）数据校验：使用ppocrlabel对ocr结果进行纠正   
5）构建csv索引数据：mllm-data.csv, 包括图片路径,ocr结果,图片size,类型,用途    
6）根据mllm-data.csv,生成arrow数据集(非必要)


### 考虑下游任务（主要数据难获取）
#### 保单托管服务（可供保险公司调用）
#### 发票信息抽取
#### 证件类抽取任务
#### 卷面评分系统



## bug
```shell
 File "D:\ProgramData\Anaconda3\envs\torch\lib\site-packages\PIL\Image.py", line 3092, in open
    fp = builtins.open(filename, "rb")
PermissionError: [Errno 13] Permission denied: 'J:/data/mllm-data/mllm-pretrain-data/train/'
# https://stackoom.com/question/485f5
```