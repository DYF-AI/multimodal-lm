
### 数据处理
1）图片来源：网上爬虫+书籍pdf   
2）数据转换：将书籍pdf转为图片   
3）图片ocr：将数据数据调用ppcor转成文本+box(目前为了进行数据校验,生成为ppocrlabel格式)    
4）数据校验：使用ppocrlabel对ocr结果进行纠正   
5）构建csv索引数据：mllm-data.csv, 包括图片路径,ocr结果,图片size,类型,用途    
6）根据mllm-data.csv,生成arrow数据集(非必要)