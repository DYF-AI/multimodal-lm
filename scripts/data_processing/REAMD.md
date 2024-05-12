
### 统一数据格式
每个数据集里面都有：metadata.json文件
```json
[
  {
    "数据来源": "",
    "数据用途": "训练|验证|测试",
    "图片路径": "image_path",
    "抽取结果": "dataset_name",
    "OCR文本": "text",
    "OCR坐标": "",
    "OCR分数": ""
  }
]
```

### 训练时,多种数据结合配置
```json
{
  "火车票": "metadata_file_path",
  "XFUND": "metadata_file_path"
}

```
