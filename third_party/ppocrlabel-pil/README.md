
### PPOCRLabel
- 使用PPOCRLabel对ocr预标注结果进行纠正
- 由于PaddleOCR代码库十分混乱,路径经常乱掉pip和代码库的代码,经常报错，ocr也是使用paddleocr的pip包;

#### 安装
```shell
pip install PPOCRLabel==2.1.3
```

#### 启动
```shell
PPOCRLabel --lang ch
```

### 修改记录
#### 修改1：
- 报错：AttributeError: 'NoneType' object has no attribute 'shape'
```angular2html
'NoneType' object has no attribute 'shape'
Traceback (most recent call last):
  File "D:\ProgramData\Anaconda3\envs\paddle\lib\site-packages\PPOCRLabel\libs\autoDialog.py", line 41, in run
    h, w, _ = cv2.imdecode(np.fromfile(Imgpath, dtype=np.uint8), 1).shape
AttributeError: 'NoneType' object has no attribute 'shape'
```
- 原因：cv2不支持读取带有中文字符路径文件
```python
  h, w, _ = cv2.imdecode(np.fromfile(Imgpath, dtype=np.uint8), 1).shape
```
- 解决：使用PIL替换cv2, 将上述代码替换为
```python
# 修改文件："D:\ProgramData\Anaconda3\envs\paddle\lib\site-packages\PPOCRLabel\libs\autoDialog.py", line 41
# 增加PIL读取图片函数
from PIL import Image
def load_image(image_path: str, return_chw: bool = True, size: tuple = None):
    image = Image.open(image_path).convert("RGB")
    if size is not None:
        image = image.resize(size)  # resize image
    image = np.asarray(image)
    image = image[:, :, ::-1]  # flip color channels from RGB to BGR
    w, h = image.shape[1], image.shape[0]  # update size after resize
    if return_chw:
        image = image.transpose(2, 0, 1)
    return image, (w, h)

# 替换上面代码
try:
    image_data, (w, h) = load_image(Imgpath, return_chw=False)
except Exception as e:
    print(f"load file {Imgpath} fail!")
    continue
```

#### 修改2
- 报错：AttributeError: 'NoneType' object has no attribute 'shape'
```python
Traceback (most recent call last):
  File "D:\ProgramData\Anaconda3\envs\paddle\lib\site-packages\PPOCRLabel\PPOCRLabel.py", line 1889, in saveFile
    self._saveFile(imgidx, mode=mode)
  File "D:\ProgramData\Anaconda3\envs\paddle\lib\site-packages\PPOCRLabel\PPOCRLabel.py", line 1934, in _saveFile
    self.openNextImg()
  File "D:\ProgramData\Anaconda3\envs\paddle\lib\site-packages\PPOCRLabel\PPOCRLabel.py", line 1880, in openNextImg
    self.loadFile(filename)
  File "D:\ProgramData\Anaconda3\envs\paddle\lib\site-packages\PPOCRLabel\PPOCRLabel.py", line 1550, in loadFile
    height, width, depth = cvimg.shape
AttributeError: 'NoneType' object has no attribute 'shape'
```
- 原因还是cv2不能读取中文路径文件
- 解决：
```python
# 修改: File "D:\ProgramData\Anaconda3\envs\paddle\lib\site-packages\PPOCRLabel\PPOCRLabel.py", line 1550, in loadFile
# cvimg = cv2.imdecode(np.fromfile(unicodeFilePath, dtype=np.uint8), 1)
cvimg, _ = load_image(unicodeFilePath, return_chw=False)
```

#### 修改3
- 报错:error: (-215:Assertion failed) _src.total() > 0 in function 'cv::warpPerspective'
  - 报错描述：在对PPOCRLABEL的框进行重新识别是，发生如下报错：
  ```
    Can not recognise the detection box in xxxx,png. Please change manually'
  
  unicodeFilePath is J:\data\mllm-data\xxxxxxxxx\wKh2CWERPJOAY2x-AAE62o598k0620.png
    OpenCV(4.2.0) C:\projects\opencv-python\opencv\modules\imgproc\src\imgwarp.cpp:3143: error: (-215:Assertion failed) _src.total() > 0 in function 'cv::warpPerspective'
    ```
    - 原因是我们的ocr预标注数据Label.txt是使用PIL读取图片数据，调用ppocr进行生产的（并不是在PPOCRLabel工具内部生产的）, 当我们修改数据框后, PPOCRLabel尝试再次使用cv2进行读取原图,此时由于cv2对路径较为敏感,经常会读取文件失败,才会出现如上情况
  
- 解决：
    依旧是修改PPOCRLabel源码, 把cv2读取改为PIL读取,就不惯着cv2的臭毛病...
    
```python
# 修改如下代码
 def reRecognition(self):
    #img = cv2.imdecode(np.fromfile(self.filePath,dtype=np.uint8),1)
    img, _ = load_image(self.filePath, return_chw=False)


def singleRerecognition(self):
      # img = cv2.imdecode(np.fromfile(self.filePath,dtype=np.uint8),1)
      img, _ = load_image(self.filePath, return_chw=False)
```


#### 修改4
- 运行PPOCRLabel源码（paddleocr使用pip安装）,报错：  AttributeError: 'Namespace' object has no attribute 'return_word_box'
```python
    - File "G:\dongyongfei786\paddle\PaddleOCR\ppstructure\predict_system.py", line 82, in __init__
    self.return_word_box = args.return_word_box
AttributeError: 'Namespace' object has no attribute 'return_word_box'
```
- 原因:
  - paddleocr使用pip安装的源码中（paddleocr=2.7.0.3）, D:\ProgramData\Anaconda3\Lib\site-packages\paddleocr\tools\infer\utility.py, 缺少
  ```python
  # extended function
    parser.add_argument("--return_word_box", type=str2bool, default=False, help='Whether return the bbox of each word (split by space) or chinese character. Only used in ppstructure for layout recovery')
```