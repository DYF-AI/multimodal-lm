
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

#### 打包成exe

```shell
(paddle) G:\dongyongfei786\multimodal-lm\third_party\ppocrlabel-pil>pyinstaller --onefile PPOCRLabel.py
```

bug1:
报错:
```shell
(paddle) G:\dongyongfei786\multimodal-lm\third_party\ppocrlabel-pil\dist>PPOCRLabel.exe
Traceback (most recent call last):
  File "paddle\fluid\ir.py", line 24, in <module>
  File "PyInstaller\loader\pyimod02_importers.py", line 419, in exec_module
  File "paddle\fluid\proto\pass_desc_pb2.py", line 16, in <module>
ModuleNotFoundError: No module named 'framework_pb2
```

解决:
```python
# 安装存在问题, 有些文件夹前面是~， 类似下面的警告， 重装paddle
# WARNING: Ignoring invalid distribution -pocrlabel (d:\programdata\anaconda3\envs\paddle\lib\site-packages)
D:\ProgramData\Anaconda3\envs\paddle\Lib\site-packages\paddle\fluid\~roto


# 修改：D:\ProgramData\Anaconda3\envs\paddle\Lib\site-packages\paddle\fluid\proto\pass_desc_pb2.py, 可解决bug1
# import framework_pb2 as framework__pb2
import paddle.fluid.proto.framework_pb2 as framework__pb2
```

bug2：
报错:
```shell
(paddle) G:\dongyongfei786\multimodal-lm\third_party\ppocrlabel-pil\dist>PPOCRLabel.exe
Traceback (most recent call last):
  File "ppocrlabel-pil\PPOCRLabel.py", line 41, in <module>
    from paddleocr import PaddleOCR, PPStructure
  File "<frozen importlib._bootstrap>", line 991, in _find_and_load
  File "<frozen importlib._bootstrap>", line 975, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 671, in _load_unlocked
  File "PyInstaller\loader\pyimod02_importers.py", line 419, in exec_module
  File "paddleocr\__init__.py", line 14, in <module>
  File "<frozen importlib._bootstrap>", line 991, in _find_and_load
  File "<frozen importlib._bootstrap>", line 975, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 671, in _load_unlocked
  File "PyInstaller\loader\pyimod02_importers.py", line 419, in exec_module
  File "paddleocr\paddleocr.py", line 21, in <module>
  File "<frozen importlib._bootstrap>", line 991, in _find_and_load
  File "<frozen importlib._bootstrap>", line 975, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 671, in _load_unlocked
  File "PyInstaller\loader\pyimod02_importers.py", line 419, in exec_module
  File "paddle\__init__.py", line 71, in <module>
  File "<frozen importlib._bootstrap>", line 991, in _find_and_load
  File "<frozen importlib._bootstrap>", line 975, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 671, in _load_unlocked
  File "PyInstaller\loader\pyimod02_importers.py", line 419, in exec_module
  File "paddle\dataset\__init__.py", line 27, in <module>
  File "<frozen importlib._bootstrap>", line 991, in _find_and_load
  File "<frozen importlib._bootstrap>", line 975, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 671, in _load_unlocked
  File "PyInstaller\loader\pyimod02_importers.py", line 419, in exec_module
  File "paddle\dataset\flowers.py", line 39, in <module>
  File "<frozen importlib._bootstrap>", line 991, in _find_and_load
  File "<frozen importlib._bootstrap>", line 975, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 671, in _load_unlocked
  File "PyInstaller\loader\pyimod02_importers.py", line 419, in exec_module
  File "paddle\dataset\image.py", line 48, in <module>
  File "subprocess.py", line 858, in __init__
  File "subprocess.py", line 1311, in _execute_child
FileNotFoundError: [WinError 2] 系统找不到指定的文件。
[35992] Failed to execute script 'PPOCRLabel' due to unhandled exception
```

解决:
https://github.com/PaddlePaddle/PaddleOCR/issues/5326
```python
# FIXME(minqiyang): this is an ugly fix for the numpy bug reported here
# https://github.com/numpy/numpy/issues/12497
if six.PY3:
    import subprocess
    import sys
    import os
    interpreter = sys.executable
    # Note(zhouwei): if use Python/C 'PyRun_SimpleString', 'sys.executable'
    # will be the C++ execubable on Windows
    if sys.platform == 'win32' and 'python.exe' not in interpreter:
        interpreter = sys.exec_prefix + os.sep + 'python.exe'
    import_cv2_proc = subprocess.Popen(
        [interpreter, "-c", "import cv2"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    out, err = import_cv2_proc.communicate()
    retcode = import_cv2_proc.poll()
    if retcode != 0:
        cv2 = None
    else:
        import cv2
else:
    try:
        import cv2
    except ImportError:
        cv2 = None

# 改为
try:
    import cv2
except:
    cv2 = None
```


bug3:
```shell
(paddle) G:\dongyongfei786\multimodal-lm\third_party\ppocrlabel-pil\dist>PPOCRLabel.exe
Traceback (most recent call last):
  File "ppocrlabel-pil\PPOCRLabel.py", line 41, in <module>
    from paddleocr import PaddleOCR, PPStructure
  File "<frozen importlib._bootstrap>", line 991, in _find_and_load
  File "<frozen importlib._bootstrap>", line 975, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 671, in _load_unlocked
  File "PyInstaller\loader\pyimod02_importers.py", line 419, in exec_module
  File "paddleocr\__init__.py", line 14, in <module>
  File "<frozen importlib._bootstrap>", line 991, in _find_and_load
  File "<frozen importlib._bootstrap>", line 975, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 671, in _load_unlocked
  File "PyInstaller\loader\pyimod02_importers.py", line 419, in exec_module
  File "paddleocr\paddleocr.py", line 43, in <module>
  File "paddleocr\paddleocr.py", line 37, in _import_file
  File "<frozen importlib._bootstrap_external>", line 839, in exec_module
  File "<frozen importlib._bootstrap_external>", line 975, in get_code
  File "<frozen importlib._bootstrap_external>", line 1032, in get_data
FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Users\\21702\\AppData\\Local\\Temp\\_MEI271362\\paddleocr\\tools/__init__.py'
[7336] Failed to execute script 'PPOCRLabel' due to unhandled exception!
```
解决:
```shell
# --collect-all  paddleocr 不再包paddleocr的错
pyinstaller --onefile --collect-all  paddleocr PPOCRLabel.py

# 尝试-F
pyinstaller -F PPOCRLabel.py

# --collect-all 可以通过加 ，但治标不治本
pyinstaller --onefile --collect-all paddleocr --collect-all pyclipper   --collect-all imghdr PPOCRLabel.py

# https://blog.csdn.net/weixin_44458631/article/details/115290619
# https://blog.csdn.net/chang1976272446/article/details/119824048

(paddle) G:\dongyongfei786\multimodal-lm\third_party\ppocrlabel-pil>
pyinstaller --onefile --collect-all paddleocr --collect-all pyclipper  --collect-all imghdr --collect-all skimage --collect-all imgaug --hidden-import scipy.io --collect-all lmdb --hidden-import PyQt5.Qt --hidden-import  PyQt5.QtCore --hidden-import PyQt5.QtGui --hidden-import PyQt5.QtWidgets PPOCRLabel.py

```

bug4:
```shell
(paddle) G:\dongyongfei786\multimodal-lm\third_party\ppocrlabel-pil\dist>PPOCRLabel.exe
Traceback (most recent call last):
  File "ppocrlabel-pil\PPOCRLabel.py", line 28, in <module>
ImportError: DLL load failed while importing QtCore: 找不到指定的程序。
[35384] Failed to execute script 'PPOCRLabel' due to unhandled exception!
```
解决：
https://zhuanlan.zhihu.com/p/524637688   pyqt5和pyqt5-qt5两个版本不一致
安装：pip install PyQt5-tools==5.15.2.3
依旧不行