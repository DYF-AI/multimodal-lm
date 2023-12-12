import os
from PIL import Image
from flask import Flask, Response, request

app = Flask(__name__)
print(type(app), app)
print(app.root_path)


@app.route('/')
def index():
    return 'Hello world!'


@app.route('/img_flip', methods=["POST"])
def process_img():
    # 接收前端传来的图片  image定义上传图片的key
    upload_img = request.files['image']
    # 获取到图片的名字
    img_name = upload_img.filename
    # 把前端上传的图片保存到后端
    upload_img.save(os.path.join('./cache', upload_img.filename))
    # 对后端保存的图片进行镜像处理
    img_path = os.path.join('./cache', upload_img.filename)
    print('path', img_path)
    img = Image.open(img_path).convert("RGB")
    img_flip = img.transpose(Image.ROTATE_180)
    img_flip.save(os.path.join('./cache', 'res_' + upload_img.filename))
    # 把图片读成二进制，返回到前端
    image = open(os.path.join('./cache', 'res_' + upload_img.filename), mode='rb')
    response = Response(image, mimetype="image/jpeg")
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
