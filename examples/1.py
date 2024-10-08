'''
  ​
  通过该程序下载的视频和音频是分成连个文件的，没有合成，
  视频为：视频名_video.mp4
  音频为：视频名_audio.mp4
  修改url的值，换成自己想下载的页面节课
  '''
# 导入requests模块，模拟发送请求
import requests
# 导入json
import json
# 导入re
import re
# 定义请求头
headers = {
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.5',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.84 Safari/537.36'
}


# 正则表达式，根据条件匹配出值
def my_match(text, pattern):
    match = re.search(pattern, text)
    print(match.group(1))
    print()
    return json.loads(match.group(1))


def download_video(old_video_url, video_url, audio_url, video_name):
    headers.update({"Referer": old_video_url})
    print("开始下载视频：%s" % video_name)
    video_content = requests.get(video_url, headers=headers)
    print('%s视频大小：' % video_name, video_content.headers['content-length'])
    audio_content = requests.get(audio_url, headers=headers)
    print('%s音频大小：' % video_name, audio_content.headers['content-length'])
    # 下载视频开始
    received_video = 0
    with open('%s_video.mp4' % video_name, 'ab') as output:
        while int(video_content.headers['content-length']) > received_video:
            headers['Range'] = 'bytes=' + str(received_video) + '-'
            response = requests.get(video_url, headers=headers)
            output.write(response.content)
            received_video += len(response.content)
    # 下载视频结束
    # 下载音频开始
    audio_content = requests.get(audio_url, headers=headers)
    received_audio = 0
    with open('%s_audio.mp4' % video_name, 'ab') as output:
        while int(audio_content.headers['content-length']) > received_audio:
            # 视频分片下载
            headers['Range'] = 'bytes=' + str(received_audio) + '-'
            response = requests.get(audio_url, headers=headers)
            output.write(response.content)
            received_audio += len(response.content)
    # 下载音频结束
    return video_name


if __name__ == '__main__':
    # 换成你要爬取的视频地址
    url = 'https://www.bilibili.com/video/av76609390'
    # 发送请求，拿回数据
    res = requests.get(url, headers=headers)
    # 视频详情json
    playinfo = my_match(res.text, '__playinfo__=(.*?)</script><script>')
    # 视频内容json
    initial_state = my_match(res.text, r'__INITIAL_STATE__=(.*?);\(function\(\)')
    # 视频分多种格式，直接取分辨率最高的视频 1080p
    video_url = playinfo['data']['dash']['video'][0]['baseUrl']
    # 取出音频地址
    audio_url = playinfo['data']['dash']['audio'][0]['baseUrl']
    video_name = initial_state['videoData']['title']
    print('视频名字为：video_name')
    print('视频地址为：', video_url)
    print('音频地址为：', audio_url)
    download_video(url, video_url, audio_url, video_name)
