# env python 3.7
# coding=utf-8

import datetime
import hmac
import hashlib
import json
from urllib.parse import urlparse
import base64
import http.client


# 获取时间
def get_current_date():
    date = datetime.datetime.strftime(datetime.datetime.utcnow(), "%a, %d %b %Y %H:%M:%S GMT")
    return date


# 计算MD5+BASE64
def getcode(bodyData):
    temp = bodyData.encode("utf-8")
    md5 = hashlib.md5()
    md5.update(temp)
    md5str = md5.digest()  # 16位
    b64str = base64.b64encode(md5str)
    return b64str


# 计算 HMAC-SHA1
def to_sha1_base64(stringToSign, secret):
    hmacsha1 = hmac.new(secret.encode(), stringToSign.encode(), hashlib.sha1)
    return base64.b64encode(hmacsha1.digest())


def demo():
    # 初始参数设置
    ak_id = 'Your AccessKey ID'
    ak_secret = 'Your AccessKey Secret'
    datetime1 = get_current_date()  # 获取时间

    # 读取本地图片
    filenamePath = "/Users/FDUHYJ/PyProj/Aliyun/Face_Recognition/attribute/img/Blackpink.jpeg"  # 测试图片存放在项目目录下
    base64_data = ''
    with open(filenamePath, "rb") as f:
        base64_data = base64.b64encode(f.read())

    # 根据实际测试需要设置自己的图片URL
    options = {
        'url': 'https://dtplus-cn-shanghai.data.aliyuncs.com/face/attribute',
        'method': 'POST',
        'body': json.dumps({"type": "1", "content": base64_data.decode()}, separators=(',', ':')),
        'headers': {
            'accept': 'application/json',
            'content-type': 'application/json',
            'date':  datetime1
        }
    }

    body = ''       # 请求body参数
    if 'body' in options:
        body = options['body']

    bodymd5 = ''    # 计算获取请求body的md5值
    if not body == '':
        bodymd5 = getcode(body)
        bodymd5 = bodymd5.decode(encoding='utf-8')

    urlPath = urlparse(options['url'])
    restUrl = urlPath[1]
    path = urlPath[2]

    # 拼接请求签名字符串
    stringToSign = options['method'] + '\n' + options['headers']['accept'] + '\n' + bodymd5 + '\n' + options['headers']['content-type'] + '\n' + options['headers']['date'] + '\n' + path
    signature = to_sha1_base64(stringToSign, ak_secret)
    signature = signature.decode(encoding='utf-8')
    authHeader = 'Dataplus ' + ak_id + ':' + signature  # 组合认证Authorization

    # 请求Header
    headers = {
        # Request headers
        'Content-Type': options['headers']['content-type'],
        'Authorization': authHeader,
        'Date': datetime1,
        'Accept': options['headers']['accept']
    }

    try:
        # 设置http请求参数
        conn = http.client.HTTPSConnection(restUrl)
        conn.request(options['method'], path, body, headers)
        response = conn.getresponse()
        data = eval(response.read())
        print("Result: ", data)
        conn.close()
    except Exception as e:
        print("[Errno {0}] {1}".format(e.errno, e.strerror))


if __name__ == '__main__':
    demo()
