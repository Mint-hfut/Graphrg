import base64
import requests
import json
import time
import os
import pandas as pd

url="http://127.0.0.1:8084/graphrag"


files = ['/root/work/GC/nanorag/清北学霸.md']

questions = ['这篇文章讲了什么？']

def test(mode, user_id, topic_id, **kwargs):
    proxies = {"http":"","https":"", }
    if mode == "index":
        filepath = kwargs['file']
        f_open = open(filepath, 'rb')
        file_b64 = base64.b64encode(f_open.read())
        f_open.close()
        data = {"file_content": file_b64.decode(), "file_paths": filepath, 'mode': mode, 'user_id': user_id, 'topic_id': topic_id}
        data = json.dumps(data)
        with requests.post(url, data=data, proxies=proxies, stream=True) as response:
            for line in response.iter_lines():
                if line:
                    print(line.decode('utf-8'))
        # response = requests.post(url, data=data, proxies=proxies)
        # print(response.json())
    if mode == "query":
        text_input = kwargs['question']
        # text = text_input.encode('utf-8')
        data = {'user_input': text_input, 'mode': mode, 'user_id': user_id, 'topic_id': topic_id}
        response = requests.post(url, data=json.dumps(data), proxies=proxies)
        print(response)
        print(response.json())

if __name__=="__main__":
    
    for file in files:
        test("index", file=file, user_id='gc2', topic_id='topic1')
    for question in questions:
        test("query", user_id='gc2', topic_id='topic1', question=question)

