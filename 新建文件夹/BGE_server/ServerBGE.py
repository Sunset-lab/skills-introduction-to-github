import os
import re
from bs4 import BeautifulSoup
import json
from argparse import ArgumentParser
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from pymilvus import connections, Collection
import json
import numpy as np
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel
from flask import Flask
from flask import request


model = BGEM3FlagModel('./model/bgem3_merge', use_fp16=False)
connections.connect("default", host="localhost", port="19530", user='root', password='Milvus')

AA
def create_database(html_text, guid, start_page, end_page):
    # 读取HTML文件
    try:
        html_content = html_text
        soup = BeautifulSoup(html_content, 'html.parser')

        results = []
        titles_with_pages = set()

        # 第一步：遍历文档，收集段落信息
        current_page = 0
        current_title = ''
        for element in soup.find_all(True):
            if element.name == 'page':
                current_page = int(element.text.split('第')[1].split('页')[0].replace('复杂', ''))
                if (start_page != 0 or end_page != 0) and (
                        current_page + 1 < start_page or current_page + 1 > end_page):
                    continue
            elif element.name == 'h1':
                if (start_page != 0 or end_page != 0) and (
                        current_page + 1 < start_page or current_page + 1 > end_page):
                    continue
                current_title = element.text.strip()
                # 删除标题中的制表符
                current_title = current_title.replace('\t', ' ')

            elif element.name == 'p':
                if (start_page != 0 or end_page != 0) and (
                        current_page + 1 < start_page or current_page + 1 > end_page):
                    continue
                paragraph_text = element.text.strip()
                # 删除\t
                paragraph_text = paragraph_text.replace('\t', ' ')
                paragraph_text = paragraph_text.replace(' ', '')

                if paragraph_text:
                    titles_with_pages.add((current_title, current_page))
                    results.append({
                        'title': current_title,
                        'paragraph': paragraph_text,
                        'page': current_page + 1
                    })

        # 第二步：重新检查每个标题，确认所有标题都被记录
        current_page = 0
        for element in soup.find_all(True):

            if element.name == 'page':
                current_page = int(element.text.split('第')[1].split('页')[0].replace('复杂', ''))
                if (start_page != 0 or end_page != 0) and (
                        current_page + 1 < start_page or current_page + 1 > end_page):
                    continue
            elif element.name == 'h1':
                if (start_page != 0 or end_page != 0) and (
                        current_page + 1 < start_page or current_page + 1 > end_page):
                    continue
                title = element.text.strip()
                # 删除标题中的制表符
                title = title.replace('\t', ' ')

                title_page_combo = (title, current_page)
                results.append({
                    'title': title,
                    'paragraph': '无文本',
                    'page': current_page + 1
                })

        collection_name = '_' + guid.replace('-', '_')
        print("开始创建数据库：", collection_name)
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
        fields = [
            FieldSchema(name="index", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="page", dtype=DataType.INT64)
        ]
        schema = CollectionSchema(fields, collection_name)
        collection = Collection(collection_name, schema)
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "IP",
            "params": {"nlist": 1024},
        }
        collection.create_index(field_name="embedding", index_params=index_params)

        collection.load()
        print("success Create collection: ", collection_name)
        print("开始插入数据：")
        data = results
        title = []
        page = []
        text = []
        embedding = []
        for i in tqdm(data):
            page.append(i['page'])
            text.append(i['paragraph'])
            title.append(i['title'])

            if i['paragraph'] == '无文本':
                sentence_embeddings_norm = model.encode(i['title'])['dense_vecs']
            else:
                sentence_embeddings_norm = model.encode(i['paragraph'])['dense_vecs']
            embedding.append(sentence_embeddings_norm)
        mr = collection.insert([embedding, text, title, page])
        collection.flush()
        print('成功提取特征并存放ai数据库', mr.succ_count)
        collection.release()
    except ValueError as e:
        print("错误：", e)



# app = Flask(__name__)


# @app.route('/', methods=['POST'])
# def login():
#     # 直接取值
#     msg = request.get_data()
#     try:
#         # print(msg.decode('utf-8'))
#         msgstd = json.loads(msg.decode('utf-8'))
#         if "start_page" in msgstd:
#             start_page = msgstd["start_page"]
#         else:
#             start_page = 0
#         if "end_page" in msgstd:
#             end_page = msgstd["end_page"]
#         else:
#             end_page = 0
#         create_database(msgstd["text"], msgstd["guid"], start_page, end_page)
#         res_send = 1
#     except Exception as e:
#         res_send = e
#     res = str(res_send)
#     return res



def login():
    # 读取 JSON 文件并直接解析为字典
    with open('data.txt', 'r', encoding='utf-8') as file:
        msg = json.load(file) 

    try:
        # 直接从字典中取值
        start_page = msg.get("start_page", 0)  # 使用 get 避免 KeyError
        end_page = msg.get("end_page", 0)
        create_database(msg["text"], msg["guid"], start_page, end_page)
        res_send = 1
    except Exception as e:
        print("错误：", e)
        res_send = e
    res = str(res_send)
    return res




if __name__ == '__main__':
    print("Starting BGE flask server in python...")
    res=login()
    print(res)
    # multiprocessing.set_start_method('spawn')

    # app.run(host="0.0.0.0", port=8867, threaded=True,debug=False)


# import os
# import sys
# import random
# from pymilvus import connections, Collection, FieldSchema, DataType, CollectionSchema
#
# # 修复路径问题
# if __name__ == "__main__":
#     sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#     os.chdir(os.path.dirname(os.path.abspath(__file__)))
#
# # 连接Milvus
# connections.connect(host="localhost", port="19530")
#
# # 定义集合结构
# fields = [
#     FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
#     FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128)
# ]
# schema = CollectionSchema(fields, description="test collection")
# collection = Collection("test_collection", schema)
#
# # 插入数据
# data = [
#     [i for i in range(100)],  # IDs
#     [[random.random() for _ in range(128)] for _ in range(100)]  # Vectors
# ]
# collection.insert(data)
# print("插入成功")
