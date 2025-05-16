# -*- coding: utf-8 -*-
from time import sleep
from threading import Thread
from queue import Queue
import concurrent.futures
from pymilvus import connections, Collection, utility
from tqdm import tqdm
# from extract_embedding import get_embedding
import os
import json
from collections import defaultdict
from argparse import ArgumentParser
from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel,FlagReranker
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from flask import Flask
from flask import request
import numpy as np

model = BGEM3FlagModel('./model/bgem3_merge', use_fp16=False)
connections.connect("default", host="localhost", port="19530", user='root', password='Milvus')
reranker = FlagReranker('./model/rerank_ft_0519merge73', use_fp16=True)


def compute_similarities(query, passages):
    """计算query与每个passage的相似度"""
    query_passage_pairs = [[query, passage] for passage in passages]
    scores = reranker.compute_score(query_passage_pairs)
    scores_normalized = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    return scores_normalized





def parse_data():
    data_list = []
    with open('指标.txt', 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if '\t' in line:
                index_name, index_definition = line.split('\t', 1)
                data = {
                    '标题': "中国石墨集团有限公司2024年报",
                    '公司名': "中国石墨集团有限公司",
                    '文件名': "9215b919f4-4c7cc6d586c50dac65267a8b2d804730",
                    '指标': index_name,
                    '释义': index_definition,
                    'qw': "其他字段"
                }

                data_list.append(data)
    return data_list

def call_reranker(datas):
    res_send = []  # 初始化默认值
    try:
        file_groups = defaultdict(list)
        for item in datas:
            file_groups["文件名"].append(item)
        grouped_data = list(file_groups.values())
        res_send = process(grouped_data)
    except Exception as e:
        print(f"Error: {e}")
        res_send = {"error": str(e)}
    return json.dumps(res_send, ensure_ascii=False, default=lambda o: o.__dict__)  # 支持对象序列化



def process(grouped_data):
    all_results = []
    for idx in range(len(grouped_data)):
        collection_name = '_' + str(grouped_data[idx][0]["文件名"]).replace('-', '_')
        print(f"正在处理集合: {collection_name}")
        collection = Collection(collection_name)
        collection.load()
        for i in grouped_data[idx]:
            query = i['指标']+i['释义']
            # print(query)
            emb = model.encode(query)['dense_vecs']
            topk = 5
            """输入问题，AI数据库中返回topk个相近的答案，过滤标题"""
            search_param = {
                "data": [emb],  # 查询的向量数据
                "anns_field": "embedding",
                "output_fields": ["text", 'title', 'page'],
                "param": {"metric_type": "IP", "params": {"nprobe": 64}},
                "limit": topk,
                "expr": 'text != "无文本"'  # 过滤条件
            }
            results = collection.search(**search_param)
            search_results = []
            for hits in results:
                for hit in hits:
                    score = round(hit.distance, 3)
                    if score >= 0.2:
                        search_results.append(
                            {"score": score, "text": hit.entity.get('text'), "title": hit.entity.get('title'),
                             "page": hit.entity.get('page')})
            passages = [item['text'] for item in search_results]
            if '释义' in i.keys():
                query = i["指标"] + i["释义"]
            else:
                query = i["指标"]
            scores = compute_similarities(query, passages)
            for score, passage_data in zip(scores, search_results):
                passage_data['score_re'] = score
            search_results.sort(key=lambda x: x['score_re'], reverse=True)
            i['ser_res'] = search_results[:5]
            all_results.append(i)
        collection.release()
        print(f"完成处理集合: {collection_name}")
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
    return all_results




if __name__ == '__main__':
    print("Starting Run...")
    datas=parse_data()
    res = call_reranker(datas)
    print(res)


