FROM milvusdb/milvus:v2.3.8-gpu
RUN mkdir workspace
COPY / /workspace/BGE_server
WORKDIR /workspace/BGE_server
RUN pip install python==3.10.14 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -U FlagEmbedding -i https://pypi.tuna.tsinghua.edu.cn/simple 
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
ENTRYPOINT ["python"]
CMD ["ServerBGE.py"]