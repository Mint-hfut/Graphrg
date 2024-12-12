import os
import logging
import json
from openai import AsyncOpenAI
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs
from sentence_transformers import SentenceTransformer
import time
from loguru import logger
import base64
import threading
from flask import Flask, request,stream_with_context, Response
from nano_graphrag.graphrag import process_to_store
logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

API_KEY = "token-abc123"
os.environ["TIKTOKEN_CACHE_DIR"]="./"
EMBED_MODEL = SentenceTransformer("./bge-large-zh-v1.5")


@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
    max_token_size=EMBED_MODEL.max_seq_length,
)
async def local_embedding(texts: list[str]):
    return EMBED_MODEL.encode(texts, normalize_embeddings=True)


async def deepseepk_model_if_cache(
        prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(
        api_key=API_KEY, base_url="http://36.137.80.249:38080/ali_Qwen2_7B"
    )
    messages = []

    cache_type = kwargs.pop("cache_type", None)
    prompt_to_cache = kwargs.pop("prompt_to_cache", None)

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    if hashing_kv is not None:
        args_hash = compute_args_hash(os.getenv("MODEL"), messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash, fields=['return','model'])
        if if_cache_return is not None:
            return if_cache_return[0]["return"]



    print("开始推理...")
    if 'max_tokens' not in kwargs:
        kwargs['max_tokens'] = 2000
    if 'response_format' in kwargs:
        kwargs.pop('response_format')
    response = await openai_async_client.chat.completions.create(
            model=os.getenv("MODEL"), messages=messages, **kwargs
        )


    
    print("*" * 200)
    print("输入：\n", messages)
    print("*" * 200)
    print("输出：\n", response)
    print("*" * 200)
    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert_async(
            process_to_store({args_hash: {"return": response.choices[0].message.content, "model": os.getenv("MODEL")}})
        )
    # -----------------------------------------------------
    return response.choices[0].message.content


async def local_llm(
        prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await deepseepk_model_if_cache(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads/'

@app.route('/graphrag', methods=['POST'])
def getResult():
    content = request.data.decode('utf-8')
    content_json = json.loads(content)
    mode = content_json["mode"]
    user_id = content_json["user_id"]
    topic_id = content_json["topic_id"]

    if mode == "index":
        dir_name, full_file_name = os.path.split(content_json["file_paths"])
        inputs = content_json["file_content"]
        input1 = base64.b64decode(inputs)
        if full_file_name.endswith("txt") or full_file_name.endswith("md"):
            text = input1.decode('utf-8')
        else:
            return (json.dumps("请上传.pdf文件或.txt文件", ensure_ascii=False))


        insert_thread = threading.Thread(target=rag.insert, args=(text, user_id,topic_id))
        insert_thread.start()
        @stream_with_context
        def generate():
            # 当线程在运行时，持续返回 "indexing..."
            while insert_thread.is_alive():
                yield json.dumps("正在创建文档图谱及社区报告，请稍等。。。", ensure_ascii=False) + "\n"
                time.sleep(5)  # 每2秒返回一次状态更新
            # 当线程完成时，返回成功消息
            yield json.dumps("文档建图成功，可选择相关问题测试。", ensure_ascii=False) + "\n"
        return Response(generate(), content_type='application/json')


    else:

        input_text = content_json["user_input"]
        response = rag.query(input_text, param=QueryParam(mode="local"), user_id=user_id, topic_id=topic_id)
        logger.info(f" response:{response}")

        return (json.dumps(response.replace("\n", ""), ensure_ascii=False))


if __name__ == '__main__':
    os.environ['MODEL'] = "ali_Qwen2_7B"
    rag = GraphRAG(
        best_model_func=local_llm,
        cheap_model_func=local_llm,
        embedding_func=local_embedding,
    )
    app.run(host='0.0.0.0', port=8084, threaded=True)
