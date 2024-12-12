import os
import json
import base64
import hashlib
import numpy as np
from typing import TypedDict, Literal, Union
from dataclasses import dataclass, asdict
from logging import getLogger
import logging
import asyncio
from time import time
from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType
from nano_graphrag.base import BaseKVStorage
f_ID = "__id__"
f_entity_name = "entity_name"
f_VECTOR = "__vector__"
f_METRICS = "__metrics__"
Data = TypedDict("Data", {f_ID: str, f_VECTOR: np.ndarray})
DataBase = TypedDict(
    "DataBase", {"embedding_dim": int, "data": list[Data], "matrix": np.ndarray}
)
Float = np.float32
logger = getLogger("nano-vectordb")

logging.basicConfig(level=logging.INFO)


def array_to_buffer_string(array: np.ndarray) -> str:
    return base64.b64encode(array.tobytes()).decode()


def buffer_string_to_array(base64_str: str, dtype=Float) -> np.ndarray:
    return np.frombuffer(base64.b64decode(base64_str), dtype=dtype)


def load_storage(file_name) -> Union[DataBase, None]:
    if not os.path.exists(file_name):
        return None
    with open(file_name) as f:
        data = json.load(f)
    data["matrix"] = buffer_string_to_array(data["matrix"]).reshape(
        -1, data["embedding_dim"]
    )
    logger.info(f"Load {data['matrix'].shape} data")
    return data


def hash_ndarray(a: np.ndarray) -> str:
    return hashlib.md5(a.tobytes()).hexdigest()


def normalize(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


@dataclass
class NanoVectorDB:
    embedding_dim: int
    metric: Literal["cosine"] = "cosine"
    storage_file: str = "nano-vectordb.json"

    def pre_process(self):
        if self.metric == "cosine":
            self.__storage["matrix"] = normalize(self.__storage["matrix"])

    def __post_init__(self):
        default_storage = {
            "embedding_dim": self.embedding_dim,
            "data": [],
            "matrix": np.array([], dtype=Float).reshape(0, self.embedding_dim),
        }
        storage: DataBase = load_storage(self.storage_file) or default_storage
        assert (
            storage["embedding_dim"] == self.embedding_dim
        ), f"Embedding dim mismatch, expected: {self.embedding_dim}, but loaded: {storage['embedding_dim']}"
        self.__storage = storage
        self.usable_metrics = {
            "cosine": self._cosine_query,
        }
        assert self.metric in self.usable_metrics, f"Metric {self.metric} not supported"
        self.pre_process()
        logger.info(f"Init {asdict(self)} {len(self.__storage['data'])} data")

    def upsert(self, datas: list[Data]):
        _index_datas = {
            data.get(f_ID, hash_ndarray(data[f_VECTOR])): data for data in datas
        }
        if self.metric == "cosine":
            for v in _index_datas.values():
                v[f_VECTOR] = normalize(v[f_VECTOR])
        report_return = {"update": [], "insert": []}
        for i, already_data in enumerate(self.__storage["data"]):
            if already_data[f_ID] in _index_datas:
                update_d = _index_datas.pop(already_data[f_ID])
                self.__storage["matrix"][i] = update_d[f_VECTOR].astype(Float)
                del update_d[f_VECTOR]
                self.__storage["data"][i] = update_d
                report_return["update"].append(already_data[f_ID])
        if len(_index_datas) == 0:
            return report_return
        report_return["insert"].extend(list(_index_datas.keys()))
        new_matrix = np.array(
            [data[f_VECTOR] for data in _index_datas.values()], dtype=Float
        )
        new_datas = []
        for new_k, new_d in _index_datas.items():
            del new_d[f_VECTOR]
            new_d[f_ID] = new_k
            new_datas.append(new_d)
        self.__storage["data"].extend(new_datas)
        self.__storage["matrix"] = np.vstack([self.__storage["matrix"], new_matrix])
        return report_return

    def get(self, ids: list[str]):
        return [data for data in self.__storage["data"] if data[f_ID] in ids]

    def delete(self, ids: list[str]):
        ids = set(ids)
        left_data = []
        delete_index = []
        for i, data in enumerate(self.__storage["data"]):
            if data["__id__"] in ids:
                delete_index.append(i)
                ids.remove(data["__id__"])
                if len(ids) == 0:
                    break
            else:
                left_data.append(data)
        self.__storage["data"] = left_data
        self.__storage["matrix"] = np.delete(
            self.__storage["matrix"], delete_index, axis=0
        )

    def save(self):
        storage = {
            **self.__storage,
            "matrix": array_to_buffer_string(self.__storage["matrix"]),
        }
        with open(self.storage_file, "w") as f:
            json.dump(storage, f, ensure_ascii=False)

    def query(
        self, query: np.ndarray, top_k: int = 10, better_than_threshold: float = None
    ):
        return self.usable_metrics[self.metric](query, top_k, better_than_threshold)

    def _cosine_query(
        self, query: np.ndarray, top_k: int, better_than_threshold: float
    ):
        query = normalize(query)
        scores = np.dot(self.__storage["matrix"], query)
        sort_index = np.argsort(scores)[-top_k:]
        sort_index = sort_index[::-1]
        results = []
        for i in sort_index:
            if better_than_threshold is not None and scores[i] < better_than_threshold:
                break
            results.append({**self.__storage["data"][i], f_METRICS: scores[i]})
        return results




@dataclass
class MilvusDB(BaseKVStorage):
    embedding_dim: int = 768
    metric: str = "COSINE"
    milvus_uri: str = "Milvus.db"  # Default Milvus URI
    milvus_user: str = ""  # Default user
    milvus_password: str = ""  # Default password
    collection_name: str = "nano_vectordb_collection"
    is_vector_db: bool = True
    user_id: str = "user_id"
    topic_id: str = "topic_id"
    def __post_init__(self):
        self.client = MilvusClient(
            uri=self.milvus_uri,
            user=self.milvus_user,
            password=self.milvus_password
        )

        self._create_collection_if_not_exists()
        logger.info(f"Init {asdict(self)}")
    
    def _create_collection_if_not_exists(self):
        if self.collection_name not in self.client.list_collections():
            if self.is_vector_db:
                self.create_vector_collection()
            else:
                self.create_json_collection()
    def create_json_collection(self):
        schema = self.client.create_schema(
                    auto_id=False,
                    enable_dynamic_field=True,
                )
        schema.add_field(field_name=f_ID, datatype=DataType.VARCHAR, is_primary=True,max_length=225)
        self.client.create_collection(collection_name=self.collection_name, schema=schema)
    def create_vector_collection(self):
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True,)
        schema.add_field(field_name=f_ID, datatype=DataType.VARCHAR, is_primary=True,max_length=225)
        schema.add_field(field_name=self.user_id, datatype=DataType.VARCHAR, max_length=225)
        schema.add_field(field_name=self.topic_id, datatype=DataType.VARCHAR, max_length=225)
        schema.add_field(field_name=f_entity_name, datatype=DataType.VARCHAR, max_length=225)
        schema.add_field(field_name=f_VECTOR, datatype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        self.client.create_collection(collection_name=self.collection_name, schema=schema)


        index_params = self.client.prepare_index_params()

        # 4.2. Add an index on the vector field.
        index_params.add_index(
            field_name=f_VECTOR,
            metric_type=self.metric,
            index_type="AUTOINDEX",
            index_name="vector_index",
            params={ "nlist": 128 }
        )

        # 4.3. Create an index file
        self.client.create_index(
            collection_name=self.collection_name,
            index_params=index_params,
            sync=False # Whether to wait for index creation to complete before returning. Defaults to True.
        )
    async def index_done_callback(self):
        pass
    async def all_keys(self) -> list[str]:
        result = await asyncio.to_thread(self.client.query,collection_name=self.collection_name,
                                         filter="", output_fields=["id"])
        return [str(item['id']) for item in result]

    async def get_by_ids(self, ids: list, fields: list = None):
        # 如果没有指定字段，默认返回所有字段
        if fields is None:
            result = await asyncio.to_thread(self.client.query, collection_name=self.collection_name,
                                             ids=[ids],)
            return result
        else:
            # 根据字段列表过滤查询结果
            result = await asyncio.to_thread(self.client.query,collection_name=self.collection_name,
                                            ids=[ids], output_fields=fields
                                        )
            return result
    async def get_by_id(self, ids: list, fields: list = None):
        result = await self.get_by_ids(ids=ids, fields=fields)
        return None if len(result)==0 else result
    async def filter_keys(self, data: list[str]) -> set:
        # 获取集合中所有的ID

        result = await asyncio.to_thread(self.client.query, collection_name=self.collection_name,
                                      ids=[idx for idx in data], output_fields=["id"])
                                    # 将查询结果中的ID提取到一个集合中
        existing_ids = set(str(item[f_ID]) for item in result)
        
        # 过滤出不在Milvus集合中的ID
        return set([s for s in data if s not in existing_ids])
    async def upsert_async(self, datas: list[Data]):

        await asyncio.to_thread(self.client.insert,
            collection_name=self.collection_name,
            data=datas
        )
        logger.info(f"Upserted {len(datas)} data points.")
        return 0
    def upsert(self, datas: list[Data]):

        self.client.insert(
            collection_name=self.collection_name,
            data=datas
        )
        logger.info(f"Upserted {len(datas)} data points.")
    def save(self):
        pass

    def get(self, ids: list[str]):
        if not ids:
            logger.warning("No ids provided for retrieval.")
            return []

        quoted_ids = [f"'{id}'" for id in ids]
        filter_expr = f"{f_ID} in [{', '.join(quoted_ids)}]"
        res = self.client.query(
            collection_name=self.collection_name,
            expr=filter_expr,
            output_fields=[f_ID, f_VECTOR]
        )
        results = []
        for entity in res:
            vector = np.array(entity[f_VECTOR], dtype=Float)
            results.append({
                f_ID: entity[f_ID],
                f_VECTOR: vector
            })
        return results


    def query(
        self, query: np.ndarray, top_k: int = 10, better_than_threshold: float = None,filter=None
    ):
        # 确保查询向量是归一化的
        query = query / np.linalg.norm(query)
        
        res = self.client.search(
            collection_name=self.collection_name,
            data=[query.tolist()],
            anns_field=f_VECTOR,
            limit=top_k,
            # output_fields=[f_ID, f_VECTOR, f_entity_name],
            output_fields=["entity_name","description","source", "target","distance","entity_type","chunk_id"],
            filter=filter,
        )
        
        results = []
        for hit in res[0]:
            if better_than_threshold is not None and hit["distance"] < better_than_threshold:
                break
            results.append({
                "entity_name": hit["entity"].get("entity_name"),
                # f_VECTOR: np.array(hit["entity"].get(f_VECTOR), dtype=np.float32),
                "description": hit["entity"].get("description"),
                "source": hit["entity"].get("source"),
                "target": hit["entity"].get("target"),
                "entity_type": hit["entity"].get("entity_type"),
                "chunk_id": hit["entity"].get("chunk_id"),
            })
        return results
    async def drop(self):
    # 使用expr="*"来删除所有数据
        await asyncio.to_thread(self.client.delete, collection_name=self.collection_name)
