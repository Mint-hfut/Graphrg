import asyncio
from dataclasses import dataclass
import numpy as np
from nano_vectordb import MilvusDB
from ._utils import logger
from .base import (
    BaseVectorStorage,

)



@dataclass
class NanoVectorDBStorage(BaseVectorStorage):

    def __post_init__(self):
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._client = MilvusDB(
        global_config = self.global_config,
        namespace = self.namespace,
        embedding_dim = self.embedding_func.embedding_dim, 
        collection_name="vector_db",
        )
        

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items()},
            }
            for k, v in data.items()
        ]
        contents = [v["description"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]

        self._client.upsert(datas=list_data)


    async def query(self, user_id, topic_id, query: str, top_k=5,):
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        results = self._client.query(
            query=embedding, top_k=top_k, better_than_threshold=0.2,
            filter=f'user_id=="{user_id}" and topic_id=="{topic_id}"',
        )

        return results

    async def index_done_callback(self):
        self._client.save()