from core.config import config

import json
import redis


class RedisHelper:
    def __init__(self, host=config.REDIS_HOST, port=6379, db=0):
        self.redis = redis.Redis(host=host, port=port, db=db)

    def set(self, key, value):
        if isinstance(value, dict):
            v = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.redis.set(key, v)
        else:
            self.redis.set(key, value)

    def get(self, key):
        value = self.redis.get(key)
        if value:
            try:
                return json.loads(value.decode("utf-8"))
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON data: {e}")
                return value.decode("utf-8")
        return None

    def get_startswith(self, prefix: str):
        for key in self.redis.keys(f"{prefix}*"):
            return self.get(key)
        return None

    def get_endwith(self, suffix: str):
        for key in self.redis.keys("*"):
            if key.decode("utf-8").endswith(suffix):
                return self.get(key)
        return None

    def get_include(self, substring: str):
        for key in self.redis.keys("*"):
            if substring in key.decode("utf-8"):
                return self.get(key)
        return None

    def get_all(self):
        cache = {}
        for key in self.redis.scan_iter():
            cache[key.decode("utf-8")] = self.get(key)
        return cache

    def delete(self, key):
        self.redis.delete(key)

    def delete_startswith(self, value: str) -> None:
        for key in self.redis.scan_iter(f"{value}*"):
            self.redis.delete(key)

    def exists(self, key):
        return self.redis.exists(key)

    def keys(self, pattern="*"):
        return self.redis.keys(pattern)

    def values(self):
        return self.redis.values()

    def scan(self, cursor=0):
        return self.redis.scan(cursor)

    def expireat(self, key, timestamp):
        self.redis.expireat(key, timestamp)

    def ttl(self, key):
        return self.redis.ttl(key)


Cache = RedisHelper()
