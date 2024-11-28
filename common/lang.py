from enum import Enum


class LangType(Enum):
    en_us = 1
    zh_cn = 2

class Lang:
    def __init__(self,lang_type = LangType.zh_cn,lang_map = {}) -> None:
        self.lang_type = lang_type
        self.map(lang_map)
    def map(self,lang_map):
        self.lang_map = lang_map
    def add_map(self,lang_type,config):
        if self.lang_map.get(lang_type,None) == None:
            self.lang_map[lang_type] = config
            return self
        for key in config:
            self.lang_map[lang_type][key] = config[key]
        return self
    def t(self,key):
        return self.lang_map[self.lang_type].get(key,key.replace('_', ' '))