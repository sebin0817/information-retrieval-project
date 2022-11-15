import jsonlines
import random

class Ranker(object):

    def __init__(self):
        self.idx = None

    def index(self):
        self.idx = []
        with jsonlines.open('./data/livivo/documents/livivo_research_data.jsonl') as reader:
            for obj in reader:
                self.idx.append(obj['DBRECORDID'])
        print(len(self.idx))
        pass

    def rank_publications(self, query, page, rpp):

        itemlist = random.choices(self.idx, k=rpp)

        return {
            'page': page,
            'rpp': rpp,
            'query': query,
            'itemlist': itemlist,
            'num_found': len(itemlist)
        }


class Recommender(object):

    def __init__(self):
        self.idx = None

    def index(self):
        self.idx = []
        with jsonlines.open('./data/livivo/documents/livivo_research_data.jsonl') as reader:
            for obj in reader:
                self.idx.append(obj['DBRECORDID'])
        pass

    def recommend_datasets(self, item_id, page, rpp):

        itemlist = []

        return {
            'page': page,
            'rpp': rpp,
            'item_id': item_id,
            'itemlist': itemlist,
            'num_found': len(itemlist)
        }

    def recommend_publications(self, item_id, page, rpp):

        itemlist = random.choices(self.idx, k=rpp)

        return {
            'page': page,
            'rpp': rpp,
            'item_id': item_id,
            'itemlist': itemlist,
            'num_found': len(itemlist)
        }
