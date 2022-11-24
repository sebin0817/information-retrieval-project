import jsonlines
import random
import unicodedata
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.tokenize import sent_tokenize , word_tokenize
import pandas as pd

class Ranker(object):

    def __init__(self):
        self.idx = None

    def index(self):
        self.idx = []
        with jsonlines.open('./data/livivo/documents/livivo_research_data.jsonl') as reader:
            for obj in reader:
                # self.idx.append(obj['ABSTRACT'])
                doc = pd.Series(obj.get('ABSTRACT'))
                doc = doc.apply(lambda x: self.strip_accents(x))
                tokenized = (doc.apply(lambda x: self.tokenize_text(x)))
                inverted_doc_indexes = {}
                files_with_index = []
                files_with_tokens = {}
                for i in tokenized.index:
                    #Clean and Tokenize text of each document
                    words = tokenized[i]
                    #Store tokens
                    files_with_tokens[i] = words

                    doc_index = self.inverted_index(words)
                    self.inverted_index_add(inverted_doc_indexes, i, doc_index)
                    files_with_index.append(i)

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

    def inverted_index(self, words):
        """Create a inverted index of words (tokens or terms) from a list of terms

        Parameters:
        words (list of str): tokenized document text

        Returns:
        Inverted index of document (dict)

        """       
        inverted = {}
        for index, word in enumerate(words):
            locations = inverted.setdefault(word, [])
            locations.append(index)
        return inverted

    def inverted_index_add(self, inverted, doc_id, doc_index):
        """Insert document id into Inverted Index

        Parameters:
        inverted (dict): Inverted Index
        doc_id (int): Id of document been added
        doc_index (dict): Inverted Index of a specific document.

        Returns:
        Inverted index of document (dict)

        """        
        for word in doc_index.keys():
            locations = doc_index[word]
            indices = inverted.setdefault(word, {})
            indices[doc_id] = locations
        return inverted

    def strip_accents(self, text):   
        nfkd = unicodedata.normalize('NFKD', text)
        newText = u"".join([c for c in nfkd if not unicodedata.combining(c)])
        return re.sub('[^a-zA-Z0-9 \\\']', ' ', newText)

    def tokenize_text(self, text):
        """Make all necessary preprocessing of text: strip accents and punctuation, remove the words only contains digit
        remove \n, tokenize our text, convert to lower case, remove stop words and 
        words with less than 2 chars.

        Parameters:
        text (str): Input text

        Returns:
        str: cleaned tokenized text

        """    
        # nltk.download('stopwords')
        WORD_MIN_LENGTH = 2
        STOP_WORDS = stopwords.words('english')
        text = self.strip_accents(text)
        text = re.sub(re.compile('\n'),' ',text)
        words = word_tokenize(text)
        words = [word.lower() for word in words]
        words = [word for word in words if word not in STOP_WORDS and len(word) >= WORD_MIN_LENGTH]
        words = [word for word in words if word.isdigit()==False]
        return words


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
