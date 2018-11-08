# -*- coding: utf-8 -*-
from arnserver import app
from flask_restful import reqparse, Resource, Api
from .dbmodels import *
import platform
from flask import Flask, request
from flask import Response
import time
import hashlib
import base64
import json
import os
import logging
from logging import handlers
from sqlalchemy import or_, and_
from datetime import datetime, timedelta
import traceback
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import or_, and_
from sqlalchemy import ext
from flask_cors import CORS

from konlpy.corpus import kobill
from konlpy.tag import Kkma

from ckonlpy.utils import load_wordset

from ckonlpy.utils import load_replace_wordpair
from ckonlpy.utils import load_ngram

from konlpy.tag import Okt,Hannanum
from konlpy.tag import Kkma
from konlpy.tag import Komoran
from gensim import corpora
from gensim import models
from gensim.corpora.textcorpus import TextCorpus
from gensim.test.utils import datapath,get_tmpfile
from gensim.similarities import Similarity

from gensim.test.utils import common_texts
from gensim.corpora import Dictionary
from gensim.models import Word2Vec
from gensim.similarities import SoftCosineSimilarity
from ckonlpy.utils import load_ngram
import numpy as np
import math
import glob
import yake
import csv
import ast
from .pred_per_epoch import *
from .train_model import  *
PRINT_LOG = True


app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:1234@ec2-18-188-25-204.us-east-2.compute.amazonaws.com:3306/db_search2'

db.init_app(app)

var_cros_v1 = {'Content-Type', 'token', 'If-Modified-Since', 'Cache-Control', 'Pragma'}
# var_cros_v2 = {'Content-Type', 'token'}
CORS(app, resources=r'/*', headers=var_cros_v1)

# Multilanguages
import sys
from importlib import reload
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")

# --------------------------------------------------------------------------------------------------------------------
#                                            Static Area
# --------------------------------------------------------------------------------------------------------------------
DAEMON_HEADERS = {'Content-type': 'application/json'}

g_platform = platform.system()

if g_platform == "Linux":
    LOG_DEFAULT_DIR = './log'
elif g_platform == "Windows":
    LOG_DEFAULT_DIR = '.'
elif g_platform == "Darwin":
    LOG_DEFAULT_DIR = '.'


# --------------------------------------------------------------------------------------------------------------------
#                                            Function Area
# --------------------------------------------------------------------------------------------------------------------

def result(code, notice, objects, meta, author):
    """
    html status code def
    [ 200 ] - OK
    [ 400 ] - Bad Request
    [ 401 ] - Unauthorized
    [ 404 ] - Not Found
    [ 500 ] - Internal Server Error
    [ 503 ] - Service Unavailable
    - by thingscare
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

    if author is None:
        author = "by sisung"

    result = {
        "status": code,
        "notice": notice,
        "author": author
    }

    log_bySisung = ''

    # [ Check ] Objects
    if objects is not None:
        result["objects"] = objects

    # [ Check ] Meta
    if meta is not None:
        result["meta"] = meta

    if code == 200:
        result["message"] = "OK"
        log_bySisung = OKBLUE
    elif code == 400:
        result["message"] = "Bad Request"
        log_bySisung = FAIL
    elif code == 401:
        result["message"] = "Unauthorized"
        log_bySisung = WARNING
    elif code == 404:
        result["message"] = "Not Found"
        log_bySisung = FAIL
    elif code == 500:
        result["message"] = "Internal Server Error"
        log_bySisung = FAIL
    elif code == 503:
        result["message"] = "Service Unavailable"
        log_bySisung = WARNING

    log_bySisung = log_bySisung + 'RES : [' + str(code) + '] ' + str(notice) + ENDC
    return result


# --------------------------------------------------------------------------------------------------------------------
#                                            Class Area
# --------------------------------------------------------------------------------------------------------------------
class Helper(object):
    @staticmethod
    def get_file_logger(app_name, filename):
        log_dir_path = LOG_DEFAULT_DIR
        try:
            if not os.path.exists(log_dir_path):
                os.mkdir(log_dir_path)

            full_path = '%s/%s' % (log_dir_path, filename)
            file_logger = logging.getLogger(app_name)
            file_logger.setLevel(logging.INFO)

            file_handler = handlers.RotatingFileHandler(
                full_path,
                maxBytes=(1024 * 1024 * 10),
                backupCount=5
            )
            formatter = logging.Formatter('%(asctime)s %(message)s')

            file_handler.setFormatter(formatter)
            file_logger.addHandler(file_handler)

            return file_logger

        except :
            return logging.getLogger(app_name)


exception_logger = Helper.get_file_logger("exception", "exception.log")
service_logger = Helper.get_file_logger("service", "service.log")


def Log(msg) :
    try :
        if PRINT_LOG == True :
            print(msg)
        service_logger.info(msg)
    except :
        print("log exception!!")

@app.errorhandler(500)
def internal_error(exception):
    exception_logger.info(traceback.format_exc())
    return 500


@app.errorhandler(404)
def internal_error(exception):
    exception_logger.info(traceback.format_exc())
    return 404


# Singleton Source from http://stackoverflow.com/questions/42558/python-and-the-singleton-pattern
class Singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.

    The decorated class can define one `__init__` function that
    takes only the `self` argument. Other than that, there are
    no restrictions that apply to the decorated class.

    To get the singleton instance, use the `Instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.

    Limitations: The decorated class cannot be inherited from.
    """

    def __init__(self, decorated):
        self._decorated = decorated

    def instance(self):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.

        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `Instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)


@Singleton
class TokenManager(object):
    def generate_token(self, userID):
        m = hashlib.sha1()

        m.update(str(userID))
        m.update(datetime.now().isoformat())

        return m.hexdigest()

    def validate_token(self, token):
        token_result = TB_LOGIN_USER.query.filter_by(token=token).first()

        if token_result is None:
            return ""
        return token_result.user_id


@app.route('/')
class Login(Resource):
    """
    [ Login ]
    For Mobile Auth
    @ GET : Returns Result
    by sisung
    """

    def __init__(self):
        self.parser = reqparse.RequestParser()
        print("login init")
        self.parser.add_argument("userID", type=str, location="json")
        self.parser.add_argument("userPW", type=str, location="json")

        self.token_manager = TokenManager.instance()

        self.user_id = self.parser.parse_args()["userID"]
        self.user_password = self.parser.parse_args()["userPW"]
        super(Login, self).__init__()

    def post(self):
        try:
            print("login start")
            objects = []
            Log("[LOGIN START...]")
            # query = "SELECT id FROM %(table_name)s WHERE user_id='%(ID)s' AND user_password=password('%(PW)s')" % {
            #     "table_name": TB_LOGIN_USER.__tablename__,
            #     "ID": self.user_id,
            #     "PW": self.user_password
            # }
            # print ("sql : " + query)
            # login_user = TB_LOGIN_USER.query.from_statement(query).first()
            # print("login start 111")
            # if login_user is not None:
            #     update_token = self.token_manager.generate_token(login_user.user_id)
            #     print("login start 222")
            #     token_input = {}
            #     token_input["token"] = update_token
            #     db.session.query(TB_LOGIN_USER).filter_by(user_id=self.user_id).update(token_input)
            #     db.session.commit()
            #     print("login start 333")
            #     objects.append({
            #         'login': True,
            #         'userOTP': False,
            #         'token': update_token
            #     })
            Log("[Login SUCCESS]")
            return result(200, "Login successful.", objects, None, "by sisung ")
        except:
            Log("[Login exception]")
            return result(400, "Login exception ", None, None, "by sisung ")
        return result(400, "Login failed.", objects, None, "by sisung")

@app.route('/')
class LoadSrcFile(Resource):
    """
    [ LoadSrcFile ]
    For Mobile Auth
    @ GET : Returns Result
    by sisung
    """

    def __init__(self):
        self.parser = reqparse.RequestParser()
        print("LoadSrcFile init")
        self.parser.add_argument("group_path", type=str, location="json")

        self.token_manager = TokenManager.instance()
        print("self.parser.parse_args() : ",self.parser.parse_args())
        self.group_path = self.parser.parse_args()["group_path"]
        self.t = Okt()
        self.ha = Hannanum()
        super(LoadSrcFile, self).__init__()

    def getKeywords(self,docs_ko):
        datas = ''
        cur = 0
        for k in docs_ko:
            # if cur > 100 :
            #     break
            cur += 1
            datas = datas + k.replace("'","")
            datas = datas + ','
        query = "SELECT idx,keydata,GET_KEYWORD_TOP('{0}',{1}) AS keyword FROM tb_temp LIMIT 1".format(datas, cur)
        # print(query)
        results = TB_TEMP.query.from_statement(query).first()
        result_str = results.keyword.replace('"','')
        db.session.commit()
        return result_str

    def getSimList(self,keywords,docs_ko,topics_ko):
        pos = lambda d: ['/'.join(p) for p in self.t.pos(d)]
        def tokenize(doc):
            return ['/'.join(t) for t in self.t.pos(doc, norm=True, stem=True)]
        doc_dict = []
        for doc in docs_ko:
            if doc.find('(') != -1 or doc.find(')') != -1 or doc.find('"') != -1 or doc.find('\"') != -1:
                continue
            doc_dict.append(doc)
        docs_ko = [''.join(d) for d in doc_dict]
        common_texts = []
        for doc in docs_ko:
            try:
                tokens = tokenize(doc)
                for token in tokens:
                    common_texts.append(token)
            except:
                pass
        commons = []
        datalist = []
        for c in common_texts:
            try:
                array2 = c.split('/')
                if len(array2) > 1 and array2[1] == 'Noun':
                    commons.append([c])
                    datalist.append(array2[0])
                    # print("==>add ", c)
            except:
                pass

        topic_dict = []
        for topic in topics_ko:
            if topic.find('(') != -1 or topic.find(')') != -1 or topic.find('"') != -1:
                continue
                topic_dict.append(topic)
        topics_ko = [''.join(d) for d in topic_dict]
        print("topics_ko",topics_ko)
        topic_common_texts = []
        for doc in topics_ko:
            try:
                tokens = tokenize(doc)
                for token in tokens:
                    topic_common_texts.append(token)
            except:
                pass
        topic_datalist = []
        for c in topic_common_texts:
            try:
                array2 = c.split('/')
                if len(array2) > 1 and array2[1] == 'Noun':
                    topic_datalist.append(array2[0])
            except:
                pass

        model = Word2Vec(commons)  # train word-vectors
        model.init_sims(replace=True)
        result_list = []
        for key in keywords:
            cur = 0
            key_list = []
            for doc in datalist:
                m_doc = key + ' ' + doc
                # print("m_doc : ", m_doc)
                try:
                    # tok = tokenize(m_doc)
                    sim = model.wv.similarity(*tokenize(m_doc))
                    key_list.append(sim)
                    # print("===> ", m_doc, ", sim : ", sim)
                except:
                    pass
                cur += 1
            topic_list = []
            for topic in topic_datalist:
                m_doc = key + ' ' + topic
                # print("m_doc : ", m_doc)
                try:
                    # tok = tokenize(m_doc)
                    sim = model.wv.similarity(*tokenize(m_doc))
                    topic_list.append(sim)
                    # print("===> ", m_doc, ", sim : ", sim)
                except:
                    pass
            query_idf = self.idf(datalist,key)
            if len(key_list) > 0:
                obj = {
                    "key":key,
                    "query_idf":query_idf,
                    "sim_cos":key_list,
                    "topic_cos": topic_list
                }
                result_list.append(obj)
        return result_list

    def insert_file_info(self,f_group_idx,f_idx_string,f_file_path,f_doc,f_topic,f_doc_words,f_topic_words):
        db.session.query(TB_FILES).filter_by(f_idx_string=f_idx_string).delete()
        db.session.commit()
        new_file_info = TB_FILES()
        new_file_info.f_group_idx = f_group_idx
        new_file_info.f_idx_string = f_idx_string
        new_file_info.f_file_path = f_file_path
        new_file_info.f_type= 0
        new_file_info.f_doc = str(f_doc).replace('"', '')
        new_file_info.f_topic = str(f_topic).replace('"','')
        new_file_info.f_doc_words = str(f_doc_words)
        new_file_info.f_topic_words = str(f_topic_words)
        new_file_info.created_date = datetime.utcnow()
        db.session.add(new_file_info)
        db.session.commit()
        print("new_file_info.idx : ",new_file_info.idx)
        return new_file_info.idx

    def insert_keyword(self,f_group_idx,f_idx_string,fk_f_idx,k_word,k_topic_idf,k_desc_idf):
        db.session.query(TB_KEYWORD).filter_by(f_group_idx=f_group_idx).filter_by(f_idx_string=f_idx_string).filter_by(k_word=k_word).delete()
        db.session.commit()
        new_keyword = TB_KEYWORD()
        new_keyword.f_group_idx = f_group_idx
        new_keyword.f_idx_string = f_idx_string
        new_keyword.fk_f_idx = fk_f_idx
        new_keyword.k_word = k_word
        new_keyword.k_topic_idf = str(k_topic_idf)
        new_keyword.k_desc_idf = str(k_desc_idf)
        new_keyword.created_date = datetime.utcnow()
        db.session.add(new_keyword)
        db.session.commit()
        return new_keyword.idx

    def insert_sim_cos(self,f_group_idx,f_idx_string,k_word,fk_k_idx,fk_f_idx,s_sim_cos_doc,s_sim_cos_topic,label):
        db.session.query(TB_SIM_COS).filter_by(f_group_idx=f_group_idx).filter_by(f_idx_string=f_idx_string).filter_by(k_word=k_word).delete()
        db.session.commit()
        new_sim_cos = TB_SIM_COS()
        new_sim_cos.f_group_idx = f_group_idx
        new_sim_cos.f_idx_string = f_idx_string
        new_sim_cos.fk_k_idx = fk_k_idx
        new_sim_cos.fk_f_idx = fk_f_idx
        new_sim_cos.k_word = k_word
        print("s_sim_cos_doc len : " , len(json.dumps(s_sim_cos_doc)))
        print("s_sim_cos_topic len : ", len(json.dumps(s_sim_cos_topic)))
        new_sim_cos.s_sim_cos_doc = json.dumps(s_sim_cos_doc)
        new_sim_cos.s_sim_cos_topic = json.dumps(s_sim_cos_topic)
        new_sim_cos.label = label
        new_sim_cos.created_date = datetime.utcnow()
        db.session.add(new_sim_cos)
        db.session.commit()
        return new_sim_cos.idx


    def idf(self,index, term):
        list_count = len(index)
        find = 0
        for d in index:
            if d == term:
                find += 1
        if find == 0:
            return 0
        return math.log(float(list_count) / find)

    def post(self):
        # try:
        print("LoadSrcFile start")


        def getKobilFilePath(path) :
            return path.split('../data/')[1]
        input_doc_dir = '../../data/data/' + str(self.group_path) + '/doc/*.txt'
        doc_files = glob.glob(input_doc_dir)

        print("doc_file len : ", len(doc_files))
        group = TB_GROUP.query.filter_by(group_name=self.group_path).first()
        cur = 0
        for f in doc_files:
            # if cur > 0:
            #     break

            cur += 1
            doc_filepath = getKobilFilePath(f)
            topic_filepath = doc_filepath.replace('doc','topic')
            # print("doc_file : ", doc_filepath)
            # print("topic_file : ", topic_filepath)
            filename = os.path.basename(doc_filepath).split('.')[0]
            print("============================== start cur : ", cur, "  filename : ",filename)
            # print("filename : ", filename)
            doc_ngrams = kobill.open(doc_filepath).read()
            docs_ko = self.ha.nouns(doc_ngrams)
            keywords = self.getKeywords(docs_ko)
            print("keywords : ", keywords)
            topic_ngrams = kobill.open(topic_filepath).read()
            topics_ko = self.ha.nouns(topic_ngrams)
            # print("topics_ko : ", topics_ko)
            sim_list = self.getSimList(keywords.split(','),docs_ko,topics_ko)
            # print("sim_list : ",sim_list)

            fk_f_idx = self.insert_file_info(group.idx,filename,doc_filepath,doc_ngrams,topic_ngrams,docs_ko,topics_ko)
            for sim in sim_list :
                print("key : ",sim['key'])
                print("query_idf : ", sim['query_idf'])
                f_k_idx = self.insert_keyword(group.idx,filename,fk_f_idx,sim['key'],sim['query_idf'],sim['query_idf'])
                print("f_k_idx : ", f_k_idx)
                f_s_idx = self.insert_sim_cos(group.idx,filename,sim['key'],fk_f_idx,f_k_idx,sim['sim_cos'],sim['topic_cos'])
                print("f_s_idx : ", f_s_idx)


        objects = []
        Log("[LoadSrcFile SUCCESS]")
        return result(200, "LoadSrcFile successful.", objects, None, "by sisung ")
        # except:
        #     Log("[Login exception]")
        #     return result(400, "Login exception ", None, None, "by sisung ")
        # return result(400, "Login failed.", objects, None, "by sisung")

@app.route('/')
class SearchQuery(Resource):
    """
    [ SearchQuery ]
    For SearchQuery
    @ GET : Returns Result
    by sisung
    """

    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument("group_path", type=str, location="json")
        self.parser.add_argument("querystring", type=str, location="json")

        self.token_manager = TokenManager.instance()
        self.group_path = self.parser.parse_args()["group_path"]
        self.querystring = self.parser.parse_args()["querystring"]
        super(SearchQuery, self).__init__()

    def post(self):
        # try:
        objects = []
        Log("[SearchQuery START...]")
        print("group_path:",self.group_path,",querystring :",self.querystring)
        group = TB_GROUP.query.filter_by(group_name=self.group_path).first()
        query_sim = TB_KEYWORD.query.filter_by(f_group_idx=group.idx).filter_by(k_word=self.querystring).first()
        print("k_topic_idf :", query_sim.k_topic_idf)
        sim_cos_list = TB_SIM_COS.query.filter_by(f_group_idx=group.idx).filter_by(k_word=self.querystring).all()
        print("sim_cos_list len :", len(sim_cos_list))
        objects = []
        for sim_cos in sim_cos_list:
            obj = {
                'f_idx_string': sim_cos.f_idx_string,
                "sim_doc": np.array(ast.literal_eval(sim_cos.s_sim_cos_topic), dtype=float),
                "sim_topic": np.array(ast.literal_eval(sim_cos.s_sim_cos_doc), dtype=float),
                'label': sim_cos.label,
            }
            # print("obj label : ",obj)
            objects.append(obj)
        query_obj = {
            "k_word": query_sim.k_word,
            "k_topic_idf": np.array(ast.literal_eval(query_sim.k_topic_idf), dtype=float),
            "k_desc_idf": np.array(ast.literal_eval(query_sim.k_desc_idf), dtype=float),
            "doc_matrix": objects
        }
        # print("query_obj : ", query_obj)


        objParam= {
            'expname': 'pacrrpub',
            'train_years': 'wt09_10',
            'test_year': 'wt15',
            'numneg': 6,
            'batch': 32,
            'winlen': 1,
            'kmaxpool': 3,
            'binmat': False,
            'context': False,
            'combine': 16,
            'iterations': 10,
            'shuffle': False,
            'parentdir': '/home/ubuntu/search/env/Scripts',
            'modelfn':'pacrr',
            'seed':7541,
            'qproximity':0,
            'maxqlen':16,
            'xfilters':'',
            'simdim':800,
            'epochs':50,
            'nsamples':2048,
            'ud':True,
            'ut':True,
            'distill':'firstk',
            'nfilter':32,
            'cascade':'',
            'type':1

        }
        print("param obj :",objParam)
        rank_idx_dict,rank_value_dict = pred(_log=None,_config=objParam,_query_obj=query_obj)
        print("rank_idx_dict :",rank_idx_dict)
        # print("rank_value_dict :", rank_value_dict)
        # for rank in range(1,len(rank_idx_dict)) :
        #     print("s_idx : ",rank_idx_dict[rank], ",values : ",rank_value_dict[rank])
        #     query = "SELECT idx,f_topic,substring(f_doc,1,80) as f_doc FROM tb_files WHERE f_group_idx={0} and f_idx_string = '{1}'  LIMIT 1".format(group.idx,rank_idx_dict[rank])
        #     results = TB_FILES.query.from_statement(query).first()
        #     obj = {
        #         'f_idx_string':rank_idx_dict[rank],
        #         'f_topic':results.f_topic,
        #         'f_doc':results.f_doc,
        #         'rank':str(rank)
        #     }
        #     objects.append(obj)
        Log("[SearchQuery SUCCESS]")
        return result(200, "SearchQuery successful.", objects, None, "by sisung ")
        # except:
        #     Log("[SearchQuery exception]")
        #     return result(400, "SearchQuery exception ", None, None, "by sisung ")
        return result(400, "SearchQuery failed.", objects, None, "by sisung")

@app.route('/')
class Trainning(Resource):
    """
    [ Trainning ]
    For Trainning
    @ GET : Returns Result
    by sisung
    """

    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument("group_path", type=str, location="json")
        self.parser.add_argument("epoch", type=str, location="json")
        self.parser.add_argument("squery_count", type=str, location="json")

        self.token_manager = TokenManager.instance()
        self.group_path = self.parser.parse_args()["group_path"]
        self.epoch = self.parser.parse_args()["epoch"]
        self.squery_count = self.parser.parse_args()["squery_count"]
        super(Trainning, self).__init__()

    def post(self):
        # try:
        query_objects={}

        Log("[Trainning START...]")
        print("group_path:",self.group_path,",epoch :",self.epoch,",squery_count :",self.squery_count)
        group = TB_GROUP.query.filter_by(group_name=self.group_path).first()
        query_sim_cos_list = TB_KEYWORD.query.filter_by(f_group_idx=group.idx).order_by('f_idx_string').limit(self.squery_count).all()
        cur = 1
        for query_sim in query_sim_cos_list :
            objects = []
            print("query_sim.k_word : ", query_sim.k_word, ",query_sim f_idx_string:",query_sim.f_idx_string,",group.idx:",group.idx)
            sim_cos_list = TB_SIM_COS.query.filter_by(f_group_idx=group.idx).filter_by(k_word=query_sim.k_word).all()
            for sim_cos in sim_cos_list :
                obj = {
                    'f_idx_string': sim_cos.f_idx_string,
                    "sim_doc": np.array(ast.literal_eval(sim_cos.s_sim_cos_topic), dtype=float),
                    "sim_topic": np.array(ast.literal_eval(sim_cos.s_sim_cos_doc), dtype=float),
                    'label': sim_cos.label,
                }
                # print("obj label : ",obj)
                objects.append(obj)
            query_obj = {
                "k_word":query_sim.k_word,
                "k_topic_idf":np.array(ast.literal_eval(query_sim.k_topic_idf), dtype=float),
                "k_desc_idf": np.array(ast.literal_eval(query_sim.k_desc_idf), dtype=float),
                "doc_matrix":objects
            }
            # print("query_obj : ", query_obj)
            query_objects[cur] = query_obj
            print("cur : ", cur, ", query_objects[cur] :" , query_objects[cur])
            cur += 1
        # print("objects len:", len(objects))
        objParam= {
            'expname': 'pacrrpub',
            'train_years': 'wt09_10',
            'numneg': 6,
            'batch': 32,
            'winlen': 1,
            'kmaxpool': 3,
            'binmat': False,
            'context': False,
            'combine': 16,
            'iterations': self.epoch,
            'shuffle': False,
            'parentdir': '/home/ubuntu/search/env/Scripts',
            'modelfn':'pacrr',
            'seed':7541,
            'qproximity':0,
            'maxqlen':16,
            'xfilters':'',
            'simdim':800,
            'epochs':self.epoch,
            'nsamples':2048,
            'ud':True,
            'ut':True,
            'distill':'firstk',
            'nfilter':32,
            'cascade':''
        }
        print("param obj :",objParam)
        train_model(_log=None, _config=objParam, _query_objects=query_objects)
        Log("[SearchQuery SUCCESS]")
        return result(200, "SearchQuery successful.", None, None, "by sisung ")
        # except:
        #     Log("[SearchQuery exception]")
        #     return result(400, "SearchQuery exception ", None, None, "by sisung ")
        # return result(400, "Trainning failed.", None, None, "by sisung")


@app.route('/')
class LoadSrcFile2(Resource):
    """
    [ LoadSrcFile2 ]
    For Mobile Auth
    @ GET : Returns Result
    by sisung
    """

    def __init__(self):
        self.parser = reqparse.RequestParser()
        print("LoadSrcFile2 init")
        self.parser.add_argument("group_path", type=str, location="json")

        self.token_manager = TokenManager.instance()
        print("self.parser.parse_args() : ",self.parser.parse_args())
        self.group_path = self.parser.parse_args()["group_path"]
        self.t = Okt()
        self.ha = Hannanum()
        super(LoadSrcFile2, self).__init__()

    def insert_file_info(self,f_group_idx,f_idx_string,f_file_path,f_doc,f_topic,f_doc_words,f_topic_words):
        db.session.query(TB_FILES).filter_by(f_idx_string=f_idx_string).delete()
        db.session.commit()
        new_file_info = TB_FILES()
        new_file_info.f_group_idx = f_group_idx
        new_file_info.f_idx_string = f_idx_string
        new_file_info.f_file_path = f_file_path
        new_file_info.f_type= 0
        new_file_info.f_doc = str(f_doc).replace('"', '')
        new_file_info.f_topic = str(f_topic).replace('"','')
        new_file_info.f_doc_words = str(f_doc_words)
        new_file_info.f_topic_words = str(f_topic_words)
        new_file_info.created_date = datetime.utcnow()
        db.session.add(new_file_info)
        db.session.commit()
        print("new_file_info.idx : ",new_file_info.idx)
        return new_file_info.idx

    def insert_keyword(self,f_group_idx,f_idx_string,fk_f_idx,k_word,k_topic_idf,k_desc_idf):
        db.session.query(TB_KEYWORD).filter_by(f_group_idx=f_group_idx).filter_by(f_idx_string=f_idx_string).filter_by(k_word=k_word).delete()
        db.session.commit()
        new_keyword = TB_KEYWORD()
        new_keyword.f_group_idx = f_group_idx
        new_keyword.f_idx_string = f_idx_string
        new_keyword.fk_f_idx = fk_f_idx
        new_keyword.k_word = k_word
        new_keyword.k_topic_idf = str(k_topic_idf)
        new_keyword.k_desc_idf = str(k_desc_idf)
        new_keyword.created_date = datetime.utcnow()
        db.session.add(new_keyword)
        db.session.commit()
        return new_keyword.idx

    def insert_sim_cos(self,f_group_idx,f_idx_string,k_word,fk_k_idx,fk_f_idx,s_sim_cos_doc,s_sim_cos_topic,label):
        db.session.query(TB_SIM_COS).filter_by(f_group_idx=f_group_idx).filter_by(f_idx_string=f_idx_string).filter_by(k_word=k_word).delete()
        db.session.commit()
        new_sim_cos = TB_SIM_COS()
        new_sim_cos.f_group_idx = f_group_idx
        new_sim_cos.f_idx_string = f_idx_string
        new_sim_cos.fk_k_idx = fk_k_idx
        new_sim_cos.fk_f_idx = fk_f_idx
        new_sim_cos.k_word = k_word
        print("s_sim_cos_doc len : " , len(json.dumps(s_sim_cos_doc)))
        print("s_sim_cos_topic len : ", len(json.dumps(s_sim_cos_topic)))
        new_sim_cos.s_sim_cos_doc = json.dumps(s_sim_cos_doc)
        new_sim_cos.s_sim_cos_topic = json.dumps(s_sim_cos_topic)
        new_sim_cos.label = label
        new_sim_cos.created_date = datetime.utcnow()
        db.session.add(new_sim_cos)
        db.session.commit()
        return new_sim_cos.idx


    def idf(self,index, term):
        list_count = len(index)
        find = 0
        for d in index:
            if d == term:
                find += 1
        if find == 0:
            return 0
        return math.log(float(list_count) / find)

    def getDataFromFile(self,dir):
        doc_files = sorted([fname for fname  in os.listdir(dir) if fname.endswith('.npy')],
               key=lambda f: int(f.rsplit(os.path.extsep, 1)[0].rsplit(None, 1)[-1]))
        # print("doc_file len : ", len(doc_files))
        cur = 0
        array = {}
        for f in doc_files:
            if cur > 5:
                break
            cur += 1
            filename = os.path.basename(f).split('.')[0]

            data = np.load(dir + "/" + f)
            array[int(filename)] = data.tolist()
        return array

    def getDataFromFileNotSort(self,dir):
        doc_files = glob.glob(dir + "/*.npy")
        cur = 0
        array = {}
        for f in doc_files:
            if cur > 100:
                break
            cur += 1
            filename = os.path.basename(f).split('.')[0]
            # print("filename : ",filename,"f : ",f)
            data = np.load(f)
            # print("data : ",data)
            print("dir : ",dir , ",file : ", filename, ",len : ",os.path.getsize (f))
            array[str(filename)] = data.tolist()
            # print("data 1: ",array[str(filename)])
        return array

    def post(self):
        # try:
        print("LoadSrcFile start")


        def getKobilFilePath(path) :
            return path.split('../data/')[1]

        fname = '/home/ubuntu/copacrr/data/qrels.adhoc.6y'
        array_adhoc = {}
        with open(fname) as f:
            content = f.readlines()
            for line in content:
                params = line.split(' ')
                # print("params : ",params)
                if len(params) > 2 :
                    array_adhoc[params[2]] = {
                        'qid':params[0],
                        'label':params[3].replace('\n','')
                    }
        # print("array_adhoc :",array_adhoc)

        #
        input_doc_dir = '/local/var/tmp/ubuntu/cosine/query_idf/topic_term_idf'
        array_idf_topic = self.getDataFromFile(input_doc_dir)
        input_doc_dir = '/local/var/tmp/ubuntu/cosine/query_idf/desc_term_idf'
        array_idf_desc = self.getDataFromFile(input_doc_dir)

        print("array_idf_topic:",array_idf_topic)
        print("array_idf_desc:", array_idf_desc)
        array_doc_topic = {}
        array_doc_desc = {}
        print("array_idf_topic len :",len(array_idf_topic))
        for idx in range(1,len(array_idf_topic) + 1) :
            input_doc_dir = '/local/var/tmp/ubuntu/cosine/topic_doc_mat/' + str(idx)
            doc_topic = self.getDataFromFileNotSort(input_doc_dir)
            array_doc_topic[idx] = doc_topic
            input_doc_dir = '/local/var/tmp/ubuntu/cosine/desc_doc_mat/' + str(idx)
            doc_desc = self.getDataFromFileNotSort(input_doc_dir)
            array_doc_desc[idx] = doc_desc
        #
        # print("array_doc_topic:",array_doc_topic)
        # print("array_doc_desc:", array_doc_desc[1])
        #
        group = TB_GROUP.query.filter_by(group_name=self.group_path).first()
        for idx in range(1, len(array_idf_topic) + 1):
            fk_f_idx = self.insert_file_info(group.idx,str(idx),'','','','','')
            print("idx : ",idx)
            print("fk_f_idx : ", fk_f_idx)
            print("str(idx) :",str(idx),",fk_f_idx:",fk_f_idx,",array_idf_topic[idx] : ",array_idf_topic[idx],",array_idf_desc[idx]:",array_idf_desc[idx])
            f_k_idx = self.insert_keyword(group.idx,str(idx),fk_f_idx,str(idx),array_idf_topic[idx],array_idf_desc[idx])
            print("f_k_idx : ", f_k_idx)
            for topic in array_doc_topic[idx]:
                # print("array_doc_topic : ",array_doc_topic[idx][topic])
                label = array_adhoc[topic]['label']
                f_s_idx = self.insert_sim_cos(group.idx, topic,str(idx),f_k_idx,fk_f_idx,array_doc_topic[idx][topic],array_doc_desc[idx][topic],int(label))

        objects = []
        Log("[LoadSrcFile SUCCESS]")
        return result(200, "LoadSrcFile successful.", objects, None, "by sisung ")
        # except:
        #     Log("[Login exception]")
        #     return result(400, "Login exception ", None, None, "by sisung ")
        # return result(400, "Login failed.", objects, None, "by sisung")

@app.route('/')
class DocDetail(Resource):
    """
    [ DocDetail ]
    For Mobile Auth
    @ GET : Returns Result
    by sisung
    """

    def __init__(self):
        self.parser = reqparse.RequestParser()
        print("DocDetail init")
        self.parser.add_argument("group_path", type=str, location="json")
        self.parser.add_argument("f_idx_string", type=str, location="json")
        self.token_manager = TokenManager.instance()
        print("self.parser.parse_args() : ",self.parser.parse_args())
        self.group_path = self.parser.parse_args()["group_path"]
        self.f_idx_string = self.parser.parse_args()["f_idx_string"]
        super(DocDetail, self).__init__()

    def post(self):
        # try:
        print("DocDetail start")
        group = TB_GROUP.query.filter_by(group_name=self.group_path).first()

        file = TB_FILES.query.filter_by(f_group_idx=group.idx).first()
        if file is None :
            return result(200, "DocDetail successful.", file.f_doc, None, "by sisung ")
        return result(400, "DocDetail not founded.", None, None, "by sisung ")

@app.route('/')
class DomainFileList(Resource):
    """
    [ DomainFileList ]
    For Mobile Auth
    @ GET : Returns Result
    by sisung
    """

    def __init__(self):
        self.parser = reqparse.RequestParser()
        print("DocDDomainFileListetail init")
        self.parser.add_argument("group_path", type=str, location="json")
        self.token_manager = TokenManager.instance()
        print("self.parser.parse_args() : ",self.parser.parse_args())
        self.group_path = self.parser.parse_args()["group_path"]
        super(DomainFileList, self).__init__()

    def post(self):
        # try:
        print("DomainFileList start")
        group = TB_GROUP.query.filter_by(group_name=self.group_path).first()
        objects = []
        files = TB_FILES.query.filter_by(f_group_idx=group.idx).limit(10).all()
        for file in files :
            objects.append({
                'f_idx_string': file.f_idx_string,
                'f_topic': file.f_topic,
                'f_doc': file.f_doc,
                'f_topic_words': file.f_topic_words,
                'f_doc_words': file.f_doc_words
            })
        return result(200, "DomainFileList successful.", objects, None, "by sisung ")

@app.route('/')
class DomainKeywordList(Resource):
    """
    [ DomainKeywordList ]
    For Mobile Auth
    @ GET : Returns Result
    by sisung
    """

    def __init__(self):
        self.parser = reqparse.RequestParser()
        print("DomainKeywordList init")
        self.parser.add_argument("group_path", type=str, location="json")
        self.token_manager = TokenManager.instance()
        print("self.parser.parse_args() : ",self.parser.parse_args())
        self.group_path = self.parser.parse_args()["group_path"]
        super(DomainKeywordList, self).__init__()

    def post(self):
        # try:
        print("DomainKeywordList start")
        group = TB_GROUP.query.filter_by(group_name=self.group_path).first()
        objects = []
        keywordlist = TB_KEYWORD.query.filter_by(f_group_idx=group.idx).limit(20).all()
        for keyword in keywordlist :
            objects.append({
                'idx': keyword.idx,
                'f_idx_string': keyword.f_idx_string,
                'k_word': keyword.k_word,
                'k_topic_idf': keyword.k_topic_idf,
                'f_topic_words': keyword.f_topic_words,
                'k_desc_idf': keyword.k_desc_idf
            })
        return result(200, "DomainKeywordList successful.", objects, None, "by sisung ")

@app.route('/')
class DomainSimMatList(Resource):
    """
    [ DomainSimMatList ]
    For Mobile Auth
    @ GET : Returns Result
    by sisung
    """

    def __init__(self):
        self.parser = reqparse.RequestParser()
        print("DomainKeywordList init")
        self.parser.add_argument("group_path", type=str, location="json")
        self.parser.add_argument("fk_k_idx", type=str, location="json")
        self.token_manager = TokenManager.instance()
        print("self.parser.parse_args() : ",self.parser.parse_args())
        self.group_path = self.parser.parse_args()["group_path"]
        self.fk_k_idx = self.parser.parse_args()["fk_k_idx"]
        super(DomainSimMatList, self).__init__()

    def post(self):
        # try:
        print("DomainSimMatList start")
        group = TB_GROUP.query.filter_by(group_name=self.group_path).first()
        objects = []
        simmatlist = TB_SIM_COS.query.filter_by(f_group_idx=group.idx).filter_by(fk_k_idx=self.fk_k_idx).limit(20).all()
        for simmat in simmatlist :
            objects.append({
                'idx': simmat.idx,
                'f_idx_string': simmat.f_idx_string,
                's_sim_cos_topic': str(simmat.s_sim_cos_topic)[:100],
                's_sim_cos_doc': str(simmat.s_sim_cos_doc)[:100],
                'label': simmat.label
            })
        return result(200, "DomainSimMatList successful.", objects, None, "by sisung ")

api = Api(app)

# Basic URI
# 로그인
api.add_resource(Login, '/Login')
api.add_resource(LoadSrcFile, '/LoadSrcFile')
api.add_resource(LoadSrcFile2, '/LoadSrcFile2')
api.add_resource(SearchQuery, '/SearchQuery')
api.add_resource(Trainning, '/Trainning')
api.add_resource(DocDetail, '/DocDetail')
api.add_resource(DomainFileList, '/DomainFileList')
api.add_resource(DomainKeywordList, '/DomainKeywordList')
api.add_resource(DomainSimMatList, '/DomainSimMatList')
