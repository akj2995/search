# -*- coding: utf-8 -*-
from arnserver import app
from flask_restful import reqparse, Resource, Api
from .models import *
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
        super(LoadSrcFile, self).__init__()


    def post(self):
        # try:
        print("LoadSrcFile start")

        def tokenize(doc):
            return ['/'.join(t) for t in t.pos(doc, norm=True, stem=True)]

        def getKobilFilePath(path) :
            return path.split('../data/')[1]

        t = Okt()
        ha = Hannanum()
        input_doc_dir = '../../data/data/' + str(self.group_path) + '/doc/*.txt'
        print("input_doc_dir : ", input_doc_dir)
        input_query_dir = './input/query'
        files = glob.glob(input_doc_dir)
        print("file len : ", len(files))
        kkma = Kkma()
        # komoran = Komoran()
        cur = 0
        for f in files:
            if cur > 0:
                break
            cur += 1
            filepath = getKobilFilePath(f)
            print("file : ", filepath)

            ngrams = kobill.open(filepath).read()
            pos = lambda d: ['/'.join(p) for p in t.pos(d)]
            docs_ko = ha.nouns(ngrams)
            print("ngrams : ",docs_ko)
            datas = ''
            for k in docs_ko :
                datas = datas + k
                datas = datas + ','
            query = "SELECT *,GET_KEYWORD_TOP('{0}',{1}) AS keyword FROM tb_temp LIMIT 1".format(datas,len(docs_ko))
            print(query)
            results = TB_TEMP.query.from_statement(query).first()
            db.session.commit()
            print("r : ",results.keyword)

            # print("result : ",results[0]['keyword'])
            # custom_kwextractor = yake.KeywordExtractor(lan="en", n=1, dedupLim=0.8, windowsSize=1, top=5)
            # # ngrams = load_ngram(filepath)
            # ngrams = kobill.open(filepath).read()
            #
            # keywords = custom_kwextractor.extract_keywords(ngrams)
            # print("keywords : ", keywords)
            # for key in keywords:
            #     print("key : ",key[1])
            # # print("ngrams : ", ngrams)

            # pos = lambda d: ['/'.join(p) for p in t.pos(d)]
            #
            #
            # # docs_ko = kkma.nouns(ngrams)
            #
            # docs_ko = komoran.nouns(ngrams)
            # # docs_ko = mecab.nouns(ngrams)
            # # docs_ko = ha.nouns(ngrams)
            # # docs_ko = ha.pos(ngrams,ntags=9)
            doc_dict = []
            for doc in docs_ko:
                if doc.find('(') != -1 or doc.find(')') != -1:
                    continue
                doc_dict.append(doc)
            docs_ko = [''.join(d) for d in doc_dict]
            print("docs_ko : ", docs_ko)
            #  # common_texts = [tokenize(doc) for doc  in docs_ko]
            common_texts = []
            for doc in docs_ko:
                # print("doc : ", doc)
                try:
                    tokens = tokenize(doc)
                    for token in tokens:
                        # print("token : ", token)
                        common_texts.append(token)
                except:
                    pass
            print("common_texts : ", common_texts)

            commons = []
            datalist = []
            cur = 0
            for c in common_texts:
                try:
                    array2 = c.split('/')
                    if len(array2) > 1 and array2[1] == 'Noun':
                        commons.append([c])
                        datalist.append(array2[0])
                        # print("==>add ", c)
                except:
                    pass

            model = Word2Vec(commons)  # train word-vectors
            model.init_sims(replace=True)
            keywords = results.keyword.split(',')
            for key in keywords:
                cur = 0
                result_list = []
                for doc in datalist:
                    m_doc = key + ' ' + doc
                    # print("m_doc : ", m_doc)
                    try:
                        tok = tokenize(m_doc)
                        # print("tok : ", tok)
                        # print("===> ",m_doc,", sim : ",model.wv.similarity(*tokenize(u'이스라엘 아랍')))
                        # print("mdoc : ",*tokenize(m_doc))
                        sim = model.wv.similarity(*tokenize(m_doc))
                        result_list.append(sim)
                        # print("===> ", m_doc, ", sim : ", sim)
                    except:
                        pass
                    cur += 1
                if len(result_list) > 0 :
                    print("sim list :", result_list)

        objects = []
        Log("[LoadSrcFile START...]")
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
        Log("[LoadSrcFile SUCCESS]")
        return result(200, "LoadSrcFile successful.", objects, None, "by sisung ")
        # except:
        #     Log("[Login exception]")
        #     return result(400, "Login exception ", None, None, "by sisung ")
        # return result(400, "Login failed.", objects, None, "by sisung")

api = Api(app)

# Basic URI
# 로그인
api.add_resource(Login, '/Login')
api.add_resource(LoadSrcFile, '/LoadSrcFile')
