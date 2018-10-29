from flask import Flask
from flask_sqlalchemy import SQLAlchemy

__author__ = 'sisung'
# -*- coding: utf-8 -*-

db = SQLAlchemy()

class TB_FILES(db.Model):
    __tablename__ = 'tb_files'
    __table_args__ = {
        'schema': 'db_search2'
    }
    idx = db.Column('idx', db.Integer, primary_key=True)
    f_idx_string = db.Column('f_idx_string', db.String(50))
    f_type = db.Column('f_type', db.Integer)
    f_topic = db.Column('f_topic', db.String(10))
    f_doc = db.Column('f_doc', db.String(1024*100))
    f_doc_words = db.Column('f_doc_words', db.String(1024*100))
    created_date = db.Column('created_date', db.DateTime)

class TB_TEMP_KEYWORD(db.Model):
    __tablename__ = 'tb_temp_keyword'
    __table_args__ = {
        'schema': 'db_search2'
    }
    idx = db.Column('idx', db.Integer, primary_key=True)
    keyword = db.Column('keyword', db.String(1024))

class TB_TEMP(db.Model):
    __tablename__ = 'tb_temp'
    __table_args__ = {
        'schema': 'db_search2'
    }
    idx = db.Column('idx', db.Integer, primary_key=True)
    keyword = db.Column('keyword', db.String(1024))
    keydata = db.Column('keydata', db.Integer)