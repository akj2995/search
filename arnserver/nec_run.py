#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'sisung'

import json
import os
import sys
import time
import logging
from logging import handlers
import threading
import signal
from datetime import datetime,timedelta
from ConfigParser import ConfigParser
import platform
from os import walk
version = "1.0.0"


current_dir = os.path.dirname(os.path.realpath(__file__))
up_dir = os.path.dirname(current_dir)
sys.path.append(up_dir + '/lib')

REQ_TIMEOUT = 3
import requests
import ctypes

import warnings
warnings.filterwarnings("ignore", category=UnicodeWarning)

def read_from_file(filename, section, required_props=None):
    config_parser = ConfigParser()
    config_parser.optionxform = str
    data = dict()

    try:
        data_set = config_parser.read(filename)
        if len(data_set) == 0:
            return None

        if section not in config_parser.sections():
            return dict()

        for k, v in config_parser.items(section):
            data[k] = v

        return data

    except IOError, e:
        print("read_from_file Open '%s' file failed: %s" % (filename, str(e)))
        return None

config = None
config = read_from_file('./config.ini', 'info')

DB_INFO = config['DB_INFO'].replace("\"","")
LDAP_INFO = config['LDAP_INFO'].replace("\"","")
CISCO_IP = config['CISCO_IP']
CISCO_ADMIN = config['CISCO_ADMIN']
CISCO_PASS = config['CISCO_PASS']
CISCO_DEFAULT_PASSWORD = config['CISCO_DEFAULT_PASSWORD']
ACTIVE = config['ACTIVE']
ACTIVE_WAS_IP = config['ACTIVE_WAS_IP']
STANBY_WAS_IP = config['STANBY_WAS_IP']

print ("DB_INFO : " + DB_INFO)
print ("LDAP_INFO : " + LDAP_INFO)
print ("CISCO_IP : " + CISCO_IP)
print ("CISCO_ADMIN : " + CISCO_ADMIN)
print ("CISCO_PASS : " + CISCO_PASS)
print ("ACTIVE : " + ACTIVE)
print ("ACTIVE_WAS_IP : " + ACTIVE_WAS_IP)
print ("STANBY_WAS_IP : " + STANBY_WAS_IP)

class NecProcess(object):
    def __init__(self, logger=None):
        print("NecProcess init")
        self.is_running = True
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger('NecProcess')
        else:
            self.logger = logger
        self.nec_config=None
        self.firstcheck = True
        self.this_was_disconect_count = 0
        self.active_was_disconnect_count = 0
        self.master_job_enable = False
        self.log_print_enable = True

    def Log(self,msg) :
        self.logger.info(msg)
        if self.log_print_enable is True :
            print(msg)

    def get_active_server_check(self):
        try:
            self.Log("get_active_server_check start ...")

            servers_url = 'https://' + ACTIVE_WAS_IP + '/AliveCheck'
            self.Log("check api url : " + servers_url)
            result = requests.get(servers_url,timeout=REQ_TIMEOUT, verify=False).text
            data = json.loads(result)
            if data['status'] == 200 :
                self.Log('ACTIVE SERVER IS ALIVE.......')
                self.this_was_disconect_count = 0
                self.active_was_disconnect_count = 0
            else :
                self.Log('ACTIVE SERVER IS DEAD.......')
                self.this_was_disconect_count = self.this_was_disconect_count + 1
                self.active_was_disconnect_count = self.active_was_disconnect_count + 1
        except:
            self.this_was_disconect_count = self.this_was_disconect_count + 1
            self.active_was_disconnect_count = self.active_was_disconnect_count + 1
            self.Log("get_active_server_check Exception")

    def get_stanby_server_check(self):
        try:
            self.Log("get_stanby_server_check start ...")

            servers_url = 'https://' + STANBY_WAS_IP + '/AliveCheck'
            self.Log("check api url : " + servers_url)
            result = requests.get(servers_url, timeout=REQ_TIMEOUT,verify=False).text
            data = json.loads(result)
            if data['status'] == 200 :
                self.Log("STANBY SERVER IS ALIVE.......")
                self.this_was_disconect_count = 0
            else :
                self.Log("STANBY SERVER IS DEAD.......")
                self.this_was_disconect_count = self.this_was_disconect_count + 1
        except:
            self.this_was_disconect_count = self.this_was_disconect_count + 1
            self.Log("get_stanby_server_check Exception")

    def get_nec_config(self):
        try:
            self.Log("get_nec_config start ...")
            servers_url = self.get_api_url() + '/UserConfig'
            self.Log("check api url : " + servers_url)
            result = requests.get(servers_url, timeout=REQ_TIMEOUT,verify=False).text
            data = json.loads(result)
            if data['status'] == 200 :
                self.nec_config = data['objects']
                self.Log('nec_config : ' + str(self.nec_config))
        except:
            self.logger.error("get_nec_config Exception")
            print("get_nec_config Exception")

    def get_ldap_to_db(self):
        try:
            self.Log("get_ldap_to_db start ...")
            servers_url = self.get_api_url() + '/LdapToDB'
            self.Log("check api url : " + servers_url)
            result = requests.get(servers_url, verify=False).text
            data = json.loads(result)
            self.Log(str(data))
        except:
            self.Log("get_ldap_to_db Exception")

    def get_check_reregister(self):
        try:
            self.Log("get_check_register_user_count start ...")
            servers_url = self.get_api_url() + '/CheckRegisterUsers'
            self.Log("check api url : " + servers_url)
            result = requests.get(servers_url,timeout=REQ_TIMEOUT,verify=False).text
            self.Log("get_check_reregister result : " + str(result))
            data = json.loads(result)
            objects = data['objects']
            self.logger.info(str(objects))
            if objects.get('registercount') is not None :
                count = objects['registercount']
                if int(count) < 1 :
                    return True
            
            return False
        except:
            self.Log("get_check_register_user_count Exception")
        return False

    def set_db_to_cisco(self):
        try:
            self.Log("set_db_to_cisco start ...")
            servers_url = self.get_api_url() + '/RegistAllUserToCisco'
            self.Log("check api url : " + servers_url)
            result = requests.get(servers_url, verify=False).text
            data = json.loads(result)
            self.Log(str(data))
        except:
            self.Log("set_db_to_cisco Exception")
        
    def get_api_url(self) :
        api_url = ''
        if ACTIVE == 'YES' : 
            api_url = 'https://' + ACTIVE_WAS_IP
        else :
            api_url = 'https://' + STANBY_WAS_IP
        return api_url

    def get_backup_db_list(self) :
        f=[]
        for(dirpath,dirnames,filenames) in walk('../../../db_back'):
            f.extend(filenames)
        return f

    def process_db_delete(self,del_day) :
        try :
            f = self.get_backup_db_list()
            print(str(f))
            date_n_days_ago = datetime.now()-timedelta(days=del_day)
            deldaytime = date_n_days_ago.strftime("%Y%m%d")
            del_file_list = []
            for x in f :
                filename = x.split(".")
                if int(filename[0]) < int(deldaytime) :
                    print("del file : " + filename[0] + ".sql")
                    del_file_list.append(x)
            for x in del_file_list :
                os.remove("../../../db_back/" + x)
                
        except :
            print("process_db_delete exception!!" )

    def run(self):
         
        while self.is_running:

            try:
                if ACTIVE == 'YES' : 
                    self.master_job_enable = True
                    self.get_active_server_check()
                else : 
                    self.get_active_server_check()
                    self.get_stanby_server_check()

                if self.this_was_disconect_count > 2 :
                    self.this_was_disconect_count = 0
                    self.Log("THIS WAS RESTART......")
                    #WAS restart
                if ACTIVE == 'NO' and  self.active_was_disconnect_count > 2: 
                   self.active_was_disconnect_count = 0
                   self.master_job_enable = True
                   self.Log("STANBY WAS is granted Master Job......")
                   #change master
                      
                self.get_nec_config()

                now = datetime.now()
                nTime = now.strftime('%H:%M')
                nowMinute = now.strftime('%M')

                #if nTime == self.nec_config['ldap_sync_time'] :
                #    self.get_ldap_to_db()

                bForceRegister = self.get_check_reregister()
                
                self.Log("firstcheck : " + str(self.firstcheck) + ", forceregister : " + str(bForceRegister) + ", nTime : " + nTime)

                #if self.nec_config is not None and nTime == self.nec_config['ldap_sync_time'] or self.get_check_reregister() == True:
                if self.nec_config is not None and nTime == self.nec_config['ldap_sync_time']:
                    self.Log("self.master_job_enable : " + str(self.master_job_enable))
                    if self.master_job_enable is True :
                        self.Log("LDAP & CISCO SYNC START......")
                        self.get_ldap_to_db()
                        if self.nec_config is not None and self.nec_config['cisco_sync'] == '1' :
                            self.set_db_to_cisco()
                        self.firstcheck = False

                if self.nec_config is not None :
                    self.process_db_delete(int(self.nec_config['db_save_days']))

                sleep_seconds = 30

                for i in range(sleep_seconds):
                    if not self.is_running:
                        break
                    time.sleep(1)

            except KeyboardInterrupt:
                self.Log('Exception KeyboardInterrupt')
                self.is_running = False

    def stop_process(self):
        self.is_running = False


if __name__ == '__main__':
    file_logger = logging.getLogger("NecProcess")
    file_logger.setLevel(logging.INFO)

    file_handler = handlers.RotatingFileHandler(
        "./log/nec_run.log",
        maxBytes=(1024 * 1024 * 1),
        backupCount=5
    )
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    file_logger.addHandler(file_handler)

    app = NecProcess(logger=file_logger)
    app.run()
    app.stop_process()

