import os, importlib
import keras.callbacks
from .utils.year_2_qids import get_train_qids, get_qrelf
from .utils.common_utils import read_qrel, config_logger, SoftFailure
from .utils.ngram_nfilter import get_ngram_nfilter
from .utils.utils import load_train_data_generator, DumpWeight, dump_modelplot,load_train_data_generator_new
import numpy as np, matplotlib as mpl
mpl.use('Agg')
mpl.rcParams.update({'font.size': 10})
import matplotlib.pyplot as plt
import pickle, logging
from .utils.config import file2name, default_params, perlf, qrelfdir, rawdoc_mat_dir
# forces the tensorflow session to be launched immediately
# it is important when the tf random seed is fixed
import keras.backend as K
K.get_session()
# import for sacred
import sacred
from sacred.utils import apply_backspaces_and_linefeeds


# set up the sacred env
ex = sacred.Experiment('train')
ex.path = 'train'
sacred.SETTINGS.HOST_INFO.CAPTURED_ENV.append('CUDA_VISIBLE_DEVICES')
sacred.SETTINGS.HOST_INFO.CAPTURED_ENV.append('USER')
ex.captured_out_filter = apply_backspaces_and_linefeeds
default_params = ex.config(default_params)



@ex.automain
def train_model(_log, _config,_query_objects):
    p = _config
    
    modelname = file2name[p['modelfn']]
    mod_model = importlib.import_module('.models.%s' % p['modelfn'],package='arnserver')
    # load the model to be employed, say from models/pacrr.py
    model_cls = getattr(mod_model, modelname)
    model_params = {k: v for k, v in p.items() if k in model_cls.params or k == 'modelfn'}
    model = model_cls(model_params, rnd_seed=p['seed'])
    # create a expid based on the configured parameters
    expid = model.params_to_string(model_params)
   
    # the model files 
    outdir='%s/train_%s/%s/model_weight/%s'%(p['parentdir'], p['train_years'], p['expname'], expid)
    # the plots for the model, the training loss etc..
    detail_outdir='%s/train_%s/%s/model_detail/'%(p['parentdir'], p['train_years'], p['expname'])

    if not os.path.isdir(detail_outdir + 'outs'):
        os.makedirs(detail_outdir + 'outs')
    
    # _log.info('Input parameters: {0}'.format(p))
    label2tlabel={4:2,3:2,2:2,1:1,0:0,-2:0}
    sample_label_prob=dict()
    # _log.info('{0} {1} {2}'.format(p['expname'], p['train_years'], sample_label_prob))

    NGRAM_NFILTER, N_GRAMS = get_ngram_nfilter(p['winlen'], p['qproximity'], p['maxqlen'], p['xfilters'])

    # _log.info('process and output to %s'%outdir)
    # _log.info('{0} {1} {2} {3} {4}'.format(p['distill'], 'NGRAM_NFILTER', NGRAM_NFILTER, 'N_GRAMS', N_GRAMS))
    if os.path.exists(outdir) and len(os.listdir(outdir)) == p['epochs']:
        # _log.info("outdir already seems to be full... exiting early")
        return

    # prepare train data
    # qids = get_train_qids(p['train_years'])
    qids = []
    for idx in range(1,len(_query_objects) + 1) :
        qids.append(idx)
    print("qids :",qids)
    # qids = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
    # qids = [1, 2, 3, 4, 5]
    # qrelf = get_qrelf(qrelfdir, p['train_years'])

    qid_cwid_label = {}
    # qid_cwid_label = read_qrel(qrelf, qids, include_spam=False)
    # print("qid_cwid_label :",qid_cwid_label)
    # for obj in _query_objects :

    qid_topic_idf = {}
    qid_desc_idf = {}
    src_sim_doc_array = dict()
    src_sim_topic_array = dict()
    import ast

    for idx in range(1, len(_query_objects) + 1):
        obj_list = _query_objects[idx]['doc_matrix']
        print("obj_list len : ",len(obj_list))
        obj_array = {}
        sim_doc_array = {}
        sim_topic_array = {}
        cur = 0
        for obj in obj_list:
            cur += 1
            obj_array[obj['f_idx_string']]=int(obj['label'])
            sim_doc_array[obj['f_idx_string']]= obj['sim_doc']
            sim_topic_array[obj['f_idx_string']] = obj['sim_topic']
        qid_cwid_label[idx] = obj_array
        # k_topic_idf = np.array([float(_query_objects[idx]['k_topic_idf'])])
        # k_desc_idf = np.array([float(_query_objects[idx]['k_desc_idf'])])
        k_topic_idf = _query_objects[idx]['k_topic_idf']
        k_desc_idf = _query_objects[idx]['k_desc_idf']
        qid_topic_idf[idx] = k_topic_idf
        qid_desc_idf[idx] = k_desc_idf
        src_sim_doc_array[idx] = sim_doc_array
        src_sim_topic_array[idx] = sim_topic_array

    # print("qid_cwid_label :",qid_cwid_label)
    train_qids =[qid for qid in qids if qid in qid_cwid_label]
    # _log.info('%s train_num %d '%(p['train_years'], len(train_qids)))

    def plot_curve_loss(epoch_train_loss, outdir, name, plot_id, series):
        epochs, losses = zip(*list(enumerate(epoch_train_loss)))
        argmin_loss_epoch =  np.argmin(epoch_train_loss)
        fig = plt.figure()
        plt.plot(epochs, losses, 'k:')
        plt.ylabel('Training Loss')
        plt.tick_params('y')
        plt.xlabel('epoches')
        plt.title('loss:%d %.3f'%(argmin_loss_epoch, epoch_train_loss[argmin_loss_epoch]))
        fig.savefig(outdir + '/' + name + '_' + plot_id + '.pdf', format='pdf')
        plt.close()


    # dump model plot
    built_model = model.build()
    model.build_predict()  # run build_predict to verify it's working
    dump_modelplot(built_model, detail_outdir + 'model_' + expid)

    # callback function, dump the model and compute ndcg/map
    dump_weight = DumpWeight(outdir, batch_size=p['batch'], nb_sample=p['nsamples'])
    # dump_weight = keras.callbacks.TensorBoard(log_dir='./graph',histogram_freq=0,write_graph=True,write_images=True)
    # keras 2 steps per epoch is number of batches per epoch, not number of samples per epoch
    steps_per_epoch = np.int(p['nsamples'] / p['batch'])

    # the generator for training data
    train_data_generator = load_train_data_generator_new(src_sim_doc_array,src_sim_topic_array,qid_topic_idf,qid_desc_idf,qids, rawdoc_mat_dir, qid_cwid_label, N_GRAMS, p,label2tlabel=label2tlabel, sample_label_prob=sample_label_prob)
    # train_data_generator=\
    #         load_train_data_generator(qids, rawdoc_mat_dir, qid_cwid_label, N_GRAMS, p,\
    #                 label2tlabel=label2tlabel, sample_label_prob=sample_label_prob)
    #
    history = built_model.fit_generator(train_data_generator, steps_per_epoch=steps_per_epoch, epochs=int(p['epochs']),
                                        verbose=0, callbacks=[dump_weight], max_q_size=15, workers=1, pickle_safe=False)
    #
    epoch_train_loss = history.history['loss']
    epoch_train_acc = history.history['acc']
    print("epoch_train_loss:",epoch_train_loss)
    print("epoch_train_acc:", epoch_train_acc)
    # plot the training loss for debugging
    plot_curve_loss(epoch_train_loss, detail_outdir, 'train_', expid, ['loss'])
    historyfile = detail_outdir + 'hist_' + expid + '.history'
    with open(detail_outdir + 'hist_' + expid + '.p', 'wb') as handle:
        pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    K.clear_session()