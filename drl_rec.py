#!encoding=utf-8
"""
    Distilled reinforcement learning framework for recommendation (DRL-Rec)
    @2021-01-04
    modriczhang
"""
import os
import sys
import math
import time
import random
import hashlib
import datetime
import subprocess
import numpy as np
import tensorflow.compat.v1 as tf
from layer_util import *
from data_reader import DataReader
from hyper_param import param_dict as pd
from replay_buffer import RB

tf.disable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)

g_working_mode = 'local_train'
g_training = False
# replay buffer
g_rb = None
# data reader
g_dr = DataReader(pd['batch_size'])

g_batch_counter = 0
g_topk_pool = [[], [], [], []]

class DRL4Rec(object):
    def __init__(self, global_step):
        self.global_step = global_step
        # cumulative loss
        self.teacher_loss_val, self.student_loss_val, self.topk_loss_val = 0.0, 0.0, 0.0
        # placeholder
        self.sph_user = tf.sparse_placeholder(tf.int32, name='sph_user')
        self.sph_doc = tf.sparse_placeholder(tf.int32, name='sph_doc')
        self.sph_con = tf.sparse_placeholder(tf.int32, name='sph_con')
        self.ph_reward = tf.placeholder(tf.float32, name='ph_reward')
        self.ph_seqlen = tf.placeholder(tf.int32, name='seq_len')
        self.seqlen = tf.identity(self.ph_seqlen)
        print('\n======\nbuilding teacher main Q-network...')
        self.TMH, self.TMQ = self.build_q_network('teacher', 'main', pd['teacher_feat_dim'])
        print('\n======\nbuilding teacher target Q-network...')
        self.TTH, self.TTQ = self.build_q_network('teacher', 'target', pd['teacher_feat_dim'])
        print('\n======\nbuilding student main Q-network...')
        self.SMH, self.SMQ = self.build_q_network('student', 'main', pd['student_feat_dim'])
        print('\n======\nbuilding student target Q-network...')
        self.STH, self.STQ = self.build_q_network('student', 'target', pd['student_feat_dim'])

        teacher_var, student_var = 0, 0
        vs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='')
        for vi in vs:
            dims = vi.get_shape().as_list()
            if 'teacher' in vi.name:
                teacher_var += dims[0] if len(dims) == 1 else dims[0] * dims[1]
            elif 'student' in vi.name:
                student_var += dims[0] if len(dims) == 1 else dims[0] * dims[1]
            else:
                pass
        print('\n\n=====Teacher Network: ' + str(teacher_var))
        print('=====Student Network: ' + str(student_var))
        print('=====Compress ratio: ' + str(1.0 - 1.0 * student_var / teacher_var))
        # teacher rl loss
        self.TNQ = tf.stop_gradient(self.TTQ)
        self.TNQ = tf.concat([self.TNQ[:, 1:], tf.zeros_like(self.TNQ)[:, :1]], axis=1)
        self.TYT = tf.reshape(self.ph_reward, [-1]) + tf.scalar_mul(tf.constant(pd['rl_gamma']),
                                                                    tf.reshape(self.TNQ, [-1]))
        self.TTDE = tf.square(self.TYT - tf.reshape(self.TMQ, [-1]))
        self.teacher_loss = tf.reduce_mean(self.TTDE)
        # teacher optimization
        vs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='teacher/main')
        vs.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='teacher/embedding'))
        self.t_grads = tf.clip_by_global_norm(tf.gradients(self.teacher_loss, vs), pd['grad_clip'])[0]
        with tf.variable_scope('opt_teacher'):
            optimizer = tf.train.AdamOptimizer(pd['teacher_lr'])
            self.teacher_opt = optimizer.apply_gradients(zip(self.t_grads, vs), global_step=global_step)
        m_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="teacher/main")
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="teacher/target")
        alpha = pd['double_networks_sync_step']
        self.teacher_sync_op = [tf.assign(t, (1.0 - alpha) * t + alpha * m) for t, m in zip(t_params, m_params)]
        # student rl loss
        self.SNQ = tf.stop_gradient(self.STQ)
        self.SNQ = tf.concat([self.SNQ[:, 1:], tf.zeros_like(self.SNQ)[:, :1]], axis=1)
        self.SYT = tf.reshape(self.ph_reward, [-1]) + tf.scalar_mul(tf.constant(pd['rl_gamma']),
                                                                    tf.reshape(self.SNQ, [-1]))
        self.STDE = tf.square(self.SYT - tf.reshape(self.SMQ, [-1]))
        self.student_rl_loss = tf.reduce_mean(self.STDE)
        # confidence
        maxe = tf.reduce_max(tf.reshape(self.TTDE, [-1])) + 1e-6
        self.guide_weights = tf.stop_gradient(1.0 - self.TTDE / maxe)

        # student guide loss
        logp = tf.nn.log_softmax(logits=tf.stop_gradient(self.TMQ) / pd['distill_temperature'], dim=-1)
        logq = tf.nn.log_softmax(logits=self.SMQ, dim=-1)
        self.student_kd_loss = tf.reduce_mean(self.kl_divergence(logp, logq))
        # student hint loss
        if pd['enable_confidence']:
            self.hint_loss = tf.reduce_mean(tf.multiply(tf.stop_gradient(self.guide_weights),
                                                        tf.reduce_mean(tf.square(
                                                            tf.reshape(self.SMH - tf.stop_gradient(self.TMH), [-1])))))
        else:
            self.hint_loss = tf.reduce_mean(tf.square(tf.reshape(self.SMH - tf.stop_gradient(self.TMH), [-1])))
        loss_weights = [0.4, 0.3, 0.3]
        if pd['enable_distill']:
            self.student_loss = loss_weights[0] * self.student_rl_loss + \
                                loss_weights[1] * self.student_kd_loss + \
                                loss_weights[2] * self.hint_loss
        else:
            self.student_loss = loss_weights[0] * self.student_rl_loss

        vs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='student/main')
        vs.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='student/embedding'))
        self.s_grads = tf.clip_by_global_norm(tf.gradients(self.student_loss, vs), pd['grad_clip'])[0]
        with tf.variable_scope('opt_student'):
            optimizer = tf.train.AdamOptimizer(pd['student_lr'])
            self.student_opt = optimizer.apply_gradients(zip(self.s_grads, vs), global_step=global_step)
        self.topk_grads = tf.clip_by_global_norm(tf.gradients(self.hint_loss, vs), pd['grad_clip'])[0]
        with tf.variable_scope('opt_topk_distill'):
            optimizer = tf.train.AdamOptimizer(pd['topk_distill_lr'])
            self.topk_distill_opt = optimizer.apply_gradients(zip(self.topk_grads, vs), global_step=global_step)
        m_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="student/main")
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="student/target")
        self.student_sync_op = [tf.assign(t, (1.0 - alpha) * t + alpha * m) for t, m in zip(t_params, m_params)]

    def kl_divergence(self, logp, logq):
        p = tf.exp(logp)
        return tf.reduce_sum(p * logp, axis=-1) - tf.reduce_sum(p * logq, axis=-1)

    def field_interact(self, fields):
        global g_training
        qkv = tf.layers.dropout(fields, rate=pd['dropout'], training=g_training)
        with tf.variable_scope('fi'):
            return multihead_attention(queries=qkv,
                                       keys=qkv,
                                       values=qkv,
                                       num_heads=pd['head_num'],
                                       dropout_rate=pd['dropout'],
                                       training=g_training,
                                       causality=False,
                                       scope='mha')

    def build_embedding_layer(self, sub_net, scope, feat_dim):
        with tf.variable_scope(sub_net, reuse=tf.AUTO_REUSE):
            feat_dict = get_embeddings(g_dr.unique_feature_num(),
                                       feat_dim,
                                       scope='embedding',
                                       zero_pad=False)
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                user_embed = tf.nn.embedding_lookup_sparse(feat_dict,
                                                           self.sph_user,
                                                           sp_weights=None,
                                                           partition_strategy='div',
                                                           combiner='mean')
                user_fields = tf.reshape(user_embed, shape=[-1, pd['user_field_num'], feat_dim])
                doc_embed = tf.nn.embedding_lookup_sparse(feat_dict,
                                                          self.sph_doc,
                                                          sp_weights=None,
                                                          partition_strategy='div',
                                                          combiner='mean')
                doc_fields = tf.reshape(doc_embed, shape=[-1, pd['doc_field_num'], feat_dim])
                con_embed = tf.nn.embedding_lookup_sparse(feat_dict,
                                                          self.sph_con,
                                                          sp_weights=None,
                                                          partition_strategy='div',
                                                          combiner='mean')
                con_fields = tf.reshape(con_embed, shape=[-1, pd['con_field_num'], feat_dim])
                return tf.concat([user_fields, doc_fields, con_fields], axis=1)

    def build_q_network(self, sub_net, scope, feat_dim):
        global g_training
        fields = self.build_embedding_layer(sub_net, scope, feat_dim)
        with tf.variable_scope(sub_net, reuse=tf.AUTO_REUSE):
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                fields_num = pd['user_field_num'] + pd['doc_field_num'] + pd['con_field_num']
                finp = tf.reshape(fields, shape=[-1, fields_num, feat_dim])
                # rnn seq embedding
                seq_embed = tf.reshape(self.field_interact(finp), (-1, self.seqlen, fields_num * feat_dim))
                gru = tf.nn.rnn_cell.GRUCell(seq_embed.get_shape().as_list()[-1])
                drop = tf.nn.rnn_cell.DropoutWrapper(gru, output_keep_prob=1.0 - pd['dropout'] if g_training else 1.)
                cell = tf.nn.rnn_cell.MultiRNNCell([drop for _ in range(pd['rnn_layer'])])
                state_embed, last_states = tf.nn.dynamic_rnn(cell=cell, dtype=tf.float32, inputs=seq_embed,
                                                             time_major=False)
                print('state_embed.shape:', state_embed.shape)
                state_dim = state_embed.get_shape().as_list()[-1]
                mlp_dims = [state_dim / 2, state_dim / 4, pd['hint_dim']]
                fc = state_embed
                for i in range(len(mlp_dims)):
                    fc = tf.layers.dense(fc, mlp_dims[i], name='mlp_' + str(i + 1), activation=tf.nn.tanh,
                                         reuse=tf.AUTO_REUSE)
                    fc = tf.layers.dropout(fc, rate=pd['dropout'], training=g_training)
                hint_layer = fc
                print('hint_layer.shape:', hint_layer.shape)
                q = tf.reshape(tf.layers.dense(hint_layer, 1), (-1, self.seqlen))
                print('q.shape:', q.shape)
                return hint_layer, q

    def teacher_learn(self, sess, ph_dict):
        loss, _ = sess.run([self.teacher_loss, self.teacher_opt], feed_dict={self.ph_reward: ph_dict['reward'],
                                                                             self.ph_seqlen: pd['rnn_seq_len'],
                                                                             self.sph_user: ph_dict['user'],
                                                                             self.sph_doc: ph_dict['doc'],
                                                                             self.sph_con: ph_dict['con']})
        self.teacher_loss_val += loss

    def student_learn(self, sess, ph_dict):
        loss, _ = sess.run([self.student_loss, self.student_opt], feed_dict={self.ph_reward: ph_dict['reward'],
                                                                             self.ph_seqlen: pd['rnn_seq_len'],
                                                                             self.sph_user: ph_dict['user'],
                                                                             self.sph_doc: ph_dict['doc'],
                                                                             self.sph_con: ph_dict['con']})
        self.student_loss_val += loss

    def inference_candi_pool(self, sess, ph_dict):
        tq, sq = sess.run([self.TMQ, self.SMQ], feed_dict={self.sph_user: ph_dict['user'],
                                                           self.sph_doc: ph_dict['doc'],
                                                           self.sph_con: ph_dict['con'],
                                                           self.ph_seqlen: 1})
        return tq, sq

    def topk_distill(self, sess, ph_dict):
        loss, _ = sess.run([self.hint_loss, self.topk_distill_opt],
                           feed_dict={self.ph_reward: ph_dict['reward'],
                                      self.ph_seqlen: 1,
                                      self.sph_user: ph_dict['user'],
                                      self.sph_doc: ph_dict['doc'],
                                      self.sph_con: ph_dict['con']})
        self.topk_loss_val += loss

    def eval(self, sess, ph_dict):
        tq, sq = sess.run([self.TMQ, self.SMQ], feed_dict={self.sph_user: ph_dict['user'],
                                                           self.sph_doc: ph_dict['doc'],
                                                           self.sph_con: ph_dict['con'],
                                                           self.ph_seqlen: pd['rnn_seq_len']})
        return tq, sq


def handle(sess, net, sess_data):
    def gen_sparse_tensor(fs):
        global g_dr
        kk, vv = [], []
        for i in range(len(fs)):
            ff = fs[i]
            assert (isinstance(ff, set))
            ff = list(ff)
            for k in range(len(ff)):
                kk.append(np.array([i, k], dtype=np.int32))
                vv.append(ff[k])
        return tf.SparseTensorValue(kk, vv, [len(fs), g_dr.unique_feature_num()])

    global g_rb, g_topk_pool
    g_rb.save(sess_data)
    while g_rb.has_batch():
        user, doc, con, rwd, rtn = g_rb.next_batch()
        cps = pd['candi_pool_size']
        g_topk_pool[0].extend(user)
        g_topk_pool[1].extend(doc)
        g_topk_pool[2].extend(con)
        g_topk_pool[3].extend(rwd)
        for i in range(len(g_topk_pool)):
            g_topk_pool[i] = g_topk_pool[i][-cps:]
        phd = {}
        rec_user = np.array(user).reshape(pd['batch_size'] * pd['rnn_seq_len'] * pd['user_field_num'])
        phd['user'] = gen_sparse_tensor(rec_user)
        rec_doc = np.array(doc).reshape(pd['batch_size'] * pd['rnn_seq_len'] * pd['doc_field_num'])
        phd['doc'] = gen_sparse_tensor(rec_doc)
        rec_con = np.array(con).reshape(pd['batch_size'] * pd['rnn_seq_len'] * pd['con_field_num'])
        phd['con'] = gen_sparse_tensor(rec_con)
        phd['reward'] = rwd
        global g_batch_counter, g_training
        g_batch_counter += 1
        if g_training:
            net.teacher_learn(sess, phd)
            net.student_learn(sess, phd)
            if len(g_topk_pool[0]) == cps:
                # eval all items in candidate pool
                topk_user = np.array(g_topk_pool[0])
                phd['user'] = gen_sparse_tensor(topk_user.reshape(-1))
                topk_doc = np.array(g_topk_pool[1])
                phd['doc'] = gen_sparse_tensor(topk_doc.reshape(-1))
                topk_con = np.array(g_topk_pool[2])
                phd['con'] = gen_sparse_tensor(topk_con.reshape(-1))
                topk_rwd = np.array(g_topk_pool[3])
                tq, sq = net.inference_candi_pool(sess, phd)
                # select top-k items of teacher and student
                tq = tq.reshape((1, -1))
                sq = sq.reshape((1, -1))
                arg_tq = np.argsort(-tq, axis=1).take(np.arange(pd['explore_filter_topk']))
                arg_sq = np.argsort(-sq, axis=1).take(np.arange(pd['explore_filter_topk']))
                ef_user = np.concatenate([topk_user.take(arg_tq, axis=0), topk_user.take(arg_sq, axis=0)], axis=0)
                ef_doc = np.concatenate([topk_doc.take(arg_tq, axis=0), topk_doc.take(arg_sq, axis=0)], axis=0)
                ef_con = np.concatenate([topk_con.take(arg_tq, axis=0), topk_con.take(arg_sq, axis=0)], axis=0)
                ef_rwd = np.concatenate([topk_rwd.take(arg_tq, axis=0), topk_rwd.take(arg_sq, axis=0)], axis=0)
                phd['user'] = gen_sparse_tensor(ef_user.reshape(-1))
                phd['doc'] = gen_sparse_tensor(ef_doc.reshape(-1))
                phd['con'] = gen_sparse_tensor(ef_con.reshape(-1))
                phd['reward'] = ef_rwd
                # top-k distillation
                if pd['enable_distill'] and pd['enable_topk_distill']:
                    net.topk_distill(sess, phd)
            if g_batch_counter % pd['double_networks_sync_freq'] == 0:
                print('Run soft replacement for main networks and target networks...')
                sess.run(net.teacher_sync_op)
                sess.run(net.student_sync_op)
        else:
            tq, sq = net.eval(sess, phd)
            tq = tq.reshape([-1])
            sq = sq.reshape([-1])
            global g_working_mode
            for i in range(len(rtn)):
                if 'local_predict' == g_working_mode:
                    print('%s %s %s' % (rwd[i], tq[i], sq[i]))


def local_run():
    global_step = tf.train.get_or_create_global_step()
    sess = tf.Session()
    net = DRL4Rec(global_step)
    saver = tf.train.Saver(max_to_keep=1)
    g_init_op = tf.global_variables_initializer()
    if os.path.exists('./ckpt'):
        model_file = tf.train.latest_checkpoint('ckpt/')
        saver.restore(sess, model_file)
    else:
        sess.run(g_init_op)
        os.system('mkdir ckpt')
    global g_batch_counter
    for k in range(pd['num_epochs'] if g_training else 1):
        if k > 0:
            g_dr.load('sample.data')
        data = g_dr.next()
        while data is not None:
            handle(sess, net, data)
            data = g_dr.next()
            if g_training and g_batch_counter % 10 == 0:
                print(
                    '>>> epoch %d --- batch %d --- teacher net loss = %f --- student net loss = %f --- top-k distill loss = %f' % (
                        k, g_batch_counter, net.teacher_loss_val / (g_batch_counter + 1e-6),
                        net.student_loss_val / (g_batch_counter + 1e-6),
                        net.topk_loss_val / (g_batch_counter + 1e-6)))
    saver.save(sess, 'ckpt/drl_rec.ckpt')


if __name__ == '__main__':
    g_working_mode = 'local_train'
    commander = {
        'local_train': local_run,
        'local_predict': local_run
    }
    if g_working_mode not in commander:
        print('your working mode(%s) not recognized!!!' % g_working_mode)
        sys.exit(1)
    g_training = True if g_working_mode == 'local_train' else False
    print('>>> working_model:', g_working_mode)
    print('>>> is_training:', g_training)
    g_dr.load('sample.data')
    g_rb = RB(pd['batch_size'], pd['rnn_seq_len'], pd['user_field_num'], pd['doc_field_num'], pd['con_field_num'])
    commander[g_working_mode]()
