#!encoding:utf-8
import sys

"""
    Static Replay Buffer
    @2020-08-17
    by modriczhang
"""

class RB(object):
    
    def __init__(self, batch_num, rnn_seq_len, user_field_num, doc_field_num, con_field_num):
        self.batch_num = batch_num
        self.rnn_seq_len = rnn_seq_len
        self.rec_step = self.batch_num * self.rnn_seq_len
        self.user_field_num = user_field_num
        self.doc_field_num = doc_field_num
        self.con_field_num = con_field_num
        self.user_buffer = []
        self.doc_buffer = []
        self.con_buffer = []
        self.reward_buffer = []
        self.return_buffer = []

    def save(self, sess_data):
        for data in sess_data:
            if len(data) != 7:
                print('sample format error, colume should be 7 not %d' % len(data))
                return False
        for data in sess_data:
            if len(data[2]) != self.user_field_num:
                print('sample format error, len(user)=%d' % len(data[2]))
                return False
            self.user_buffer.append(data[2])
            if len(data[3]) != self.doc_field_num:
                print('sample format error, len(doc)=%d' % len(data[3]))
                return False
            self.doc_buffer.append(data[3])
            if len(data[4]) != self.con_field_num:
                print('sample format error, len(con)=%d' % len(data[4]))
                return False
            self.con_buffer.append(data[4])
            self.reward_buffer.append(data[5])
            self.return_buffer.append(data[6])
    
    def has_batch(self):
        return len(self.reward_buffer) >= self.rec_step
    
    def dump(self):
        return 'buffer_size:' + str(len(self.user_buffer)) + ',' \
                              + str(len(self.doc_buffer)) + ','  \
                              + str(len(self.con_buffer))

    def next_batch(self):
        if not self.has_batch():
            raise Exception('replay buffer is almost empty')
        user = self.user_buffer[:self.rec_step]
        self.user_buffer = self.user_buffer[self.rec_step:]
        doc = self.doc_buffer[:self.rec_step]
        self.doc_buffer = self.doc_buffer[self.rec_step:]
        con = self.con_buffer[:self.rec_step]
        self.con_buffer = self.con_buffer[self.rec_step:]
        rwd = self.reward_buffer[:self.rec_step]
        self.reward_buffer = self.reward_buffer[self.rec_step:]
        rtn = self.return_buffer[:self.rec_step]
        self.return_buffer = self.return_buffer[self.rec_step:]
        return user, doc, con, rwd, rtn
