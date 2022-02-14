#!encoding:utf-8
"""
    Model Hyper Parameter Dict
    @2021-01-04
    by modriczhang
"""
param_dict = {
    'num_epochs': 1,  # training epoch
    'teacher_feat_dim': 16,  # feature embedding dimension
    'student_feat_dim': 8,  # feature embedding dimension
    'user_field_num': 5,  # number of user feature fields
    'doc_field_num': 5,  # number of item feature fields
    'con_field_num': 5,  # number of context feature fields
    'num_epochs': 10,  # training epoch
    'batch_size': 256,  # batch size
    'teacher_lr': 0.0002,  # learning rate
    'student_lr': 0.0002,  # learning rate
    'topk_distill_lr': 0.0001,  # learning rate
    'enable_topk_distill': 1,  # learning rate
    'enable_distill': 1,  # learning rate
    'enable_confidence': 1,  # learning rate
    'distill_temperature': 10.0,  # learning rate
    'hint_dim': 32,  # learning rate
    'dropout': 0.3,  # learning rate of q/value network
    'rl_gamma': 0.4,  # gamma in rl
    'grad_clip': 5,  # grad clip
    'rnn_seq_len': 10,  # rnn sequence length
    'rnn_layer': 1,  # layer number of RNN
    'candi_pool_size': 1000,  # candidate pool of top-k distillation
    'explore_filter_topk': 10,  # topk of explore filter module
    'head_num': 2,  # head number for all self-attention units
    'encoder_layer': 1,  # encoder layer
    'decoder_layer': 1,  # decoder layer
    'double_networks_sync_freq': 10,  # sync frequency for both policy and value network
    'double_networks_sync_step': 0.005,  # sync step is designed with reference to DeepMind
}
