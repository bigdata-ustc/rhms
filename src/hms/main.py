# -*- coding: utf-8 -*-

import os
import logging
import random
import torch
import torch.nn as nn

from config import get_args
from model import Encoder, Decoder, Seq2seq
from utils import DataLoader, Checkpoint, Evaluator, SupervisedTrainer

def init_env():
    args = get_args()

    args.use_cuda = args.cuda_id is not None
    if args.use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
        if not torch.cuda.is_available():
            args.use_cuda = False

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    log_path = args.log
    if log_path is None:
        log_path = os.path.join("experiment", f"{args.checkpoint}.log")
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", filename=log_path)
    logging.info('\n' + '\n'.join([f"\t{'['+k+']':20}\t{v}" for k, v in dict(args._get_kwargs()).items()]))
    return args

def init_ckpt(args):
    checkpoint_path = os.path.join("experiment", args.checkpoint)
    if not os.path.exists(checkpoint_path):
        logging.info(f'create checkpoint directory {checkpoint_path} ...')
        os.makedirs(checkpoint_path)
    Checkpoint.set_ckpt_path(checkpoint_path)
    return

def init_model_data(args):
    data_loader = DataLoader(
        mode=args.mode,
        embed_dim=args.embed
    )

    embed_model = nn.Embedding(data_loader.vocab_len, args.embed)
    if not args.bert:
        embed_model.weight.data.copy_(data_loader.embed_vectors)
    encode_model = Encoder(
        embed_model=embed_model,
        hidden_dim=args.hidden,
        span_size=data_loader.span_size,
        dropout=args.dropout,
        bert_path=args.bert
    )
    
    decode_model = Decoder(
        embed_model=embed_model,
        op_set=data_loader.op_set,
        vocab_dict=data_loader.vocab_dict,
        class_list=data_loader.class_list,
        hidden_dim=args.hidden,
        dropout=args.dropout,
        use_cuda=args.use_cuda
    )
    
    seq2seq = Seq2seq(encode_model, decode_model)
    return seq2seq, data_loader

def train(args):
    seq2seq, data_loader = init_model_data(args)
    if args.use_cuda:
        seq2seq = seq2seq.cuda()

    st = SupervisedTrainer(
        class_dict=data_loader.class_dict,
        class_list=data_loader.class_list,
        use_cuda=args.use_cuda
    )

    logging.info('start training ...')
    max_ans_acc = st.train(
        model=seq2seq, 
        data_loader=data_loader,
        batch_size=args.batch,
        n_epoch=args.epoch,
        resume=args.resume,
        optim_lr=args.lr,
        optim_weight_decay=args.weight_decay,
        scheduler_step_size=args.step,
        scheduler_gamma=args.gamma,
        test_beam=args.beam
    )
    return max_ans_acc

def test(args):
    seq2seq, data_loader = init_model_data(args)
    resume_checkpoint = Checkpoint.load(model_only=True)
    seq2seq.load_state_dict(resume_checkpoint.model)
    if args.use_cuda:
        seq2seq = seq2seq.cuda()

    evaluator = Evaluator(
        class_dict=data_loader.class_dict,
        class_list=data_loader.class_list,
        use_cuda=args.use_cuda
    )
    test_temp_acc, test_ans_acc = evaluator.evaluate(
        model=seq2seq,
        data_loader=data_loader,
        data_list=data_loader.test_list,
        template_flag=True,
        template_len=False,
        batch_size=1,
        beam_width=args.beam,
        test_log=args.test_log
    )
    logging.info(f"temp_acc: {test_temp_acc}, ans_acc: {test_ans_acc}")
    return test_ans_acc

def run_mode(args):
    if args.mode == "train":
        acc = train(args)
    elif args.mode == "test":
        acc = test(args)
    else:
        logging.info(f"unknown mode {args.mode}")
        acc = None
    return acc

def main():
    args = init_env()
    init_ckpt(args)
    acc = run_mode(args)
    logging.info(f"accuracy: {acc}")
    return

if __name__ == "__main__":
    main()
