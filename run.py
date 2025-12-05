import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
import numpy as np
import pandas as pd
import re

# --------------------------
# LOAD .ENV FOR HF TOKEN
# --------------------------
from dotenv import load_dotenv
load_dotenv()  # Loads .env from project root
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
# --------------------------

os.environ["TOKENIZERS_PARALLELISM"] = "true"

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task: M multivariate, S univariate, MS multi→single')
    parser.add_argument('--target', type=str, default='OT', help='target feature')
    parser.add_argument('--freq', type=str, default='h', help='freq encoding')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='label length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction length')
    parser.add_argument('--inverse', action='store_true', default=False)

    # model define
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--dec_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--activation', type=str, default='gelu')

    # optimization
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=5)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)

    # --------------------------
    # LLM + GPT-2 SETTINGS
    # --------------------------
    parser.add_argument('--llm_model', type=str, default="GPT2", help='LLM model (GPT2/BERT/LLAMA)')
    parser.add_argument('--llm_dim', type=int, default=768)
    parser.add_argument('--llm_layers', type=int, default=6)
    
    # load HuggingFace token from .env automatically
    parser.add_argument('--huggingface_token', type=str, default=HF_TOKEN,
                        help='HuggingFace token loaded from .env')

    parser.add_argument('--text_path', type=str, default="None")
    parser.add_argument('--text_len', type=int, default=3)
    parser.add_argument('--prompt_weight', type=float, default=0.01)
    parser.add_argument('--pool_type', type=str, default='avg')
    parser.add_argument('--date_name', type=str, default='date')
    parser.add_argument('--seed', type=int, default=2024)

    args = parser.parse_args()

    # --------------------------
    # FIX MODEL SETTINGS FOR GPT-2
    # --------------------------

    if args.llm_model == "GPT2":
        args.llm_dim = 768
    if args.llm_model == "GPT2L":
        args.llm_dim = 1280
    if args.llm_model == "GPT2XL":
        args.llm_dim = 1600

    # --------------------------
    # FORCE UNIVARIATE MODE FOR STOCK PRICE TRAINING
    # --------------------------
    args.features = 'S'
    args.enc_in = 1
    args.dec_in = 1
    args.c_out = 1

    print("✔ Using univariate mode (stock price forecasting).")
    print("✔ HF Token loaded:", args.huggingface_token is not None)

    # --------------------------
    # SEEDING + GPU
    # --------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.use_gpu = True if torch.cuda.is_available() else False
    print("CUDA Available:", torch.cuda.is_available())

    print("\n===== FINAL ARGUMENTS =====")
    print_args(args)

    # --------------------------
    # RUN THE EXPERIMENT
    # --------------------------

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    else:
        Exp = Exp_Long_Term_Forecast  # default

    exp = Exp(args)

    if args.is_training:
        setting = f"{args.model_id}_{args.model}_{args.pred_len}"
        print(f"\n🚀 Training Started: {setting}\n")
        exp.train(setting)

        print(f"\n🔍 Testing Started: {setting}\n")
        exp.test(setting, test=1)

        torch.cuda.empty_cache()

    else:
        setting = f"{args.model_id}_{args.model}_{args.pred_len}"
        print(f"\n🔍 Testing Only: {setting}\n")
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
