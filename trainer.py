"""transformers의 Trainer 활용 학습 시 사용되는 모듈"""

import argparse
import os
import warnings
from tokenization import load_tokenizer
from transformers import TrainingArguments, Trainer
import wandb
from models import load_model
from dataset import REDatasetForTrainer, split_data, load_data
from tokenization import tokenize
from preprocessing import preprocess_text
from utils import get_timestamp, get_timestamp, set_seed, save_json
from evaluation import compute_metrics
from config import ModelType, Config, Optimizer, PreTrainedType, PreProcessType, Loss

warnings.filterwarnings("ignore")
os.environ["WANDB_LOG_MODEL"] = "true"
os.environ["WANDB_WATCH"] = "all"

TOTAL_SAMPLES = 9000


def train(
    model_type: str = ModelType.VanillaBert,  # 불러올 모델 프레임
    pretrained_type: str = PreTrainedType.MultiLingual,  # 모델에 활용할 Pretrained BERT Backbone 이름
    num_classes: int = Config.NumClasses,  # 카테고리 수
    pooler_idx: int = 0,  # 인코딩 결과로부터 추출할 hidden state. 0: [CLS]
    dropout: float = 0.8,
    load_state_dict: str = None,  # (optional) 저장한 weight 경로
    data_root: str = Config.Train,  # 학습 데이터 경로
    preprocess_type: str = PreProcessType.Base,  # 텍스트 전처리 타입
    epochs: int = Config.Epochs,
    valid_size: float = Config.ValidSize,  # 학습 데이터 중 검증에 활용할 데이터 비율
    train_batch_size: int = Config.Batch32,
    valid_batch_size: int = 512,
    optim_type: str = Optimizer.Adam,
    loss_type: str = Loss.CE,
    lr: float = Config.LR,
    lr_scheduler: str = Optimizer.CosineAnnealing,
    device: str = Config.Device,
    seed: int = Config.Seed,
    save_path: str = Config.CheckPoint,
):
    K_FOLD = False
    set_seed(seed)

    # tokenization phase
    tokenizer = load_tokenizer(model_type=model_type, preprocess_type=preprocess_type)

    # Load data
    # Monolingual K-Fold DataL just for KOR
    if os.path.basename(data_root).startswith("kfold") and "monolingual" in data_root:
        print(f"TRAIN TYPE: monolingual K-Fold - {os.path.basename(data_root)}")
        K_FOLD = True
        fold_num = int(os.path.basename(data_root).split("_")[1])
        train_data = load_data(f"./preprocessed/kfold_{fold_num}_train_monolingual.csv")
        valid_data = load_data(f"./preprocessed/kfold_{fold_num}_test_monolingual.csv")
        train_data = preprocess_text(train_data, model_type, preprocess_type)
        valid_data = preprocess_text(valid_data, model_type, preprocess_type)

    # Multilingual K-Fold DataL for KOR, ENG, FRN, SPN
    elif (
        os.path.basename(data_root).startswith("kfold") and "multilingual" in data_root
    ):
        print(f"TRAIN TYPE: multilingual K-Fold - {os.path.basename(data_root)}")
        K_FOLD = True
        fold_num = int(os.path.basename(data_root).split("_")[1])
        train_data = load_data(
            f"./preprocessed/kfold_{fold_num}_train_multilingual.csv"
        )
        valid_data = load_data(f"./preprocessed/kfold_{fold_num}_test_multilingual.csv")
        train_data = preprocess_text(train_data, model_type, preprocess_type)
        valid_data = preprocess_text(valid_data, model_type, preprocess_type)

    # For singualar model which contains monolingual, multilingual
    else:
        print("TRAIN TYPE: singular")
        data = load_data(data_root)
        data = preprocess_text(data, model_type, preprocess_type)

    # for K-Fold learning
    if K_FOLD:
        tokenized_train = tokenize(train_data, tokenizer)
        tokenized_valid = tokenize(valid_data, tokenizer)

        train_labels = train_data["label"].values
        valid_labels = valid_data["label"].values

        train_dataset = REDatasetForTrainer(tokenized_train, train_labels)
        valid_dataset = REDatasetForTrainer(tokenized_valid, valid_labels)

    # for singular learning
    elif valid_size > 0:
        train_data, valid_data = split_data(data, valid_size)
        tokenized_train = tokenize(train_data, tokenizer)
        tokenized_valid = tokenize(valid_data, tokenizer)

        train_labels = train_data["label"].values
        valid_labels = valid_data["label"].values

        train_dataset = REDatasetForTrainer(tokenized_train, train_labels)
        valid_dataset = REDatasetForTrainer(tokenized_valid, valid_labels)

    # for singular learning with WHOLE train data. TODO: maybe insufficient implementation
    else:
        tokenized_train = tokenize(data, tokenizer)
        train_labels = data["label"].values
        train_dataset = REDatasetForTrainer(tokenized_train, train_labels)

    model = load_model(
        model_type, pretrained_type, num_classes, load_state_dict, pooler_idx, dropout
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.train()

    # train configuration phase
    training_args = TrainingArguments(
        output_dir=save_path,
        save_total_limit=1,
        save_steps=200,
        evaluation_strategy="epoch",
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=valid_batch_size,
        warmup_steps=300,
        weight_decay=0.01,
        dataloader_num_workers=2,
        label_smoothing_factor=0.5,
        logging_dir="./logs",
        logging_steps=400,
        report_to="wandb",
        run_name=RUN_NAME,
    )

    # train & validation phase
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    TIMESTAMP = get_timestamp()  # used as an identifier in model save phase
    LOAD_STATE_DICT = None

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default=ModelType.XLMSequenceClfL)
    parser.add_argument(
        "--pretrained-type", type=str, default=PreTrainedType.XLMRobertaL
    )
    parser.add_argument("--num-classes", type=int, default=Config.NumClasses)
    parser.add_argument("--pooler-idx", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--load-state-dict", type=str, default=LOAD_STATE_DICT)
    parser.add_argument("--data-root", type=str, default=Config.TrainMulti5)
    parser.add_argument("--preprocess-type", type=str, default=PreProcessType.ES)
    parser.add_argument("--epochs", type=int, default=Config.Epochs)
    parser.add_argument("--valid-size", type=int, default=0.1)
    parser.add_argument("--train-batch-size", type=int, default=Config.Batch32)
    parser.add_argument("--valid-batch-size", type=int, default=512)
    parser.add_argument("--optim-type", type=str, default=Optimizer.AdamW)
    parser.add_argument("--loss-type", type=str, default=Loss.CE)
    parser.add_argument("--lr", type=float, default=Config.LRRoberta)
    parser.add_argument("--lr-scheduler", type=str, default=Optimizer.LinearWarmUp)
    parser.add_argument("--device", type=str, default=Config.Device)
    parser.add_argument("--seed", type=int, default=Config.Seed)
    parser.add_argument("--save-path", type=str, default=Config.CheckPoint)

    args = parser.parse_args()

    # register logs to wandb
    RUN_NAME = (
        TIMESTAMP + "_" + args.model_type
    )  # save file name: [MODEL-TYPE]_[PRETRAINED-TYPE]_[EPOCH][ACC][LOSS][ID].pth
    run = wandb.init(
        project="pstage-klue",
        name=RUN_NAME,
        tags=[
            args.model_type,
            os.path.basename(args.pretrained_type),
            str(args.num_classes),
        ],
        group=args.model_type,
    )
    wandb.config.update(args)

    # make checkpoint directory to save model during train
    checkpoint_dir = f"{os.path.basename(args.model_type)}_{os.path.basename(args.pretrained_type)}_{TIMESTAMP}"
    if checkpoint_dir not in os.listdir(args.save_path):
        os.mkdir(os.path.join(args.save_path, checkpoint_dir))
    args.save_path = os.path.join(args.save_path, checkpoint_dir)

    # save param dict
    save_param = vars(args)
    save_param["device"] = save_param["device"].type
    save_json(os.path.join(args.save_path, "param_dict.json"), save_param)

    print("=" * 100)
    print(args)
    print("=" * 100)
    train(**vars(args))

    run.finish()  # finish wandb's session
