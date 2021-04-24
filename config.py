import os
from dataclasses import dataclass
import torch


TRAIN = 
TRAIN_UP = 
TRAIN_AUG = 
TRAIN_41 = 
TRAIN_BIN = 
TEST = "./input/data/test/test.tsv"
LABEL = "./input/data/label_type.pkl"
LABEL41 = "./input/data/label_type41.pkl"
LOGS = "./logs"
CKPT = "./saved_models"

DOT = "."


@dataclass
class Config:
    """notebooks 디렉토리의 주피터 환경에서 아래의 configuration을 활용할 수 있도록 구성
    DOT + SOMETHING: '../something/something' <- 디렉토리 경로를 바꿔주게 됨
    """

    Train: str = "./input/data/train/train.tsv"
    TrainUp: str = "./preprocessed/train_upsampled.csv"
    TrainAug: str = "./preprocessed/train_augmented.csv"
    Train41: str = "./preprocessed/train41.csv" # 관계 없음(0) 레이블을 제외한 41가지 클래스의 데이터
    TrainBin: str = "./preprocessed/train_binary.csv" # 관계 있음/없음의 이진 분류 데이터
    
    # 단일 언어 K-Fold 앙상블을 위한 데이터셋
    TrainMono1: str = './preprocessed/kfold_0_train_monolingual.csv'
    TrainMono2: str = './preprocessed/kfold_1_train_monolingual.csv'
    TrainMono3: str = './preprocessed/kfold_2_train_monolingual.csv'
    TrainMono4: str = './preprocessed/kfold_3_train_monolingual.csv'
    TrainMono5: str = './preprocessed/kfold_4_train_monolingual.csv'
    
    #
    TestMono1: str = './preprocessed/kfold_0_test_monolingual.csv'
    TestMono2: str = './preprocessed/kfold_1_test_monolingual.csv'
    TestMono3: str = './preprocessed/kfold_2_test_monolingual.csv'
    TestMono4: str = './preprocessed/kfold_3_test_monolingual.csv'
    TestMono5: str = './preprocessed/kfold_4_test_monolingual.csv'
        
    
    # 다중 언어 단일 모델 학습을 위한 데이터셋
    TrainMultiAll: str = './preprocessed/train_multilingual.csv' # 학습 데이터

    ValidMultiSingularEn: str = "./preprocessed/valid_en_multilingual_singular.csv" # 검증 데이터(영어)
    ValidMultiSingularFr: str = "./preprocessed/valid_fr_multilingual_singular.csv" # 검증 데이터(프랑스어)
    ValidMultiSingularKr: str = "./preprocessed/valid_kr_multilingual_singular.csv" # 검증 데이터(한국어)
    ValidMultiSingularSp: str = "./preprocessed/valid_sp_multilingual_singular.csv" # 검증 데이터(스페인어)
    
    # 다중 언어 K-Fold 앙상블을 위한 데이터셋
    TrainMulti1: str = './preprocessed/kfold_0_train_multilingual.csv' # 학습 데이터(fold1)
    TrainMulti2: str = './preprocessed/kfold_1_train_multilingual.csv' # 학습 데이터(fold2)
    TrainMulti3: str = './preprocessed/kfold_2_train_multilingual.csv' # 학습 데이터(fold3)
    TrainMulti4: str = './preprocessed/kfold_3_train_multilingual.csv' # 학습 데이터(fold4)
    TrainMulti5: str = './preprocessed/kfold_4_train_multilingual.csv' # 학습 데이터(fold5)

    TestMulti1: str = './preprocessed/kfold_0_test_multilingual.csv'# 검증 데이터(fold1)
    TestMulti2: str = './preprocessed/kfold_1_test_multilingual.csv'# 검증 데이터(fold2)
    TestMulti3: str = './preprocessed/kfold_2_test_multilingual.csv'# 검증 데이터(fold3)
    TestMulti4: str = './preprocessed/kfold_3_test_multilingual.csv'# 검증 데이터(fold4)
    TestMulti5: str = './preprocessed/kfold_4_test_multilingual.csv'# 검증 데이터(fold5)
    
    # 다중 언어 K-Fold 학습 시 fold별 검증 데이터
    ValidFold0kr: str = "./preprocessed/kfold_0_test_multilingual_kr.csv"
    ValidFold1kr: str = "./preprocessed/kfold_1_test_multilingual_kr.csv"
    ValidFold2kr: str = "./preprocessed/kfold_2_test_multilingual_kr.csv"
    ValidFold3kr: str = "./preprocessed/kfold_3_test_multilingual_kr.csv"
    ValidFold4kr: str = "./preprocessed/kfold_4_test_multilingual_kr.csv"
    
    # 프랑스어
    ValidFold0fr: str = "./preprocessed/kfold_0_test_multilingual_fr.csv"
    ValidFold1fr: str = "./preprocessed/kfold_1_test_multilingual_fr.csv"
    ValidFold2fr: str = "./preprocessed/kfold_2_test_multilingual_fr.csv"
    ValidFold3fr: str = "./preprocessed/kfold_3_test_multilingual_fr.csv"
    ValidFold4fr: str = "./preprocessed/kfold_4_test_multilingual_fr.csv"
    
    # 스페인어
    ValidFold0sp: str = "./preprocessed/kfold_0_test_multilingual_sp.csv"
    ValidFold1sp: str = "./preprocessed/kfold_1_test_multilingual_sp.csv"
    ValidFold2sp: str = "./preprocessed/kfold_2_test_multilingual_sp.csv"
    ValidFold3sp: str = "./preprocessed/kfold_3_test_multilingual_sp.csv"
    ValidFold4sp: str = "./preprocessed/kfold_4_test_multilingual_sp.csv"
    
    # 영어
    ValidFold0en: str = "./preprocessed/kfold_0_test_multilingual_en.csv"
    ValidFold1en: str = "./preprocessed/kfold_1_test_multilingual_en.csv"
    ValidFold2en: str = "./preprocessed/kfold_2_test_multilingual_en.csv"
    ValidFold3en: str = "./preprocessed/kfold_3_test_multilingual_en.csv"
    ValidFold4en: str = "./preprocessed/kfold_4_test_multilingual_en.csv"
    
    
    TestEn: str = "./preprocessed/multilingual/test/test_english.csv"
    TestFr: str = "./preprocessed/multilingual/test/test_french.csv"
    TestKr: str = "./preprocessed/multilingual/test/test_korean.csv"
    TestSp: str = "./preprocessed/multilingual/test/test_spanish.csv"
            
        
    Test: str = TEST if os.path.isfile(TEST) else DOT + TEST
    SamplingWeights: str = './preprocessed/sampling_weights.pkl'
    SamplingWeightsP: str = './preprocessed/sampling_weights_modified.pkl'
    ValidSize: float = 0.1
    Label: str = LABEL if os.path.isfile(LABEL) else DOT + LABEL
    Label41: str = LABEL41 if os.path.isfile(LABEL41) else DOT + LABEL41
    Logs: str = LOGS if os.path.isfile(LOGS) else DOT + LOGS
    NumClasses: int = 42
    NumBinary: int = 2
    Num41: int = 41
    
    Epochs: int = 10

    Batch8: int = 8
    Batch16: int = 16
    Batch32: int = 32
    Batch64: int = 64
    Batch128: int = 128

    LRFaster: float = 5e-5
    LRFast: float = 25e-6
    LR: float = 1e-6
    LRSlow: float = 25e-7
    LRSlower: float = 1e-7
    LRRoberta: float = 1e-5

    Seed: int = 42
    Device: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CheckPoint: str = "./saved_models"
    SavePath: str = "./predictions"


@dataclass
class Optimizer:
    Adam: str = "Adam"
    AdamP: str = "AdamP"
    AdamW: str = "AdamW"
    SGD: str = "SGD"
    Momentum: str = "Momentum"
    CosineAnnealing: str = "CosineScheduler"
    LinearWarmUp = "LinearWarmUp"


@dataclass
class Loss:
    CE: str = "crossentropyloss"
    LS: str = "labelsmoothingLoss"


@dataclass
class PreProcessType:
    Base: str = "Base"  # No preprocessing
    ES: str = (
        "EntitySeparation"  # Entity Separation, method as baseline of boostcamp itself
    )
    ESP: str = "ESPositionEmbedding"  # Entity Separation with Position Embedding, add scalar for each values in entities
    EM: str = "EntityMarker"  # Entity Marking
    EMSP: str = "EntityMarkerSeparationPositionEmbedding"


@dataclass
class ModelType:
    VanillaBert: str = "VanillaBert"
    VanillaBert_v2: str = "VanillaBert_v2"
    Base: str = "BertModel"
    SequenceClf: str = "BertForSequenceClassification"
    KoELECTRAv3: str = "KoELECTRAv3"
    KoBert: str = "monologg/kobert"

    XLMSequenceClf: str = "XLMSequenceClf"
    XLMSequenceClfL: str = "XLMSequenceClfLarge"
    XLMBase: str = "XLMBase"


@dataclass
class PreTrainedType:
    MultiLingual: str = "bert-base-multilingual-cased"
    BaseUncased: str = "bert-base-uncased"
    KoELECTRAv3: str = "monologg/koelectra-base-v3-discriminator"
    KoBert: str = "monologg/kobert"
    XLMRoberta: str = "xlm-roberta-base"
    XLMRobertaL: str = "xlm-roberta-large"
        
