from dataclasses import dataclass
import torch


@dataclass
class Config:
    """notebooks 디렉토리의 주피터 환경에서 아래의 configuration을 활용할 수 있도록 구성
    DOT + SOMETHING: '../something/something' <- 디렉토리 경로를 바꿔주게 됨
    """

    ValidSize: float = 0.1
    Label: str = "./input/data/label_type.pkl"
    Label41: str = "./input/data/label_type41.pkl"
    Logs: str = "./logs"
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

    # Data Configuration
    Train: str = "./input/data/train/train.tsv"
    TrainUp: str = "./preprocessed/train_upsampled.csv"
    TrainAug: str = "./preprocessed/train_augmented.csv"
    Train41: str = "./preprocessed/train41.csv" # 관계 없음(0) 레이블을 제외한 41가지 클래스의 데이터
    TrainBin: str = "./preprocessed/train_binary.csv" # 관계 있음/없음의 이진 분류 데이터

    # 리더보드 제출용 테스트 데이터
    TestEn: str = "./preprocessed/test/test_english.csv" # 영어
    TestFr: str = "./preprocessed/test/test_french.csv" # 프랑스어
    TestKr: str = "./preprocessed/test/test_korean.csv" # 한국어
    TestSp: str = "./preprocessed/test/test_spanish.csv" # 스페인어
    
    # 단일 언어 K-Fold 학습을 위한 데이터셋
    TrainMono1: str = './preprocessed/kfold_monolingual/train/kfold_0_train_monolingual.csv' # 학습 데이터(fold1)
    TrainMono2: str = './preprocessed/kfold_monolingual/train/kfold_1_train_monolingual.csv' # 학습 데이터(fold2)
    TrainMono3: str = './preprocessed/kfold_monolingual/train/kfold_2_train_monolingual.csv' # 학습 데이터(fold3)
    TrainMono4: str = './preprocessed/kfold_monolingual/train/kfold_3_train_monolingual.csv' # 학습 데이터(fold4)
    TrainMono5: str = './preprocessed/kfold_monolingual/train/kfold_4_train_monolingual.csv' # 학습 데이터(fold5)
    
    ValidMono1: str = './preprocessed/kfold_monolingual/valid/kfold_0_test_monolingual.csv' # 검증 데이터(fold1)
    ValidMono2: str = './preprocessed/kfold_monolingual/valid/kfold_1_test_monolingual.csv' # 검증 데이터(fold2)
    ValidMono3: str = './preprocessed/kfold_monolingual/valid/kfold_2_test_monolingual.csv' # 검증 데이터(fold3)
    ValidMono4: str = './preprocessed/kfold_monolingual/valid/kfold_3_test_monolingual.csv' # 검증 데이터(fold4)
    ValidMono5: str = './preprocessed/kfold_monolingual/valid/kfold_4_test_monolingual.csv' # 검증 데이터(fold5)
        
    
    # 다중 언어 단일 모델 학습을 위한 데이터셋
    TrainMulti36k: str = './preprocessed/singular_multilingual_36k/train_multilingual.csv' # 학습 데이터

    ValidMultiSingularEn: str = "./preprocessed/valid_en_multilingual_singular.csv" # 검증 데이터(영어)
    ValidMultiSingularFr: str = "./preprocessed/valid_fr_multilingual_singular.csv" # 검증 데이터(프랑스어)
    ValidMultiSingularKr: str = "./preprocessed/valid_kr_multilingual_singular.csv" # 검증 데이터(한국어)
    ValidMultiSingularSp: str = "./preprocessed/valid_sp_multilingual_singular.csv" # 검증 데이터(스페인어)
    
    # 다중 언어 K-Fold 학습을 위한 데이터셋
    TrainMulti1: str = './preprocessed/kfold_multilingual/train/kfold_0_train_multilingual.csv' # 학습 데이터(fold1)
    TrainMulti2: str = './preprocessed/kfold_multilingual/train/kfold_1_train_multilingual.csv' # 학습 데이터(fold2)
    TrainMulti3: str = './preprocessed/kfold_multilingual/train/kfold_2_train_multilingual.csv' # 학습 데이터(fold3)
    TrainMulti4: str = './preprocessed/kfold_multilingual/train/kfold_3_train_multilingual.csv' # 학습 데이터(fold4)
    TrainMulti5: str = './preprocessed/kfold_multilingual/train/kfold_4_train_multilingual.csv' # 학습 데이터(fold5)

    # 다중 언어 K-Fold 학습 시 fold별 검증 데이터
    ValidFold0kr: str = "./preprocessed/valid/fold0/kfold_0_test_multilingual_kr.csv" # 한국어
    ValidFold1kr: str = "./preprocessed/valid/fold1/kfold_1_test_multilingual_kr.csv"
    ValidFold2kr: str = "./preprocessed/valid/fold2/kfold_2_test_multilingual_kr.csv"
    ValidFold3kr: str = "./preprocessed/valid/fold3/kfold_3_test_multilingual_kr.csv"
    ValidFold4kr: str = "./preprocessed/valid/fold4/kfold_4_test_multilingual_kr.csv"
    
    ValidFold0fr: str = "./preprocessed/valid/fold0/kfold_0_test_multilingual_fr.csv" # 프랑스어
    ValidFold1fr: str = "./preprocessed/valid/fold1/kfold_1_test_multilingual_fr.csv"
    ValidFold2fr: str = "./preprocessed/valid/fold2/kfold_2_test_multilingual_fr.csv"
    ValidFold3fr: str = "./preprocessed/valid/fold3/kfold_3_test_multilingual_fr.csv"
    ValidFold4fr: str = "./preprocessed/valid/fold4/kfold_4_test_multilingual_fr.csv"
    
    ValidFold0sp: str = "./preprocessed/valid/fold0/kfold_0_test_multilingual_sp.csv" # 스페인어
    ValidFold1sp: str = "./preprocessed/valid/fold1/kfold_1_test_multilingual_sp.csv"
    ValidFold2sp: str = "./preprocessed/valid/fold2/kfold_2_test_multilingual_sp.csv"
    ValidFold3sp: str = "./preprocessed/valid/fold3/kfold_3_test_multilingual_sp.csv"
    ValidFold4sp: str = "./preprocessed/valid/fold4/kfold_4_test_multilingual_sp.csv"
    
    ValidFold0en: str = "./preprocessed/valid/fold0/kfold_0_test_multilingual_en.csv" # 영어
    ValidFold1en: str = "./preprocessed/valid/fold1/kfold_1_test_multilingual_en.csv"
    ValidFold2en: str = "./preprocessed/valid/fold2/kfold_2_test_multilingual_en.csv"
    ValidFold3en: str = "./preprocessed/valid/fold3/kfold_3_test_multilingual_en.csv"
    ValidFold4en: str = "./preprocessed/valid/fold4/kfold_4_test_multilingual_en.csv"
    

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
    BertBase: str = "BertModel"
    BertSequenceClf: str = "BertForSequenceClassification"
    
    VanillaBert: str = "VanillaBert"
    VanillaBert_v2: str = "VanillaBert_v2"
    
    
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
        
