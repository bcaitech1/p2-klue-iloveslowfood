from torch import nn
from transformers import (
    BertModel,
    BertConfig,
    BertForSequenceClassification,
    ElectraModel,
    XLMRobertaForSequenceClassification,
    XLMRobertaConfig,
)
from config import ModelType, Config, ModelType, PreTrainedType


def load_model(
    model_type: str = ModelType.SequenceClf,
    pretrained_type: str = PreTrainedType.MultiLingual,
    num_classes: int = Config.NumClasses,
    load_state_dict: str = None,
    pooler_idx: int = 0,  # get last hidden state from CLS
    dropout: float = 0.5,
):
    """설정에 맞는 모델을 리턴하는 함수. HuggingFace의 transformers 라이브러리를 바탕으로 둠

    Args:
        model_type (str, optional): 불러올 모델 아키텍쳐. Defaults to ModelType.SequenceClf.
        pretrained_type (str, optional): 불러올 사전 학습 파라미터. Defaults to PreTrainedType.MultiLingual.
        num_classes (int, optional): 분류할 카테고리 수. Defaults to Config.NumClasses.
        load_state_dict (str, optional): 사전 학습한 파라미터를 불러올 경우 해당 경로를 입력. Defaults to None.
        pooler_idx (int, optional):
            마지막 hidden state의 몇 번째 벡터를 가져올 지 설정 0으로 설정할 경우 [CLS] 토큰에 대한 hidden state를 추출.
            VanillaBert, VanillaBert_v2에만 적용 가능. Defaults to 0.

    Returns:
        model(torch.nn): 학습에 활용할 모델
    """
    print("Load Model...", end="\t")
    # make BERT configuration

    # BertModel
    if model_type == ModelType.BertBase:
        if load_state_dict is not None:
            model = BertModel.from_pretrained(load_state_dict)

        else:
            bert_config = BertConfig.from_pretrained(pretrained_type)
            bert_config.num_labels = num_classes
            model = BertModel.from_pretrained(pretrained_type, config=bert_config)

    # XLM Roberta Base
    elif model_type == ModelType.XLMSequenceClf:
        if load_state_dict is not None:
            model = XLMRobertaForSequenceClassification.from_pretrained(load_state_dict)
        else:
            config = XLMRobertaConfig.from_pretrained(PreTrainedType.XLMRoberta)
            config.num_labels = num_classes
            model = XLMRobertaForSequenceClassification.from_pretrained(
                PreTrainedType.XLMRoberta, config=config
            )

    # XLM Roberta Large
    elif model_type == ModelType.XLMSequenceClfL:
        if load_state_dict is not None:
            model = XLMRobertaForSequenceClassification.from_pretrained(load_state_dict)
        else:
            config = XLMRobertaConfig.from_pretrained(PreTrainedType.XLMRobertaL)
            config.num_labels = num_classes
            model = XLMRobertaForSequenceClassification.from_pretrained(
                PreTrainedType.XLMRobertaL, config=config
            )

    # Bert for Sequence Classification
    elif model_type == ModelType.BertSequenceClf:
        if load_state_dict is not None:
            print(f"Load params from {load_state_dict}...", end="\t")
            model = BertForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=load_state_dict
            )
            print("done!")

        else:
            bert_config = BertConfig.from_pretrained(pretrained_type)
            bert_config.num_labels = num_classes
            model = BertForSequenceClassification.from_pretrained(
                pretrained_type, config=bert_config
            )

    elif model_type == ModelType.VanillaBert:
        bert_config = BertConfig.from_pretrained(pretrained_type)
        bert_config.num_labels = num_classes
        model = VanillaBert(
            model_type=ModelType.BertSequenceClf,
            pretrained_type=pretrained_type,
            num_labels=num_classes,
            pooler_idx=pooler_idx,
            dropout=dropout,
        )

    elif model_type == ModelType.VanillaBert_v2:
        bert_config = BertConfig.from_pretrained(pretrained_type)
        bert_config.num_labels = num_classes
        model = VanillaBert_v2(
            model_type=ModelType.SequenceClf,
            pretrained_type=pretrained_type,
            num_labels=num_classes,
            pooler_idx=pooler_idx,
        )
    elif model_type == ModelType.KoELECTRAv3:
        model = VanillaKoElectra(num_classes=num_classes)

    elif model_type == ModelType.XLMSequenceClfL:
        config = XLMRobertaConfig.from_pretrained(model_type)
        config.num_labels = num_classes
        model = XLMRobertaForSequenceClassification.from_pretrained(
            model_type, config=config
        )

    else:
        raise NotImplementedError()

    print("done!")
    return model


class VanillaKoBert(nn.Module):
    def __init__(self, num_classes, pooler_idx: int = 0):
        super(VanillaKoBert, self).__init__()
        self.backbone = BertModel.from_pretrained(PreTrainedType.KoBert)
        self.linear = nn.Linear(in_features=768, out_features=num_classes)
        self.idx = pooler_idx

    def forward(self, input_ids, token_type_ids, attention_mask):
        x = self.backbone(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        x = x.last_hidden_state[:, self.idx, :]
        output = self.linear(x)
        return output

    def resize_token_embeddings(self, new_num_tokens: int):
        self.backbone.resize_token_embeddings(new_num_tokens)


class VanillaKoElectra(nn.Module):
    def __init__(self, num_classes, pooler_idx: int = 0):
        super(VanillaKoElectra, self).__init__()
        self.backbone = ElectraModel.from_pretrained(PreTrainedType.KoELECTRAv3)
        self.linear = nn.Linear(in_features=768, out_features=num_classes)
        self.idx = pooler_idx

    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        x = self.backbone(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        x = x.last_hidden_state[:, self.idx, :]
        output = self.linear(x)
        return output

    def resize_token_embeddings(self, new_num_tokens: int):
        self.backbone.resize_token_embeddings(new_num_tokens)


class VanillaBert_v2(nn.Module):
    def __init__(
        self,
        model_type: str = ModelType.SequenceClf, 
        pretrained_type: str = PreTrainedType.MultiLingual,
        num_labels: int = Config.NumClasses,
        pooler_idx: int = 0,
    ):
        super(VanillaBert_v2, self).__init__()
        bert = self.load_bert(
            model_type=model_type,
            pretrained_type=pretrained_type,
            num_labels=num_labels,
        )
        self.backbone = bert.bert
        self.dropout = bert.dropout
        self.clf = bert.classifier
        self.idx = 0 if pooler_idx == 0 else pooler_idx

    def forward(self, input_ids, token_type_ids, attention_mask):
        x = self.backbone(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        x = x.last_hidden_state[:, self.idx, :]
        x = self.dropout(x)
        output = self.clf(x)
        return output

    @staticmethod
    def load_bert(model_type, pretrained_type, num_labels):
        config = BertConfig.from_pretrained(pretrained_type)
        config.num_labels = num_labels
        if model_type == ModelType.SequenceClf:
            model = BertForSequenceClassification.from_pretrained(
                pretrained_type, config=config
            )
        else:
            raise NotImplementedError()

        return model


class VanillaBert(nn.Module):
    def __init__(
        self,
        model_type: str = ModelType.SequenceClf,  # BertForSequenceClassification
        pretrained_type: str = PreTrainedType.MultiLingual,  # bert-base-multilingual-cased
        num_labels: int = Config.NumClasses,  # 42
        pooler_idx: int = 0,
        dropout: float = 0.5,
    ):
        super(VanillaBert, self).__init__()
        # BERT로부터 얻은 128(=max_length)개 hidden state 중 몇 번째를 활용할 지 결정. Default - 0(CLS 토큰의 인덱스)
        self.idx = 0 if pooler_idx == 0 else pooler_idx
        self.backbone = self.load_bert(
            model_type=model_type,
            pretrained_type=pretrained_type,
        )
        self.layernorm = nn.LayerNorm(768)  # 768: output length of backbone, BERT
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_features=768, out_features=num_labels)

    def forward(self, input_ids, token_type_ids, attention_mask):
        x = self.backbone(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        # backbone으로부터 얻은 128(토큰 수)개 hidden state 중 어떤 것을 활용할 지 결정. Default - 0(CLS 토큰)
        x = x.last_hidden_state[:, self.idx, :]
        x = self.layernorm(x)
        x = self.dropout(x)
        x = self.relu(x)
        output = self.linear(x)
        return output

    @staticmethod
    def load_bert(model_type, pretrained_type):
        if model_type == ModelType.SequenceClf:
            model = BertForSequenceClassification.from_pretrained(pretrained_type)
            model = model.bert  # 마지막 레이어을 제외한 BERT 아키텍쳐만을 backbone으로 사용
        else:
            raise NotImplementedError()

        return model


# for debugging
if __name__ == "__main__":
    model = load_model(
        model_type=ModelType.SequenceClf,
        pretrained_type=PreTrainedType.MultiLingual,
        num_classes=Config.Num41,
        load_state_dict="./saved_models/BertForSequenceClassification_bert-base-multilingual-cased_20210421122305/checkpoint-1000/",
    )
