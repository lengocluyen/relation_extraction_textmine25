import torch
from torch import nn
from transformers import PreTrainedModel


class BertBasedModel(nn.Module):
    def __init__(
        self,
        name: str,
        embedding_model: PreTrainedModel,
        head_model: nn.Sequential,
        # multilabel: bool = False,
    ):
        super(BertBasedModel, self).__init__()
        self.name = name
        self.embedding_model = embedding_model
        self.head_model = head_model
        # if multilabel:  # for multilabel classification
        #     head_model.add_module(
        #         name="multilabel_output_activation", module=nn.Sigmoid()
        #     )
        # else:  # for no-multilabel classification
        #     head_model.add_module(
        #         name="unilabel_output_activation", module=nn.Softmax()
        #     )

    def forward(
        self,
        input_ids: torch.tensor,
        attn_mask: torch.tensor,
        token_type_ids: torch.tensor,
    ) -> torch.tensor:
        output = self.embedding_model(
            input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids
        )
        output = self.head_model(output.pooler_output)
        return output


class BertMlp(BertBasedModel):
    def __init__(
        self,
        name: str,
        embedding_model: PreTrainedModel,
        embedding_size: int,
        hidden_dim: int,
        n_classes: int,
        dropout_rate: float = 0.1,
        # multilabel: bool = False,
    ):
        head_model = nn.Sequential(
            nn.Linear(embedding_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, n_classes),
        )
        # initialize the weight of the head
        # nn.init.kaiming_uniform_(head_model.weight, a=torch.math.sqrt(5))
        # super(BertMlp, self).__init__(name, embedding_model, head_model, multilabel)
        super(BertMlp, self).__init__(name, embedding_model, head_model)


class BertCnn1d(BertBasedModel):
    def __init__(
        self,
        name: str,
        embedding_model: PreTrainedModel,
        embedding_size: int,
        hidden_dim: int,
        n_classes: int,
        dropout_rate: float = 0.1,
        kernel=3,
        frac=2,
    ):
        head_model = nn.Sequential(
            nn.Conv1d(embedding_size, embedding_size // frac, kernel, padding=2),
            nn.ReLU(),
            nn.Conv1d(
                embedding_size // frac, embedding_size // frac, kernel, padding=2
            ),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Linear(embedding_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, n_classes),
        )
        # initialize the weight of the head
        # nn.init.kaiming_uniform_(head_model.weight, a=torch.math.sqrt(5))
        # super(BertMlp, self).__init__(name, embedding_model, head_model, multilabel)
        super(BertMlp, self).__init__(name, embedding_model, head_model)

    def forward(
        self,
        input_ids: torch.tensor,
        attn_mask: torch.tensor,
        token_type_ids: torch.tensor,
    ) -> torch.tensor:
        output = self.embedding_model(
            input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids
        )
        output = self.head_model(output.pooler_output)
        return output
