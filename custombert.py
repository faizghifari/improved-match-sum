import torch
import torch.nn as nn
from transformers import BertConfig
from transformers.models.bert.modeling_bert import (
    BertModel, 
    BertEmbeddings, 
    BertEncoder, 
    BertPooler,
    # BertPreTrainedModel, 
    # RobertaModel,
    # BertForSequenceClassification, 
)

# >>>>>>>>>> POOLER <<<<<<<<<<<<
class MeanMaxTokensBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        print("MEANMAXPOOLER")
        self.linear = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.act = nn.Tanh()

    def forward(self, hidden_states, *args, **kwargs):
        other_token_tensor = hidden_states[:, 1:]
        mean_token_tensor = torch.mean(other_token_tensor, dim=1)
        max_token_tensor = torch.max(other_token_tensor, dim=1)
        max_token_tensor = max_token_tensor.values
        mmt_tensor = torch.cat((mean_token_tensor, max_token_tensor), dim=1)
        pooled_out = self.linear(mmt_tensor)
        pooled_out = self.act(pooled_out)
        return pooled_out

class LSTMPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        print("LSTMPOOLER")
        self.model_config = config
        self.hidden_size = config.hidden_size
        self.rnn_pool_layer = config.rnn_pool_layer if hasattr(config, "rnn_pool_layer") else 1
        self.rnn_pool_dropout = config.rnn_pool_dropout if hasattr(config, "rnn_pool_dropout") else 0.5
        self.rnn_pool_bidirect = config.rnn_pool_bidirect if hasattr(config, "rnn_pool_bidirect") else False

        self.rnn_layer = nn.LSTM(input_size=self.hidden_size,
                                hidden_size=self.hidden_size,
                                num_layers=self.rnn_pool_layer,
                                dropout=self.rnn_pool_dropout,
                                bidirectional=self.rnn_pool_bidirect,
                                batch_first=True
                                )
        if self.rnn_pool_bidirect:
            self.fc_layer = nn.Linear(in_features=self.hidden_size*2,
                                      out_features=self.hidden_size)
        else:
            self.fc_layer = nn.Linear(in_features=self.hidden_size,
                                      out_features=self.hidden_size)

    def forward(self, hidden_states, *args, **kwargs):
        rnn_output, (hidden, cell) = self.rnn_layer(hidden_states) # LSTM
        if self.rnn_pool_bidirect:
          hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
          hidden = hidden[-1,:,:]
        output = self.fc_layer(hidden)
        output = output.squeeze(1)
        return output
    
class GRUPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        print("GRUPOOLER")
        self.model_config = config
        self.hidden_size = config.hidden_size
        self.rnn_pool_layer = config.rnn_pool_layer if hasattr(config, "rnn_pool_layer") else 1
        self.rnn_pool_dropout = config.rnn_pool_dropout if hasattr(config, "rnn_pool_dropout") else 0.5
        self.rnn_pool_bidirect = config.rnn_pool_bidirect if hasattr(config, "rnn_pool_bidirect") else False

        self.rnn_layer = nn.GRU(input_size=self.hidden_size,
                                hidden_size=self.hidden_size,
                                num_layers=self.rnn_pool_layer,
                                dropout=self.rnn_pool_dropout,
                                bidirectional=self.rnn_pool_bidirect,
                                batch_first=True
                                )
        if self.rnn_pool_bidirect:
            self.fc_layer = nn.Linear(in_features=self.hidden_size*2,
                                      out_features=self.hidden_size)
        else:
            self.fc_layer = nn.Linear(in_features=self.hidden_size,
                                      out_features=self.hidden_size)

    def forward(self, hidden_states, *args, **kwargs):
        rnn_output, hidden = self.rnn_layer(hidden_states) # GRU, RNN
        if self.rnn_pool_bidirect:
          hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
          hidden = hidden[-1,:,:]
        output = self.fc_layer(hidden)
        output = output.squeeze(1)
        return output


# >>>>>>>>>> BERT <<<<<<<<<<<<
class CustomBertConfig(BertConfig):
    def __init__(self, pooling_layer_type="CLS", **kwargs):
        super().__init__(**kwargs)
        self.pooling_layer_type = pooling_layer_type

class CustomBertModel(BertModel):

    def __init__(self, config: CustomBertConfig):
        super(BertModel, self).__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        if config.pooling_layer_type == "CLS":
            self.pooler = BertPooler(config) # See src/transformers/models/bert/modeling_bert.py#L869
        elif config.pooling_layer_type == "MEAN_MAX":
            self.pooler = MeanMaxTokensBertPooler(config)
        elif config.pooling_layer_type == "LSTM":
            self.pooler = LSTMPooler(config)
        elif config.pooling_layer_type == "GRU":
            self.pooler = GRUPooler(config)
        else:
            raise ValueError(f"Wrong pooling_layer_type: {config.pooling_layer_type}")

        self.init_weights()

    @property
    def pooling_layer_type(self):
        return self.config.pooling_layer_type
