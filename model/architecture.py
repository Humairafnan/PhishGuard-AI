import torch
import torch.nn as nn


class ElectraCNNClassifier(nn.Module):
    """
    Final ELECTRA-CNN Hybrid architecture used by the trained checkpoint.

    Architecture:
    ELECTRA [CLS] vector: 768 dimensions
    CNN branch k=3:       128 dimensions after global max pooling
    CNN branch k=5:       128 dimensions after global max pooling
    Fused vector:         1024 dimensions
    Classifier:           1024 -> 256 -> 2
    """

    def __init__(self, electra_model, num_classes=2):
        super().__init__()

        self.electra = electra_model

        hidden_size = electra_model.config.hidden_size

        self.conv1 = nn.Conv1d(
            hidden_size,
            128,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv1d(
            hidden_size,
            128,
            kernel_size=5,
            padding=2
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.3)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None
    ):

        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds
        )

        hidden_states = outputs.last_hidden_state


        # ELECTRA CLS representation
        cls_vector = hidden_states[:, 0, :]


        # CNN feature extraction
        x = hidden_states.permute(0, 2, 1)

        conv3 = self.relu(self.conv1(x))
        conv5 = self.relu(self.conv2(x))


        # Global max pooling
        pool3 = torch.max(conv3, dim=2)[0]
        pool5 = torch.max(conv5, dim=2)[0]


        cnn_features = torch.cat(
            [pool3, pool5],
            dim=1
        )


        # Feature fusion
        combined = torch.cat(
            [cls_vector, cnn_features],
            dim=1
        )

        combined = self.dropout(combined)


        return self.classifier(combined)