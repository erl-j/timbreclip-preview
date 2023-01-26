import torch
import numpy as np


# inspired by https://github.com/descriptinc/lyrebird-wav2clip
class CustomClipLoss(torch.nn.Module):
    def __init__(self):
        super(CustomClipLoss, self).__init__()
        # self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_audio = torch.nn.CrossEntropyLoss()
        self.loss_text = torch.nn.CrossEntropyLoss()

    def forward(self, audio_features, text_features, matches):
        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_audio = logit_scale * audio_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ audio_features.t()

        batch_size = audio_features.shape[0]
        # ground_truth = torch.arange(batch_size, dtype=torch.long, device=audio_features.device)
        # print(ground_truth.shape)

        return (
            (
                self.loss_audio(logits_per_audio, matches)
                * self.loss_text(logits_per_text.T, matches)
            )
            / 2
        ) / (audio_features.shape[-1])


def get_first_match_rank(documents, queries, matches):
    documents = documents / documents.norm(dim=-1, keepdim=True)
    queries = queries / queries.norm(dim=-1, keepdim=True)
    cos = documents @ queries.t()

    # sort columns of matches by columns of sort
    ranks = torch.zeros(cos.shape[-1])
    for q in range(cos.shape[-1]):
        r = torch.argsort(cos[:, q], descending=True)
        sm = matches[:, q][r]
        ranks[q] = np.nonzero(sm)[0]

    return torch.mean(ranks) / cos.shape[0]
