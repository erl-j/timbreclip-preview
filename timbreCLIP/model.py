import numpy as np
import pytorch_lightning as pl
import torch
import wav2clip

from .lib import CustomClipLoss, get_first_match_rank


class TimbreCLIP(pl.LightningModule):
    def __init__(
        self, embedding_size, start_from_wav2clip=False, use_wav2clip_architecture=False
    ):
        super().__init__()
        model = wav2clip.get_model()

        if use_wav2clip_architecture:
            print(model.transform.sequential[3])
            model.transform.sequential[3] = torch.nn.Linear(
                in_features=512, out_features=embedding_size, bias=True
            )

        else:
            model.transform = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(
                    in_features=512, out_features=embedding_size, bias=True
                ),
            )

        def weights_init(m):
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)

        if not start_from_wav2clip:
            model.apply(weights_init)

        self.projection_layer = model
        self.loss_fn = CustomClipLoss()

    def forward(self, waveform):
        return self.projection_layer(waveform)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return optimizer

    def training_step(self, batch, batch_idx):
        waveform = batch["waveform"]
        text_embeddings = batch["text_embeddings"]
        output = self.forward(waveform)
        loss = self.loss_fn(output, text_embeddings, batch["audiomatchestext"])
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch["waveform"].shape[0],
        )
        return {
            "loss": loss,
            "log": {"validation_loss": loss.detach().cpu()},
            "audio_embedding": output.detach().cpu(),
            "text_embedding": text_embeddings.detach().cpu(),
            "audiomatchestext": batch["audiomatchestext"].cpu(),
        }

    def validation_step(self, batch, batch_idx):
        waveform = batch["waveform"]
        text_embeddings = batch["text_embeddings"]
        output = self.forward(waveform)
        loss = self.loss_fn(output, text_embeddings, batch["audiomatchestext"])
        self.log(
            "validation_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch["waveform"].shape[0],
        )
        return {
            "loss": loss.detach().cpu(),
            "log": {"validation_loss": loss.detach().cpu()},
            "audio_embedding": output.detach().cpu(),
            "text_embedding": text_embeddings.detach().cpu(),
            "audiomatchestext": batch["audiomatchestext"].cpu(),
        }

    def validation_epoch_end(self, outputs):
        audio_embeddings = []
        text_embeddings = []

        rank = []

        for output in outputs:
            audio_embeddings = output["audio_embedding"]
            text_embeddings = output["text_embedding"]
            matches = output["audiomatchestext"]
            rank.append(
                get_first_match_rank(audio_embeddings, text_embeddings, matches)
            )

        mean_rank = np.mean(rank)

        # avg_loss = torch.stack([x["validation_loss"] for x in outputs]).mean()
        self.log("val_mean_rank", mean_rank, prog_bar=True)
        # return {
        #     "mean_rank": mean_rank,
        # }

    def training_epoch_end(self, outputs):
        audio_embeddings = []
        text_embeddings = []

        rank = []

        for output in outputs:
            audio_embeddings = output["audio_embedding"]
            text_embeddings = output["text_embedding"]
            matches = output["audiomatchestext"]
            rank.append(
                get_first_match_rank(audio_embeddings, text_embeddings, matches)
            )

        mean_rank = np.mean(rank)

        # avg_loss = torch.stack([x["validation_loss"] for x in outputs]).mean()
        self.log("mean_rank", mean_rank, prog_bar=True)
        # return {
        #     "mean_rank": mean_rank,
        # }
