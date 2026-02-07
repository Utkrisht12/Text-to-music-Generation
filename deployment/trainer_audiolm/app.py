import argparse
import os
import traceback
import urllib.request
from glob import glob

import laion_clap
import torch
import torchaudio
import wandb
from audiolm_pytorch import (
    AudioLM,
    CoarseTransformer,
    CoarseTransformerTrainer,
    EncodecWrapper,
    FineTransformer,
    FineTransformerTrainer,
    HubertWithKmeans,
    SemanticTransformer,
    SemanticTransformerTrainer,
)
from audiolm_pytorch.utils import AudioConditionerBase
from beartype import beartype
from beartype.typing import Tuple
from dotenv import load_dotenv
from einops import rearrange, repeat
from huggingface_hub import hf_hub_download
from torch import nn
from vector_quantize_pytorch import ResidualVQ

load_dotenv()
wandb.login(key=os.environ["WANDB_KEY"])

RESULTS_ROOT_PATH = "/mnt/data/results"

dataset_folder = "/mnt/data/data/"
hubert_ckpt = "hubert/hubert_base_ls960.pt"
hubert_quantizer = f"hubert/hubert_base_ls960_L9_km500.bin"  # listed in row "HuBERT Base (~95M params)", column Quantizer
using_encodec = True  # TODO


CLAP_REPO_ID = "lukewys/laion_clap"
CLAP_VERSION = "music_audioset_epoch_15_esc_90.14.pt"
clap_ckp_path = hf_hub_download(repo_id=CLAP_REPO_ID, filename=CLAP_VERSION)

# hubert checkpoints can be downloaded at
# https://github.com/facebookresearch/fairseq/tree/main/examples/hubert
if not os.path.isdir("hubert"):
    os.makedirs("hubert")
if not os.path.isfile(hubert_ckpt):
    hubert_ckpt_download = f"https://dl.fbaipublicfiles.com/{hubert_ckpt}"
    urllib.request.urlretrieve(hubert_ckpt_download, f"./{hubert_ckpt}")
if not os.path.isfile(hubert_quantizer):
    hubert_quantizer_download = (
        f"https://dl.fbaipublicfiles.com/{hubert_quantizer}"
    )
    urllib.request.urlretrieve(
        hubert_quantizer_download, f"./{hubert_quantizer}"
    )


def custom_train(trainer, log_fn):
    while trainer.steps < trainer.num_train_steps:
        torch.cuda.empty_cache()
        try:
            logs = trainer.train_step()
            log_fn(logs)
        except RuntimeError as e:
            print(traceback.format_exc())
            print(e)

    model_path = str(trainer.results_folder / f"transformer.last.pt")
    trainer.save(model_path)
    trainer.print("training complete")


# Based on MuLaNEmbedQuantizer implementation on https://github.com/lucidrains/musiclm-pytorch
def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class CLAPEmbedQuantizer(AudioConditionerBase):
    @beartype
    def __init__(
        self,
        clap: laion_clap.CLAP_Module,
        conditioning_dims: Tuple[int, ...],
        rq_num_quantizers=8,
        rq_ema_decay=0.9,
        codebook_size=1024,
        namespaces: Tuple[str, ...] = ("semantic", "coarse", "fine"),
    ):
        super().__init__()
        self.clap = clap

        assert len(namespaces) > 0
        self.namespaces = namespaces
        self.conditioning_dims = conditioning_dims

        assert len(conditioning_dims) == len(
            namespaces
        ), "number of conditioning dimensions must be equal to number of namespaces"

        dim = clap.model.audio_projection[-1].out_features

        self.rq = ResidualVQ(
            dim=dim,
            num_quantizers=rq_num_quantizers,
            codebook_size=codebook_size,
            decay=rq_ema_decay,
            commitment_weight=0,  # only use EMA to update codebooks
            kmeans_init=True,
            threshold_ema_dead_code=2,
            quantize_dropout=False,  # no quantize dropout
        )

        self.dim = dim
        self.num_codebooks = rq_num_quantizers

        self.cond_embeddings = nn.ParameterDict({})

        for namespace, conditioning_dim in zip(namespaces, conditioning_dims):
            cond_embeddings = nn.Parameter(
                torch.randn(rq_num_quantizers, codebook_size, conditioning_dim)
            )
            nn.init.normal_(cond_embeddings, std=0.02)

            self.cond_embeddings[namespace] = cond_embeddings

        self.set_default_namespace(namespaces[0])

    def parameters(self):
        return self.cond_embeddings.parameters()

    def set_default_namespace(self, namespace):
        self._default_namespace = namespace

    def forward(self, wavs=None, texts=None, namespace=None):
        assert exists(wavs) ^ exists(texts)

        namespace = default(namespace, self._default_namespace)
        assert namespace in self.namespaces, f"namespace {namespace} not found"
        cond_embeddings = self.cond_embeddings[namespace]

        with torch.no_grad():
            self.clap.eval()

            # sound and language live in joint embedding space because of contrastive learning

            if exists(wavs):
                latents = self.clap.get_audio_embedding_from_data(
                    x=wavs, use_tensor=True
                )
            elif exists(texts):
                latents = self.clap.get_text_embedding(texts, use_tensor=True)
            else:
                raise RuntimeError()

        _, indices, _ = self.rq(latents)

        batch, num_codebooks, dim = (
            indices.shape[0],
            self.num_codebooks,
            cond_embeddings.shape[-1],
        )

        cond_embeddings = repeat(cond_embeddings, "q c d -> b q c d", b=batch)
        indices = repeat(indices, "b q -> b q 1 d", q=num_codebooks, d=dim)

        cond_embeddings = cond_embeddings.gather(2, indices)
        return rearrange(cond_embeddings, "b q 1 d -> b q d")


DATA_SIZE = 96113
EPOCHS = 15


def get_trainer(
    trainer_type,
    soundstream,
    wav2vec,
    quantizer,
    batch_size=1,
    data_max_lenght_sec=10,
    **kwargs,
):
    torch.cuda.empty_cache()
    print(f"Getting trainer {trainer_type}...")
    total_steps = DATA_SIZE * EPOCHS + 1
    save_every = 300
    if trainer_type == "semantic":
        semantic_transformer = SemanticTransformer(
            num_semantic_tokens=wav2vec.codebook_size,
            dim=1024,
            depth=6,
            audio_text_condition=True,
        ).cuda()

        semantic_trainer = SemanticTransformerTrainer(
            transformer=semantic_transformer,
            wav2vec=wav2vec,
            audio_conditioner=quantizer,
            folder=dataset_folder,
            batch_size=batch_size,
            data_max_length_seconds=data_max_lenght_sec,
            save_results_every=30,
            save_model_every=save_every,
            num_train_steps=total_steps,
            results_folder=RESULTS_ROOT_PATH + "/results_semantic",
            force_clear_prev_results=False,
        )
        return semantic_trainer
    elif trainer_type == "coarse":
        coarse_transformer = CoarseTransformer(
            num_semantic_tokens=wav2vec.codebook_size,
            codebook_size=1024,
            num_coarse_quantizers=3,
            dim=1024,
            depth=6,
            audio_text_condition=True,
        )

        coarse_trainer = CoarseTransformerTrainer(
            transformer=coarse_transformer,
            codec=soundstream,
            wav2vec=wav2vec,
            audio_conditioner=quantizer,
            folder=dataset_folder,
            batch_size=batch_size,
            data_max_length_seconds=data_max_lenght_sec,
            save_results_every=save_every,
            save_model_every=save_every,
            num_train_steps=total_steps,
            results_folder=RESULTS_ROOT_PATH + "/results_coarse",
            force_clear_prev_results=False,
        )
        return coarse_trainer
    elif trainer_type == "fine":
        fine_transformer = FineTransformer(
            num_coarse_quantizers=3,
            num_fine_quantizers=5,
            codebook_size=1024,
            dim=1024,
            depth=6,
            audio_text_condition=True,
        )

        fine_trainer = FineTransformerTrainer(
            transformer=fine_transformer,
            codec=soundstream,
            audio_conditioner=quantizer,
            folder=dataset_folder,
            batch_size=batch_size,
            data_max_length_seconds=data_max_lenght_sec,
            save_results_every=save_every,
            save_model_every=save_every,
            num_train_steps=total_steps,
            results_folder=RESULTS_ROOT_PATH + "/results_fine",
            force_clear_prev_results=False,
        )
        return fine_trainer
    else:
        raise RuntimeError("Invalid type")


def get_auxiliary_model():
    clap_model = laion_clap.CLAP_Module(
        enable_fusion=False, amodel="HTSAT-base"
    )
    clap_model.load_ckpt(clap_ckp_path)
    soundstream = EncodecWrapper()
    wav2vec = HubertWithKmeans(
        checkpoint_path=f"./{hubert_ckpt}", kmeans_path=f"./{hubert_quantizer}"
    )
    quantizer = CLAPEmbedQuantizer(
        clap=clap_model,
        conditioning_dims=(
            1024,
            1024,
            1024,
        ),  # say all three transformers have model dimensions of 1024
        namespaces=("semantic", "coarse", "fine"),
    ).cuda()
    return soundstream, wav2vec, quantizer


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="Train program for transformers of AudioLM"
    )
    parser.add_argument(
        "model_to_train", choices=["fine", "coarse", "semantic", "inference"]
    )
    parser.add_argument("-l", "--load_ckpt", default=False, type=bool)
    parser.add_argument("-n", "--run_number", default=0)
    parser.add_argument("-r", "--resume", default=False, type=bool)
    parser.add_argument("-bs", "--batch_size", default=1, type=int)
    parser.add_argument("-len", "--max_length", default=2048, type=int)
    parser.add_argument("-dls", "--data_max_lenght_sec", default=10, type=int)
    parser.add_argument("-s", "--semantic_load_ckp", default=None)
    parser.add_argument("-c", "--coarse_load_ckp", default=None)
    parser.add_argument("-f", "--fine_load_ckp", default=None)
    args = parser.parse_args()
    checkpoints = dict(
        semantic=args.semantic_load_ckp,
        coarse=args.coarse_load_ckp,
        fine=args.fine_load_ckp,
    )
    config = dict(
        batch_size=args.batch_size,
        data_max_lenght_sec=args.data_max_lenght_sec,
    )
    return (
        args.model_to_train,
        args.load_ckpt,
        args.run_number,
        args.resume,
        config,
        checkpoints,
    )


def get_last_checkpoint(model_to_train):
    dirs = glob(f"{RESULTS_ROOT_PATH}/results_{model_to_train}/*")
    if len(dirs) == 0:
        return None
    return sorted(dirs, key=lambda x: int(x.split(".")[-2]))[-1]


def main():
    (
        model_to_train,
        load_ckpt,
        run_number,
        resume,
        config,
        checkpoints,
    ) = parse_arguments()

    inference = model_to_train == "inference"

    output_path = RESULTS_ROOT_PATH + f"/generated/out{run_number}.wav"
    sample_rate = 44100

    auxiliar_models = get_auxiliary_model()
    soundstream, wav2vec, quantizer = auxiliar_models
    # Training
    if not inference:
        run = wandb.init(
            name=f"{model_to_train}_{run_number}",
            resume=resume,
            project="project",
            entity="cmu-idl",
            config=config,
        )
        print(run.id)

        trainer = get_trainer(
            model_to_train,
            *auxiliar_models,
            batch_size=config["batch_size"],
            data_max_lenght_sec=config["data_max_lenght_sec"],
        )
        if load_ckpt:
            path = (
                checkpoints[model_to_train]
                if checkpoints[model_to_train] is not None
                else get_last_checkpoint(model_to_train)
            )
            if path is not None:
                print(f"Checkpoint to be loaded {path} ............")
                trainer.load(path)
                print(f"Checkpoint in {path} succesfully loaded!")
        custom_train(trainer, wandb.log)

    if inference:
        # Everything together
        semantic_trainer = get_trainer("semantic", *auxiliar_models)
        semantic_trainer.load(checkpoints["semantic"])
        coarse_trainer = get_trainer("coarse", *auxiliar_models)
        coarse_trainer.load(checkpoints["coarse"])
        fine_trainer = get_trainer("fine", *auxiliar_models)
        fine_trainer.load(checkpoints["fine"])
        audiolm = AudioLM(
            wav2vec=wav2vec,
            codec=soundstream,
            semantic_transformer=semantic_trainer.transformer,
            coarse_transformer=coarse_trainer.transformer,
            fine_transformer=fine_trainer.transformer,
        )

        text_embeddings = quantizer(
            texts=["Experimental high quality metal song with flute", ""]
        )[0:1]

        generated_wav = audiolm(
            text_embeds=text_embeddings, batch_size=1, max_length=2048
        )

        torchaudio.save(output_path, generated_wav.cpu(), sample_rate)


torch.cuda.empty_cache()
if __name__ == "__main__":
    main()
