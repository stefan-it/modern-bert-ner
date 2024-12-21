import logging

from dataclasses import dataclass

from flair import set_seed

from flair.data import MultiCorpus
from flair.embeddings import TransformerWordEmbeddings, TransformerDocumentEmbeddings
from flair.models import SequenceTagger, TextClassifier
from flair.trainers import ModelTrainer
from flair.trainers.plugins.loggers.tensorboard import TensorboardLogger

from pathlib import Path

from typing import List

from conll03_english import CONLL_03_ENGLISH

logger = logging.getLogger("flair")
logger.setLevel(level="INFO")


@dataclass
class ExperimentConfiguration:
    batch_size: int
    learning_rate: float
    epoch: int
    context_size: int
    seed: int
    base_model: str
    base_model_short: str
    task: str
    datasets: List[str]
    layers: str = "-1"
    subtoken_pooling: str = "first"
    use_crf: bool = False
    use_tensorboard: bool = True


def run_experiment_token_classification(experiment_configuration: ExperimentConfiguration) -> str:
    set_seed(experiment_configuration.seed)

    corpora = []

    label_type = experiment_configuration.task

    if experiment_configuration.task == "ner":
        label_type = "ner"

        for dataset in experiment_configuration.datasets:
            # E.g. xtreme/tr
            if dataset == "conll03_english":
                corpora.append(CONLL_03_ENGLISH())

    corpora: MultiCorpus = MultiCorpus(corpora=corpora, sample_missing_splits=False)

    label_dictionary = corpora.make_label_dictionary(label_type=label_type)
    logger.info("Label Dictionary: {}".format(label_dictionary.get_items()))
    logger.info("Layers: {}".format(experiment_configuration.layers))

    embeddings = TransformerWordEmbeddings(
        model=experiment_configuration.base_model,
        layers=experiment_configuration.layers,
        subtoken_pooling=experiment_configuration.subtoken_pooling,
        fine_tune=True,
        use_context=experiment_configuration.context_size,
    )

    tagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=label_dictionary,
        tag_type=label_type,
        use_crf=experiment_configuration.use_crf,
        use_rnn=False,
        reproject_embeddings=False,
    )

    trainer = ModelTrainer(tagger, corpora)

    output_path_parts = [
        "flair",
        experiment_configuration.task,
        "_".join(experiment_configuration.datasets),
        experiment_configuration.base_model_short,
        f"bs{experiment_configuration.batch_size}",
        f"e{experiment_configuration.epoch}",
        f"cs{experiment_configuration.context_size}",
        f"lr{experiment_configuration.learning_rate}",
        str(experiment_configuration.seed)
    ]

    output_path = "-".join(output_path_parts)

    plugins = []

    if experiment_configuration.use_tensorboard:
        logger.info("TensorBoard logging is enabled")

        tb_path = Path(f"{output_path}/runs")
        tb_path.mkdir(parents=True, exist_ok=True)

        plugins.append(TensorboardLogger(log_dir=str(tb_path), comment=output_path))

    trainer.fine_tune(
        output_path,
        learning_rate=experiment_configuration.learning_rate,
        mini_batch_size=experiment_configuration.batch_size,
        max_epochs=experiment_configuration.epoch,
        shuffle=True,
        embeddings_storage_mode='none',
        weight_decay=0.,
        use_final_model_for_eval=False,
        plugins=plugins,
    )

    # Finally, print model card for information
    tagger.print_model_card()

    return output_path

def run_experiment(experiment_configuration: ExperimentConfiguration) -> str:
    if experiment_configuration.task in ["ner"]:
        return run_experiment_token_classification(experiment_configuration)
