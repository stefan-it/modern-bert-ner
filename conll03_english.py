import flair

from flair.datasets.sequence_labeling import ColumnCorpus

from huggingface_hub import hf_hub_download

from pathlib import Path
from typing import Optional, Union


class CONLL_03_ENGLISH(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        column_format = {0: "text", 1: "pos", 2: "np", 3: "ner"}

        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        for dataset_file in ["eng.train", "eng.testa", "eng.testb"]:
            if not (data_folder / dataset_file).exists():
                # Download it from hub - ask @stefan-it for permission
                hf_hub_download(repo_id="stefan-it/conll03-english", repo_type="dataset",
                                filename=dataset_file, token=True, local_dir=data_folder)

        super().__init__(
            data_folder,
            column_format=column_format,
            in_memory=in_memory,
            document_separator_token="-DOCSTART-",
            **corpusargs,
        )