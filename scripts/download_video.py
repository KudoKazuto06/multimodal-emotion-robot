from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="declare-lab/MELD",
    repo_type="dataset",
    filename="MELD.Raw.tar.gz",
    local_dir="../data/raw"
)