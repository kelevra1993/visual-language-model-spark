# Foundation Model Data Downloader

Instructions for `vision_training/data/classification/foundation_model_image_downloader.py`, which downloads Tencent
ML-Images (ImageNet-22k + Open Images, 11,166 WordNet categories, multi-label) and builds the classification manifests.

## Before Running

1. **ImageNet-22k access**: click "Access repository" on https://huggingface.co/datasets/timm/imagenet-22k-wds with
   your Hugging Face account (approved automatically). Without it the script stops at the start with that link.
2. **ImageNet-1k test images (exclusion of their copies)**: `imagenet-1k/` does not exist yet. Either run
   `vision_training/data/classification/imagenet_download.py` first, or set `excluded_images_folder = None` in the main
   block.
3. **Disk space**: about 1.7 TB for this dataset, plus about 150 GB of ImageNet-1k images (once its parquet files are
   deleted), against 2.2 TB free. The script stops cleanly below `minimum_free_gigabytes` (100 GB) and resumes on the
   next run.

## Running

Set the paths in the main block, then run in tmux (`.venv/bin/python`, since `uv run` breaks when the output is sent to
a log file):

```
.venv/bin/python -m vision_training.data.classification.foundation_model_image_downloader
```

Every step skips the work already done, so it can be run again after any stop.

## Decisions

| Topic | Decision |
|-------|----------|
| ImageNet source | `timm/imagenet-22k-wds` (fall11), 4,608 shards of 0.29 GB, each deleted after use |
| Open Images source | Public S3 bucket through `boto3` (unsigned) |
| Tencent lists | Fetched from Google Drive with `gdown`; if Drive refuses, the error gives the link and the target path |
| Matching | Tencent URL to Open Images identifier by Flickr photo number; unmatched URLs reported |
| Images | Shorter side 512 (never enlarged), JPEG quality 85, Open Images rotation applied |
| Labels | Every label kept (all confidences, ancestors included); confidences stored in `category_confidences` |
| Duplicates | Merged into one sample with combined labels (test copy wins over training copies) |
| Exclusion | Copies of the ImageNet-1k test images removed from every split |
| Splits | Tencent validation images = test; 50,000 training images held out as validation by file name hash |
| Layout | `images/imagenet/<wnid>/`, `images/openimages/<first 2 characters>/`, `manifests/<split>/{data,labels.json}` |
| Class names | First synonym, WordNet identifier appended when two classes share a name |

## Changes Forced By The Data

- **Open Images comes from archives**: only the 1.74M boxable training images exist as single files on S3 (already in
  `open-images-v7`). The full training set is only in 16 `.tar.gz` archives (548 GB), streamed without being saved, at
  about 24 MB/s per archive. Progress is saved every 1,000 images; after an interruption the archive is downloaded again
  but stored images are skipped. The local copy is only used for the validation images.
- **About 1.07M Open Images images are lost**: 15.6% of the Tencent Open Images URLs are no longer in Open Images, so
  about 5.87M Open Images images instead of 6.9M. They are listed in `reports/`.
- **Duplicates use a perceptual hash, not MD5**: the ImageNet-22k copy was re-encoded, so byte hashes could not match
  across sources or with ImageNet-1k. A 128-bit thumbnail hash is used: identical hash for merging duplicates, at most 3
  differing bits for the ImageNet-1k exclusion. Nearly uniform images get no hash and are never merged.

## Dataset Folder

- `annotations/`: Tencent lists, category dictionary, Open Images metadata.
- `images/`: the stored images.
- `progress/`: records of finished shards, batches and archives (resumption).
- `manifests/`: `train`, `validation`, `test` splits and `label_mapping.json`.
- `reports/`: images of the lists left out, `summary.json` with the counts.

## Risk To Check Before Training

The train manifest will hold about 16M samples (about 3 GB of JSON). `ClassificationDataset` loads it into Python
objects, which may use a lot of RAM once the data loader workers start.

## Dependencies

`gdown` and `boto3` were added to `pyproject.toml` and `uv.lock`.
