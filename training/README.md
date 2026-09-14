# Fair Copy — Model Training

Outline of how to use the files in this directory to train a model for Fair Copy.
Technically it's three models currently

## Architecture: three independent binary classifiers, not one joint model

Fair Copy's four labels (Factual Reporting, Opinion/Editorial, Hyperpartisan, Clickbait) are backed by three separately fine-tuned DistilBERT models, not one multi-label model. This is a deliberate choice, not a simplification of convenience: no available dataset has all four labels for the same text, so a single joint model isn't trainable from what exists.

| Model | Labels it predicts | Dataset | Size |
|---|---|---|---|
| Hyperpartisan | Hyperpartisan / Not | SemEval-2019 Task 4 (by-article corpus) | 645 articles |
| Clickbait | Clickbait / Not | Webis Clickbait Corpus 2017 | 19,484 teasers |
| Factual/Opinion | Factual Reporting / Opinion-Editorial | Self-collected via RSS (distant supervision) | 328 articles |

At inference time, `backend/main.py`'s `classify_text()` calls all three models and combines their outputs into the app's four-label response.

## The datasets

**Hyperpartisan** - 645 articles
(Source: https://zenodo.org/records/5776081 - articles-training-byarticle-*.zip)
SemEval's by-article corpus is the *only* publicly released hand-labeled subset.

This data was manually downloaded and added to the project where it was prepared and processed to be used as training for a classifying transformer.

**Clickbait** - 19,484 teasers
(Source:https://zenodo.org/records/5530410 - clickbait17-train-170630.zip)
Webis Clickbait Corpus 2017. Training text is the tweet/teaser (`postText`), not the linked article body, since that's what the corpus's ground truth actually judges.

This data was manually downloaded and added to the project where it was prepared and processed to be used as training for a classifying transformer.

**Factual/Opinion** - 328 articles
(Source: NY Times, Washington Post, The Gaurdian, LA Times) No pre-existing labeled dataset exists for this distinction, so it was built via distant supervision: pulling articles from publishers' own RSS feeds, using which feed (Opinion section vs. straight-news section) as the label. RSS feeds only expose a rolling window of recent entries (not a queryable archive), so this collection is a snapshot, not exhaustive -- periodic re-collection would be
needed to grow this dataset further.

## Pipeline, in order

1. **Collect / download raw data**
   - Hyperpartisan, Clickbait: manual download from Zenodo (see links above), unzipped into `data/hyperpartisan/` and `data/clickbait/`
   - Factual/Opinion: `python collect_factual_opinion.py` -- fetches article text live via Jina + trafilatura (the same extraction pipeline the app
     itself uses), rate-paced under Jina's free-tier limits, checkpointed so an interrupted run can resume. Writes to `data/factual_opinion/raw_collected.jsonl`.
        - **WARNING**: costs Jina tokens

2. **Prepare** — flattens raw source data down to `{"text": ..., "label": ...}`, one file per dataset:

    ```
    python prepare_hyperpartisan.py
    python prepare_clickbait.py
    python prepare_factual_opinion.py
    ```

   **Outputs**: data/hyperpartisan_prepared.jsonl, data/clickbait_prepared.jsonl, data/factual_opinion_prepared.jsonl.

3. **Split** — stratified train/validation split (85/15), preserving each dataset's class ratio in both halves:
    ```
    python split_data.py
    ```

4. **Train** — fine-tunes DistilBERT per dataset, with class weighting to account for label imbalance in all three datasets:
    ```
    python train_classifier.py --dataset hyperpartisan
    python train_classifier.py --dataset clickbait
    python train_classifier.py --dataset factual_opinion
    ```
    Saves each to models/<dataset>/final/.

5. **Push to Hugging Face Hub** — makes the models downloadable by anyone running the app (see backend/README.md — this is how the app loads them, the same mechanism the R&D phase used for the public bart-large-mnli model, just pointed at these instead):
    ```
    python push_to_hub.py
    ```

   Requires a Hugging Face account and a Write-type token to push. Repos: mackyoop/fair-copy-hyperpartisan, mackyoop/fair-copy-clickbait, mackyoop/fair-copy-factual-opinion.

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Hyperpartisan | 0.835 | 0.738 | 0.861 | 0.795 |
| Clickbait | 0.849 | 0.687 | 0.690 | 0.689 |
| Factual/Opinion | 0.920 | 0.889 | 0.960 | 0.923 |

All three outperform the RnD phase's zero-shot baseline (facebook/bart-large-mnli), consistent with published findings that fine-tuned task-specific models outperform zero-shot classification on this kind of task.

## Known limitations, stated plainly

- **Small validation sets.** Hyperpartisan (97) and especially Factual/Opinion (50) are small enough that a handful of flipped predictions would visibly move the reported F1 — these numbers should be read as directionally strong, not precise.
- **512-token truncation.** DistilBERT's context window is 512 tokens; longer articles are truncated at inference (and were truncated during training). The model only ever sees roughly the first ~512 tokens of a long article.
- **Factual/Opinion labels are a proxy, not independent judgment.** A label reflects which RSS section a publisher filed the article under, not an independent assessment of whether it's actually opinion or news writing.
- **Class imbalance** exists in all three datasets (worst in Clickbait, ~24%/76%) — addressed via class weighting during training, not by discarding data.
- **Hyperpartisan's "Not Hyperpartisan" combines what should arguably be two different things** (calm factual reporting and mild non-hyperpartisan opinion) since SemEval only labels the hyperpartisan/not distinction. The separate Factual/Opinion model is what actually distinguishes those two cases at inference time — this is why three models exist instead of two.