# ConvDR

This repo contains code and data for SIGIR 2021 paper ["Few-Shot Conversational Dense Retrieval"](https://arxiv.org/pdf/2105.04166.pdf).

## Prerequisites

Install dependencies:

```bash
git clone https://github.com/thunlp/ConvDR.git
cd ConvDR
pip install -r requirements.txt
```

We recommend set `PYTHONPATH` before running the code:

```bash
export PYTHONPATH=${PYTHONPATH}:`pwd`
```

To train ConvDR, we need trained ad hoc dense retrievers. We use [ANCE](https://github.com/microsoft/ANCE) for both tasks. Please downloads those checkpoints here: [TREC CAsT](https://webdatamltrainingdiag842.blob.core.windows.net/semistructstore/OpenSource/Passage_ANCE_FirstP_Checkpoint.zip) and [OR-QuAC](https://data.thunlp.org/convdr/ad-hoc-ance-orquac.cp). For TREC CAsT, we directly use the official model trained on MS MARCO Passage Retrieval task. For OR-QuAC, we initialize the retriever from the official model trained on NQ and TriviaQA, and continue training on OR-QuAC with manually reformulated questions using the ANCE codebase.

The following code downloads those checkpoints and store them in `./checkpoints`.

```bash
mkdir checkpoints
wget https://webdatamltrainingdiag842.blob.core.windows.net/semistructstore/OpenSource/Passage_ANCE_FirstP_Checkpoint.zip
wget https://data.thunlp.org/convdr/ad-hoc-ance-orquac.cp
unzip Passage_ANCE_FirstP_Checkpoint.zip
mv "Passage ANCE(FirstP) Checkpoint" ad-hoc-ance-msmarco
```

## Data Preparation

By default, we expect raw data to be stored in `./datasets/raw` and processed data to be stored in `./datasets`:

```bash
mkdir datasets
mkdir datasets/raw
```

### TREC CAsT

#### CAsT shared files download

Use the following commands to download the document collection for CAsT-19 & CAsT-20 as well as the MARCO duplicate file:

```bash
cd datasets/raw
wget https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz -O msmarco.tsv
wget http://trec-car.cs.unh.edu/datareleases/v2.0/paragraphCorpus.v2.0.tar.xz
wget http://boston.lti.cs.cmu.edu/Services/treccast19/duplicate_list_v1.0.txt
```

#### CAsT-19 files download

Download necessary files for CAsT-19 and store them into `./datasets/raw/cast-19`:

```bash
mkdir datasets/raw/cast-19
cd datasets/raw/cast-19
wget https://raw.githubusercontent.com/daltonj/treccastweb/master/2019/data/evaluation/evaluation_topics_v1.0.json
wget https://raw.githubusercontent.com/daltonj/treccastweb/master/2019/data/evaluation/evaluation_topics_annotated_resolved_v1.0.tsv
wget https://trec.nist.gov/data/cast/2019qrels.txt
```

#### CAsT-20 files download

Download necessary files for CAsT-20 and store them into `./datasets/raw/cast-20`:

```bash
mkdir datasets/raw/cast-20
cd datasets/raw/cast-20
wget https://raw.githubusercontent.com/daltonj/treccastweb/master/2020/2020_automatic_evaluation_topics_v1.0.json
wget https://raw.githubusercontent.com/daltonj/treccastweb/master/2020/2020_manual_evaluation_topics_v1.0.json
wget https://trec.nist.gov/data/cast/2020qrels.txt
```

#### CAsT preprocessing

Use the scripts `./data/preprocess_cast19` and `./data/preprocess_cast20` to preprocess raw CAsT files:

```bash
mkdir datasets/cast-19
mkdir datasets/cast-shared
python data/preprocess_cast19.py  --car_cbor=datasets/raw/dedup.articles-paragraphs.cbor  --msmarco_collection=datasets/raw/msmarco.tsv  --duplicate_file=datasets/raw/duplicate_list_v1.0.txt  --cast_dir=datasets/raw/cast-19/  --out_data_dir=datasets/cast-19  --out_collection_dir=datasets/cast-shared
```

```bash
mkdir datasets/cast-20
mkdir datasets/cast-shared
python data/preprocess_cast20.py  --car_cbor=datasets/raw/dedup.articles-paragraphs.cbor  --msmarco_collection=datasets/raw/msmarco.tsv  --duplicate_file=datasets/raw/duplicate_list_v1.0.txt  --cast_dir=datasets/raw/cast-20/  --out_data_dir=datasets/cast-20  --out_collection_dir=datasets/cast-shared
```

### OR-QuAC

#### OR-QuAC files download

Download necessary OR-QuAC files and store them into `./datasets/raw/or-quac`:

```bash
mkdir datasets/raw/or-quac
cd datasets/raw/or-quac
wget https://ciir.cs.umass.edu/downloads/ORConvQA/all_blocks.txt.gz
wget https://ciir.cs.umass.edu/downloads/ORConvQA/qrels.txt.gz
gzip -d *.txt.gz
mkdir preprocessed
cd preprocessed
wget https://ciir.cs.umass.edu/downloads/ORConvQA/preprocessed/train.txt
wget https://ciir.cs.umass.edu/downloads/ORConvQA/preprocessed/test.txt
wget https://ciir.cs.umass.edu/downloads/ORConvQA/preprocessed/dev.txt
```

#### OR-QuAC preprocessing

Use the scripts `./data/preprocess_orquac` to preprocess OR-QuAC files:

```bash
mkdir datasets/or-quac
python data/preprocess_orquac.py  --orquac_dir=datasets/raw/or-quac  --output_dir=datasets/or-quac
```

### Generate Document Embeddings

Our code is based on ANCE and we have a similar embedding inference pipeline, where the documents are first tokenized and converted to token ids and then the token ids are used for embedding inference. We create sub-directories `tokenized` and `embeddings` inside `./datasets/cast-shared` and `./datasets/or-quac` to store the tokenized documents and document embeddings, respectively:

```bash
mkdir datasets/cast-shared/tokenized
mkdir datasets/cast-shared/embeddings
mkdir datasets/or-quac/tokenized
mkdir datasets/or-quac/embeddings
```

Run `./data/tokenizing.py` to tokenize documents in parallel:

```bash
# CAsT
python data/tokenizing.py  --collection=datasets/cast-shared/collection.tsv  --out_data_dir=datasets/cast-shared/tokenized  --model_name_or_path=checkpoints/ad-hoc-ance-msmarco --model_type=rdot_nll
# OR-QuAC
python data/tokenizing.py  --collection=datasets/or-quac/collection.tsv  --out_data_dir=datasets/or-quac/tokenized  --model_name_or_path=bert-base-uncased --model_type=dpr
```

After tokenization, run `./drivers/gen_passage_embeddings.py` to generate document embeddings:

```bash
# CAsT
python -m torch.distributed.launch --nproc_per_node=$gpu_no python drivers/gen_passage_embeddings.py  --data_dir=datasets/cast-shared/tokenized  --checkpoint=checkpoints/ad-hoc-ance-msmarco  --output_dir=datasets/cast-shared/embeddings  --model_type=rdot_nll
# OR-QuAC
python -m torch.distributed.launch --nproc_per_node=$gpu_no python drivers/gen_passage_embeddings.py  --data_dir=datasets/or-quac/tokenized  --checkpoint=checkpoints/ad-hoc-ance-orquac.cp  --output_dir=datasets/or-quac/embeddings  --model_type=dpr
```

Note that we follow the ANCE implementation and this step takes up a lot of memory. To generate all 38M CAsT document embeddings safely, the machine should have at least 200GB memory. It's possible to save memory by generating a part at a time, and we may update the implementation in the future.

## ConvDR Training

Now we are all prepared: we have downloaded & preprocessed data, and we have obtained document embeddings. Simply run `./drivers/run_convdr_train.py` to train a ConvDR using KD (MSE) loss:

```bash
# CAsT-19, KD loss only, five-fold cross-validation
python drivers/run_convdr_train.py  --output_dir=checkpoints/convdr-kd-cast19  --model_name_or_path=checkpoints/ad-hoc-ance-msmarco  --train_file=datasets/cast-19/eval_topics.jsonl  --query=no_res  --per_gpu_train_batch_size=4  --learning_rate=1e-5   --log_dir=logs/convdr_kd_cast19  --num_train_epochs=8  --model_type=rdot_nll  --cross_validate
# CAsT-20, KD loss only, five-fold cross-validation, use automatic canonical responses, set a longer length
python drivers/run_convdr_train.py  --output_dir=checkpoints/convdr-kd-cast20  --model_name_or_path=checkpoints/ad-hoc-ance-msmarco  --train_file=datasets/cast-20/eval_topics.jsonl  --query=auto_can  --per_gpu_train_batch_size=4  --learning_rate=1e-5   --log_dir=logs/convdr_kd_cast20  --num_train_epochs=8  --model_type=rdot_nll  --cross_validate  --max_concat_length=512
# OR-QuAC, KD loss only
python drivers/run_convdr_train.py  --output_dir=checkpoints/convdr-kd-orquac.cp  --model_name_or_path=checkpoints/ad-hoc-ance-orquac.cp  --train_file=datasets/or-quac/train.jsonl  --query=no_res  --per_gpu_train_batch_size=4  --learning_rate=1e-5  --log_dir=logs/convdr_kd_orquac  --num_train_epochs=1  --model_type=dpr  --log_steps=100
```

Note that for CAsT-20, it's better to first pretrain the model on CANARD and then do cross-validation:

```bash
# Pretrain on CANARD (use preprocessed OR-QuAC)
python drivers/run_convdr_train.py  --output_dir=checkpoints/convdr-kd-cast20-warmup  --model_name_or_path=checkpoints/ad-hoc-ance-msmarco  --train_file=datasets/or-quac/train.jsonl  --query=man_can  --per_gpu_train_batch_size=4  --learning_rate=1e-5   --log_dir=logs/convdr_kd_cast20_warmup  --num_train_epochs=1  --model_type=rdot_nll  --log_steps=100  --max_concat_length=512
# Do cross-validation on CAsT-20; Set model_name_or_path to the pretrained model and specify teacher_model to the ad hoc model
python drivers/run_convdr_train.py  --output_dir=checkpoints/convdr-kd-cast20  --model_name_or_path=checkpoints/convdr-kd-cast20-warmup  --teacher_model=checkpoints/ad-hoc-ance-msmarco  --train_file=datasets/cast-20/eval_topics.jsonl  --query=auto_can  --per_gpu_train_batch_size=4  --learning_rate=1e-5   --log_dir=logs/convdr_kd_cast20  --num_train_epochs=8  --model_type=rdot_nll  --cross_validate  --max_concat_length=512
```

To use ranking loss, we need to find negative documents for each query. We use top retrieved negatives documents from the ranking results of **manual** queries. So we need to first perform retrieval using the manual queries:

```bash
# CAsT-19
python drivers/run_convdr_inference.py  --model_path=checkpoints/ad-hoc-ance-msmarco  --eval_file=datasets/cast-19/eval_topics.jsonl  --query=target  --per_gpu_eval_batch_size=8  --ann_data_dir=datasets/cast-19/embeddings  --qrels=datasets/cast-19/qrels.tsv  --processed_data_dir=datasets/cast-19/tokenized  --raw_data_dir=datasets/cast-19   --output_file=results/cast-19/manual_ance.jsonl  --output_trec_file=results/cast-19/manual_ance.trec  --model_type=rdot_nll  --output_query_type=manual  --use_gpu
# OR-QuAC, inference on train, set query to "target" to use manual queries directly
python drivers/run_convdr_inference.py  --model_path=checkpoints/ad-hoc-ance-orquac.cp  --eval_file=datasets/or-quac/train.jsonl  --query=target  --per_gpu_eval_batch_size=8  --ann_data_dir=datasets/or-quac/embeddings  --qrels=datasets/or-quac/qrels.tsv  --processed_data_dir=datasets/or-quac/tokenized  --raw_data_dir=datasets/or-quac   --output_file=results/or-quac/manual_ance_train.jsonl  --output_trec_file=results/or-quac/manual_ance_train.trec  --model_type=dpr  --output_query_type=train.manual  --use_gpu
```

After the retrieval finishes, we can select negative documents from manual runs and supplement the original training files with them:

```bash
# CAsT-19
python data/gen_ranking_data.py  --train=datasets/cast-19/eval_topics.jsonl  --run=results/cast-19/manual_ance.trec  --output=datasets/cast-19/eval_topics.rank.jsonl  --qrels=datasets/cast-19/qrels.tsv  --collection=datasets/cast-shared/collection.tsv  --cast
# OR-QuAC
python data/gen_ranking_data.py  --train=datasets/or-quac/train.jsonl  --run=results/or-quac/manual_ance_train.trec  --output=datasets/or-quac/train.rank.jsonl  --qrels=datasets/or-quac/qrels.tsv  --collection=datasets/or-quac/collection.jsonl
```

Now we are able to use the ranking loss, with the `--ranking_task` flag on:

```bash
# CAsT-19, Multi-task
python drivers/run_convdr_train.py  --output_dir=checkpoints/convdr-multi-cast19  --model_name_or_path=checkpoints/ad-hoc-ance-msmarco  --train_file=datasets/cast-19/eval_topics.rank.jsonl  --query=no_res  --per_gpu_train_batch_size=4  --learning_rate=1e-5   --log_dir=logs/convdr_multi_cast19  --num_train_epochs=8  --model_type=rdot_nll  --cross_validate  --ranking_task
# OR-QuAC, Multi-task
python drivers/run_convdr_train.py  --output_dir=checkpoints/convdr-multi-orquac.cp  --model_name_or_path=checkpoints/ad-hoc-ance-orquac.cp  --train_file=datasets/or-quac/train.rank.jsonl  --query=no_res  --per_gpu_train_batch_size=4  --learning_rate=1e-5  --log_dir=logs/convdr_multi_orquac  --num_train_epochs=1  --model_type=dpr  --log_steps=100  --ranking_task
```

To disable the KD loss, simply set the `--no_mse` flag.

## ConvDR Inference

Run `./drivers/run_convdr_inference.py` to get inference results. `output_file` is the [OpenMatch](https://github.com/thunlp/OpenMatch)-format file for reranking, and `output_trec_file` is the TREC-style run file which can be evaluated by the [trec_eval](https://github.com/usnistgov/trec_eval) tool.

```bash
# OR-QuAC
python drivers/run_convdr_inference.py  --model_path=checkpoints/convdr-multi-orquac.cp  --eval_file=datasets/or-quac/test.jsonl  --query=no_res  --per_gpu_eval_batch_size=8  --cache_dir=../ann_cache_dir  --ann_data_dir=datasets/or-quac/embeddings  --qrels=datasets/or-quac/qrels.tsv  --processed_data_dir=datasets/or-quac/tokenized  --raw_data_dir=datasets/or-quac   --output_file=results/or-quac/multi_task.jsonl  --output_trec_file=results/or-quac/multi_task.trec  --model_type=dpr  --output_query_type=test.raw  --use_gpu
# CAsT-19
python drivers/run_convdr_inference.py  --model_path=checkpoints/convdr-kd-cast19  --eval_file=datasets/cast-19/eval_topics.jsonl  --query=no_res  --per_gpu_eval_batch_size=8  --cache_dir=../ann_cache_dir  --ann_data_dir=datasets/cast-19/embeddings  --qrels=datasets/cast-19/qrels.tsv  --processed_data_dir=datasets/cast-19/tokenized  --raw_data_dir=datasets/cast-19   --output_file=results/cast-19/kd.jsonl  --output_trec_file=results/cast-19/kd.trec  --model_type=rdot_nll  --output_query_type=raw  --use_gpu  --cross_validation
```

The query embedding inference always takes the first GPU. If you set the `--use_gpu` flag (recommended), the retrieval will be performed on the remaining GPUs. The retrieval process consumes a lot of GPU resources. To reduce the resource usage, we split all document embeddings into several blocks, perform searching one-by-one and finally combine the results. If you have enough GPU resources, you can modify the code to perform searching all at once.

## Download Trained Models

Three trained models can be downloaded with the following link: [CAsT19-KD-CV-Fold1](https://data.thunlp.org/convdr/convdr-kd-cast19-1.zip), [CAsT20-KD-Warmup-CV-Fold2](https://data.thunlp.org/convdr/convdr-kd-cast20-2.zip) and [ORQUAC-Multi](https://data.thunlp.org/convdr/convdr-multi-orquac.cp).

## Results

[Download ConvDR and baseline runs on CAsT](https://drive.google.com/file/d/1F0RwA9sZscUAyE0IyQ7PMrgzNVqDnho5/view?usp=sharing)

## Contact

Please send email to ~~yus17@mails.tsinghua.edu.cn~~ yushi17@foxmail.com.
