Requires Tensorflow version 1.9

#### Data Preprocessing:
1. First download/extract the [UMLS](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html). This project assumes the UMLS files are laid out as such:
    ```
    <UMLS_DIR>/
        META/
            MRCONSO.RRF
            MRSTY.RRF
        NET/
            SRSTRE1
    ```
2. Create Metathesaurus triples

    ```bash
    python -m eukg.data.create_triples <UMLS_DIR>
    ```
   This will create the Metathesaurus train/test triples in data/metathesaurus.
3. Create Semantic Network Triples
    ```bash
    python -m eukg.data.create_triples <UMLS_DIR>
    ```

#### Training:
To train the Metathesaurus Discriminator:
```bash
python -m eukg.train --mode=disc --model=transd --run_name=transd-disc --no_semantic_network
```
To train the both Metathesaurus and Semantic Network Discriminators:
```bash
python -m eukg.train --mode=disc --model=transd --run_name=transd-sn-disc
```
To train the Metathesaurus Generator:
```bash
python -m eukg.train --mode=gen --model=distmult --run_name=dm-gen --no_semantic_network --no_semantic_network
```
To train the Metathesaurus and Semantic Network Generators:
```bash
python -m eukg.train --mode=gen --model=distmult --run_name=dm-sn-gen
```
To train the full GAN model:
```bash
python -m eukg.train --mode=gan --model=transd --run_name=gan --dis_run_name=transd-sn-disc --gen_run_name=dm-sn-gen
```
Note that the GAN model requires a pretrained discriminator and generator