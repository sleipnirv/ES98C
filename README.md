# ES98C — Data-driven prediction of electrochemical corrosion kinetics

This repository contains the code for an end-to-end pipeline that extracts
electrochemical corrosion data from published papers (from both tables and
polarisation-curve figures) and trains a regression model to predict the
corrosion current density across material systems.

## Repository layout

- `project_v2.py` — main entry point: runs the text pathway and the image
  pathway for every PDF in `source_papers/`.
- `curve_pipeline.py` — the image pathway (page pre-filter → panel crop →
  curve extraction → legend matching → cleaning → record emission).
- `data_cleaner/` — Butler–Volmer-based outlier rejection and global cleaning.
- `page_analyzer/` — page-level VLM analysis and Tafel-panel cropping.
- `modeling/model.py` — k-means clustering + RF/GB regression + feature ablation.
- `mkreports/` — figure-generation scripts (workflow diagrams, results figures).
- `chart_extractor/` — chart-curve extraction (LineFormer + ChartDete + VLM OCR),
  adapted from
  [extract-line-chart-data](https://github.com/tdsone/extract-line-chart-data).
- `cleaned_data.csv` — the modelling dataset (147 records).

## Installation

1. Create and activate a Python 3.10 virtual environment.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Install the mmdet fork used by ChartDete (obtain the `third_party/ChartDete`
   directory from the original `extract-line-chart-data` repository, then):

   ```bash
   pip install -e chart_extractor/third_party/ChartDete/mmdet
   ```

   On Windows there is no CUDA wheel for `mmcv-full`; install the CPU-only build
   (see the comment in `requirements.txt`).

4. Download the LineFormer and ChartDete model checkpoints from the original
   [extract-line-chart-data](https://github.com/tdsone/extract-line-chart-data)
   repository and place them as described in that repository's README.

5. Run a local Ollama instance with the `qwen3-vl:8b-instruct` model:

   https://ollama.com/library/qwen3-vl

## Usage

```bash
python project_v2.py
```

The pipeline reads PDFs from `source_papers/` and writes
`extracted_data_text.csv` and `extracted_data_image.csv`.

## Validation paper

The end-to-end example used throughout the project is:

> R. Yamanoglu, E. Fazakas, F. Ahnia, D. Alontseva, F. Khoshnaw, "Pitting
> corrosion behaviour of austenitic stainless-steel coated on Ti6Al4V alloy in
> chloride solutions", *Advances in Materials Science*, vol. 21, no. 2,
> pp. 5–15, 2021.

It is open access (CC BY-NC-ND 3.0) and can be downloaded from:

https://doi.org/10.2478/adms-2021-0007
