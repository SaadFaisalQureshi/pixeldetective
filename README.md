---
title: PixelDetective
emoji: 🔎
colorFrom: teal
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
python_version: "3.11"
app_file: app.py
---

# PixelDetective

PixelDetective compares four binary image manipulation classifiers for AIC354 at COMSATS University Islamabad, Lahore Campus. All four systems use frozen pretrained encoders from Hugging Face and train a logistic classifier on CASIA v2.

- [EfficientNet-B0](https://huggingface.co/google/efficientnet-b0)
- [ViT-Tiny](https://huggingface.co/WinKawaks/vit-tiny-patch16-224)
- [DINOv2 Base](https://huggingface.co/facebook/dinov2-base)
- [ConvNeXt V2 Tiny](https://huggingface.co/facebook/convnextv2-tiny-22k-224)
- [CASIA v2 Kaggle dataset](https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset)
- [Instructor reference repository](https://github.com/haqnawaz99/ml-teaching-notebooks)

## Changes from the Colab draft

The original CASIA preparation code and its split are retained. The uploaded draft initialized both models with TorchVision weights and could train and evaluate them once run. This version loads fixed checkpoints directly from Hugging Face, trains small logistic heads, compares both with a majority baseline, and supplies trained head artifacts and held-out predictions. The Gradio app uses the same preprocessing and scoring code as evaluation.

Kaggle archive SHA-256: `8d2e1513f83c0dd1593f8965cb2e72399191ebaf98896b5fbb8885d6a9dfefcb`.

## Reproduce the experiment

Open `PixelDetective.ipynb` in Colab. Follow its setup cell to stage this ZIP and display the bundled evaluation results. For a fresh run, select a GPU runtime and enable the dataset download and training cells. The experiment uses all readable authentic and manipulated images in the selected release across train, validation, and test splits. The original project split keeps exact decoded pixel duplicates, known shared CASIA source IDs and matching coarse thumbnail fingerprints together. Oversized connected groups remain in training. Unlisted relationships and unmatched near-duplicates can still cross splits. The held-out test is compared both across all file formats and on JPEGs alone.

On a local machine with Python 3.11 or 3.12:

```sh
python -m pip install -r requirements-train.txt
python train.py --data /path/to/extracted/CASIA --output artifacts --batch-size 32
python app.py
```

The data argument can be a parent of the CASIA folders. It searches for supported images directly inside `Au` and `Tp` directories. Use the original ZIP release. Avoid combining copies of the dataset under the same root. On GPU memory errors, reduce the batch size to 16 or 8. The downloaded encoders are cached by Hugging Face and their exact revisions are recorded.

## Measured results

| Model | Validation macro F1 | Test accuracy | Test macro F1 | Manipulated recall | JPEG-only macro F1 |
|---|---:|---:|---:|---:|---:|
| EfficientNet-B0 | 0.615 | 70.31% | 0.613 | 47.6% | 0.528 |
| ViT-Tiny | 0.667 | 72.53% | 0.662 | 62.8% | 0.528 |
| DINOv2 Base | 0.654 | 72.53% | 0.636 | 49.6% | 0.541 |
| ConvNeXt V2 Tiny | 0.681 | 73.89% | 0.671 | 61.1% | 0.557 |
| Majority class | — | 76.72% | 0.434 | 0.0% | — |

The same split included 8,370 training, 2,122 validation and 2,122 test images. Validation macro F1 selected **ConvNeXt V2 Tiny**. Its test macro F1 was 0.671 compared with 0.662 for ViT-Tiny. It detected 302 of 494 manipulated images, including 68 of 111 manipulated JPEGs. The majority baseline predicted Authentic for every image.

## Local upload verification

The Gradio upload endpoint was tested with these held-out JPEGs using all four models.

| Filename | Actual | EfficientNet-B0 | ViT-Tiny | DINOv2 Base | ConvNeXt V2 Tiny |
| --- | --- | --- | --- | --- | --- |
| `Au_ani_00002.jpg` | Authentic | Manipulated | Authentic | Authentic | Manipulated |
| `Tp_D_CNN_M_N_pla00042_pla00042_10976.jpg` | Manipulated | Authentic | Manipulated | Manipulated | Manipulated |

## Deploy

Create a Hugging Face Space using the Gradio SDK. Upload `README.md`, `app.py`, `modeling.py`, `requirements.txt`, and the `artifacts` directory containing all four trained heads and `metrics.json`. Use CPU Basic. The first prediction may take longer while the encoders download. Once the Space shows Running, verify an upload and the default model and the model selector from a browser with the same access as your evaluator. Paste the verified Space URL in the very first cell of the notebook.

## Submit

Submit the executed notebook, source code, and all four trained JSON head files. Download CASIA v2 directly from the Kaggle link above for the dataset requirement. The Kaggle release lists the dataset license as Unknown. The JSON heads contain the learned weights and bias, the class order, and the exact pretrained backbone revisions they require. Include `split.csv`, `unreadable.csv`, plus `test_predictions.csv` and `metrics.json` for reproducibility. The assignment also asks you to run and review the instructor's two emotion-classification notebooks separately.

PixelDetective offers image-level classification. The displayed sigmoid scores have not been calibrated. A strong score can reflect dataset-specific shortcuts, including compression and file-format patterns. Report performance as applying to the held-out CASIA sample and distinguish it from performance on unfamiliar images.

## Instructor reference notebooks

I executed both reference notebooks' code cells locally using their bundled CSV files. The binary Joy/Not Joy notebook selected the balanced decision tree (Joy F1 of 0.8146). The six-class notebook selected Linear SVM (macro F1 of 0.7805). Both fit TF-IDF on training text and transform the held-out text with the resulting vocabulary. These emotion scores belong to the instructor examples. The assignment specifies running them on Colab, so rerun the original notebooks there for that submission requirement.

The benchmark uses 128 × 128 RGB inputs, with 126 × 126 for DINOv2 to match its 14-pixel patch grid. The ViT-Tiny and DINOv2 encoders interpolate their pretrained positional embeddings. ConvNeXt uses its checkpoint’s resize and center-crop processor at the target size. Resizing can remove small edits.

## Streamlit app

Run `python -m streamlit run streamlit_app.py` locally or select `streamlit_app.py` as the Community Cloud entry point. The analysis tab supports single-model and four-model comparison modes. The benchmark tab reads the saved CASIA results, including a confusion matrix per model and the majority-class baseline. Comparison loads encoders one at a time to limit memory use.

The optional score sensitivity overlay measures the change in the manipulation score after covering individual tiles in the displayed preview. Positive drops are highlighted. The overlay is an explanation of score sensitivity and may highlight areas outside the ground-truth edit.

## Additional benchmark experiments

Install `requirements-train.txt` and pass a data directory that contains the paths in the relevant CSV. Outputs are separate from the original CASIA benchmark.

```sh
python experiments.py robustness --data /path/to/casia --models convnext vit --output artifacts/robustness_casia.json
python experiments.py robustness --data /path/to/external --external-manifest external.csv --output artifacts/robustness_external.json
python experiments.py resolution --data /path/to/casia --output artifacts/resolution.json
python experiments.py localization --data /path/to/masked --mask-manifest masks.csv --output artifacts/localization.json
```

- Robustness compares original inputs, JPEG quality 65, and half-size inputs. The external CSV uses `path,label` with labels 0 for authentic and 1 for manipulated. Use images independent of CASIA and of any training or tuning data. The app displays both robustness reports separately when present.
- Resolution fits a fresh 224 × 224 ConvNeXt classifier using the saved split, chooses its regularization on validation macro F1, and reports test metrics plus median warm embedding time against 128 × 128 on the same device. Use a GPU if available and reduce `--batch-size` if necessary.
- Localization measures overlap between the score sensitivity grid and binary ground-truth masks at a fixed score-drop threshold of 0.10. The mask CSV uses `path,mask`. Supply edited photos with same-size masks where white represents edited pixels. Mean IoU measures overlap for this explanation method.
