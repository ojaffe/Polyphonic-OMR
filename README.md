# Polyphonic-OMR
 PyTorch code for end-to-end Optical Music Recognition (OMR) on polyphonic scores. Uses a CRNN to encode the image, then a transformer-decoder which acts as an autoregressive language model. Images are required to be in a specific format, which can be created from the preprocessing steps explained below.
 
We recommend a GPU with at least 8GB of VRAM to replicate our results.

## Dataset Creation
This requires (MusicXML, PNG) examples. If you don't have a dataset already, we support the creation of it by scraping and processing MuseScore files, found [here](https://github.com/Xmader/musescore-dataset). This will convert scraped MSCZ files into an image and sequence of symbolic labels, an example of which is `clef-G2 + keySig-FM + timeSig-4/4 + note-A4_quarter...`

Requirements:
1. Make sure you have [MuseScore](https://musescore.org/en) installed and locate the Plugins folder for it
2. Clone and install the plugin https://github.com/ojaffe/batch_export
3. Open Plugins -> Plugin Manager in MuseScore and make sure that new plugins (Batch Convert) is enabled

For downloading the data we support Linux and Windows. Download the .csv file of mscz files at the [previous link](https://github.com/Xmader/musescore-dataset) to `data/download/csv_file.csv`. Run the appropriate script in ./data/download/

Then preprocess the files in ./data/clean/ with the following commands:
1. Run "Batch Convert" in MuseScore on all MSCZ files to MusicXML, select "Resize Export"
2. Run removecredits.py on the generated MusicXML files
3. Run "Batch Convert" in MuseScore on the cleaned MusicXML files to MSCZ
4. Run "Batch Convert" in MuseScore on the new MSCZ to MusicXML and PNG
5. Run genlabels.py to generate labels for the MusicXML files
6. Run clean.py with appropriate arguments to clean the data, as needed

A dataset of 177k examples from the end of step 4 is available [here](https://drive.google.com/file/d/1dO81DS4cvIMuDXYgIha407P6zhlg4W67/view?usp=sharing).

Credits to [the following repository](https://github.com/sachindae/polyphonic-omr), which was a result from the paper: `"An Empirical Evaluation of End-to-End Polyphonic Optical Music Recognition"`

## Training
Change the config file according to your needs, then run:
```
python3 train/train.py configs/default.yaml
```
We log loss, accuracy, and occasional generations from the model during training with `tensorboard`. To view this, run:
```
python3 -m tensorboard.main --logdir log
```
