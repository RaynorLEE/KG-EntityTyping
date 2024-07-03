# The Integration of Semantic and Structural Knowledge in Knowledge Graph Entity Typing (SSET)
#### This repo provides the source code & data of our paper: "The Integration of Semantic and Structural Knowledge in Knowledge Graph Entity Typing" published in the proceedings of the 2024 Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL2024 main, Oral)

## Dependencies
* python=3.10
* PyTorch 1.13.1+cu117
* transformers 4.33.1

## Running the code
### Dataset
* You may download the datasets and model checkpoints from this [link](https://drive.google.com/drive/folders/1kUwUkf80Ved9IJ0k_Njd1PX5a5JvAo7_?usp=sharing).

### Model training and testing
* Please see commands.sh file


* **Note:** Before running, you need to create the ./save folder first.

## Citation
If you find this code useful, please consider citing the following paper.
```
@inproceedings{li-etal-2024-integration,
    title = "The Integration of Semantic and Structural Knowledge in Knowledge Graph Entity Typing",
    author = "Li, Muzhi  and
      Hu, Minda  and
      King, Irwin  and
      Leung, Ho-fung",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.369",
    pages = "6625--6638",
}
```

## Acknowledgement
We refer to the code of [TET](https://github.com/zhiweihu1103/ET-TET) and [MiNer](https://github.com/jinzhuoran/MiNer/). Thanks for their contributions.
