# The Integration of Semantic and Structural aware Knowledge Graph Entity Typing (SSET)
#### This repo provides the source code & data of our paper: "The Integration of Semantic and Structural Knowledge in Knowledge Graph Entity Typing" that will be published in the proceedings of the 2024 Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL2024 main)

## Dependencies
* python=3.10
* PyTorch 1.13.1+cu117
* transformers 4.33.1

## Running the code
### Dataset
* You may download the datasets and model checkpoints from this [link](https://drive.google.com/drive/folders/1kUwUkf80Ved9IJ0k_Njd1PX5a5JvAo7_?usp=sharing).

### Model training and testing
* Please see commands.sh file


* **Note:** Before running, you need to create the ./logs folder first.

## Citation
If you find this code useful, please consider citing the following paper.
We will alter the citation information once the final version of this paper is released by the conference. 
```
@misc{li2024integration,
      title={The Integration of Semantic and Structural Knowledge in Knowledge Graph Entity Typing}, 
      author={Muzhi Li and Minda Hu and Irwin King and Ho-fung Leung},
      year={2024},
      eprint={2404.08313},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Acknowledgement
We refer to the code of [TET](https://github.com/zhiweihu1103/ET-TET) and [MiNer](https://github.com/jinzhuoran/MiNer/). Thanks for their contributions.
