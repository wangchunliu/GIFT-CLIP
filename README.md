# GIFT-CLIP


>This paper introduces an end-to-end framework GIFT-CLIP. 


## 📚 Dataset Download
Training datasets are available [`here`](https://drive.google.com/drive/folders/1kmH33IAclyZcHAbchtt2Hk_9YAw6PtJx?usp=sharing) .

## 📕 Code Path

#### Code Structures
There are four parts in the code.
- **model**: It contains the main files for GIFT-CLIP network.
- **data**: It contains the pre-training data splits and downstream dataset.
- **checkpoints**: It saves checkpoint for reloading.
- **script**: The training scripts for GIFT-CLIP.

## 🔬 Dependencies

- ```Python 3```
- ```PyTorch >= 1.8.0```
- ```Transformers>= 4.11.3```
- ```NumPy```
- All experiments are performed with one A100 GPU.

## 🚀 Train & Eval

The training script:
```shell
bash script/run_pairwise.sh
```

**Note**: 
- you can open the `.sh` file for <a href="#Parameter">parameter</a> modification.

## 🤝 Cite:
Please consider citing this paper if you use the ```code``` or ```data``` from our work.
Thanks a lot :)


