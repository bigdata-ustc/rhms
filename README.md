 # Learning Relation-Enhanced Hierarchical Solver for Math Word Problems
Source code for paper *Learning Relation-Enhanced Hierarchical Solver for Math Word Problems*.

 ## Dependencies
- python >= 3.6

- stanfordcorenlp
- torch

 ## Usage
- Preprocess dataset
```bash
python3 src/dataprocess/math23k.py
python3 src/dataprocess/similarity.py
```
- Train and test model
```bash
python3 src/rhms/main.py
```
For running arguments, please refer to [src/rhms/config.py](src/rhms/config.py).

 ## Citation
If you find our work helpful, please consider citing our paper.
```
@article{lin2024learning,
  title={Learning Relation-Enhanced Hierarchical Solver for Math Word Problems},
  author={Lin, Xin and Huang, Zhenya and Zhao, Hongke and Chen, Enhong and Liu, Qi and Lian, Defu and Li, Xin and Wang, Hao},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2024},
  volume={35},
  number={10},
  pages={13830--13844},
  publisher={IEEE}
}
```
```
@inproceedings{lin2021hms,
  title={HMS: A Hierarchical Solver with Dependency-Enhanced Understanding for Math Word Problem},
  author={Lin, Xin and Huang, Zhenya and Zhao, Hongke and Chen, Enhong and Liu, Qi and Wang, Hao and Wang, Shijin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={5},
  pages={4232--4240},
  year={2021}
}
```
