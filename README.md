<img src="tabstar_logo.png" alt="TabSTAR Logo" width="50%">

**Welcome to the TabSTAR Research repo!**

🚧 This repository is still work in progress. 🚧

---

📚 **Resources**

* 📄 **Paper**: [TabSTAR: A Foundation Tabular Model With Semantically Target-Aware Representations](https://arxiv.org/abs/2505.18125)

* 🌐 **Project Website**: [TabSTAR](https://eilamshapira.com/TabSTAR/)

___

To install the repository, do:

```commandline
source init.sh
```

The main scripts provided are:

* `do_pretrain` which pretrains a TabSTAR model.
* `do_finetune` which finetunes a pretrained TabSTAR model on a downstream task.
* `do_baseline` which runs a baseline model on a downstream task.

<img src="tabstar_arch.png" alt="TabSTAR Arch" width="100%">


To pretrain TabSTAR, run the following command, controlling for the number of datasets:
```commandline
python do_pretrain.py --n_datasets=256
```

For debugging purpose, you can decrease the number, but this will harm downstream task performance.
At the end of the pretraining, you will get the name of the `pretrain_exp` which should be passed for finetuning for a given downstream task:
```commandline
python do_finetune.py --pretrain_exp=MY_PRETRAINED_EXP --dataset_id=46655
```

To compare the performance with a baseline, choose a model and a dataset and run:
```commandline
python do_baseline.py --model=rf --dataset_id=46655
```

## Using Custom Datasets

You can now use your own CSV datasets with TabSTAR! For detailed instructions, see [CUSTOM_DATASET_USAGE.md](CUSTOM_DATASET_USAGE.md).

**Quick example:**
```commandline
# Using the helper script (recommended)
python use_custom_dataset.py --csv_path your_data.csv --target_column target_col --pretrain_exp YOUR_MODEL

# Or directly with do_finetune.py
python do_finetune.py --pretrain_exp=YOUR_MODEL --dataset_id=custom --custom_csv_path=your_data.csv --custom_target_column=target_col
```
