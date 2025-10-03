[W1002 17:03:02.052553035 Utils.hpp:137] Warning: Environment variable NCCL_ASYNC_ERROR_HANDLING is deprecated; use TORCH_NCCL_ASYNC_ERROR_HANDLING instead (function operator())
[W1002 17:03:02.057094436 Utils.hpp:137] Warning: Environment variable NCCL_ASYNC_ERROR_HANDLING is deprecated; use TORCH_NCCL_ASYNC_ERROR_HANDLING instead (function operator())
[W1002 17:03:02.059366403 Utils.hpp:137] Warning: Environment variable NCCL_ASYNC_ERROR_HANDLING is deprecated; use TORCH_NCCL_ASYNC_ERROR_HANDLING instead (function operator())
[W1002 17:03:02.060481232 Utils.hpp:137] Warning: Environment variable NCCL_ASYNC_ERROR_HANDLING is deprecated; use TORCH_NCCL_ASYNC_ERROR_HANDLING instead (function operator())
Using device: cuda:0, mixed_precision=fp16
/home/spark/miniconda3/lib/python3.13/site-packages/torch/nn/modules/transformer.py:392: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(
/home/spark/miniconda3/lib/python3.13/site-packages/torch/nn/modules/transformer.py:392: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(
/home/spark/miniconda3/lib/python3.13/site-packages/torch/nn/modules/transformer.py:392: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(
/home/spark/miniconda3/lib/python3.13/site-packages/torch/nn/modules/transformer.py:392: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(
[rank0]:[W1002 17:05:32.830962011 Utils.hpp:112] Warning: Environment variable NCCL_ASYNC_ERROR_HANDLING is deprecated; use TORCH_NCCL_ASYNC_ERROR_HANDLING instead (function operator())
[rank1]:[W1002 17:05:32.831952197 Utils.hpp:112] Warning: Environment variable NCCL_ASYNC_ERROR_HANDLING is deprecated; use TORCH_NCCL_ASYNC_ERROR_HANDLING instead (function operator())
[rank2]:[W1002 17:05:32.833430612 Utils.hpp:112] Warning: Environment variable NCCL_ASYNC_ERROR_HANDLING is deprecated; use TORCH_NCCL_ASYNC_ERROR_HANDLING instead (function operator())
[rank3]:[W1002 17:05:32.833917430 Utils.hpp:112] Warning: Environment variable NCCL_ASYNC_ERROR_HANDLING is deprecated; use TORCH_NCCL_ASYNC_ERROR_HANDLING instead (function operator())
[epoch 1 step 400] train_loss=2.0061
  VAL: loss=1.7368 acc=0.4704 f1w=0.3387 mae=0.8997
  >> saving /home/spark/xinze-project/Pred_ADNC/transformer_ADNC/outputs_adnc_ordinal_A9_depth6_e3/best_epoch1_step400_mae0.8997.pt
[epoch 1 step 800] train_loss=1.7481
  VAL: loss=1.6732 acc=0.4768 f1w=0.3600 mae=0.8623
  >> saving /home/spark/xinze-project/Pred_ADNC/transformer_ADNC/outputs_adnc_ordinal_A9_depth6_e3/best_epoch1_step800_mae0.8623.pt
[epoch 1 step 1200] train_loss=1.6607
  VAL: loss=1.5924 acc=0.4926 f1w=0.4118 mae=0.7984
  >> saving /home/spark/xinze-project/Pred_ADNC/transformer_ADNC/outputs_adnc_ordinal_A9_depth6_e3/best_epoch1_step1200_mae0.7984.pt
[epoch 1 step 1600] train_loss=1.5746
  VAL: loss=1.5523 acc=0.5033 f1w=0.4314 mae=0.7651
  >> saving /home/spark/xinze-project/Pred_ADNC/transformer_ADNC/outputs_adnc_ordinal_A9_depth6_e3/best_epoch1_step1600_mae0.7651.pt
[epoch 1 step 2000] train_loss=1.5520
  VAL: loss=1.5196 acc=0.5073 f1w=0.4180 mae=0.7650
  >> saving /home/spark/xinze-project/Pred_ADNC/transformer_ADNC/outputs_adnc_ordinal_A9_depth6_e3/best_epoch1_step2000_mae0.7650.pt
[epoch 1 step 2400] train_loss=1.5228
  VAL: loss=1.4897 acc=0.5182 f1w=0.4314 mae=0.7324
  >> saving /home/spark/xinze-project/Pred_ADNC/transformer_ADNC/outputs_adnc_ordinal_A9_depth6_e3/best_epoch1_step2400_mae0.7324.pt
[epoch 1 step 2800] train_loss=1.5035
  VAL: loss=1.4786 acc=0.5132 f1w=0.4415 mae=0.7212
  >> saving /home/spark/xinze-project/Pred_ADNC/transformer_ADNC/outputs_adnc_ordinal_A9_depth6_e3/best_epoch1_step2800_mae0.7212.pt
[epoch 1 step 3200] train_loss=1.5126
  VAL: loss=1.5282 acc=0.5100 f1w=0.4263 mae=0.7482
[epoch 1 step 3600] train_loss=1.5493
  VAL: loss=1.5781 acc=0.5006 f1w=0.4032 mae=0.7785
[epoch 1 step 4000] train_loss=1.5723
  VAL: loss=1.7695 acc=0.4930 f1w=0.3737 mae=0.8207
[epoch 1 step 4400] train_loss=1.5779
  VAL: loss=1.6382 acc=0.4968 f1w=0.3900 mae=0.7975
