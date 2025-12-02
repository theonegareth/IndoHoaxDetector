user@MSI:/home$ /bin/python3 "/home/user/Machine Learning/IndoHoaxDetector/train_indobert.py"
2025-11-24 11:58:28.774132: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-11-24 11:58:30.684369: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-11-24 11:58:32.662060: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
/home/user/.local/lib/python3.12/site-packages/matplotlib/projections/__init__.py:63: UserWarning: Unable to import Axes3D. This may be due to multiple versions of Matplotlib being installed (e.g. as a system package and as a pip package). As a result, the 3D projection is not available.
  warnings.warn("Unable to import Axes3D. This may be due to multiple versions of "
[INFO] Loading labeled data from: user/Machine Learning/IndoHoaxDetector/preprocessed_data_FINAL_FINAL.csv
[INFO] Using pre-cleaned text from dataset.
[INFO] Using device: cuda
[INFO] Loading IndoBERT: indobenchmark/indobert-base-p1
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at indobenchmark/indobert-base-p1 and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Map: 100%|█████████████████████████████████████████████████████████████████████████████| 62972/62972 [00:05<00:00, 11544.26 examples/s]
[INFO] Trainer output: /home/user/Machine Learning/IndoHoaxDetector/indobert_model/training; logs: /home/user/Machine Learning/IndoHoaxDetector/indobert_model/logs
[DEBUG] TrainingArguments init params: self, output_dir, overwrite_output_dir, do_train, do_eval, do_predict, eval_strategy, prediction_loss_only, per_device_train_batch_size, per_device_eval_batch_size, per_gpu_train_batch_size, per_gpu_eval_batch_size, gradient_accumulation_steps, eval_accumulation_steps, eval_delay, torch_empty_cache_steps, learning_rate, weight_decay, adam_beta1, adam_beta2, adam_epsilon, max_grad_norm, num_train_epochs, max_steps, lr_scheduler_type, lr_scheduler_kwargs, warmup_ratio, warmup_steps, log_level, log_level_replica, log_on_each_node, logging_dir, logging_strategy, logging_first_step, logging_steps, logging_nan_inf_filter, save_strategy, save_steps, save_total_limit, save_safetensors, save_on_each_node, save_only_model, restore_callback_states_from_checkpoint, no_cuda, use_cpu, use_mps_device, seed, data_seed, jit_mode_eval, bf16, fp16, fp16_opt_level, half_precision_backend, bf16_full_eval, fp16_full_eval, tf32, local_rank, ddp_backend, tpu_num_cores, tpu_metrics_debug, debug, dataloader_drop_last, eval_steps, dataloader_num_workers, dataloader_prefetch_factor, past_index, run_name, disable_tqdm, remove_unused_columns, label_names, load_best_model_at_end, metric_for_best_model, greater_is_better, ignore_data_skip, fsdp, fsdp_min_num_params, fsdp_config, fsdp_transformer_layer_cls_to_wrap, accelerator_config, parallelism_config, deepspeed, label_smoothing_factor, optim, optim_args, adafactor, group_by_length, length_column_name, report_to, project, trackio_space_id, ddp_find_unused_parameters, ddp_bucket_cap_mb, ddp_broadcast_buffers, dataloader_pin_memory, dataloader_persistent_workers, skip_memory_metrics, use_legacy_prediction_loop, push_to_hub, resume_from_checkpoint, hub_model_id, hub_strategy, hub_token, hub_private_repo, hub_always_push, hub_revision, gradient_checkpointing, gradient_checkpointing_kwargs, include_inputs_for_metrics, include_for_metrics, eval_do_concat_batches, fp16_backend, push_to_hub_model_id, push_to_hub_organization, push_to_hub_token, mp_parameters, auto_find_batch_size, full_determinism, torchdynamo, ray_scope, ddp_timeout, torch_compile, torch_compile_backend, torch_compile_mode, include_tokens_per_second, include_num_input_tokens_seen, neftune_noise_alpha, optim_target_modules, batch_eval_metrics, eval_on_start, use_liger_kernel, liger_kernel_config, eval_use_gather_object, average_tokens_across_devices
/home/user/Machine Learning/IndoHoaxDetector/train_indobert.py:184: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.
  trainer = Trainer(
[INFO] Starting fine-tuning...
{'loss': 0.1413, 'grad_norm': 6.066575527191162, 'learning_rate': 1.8943579972478037e-05, 'epoch': 0.16}                               
{'loss': 0.079, 'grad_norm': 2.5585596561431885, 'learning_rate': 1.788504287075262e-05, 'epoch': 0.32}                                
{'loss': 0.0669, 'grad_norm': 6.151716232299805, 'learning_rate': 1.6826505769027206e-05, 'epoch': 0.48}                               
{'loss': 0.0569, 'grad_norm': 0.012287363409996033, 'learning_rate': 1.576796866730179e-05, 'epoch': 0.64}                             
{'loss': 0.0528, 'grad_norm': 0.09763805568218231, 'learning_rate': 1.4709431565576376e-05, 'epoch': 0.79}                             
{'loss': 0.0486, 'grad_norm': 5.14717435836792, 'learning_rate': 1.3650894463850958e-05, 'epoch': 0.95}                                
{'eval_loss': 0.03707018122076988, 'eval_accuracy': 0.9922191345772132, 'eval_runtime': 83.2394, 'eval_samples_per_second': 151.311, 'eval_steps_per_second': 9.467, 'epoch': 1.0}                                                                                            
{'loss': 0.0229, 'grad_norm': 0.0047888318076729774, 'learning_rate': 1.2592357362125544e-05, 'epoch': 1.11}                           
{'loss': 0.019, 'grad_norm': 0.043374646455049515, 'learning_rate': 1.1533820260400128e-05, 'epoch': 1.27}                             
{'loss': 0.0191, 'grad_norm': 0.03199789300560951, 'learning_rate': 1.0475283158674712e-05, 'epoch': 1.43}                             
{'loss': 0.0247, 'grad_norm': 0.016201680526137352, 'learning_rate': 9.416746056949296e-06, 'epoch': 1.59}                             
{'loss': 0.021, 'grad_norm': 0.003989357966929674, 'learning_rate': 8.35820895522388e-06, 'epoch': 1.75}                               
{'loss': 0.0171, 'grad_norm': 0.013209746219217777, 'learning_rate': 7.299671853498465e-06, 'epoch': 1.91}                             
{'eval_loss': 0.033595748245716095, 'eval_accuracy': 0.993092497022628, 'eval_runtime': 83.3444, 'eval_samples_per_second': 151.12, 'eval_steps_per_second': 9.455, 'epoch': 2.0}                                                                                             
{'loss': 0.0149, 'grad_norm': 0.0014694147976115346, 'learning_rate': 6.24113475177305e-06, 'epoch': 2.06}                             
{'loss': 0.0042, 'grad_norm': 0.0010297272820025682, 'learning_rate': 5.182597650047635e-06, 'epoch': 2.22}                            
{'loss': 0.0052, 'grad_norm': 0.0013135488843545318, 'learning_rate': 4.124060548322219e-06, 'epoch': 2.38}                            
{'loss': 0.0059, 'grad_norm': 0.0010974216274917126, 'learning_rate': 3.0655234465968036e-06, 'epoch': 2.54}                           
{'loss': 0.003, 'grad_norm': 0.0037105802912265062, 'learning_rate': 2.0069863448713877e-06, 'epoch': 2.7}                             
{'loss': 0.0033, 'grad_norm': 0.002627098932862282, 'learning_rate': 9.484492431459723e-07, 'epoch': 2.86}                             
{'eval_loss': 0.03967500478029251, 'eval_accuracy': 0.9938070662961492, 'eval_runtime': 83.2653, 'eval_samples_per_second': 151.264, 'eval_steps_per_second': 9.464, 'epoch': 3.0}                                                                                            
{'train_runtime': 3660.1704, 'train_samples_per_second': 41.291, 'train_steps_per_second': 2.581, 'train_loss': 0.032396528182540026, 'epoch': 3.0}                                                                                                                           
100%|████████████████████████████████████████████████████████████████████████████████████████████| 9447/9447 [1:01:00<00:00,  2.58it/s]
[INFO] Saving fine-tuned IndoBERT to: /home/user/Machine Learning/IndoHoaxDetector/indobert_model
[INFO] Evaluating on validation set...
100%|████████████████████████████████████████████████████████████████████████████████████████████████| 788/788 [01:22<00:00,  9.50it/s]
Validation Classification report:
              precision    recall  f1-score   support

    FAKTA(0)     0.9915    0.9969    0.9942      6691
     HOAX(1)     0.9964    0.9903    0.9934      5904

    accuracy                         0.9938     12595
   macro avg     0.9940    0.9936    0.9938     12595
weighted avg     0.9938    0.9938    0.9938     12595

Confusion matrix [[TN, FP], [FN, TP]]:
[[6670   21]
 [  57 5847]]

[INFO] IndoBERT fine-tuning complete. Model saved to /home/user/Machine Learning/IndoHoaxDetector/indobert_model