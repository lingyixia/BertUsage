 python bert/run_classifier.py --task_name=mytask --do_train=true --do_eval=true --data_dir=data --vocab_file=../bertpremodel/vocab.txt --bert_config_file=../bertpremodel/bert_config.json --init_checkpoint=../bertpremodel/bert_model.ckpt --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=1.0 --output_dir=output