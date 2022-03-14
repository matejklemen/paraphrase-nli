EXPERIMENT_DIR=$1

echo "Input copy BLEU score:"
sacrebleu ${EXPERIMENT_DIR}/test_ref.txt -i ${EXPERIMENT_DIR}/test_input_copy.txt -m bleu -b -w 4

echo "Input copy BERTscore:"
bert-score -r ${EXPERIMENT_DIR}/test_ref.txt -c ${EXPERIMENT_DIR}/test_input_copy.txt --lang en --model roberta-base --rescale_with_baseline 2> /dev/null | tail -1
