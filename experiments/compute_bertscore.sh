EXPERIMENT_DIR=$1

echo "Dev BERT scores:"
echo "- greedy decoding:"
bert-score -r ${EXPERIMENT_DIR}/dev_ref.txt -c ${EXPERIMENT_DIR}/dev_greedy_hyp.txt --lang en --model roberta-base --rescale_with_baseline 2> /dev/null | tail -1
echo "- beam search decoding:"
bert-score -r ${EXPERIMENT_DIR}/dev_ref.txt -c ${EXPERIMENT_DIR}/dev_beam_hyp.txt --lang en --model roberta-base --rescale_with_baseline 2> /dev/null | tail -1
echo "- top K decoding:"
bert-score -r ${EXPERIMENT_DIR}/dev_ref.txt -c ${EXPERIMENT_DIR}/dev_top_k_hyp.txt --lang en --model roberta-base --rescale_with_baseline 2> /dev/null | tail -1
echo "- top P decoding:"
bert-score -r ${EXPERIMENT_DIR}/dev_ref.txt -c ${EXPERIMENT_DIR}/dev_top_p_hyp.txt --lang en --model roberta-base --rescale_with_baseline 2> /dev/null | tail -1

echo "Test BERT scores:"
echo "- greedy decoding:"
bert-score -r ${EXPERIMENT_DIR}/test_ref.txt -c ${EXPERIMENT_DIR}/test_greedy_hyp.txt --lang en --model roberta-base --rescale_with_baseline 2> /dev/null | tail -1
echo "- beam search decoding:"
bert-score -r ${EXPERIMENT_DIR}/test_ref.txt -c ${EXPERIMENT_DIR}/test_beam_hyp.txt --lang en --model roberta-base --rescale_with_baseline 2> /dev/null | tail -1
echo "- top K decoding:"
bert-score -r ${EXPERIMENT_DIR}/test_ref.txt -c ${EXPERIMENT_DIR}/test_top_k_hyp.txt --lang en --model roberta-base --rescale_with_baseline 2> /dev/null | tail -1
echo "- top P decoding:"
bert-score -r ${EXPERIMENT_DIR}/test_ref.txt -c ${EXPERIMENT_DIR}/test_top_p_hyp.txt --lang en --model roberta-base --rescale_with_baseline 2> /dev/null | tail -1
