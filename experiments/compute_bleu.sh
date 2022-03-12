EXPERIMENT_DIR=$1

echo "Dev BLEU scores:"
echo "- greedy decoding:"
sacrebleu ${EXPERIMENT_DIR}/dev_ref.txt -i ${EXPERIMENT_DIR}/dev_greedy_hyp.txt -m bleu -b -w 4
echo "- beam search decoding:"
sacrebleu ${EXPERIMENT_DIR}/dev_ref.txt -i ${EXPERIMENT_DIR}/dev_beam_hyp.txt -m bleu -b -w 4
echo "- top K decoding:"
sacrebleu ${EXPERIMENT_DIR}/dev_ref.txt -i ${EXPERIMENT_DIR}/dev_top_k_hyp.txt -m bleu -b -w 4
echo "- top P decoding:"
sacrebleu ${EXPERIMENT_DIR}/dev_ref.txt -i ${EXPERIMENT_DIR}/dev_top_p_hyp.txt -m bleu -b -w 4

echo "Test BLEU scores:"
echo "- greedy decoding:"
sacrebleu ${EXPERIMENT_DIR}/test_ref.txt -i ${EXPERIMENT_DIR}/test_greedy_hyp.txt -m bleu -b -w 4
echo "- beam search decoding:"
sacrebleu ${EXPERIMENT_DIR}/test_ref.txt -i ${EXPERIMENT_DIR}/test_beam_hyp.txt -m bleu -b -w 4
echo "- top K decoding:"
sacrebleu ${EXPERIMENT_DIR}/test_ref.txt -i ${EXPERIMENT_DIR}/test_top_k_hyp.txt -m bleu -b -w 4
echo "- top P decoding:"
sacrebleu ${EXPERIMENT_DIR}/test_ref.txt -i ${EXPERIMENT_DIR}/test_top_p_hyp.txt -m bleu -b -w 4
