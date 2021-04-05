- NOTE: test accuracies are reported only for convenience (so I don't have to rerun it later), **do not** use them for 
  selecting the best model!

| Model settings                                                                          | Val. acc. | Test acc. | 
| --------------------------------------------------------------------------------------- | --------- | --------- |
| bert-base-uncased, max_seq_len=41, lr=2e-5, B=256                                       |  0.8715   |  0.8679   |
| bert-base-uncased, max_seq_len=50, lr=2e-5, B=256                                       |  0.8704   |  0.8696   |
| bert-base-cased, max_seq_len=41, lr=2e-5, B=256                                         |  0.8712   |  0.8691   |
| [T] bert-base-cased, max_seq_len=51, lr=2e-5, B=256                                     |  0.8752   |  0.8733   |
| bert-large-uncased, max_seq_len=41, lr=2e-5, B=64                                       |  0.8962   |  0.8904   |
| [T] bert-large-uncased, max_seq_len=50, lr=2e-5, B=48                                   |  0.9028   |  0.8950   |
| bert-large-cased, max_seq_len=41, lr=2e-5, B=64                                         |  0.8957   |  0.8923   |
| bert-large-cased, max_seq_len=51, lr=2e-5, B=64                                         |  0.8961   |  0.8874   |
| bert-base-mtl-uncased, max_seq_len=43, lr=2e-5, B=256                                   |  0.8613   |  0.8590   |
| bert-base-mtl-uncased, max_seq_len=54, lr=2e-5, B=256                                   |  0.8671   |  0.8649   |
| bert-base-mtl-cased, max_seq_len=44, lr=2e-5, B=256                                     |  0.8605   |  0.8595   |
| [T] bert-base-mtl-cased, max_seq_len=56, lr=2e-5, B=256                                 |  0.8685   |  0.8661   |
| [T] roberta-base, max_seq_len=42, lr=2e-5, B=256                                        |  0.8919   |  0.8877   |
| roberta-base, max_seq_len=52, lr=2e-5, B=256                                            |  0.8834   |  0.8818   |
| roberta-large, max_seq_len=42, lr=2e-5, B=64                                            |  0.9034   |  0.9015   |
| [T] roberta-large, max_seq_len=52, lr=2e-5, B=48                                        |  0.9064   |  0.9039   |
| xlm-roberta-base, max_seq_len=47, lr=2e-5, B=256                                        |  0.8677   |  0.8663   |
| [T] xlm-roberta-base, max_seq_len=58, lr=2e-5, B=256                                    |  0.8730   |  0.8715   |
| [T] **ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli (**obtained online**)        |  0.9255   |  0.9185   |
