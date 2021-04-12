- number of extracted paraphrases (train+validation+test):
- NOTE: bert-base-cased is mostly used for sanity checking

| Strategy\model                   | bert-base-cased | roberta-base | roberta-large |  roberta-large-stronger |
| -------------------------------- | --------------- | ------------ | ------------- | ----------------------- |
| argmax                           |  17891          |  17731       |  16617        |  14494                  |
| thresh=0.5                       |  16433          |  17172       |  15628        |  14345                  |
| thresh=0.75                      |  10422          |  11916       |  10065        |  11045                  |
| thresh=0.9                       |  5924           |  8030        |  6486         |  8265                   |
| MCD_N, argmax                    |  TBD            |  TBD         |  TBD          |  TBD                    |
| MCD_N, thresh=0.5                |  TBD            |  TBD         |  TBD          |  TBD                    |
| MCD_N, thresh=0.75               |  TBD            |  TBD         |  TBD          |  TBD                    |
| MCD_N, thresh=0.9                |  TBD            |  TBD         |  TBD          |  TBD                    |
