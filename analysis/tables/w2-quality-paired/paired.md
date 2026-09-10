# TMMLU+ paired comparison against `bf16` (same items, exact McNemar)

| cell | n shared | accuracy | baseline accuracy | delta | 95% paired CI | only cell right | only baseline right | McNemar p |
|---|---|---|---|---|---|---|---|---|
| awq | 19680 | 0.5797 | 0.5911 | -0.0113 | [-0.0164, -0.0063] | 1110 | 1333 | 6.98e-06 |
| fp8 | 19680 | 0.5909 | 0.5911 | -0.0002 | [-0.0029, +0.0027] | 413 | 416 | 0.945 |
| gptq | 19680 | 0.5703 | 0.5911 | -0.0208 | [-0.0255, -0.0156] | 1020 | 1429 | 1.41e-16 |
