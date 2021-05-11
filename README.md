# Tausworthe RNG
---

## Abstract
---
This project presents the Tausworthe pseudo-random number generator as illustrated in Module 6 of ISYE 6644. The generator is excellent for producing $\mathcal U(0,1)$ pseudo-random numbers which can in turn be used for a plethora of other random variates via inverse transform and alternative methods. The generator presented here is tested with the $\chi^2$ *goodness-of-fit test* and the *Kolmogorov-Smirnov goodness-of-fit test* both implemented in the `scipy.stats` package. Runs tests for independence *Above and Below the Mean and Up and Down* are also conducted via my own implementation following the definitions in Module 6. The generator passes all tests swimmingly! Finally, some $\mathcal N(0, 1)$ variates are produced via the Box-Muller method, plotted, and tested with the *Anderson-Darling goodness-of-fit test* from `scipy.stats`. 

## Contents
---
- `README.md`: (you're reading me)
- `Tausworthe.py`: file of all code and functions related to my implementation of the Tausworthe generator.
- `environment.yml`: environment file just in case there's some funny business or problems when running code. 
- `Tausworthe_notebook.ipynb`: jupyter notebook of the report and code demonstration from `Tausworthe.py` (message me for access). 
