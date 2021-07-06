# Uniformity tests

Implementation in Python of statistical tests of uniformity, to be used for cryptographic purposes. These tests were initially described in the article "A comparison of uniformity tests" by Y. Marhuenda, D. Morales, and M. C. Pardo

In addition, we show the following results, obtained after experimenting with the battery to verify its correct implementation:

- Kolmogorov-Smirnov test on the lists of p-values obtained in each test with four generators (dev / urandom, CryptGenRandom (), python.secrets(), and qRNG)
- Graphical distribution of common statistics (mean, standard deviation, and quartiles) of the p-values
- Graphical distribution of failed tests for three levels of significance (α = 0.05, α = 0.01 and α = 0.001)
- Distribution of p-values in a histogram
