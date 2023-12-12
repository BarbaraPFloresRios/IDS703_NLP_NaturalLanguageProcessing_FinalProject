## Textual Crossroads: Classifying Documents between the Bible and Hamlet

<img src="https://t4.ftcdn.net/jpg/03/32/63/37/360_F_332633731_eL5iv27kZRe6p8J8bqQtIOV4XYV5mkdH.jpg" alt="Descripción opcional" style="width: 100%; height: auto;">


Daniela Jiménez

Shaila Güereca

Bárbara Flores


#### Objective

The objective of this study is to classify documents, distinguishing between the Bible and Hamlet, two emblematic literary works with markedly different styles and themes. To achieve this, we will employ a generative method like Naive Bayes and a discriminative method such as a Neural Network. Our aim is to identify the distinctive patterns that characterize these works, ultimately developing a robust classifier capable of discerning the subtle differences defining the essence of the Bible and the poetic expression of Hamlet. In exploring textual crossroads, we aspire to not only contribute to a deeper understanding of the complexities that set apart these two timeless literary influences but also to comprehend the differences between these two methods of generative and discriminative classification.


#### Requirements

To run this project, it is necessary to...

#### Results
```bash
DOCUMENT INFORMATION

h0 = Hamlet by Shakespeare
Number of sentences in h0: 3,106
Number of unique words in h0: 5,447

h1 = Bible
Number of sentences in h1: 3,106
Number of unique words in h1: 3,975

__________________________________________________

RESULTS NAIVE BAYES

Naive Bayes (train): 0.9962439635127884
Naive Bayes (test): 0.9967793880837359

Confusion Matrix for Naive Bayes (Train)
╒══════════╤═══════════════╤═══════════════╕
│          │   Predicted 0 │   Predicted 1 │
╞══════════╪═══════════════╪═══════════════╡
│ Actual 0 │          2787 │             7 │
├──────────┼───────────────┼───────────────┤
│ Actual 1 │            14 │          2783 │
╘══════════╧═══════════════╧═══════════════╛

Confusion Matrix for Naive Bayes (Test)
╒══════════╤═══════════════╤═══════════════╕
│          │   Predicted 0 │   Predicted 1 │
╞══════════╪═══════════════╪═══════════════╡
│ Actual 0 │           311 │             1 │
├──────────┼───────────────┼───────────────┤
│ Actual 1 │             1 │           308 │
╘══════════╧═══════════════╧═══════════════╛

Misclassified Sentences for Naive Bayes (Test)

- ' , : Happily a an childe come for he is man old s say second the them they time to twice
- ? Is alive he yet
__________________________________________________


RESULTS LOGISTIC REGRESSION WITH TF-IDF

Logistic Regression with TF-IDF (train): 0.9998211411196566
Logistic Regression with TF-IDF (test): 0.9838969404186796

Confusion Matrix for Logistic Regression with TF-IDF (Train)
╒══════════╤═══════════════╤═══════════════╕
│          │   Predicted 0 │   Predicted 1 │
╞══════════╪═══════════════╪═══════════════╡
│ Actual 0 │          2794 │             0 │
├──────────┼───────────────┼───────────────┤
│ Actual 1 │             1 │          2796 │
╘══════════╧═══════════════╧═══════════════╛

Confusion Matrix for Logistic Regression with TF-IDF (Test)
╒══════════╤═══════════════╤═══════════════╕
│          │   Predicted 0 │   Predicted 1 │
╞══════════╪═══════════════╪═══════════════╡
│ Actual 0 │           306 │             6 │
├──────────┼───────────────┼───────────────┤
│ Actual 1 │             4 │           305 │
╘══════════╧═══════════════╧═══════════════╛

Misclassified Sentences for Logistic Regression with TF-IDF (Test)

- . Thus father for him his wept
- , ? after hast hotly is me my pursued sin so that thou what
- ' , : Happily a an childe come for he is man old s say second the them they time to twice
- Words
- , : Ambition Ambitious Dreame Which a are dreames for indeed is meerely shadow substance very
- , . And O away halfe it liue of other part purer throw with worser
- , And Obiect This heerein made of present probation the truth
- , . 45 7 : And God and before by deliverance earth great in lives me posterity preserve save sent the your
- Our Sonne shall win
- ? Is alive he yet
__________________________________________________


SYNTHETIC DATA INFORMATION

h0 = Synthetic Hamlet by Shakespeare
Number of sentences in synthetic data h0: 1,000
Number of unique words in synthetic data h0: 4,085

h1 = Synthetic Bible
Number of sentences in synthetic data h1: 1,000
Number of unique words in synthetic data h1: 2,431

__________________________________________________

SYNTHETIC RESULTS


RESULTS NAIVE BAYES ON SYNTHETIC DATA

Naive Bayes on Synthetic Data (train): 0.9988888888888889
Naive Bayes on Synthetic Data (test): 0.985

Confusion Matrix for Naive Bayes on Synthetic Data (Train)
╒══════════╤═══════════════╤═══════════════╕
│          │   Predicted 0 │   Predicted 1 │
╞══════════╪═══════════════╪═══════════════╡
│ Actual 0 │           896 │             2 │
├──────────┼───────────────┼───────────────┤
│ Actual 1 │             0 │           902 │
╘══════════╧═══════════════╧═══════════════╛

Confusion Matrix for Naive Bayes on Synthetic Data (Test)
╒══════════╤═══════════════╤═══════════════╕
│          │   Predicted 0 │   Predicted 1 │
╞══════════╪═══════════════╪═══════════════╡
│ Actual 0 │            99 │             3 │
├──────────┼───────────────┼───────────────┤
│ Actual 1 │             0 │            98 │
╘══════════╧═══════════════╧═══════════════╛

Misclassified Sentences for Naive Bayes on Synthetic Data (Test)

- , . all and brokenfooted cast fire found inward much on pitch s tell vain
- Carowses Partizan Peace Reads his say she speechlesse these to vnseene working your
- . : Arme Choller Lordship Robin a and but by of rac the there willing
__________________________________________________

RESULTS LOGISTIC REGRESSION WITH TF-IDF ON SYNTHETIC DATA

Logistic Regression with TF-IDF on Synthetic Data (train): 1.0
Logistic Regression with TF-IDF on Synthetic Data (test): 0.97

Confusion Matrix for Logistic Regression with TF-IDF on Synthetic Data (Train)
╒══════════╤═══════════════╤═══════════════╕
│          │   Predicted 0 │   Predicted 1 │
╞══════════╪═══════════════╪═══════════════╡
│ Actual 0 │           898 │             0 │
├──────────┼───────────────┼───────────────┤
│ Actual 1 │             0 │           902 │
╘══════════╧═══════════════╧═══════════════╛

Confusion Matrix for Logistic Regression with TF-IDF on Synthetic Data (Test)
╒══════════╤═══════════════╤═══════════════╕
│          │   Predicted 0 │   Predicted 1 │
╞══════════╪═══════════════╪═══════════════╡
│ Actual 0 │           100 │             2 │
├──────────┼───────────────┼───────────────┤
│ Actual 1 │             4 │            94 │
╘══════════╧═══════════════╧═══════════════╛

Misclassified Sentences for Logistic Regression with TF-IDF on Synthetic Data (Test)

- , And The concernings face hind my of
- Carowses Partizan Peace Reads his say she speechlesse these to vnseene working your
- Adue Foole Isaac Laban circumcised eldest lovest man one out peace see
- , 43 Beards LORD Noble down in judge ten them vnderstanding wages would
- . : Beare Naturall be he me of one shame to willing
- And Beard Loue Queene With and anoint daughers him in numbering of sometimes the thus
```
