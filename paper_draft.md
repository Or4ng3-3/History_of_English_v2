# Reconstructing Old English from Modern Germanic Cognates with Joint Grapheme-Phoneme Modeling

## Abstract

This paper studies Old English reconstruction from modern Germanic cognates. Traditionally, historical linguists reconstruct earlier forms through the comparative method: they compare cognates across related languages, identify regular sound correspondences, and infer the most plausible ancestral form. Building on *Ab Antiquo: Neural Proto-language Reconstruction*, this project asks whether part of that comparative reasoning can be modeled computationally. I shift the target family from Romance to Germanic and replace the recurrent sequence model with a T5-style encoder-decoder, specifically a byte-level variant that can handle IPA symbols and historical characters without tokenization problems. I compare two input settings: a shape-only baseline and a joint grapheme-phoneme model that combines orthography with IPA for each cognate. The data come from a reconstructed Germanic root dataset collected from Kakki and aligned with cognate forms in modern English, German, Dutch, and Swedish. On the held-out development set, the shape-only system reaches an average edit distance of 1.37 and an exact-match rate of 32.99%, while the joint system reaches an average edit distance of 1.44 and an exact-match rate of 30.41% on word form. Although the joint model is not uniformly better at the aggregate level, it produces clear gains on phonologically transparent cases and gives a more linguistically informed output space. The results suggest that combining graphemic and phonological information helps most when regular sound correspondences are visible, but may hurt when orthography itself carries a conservative historical clue.

## 1. Introduction

Historical linguistics relies on the comparative method: if cognate forms across daughter languages show regular correspondences, then the ancestral form can often be reconstructed. In practice, this work is slow, interpretive, and dependent on the availability of well-curated cognate sets. Recent neural approaches have shown that sequence models can learn many of the regularities behind historical sound change and can assist reconstruction tasks.

The present study asks a narrower but useful question: can we reconstruct Old English forms from modern Germanic cognates by giving the model not only spelling information, but also phonological evidence? This is a natural extension of the idea explored in *Ab Antiquo*, but adapted to a different family and a different target. Instead of Latin reconstruction from Romance descendants, we work on Germanic data and aim at Old English forms. That shift matters because the data are smaller, the correspondences are noisier, and orthography alone is often less stable across the family.

My main contribution is a joint grapheme-phoneme setup. Each cognate is represented as a word form plus an IPA transcription, so the model can exploit both orthographic similarity and sound-level regularity. The paper compares this setting against a shape-only baseline, analyzes the development results, and studies where the joint representation helps or hurts.

## 2. Related Work

This work sits at the intersection of historical linguistics and neural sequence modeling.

The most direct predecessor is *Ab Antiquo: Neural Proto-language Reconstruction* (Meloni, Ravfogel, and Goldberg, 2021). That paper frames proto-word reconstruction as a sequence-to-sequence task over cognate sets and shows that neural models can outperform earlier methods. It also highlights a central lesson that still matters here: errors are not random, but often reflect the opacity of the historical change.

Earlier computational historical linguistics work laid the foundation for that result. Bouchard-Côté et al. developed probabilistic models of sound change and showed that ancient word forms can be reconstructed by explicitly modeling diachronic correspondences. Ciobanu and Dinu later studied cognate detection and Latin proto-word reconstruction, including sequence and alignment-based methods. Rama et al. also evaluated whether automatic cognate methods are strong enough for historical reconstruction tasks.

On the modeling side, T5 provides a useful abstraction because it turns every task into text-to-text generation. ByT5 strengthens that idea by working directly at the byte level, which is especially attractive for historical reconstruction: IPA symbols, macrons, thorn, ash, and other non-ASCII characters are handled naturally, and the model does not depend on fragile tokenization choices. For this reason, ByT5 is a better fit here than a standard wordpiece model.

Recent work has also continued to push neural reconstruction forward, including unsupervised approaches to protolanguage reconstruction. Taken together, these studies suggest that the key challenge is not whether neural models can learn correspondences at all, but how to present the input so that the model sees the right mix of phonological and orthographic evidence.

## 3. Methodology

### 3.1 Task Definition

The task is a supervised sequence-to-sequence reconstruction problem. For each training example, the model receives several modern Germanic cognates placed together in one input string. The label is the reconstructed ancestral form used as the target. In this project, the practical target is the Old English form associated with a reconstructed Proto-Germanic root in the dataset. Each example may contain English, German, Dutch, and Swedish cognates; if a language has no available cognate, that language is simply omitted from the input.

For example, a shape-only training instance has the following structure:

```text
Input:  English: free | German: frei | Dutch: vrij | Swedish: fri
Target: frēo
```

Another example with a missing English cognate may look like this:

```text
Input:  German: Ruf | Dutch: roep | Swedish: rop
Target: hrōp
```

This format is important because the model is not trained on single word pairs. It is trained to read a small cognate set as a multi-source input and to infer the historical target from cross-linguistic correspondences. The language labels are kept in the input so that the model can learn that the same letter or sound may provide different evidence depending on the source language.

### 3.2 Data Source

The dataset was collected from Kakki, where linguistically reconstructed Germanic roots are aligned with corresponding lexical items in several modern languages. In the project files, `germanic_parallel_dataset.csv` contains the Proto-Germanic ancestor, meaning, Old English target, and modern English, German, Dutch, and Swedish forms. The table contains 2,585 entries.

The raw data therefore has a structure like this:

| Proto-Germanic | Meaning | Old English target | English | German | Dutch | Swedish |
| --- | --- | --- | --- | --- | --- | --- |
| `frijaz` | free | `frēo` | free | frei | vrij | fri |
| `ab` | away | `ob` | of | ab | af | av |

For evaluation, I use `actual_validation_set.csv`, which contains 388 held-out examples in the same multi-source style. The model is evaluated by comparing its predicted Old English form with the gold target using edit distance.

### 3.3 Shape-Only Baseline

The first setting uses only written forms. This is closest to the orthographic reconstruction setup in *Ab Antiquo*. All available modern cognates are concatenated into a single string, separated by vertical bars. The label is the Old English word form:

```text
Input:  English: warm | German: warm | Dutch: warm | Swedish: varm
Target: wearm
```

In this design, the model must learn historical correspondences from spelling alone. This is a useful baseline because many cognate relationships are visible in the written forms. However, it also has an obvious limitation: spelling is not always a direct representation of sound. For example, English, German, Dutch, and Swedish may use similar letters for different sounds, or different letters for historically related sounds.

### 3.4 Joint Grapheme-Phoneme Input

The second setting is the main experimental improvement. Instead of giving only the written cognate forms, I attach an IPA transcription to each modern word. The target is also expanded from only the Old English spelling to the Old English spelling plus an IPA form:

```text
Input:  English: free [/fɹiː/] | German: frei [/fraɪ/] | Dutch: vrij [/vrɛɪ/] | Swedish: fri [/friː/]
Target: frēo [/freːo/]
```

Another example from the evaluation set is:

```text
Input:  German: Ruf [/ruːf/] | Dutch: roep [/rup/] | Swedish: rop [/ruːp/]
Target: hrōp [/hroːp/]
```

This joint format gives the model two kinds of evidence at the same time. The written form may preserve historical spelling clues, while the IPA transcription makes phonological correspondences explicit. This is especially useful for Germanic reconstruction because modern spellings are unevenly conservative and do not always reveal the same sound information across languages.

The modern IPA transcriptions are generated with `phonemizer` using the `espeak-ng` backend. The Old English IPA labels are generated by a rule-based converter. The converter handles macron vowels, Old English diphthongs, palatalization of `c` and `g` before front vowels, `sc` as a palatal fricative, `cg` as an affricate, and voicing of some intervocalic fricatives. These rules are not meant to replace expert phonological analysis, but they provide a consistent training signal for the joint model.

### 3.5 Why T5 and ByT5

I use a T5-style encoder-decoder because the task is naturally generative. The model must read a bundle of cognates and generate a historical form, which is exactly the kind of mapping that sequence-to-sequence models handle well. T5 also makes it easy to represent all inputs in one uniform text format, rather than designing separate alignment features for each language pair.

The implementation uses `google/byt5-small`, a byte-level T5 model. This choice is especially important for this project. The data include IPA symbols, macron vowels such as `ē` and `ā`, and Old English characters such as `þ`, `ð`, `æ`, and `ġ`. A byte-level model can process these symbols directly, while a standard subword tokenizer may split them unpredictably or treat them as rare fragments. ByT5 therefore reduces tokenization problems and lets the model learn from the actual historical and phonetic strings.

### 3.6 Improvements over *Ab Antiquo*

This experiment follows the general idea of *Ab Antiquo*: proto-form reconstruction can be treated as neural sequence-to-sequence learning from cognate sets. However, the present project changes the original design in three important ways.

First, the language family and target are different. *Ab Antiquo* reconstructs Latin from Romance languages. This project reconstructs Old English forms from modern Germanic evidence, using English, German, Dutch, and Swedish cognates aligned with reconstructed Germanic roots.

Second, the model architecture changes. *Ab Antiquo* uses a recurrent neural network architecture. This project uses a T5-style encoder-decoder, specifically ByT5, which is more flexible for text-to-text generation and more robust for non-ASCII historical and phonetic symbols.

Third, the representation changes. *Ab Antiquo* compares orthographic and phonetic datasets as separate settings. In contrast, this project compares a shape-only baseline with a joint grapheme-phoneme setting, where spelling and IPA are placed together in the same training example. The goal is not merely to ask whether spelling or sound is better, but whether the model can benefit from seeing both at once.

### 3.7 Training and Evaluation

The joint model is trained on `germanic_joint_ipa_dataset.csv`. The script constructs `input_text` and `target_text` fields, then splits the data with a 15% validation set. Inputs are truncated at 256 tokens and targets at 64 tokens. Training uses a learning rate of `2e-4`, batch size 8, seven epochs, beam-style generation at inference time, and checkpoint selection by validation loss.

Evaluation follows the metric used in *Ab Antiquo*: edit distance between the predicted historical word and the gold form. I report average edit distance and the percentage of predictions within edit-distance thresholds from 0 to 4. Exact match corresponds to edit distance 0.

## 4. Results Analysis

I compare two evaluation files:

- `eval/dev_evaluation_results_nosound.csv` for the shape-only setting
- `eval/joint_dev_evaluation_results.csv` for the joint grapheme-phoneme setting

The main development results are:

| Setting | Avg. edit distance | Exact match | <= 1 | <= 2 | <= 3 | <= 4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Shape-only | 1.37 | 32.99% | 57.22% | 79.64% | 95.36% | 98.71% |
| Joint grapheme-phoneme | 1.44 | 30.41% | 56.19% | 79.12% | 92.53% | 98.71% |

The first point is that both models usually get close to the target even when they do not produce an exact match. In both settings, nearly 80% of predictions are within two edits of the gold form, and 98.71% are within four edits. This means that many errors are not complete failures. They are often small mistakes in vowel quality, length, palatal marking, derivational ending, or the presence of an initial consonant such as `h`. This pattern is similar to the observation in *Ab Antiquo*: neural reconstruction errors tend to cluster around historically meaningful alternations rather than being random strings.

The second point is that the joint model is slightly worse on aggregate. The shape-only model has a lower average edit distance, 1.37 compared with 1.44, and a higher exact-match rate, 32.99% compared with 30.41%. This result is important because it prevents a simple conclusion that "adding phonology always helps." Across the 386 inputs that can be matched directly between the two result files, the joint model improves 55 items, worsens 76, and leaves the rest unchanged. The phonological representation is therefore selective: it helps in some linguistically transparent cases, but it also adds another source of noise.

The third point concerns comparison with *Ab Antiquo*. Meloni, Ravfogel, and Goldberg report much higher scores for Romance-to-Latin reconstruction. Their orthographic model reaches 64.1% exact reconstruction and an average edit distance of 0.65; even their phonetic model reaches 50.0% exact reconstruction and an average edit distance of 1.022. My numbers are lower. This does not necessarily mean that the model architecture is weaker. The experimental setting is different in several ways.

First, the dataset is smaller. *Ab Antiquo* uses 8,799 cognate sets after cleaning and extension, with 7,038 training examples. My dataset contains 2,585 entries, and the held-out development set contains 388 examples. A neural sequence model has fewer opportunities to learn rare correspondences, suffix patterns, and exceptional forms.

Second, the language-family setup is different. The Romance languages used in *Ab Antiquo* are all direct descendants of Latin, and the paper uses a large, carefully cleaned Romance dataset with multiple daughter languages. In my experiment, the input languages come from different branches of the Germanic family: English, German, Dutch, and Swedish do not preserve the same parts of the ancestral system equally well. Modern English is especially problematic because it has undergone heavy sound change, large-scale borrowing, and major orthographic conservatism. German, Dutch, and Swedish also differ in how they reflect Proto-Germanic and Old English patterns. The model is therefore not only learning spelling correspondences; it must also handle a more uneven distribution of historical evidence.

Third, the target forms themselves contain difficult historical details. Old English reconstruction involves vowel length, front rounded vowels, breaking, i-mutation, palatalized consonants, and special characters such as `ċ`, `ġ`, `þ`, `ð`, and `æ`. Many of these distinctions are weakly visible or invisible in the modern cognates. For example, modern spellings may preserve a related root but not the Old English vowel quality or palatal marking. This makes exact-match accuracy harsh: a prediction can be historically plausible and still lose one or two edit-distance points.

For these reasons, the lower accuracy should not be explained only by saying that Germanic is "more complex" in an absolute sense. A more precise explanation is that this Germanic reconstruction setup is harder for the model: the dataset is smaller, the daughter languages are more uneven as evidence for Old English, and the target notation demands fine-grained historical distinctions.

## 5. Case Study

### 5.1 When Joint Modeling Helps

The strongest gains appear when the historical correspondence is phonologically transparent and the IPA input exposes information that the spelling-only model misses. In these cases, adding IPA does not merely add decoration; it changes what evidence the model can use.

- `English: offlay | German: ablegen | Dutch: afleggen | Swedish: avlägga`
  - shape-only: `afleggan`
  - joint: `ofleċġan`
  - gold: `ofleċġan`

In this example, the shape-only model is attracted to the modern continental spelling pattern `ab-/af-/av-` and produces `afleggan`. That output is close to the modern cognates, but it misses two Old English features: the `of-` prefix and the palatalized spelling `ċġ`. The joint input gives extra phonological evidence: German `ablegen` is transcribed with initial `/ap-/`, Dutch `afleggen` with `/ɑf-/`, and Swedish `avlägga` with `/av-/`. These forms show a labial consonant alternation around the prefix, while the English form `offlay` also points toward an `of-` type form. The IPA helps the model treat these as related phonological variants rather than simply copying the most visible written prefix. It also helps with the consonant cluster near the end, where written `gg` or `g` does not by itself tell the model to produce the Old English palatalized `ċġ`.

- `English: day | German: Tag | Dutch: dag | Swedish: dag`
  - shape-only: `dagu`
  - joint: `dæġ`
  - gold: `dæġ`

This is a clear case where spelling alone is misleading. The written forms `Tag`, `dag`, and `dag` strongly suggest a surface form with `d-a-g`, so the shape-only model predicts `dagu`. The joint input adds the actual pronunciations: English `/deɪ/`, German `/tɑːk/`, Dutch `/dɑx/`, and Swedish `/dɑːɡ/`. These show that the final consonant has different modern reflexes: English has lost it in the surface pronunciation, German has final devoicing, Dutch has a fricative, and Swedish keeps a voiced stop. Seeing these sound differences together helps the model infer that the Old English target is not just a direct copy of modern `dag`, but the historical form `dæġ`. In other words, IPA makes the sound correspondence visible behind misleadingly similar spellings.

- `English: wound | Dutch: wonden`
  - shape-only: `wundan`
  - joint: `wundian`
  - gold: `wundian`

Here the difference is morphological rather than only segmental. The shape-only model predicts `wundan`, which is close in root shape but misses the verbal suffix `-ian`. The joint input contains English `/wuːnd/` and Dutch `/ʋɔndən/`, showing a stable root with a nasal-stop cluster but also a Dutch form with a verbal ending. The IPA does not directly spell out Old English `-ian`, but it helps the model separate the root `wund-` from the inflectional material in the modern form. The joint model therefore recovers the correct Old English verb `wundian`. This suggests that phonology can support not only consonant recovery but also morphological classification when the modern cognates include derivational or inflectional endings.

Together, these examples show the main advantage of the joint representation. The shape-only model often follows the most obvious modern spelling. The joint model can instead compare spelling with sound and recover a less surface-like but more historically appropriate form.

### 5.2 When Joint Modeling Hurts

The joint model can also over-correct. By "over-correct," I mean that the model applies a historical-looking change even though the simpler spelling-based reconstruction is already correct. We can tell that this is happening when the shape-only model exactly matches the gold form, but the joint model changes the vowel, adds an initial consonant, or otherwise produces a form that looks plausible in Old English but is wrong for that item.

- `English: leave | German: Laube | Dutch: love`
  - shape-only: `lēaf`
  - joint: `lāf`
  - gold: `lēaf`

The shape-only prediction is exactly correct: `lēaf`. The joint model changes it to `lāf`, replacing the front long vowel `ēa/ē`-type spelling with `ā`. This is an over-correction because the IPA forms contain strong back-vowel evidence: German `Laube` has `/aʊ/`, Dutch `love` has `/oː/`, and English `leave` has `/iː/`. These forms are not phonetically uniform, and the joint model appears to generalize from the back-vowel evidence toward `ā`. However, the correct Old English form preserves the fronted written form `lēaf`. In this case, orthography is the better historical clue than the modern sound shapes.

- `English: reif | German: Raub | Dutch: roof | Swedish: rov`
  - shape-only: `rēaf`
  - joint: `hrāf`
  - gold: `rēaf`

This example shows two over-corrections at once. The joint model changes the vowel from `ēa/ē` to `ā`, and it adds an initial `h`, producing `hrāf`. The added `h` is not random: Old English has many inherited `hr-` clusters, and the model has learned that some modern `r-` words correspond to older `hr-` forms. The problem is that this pattern is applied where it does not belong. The modern IPA forms all begin with an `r`-like sound and do not provide direct evidence for an initial `h`. The model is therefore using a learned historical pattern too broadly. The shape-only model avoids this by staying close to the spelling pattern and correctly predicts `rēaf`.

- `English: red | German: rot | Dutch: rood | Swedish: röd`
  - shape-only: `rēad`
  - joint: `hrēd`
  - gold: `rēad`

The same tendency appears here. The correct form is `rēad`, and the shape-only model gets it exactly right. The joint model predicts `hrēd`, again inserting initial `h` and also simplifying the vowel sequence. This looks like the model has learned a real Old English-looking pattern, but it has not learned the conditions under which the pattern should apply. From a linguistic point of view, this is a conditioning problem: historical correspondences are not just global substitutions. Whether an `h` is reconstructed before `r`, or whether a vowel should be `ēa` rather than `ē`, depends on lexical history and phonological environment. The joint model sees the IPA forms `/ɹɛd/`, `/roːt/`, `/roːt/`, and `/røːd/`, which emphasize an `r` onset and varying vowels, but they do not encode enough information to recover the exact Old English form by themselves.

These failures show the danger of adding phonology without enough data or stronger linguistic constraints. IPA can make sound correspondences clearer, but it can also encourage the model to treat a frequent historical pattern as more general than it really is. In these cases, the joint model is not making nonsense predictions; it is making plausible but overgeneralized historical predictions.

## 6. Discussion

The broader lesson is that joint modeling is useful, but not as a blunt replacement for shape-only prediction. It is best understood as an added source of evidence that helps when the historical change is regular and recoverable from sound. When the evidence is ambiguous, the extra signal may increase confidence in the wrong analysis.

This also explains why the comparison to *Ab Antiquo* should be cautious. That paper worked on a larger and more widely studied Romance dataset; this project works on a smaller Germanic dataset with a different target and a different evidence profile. The task is therefore harder in some respects, and improvements should be judged in linguistic terms, not just by raw aggregate accuracy.

Two limitations are worth stating explicitly. First, the dataset is relatively small, so the joint model has less evidence than large-scale reconstruction systems. Second, the Old English IPA labels are generated by hand-written rules rather than by human annotation, which keeps the output consistent but also makes the setup dependent on those phonological assumptions. The experiment is therefore best read as a controlled test of representation choice, not as a final description of Old English phonology.

## 7. Conclusion

This paper presents a neural Old English reconstruction experiment based on modern Germanic cognates. Compared with the original *Ab Antiquo* setup, the main changes are the use of a T5-style byte-level model, a Germanic rather than Romance target, and a joint grapheme-phoneme representation. The results show that the joint model does not improve every metric on this development set, but it does recover several historically important correspondences more reliably and produces better outputs on sound-transparent examples.

The overall conclusion is modest but useful: orthography and phonology are complementary, not interchangeable. A reconstruction model should be allowed to see both, but it still needs enough structure and data to decide when one source should dominate the other. In future work, a larger Germanic dataset and more careful human validation of the phonological labels would make it easier to separate modeling gains from annotation choices.

## References

- Meloni, C., Ravfogel, S., & Goldberg, Y. (2021). *Ab Antiquo: Neural Proto-language Reconstruction*.
- Raffel, C. et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*.
- Xue, L. et al. (2021). *ByT5: Towards a token-free future with pre-trained byte-to-byte models*.
- Bouchard-Côté, A., Griffiths, T. L., & Klein, D. (2009). *Improved reconstruction of proto-language word forms*.
- Bouchard-Côté, A. et al. (2013). *Automated reconstruction of ancient languages using probabilistic models of sound change*.
- Ciobanu, A. M., & Dinu, L. P. (2018). *Ab initio: Automatic Latin proto-word reconstruction*.
- Rama, T., List, J.-M., Wahle, J., & Jäger, G. (2018). *Are automatic methods for cognate detection good enough for phylogenetic reconstruction in historical linguistics?*
