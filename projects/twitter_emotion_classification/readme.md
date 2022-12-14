
![](https://i.imgur.com/pb2ocRH.png)


### Project Keywords 📒

`Sentiment` `Multiclass` `Emotion` `Classification` `NLP` `HuggingFace` `Transformers`

<br>

### Project Background 📥

- The emergence of social media networks/platforms has allowed people to writing posts, sharing photos and videos
- One of those social medial networks is `Twitter`; it is one of the most popular social media platforms in the world
- Twitter has 330 million monthly active users and 500 million tweets sent each day, which makes it an invaluable tool in ML/DL applications
- Understanding the `sentiment` (tone) of tweets is important for a variety of reasons: `business marketing`, `political views`, `behavior analysis`, just to mention a few
- `Emotions` are important in `Sentiment` analysis (not only `positive` & `negative`) as it gives a more detail about a person's opinion about a `topic` or `product`

<br>

### Key Terms ❓

`Sentiment analysis` | **[Section 5](https://www.kaggle.com/code/shtrausslearning/natural-language-processing#5-%7C-Sentiment-Analysis)**

>  Identify & classify the sentiments/tones that are expressed in the text source. social media posts are often useful in generating a vast amount of sentiment data upon analysis, which is useful in understanding the opinion of the people about a variety of topics

`Tokenisation` | **[Section 1](https://www.kaggle.com/code/shtrausslearning/natural-language-processing)**

>  Process of breaking up a sequence of text into pieces such as words, keywords, phrases, symbols and other elements called tokens, several types of techniques exist; `character`, `sentence`, `word` & `subword` tokenisation

`Embedding` | **[Section 4](https://www.kaggle.com/code/shtrausslearning/natural-language-processing#3-%7C-Advanced-Feature-Generation)**

> An embedding is the mapping of a discrete — categorical — variable or text to a vector of continuous numbers in vecor form 

<br>

### Kaggle Notebook 📖

Kaggle offer a very neat `ipynb` render, if you'd like to read the notebook on Kaggle, links are provided below:

- **[Twitter Emotion Classification](https://www.kaggle.com/code/shtrausslearning/twitter-emotion-classification)**

<br>

### Project Aim 🎯 

- We need to build a model based sentiment analysis tool that will be able to automatically identify emotional states (eg. anger, joy) that people express about your company's product on twitter
- Aside from the standard approach to `validation`, we'll apply the models on some recent `tweet` examples

<br>

### Target Labels 🏷️

Muliclass classification
- `surprise` `love` `fear` `anger` `sadness` `joy`

<br>

### Project Pipeline 📑

Kaggle notebook workflow

- `1` Generating Dataset
- `2` Exploratory Data Analysis
- `3` Tokenisation
- `4` Pretrained Embedding Model
- `5` Fine-Tuning Transformers
- `6` Model Error Analysis

<br>

### GitHub Folder Contents 📁




<br>


### Main Takeaways 📤

<br>

I. As with `CV` applications, transfer learning plays an important role in `NLP` in creating high accuracy classifiers

***

`Embedding` based `LogisticRegression` model inferior to fine tuned `tranformers` (transfer learning approach)

- Embedding LogisticRegression Model Accuracy: **0.634**
- Fine-Tuned DistilBERT Model Accuracy **0.933**

Confusion Matrix Data for the two models:

| Embedding LogisticRegression Model | Fine-Tuned DistilBERT Model |
| - | - |
| ![](https://www.kaggleusercontent.com/kf/113303788/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..BTBKIdhiDYgPCX7A_mj0pg.2Skv0-MVTR34K-8BixmuUOTYgKyY8ALAPUYftoK4AkVSkQLfz83vv2Ttin4gDPRcvsz9KuVv-ltVVDS0oMCB_CikbIoa6nvAddPiq8gdznTeepc3Qsd24FP2McI_xELYwSIJrVxkUmdCjiQrDyW-VMcnBkXlqCvqr9qz-B0MaqTL1uwcx-3aI1kc_8dXve4s_BvVR7dT76W59XTLc7D4CcVxQT5jbBwO5iLNYsqrnD1P0F6uGNh6rm9ANsNclv-ka57U7S9iQNqANwZtcFLKabiiH5HRh-Nzg3tDm7a7stVf7_LErWnUuW5w2fxNDuDmHZE36JRQLbQHKi5ZfrBD-RmyjYwvG1w3aiwIMvu-G4l-2ltBZvOjUZC_y69BLlTQFsP211aLUV-ahBaQahimWPLKr_u7T9ibCEcjhfXbCINOJsN5HnbBLNrxusfEDpIM7C06fCCGsOe0BnPpPhVCR7fiBh09vjmXtAT-kziFAQZvTdxf_W8dHD7p167lDThY617VYg9SXEqjUHzFaMkDenetf0ZiCbwtKY41Cb-bky5CmRiDQeYrWPc2aHvzStlUoD1k1ra52wfSdRn5ePrgbmNWtQB2BRW9pIzzUIkfBPOEGLZ8-95mB_Wj83o3qETvSyTrZs5iseRr5KsBrANlPjJJnxIqMwPM1bhvM8ywH3UPF1FFRPPxFQPEHOCJGXTM.6GrF35sEsLyoUZ2Du4GB4w/__results___files/__results___62_0.png) | ![](https://www.kaggleusercontent.com/kf/113303788/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..BTBKIdhiDYgPCX7A_mj0pg.2Skv0-MVTR34K-8BixmuUOTYgKyY8ALAPUYftoK4AkVSkQLfz83vv2Ttin4gDPRcvsz9KuVv-ltVVDS0oMCB_CikbIoa6nvAddPiq8gdznTeepc3Qsd24FP2McI_xELYwSIJrVxkUmdCjiQrDyW-VMcnBkXlqCvqr9qz-B0MaqTL1uwcx-3aI1kc_8dXve4s_BvVR7dT76W59XTLc7D4CcVxQT5jbBwO5iLNYsqrnD1P0F6uGNh6rm9ANsNclv-ka57U7S9iQNqANwZtcFLKabiiH5HRh-Nzg3tDm7a7stVf7_LErWnUuW5w2fxNDuDmHZE36JRQLbQHKi5ZfrBD-RmyjYwvG1w3aiwIMvu-G4l-2ltBZvOjUZC_y69BLlTQFsP211aLUV-ahBaQahimWPLKr_u7T9ibCEcjhfXbCINOJsN5HnbBLNrxusfEDpIM7C06fCCGsOe0BnPpPhVCR7fiBh09vjmXtAT-kziFAQZvTdxf_W8dHD7p167lDThY617VYg9SXEqjUHzFaMkDenetf0ZiCbwtKY41Cb-bky5CmRiDQeYrWPc2aHvzStlUoD1k1ra52wfSdRn5ePrgbmNWtQB2BRW9pIzzUIkfBPOEGLZ8-95mB_Wj83o3qETvSyTrZs5iseRr5KsBrANlPjJJnxIqMwPM1bhvM8ywH3UPF1FFRPPxFQPEHOCJGXTM.6GrF35sEsLyoUZ2Du4GB4w/__results___files/__results___76_0.png)

<br>

II. The **DistilBERT** is quite confident in its predictions to classify new twitter data

***

Unseen `tweet` sentiment classified by `transformer` as **joy** (label 1) 

> 'I watched a movie last night, it was quite brilliant'

```python

[{'label': 'LABEL_0', 'score': 0.004893782548606396},
 {'label': 'LABEL_1', 'score': 0.9663578867912292},
 {'label': 'LABEL_2', 'score': 0.003938842564821243},
 {'label': 'LABEL_3', 'score': 0.00875843595713377},
 {'label': 'LABEL_4', 'score': 0.004418900702148676},
 {'label': 'LABEL_5', 'score': 0.011632120236754417}]

```







