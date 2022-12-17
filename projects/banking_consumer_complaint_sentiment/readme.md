
![](https://i.imgur.com/bj5ibCe.png)

### Project Keywords 📒

`Sentiment` `Multiclass` `Multi-Model` `Banking` `Classification` `NLP` `HuggingFace` `Transformers` 

<br>

### Project Background 📥

<b><span style='color:#F1A424'>Consumer Feedback</span></b>
- **<span style='color:#FFC300'>Consumer feedback</span>** is an important part of day to day financial business operations
- Companies offering products must be able to know what their consumers think of their products
    - Eg. positive & negative feedback & can be obtained from a number of sources (eg. twitter)
- In this case, we obtain data from a **[database](https://www.consumerfinance.gov/data-research/consumer-complaints/)**, which registers consumers feedback of financial products
- Customers have specific `issue` on a number of `topics` they want want the company to address 
- The form of consumer communication with the financial institution is via the `web` and **not in person**


**<span style='color:#FFC300'>Can your customers tell you something important?</span>** | **[Source](https://www.startquestion.com/blog/7-reasons-why-customer-feedback-is-important-to-your-business/)**

> If you run your own business, I know you do your best to please your customers <br>
> Satisfy their needs, and keep them loyal to your brand.  <br>
> But how can you be sure that your efforts bring desired results?  <br>
> If you do not try to find out what your clients think about your service <br>
> You will never be able to give them the best customer experience. <br>
> **<mark style="background-color:#FFC300;color:white;border-radius:5px;opacity:0.7">Their opinions</mark>** about their experience with your brand are helpful information<br> 
> That you can **<mark style="background-color:#FFC300;color:white;border-radius:5px;opacity:0.7">use to adjust your business to fit their needs more accurately</mark>**

- The source clearly outlines that customer feedback is quite critical for any business
- **<span style='color:#FFC300'>Customer feedback</span>** in our problem is related to a consumer having an `issue` with a particualar financial `product` or alike, the complaint is sent via the web

<br>

### Project Aim 🎯 

- In this project, we'll be utilising **<span style='color:#FFC300'>machine learning</span>**, to create a model(s) that will be able to **<span style='color:#FFC300'>classify the type of complaint</mark>** (as we did above) (by both `product` & `issue`)
- Such a model(s) can be useful for a company to quickly understand the **type of complaint** (What is the issue?) & which **product** it is related to, after whcih they can appoint a specific **financial expert** that will be able to solve the problem 
- Our approach will include **<span style='color:#FFC300'>separate models</span>**, that will be in charge of classifying data on different subsets of data
  - The **first model** (`M1`) will be in charge of classifying a  **<mark style="background-color:#FFC300;color:white;border-radius:5px;opacity:1.0">product</mark>** based on the customer's input complaint (`text`)
  - The **second model** (`M2`) will be in charge of classifying the particular **<mark style="background-color:#FFC300;color:white;border-radius:5px;opacity:1.0">issue</mark>** to which the complaint belongs to (`text`)

<br>

### Kaggle Notebook 📖

Kaggle offer a very neat `ipynb` render, if you'd like to read the notebook on Kaggle, links are provided below:

- **[Banking Consumer Complaints Analysis](https://www.kaggle.com/code/shtrausslearning/banking-consumer-complaints-analysis)**
- **[Hidden State Data - for LogisticRegression Model](https://www.kaggle.com/datasets/shtrausslearning/hiddenstatedata)**
