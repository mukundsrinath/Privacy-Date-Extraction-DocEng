# Date Extraction from Privacy Policies

## Usage Information

This repository contains the corpus and code for in the following paper:

Mukund Srinath, Lee Matheson, Pranav Narayanan Venkit, Gabriela Zanfir-Fortuna, Florian Schaub, C. Lee Giles, and Shomir Wilson. 2023. Privacy Now or Never: Large-Scale Extraction and Analysis of Dates in Privacy Policy Text. In Proceedings of the ACM Symposium on Document Engineering 2023 (DocEng '23). Association for Computing Machinery, New York, NY, USA, Article 24, 1–4.
[https://doi.org/10.1145/3573128.3609342]

The dataset, and code are (C) Copyright 2023, all rights reserved. For technical questions about this data, please contact Mukund Srinath (mukund@psu.edu). For licensing questions, please contact Dr. Shomir Wilson (shomir@psu.edu).

For research, teaching, and scholarship purposes, the corpus is available under a CC BY-NC-SA license. Please contact us for any requests regarding commercial use.

## Corpus Contents

The csv file contains text extracted from privacy policies that contains candidate date instances labeled into 'updated date', 'effective date' and 'other' dates. The labeling was conducted by one of the authors of the paper and verified by another author. 

If you make use of this dataset in your reserach please cite our paper:

```
@inproceedings{10.1145/3573128.3609342, 
  author = {Srinath, Mukund and Matheson, Lee and Venkit, Pranav Narayanan and Zanfir-Fortuna, Gabriela and Schaub, Florian and Giles, C. Lee and Wilson, Shomir}, 
  title = {Privacy Now or Never: Large-Scale Extraction and Analysis of Dates in Privacy Policy Text}, 
  year = {2023}, 
  isbn = {9798400700279}, 
  publisher = {Association for Computing Machinery}, 
  address = {New York, NY, USA}, 
  url = {https://doi.org/10.1145/3573128.3609342}, 
  doi = {10.1145/3573128.3609342}, 
  abstract = {The General Data Protection Regulation (GDPR) and other recent privacy laws require organizations to post their privacy policies, and place specific expectations on organisations' privacy practices. Privacy policies take the form of documents written in natural language, and one of the expectations placed upon them is that they remain up to date. To investigate legal compliance with this recency requirement at a large scale, we create a novel pipeline that includes crawling, regex-based extraction, candidate date classification and date object creation to extract updated and effective dates from privacy policies written in English. We then analyze patterns in policy dates using four web crawls and find that only about 40\% of privacy policies online contain a date, thereby making it difficult to assess their regulatory compliance. We also find that updates in privacy policies are temporally concentrated around passage of laws regulating digital privacy (such as the GDPR), and that more popular domains are more likely to have policy dates as well as more likely to update their policies regularly.}, 
  booktitle = {Proceedings of the ACM Symposium on Document Engineering 2023}, 
  articleno = {24}, 
  numpages = {4}, 
  keywords = {date extraction, privacy policy, crawling}, 
  location = {Limerick, Ireland}, 
  series = {DocEng '23} 
}
```

## Credits

This documentation was written by Mukund Srinath.

This work was partly supported by NSF awards #2105736, #2105734, and #2105745.
