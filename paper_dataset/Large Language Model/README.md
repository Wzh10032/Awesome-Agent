# Awesome Large Language Model Research

> 自动生成的论文列表，更新于: 2025-06-03

## 目录
- [Critique of LLMs](#critique-of-llms)
- [European Language Models](#european-language-models)
- [Language-Specific Models](#language-specific-models)
- [Low-Resource Languages](#low-resource-languages)
- [Multilingual Language Models](#multilingual-language-models)
- [Psycholinguistics and LLMs](#psycholinguistics-and-llms)
- [Theoretical Analysis of LLMs](#theoretical-analysis-of-llms)
- [Vision-Language Models](#vision-language-models)

---

## Critique of LLMs

### [A Precis of Language Models are not Models of Language](http://arxiv.org/pdf/2205.07634v1)

**Authors**: Csaba Veres

**Published**: May 16, 2022 | **Updated**: May 16, 2022 | **Arxiv ID**: [2205.07634v1](http://arxiv.org/pdf/2205.07634v1)

**Abstract**:
> Natural Language Processing is one of the leading application areas in the
> current resurgence of Artificial Intelligence, spearheaded by Artificial Neural
> Networks. We show that despite their many successes at performing linguistic
> tasks, Large Neural Language Models are ill-suited as comprehensive models of
> natural language. The wider implication is that, in spite of the often
> overbearing optimism about AI, modern neural models do not represent a
> revolution in our understanding of cognition.

---

## European Language Models

### [A Survey of Large Language Models for European Languages](http://arxiv.org/pdf/2408.15040v2)

**Authors**: Wazir Ali, Sampo Pyysalo

**Published**: Aug 27, 2024 | **Updated**: Aug 28, 2024 | **Arxiv ID**: [2408.15040v2](http://arxiv.org/pdf/2408.15040v2)

**Abstract**:
> Large Language Models (LLMs) have gained significant attention due to their
> high performance on a wide range of natural language tasks since the release of
> ChatGPT. The LLMs learn to understand and generate language by training
> billions of model parameters on vast volumes of text data. Despite being a
> relatively new field, LLM research is rapidly advancing in various directions.
> In this paper, we present an overview of LLM families, including LLaMA, PaLM,
> GPT, and MoE, and the methods developed to create and enhance LLMs for official
> European Union (EU) languages. We provide a comprehensive summary of common
> monolingual and multilingual datasets used for pretraining large language
> models.

---

## Language-Specific Models

### [Cedille: A large autoregressive French language model](http://arxiv.org/pdf/2202.03371v1)

**Authors**: Martin Müller, Florian Laurent

**Published**: Feb 07, 2022 | **Updated**: Feb 07, 2022 | **Arxiv ID**: [2202.03371v1](http://arxiv.org/pdf/2202.03371v1)

**Abstract**:
> Scaling up the size and training of autoregressive language models has
> enabled novel ways of solving Natural Language Processing tasks using zero-shot
> and few-shot learning. While extreme-scale language models such as GPT-3 offer
> multilingual capabilities, zero-shot learning for languages other than English
> remain largely unexplored. Here, we introduce Cedille, a large open source
> auto-regressive language model, specifically trained for the French language.
> Our results show that Cedille outperforms existing French language models and
> is competitive with GPT-3 on a range of French zero-shot benchmarks.
> Furthermore, we provide an in-depth comparison of the toxicity exhibited by
> these models, showing that Cedille marks an improvement in language model
> safety thanks to dataset filtering.

---

## Low-Resource Languages

### [Goldfish: Monolingual Language Models for 350 Languages](http://arxiv.org/pdf/2408.10441v1)

**Authors**: Tyler A. Chang, Catherine Arnett, Zhuowen Tu, Benjamin K. Bergen

**Published**: Aug 19, 2024 | **Updated**: Aug 19, 2024 | **Arxiv ID**: [2408.10441v1](http://arxiv.org/pdf/2408.10441v1)

**Abstract**:
> For many low-resource languages, the only available language models are large
> multilingual models trained on many languages simultaneously. However, using
> FLORES perplexity as a metric, we find that these models perform worse than
> bigrams for many languages (e.g. 24% of languages in XGLM 4.5B; 43% in BLOOM
> 7.1B). To facilitate research that focuses on low-resource languages, we
> pre-train and release Goldfish, a suite of monolingual autoregressive
> Transformer language models up to 125M parameters for 350 languages. The
> Goldfish reach lower FLORES perplexities than BLOOM, XGLM, and MaLA-500 on 98
> of 204 FLORES languages, despite each Goldfish model being over 10x smaller.
> However, the Goldfish significantly underperform larger multilingual models on
> reasoning benchmarks, suggesting that for low-resource languages,
> multilinguality primarily improves general reasoning abilities rather than
> basic text generation. We release models trained on 5MB (350 languages), 10MB
> (288 languages), 100MB (166 languages), and 1GB (83 languages) of text data
> where available. The Goldfish models are available as baselines, fine-tuning
> sources, or augmentations to existing models in low-resource NLP research, and
> they are further useful for crosslinguistic studies requiring maximally
> comparable models across languages.

---

## Multilingual Language Models

### [Lost in Translation: Large Language Models in Non-English Content Analysis](http://arxiv.org/pdf/2306.07377v1)

**Authors**: Gabriel Nicholas, Aliya Bhatia

**Published**: Jun 12, 2023 | **Updated**: Jun 12, 2023 | **Arxiv ID**: [2306.07377v1](http://arxiv.org/pdf/2306.07377v1)

**Abstract**:
> In recent years, large language models (e.g., Open AI's GPT-4, Meta's LLaMa,
> Google's PaLM) have become the dominant approach for building AI systems to
> analyze and generate language online. However, the automated systems that
> increasingly mediate our interactions online -- such as chatbots, content
> moderation systems, and search engines -- are primarily designed for and work
> far more effectively in English than in the world's other 7,000 languages.
> Recently, researchers and technology companies have attempted to extend the
> capabilities of large language models into languages other than English by
> building what are called multilingual language models.
>   In this paper, we explain how these multilingual language models work and
> explore their capabilities and limits. Part I provides a simple technical
> explanation of how large language models work, why there is a gap in available
> data between English and other languages, and how multilingual language models
> attempt to bridge that gap. Part II accounts for the challenges of doing
> content analysis with large language models in general and multilingual
> language models in particular. Part III offers recommendations for companies,
> researchers, and policymakers to keep in mind when considering researching,
> developing and deploying large and multilingual language models.

---

### [How Good are Commercial Large Language Models on African Languages?](http://arxiv.org/pdf/2305.06530v1)

**Authors**: Jessica Ojo, Kelechi Ogueji

**Published**: May 11, 2023 | **Updated**: May 11, 2023 | **Arxiv ID**: [2305.06530v1](http://arxiv.org/pdf/2305.06530v1)

**Abstract**:
> Recent advancements in Natural Language Processing (NLP) has led to the
> proliferation of large pretrained language models. These models have been shown
> to yield good performance, using in-context learning, even on unseen tasks and
> languages. They have also been exposed as commercial APIs as a form of
> language-model-as-a-service, with great adoption. However, their performance on
> African languages is largely unknown. We present a preliminary analysis of
> commercial large language models on two tasks (machine translation and text
> classification) across eight African languages, spanning different language
> families and geographical areas. Our results suggest that commercial language
> models produce below-par performance on African languages. We also find that
> they perform better on text classification than machine translation. In
> general, our findings present a call-to-action to ensure African languages are
> well represented in commercial large language models, given their growing
> popularity.

---

## Psycholinguistics and LLMs

### [Beyond the limitations of any imaginable mechanism: large language models and psycholinguistics](http://arxiv.org/pdf/2303.00077v1)

**Authors**: Conor Houghton, Nina Kazanina, Priyanka Sukumaran

**Published**: Feb 28, 2023 | **Updated**: Feb 28, 2023 | **Arxiv ID**: [2303.00077v1](http://arxiv.org/pdf/2303.00077v1)

**Abstract**:
> Large language models are not detailed models of human linguistic processing.
> They are, however, extremely successful at their primary task: providing a
> model for language. For this reason and because there are no animal models for
> language, large language models are important in psycholinguistics: they are
> useful as a practical tool, as an illustrative comparative, and
> philosophically, as a basis for recasting the relationship between language and
> thought.

---

## Theoretical Analysis of LLMs

### [Modelling Language](http://arxiv.org/pdf/2404.09579v1)

**Authors**: Jumbly Grindrod

**Published**: Apr 15, 2024 | **Updated**: Apr 15, 2024 | **Arxiv ID**: [2404.09579v1](http://arxiv.org/pdf/2404.09579v1)

**Abstract**:
> This paper argues that large language models have a valuable scientific role
> to play in serving as scientific models of a language. Linguistic study should
> not only be concerned with the cognitive processes behind linguistic
> competence, but also with language understood as an external, social entity.
> Once this is recognized, the value of large language models as scientific
> models becomes clear. This paper defends this position against a number of
> arguments to the effect that language models provide no linguistic insight. It
> also draws upon recent work in philosophy of science to show how large language
> models could serve as scientific models.

---

## Vision-Language Models

### [Enhance Reasoning Ability of Visual-Language Models via Large Language Models](http://arxiv.org/pdf/2305.13267v1)

**Authors**: Yueting Yang, Xintong Zhang, Wenjuan Han

**Published**: May 22, 2023 | **Updated**: May 22, 2023 | **Arxiv ID**: [2305.13267v1](http://arxiv.org/pdf/2305.13267v1)

**Abstract**:
> Pre-trained visual language models (VLM) have shown excellent performance in
> image caption tasks. However, it sometimes shows insufficient reasoning
> ability. In contrast, large language models (LLMs) emerge with powerful
> reasoning capabilities. Therefore, we propose a method called TReE, which
> transfers the reasoning ability of a large language model to a visual language
> model in zero-shot scenarios. TReE contains three stages: observation,
> thinking, and re-thinking. Observation stage indicates that VLM obtains the
> overall information of the relative image. Thinking stage combines the image
> information and task description as the prompt of the LLM, inference with the
> rationals. Re-Thinking stage learns from rationale and then inference the final
> result through VLM.

---

