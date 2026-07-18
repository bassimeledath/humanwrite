<!--
Source: https://rosmine.ai/2026/05/18/fixing-llm-writing-with-distribution-fine-tuning/
Archived from: https://web.archive.org/web/20260522230330id_/https://rosmine.ai/2026/05/18/fixing-llm-writing-with-distribution-fine-tuning/
Snapshot retrieved: 2026-07-15
-->

<div class="wp-site-blocks">

<div class="wp-block-group has-global-padding is-layout-constrained wp-block-group-is-layout-constrained">

<div class="wp-block-columns alignwide is-not-stacked-on-mobile is-layout-flex wp-container-core-columns-is-layout-243bd399 wp-block-columns-is-layout-flex">

<div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow" style="flex-basis:33.33%">

</div>

<div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow" style="flex-basis:66.66%">

</div>

</div>

<div class="wp-block-columns alignwide is-layout-flex wp-container-core-columns-is-layout-243bd399 wp-block-columns-is-layout-flex">

<div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow" style="flex-basis:33.33%">

<div class="wp-block-template-part">

# <a href="https://rosmine.ai" target="_self" rel="home">Rosmine ML Blog</a>

<div class="wp-block-spacer" style="height:20px" aria-hidden="true">

</div>

- - <a href="https://rosmine.ai/25-2/" class="wp-block-pages-list__item__link wp-block-navigation-item__content">About</a>
  - <a href="https://rosmine.ai/ai-advising/" class="wp-block-pages-list__item__link wp-block-navigation-item__content">AI Advising</a>

<div class="wp-block-spacer" style="height:60px" aria-hidden="true">

</div>

</div>

</div>

<div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow" style="flex-basis:66.66%">

<div class="wp-block-group has-global-padding is-layout-constrained wp-block-group-is-layout-constrained">

<div class="wp-block-group is-vertical is-layout-flex wp-container-core-group-is-layout-1ccc30ea wp-block-group-is-layout-flex">

<div class="wp-block-group has-global-padding is-layout-constrained wp-block-group-is-layout-constrained">

# Fixing LLM writing with Distribution Fine Tuning

</div>

<div class="h1 { font-family: 'Roboto', sans-serif; } wp-block-template-part">

<div class="wp-block-group is-nowrap is-layout-flex wp-container-core-group-is-layout-bf432786 wp-block-group-is-layout-flex">

</div>

</div>

</div>

<div class="wp-block-spacer" style="height:5px" aria-hidden="true">

</div>

<div class="entry-content wp-block-post-content is-layout-flow wp-block-post-content-is-layout-flow">

  
Abstract/TLDR: LLMs are notoriously formulaic at writing, overusing certain tokens or phrases. I show that models trained with SFT fail to match the distribution of the training data by using Maximum Mean Discrepancy (MMD), Judge Model Quality (JMQ), and L2 Token Distribution.

To fix this, I created a new training algorithm, Distribution Fine Tuning (DFT), an LLM post training step that makes the distribution of model outputs better match the training distribution (improving MMD by 49% and JMQ by 63%). The model trained with DFT is much better at writing than an SFT baseline, improving creativity scores by +164%, as well as coherence (+28%), clarity (+16%), meaningful detail (+146%) and it does not have any overused “slop signs” like too many emdashes, or “it’s not X, it’s Y”.

A demo (14B param model) is available at <https://dft.rosmine.ai/>

Models trained with DFT have much more human writing style, a sample of 100 model outputs scored as 100% human written by [Pangram AI detector](https://www.pangram.com/)

<figure class="wp-block-image size-large">
<img src="https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/05/image-4.png?resize=1024%2C642&amp;ssl=1" class="wp-image-1165" data-recalc-dims="1" data-fetchpriority="high" decoding="async" data-attachment-id="1165" data-permalink="https://rosmine.ai/2026/05/18/fixing-llm-writing-with-distribution-fine-tuning/image-51/" data-orig-file="https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/05/image-4.png?fit=1124%2C705&amp;ssl=1" data-orig-size="1124,705" data-comments-opened="1" data-image-meta="{&quot;aperture&quot;:&quot;0&quot;,&quot;credit&quot;:&quot;&quot;,&quot;camera&quot;:&quot;&quot;,&quot;caption&quot;:&quot;&quot;,&quot;created_timestamp&quot;:&quot;0&quot;,&quot;copyright&quot;:&quot;&quot;,&quot;focal_length&quot;:&quot;0&quot;,&quot;iso&quot;:&quot;0&quot;,&quot;shutter_speed&quot;:&quot;0&quot;,&quot;title&quot;:&quot;&quot;,&quot;orientation&quot;:&quot;0&quot;}" data-image-title="image" data-image-description="" data-image-caption="" data-large-file="https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/05/image-4.png?fit=1024%2C642&amp;ssl=1" srcset="https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/05/image-4.png?resize=1024%2C642&amp;ssl=1 1024w, https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/05/image-4.png?resize=300%2C188&amp;ssl=1 300w, https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/05/image-4.png?resize=768%2C482&amp;ssl=1 768w, https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/05/image-4.png?w=1124&amp;ssl=1 1124w" sizes="(max-width: 1000px) 100vw, 1000px" width="1024" height="642" alt="Screenshot of an article titled &#39;Adam Smith: Moral Philosopher&#39; discussing the relationship between Adam Smith&#39;s moral sentiments and economics." />
</figure>

## Outline

1.  Key Metrics: Quantifying output quality
    - Define the key metrics for measuring text quality: MMD, JMQ, and Token L2 distance.
2.  The Problem: SFT is not all you need
    - Use these metrics to quantify how SFT fails to capture the training data distribution.
3.  Sample Model Outputs
    - Samples to see how DFT improves output.
4.  Results
    - Defines the “super baseline” and shows DFT improvement on key metrics.
5.  Next Steps for DFT
    - Collabs, Open weight model, Large model
6.  Unverified hype/speculation + Limitations
    - Potential Extensions of DFT, as well as drawbacks
7.  Anti-slop considerations + Future Vision
    - How I plan to use DFT to reduce slop
8.  Prior Work
    - Other papers that have quantified failures of SFT and proposed solutions
9.  Appendices
    - Deeper data dives, including DFT vs. SFT on 6 other metrics, dataset details, token frequency analysis, effect of data size, comparison with other models, fine grained judge model analysis, and quantification of slop signs in DFT output vs. human text.

## Key Metrics: Quantifying output quality

Slop. It’s not just annoying — it’s exhausting. You’re absolutely right to be annoyed by it, and in this blog I will delve into a solution.

You’ve probably noticed most models have their favorite words or phrases they overuse, like “—”, “it’s not X, it’s Y”, or “delve”. Before investigating the solution, I first address the metrics I use to measure output quality. Instead of measuring “quality” itself, which is not well defined, I measure similarity to human writing samples.

Metrics:

**N-gram Token distribution L2 distance**: This metric captures word choice similarity, and is useful for detecting overuse of certain words/phrases, like emdashes.

Given a set of writing samples, compute the N-gram token distribution as the number of times each N-gram appears over total number of N-grams, so dimension *i* measures the frequency of token *i*. To compare the two distributions, I use L2 (euclidean) distance<sup><a href="javascript:void(0)" role="button" aria-pressed="false" aria-describedby="mfn-content-00000000000007aa0000000000000000_753-1">1</a></sup><span id="mfn-content-00000000000007aa0000000000000000_753-1" class="modern-footnotes-footnote__note" role="tooltip" tabindex="0" mfn="1"> Note that metrics like KL or JS Divergence do not work well here because there are generally many tokens with that appear in reference but not output, or vice versa, and these have outsized contribution to the overall metric. </span> I primarily focus on L2 distance for 1-grams, see Appendix 3 for L2 on 2-grams and 3-grams.

**Maximum Mean Discrepancy (MMD, [Gretton](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf))**: This metric gets embedding for each text sample, and computes a distance between the embedding distributions. Since it’s using embeddings, it measures content similarity. For example, it captures if LLM outputs are overly generic and don’t go into detail, or if they overuse a certain concept (like [goblins](https://openai.com/index/where-the-goblins-came-from/)).

More specifically, given distributions P and Q, MMD compares the average distance from samples from the same distribution (first 2 terms in the formula) with the average distance between distributions. It will be 0 if and only if the two distributions are the same. To compute the distances the formula uses an embedding model ([Llama-embed-nemotron-8B](https://huggingface.co/nvidia/llama-embed-nemotron-8b), [Babakhin](https://arxiv.org/abs/2511.07025)) and a Gaussian RBF kernel k.

<div class="wp-block-math">

$$\begin{matrix}
{{MMD}^{2}(P,Q)} & {= {\mathbb{E}}_{x,x^{\prime} \sim P}\lbrack k(x,x^{\prime})\rbrack + {\mathbb{E}}_{y,y^{\prime} \sim Q}\lbrack k(y,y^{\prime})\rbrack} \\
 & {\quad - 2\,{\mathbb{E}}_{x \sim P,\; y \sim Q}\lbrack k(x,y)\rbrack} \\
\end{matrix}$$

</div>

I use MMD instead of other distances using embedding metrics since it was designed to test whether two sets of samples come from the same distribution, which aligns with the primary goal of DFT.

**Judge Model Quality (JMQ)**: This metric gives a judge model<sup><a href="javascript:void(0)" role="button" aria-pressed="false" aria-describedby="mfn-content-00000000000007aa0000000000000000_753-2">2</a></sup><span id="mfn-content-00000000000007aa0000000000000000_753-2" class="modern-footnotes-footnote__note" role="tooltip" tabindex="0" mfn="2"> GPT5.4-mini, with prompts in randomized order, to prevent positional bias </span>, a prompt and completions from human vs. model output. Judge Model Quality score (JMQ) is defined as 2 times the win rate for model outputs. (Since the goal is to match human text, the optimal score here is a 50% win rate. I multiply by 2 so that the range is 0-1.0). For the main body of this post, I focus on overall quality for JMQ. For fine grained analysis of creativity, coherence, depth, etc. as well as comparison of different judge models, see Appendix 9.

We now use these metrics to quantify how models trained with SFT fail to match the training data distribution.

## The Problem: SFT is not all you need

When training a frontier model, there are many post training steps, such as RLHF. A reasonable hypothesis for why LLM text has so many slop signs is from reward hacking during these steps, and that if you removed these steps then it would be easy to create higher quality writing.

However, there are still clear differences between SFT model outputs and human samples. To quantify this, I started from models in the Qwen3 family ([Yang](https://arxiv.org/abs/2505.09388))<sup><a href="javascript:void(0)" role="button" aria-pressed="false" aria-describedby="mfn-content-00000000000007aa0000000000000000_753-3">3</a></sup><span id="mfn-content-00000000000007aa0000000000000000_753-3" class="modern-footnotes-footnote__note" role="tooltip" tabindex="0" mfn="3">I initially tried starting from Qwen3 Base models, however these had bad MMD and JMQ scores due to poor instruction following. The instruct tuned Qwen3 models (e.g. [14B](https://huggingface.co/Qwen/Qwen3-14B)) performed much better.</span> and trained on a subset of fineweb (see Appendix 2 for data details). I then graphed how MMD, JMQ, and L2 Token Distance depend on sampler settings between the model output and human samples on a held out test set. These graphs show the dependence on temperature, see appendix for top_p and top_k.

Models are trained on a subset of 185K samples from fineweb (See Appendix 2 for details). Results are calculated on a set of 2000 held-out samples (JMQ only uses first 400 of these).

Note that in the graphs below, the dotted line for DFT is from a fixed sampler setting. I extended it as a line instead of a point to more dramatically emphasize how a single DFT model beats SFT at any hyperparameter setting.

<figure class="wp-block-image size-large">
<img src="https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/05/image-1.png?resize=1024%2C315&amp;ssl=1" class="wp-image-1045" data-recalc-dims="1" decoding="async" data-attachment-id="1045" data-permalink="https://rosmine.ai/2026/05/18/fixing-llm-writing-with-distribution-fine-tuning/image-47/" data-orig-file="https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/05/image-1.png?fit=1656%2C510&amp;ssl=1" data-orig-size="1656,510" data-comments-opened="1" data-image-meta="{&quot;aperture&quot;:&quot;0&quot;,&quot;credit&quot;:&quot;&quot;,&quot;camera&quot;:&quot;&quot;,&quot;caption&quot;:&quot;&quot;,&quot;created_timestamp&quot;:&quot;0&quot;,&quot;copyright&quot;:&quot;&quot;,&quot;focal_length&quot;:&quot;0&quot;,&quot;iso&quot;:&quot;0&quot;,&quot;shutter_speed&quot;:&quot;0&quot;,&quot;title&quot;:&quot;&quot;,&quot;orientation&quot;:&quot;0&quot;}" data-image-title="image" data-image-description="" data-image-caption="" data-large-file="https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/05/image-1.png?fit=1024%2C315&amp;ssl=1" srcset="https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/05/image-1.png?resize=1024%2C315&amp;ssl=1 1024w, https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/05/image-1.png?resize=300%2C92&amp;ssl=1 300w, https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/05/image-1.png?resize=768%2C237&amp;ssl=1 768w, https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/05/image-1.png?resize=1536%2C473&amp;ssl=1 1536w, https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/05/image-1.png?resize=1200%2C370&amp;ssl=1 1200w, https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/05/image-1.png?w=1656&amp;ssl=1 1656w" sizes="(max-width: 1000px) 100vw, 1000px" width="1024" height="315" alt="Line graphs displaying the results of a temperature sweep experiment across three metrics: L2_1gram, Judge Model Quality, and MMD. Each graph shows the relationship between temperature and the respective metric, with data points representing different models (14B, 8B, DFT) indicated by different colors." />
</figure>

These graphs suggest SFT is failing to fully capture the training data distribution. If it was perfectly matching the training data distribution, then MMD would be 0 and JMQ would be 1.0.

There are many proposed explanations for why SFT fails to match the training distribution, including exposure bias ([Ranzato](https://arxiv.org/abs/1511.06732), [Bengio](https://arxiv.org/abs/1506.03099)), unreliable tail probabilities ([Holtzman](https://arxiv.org/abs/1904.09751)), likelihood objective ([Welleck](https://arxiv.org/abs/1908.04319)), miscalibration ([Braverman](https://arxiv.org/abs/1906.05664)), local typicality ([Meister](https://arxiv.org/abs/2202.00666)), etc. However, this simplest explanation is that at a high level, SFT focuses on training individual samples and in doing so misses out on distribution level information. DFT trains at this higher level, optimizing the distribution of outputs so that it better matches the training data.

## Sample Model Outputs

I compare outputs for the same prompt from the SFT model with best judge preference (14B, T=0.7), the best token distribution (14B, T=1.0), and DFT.

I only include a short section of each output, to highlight differences without requiring reading the full text.

Prompt (including outline of response):

<figure class="wp-block-image size-full">
<img src="https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/03/image-13.png?resize=1000%2C170&amp;ssl=1" class="wp-image-906" data-recalc-dims="1" decoding="async" data-attachment-id="906" data-permalink="https://rosmine.ai/2026/05/18/fixing-llm-writing-with-distribution-fine-tuning/image-35/" data-orig-file="https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/03/image-13.png?fit=1000%2C170&amp;ssl=1" data-orig-size="1000,170" data-comments-opened="1" data-image-meta="{&quot;aperture&quot;:&quot;0&quot;,&quot;credit&quot;:&quot;&quot;,&quot;camera&quot;:&quot;&quot;,&quot;caption&quot;:&quot;&quot;,&quot;created_timestamp&quot;:&quot;0&quot;,&quot;copyright&quot;:&quot;&quot;,&quot;focal_length&quot;:&quot;0&quot;,&quot;iso&quot;:&quot;0&quot;,&quot;shutter_speed&quot;:&quot;0&quot;,&quot;title&quot;:&quot;&quot;,&quot;orientation&quot;:&quot;0&quot;}" data-image-title="image" data-image-description="" data-image-caption="" data-large-file="https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/03/image-13.png?fit=1000%2C170&amp;ssl=1" srcset="https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/03/image-13.png?w=1000&amp;ssl=1 1000w, https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/03/image-13.png?resize=300%2C51&amp;ssl=1 300w, https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/03/image-13.png?resize=768%2C131&amp;ssl=1 768w" sizes="(max-width: 1000px) 100vw, 1000px" width="1000" height="170" alt="Text excerpt discussing Adam Smith&#39;s philosophical and economic systems, focusing on themes like moral sympathy versus self-interest and critiques of his theories." />
</figure>

SFT T=0.7

<figure class="wp-block-image size-full">
<img src="https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/03/image-14.png?resize=1005%2C202&amp;ssl=1" class="wp-image-908" data-recalc-dims="1" loading="lazy" decoding="async" data-attachment-id="908" data-permalink="https://rosmine.ai/2026/05/18/fixing-llm-writing-with-distribution-fine-tuning/image-36/" data-orig-file="https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/03/image-14.png?fit=1005%2C202&amp;ssl=1" data-orig-size="1005,202" data-comments-opened="1" data-image-meta="{&quot;aperture&quot;:&quot;0&quot;,&quot;credit&quot;:&quot;&quot;,&quot;camera&quot;:&quot;&quot;,&quot;caption&quot;:&quot;&quot;,&quot;created_timestamp&quot;:&quot;0&quot;,&quot;copyright&quot;:&quot;&quot;,&quot;focal_length&quot;:&quot;0&quot;,&quot;iso&quot;:&quot;0&quot;,&quot;shutter_speed&quot;:&quot;0&quot;,&quot;title&quot;:&quot;&quot;,&quot;orientation&quot;:&quot;0&quot;}" data-image-title="image" data-image-description="" data-image-caption="" data-large-file="https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/03/image-14.png?fit=1005%2C202&amp;ssl=1" srcset="https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/03/image-14.png?w=1005&amp;ssl=1 1005w, https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/03/image-14.png?resize=300%2C60&amp;ssl=1 300w, https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/03/image-14.png?resize=768%2C154&amp;ssl=1 768w" sizes="auto, (max-width: 1000px) 100vw, 1000px" width="1005" height="202" alt="Text excerpt discussing Adam Smith&#39;s moral philosophy and economic theory, highlighting Julian Hoppit&#39;s critique of Smith&#39;s concept of self-interest." />
</figure>

T=1.0

<figure class="wp-block-image size-full">
<img src="https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/03/image-15.png?resize=1000%2C247&amp;ssl=1" class="wp-image-909" data-recalc-dims="1" loading="lazy" decoding="async" data-attachment-id="909" data-permalink="https://rosmine.ai/2026/05/18/fixing-llm-writing-with-distribution-fine-tuning/image-37/" data-orig-file="https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/03/image-15.png?fit=1000%2C247&amp;ssl=1" data-orig-size="1000,247" data-comments-opened="1" data-image-meta="{&quot;aperture&quot;:&quot;0&quot;,&quot;credit&quot;:&quot;&quot;,&quot;camera&quot;:&quot;&quot;,&quot;caption&quot;:&quot;&quot;,&quot;created_timestamp&quot;:&quot;0&quot;,&quot;copyright&quot;:&quot;&quot;,&quot;focal_length&quot;:&quot;0&quot;,&quot;iso&quot;:&quot;0&quot;,&quot;shutter_speed&quot;:&quot;0&quot;,&quot;title&quot;:&quot;&quot;,&quot;orientation&quot;:&quot;0&quot;}" data-image-title="image" data-image-description="" data-image-caption="" data-large-file="https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/03/image-15.png?fit=1000%2C247&amp;ssl=1" srcset="https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/03/image-15.png?w=1000&amp;ssl=1 1000w, https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/03/image-15.png?resize=300%2C74&amp;ssl=1 300w, https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/03/image-15.png?resize=768%2C190&amp;ssl=1 768w" sizes="auto, (max-width: 1000px) 100vw, 1000px" width="1000" height="247" alt="Text excerpt discussing Adam Smith&#39;s concept of the &#39;invisible hand&#39; in the context of philosophy and synergy." />
</figure>

DFT

<figure class="wp-block-image size-full">
<img src="https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/03/image-16.png?resize=997%2C171&amp;ssl=1" class="wp-image-910" data-recalc-dims="1" loading="lazy" decoding="async" data-attachment-id="910" data-permalink="https://rosmine.ai/2026/05/18/fixing-llm-writing-with-distribution-fine-tuning/image-38/" data-orig-file="https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/03/image-16.png?fit=997%2C171&amp;ssl=1" data-orig-size="997,171" data-comments-opened="1" data-image-meta="{&quot;aperture&quot;:&quot;0&quot;,&quot;credit&quot;:&quot;&quot;,&quot;camera&quot;:&quot;&quot;,&quot;caption&quot;:&quot;&quot;,&quot;created_timestamp&quot;:&quot;0&quot;,&quot;copyright&quot;:&quot;&quot;,&quot;focal_length&quot;:&quot;0&quot;,&quot;iso&quot;:&quot;0&quot;,&quot;shutter_speed&quot;:&quot;0&quot;,&quot;title&quot;:&quot;&quot;,&quot;orientation&quot;:&quot;0&quot;}" data-image-title="image" data-image-description="" data-image-caption="" data-large-file="https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/03/image-16.png?fit=997%2C171&amp;ssl=1" srcset="https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/03/image-16.png?w=997&amp;ssl=1 997w, https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/03/image-16.png?resize=300%2C51&amp;ssl=1 300w, https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/03/image-16.png?resize=768%2C132&amp;ssl=1 768w" sizes="auto, (max-width: 997px) 100vw, 997px" width="997" height="171" alt="A close-up of a text excerpt discussing the concept of the &#39;invisible hand&#39; in relation to Adam Smith&#39;s economic theories and critiques surrounding materialism and theology." />
</figure>

Some differences to notice:

SFT T=0.7  
– repetitive structure. All the sentences except for 2 start with “Smith’s”  
– the text is generic, without deeper details

SFT T=1.0  
– Big transitions, e.g. from “systematic way of thinking about synergy” to “The natural world or organisms” to economic theory.  
– Non english characters randomly added, “拭엥”<sup><a href="javascript:void(0)" role="button" aria-pressed="false" aria-describedby="mfn-content-00000000000007aa0000000000000000_753-4">4</a></sup><span id="mfn-content-00000000000007aa0000000000000000_753-4" class="modern-footnotes-footnote__note" role="tooltip" tabindex="0" mfn="4">The first token is a Chinese character meaning “to wipe”, the second is Korean for “huh?” or “what?”. Perhaps the model was also surprised that it switched to Chinese.</span>

Not all SFT writing samples are this bad, these examples were chosen because they demonstrated repetitiveness at low temperatures and incoherence at higher temps for the same prompt. At intermediate temperatures these same problems happen, just at different frequencies, for example at temp=0.8, 44% of outputs have similar amounts of repetitiveness, and at temp=0.9, over 9% of outputs have non-English characters. See Appendix 1 for more details.

## Results

The optimal value for MMD vs. JMQ vs. L2 metrics in the graphs for SFT above use different sampling parameters. L2 metrics are optimized by Temp=1, but MMD and JMQ are optimized at lower temps. For a strong baseline, I ran hyperparameter search over several learning rates, lora vs. full fine tuning, and different sampler settings, then used the best metric value over all hparam configurations as the value for the a “super baseline”. This means the super-baseline has metric scores better than is possible for any single hyperparameter setting.

(Also, note that there is randomness in the evaluation metrics, and the max over noisy estimates gives an overoptimistic estimate of the true value, which makes the super baseline even more difficult to beat)

Important note: I only do this “max over all hparam configurations” for the baseline. For DFT results I use a single model with a fixed set of hparam values.

Distribution Fine Tuning (DFT) outperforms the SFT superbaseline. A 4B model trained with DFT beats a 14B superbaseline at MMD and an 8B superbaseline at JMQ.

| Model             | MMD↓      | JMQ↑     | Token L2↓  |
|:------------------|:----------|:---------|:-----------|
| 4B SuperBaseline  | 0.047     | 0.27     | **0.0040** |
| 4B DFT            | **0.025** | **0.4**  | 0.0042     |
| 8B SuperBaseline  | 0.041     | 0.37     | 0.0040     |
| 8B DFT            | **0.023** | **0.56** | **0.0031** |
| 14B SuperBaseline | 0.037     | 0.49     | 0.0039     |
| 14B DFT           | **0.018** | **0.80** | **0.0036** |

These results do not require extreme compute; all training was done on [my local 6x 6000 Ada server](https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/).

## Next steps for DFT

DFT is a proprietary training algorithm, however, I’m currently offering a beta for a model training service where I will train your model for you using DFT. This will start with just 1-2 collaborations in the beta, and extend it after those complete. If you are interested, please contact hello@rosmine.ai

I also want to train both a small open weights model, as well as a large model with DFT. For this demo, I focused on web content like blogs and news articles. If you are interested in other use cases, (e.g. creative writing, e-mails, movie scripts, etc.) please let me know so I can focus my efforts on what people want. Feel free to reach out by email, or [tag/DM me on X](https://x.com/rosmine)

## Unverified hype/speculation

The DFT algorithm is not specific to writing, at its core it is just a better way to make model outputs better match the training data distribution. I hope to apply it to other use cases beyond writing. For example, it could be a replacement for SFT that gives more accurate outputs, or it could be used for audio to make better AI generated music. However, I’ve focused all my compute making sure the writing models are good, so I don’t have any experiment results yet for other use cases.

## Limitations

Most model training was done on my home GPU server, the exceptions were using cloud H100’s for the superbaseline full fine tuning, which is too slow on my server. All DFT training used a sequence of LoRA ([Hu](https://arxiv.org/abs/2106.09685), [Lialin](https://arxiv.org/html/2307.05695v4)). As seen in the graphs above, the larger models have better MMD/token distribution/JMQ scores, so there will be some improvements just from scaling up the size of the baseline. However, existing models all have very clear LLM generated style, so I believe that even at the largest model sizes, there will still be room for DFT to improve outputs.

Note that the demo was trained on a subset of fineweb, a collection of web documents. This should make the demo models good at blogs/news articles, but it is unlikely to do as well at creative writing, since it has not been trained for that use case yet.

## Anti-slop considerations + Future Vision

Since DFT outputs are much more humanlike than other model outputs, this technology has potential for abuse for spammers/misinformation/social media slop accounts, so I want to address mitigations I’ve implemented, and my thoughts on the problem of slop.

LLMs are not the cause of slop. Lack of effort/care is. If you spend days researching and planning a blog post, and put all the information into a detailed, well-structured outline, and ask ChatGPT to generate the post based on the outline, then the output will be interesting to read, even if the text has a lot of em-dashes.

To encourage more thought, I’ve added formatted the input so you need to add a prompt, outline of the response (including any stats/quotes), writing style, and use case. These extra inputs are not required for the DFT algorithm to work, they’re just there to force people to think more about what they’re writing.

To prevent blindly copy-pasting from the demo without carefully reading the output, I’ve injected random fruit or cute animals into the output (e.g. model output is “DFT is awesome”, but when you copy and paste, it could be “otter is awesome”).

Also, there is no public API, to prevent automated use.

I want this to be a tool that allows people to write better by letting them focus on the content of what they write, without needing to waste time typing out every individual character. It should let people who have done cool projects share their work in ways that other people will understand and appreciate, without being bottlenecked by their writing abilities.

Future versions of this product will be an “IDE for writing” that give you more fine grained control, editing, and automated checks (“unit tests, but for writing”) to make sure your writing is good. It will extend to all types of writing, such as scripts, speeches, e-mails, etc.

Right now, LLMs for writing are like GPT4 for coding. People think that LLMs help them write, but it’s actually just adding bugs faster. I’m making the next generation of LLMs for writing, where “written with LLM” guarantees clear, engaging text that you won’t be annoyed to read.

## Prior Work

Other work has measured the difference between training data and model outputs ([Pillutla](https://arxiv.org/abs/2102.01454)<sup><a href="javascript:void(0)" role="button" aria-pressed="false" aria-describedby="mfn-content-00000000000007aa0000000000000000_753-5">5</a></sup><span id="mfn-content-00000000000007aa0000000000000000_753-5" class="modern-footnotes-footnote__note" role="tooltip" tabindex="0" mfn="5">As seen in Appendix 3, MAUVE saturates, scoring .997+ for 4B baseline</span>, [Alihosseini](https://arxiv.org/abs/1904.03971)<sup><a href="javascript:void(0)" role="button" aria-pressed="false" aria-describedby="mfn-content-00000000000007aa0000000000000000_753-6">6</a></sup><span id="mfn-content-00000000000007aa0000000000000000_753-6" class="modern-footnotes-footnote__note" role="tooltip" tabindex="0" mfn="6">This paper suggests FID using BERT. I test FID in Appendix 3, but don’t use it for a key metric since Frechet metrics assume the underlying distribution is Gaussian, which is not true for language</span>)), but the combination of MMD, L2 Token distribution, and JMQ gives the fullest picture of both content and style.

There is existing work using imitation learning make the model better match the training data distribution, specifically TextGail ([Qingyang](https://arxiv.org/abs/2004.13796), [Ho](https://arxiv.org/abs/1606.03476)), and IQLearn ([Wulfmeier](https://arxiv.org/abs/2409.01369)). However these performed poorly in this case. Textgail outperformed SFT when the output was restricted to only 64 tokens, but failed when scaled up to length 1024 due to training instabilities, despite many attempts and modifications. For fixed sampling parameters, IQLearn could outperform SFT for certain metrics (e.g. at temp 0.9 for a 4B model, IQLearn improves JMQ from .09 to .25, and improves rouge .491 to 498, consistent with ([Wulfmeier](https://arxiv.org/abs/2409.01369))), there was no single sampler setting for IQLearn that could beat the super baseline.

------------------------------------------------------------------------

If you want to cite this, please use:

> @misc{rosmine_DFT,  
> author = {Rosmine},  
> title = {Fixing LLM writing with Distribution Fine Tuning},  
> year = {2026},  
> month = May,  
> howpublished = {\url{<https://rosmine.ai/?p=753>},  
> }

## Appendix 1: Quantifying repetitiveness and non-English tokens

The table below continues the analysis from Section 3, quantifying the number of samples with non-English characters.

<figure class="wp-block-table">
<table class="has-fixed-layout">
<tbody>
<tr class="odd">
<td>Model</td>
<td>% texts with non-English characters</td>
</tr>
<tr class="even">
<td>Human training data</td>
<td>0.1%</td>
</tr>
<tr class="odd">
<td>SFT T=0.7</td>
<td>1%</td>
</tr>
<tr class="even">
<td>SFT T=0.8</td>
<td>3.2%</td>
</tr>
<tr class="odd">
<td>SFT T=0.9</td>
<td>9.1%</td>
</tr>
<tr class="even">
<td>SFT T=1.0</td>
<td>36.45%</td>
</tr>
<tr class="odd">
<td>DFT</td>
<td>8.1%</td>
</tr>
</tbody>
</table>
</figure>

To quantify repetitiveness, I measured what percentage of texts have at least 3 sentences in a row where each sentence starts with the same word<sup><a href="javascript:void(0)" role="button" aria-pressed="false" aria-describedby="mfn-content-00000000000007aa0000000000000000_753-7">7</a></sup><span id="mfn-content-00000000000007aa0000000000000000_753-7" class="modern-footnotes-footnote__note" role="tooltip" tabindex="0" mfn="7">It seemed suspicious that 17.4% of human generated text has 3 sentences with the same start word in a row, that seems too high. Inspecting the data, all the examples look valid, so it’s unlikely to be a bug. Mostly this is from lists, e.g. a list of questions that start with “How”, or 3 nouns in a row (“The Lamu Archipelago…”, “The islands lie between…”, “The largest island…”) </span>

<figure class="wp-block-table">
<table class="has-fixed-layout">
<tbody>
<tr class="odd">
<td>Model</td>
<td>% texts where 3 consecutive sentences start with the same word</td>
</tr>
<tr class="even">
<td>Human</td>
<td>17.4%</td>
</tr>
<tr class="odd">
<td>SFT T=0.7</td>
<td>53.3%</td>
</tr>
<tr class="even">
<td>SFT T=0.8</td>
<td>43.8%</td>
</tr>
<tr class="odd">
<td>SFT T=0.9</td>
<td>29.9%</td>
</tr>
<tr class="even">
<td>SFT T=1.0</td>
<td>16.3%</td>
</tr>
<tr class="odd">
<td>DFT</td>
<td>18.6%</td>
</tr>
</tbody>
</table>
</figure>

You can see that at all temperatures, a large portion of samples have either repetitive outputs, or random non-English characters.

## Appendix 2: Data details

I used a subset of fineweb ([Penedo](https://arxiv.org/abs/2406.17557)), a collection of high quality web documents. I found that many of the fineweb samples included parts that were not suitable for quality writing output, for example, contact information at the end of a webpage, captions for images/figures that were not included in the writing sample, etc. I used Qwen3-32B to parse the documents line by line, and remove any line that was not relevant to the main writing sample.

To create input/output pairs from these cleaned writing samples, I used Qwen3-32B to generate a prompt that would result in the output. I used several different variants of the prompt to ensure diveresity (e.g. asking for longer/shorter/more detailed prompts). Similarly I used it to extract the use case and style. For the outlines I used GPT-5-mini to write the outline of the response, including any facts/quotes that appeared. In the training/test mix, 25% of the samples had an empty outline, all other samples had outlines.

I also added flags for target length in tokens, and if emdashes are allowed. The final prompt specifies length, whether emdashes are allowed or not, the use case, the style, the prompt, and an outline of the response.

## Appendix 3: More Metrics

To show that DFT is not just overfitting to certain metrics, I use 7 more metrics to analyze text quality.

Metric Explanations:

- JSD on token 1-grams: Compute the token distribution of model and human outputs, then compute Jensen Shannon Divergence. This has very large contributions from tokens that appear in one output and not the other.
- FID ([Heusel](https://arxiv.org/abs/1706.08500)): Frechet Inception Distance, using a SOTA embedding model (  
  nvidia/llama-embed-nemotron-8b) to compute the embeddings. One drawback of this metric is that FID assumes the distributions are Gaussian, and text outputs generally are not Gaussian.
- L2 distance on token 2-grams, 3-grams: self explanatory
- Chrf ([Popović](https://aclanthology.org/W15-3049/)): Computes the F-score of character level n-grams between model output and reference
- BLEU vs human reference ([Papineni](https://dl.acm.org/doi/10.3115/1073083.1073135)). At a high level, BLEU computes clipped n-gram precision for n=1..4, then computes the geometric mean of these precisions, with a brevity benalty. BLEU is most relevant when there is a gold standard response, for example in translation. Here it is not quite as relevant, because prompts can be open ended (e.g. “write an essay about LLMs” could have many good completions that have very little overlap with each other).
- MAUVE ([Pillutla](https://arxiv.org/abs/2102.01454)): Embeds each texts from model and reference into feature space, quantize by clustering, compute divergence at different False positive/False Negative weightings, and return area under the curve. I found that MAUVE saturated too quickly, even 4B baseline could get .997+. The [MAUVE github](https://github.com/krishnap25/mauve) suggests that when MAUVE score saturates like this, you can increase the mauve_scaling_parameter. However, that also increases the between run variance, and as you can see by comparing 4B to 8B SuperBaseline, the variance is already high enough to make it difficult to compare models accurately (8B should be better than 4B, but it is not).

<figure class="wp-block-table">
<table class="has-fixed-layout">
<tbody>
<tr class="odd">
<td>Metric</td>
<td>4B SFT SuperBaseline</td>
<td>4B DFT</td>
</tr>
<tr class="even">
<td>JSD on token 1-gram ↓</td>
<td>0.025</td>
<td>0.024</td>
</tr>
<tr class="odd">
<td>FID ↓</td>
<td>235</td>
<td>182</td>
</tr>
<tr class="even">
<td>L2 distance on token 2-grams ↓</td>
<td>0.0018</td>
<td>0.0019</td>
</tr>
<tr class="odd">
<td>L2 distance on token 3-grams ↓</td>
<td>0.00141</td>
<td>0.00140</td>
</tr>
<tr class="even">
<td>ChrF ↑</td>
<td>45.4</td>
<td>46.2</td>
</tr>
<tr class="odd">
<td>BLEU vs. human reference ↑</td>
<td>0.057</td>
<td>0.042</td>
</tr>
<tr class="even">
<td>MAUVE ↑</td>
<td>.997</td>
<td>.996</td>
</tr>
</tbody>
</table>
</figure>

8B

<figure class="wp-block-table">
<table class="has-fixed-layout">
<tbody>
<tr class="odd">
<td>Metric</td>
<td>8B SFT SuperBaseline</td>
<td>8B DFT</td>
</tr>
<tr class="even">
<td>JSD on token 1-gram ↓</td>
<td>0.026</td>
<td>0.023</td>
</tr>
<tr class="odd">
<td>FID ↓</td>
<td>223</td>
<td>176</td>
</tr>
<tr class="even">
<td>L2 distance on token 2-grams ↓</td>
<td>0.0019</td>
<td>0.0016</td>
</tr>
<tr class="odd">
<td>L2 distance on token 3-grams ↓</td>
<td>0.0014</td>
<td>0.0014</td>
</tr>
<tr class="even">
<td>ChrF ↑</td>
<td>45.7</td>
<td>46.2</td>
</tr>
<tr class="odd">
<td>BLEU vs. human reference ↑</td>
<td>0.058*</td>
<td>0.045</td>
</tr>
<tr class="even">
<td>MAUVE ↑</td>
<td>0.996</td>
<td>0.997</td>
</tr>
</tbody>
</table>
</figure>

14B

<figure class="wp-block-table">
<table class="has-fixed-layout">
<tbody>
<tr class="odd">
<td>Metric</td>
<td>14B SFT SuperBaseline</td>
<td>14B DFT</td>
</tr>
<tr class="even">
<td>JSD on token 1-gram ↓</td>
<td>.025</td>
<td>.023</td>
</tr>
<tr class="odd">
<td>FID ↓</td>
<td>201</td>
<td>164</td>
</tr>
<tr class="even">
<td>L2 distance on token 2-grams ↓</td>
<td>0.0017</td>
<td>.0018</td>
</tr>
<tr class="odd">
<td>L2 distance on token 3-grams ↓</td>
<td>0.0015</td>
<td>0.0014</td>
</tr>
<tr class="even">
<td>ChrF ↑</td>
<td>46.1</td>
<td>46.7</td>
</tr>
<tr class="odd">
<td>BLEU vs. human reference ↑</td>
<td>0.062</td>
<td>0.051</td>
</tr>
<tr class="even">
<td>MAUVE ↑</td>
<td>0.997</td>
<td>0.997</td>
</tr>
</tbody>
</table>
</figure>

#### Why is BLEU worse?

The formula for BLEU is quite complex, but main idea is that it looks at 1,2,3, and 4 grams in the model output and checks how many of those appear in the reference. Analyzing the outputs of the SFT Superbaseline vs. DFT, I noticed that this value from the superbaseline comes from sampling parameters temp = .8, top_k=2. This is quite a low value for top_k, and low top_k like this typically lead to bland/repetitive outputs. For a closer analysis, I saw the main difference in the BLEU score was being driven by fewer 3 and 4 grams appearing in the reference. I made a table of the top 3 and 4 grams in SFT output vs. DFT output with counts:

<figure class="wp-block-table">
<table class="has-fixed-layout">
<tbody>
<tr class="odd">
<td>3 Gram</td>
<td>SFT output count</td>
<td>DFT output count</td>
</tr>
<tr class="even">
<td>be used to</td>
<td>621</td>
<td>55</td>
</tr>
<tr class="odd">
<td>It is a</td>
<td>520</td>
<td>42</td>
</tr>
<tr class="even">
<td>can be used</td>
<td>509</td>
<td>71</td>
</tr>
<tr class="odd">
<td>the number of</td>
<td>501</td>
<td>134</td>
</tr>
<tr class="even">
<td>one of the</td>
<td>430</td>
<td>408</td>
</tr>
</tbody>
</table>
</figure>

In this case the BLEU score was higher because it overuses common grammatical patterns. That’s exactly what I want to avoid with DFT, so I concluded that BLEU vs. human reference is not a good metric to use.

## Appendix 4: Token Analysis

When changing the sampling parameters, I saw that this increases the L2 distance between the output vs. reference token distribution. What tokens are the main contributors?

I took the top 1000 tokens, and ordered them by which had the largest relative difference compared to their frequency in human data. This is for a 14B model trained with SFT, using T=0.8

<figure class="wp-block-table">
<table class="has-fixed-layout">
<tbody>
<tr class="odd">
<td>Token</td>
<td>Relative Frequency compared to human output</td>
</tr>
<tr class="even">
<td>” the”</td>
<td>+19%</td>
</tr>
<tr class="odd">
<td>“.”</td>
<td>+19%</td>
</tr>
<tr class="even">
<td>” is”</td>
<td>+44%</td>
</tr>
<tr class="odd">
<td>” The”</td>
<td>+90%</td>
</tr>
<tr class="even">
<td>” a”</td>
<td>+15%</td>
</tr>
<tr class="odd">
<td>” to”</td>
<td>+11%</td>
</tr>
<tr class="even">
<td>” that”</td>
<td>+25%</td>
</tr>
<tr class="odd">
<td>” are”</td>
<td>+31%</td>
</tr>
<tr class="even">
<td>” was”</td>
<td>+49%</td>
</tr>
<tr class="odd">
<td>” of”</td>
<td>+5%</td>
</tr>
</tbody>
</table>
</figure>

These tokens are already very common. Part of the reason they contribute the most to the L2 distance is because they are so common (e.g. if the frequency is 1% more common on these tokens, it contributes much more to the metric than being 1% more common for a rare token). For 14B SFT model with temp=0.8, the top 10 tokens contribute 87.2% of the L2<sup>2</sup> value (you need to square L2 for top X contribute Y metrics to make sense), so most of the token distance comes from common tokens that become more common.

## Appendix 5: Metric graphs for top_p and top_k

<figure class="wp-block-image size-large">
<img src="https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/05/image-2.png?resize=1024%2C316&amp;ssl=1" class="wp-image-1049" data-recalc-dims="1" loading="lazy" decoding="async" data-attachment-id="1049" data-permalink="https://rosmine.ai/2026/05/18/fixing-llm-writing-with-distribution-fine-tuning/image-48/" data-orig-file="https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/05/image-2.png?fit=1654%2C510&amp;ssl=1" data-orig-size="1654,510" data-comments-opened="1" data-image-meta="{&quot;aperture&quot;:&quot;0&quot;,&quot;credit&quot;:&quot;&quot;,&quot;camera&quot;:&quot;&quot;,&quot;caption&quot;:&quot;&quot;,&quot;created_timestamp&quot;:&quot;0&quot;,&quot;copyright&quot;:&quot;&quot;,&quot;focal_length&quot;:&quot;0&quot;,&quot;iso&quot;:&quot;0&quot;,&quot;shutter_speed&quot;:&quot;0&quot;,&quot;title&quot;:&quot;&quot;,&quot;orientation&quot;:&quot;0&quot;}" data-image-title="image" data-image-description="" data-image-caption="" data-large-file="https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/05/image-2.png?fit=1024%2C316&amp;ssl=1" srcset="https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/05/image-2.png?resize=1024%2C316&amp;ssl=1 1024w, https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/05/image-2.png?resize=300%2C93&amp;ssl=1 300w, https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/05/image-2.png?resize=768%2C237&amp;ssl=1 768w, https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/05/image-2.png?resize=1536%2C474&amp;ssl=1 1536w, https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/05/image-2.png?resize=1200%2C370&amp;ssl=1 1200w, https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/05/image-2.png?w=1654&amp;ssl=1 1654w" sizes="auto, (max-width: 1000px) 100vw, 1000px" width="1024" height="316" alt="Line graphs depicting the relationship between top_p values and various metrics (l2_1gram, Judge Model Quality, and MMD) at a constant temperature of 0.8 and topk of -1. Each metric is represented in separate panels with different colored lines for various configurations." />
</figure>

<figure class="wp-block-image size-large">
<img src="https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/05/image-3.png?resize=1024%2C314&amp;ssl=1" class="wp-image-1050" data-recalc-dims="1" loading="lazy" decoding="async" data-attachment-id="1050" data-permalink="https://rosmine.ai/2026/05/18/fixing-llm-writing-with-distribution-fine-tuning/image-49/" data-orig-file="https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/05/image-3.png?fit=1655%2C507&amp;ssl=1" data-orig-size="1655,507" data-comments-opened="1" data-image-meta="{&quot;aperture&quot;:&quot;0&quot;,&quot;credit&quot;:&quot;&quot;,&quot;camera&quot;:&quot;&quot;,&quot;caption&quot;:&quot;&quot;,&quot;created_timestamp&quot;:&quot;0&quot;,&quot;copyright&quot;:&quot;&quot;,&quot;focal_length&quot;:&quot;0&quot;,&quot;iso&quot;:&quot;0&quot;,&quot;shutter_speed&quot;:&quot;0&quot;,&quot;title&quot;:&quot;&quot;,&quot;orientation&quot;:&quot;0&quot;}" data-image-title="image" data-image-description="" data-image-caption="" data-large-file="https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/05/image-3.png?fit=1024%2C314&amp;ssl=1" srcset="https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/05/image-3.png?resize=1024%2C314&amp;ssl=1 1024w, https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/05/image-3.png?resize=300%2C92&amp;ssl=1 300w, https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/05/image-3.png?resize=768%2C235&amp;ssl=1 768w, https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/05/image-3.png?resize=1536%2C471&amp;ssl=1 1536w, https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/05/image-3.png?resize=1200%2C368&amp;ssl=1 1200w, https://i0.wp.com/rosmine.ai/wp-content/uploads/2026/05/image-3.png?w=1655&amp;ssl=1 1655w" sizes="auto, (max-width: 1000px) 100vw, 1000px" width="1024" height="314" alt="Three line graphs displaying results for different metrics: I2_1gram on the left, Judge Model Quality in the center, and MMD on the right. Each graph plots top_k values on the x-axis (ranging from 1 to 300) against respective metric values on the y-axis, with lines representing different model sizes (14B, 8B, and DFT)." />
</figure>

## Appendix 6: Effect of Data size

How would the baseline perform if we had 2x as much data?

To answer this, I used the cheaper alternative of taking a checkpoint with half or 1/4 as much data using that for evaluations.

The table below shows comparison of superbaseline scores at quarter data, half data and full data. Note that we are training on web documents which are similar to the pretraining data, so most of the learning is just matching formatting. In the table below, we see continued improvement from quarter data to half data, but at full data results have plateaued, or even started to regress a little.

Note that the amount of data we are using is a tiny fraction of the total pretraining data, so it’s reasonable to expect that past a certain point, improvement would be negligible compared to noise in training/eval processes. If we trained on 100x data we would probably see continued improvement in SFT superbaseline scores, but DFT can get us improvement much more cheaply.

<figure class="wp-block-table">
<table class="has-fixed-layout">
<tbody>
<tr class="odd">
<td>Super Baseline</td>
<td>MMD↓</td>
<td>JMQ↑</td>
<td>Token L2↓</td>
</tr>
<tr class="even">
<td>4B-quarter data</td>
<td>.0497</td>
<td>.23</td>
<td>.0068</td>
</tr>
<tr class="odd">
<td>4B-half data</td>
<td>.0466</td>
<td>.255</td>
<td>.0040</td>
</tr>
<tr class="even">
<td>4B-full data</td>
<td>.0477</td>
<td>.265</td>
<td>.0043</td>
</tr>
<tr class="odd">
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr class="even">
<td>8B-quarter data</td>
<td>.042</td>
<td>.34</td>
<td>.006</td>
</tr>
<tr class="odd">
<td>8B-half data</td>
<td>.041</td>
<td>.37</td>
<td>.0046</td>
</tr>
<tr class="even">
<td>8B-full data</td>
<td>.045</td>
<td>.335</td>
<td>.0040</td>
</tr>
<tr class="odd">
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr class="even">
<td>14B-quarter data</td>
<td>.0379</td>
<td>.475</td>
<td>.0062</td>
</tr>
<tr class="odd">
<td>14B-half data</td>
<td>.037</td>
<td>.49</td>
<td>.0039</td>
</tr>
<tr class="even">
<td>14B-full data</td>
<td>.038</td>
<td>.44</td>
<td>.0045</td>
</tr>
</tbody>
</table>
</figure>

## Appendix 8: Comparison with other models

In this section I use the same test set to evaluate other models. This is not a very good comparison because DFT is an algorithm to better match the training data distribution, and the metrics (with the exception of JMQ) measure distributional similarity on the test set. Since other models have been trained on different data and have other post training steps, we should not expect them to do well.

A few notes: As expected, MMD scores are much larger for other models. For Judge Model Quality, remember that this is the win rate times 2, so it’s possible to go above 1.0. One note of caution: Judge models are known to prefer outputs from LLMs ([Laurito](https://arxiv.org/abs/2407.12856)), especially their own ([Panickssery](https://arxiv.org/abs/2404.13076)). So MMD and Token L2 are unfair against other models, and JMQ is unfair in their advantage.

<figure class="wp-block-table">
<table class="has-fixed-layout">
<tbody>
<tr class="odd">
<td>Model</td>
<td>MMD↓</td>
<td>JMQ↑</td>
<td>Token L2↓</td>
</tr>
<tr class="even">
<td>DFT 14B</td>
<td>0.018</td>
<td>0.73</td>
<td>0.0039</td>
</tr>
<tr class="odd">
<td>Claude 4.6</td>
<td>0.16</td>
<td>1.88</td>
<td>.023</td>
</tr>
<tr class="even">
<td>Gemini3.1Pro</td>
<td>0.17</td>
<td>1.86</td>
<td>.025</td>
</tr>
<tr class="odd">
<td>Kimi 2.5</td>
<td>0.16</td>
<td>1.84</td>
<td>.031</td>
</tr>
<tr class="even">
<td>GPT 5.4</td>
<td>.11</td>
<td>1.96</td>
<td>.035</td>
</tr>
</tbody>
</table>
</figure>

## Appendix 9: Fine Grained Judge model analysis

The Judge Model Quality scores above evaluate the overall quality of the sample. In this section, we evaluate different criteria that affect writing quality, specifically clarity, coherence, creativity, depth, and prompt relevance.

The full super-baseline has 132 different evaluation sets for a single model size. Running all of these for each evaluation dimension would cost too much, so I used the same hyperparameters that maximized overall judge model quality for the 14B parameter model.

The judge model prompts are at the end of this appendix.

From this we see improvement in all dimensions, with greatest improvement in Depth (going deeper into details about the subject instead of generic surface level descriptions) and Creativity.

<figure class="wp-block-table">
<table class="has-fixed-layout">
<tbody>
<tr class="odd">
<td>Eval Dimension</td>
<td>14B Baseline</td>
<td>14B DFT</td>
</tr>
<tr class="even">
<td>Clarity</td>
<td>70.5</td>
<td>82.0</td>
</tr>
<tr class="odd">
<td>Coherence</td>
<td>54.5</td>
<td>70.0</td>
</tr>
<tr class="even">
<td>Creativity</td>
<td>32.5</td>
<td>86.0</td>
</tr>
<tr class="odd">
<td>Depth</td>
<td>35.5</td>
<td>87.5</td>
</tr>
<tr class="even">
<td>Prompt Relevance</td>
<td>44.0</td>
<td>75.0</td>
</tr>
</tbody>
</table>
</figure>

I also compared the overall quality for different judge models, evaluating with Claude, Gemini and Grok models of different sizes. We see that although there is a high variance in baseline score across different models, we see consistent improvement with DFT.

<figure class="wp-block-table">
<table class="has-fixed-layout">
<tbody>
<tr class="odd">
<td>Model</td>
<td>14B Baseline</td>
<td>14B DFT</td>
<td>Relative change</td>
</tr>
<tr class="even">
<td>Claude Sonnet 4.6</td>
<td>37</td>
<td>51</td>
<td>+37%</td>
</tr>
<tr class="odd">
<td>Claude Sonnet</td>
<td>19</td>
<td>46</td>
<td>+142%</td>
</tr>
<tr class="even">
<td>Gemini 3.1 Flash</td>
<td>48</td>
<td>62</td>
<td>+29%</td>
</tr>
<tr class="odd">
<td>Gemini 3.1 Pro Preview</td>
<td>62</td>
<td>72</td>
<td>+16%</td>
</tr>
<tr class="even">
<td>Grok 3 mini</td>
<td>54</td>
<td>84</td>
<td>+56%</td>
</tr>
<tr class="odd">
<td>Grok 4.3</td>
<td>35</td>
<td>65</td>
<td>+86%</td>
</tr>
<tr class="even">
<td>GPT 5.5</td>
<td>37</td>
<td>41</td>
<td>+11%</td>
</tr>
</tbody>
</table>
</figure>

### Judge Prompts

**Overall Quality:**

Given the a prompt, candidate A, and candidate B, choose which is the better response, based off of quality of the writing and following the prompt. Return which one is better, either A or B:

=== PROMPT ===

{prompt}

=== CANDIDATE A ===

{candidate_a}

=== CANDIDATE B ===

{candidate_b}

------------------------------------------------------------------------

All prompts end with the same prompt/candidate_a/candidate_b text, so I remove that in following prompts for simplicity

**Coherence:**

Given the prompt, candidate A, and candidate B, choose which response is more coherent.

Coherence means the response has a clear through-line, logical progression, consistent claims or story details, and no confusing jumps or contradictions.

Return only A or B.

------------------------------------------------------------------------

**Creativity:**

Given the prompt, candidate A, and candidate B, choose which response is more creative.

Creativity means the response uses fresh ideas, vivid details, distinctive phrasing, and non-generic development while still fitting the assignment.

Return only A or B.

------------------------------------------------------------------------

**Clarity:**

Given the prompt, candidate A, and candidate B, choose which response is clearer.

Clarity means the response is easy to follow, precise, readable, well organized, and avoids muddled wording or unnecessary ambiguity.

Return only A or B.

------------------------------------------------------------------------

**Prompt Relevance**

Given the prompt, candidate A, and candidate B, choose which response is more relevant to the prompt.

Relevance means the response follows the user’s assignment, respects stated constraints, addresses the requested topic, and avoids drifting into unrelated material.

Return only A or B.

------------------------------------------------------------------------

**Depth:**

Given the prompt, candidate A, and candidate B, choose which response has more depth.

Depth means the response goes beyond surface-level points, develops ideas with substance, gives meaningful detail, and shows insight appropriate to the assignment.

Return only A or B.

------------------------------------------------------------------------

## Appendix 10: Output Diversity

LLMs tend to overuse certain words and grammatical phrases. It’s more than just the common obvious ones like emdashes and delve. For example, multiple models frequently use “Elara Voss” ([Read](https://maxread.substack.com/p/who-is-elara-voss)) when then need to come up with a name.

To quantify diversity, I used self-BLEU ([Zhu](https://arxiv.org/abs/1802.01886)) to compare model outputs with the outputs from different prompts. Here lower is better. I don’t use the SFT super baseline here because it’s possible to make self-BLEU arbitrarily low by setting the temperature high and making the output random gibberish. For reference, I give the value for the SFT models with temp=0.9.

<figure class="wp-block-table">
<table class="has-fixed-layout">
<tbody>
<tr class="odd">
<td>Model Size</td>
<td>Human reference self-BLEU</td>
<td>SFT temp=0.9</td>
<td>DFT</td>
</tr>
<tr class="even">
<td>4B</td>
<td>0.061</td>
<td>0.080</td>
<td>0.064</td>
</tr>
<tr class="odd">
<td>8B</td>
<td>0.061</td>
<td>0.0811</td>
<td>0.066</td>
</tr>
<tr class="even">
<td>14B</td>
<td>0.061</td>
<td>0.0790</td>
<td>0.063</td>
</tr>
</tbody>
</table>
</figure>

A more concrete way to measure output diversity is to show that DFT avoids overusing the same tokens, like how other LLMs do with emdash, which I do in the next Appendix.

## Appendix 11: Avoiding slop signs

To measure tokens that get overused, I first filtered to all tokens that appear in at least 2% of LLM responses (this is necessary since doing this analysis on rarer tokens results in noise from low frequency tokens)

I then compared the token frequencies with a set of human responses to find any tokens that were overused in DFT responses compared to human writing. I did this by using two sets of prompts, A and B, and comparing LLM-A vs human B and human-A vs human-B for a baseline. From this we see that any tokens that seem overused (high relative diff in token frequency) match the same overuse as human vs. human (i.e. the overuse is just due to noise)

DFT vs. Human

The “Human Frequency” column counts the percentage of outputs that contain this token, not the percentage of all tokens

<figure class="wp-block-table">
<table class="has-fixed-layout">
<tbody>
<tr class="odd">
<td>Token</td>
<td>Relative Diff</td>
<td>Human Frequency</td>
</tr>
<tr class="even">
<td>file</td>
<td>5.7</td>
<td>1.1%</td>
</tr>
<tr class="odd">
<td>smoking</td>
<td>3.4</td>
<td>0.7%</td>
</tr>
<tr class="even">
<td>routine</td>
<td>2.8</td>
<td>0.8%</td>
</tr>
</tbody>
</table>
</figure>

Human vs. Human

<figure class="wp-block-table">
<table class="has-fixed-layout">
<tbody>
<tr class="odd">
<td>Token</td>
<td>Relative Diff</td>
<td>Human Frequency</td>
</tr>
<tr class="even">
<td>file</td>
<td>4.1</td>
<td>1.1%</td>
</tr>
<tr class="odd">
<td>faith</td>
<td>3.4</td>
<td>0.9%</td>
</tr>
<tr class="even">
<td>items</td>
<td>2.7</td>
<td>0.9%</td>
</tr>
</tbody>
</table>
</figure>

For reference, here’s the table for GPT 5:

The first set of tokens are overused because they appear in ~0.1-0.3 percent of human outputs in the dataset, but appear in 5-10% of GPT 5 outputs. When ranked by relative increase, emdashes actually have a much lower relative increase, but that’s mostly because they already appear in 18.6% of human outputs in the dataset. When you look at the tokens after that (targeted, identity, etc.) you get overused tokens that are more comparable to the rates seen from DFT. So if you can’t tell that GPT 5 is overusing targeted/identity/trust, then you shouldn’t see any overused tokens from a DFT model.

<figure class="wp-block-table">
<table class="has-fixed-layout">
<tbody>
<tr class="odd">
<td>Token</td>
<td>Relative Diff</td>
<td>Human Frequency</td>
</tr>
<tr class="even">
<td>corridors</td>
<td>45.2</td>
<td>0.1%</td>
</tr>
<tr class="odd">
<td>norms</td>
<td>43.1</td>
<td>0.1%</td>
</tr>
<tr class="even">
<td>align</td>
<td>36.0</td>
<td>0.2%</td>
</tr>
<tr class="odd">
<td>metrics</td>
<td>27.2</td>
<td>0.2%</td>
</tr>
<tr class="even">
<td>engagement</td>
<td>26.5</td>
<td>0.2%</td>
</tr>
<tr class="odd">
<td>…</td>
<td>…</td>
<td>…</td>
</tr>
<tr class="even">
<td>—</td>
<td>5.1</td>
<td>18.6%</td>
</tr>
<tr class="odd">
<td>targeted</td>
<td>5.1</td>
<td>1.6%</td>
</tr>
<tr class="even">
<td>identity</td>
<td>5.0</td>
<td>1%</td>
</tr>
<tr class="odd">
<td>trust</td>
<td>4.9</td>
<td>1.2%</td>
</tr>
</tbody>
</table>
</figure>

A stronger eval would be to run a statistical test to see if there is any difference between token distributions. The DFT outputs do not pass that test yet, but I hope to fix that with a larger model.

## Footnotes

These should be available by clicking the footnote number. We repeat them for those reading the blog offline

1.  Note that metrics like KL or JS Divergence do not work well here because there are generally many tokens with that appear in reference but not output, or vice versa, and these have outsized contribution to the overall metric.
2.  GPT5.4-mini, with prompts in randomized order, to prevent positional bias
3.  I initially tried starting from Qwen3 Base models, however these had bad MMD and JMQ scores due to poor instruction following. The instruct tuned Qwen3 models (e.g. [14B](https://huggingface.co/Qwen/Qwen3-14B)) performed much better.
4.  The first is a Chinese character meaning “to wipe”, the second is Korean for “huh?” or “what?”. Perhaps the model was also surprised that it switched to Chinese.
5.  As seen in Appendix 3, MAUVE saturates, scoring .997+ for 4B baseline
6.  This paper suggests FID using BERT. I test FID in Appendix 3, but don’t use it for a key metric since Frechet metrics assume the underlying distribution is Gaussian, which is not true for language
7.  It seemed suspicious that 17.4% of human generated text has 3 sentences with the same start word in a row, that seems too high. Inspecting the data, all the examples look valid, so it’s unlikely to be a bug. Mostly this is from lists, e.g. a list of questions that start with “How”, or 3 nouns in a row (“The Lamu Archipelago…”, “The islands lie between…”, “The largest island…”)

## References

Alihosseini, Danial, Ehsan Montahaei, and Mahdieh Soleymani Baghshah. “Jointly measuring diversity and quality in text generation models.” *Proceedings of the Workshop on Methods for Optimizing and Evaluating Neural Language Generation*. 2019.

Babakhin, Yauhen, et al. “Llama-Embed-Nemotron-8B: A Universal Text Embedding Model for Multilingual and Cross-Lingual Tasks.” *arXiv preprint arXiv:2511.07025* (2025). <https://arxiv.org/abs/2511.07025> <https://huggingface.co/nvidia/llama-embed-nemotron-8b>

Braverman, Mark, et al. “Calibration, entropy rates, and memory in language models.” *International Conference on Machine Learning*. PMLR, 2020.

Borgwardt, Karsten M., et al. “Integrating structured biological data by kernel maximum mean discrepancy.” *Bioinformatics* 22.14 (2006): e49-e57. <https://academic.oup.com/bioinformatics/article/22/14/e49/228383>

Heusel, Martin, et al. “Gans trained by a two time-scale update rule converge to a local nash equilibrium.” *Advances in neural information processing systems* 30 (2017). <https://arxiv.org/abs/1706.08500> (FID)

Ho, Jonathan, and Stefano Ermon. “Generative adversarial imitation learning.” *Advances in neural information processing systems* 29 (2016).

Holtzman, Ari, et al. “The curious case of neural text degeneration.” *arXiv preprint arXiv:1904.09751* (2019).

Hu, Edward J., et al. “Lora: Low-rank adaptation of large language models.” *Iclr* 1.2 (2022): 3.

Kimi Team. “Kimi k2: Open agentic intelligence.” *arXiv preprint arXiv:2507.20534* (2025).

Laurito, Walter, et al. “AI–AI bias: Large language models favor communications generated by large language models.” *Proceedings of the National Academy of Sciences* 122.31 (2025): e2415697122.

Lialin, Vladislav, et al. “Relora: High-rank training through low-rank updates, 2023.” *URL <a href="https://arxiv" rel="nofollow">https://arxiv</a>. org/abs/2307.05695*.

Meister, Clara, et al. “Locally typical sampling.” *Transactions of the Association for Computational Linguistics* 11 (2023): 102-121.

OpenAI. “Where the goblins came from” <https://openai.com/index/where-the-goblins-came-from/>

Pangram AI detector. <https://www.pangram.com/>

Panickssery, Arjun, Samuel R. Bowman, and Shi Feng. “Llm evaluators recognize and favor their own generations.” *Advances in Neural Information Processing Systems* 37 (2024): 68772-68802.

Papineni, Kishore, et al. “Bleu: a method for automatic evaluation of machine translation.” *Proceedings of the 40th annual meeting of the Association for Computational Linguistics*. 2002. <https://dl.acm.org/doi/10.3115/1073083.1073135>

Penedo, Guilherme, et al. “The fineweb datasets: Decanting the web for the finest text data at scale.” *Advances in Neural Information Processing Systems* 37 (2024): 30811-30849. <https://arxiv.org/abs/2406.17557>

Pillutla, Krishna, et al. “Mauve: Measuring the gap between neural text and human text using divergence frontiers.” *Advances in Neural Information Processing Systems* 34 (2021): 4816-4828. <https://arxiv.org/abs/2102.01454>

Popović, Maja. “chrF: character n-gram F-score for automatic MT evaluation.” *Proceedings of the tenth workshop on statistical machine translation*. 2015. <https://aclanthology.org/W15-3049/>

Ranzato, Marc’Aurelio, et al. “Sequence level training with recurrent neural networks.” *arXiv preprint arXiv:1511.06732* (2015).

Rosmine. “Was my \$48K GPU server worth it?” <https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/>

Read, Max. “Who is Elara Voss?” <https://maxread.substack.com/p/who-is-elara-voss>

Welleck, Sean, et al. “Neural text generation with unlikelihood training.” *arXiv preprint arXiv:1908.04319* (2019).

Wu, Qingyang, Lei Li, and Zhou Yu. “Textgail: Generative adversarial imitation learning for text generation.” *Proceedings of the AAAI Conference on Artificial Intelligence*. Vol. 35. No. 16. 2021.

Wulfmeier, Markus, et al. “Imitating language via scalable inverse reinforcement learning.” *Advances in Neural Information Processing Systems* 37 (2024): 90714-90735.

Yang, An, et al. “Qwen3 technical report.” *arXiv preprint arXiv:2505.09388* (2025). <https://arxiv.org/abs/2505.09388>

Zhu, Yaoming, et al. “Texygen: A benchmarking platform for text generation models.” *The 41st international ACM SIGIR conference on research & development in information retrieval*. 2018.

<div class="sharedaddy sd-sharing-enabled">

<div class="robots-nocontent sd-block sd-social sd-social-icon-text sd-sharing">

### Share this:

<div class="sd-content">

- <a href="https://rosmine.ai/2026/05/18/fixing-llm-writing-with-distribution-fine-tuning/?share=twitter" class="share-twitter sd-button share-icon" rel="nofollow noopener noreferrer" data-shared="sharing-twitter-753" target="_blank" aria-labelledby="sharing-twitter-753"><span id="sharing-twitter-753" hidden="">Share on X (Opens in new window)</span> <span>X</span></a>
- <a href="https://rosmine.ai/2026/05/18/fixing-llm-writing-with-distribution-fine-tuning/?share=facebook" class="share-facebook sd-button share-icon" rel="nofollow noopener noreferrer" data-shared="sharing-facebook-753" target="_blank" aria-labelledby="sharing-facebook-753"><span id="sharing-facebook-753" hidden="">Share on Facebook (Opens in new window)</span> <span>Facebook</span></a>
- 

</div>

</div>

</div>

<div id="like-post-wrapper-234301566-753-6a10e0c004ed7" class="sharedaddy sd-block sd-like jetpack-likes-widget-wrapper jetpack-likes-widget-unloaded" data-src="https://widgets.wp.com/likes/?ver=15.9-a.3#blog_id=234301566&amp;post_id=753&amp;origin=rosmine.ai&amp;obj_id=234301566-753-6a10e0c004ed7" data-name="like-post-frame-234301566-753-6a10e0c004ed7" data-title="Like or Reblog">

### Like this:

<div class="likes-widget-placeholder post-likes-widget-placeholder" style="height: 55px;">

<span class="button">Like</span> <span class="loading"><img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iamV0cGFjay1zcGlubmVyIiB3aWR0aD0iMTgiIGhlaWdodD0iMTgiIHZpZXdib3g9IjAgMCAxMDAgMTAwIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGFyaWEtaGlkZGVuPSJ0cnVlIiBmb2N1c2FibGU9ImZhbHNlIj48Y2lyY2xlIGN4PSI1MCIgY3k9IjUwIiByPSI0NiIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjZGRkIiBzdHJva2Utd2lkdGg9IjgiPjwvY2lyY2xlPjxwYXRoIGQ9Ik0gNTAgNCBBIDQ2IDQ2IDAgMCAxIDk2IDUwIiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSI4IiBzdHJva2UtbGluZWNhcD0icm91bmQiPjxhbmltYXRldHJhbnNmb3JtIGF0dHJpYnV0ZW5hbWU9InRyYW5zZm9ybSIgdHlwZT0icm90YXRlIiBkdXI9IjEuNHMiIGZyb209IjAgNTAgNTAiIHRvPSIzNjAgNTAgNTAiIHJlcGVhdGNvdW50PSJpbmRlZmluaXRlIj48L2FuaW1hdGV0cmFuc2Zvcm0+PC9wYXRoPjwvc3ZnPg==" class="jetpack-spinner" /><span class="screen-reader-text">Loading…</span></span>

</div>

<span class="sd-text-color"></span><span class="sd-link-color"></span>

</div>

</div>

<div class="wp-block-spacer" style="height:30px" aria-hidden="true">

</div>

<div id="respond" class="comment-respond wp-block-post-comments-form">

### Leave a Reply<span class="small"><a href="/2026/05/18/fixing-llm-writing-with-distribution-fine-tuning/#respond" id="cancel-comment-reply-link" rel="nofollow" style="display:none;">Cancel reply</a></span>

</div>

<div class="wp-block-comments">

## 4 responses to “Fixing LLM writing with Distribution Fine Tuning”

1.  <div id="comment-28">

    <div class="wp-block-columns is-layout-flex wp-container-core-columns-is-layout-f56f613f wp-block-columns-is-layout-flex">

    <div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow" style="flex-basis:40px">

    <div class="wp-block-avatar">

    <img src="https://secure.gravatar.com/avatar/?s=40&amp;d=identicon&amp;r=g" class="avatar avatar-40 photo avatar-default wp-block-avatar__image" style="border-radius:20px;" srcset="https://secure.gravatar.com/avatar/?s=80&amp;d=identicon&amp;r=g 2x" loading="lazy" decoding="async" width="40" height="40" alt=" Avatar" />

    </div>

    </div>

    <div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow">

    <div class="wp-block-comment-author-name has-small-font-size">

    Anonymous

    </div>

    <div class="wp-block-group is-layout-flex wp-block-group-is-layout-flex" style="margin-top:0px;margin-bottom:0px">

    <div class="wp-block-comment-date has-small-font-size">

    [May 18, 2026](https://rosmine.ai/2026/05/18/fixing-llm-writing-with-distribution-fine-tuning/comment-page-1/#comment-28)

    </div>

    </div>

    <div class="wp-block-comment-content">

    Great stuff! Thank you for publishing it

    <div id="like-comment-wrapper-234301566-28-6a10e0c00d1af" class="jetpack-comment-likes-widget-wrapper jetpack-likes-widget-unloaded" data-src="https://widgets.wp.com/likes/#blog_id=234301566&amp;comment_id=28&amp;origin=rosmine.ai&amp;obj_id=234301566-28-6a10e0c00d1af" data-name="like-comment-frame-234301566-28-6a10e0c00d1af">

    <div class="likes-widget-placeholder comment-likes-widget-placeholder comment-likes">

    <span class="loading">Loading...</span>

    </div>

    <div class="comment-likes-widget jetpack-likes-widget comment-likes">

    <span class="comment-like-feedback"></span><span class="sd-text-color"></span><span class="sd-link-color"></span>

    </div>

    </div>

    </div>

    <div class="wp-block-comment-reply-link has-small-font-size">

    <a href="https://rosmine.ai/2026/05/18/fixing-llm-writing-with-distribution-fine-tuning/comment-page-1/?replytocom=28#respond" class="comment-reply-link" rel="nofollow" data-commentid="28" data-postid="753" data-belowelement="comment-28" data-respondelement="respond" data-replyto="Reply to Anonymous" aria-label="Reply to Anonymous">Reply</a>

    </div>

    </div>

    </div>

    </div>

2.  <div id="comment-29">

    <div class="wp-block-columns is-layout-flex wp-container-core-columns-is-layout-f56f613f wp-block-columns-is-layout-flex">

    <div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow" style="flex-basis:40px">

    <div class="wp-block-avatar">

    <img src="https://secure.gravatar.com/avatar/?s=40&amp;d=identicon&amp;r=g" class="avatar avatar-40 photo avatar-default wp-block-avatar__image" style="border-radius:20px;" srcset="https://secure.gravatar.com/avatar/?s=80&amp;d=identicon&amp;r=g 2x" loading="lazy" decoding="async" width="40" height="40" alt=" Avatar" />

    </div>

    </div>

    <div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow">

    <div class="wp-block-comment-author-name has-small-font-size">

    Anonymous

    </div>

    <div class="wp-block-group is-layout-flex wp-block-group-is-layout-flex" style="margin-top:0px;margin-bottom:0px">

    <div class="wp-block-comment-date has-small-font-size">

    [May 18, 2026](https://rosmine.ai/2026/05/18/fixing-llm-writing-with-distribution-fine-tuning/comment-page-1/#comment-29)

    </div>

    </div>

    <div class="wp-block-comment-content">

    Your Twitter account linked in “About” doesn’t exist (great work btw)

    <div id="like-comment-wrapper-234301566-29-6a10e0c00df12" class="jetpack-comment-likes-widget-wrapper jetpack-likes-widget-unloaded" data-src="https://widgets.wp.com/likes/#blog_id=234301566&amp;comment_id=29&amp;origin=rosmine.ai&amp;obj_id=234301566-29-6a10e0c00df12" data-name="like-comment-frame-234301566-29-6a10e0c00df12">

    <div class="likes-widget-placeholder comment-likes-widget-placeholder comment-likes">

    <span class="loading">Loading...</span>

    </div>

    <div class="comment-likes-widget jetpack-likes-widget comment-likes">

    <span class="comment-like-feedback"></span><span class="sd-text-color"></span><span class="sd-link-color"></span>

    </div>

    </div>

    </div>

    <div class="wp-block-comment-reply-link has-small-font-size">

    <a href="https://rosmine.ai/2026/05/18/fixing-llm-writing-with-distribution-fine-tuning/comment-page-1/?replytocom=29#respond" class="comment-reply-link" rel="nofollow" data-commentid="29" data-postid="753" data-belowelement="comment-29" data-respondelement="respond" data-replyto="Reply to Anonymous" aria-label="Reply to Anonymous">Reply</a>

    </div>

    </div>

    </div>

    1.  <div id="comment-33">

        <div class="wp-block-columns is-layout-flex wp-container-core-columns-is-layout-f56f613f wp-block-columns-is-layout-flex">

        <div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow" style="flex-basis:40px">

        <div class="wp-block-avatar">

        <img src="https://secure.gravatar.com/avatar/a65fe49dd37df098b169a81cf418108ee6ec0d25f2c62bd5c1d983ba79299285?s=40&amp;d=identicon&amp;r=g" class="avatar avatar-40 photo wp-block-avatar__image" style="border-radius:20px;" srcset="https://secure.gravatar.com/avatar/a65fe49dd37df098b169a81cf418108ee6ec0d25f2c62bd5c1d983ba79299285?s=80&amp;d=identicon&amp;r=g 2x" loading="lazy" decoding="async" width="40" height="40" alt="Ben Rosmine Avatar" />

        </div>

        </div>

        <div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow">

        <div class="wp-block-comment-author-name has-small-font-size">

        Ben Rosmine

        </div>

        <div class="wp-block-group is-layout-flex wp-block-group-is-layout-flex" style="margin-top:0px;margin-bottom:0px">

        <div class="wp-block-comment-date has-small-font-size">

        [May 19, 2026](https://rosmine.ai/2026/05/18/fixing-llm-writing-with-distribution-fine-tuning/comment-page-1/#comment-33)

        </div>

        </div>

        <div class="wp-block-comment-content">

        thank you! fixed

        <div id="like-comment-wrapper-234301566-33-6a10e0c00ea91" class="jetpack-comment-likes-widget-wrapper jetpack-likes-widget-unloaded" data-src="https://widgets.wp.com/likes/#blog_id=234301566&amp;comment_id=33&amp;origin=rosmine.ai&amp;obj_id=234301566-33-6a10e0c00ea91" data-name="like-comment-frame-234301566-33-6a10e0c00ea91">

        <div class="likes-widget-placeholder comment-likes-widget-placeholder comment-likes">

        <span class="loading">Loading...</span>

        </div>

        <div class="comment-likes-widget jetpack-likes-widget comment-likes">

        <span class="comment-like-feedback"></span><span class="sd-text-color"></span><span class="sd-link-color"></span>

        </div>

        </div>

        </div>

        <div class="wp-block-comment-reply-link has-small-font-size">

        <a href="https://rosmine.ai/2026/05/18/fixing-llm-writing-with-distribution-fine-tuning/comment-page-1/?replytocom=33#respond" class="comment-reply-link" rel="nofollow" data-commentid="33" data-postid="753" data-belowelement="comment-33" data-respondelement="respond" data-replyto="Reply to Ben Rosmine" aria-label="Reply to Ben Rosmine">Reply</a>

        </div>

        </div>

        </div>

        </div>

    </div>

3.  <div id="comment-39">

    <div class="wp-block-columns is-layout-flex wp-container-core-columns-is-layout-f56f613f wp-block-columns-is-layout-flex">

    <div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow" style="flex-basis:40px">

    <div class="wp-block-avatar">

    </div>

    </div>

    <div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow">

    <div class="wp-block-comment-author-name has-small-font-size">

    <a href="https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/" rel="external nofollow ugc" target="_self">Was my $48K GPU server worth it? – Rosmine ML Blog</a>

    </div>

    <div class="wp-block-group is-layout-flex wp-block-group-is-layout-flex" style="margin-top:0px;margin-bottom:0px">

    <div class="wp-block-comment-date has-small-font-size">

    [May 21, 2026](https://rosmine.ai/2026/05/18/fixing-llm-writing-with-distribution-fine-tuning/comment-page-1/#comment-39)

    </div>

    </div>

    <div class="wp-block-comment-content">

    \[…\] The point of buying the server wasn’t to save money, it was to build something cool. I spent a long time trying high risk/high reward experiments and failing. But now I have something good. I’ve solved a major problem with LLMs. And I’m launching next Monday so we will soon see if it’s actually a breakthrough or just LLM psychosis 🙂 (UPDATE: Launch was a success! 400K+ views, and multiple collab opportunities. Read more here) \[…\]

    <div id="like-comment-wrapper-234301566-39-6a10e0c00f67c" class="jetpack-comment-likes-widget-wrapper jetpack-likes-widget-unloaded" data-src="https://widgets.wp.com/likes/#blog_id=234301566&amp;comment_id=39&amp;origin=rosmine.ai&amp;obj_id=234301566-39-6a10e0c00f67c" data-name="like-comment-frame-234301566-39-6a10e0c00f67c">

    <div class="likes-widget-placeholder comment-likes-widget-placeholder comment-likes">

    <span class="loading">Loading...</span>

    </div>

    <div class="comment-likes-widget jetpack-likes-widget comment-likes">

    <span class="comment-like-feedback"></span><span class="sd-text-color"></span><span class="sd-link-color"></span>

    </div>

    </div>

    </div>

    <div class="wp-block-comment-reply-link has-small-font-size">

    <a href="https://rosmine.ai/2026/05/18/fixing-llm-writing-with-distribution-fine-tuning/comment-page-1/?replytocom=39#respond" class="comment-reply-link" rel="nofollow" data-commentid="39" data-postid="753" data-belowelement="comment-39" data-respondelement="respond" data-replyto="Reply to Was my $48K GPU server worth it? – Rosmine ML Blog" aria-label="Reply to Was my $48K GPU server worth it? – Rosmine ML Blog">Reply</a>

    </div>

    </div>

    </div>

    </div>

</div>

<div class="wp-block-group is-nowrap is-layout-flex wp-container-core-group-is-layout-08aa00d2 wp-block-group-is-layout-flex">

<div class="post-navigation-link-next wp-block-post-navigation-link">

</div>

</div>

</div>

</div>

</div>

</div>

</div>

![](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdib3g9IjAgMCAwIDAiIHdpZHRoPSIwIiBoZWlnaHQ9IjAiIGZvY3VzYWJsZT0iZmFsc2UiIHJvbGU9Im5vbmUiIHN0eWxlPSJ2aXNpYmlsaXR5OiBoaWRkZW47IHBvc2l0aW9uOiBhYnNvbHV0ZTsgbGVmdDogLTk5OTlweDsgb3ZlcmZsb3c6IGhpZGRlbjsiPjxkZWZzPjxmaWx0ZXIgaWQ9IndwLWR1b3RvbmUtMDAwMDAwLWZlZmNmOS0yIj48ZmVjb2xvcm1hdHJpeCBjb2xvci1pbnRlcnBvbGF0aW9uLWZpbHRlcnM9InNSR0IiIHR5cGU9Im1hdHJpeCIgdmFsdWVzPSIgLjI5OSAuNTg3IC4xMTQgMCAwIC4yOTkgLjU4NyAuMTE0IDAgMCAuMjk5IC41ODcgLjExNCAwIDAgLjI5OSAuNTg3IC4xMTQgMCAwICI+PC9mZWNvbG9ybWF0cml4PjxmZWNvbXBvbmVudHRyYW5zZmVyIGNvbG9yLWludGVycG9sYXRpb24tZmlsdGVycz0ic1JHQiI+PGZlZnVuY3IgdHlwZT0idGFibGUiIHRhYmxldmFsdWVzPSIwIDAuOTk2MDc4NDMxMzcyNTUiPjwvZmVmdW5jcj48ZmVmdW5jZyB0eXBlPSJ0YWJsZSIgdGFibGV2YWx1ZXM9IjAgMC45ODgyMzUyOTQxMTc2NSI+PC9mZWZ1bmNnPjxmZWZ1bmNiIHR5cGU9InRhYmxlIiB0YWJsZXZhbHVlcz0iMCAwLjk3NjQ3MDU4ODIzNTI5Ij48L2ZlZnVuY2I+PGZlZnVuY2EgdHlwZT0idGFibGUiIHRhYmxldmFsdWVzPSIxIDEiPjwvZmVmdW5jYT48L2ZlY29tcG9uZW50dHJhbnNmZXI+PGZlY29tcG9zaXRlIGluMj0iU291cmNlR3JhcGhpYyIgb3BlcmF0b3I9ImluIj48L2ZlY29tcG9zaXRlPjwvZmlsdGVyPjwvZGVmcz48L3N2Zz4=)

<div class="jetpack-subscription-modal">

<div class="jetpack-subscription-modal__modal-content">

<div class="wp-block-group has-border-color jetpack-subscription-modal__modal-content-form is-layout-flow wp-block-group-is-layout-flow" style="border-color:#dddddd;border-width:1px;margin-top:0;margin-bottom:0;padding:32px">

## Discover more from Rosmine ML Blog

Subscribe now to keep reading and get access to the full archive.

<div class="wp-block-jetpack-subscriptions__supports-newline is-style-compact wp-block-jetpack-subscriptions">

<div class="wp-block-jetpack-subscriptions__container is-not-subscriber">

<div class="wp-block-jetpack-subscriptions__form-elements">

Type your email…

Subscribe

</div>

</div>

</div>

[Continue reading](#)

</div>

</div>

</div>

<div style="display:none">

<div class="grofile-hash-map-e534d7825b9ef2199b6b9fa3de0bc566">

</div>

</div>

<div id="jp-carousel-loading-overlay">

<div id="jp-carousel-loading-wrapper">

<span id="jp-carousel-library-loading"><img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iamV0cGFjay1zcGlubmVyIiB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIHZpZXdib3g9IjAgMCAxMDAgMTAwIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGFyaWEtaGlkZGVuPSJ0cnVlIiBmb2N1c2FibGU9ImZhbHNlIj48Y2lyY2xlIGN4PSI1MCIgY3k9IjUwIiByPSI0NiIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjZGRkIiBzdHJva2Utd2lkdGg9IjgiPjwvY2lyY2xlPjxwYXRoIGQ9Ik0gNTAgNCBBIDQ2IDQ2IDAgMCAxIDk2IDUwIiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSI4IiBzdHJva2UtbGluZWNhcD0icm91bmQiPjxhbmltYXRldHJhbnNmb3JtIGF0dHJpYnV0ZW5hbWU9InRyYW5zZm9ybSIgdHlwZT0icm90YXRlIiBkdXI9IjEuNHMiIGZyb209IjAgNTAgNTAiIHRvPSIzNjAgNTAgNTAiIHJlcGVhdGNvdW50PSJpbmRlZmluaXRlIj48L2FuaW1hdGV0cmFuc2Zvcm0+PC9wYXRoPjwvc3ZnPg==" class="jetpack-spinner" /></span>

</div>

</div>

<div class="jp-carousel-overlay" style="display: none;">

<div class="jp-carousel-container">

<div class="jp-carousel-wrap swiper jp-carousel-swiper-container jp-carousel-transitions" itemscope="" itemtype="https://schema.org/ImageGallery">

<div class="jp-carousel swiper-wrapper">

</div>

<div class="jp-swiper-button-prev swiper-button-prev">

![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjUiIGhlaWdodD0iMjQiIHZpZXdib3g9IjAgMCAyNSAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgICAgICAgICAgICAgICAgICAgICAgPG1hc2sgaWQ9Im1hc2tQcmV2IiBtYXNrLXR5cGU9ImFscGhhIiBtYXNrdW5pdHM9InVzZXJTcGFjZU9uVXNlIiB4PSI4IiB5PSI2IiB3aWR0aD0iOSIgaGVpZ2h0PSIxMiI+CiAgICAgICAgICAgICAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMTYuMjA3MiAxNi41OUwxMS42NDk2IDEyTDE2LjIwNzIgNy40MUwxNC44MDQxIDZMOC44MzM1IDEyTDE0LjgwNDEgMThMMTYuMjA3MiAxNi41OVoiIGZpbGw9IndoaXRlIj48L3BhdGg+CiAgICAgICAgICAgICAgICAgICAgICAgIDwvbWFzaz4KICAgICAgICAgICAgICAgICAgICAgICAgPGcgbWFzaz0idXJsKCNtYXNrUHJldikiPgogICAgICAgICAgICAgICAgICAgICAgICAgICAgPHJlY3QgeD0iMC41NzkxMDIiIHdpZHRoPSIyMy44ODIzIiBoZWlnaHQ9IjI0IiBmaWxsPSIjRkZGRkZGIj48L3JlY3Q+CiAgICAgICAgICAgICAgICAgICAgICAgIDwvZz4KICAgICAgICAgICAgICAgICAgICA8L3N2Zz4=)

</div>

<div class="jp-swiper-button-next swiper-button-next">

![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjUiIGhlaWdodD0iMjQiIHZpZXdib3g9IjAgMCAyNSAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgICAgICAgICAgICAgICAgICAgICAgPG1hc2sgaWQ9Im1hc2tOZXh0IiBtYXNrLXR5cGU9ImFscGhhIiBtYXNrdW5pdHM9InVzZXJTcGFjZU9uVXNlIiB4PSI4IiB5PSI2IiB3aWR0aD0iOCIgaGVpZ2h0PSIxMiI+CiAgICAgICAgICAgICAgICAgICAgICAgICAgICA8cGF0aCBkPSJNOC41OTgxNCAxNi41OUwxMy4xNTU3IDEyTDguNTk4MTQgNy40MUwxMC4wMDEyIDZMMTUuOTcxOCAxMkwxMC4wMDEyIDE4TDguNTk4MTQgMTYuNTlaIiBmaWxsPSJ3aGl0ZSI+PC9wYXRoPgogICAgICAgICAgICAgICAgICAgICAgICA8L21hc2s+CiAgICAgICAgICAgICAgICAgICAgICAgIDxnIG1hc2s9InVybCgjbWFza05leHQpIj4KICAgICAgICAgICAgICAgICAgICAgICAgICAgIDxyZWN0IHg9IjAuMzQzNzUiIHdpZHRoPSIyMy44ODIyIiBoZWlnaHQ9IjI0IiBmaWxsPSIjRkZGRkZGIj48L3JlY3Q+CiAgICAgICAgICAgICAgICAgICAgICAgIDwvZz4KICAgICAgICAgICAgICAgICAgICA8L3N2Zz4=)

</div>

</div>

<div class="jp-carousel-close-hint">

![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjUiIGhlaWdodD0iMjQiIHZpZXdib3g9IjAgMCAyNSAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgICAgICAgICAgICAgICAgICA8bWFzayBpZD0ibWFza0Nsb3NlIiBtYXNrLXR5cGU9ImFscGhhIiBtYXNrdW5pdHM9InVzZXJTcGFjZU9uVXNlIiB4PSI1IiB5PSI1IiB3aWR0aD0iMTUiIGhlaWdodD0iMTQiPgogICAgICAgICAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMTkuMzE2NiA2LjQxTDE3LjkxMzUgNUwxMi4zNTA5IDEwLjU5TDYuNzg4MzQgNUw1LjM4NTI1IDYuNDFMMTAuOTQ3OCAxMkw1LjM4NTI1IDE3LjU5TDYuNzg4MzQgMTlMMTIuMzUwOSAxMy40MUwxNy45MTM1IDE5TDE5LjMxNjYgMTcuNTlMMTMuNzU0IDEyTDE5LjMxNjYgNi40MVoiIGZpbGw9IndoaXRlIj48L3BhdGg+CiAgICAgICAgICAgICAgICAgICAgPC9tYXNrPgogICAgICAgICAgICAgICAgICAgIDxnIG1hc2s9InVybCgjbWFza0Nsb3NlKSI+CiAgICAgICAgICAgICAgICAgICAgICAgIDxyZWN0IHg9IjAuNDA5NjY4IiB3aWR0aD0iMjMuODgyMyIgaGVpZ2h0PSIyNCIgZmlsbD0iI0ZGRkZGRiI+PC9yZWN0PgogICAgICAgICAgICAgICAgICAgIDwvZz4KICAgICAgICAgICAgICAgIDwvc3ZnPg==)

</div>

<div class="jp-carousel-info">

<div class="jp-carousel-info-footer">

<div class="jp-carousel-pagination-container">

<div class="jp-swiper-pagination swiper-pagination">

</div>

<div class="jp-carousel-pagination">

</div>

</div>

<div class="jp-carousel-photo-title-container">

## 

</div>

<div class="jp-carousel-photo-icons-container">

<a href="#" class="jp-carousel-icon-btn jp-carousel-icon-info" aria-label="Toggle photo metadata visibility"><span class="jp-carousel-icon"> <img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjUiIGhlaWdodD0iMjQiIHZpZXdib3g9IjAgMCAyNSAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgPG1hc2sgaWQ9Im1hc2tJbmZvIiBtYXNrLXR5cGU9ImFscGhhIiBtYXNrdW5pdHM9InVzZXJTcGFjZU9uVXNlIiB4PSIyIiB5PSIyIiB3aWR0aD0iMjEiIGhlaWdodD0iMjAiPgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgPHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBjbGlwLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik0xMi43NTM3IDJDNy4yNjA3NiAyIDIuODAyNzMgNi40OCAyLjgwMjczIDEyQzIuODAyNzMgMTcuNTIgNy4yNjA3NiAyMiAxMi43NTM3IDIyQzE4LjI0NjYgMjIgMjIuNzA0NiAxNy41MiAyMi43MDQ2IDEyQzIyLjcwNDYgNi40OCAxOC4yNDY2IDIgMTIuNzUzNyAyWk0xMS43NTg2IDdWOUgxMy43NDg4VjdIMTEuNzU4NlpNMTEuNzU4NiAxMVYxN0gxMy43NDg4VjExSDExLjc1ODZaTTQuNzkyOTIgMTJDNC43OTI5MiAxNi40MSA4LjM2NTMxIDIwIDEyLjc1MzcgMjBDMTcuMTQyIDIwIDIwLjcxNDQgMTYuNDEgMjAuNzE0NCAxMkMyMC43MTQ0IDcuNTkgMTcuMTQyIDQgMTIuNzUzNyA0QzguMzY1MzEgNCA0Ljc5MjkyIDcuNTkgNC43OTI5MiAxMloiIGZpbGw9IndoaXRlIj48L3BhdGg+CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIDwvbWFzaz4KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgPGcgbWFzaz0idXJsKCNtYXNrSW5mbykiPgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgPHJlY3QgeD0iMC44MTI1IiB3aWR0aD0iMjMuODgyMyIgaGVpZ2h0PSIyNCIgZmlsbD0iI0ZGRkZGRiI+PC9yZWN0PgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICA8L2c+CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgPC9zdmc+" /> </span></a> <a href="#" class="jp-carousel-icon-btn jp-carousel-icon-comments" aria-label="Toggle photo comments visibility"><span class="jp-carousel-icon"> <img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjUiIGhlaWdodD0iMjQiIHZpZXdib3g9IjAgMCAyNSAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgPG1hc2sgaWQ9Im1hc2tDb21tZW50cyIgbWFzay10eXBlPSJhbHBoYSIgbWFza3VuaXRzPSJ1c2VyU3BhY2VPblVzZSIgeD0iMiIgeT0iMiIgd2lkdGg9IjIxIiBoZWlnaHQ9IjIwIj4KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIDxwYXRoIGZpbGwtcnVsZT0iZXZlbm9kZCIgY2xpcC1ydWxlPSJldmVub2RkIiBkPSJNNC4zMjcxIDJIMjAuMjQ4NkMyMS4zNDMyIDIgMjIuMjM4OCAyLjkgMjIuMjM4OCA0VjE2QzIyLjIzODggMTcuMSAyMS4zNDMyIDE4IDIwLjI0ODYgMThINi4zMTcyOUwyLjMzNjkxIDIyVjRDMi4zMzY5MSAyLjkgMy4yMzI1IDIgNC4zMjcxIDJaTTYuMzE3MjkgMTZIMjAuMjQ4NlY0SDQuMzI3MVYxOEw2LjMxNzI5IDE2WiIgZmlsbD0id2hpdGUiPjwvcGF0aD4KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgPC9tYXNrPgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICA8ZyBtYXNrPSJ1cmwoI21hc2tDb21tZW50cykiPgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgPHJlY3QgeD0iMC4zNDY2OCIgd2lkdGg9IjIzLjg4MjMiIGhlaWdodD0iMjQiIGZpbGw9IiNGRkZGRkYiPjwvcmVjdD4KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgPC9nPgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIDwvc3ZnPg==" /> <span class="jp-carousel-has-comments-indicator" aria-label="This image has comments."></span> </span></a>

</div>

</div>

<div class="jp-carousel-info-extra">

<div class="jp-carousel-info-content-wrapper">

<div class="jp-carousel-photo-title-container">

## 

</div>

<div class="jp-carousel-comments-wrapper">

<div id="jp-carousel-comments-loading">

Loading Comments...

</div>

<div class="jp-carousel-comments">

</div>

<div id="jp-carousel-comment-form-container">

<span id="jp-carousel-comment-form-spinner"><img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iamV0cGFjay1zcGlubmVyIiB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHZpZXdib3g9IjAgMCAxMDAgMTAwIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGFyaWEtaGlkZGVuPSJ0cnVlIiBmb2N1c2FibGU9ImZhbHNlIj48Y2lyY2xlIGN4PSI1MCIgY3k9IjUwIiByPSI0NiIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjZGRkIiBzdHJva2Utd2lkdGg9IjgiPjwvY2lyY2xlPjxwYXRoIGQ9Ik0gNTAgNCBBIDQ2IDQ2IDAgMCAxIDk2IDUwIiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSI4IiBzdHJva2UtbGluZWNhcD0icm91bmQiPjxhbmltYXRldHJhbnNmb3JtIGF0dHJpYnV0ZW5hbWU9InRyYW5zZm9ybSIgdHlwZT0icm90YXRlIiBkdXI9IjEuNHMiIGZyb209IjAgNTAgNTAiIHRvPSIzNjAgNTAgNTAiIHJlcGVhdGNvdW50PSJpbmRlZmluaXRlIj48L2FuaW1hdGV0cmFuc2Zvcm0+PC9wYXRoPjwvc3ZnPg==" class="jetpack-spinner" /></span>

<div id="jp-carousel-comment-post-results">

</div>

Write a Comment...

<div id="jp-carousel-comment-form-submit-and-info-wrapper">

<div id="jp-carousel-comment-form-commenting-as">

Email

Name

Website

</div>

</div>

</div>

</div>

<div class="jp-carousel-image-meta">

<div class="jp-carousel-title-and-caption">

<div class="jp-carousel-photo-info">

### 

</div>

<div class="jp-carousel-photo-description">

</div>

</div>

<a href="#" class="jp-carousel-image-download" target="_blank" style="display: none;"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjUiIGhlaWdodD0iMjQiIHZpZXdib3g9IjAgMCAyNSAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgPG1hc2sgaWQ9Im1hc2swIiBtYXNrLXR5cGU9ImFscGhhIiBtYXNrdW5pdHM9InVzZXJTcGFjZU9uVXNlIiB4PSIzIiB5PSIzIiB3aWR0aD0iMTkiIGhlaWdodD0iMTgiPgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgPHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBjbGlwLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik01Ljg0NjE1IDVWMTlIMTkuNzc3NVYxMkgyMS43Njc3VjE5QzIxLjc2NzcgMjAuMSAyMC44NzIxIDIxIDE5Ljc3NzUgMjFINS44NDYxNUM0Ljc0MTU5IDIxIDMuODU1OTYgMjAuMSAzLjg1NTk2IDE5VjVDMy44NTU5NiAzLjkgNC43NDE1OSAzIDUuODQ2MTUgM0gxMi44MTE4VjVINS44NDYxNVpNMTQuODAyIDVWM0gyMS43Njc3VjEwSDE5Ljc3NzVWNi40MUw5Ljk5NTY5IDE2LjI0TDguNTkyNjEgMTQuODNMMTguMzc0NCA1SDE0LjgwMloiIGZpbGw9IndoaXRlIj48L3BhdGg+CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIDwvbWFzaz4KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgPGcgbWFzaz0idXJsKCNtYXNrMCkiPgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgPHJlY3QgeD0iMC44NzA2MDUiIHdpZHRoPSIyMy44ODIzIiBoZWlnaHQ9IjI0IiBmaWxsPSIjRkZGRkZGIj48L3JlY3Q+CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIDwvZz4KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICA8L3N2Zz4=" /> <span class="jp-carousel-download-text"></span></a>

<div class="jp-carousel-image-map" style="display: none;">

</div>

</div>

</div>

</div>

</div>

</div>

</div>

<div class="iframe">

</div>

<div id="likes-other-gravatars" role="dialog" aria-hidden="true" tabindex="-1">

<div class="likes-text">

%d

</div>

</div>
