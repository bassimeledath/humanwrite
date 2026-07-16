<!--
Source: https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/
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

# Was my \$48K GPU server worth it?

</div>

<div class="h1 { font-family: 'Roboto', sans-serif; } wp-block-template-part">

<div class="wp-block-group is-nowrap is-layout-flex wp-container-core-group-is-layout-bf432786 wp-block-group-is-layout-flex">

</div>

</div>

</div>

<div class="wp-block-spacer" style="height:5px" aria-hidden="true">

</div>

<div class="entry-content wp-block-post-content is-layout-flow wp-block-post-content-is-layout-flow">

<figure class="wp-block-image size-large">
<img src="https://i0.wp.com/rosmine.ai/wp-content/uploads/2024/06/img_6161.jpg?resize=4032%2C3024&amp;ssl=1" class="wp-image-154" data-recalc-dims="1" data-fetchpriority="high" decoding="async" data-attachment-id="154" data-permalink="https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/img_6161/" data-orig-file="https://i0.wp.com/rosmine.ai/wp-content/uploads/2024/06/img_6161.jpg?fit=4032%2C3024&amp;ssl=1" data-orig-size="4032,3024" data-comments-opened="1" data-image-meta="{&quot;aperture&quot;:&quot;0&quot;,&quot;credit&quot;:&quot;&quot;,&quot;camera&quot;:&quot;&quot;,&quot;caption&quot;:&quot;&quot;,&quot;created_timestamp&quot;:&quot;0&quot;,&quot;copyright&quot;:&quot;&quot;,&quot;focal_length&quot;:&quot;0&quot;,&quot;iso&quot;:&quot;0&quot;,&quot;shutter_speed&quot;:&quot;0&quot;,&quot;title&quot;:&quot;&quot;,&quot;orientation&quot;:&quot;0&quot;}" data-image-title="IMG_6161" data-image-description="" data-image-caption="" data-large-file="https://i0.wp.com/rosmine.ai/wp-content/uploads/2024/06/img_6161.jpg?fit=1024%2C768&amp;ssl=1" width="4032" height="3024" />
</figure>

In 2024 I quit my FAANG job to become an independent researcher. To do this I needed GPUs, so I built “grumbl”, a 6x 6000 Ada GPU server.

This blog describes the build, some of the issues I faced, and answers the question “was it worth it to build the server myself, or should I have rented cloud GPUs?”

(It’s called “grumbl” because apparently I cannot spell “GPUs”)

<figure class="wp-block-image size-large">
<img src="https://i0.wp.com/rosmine.ai/wp-content/uploads/2024/06/gruml_name-4.png?resize=684%2C311&amp;ssl=1" class="wp-image-94" data-recalc-dims="1" decoding="async" data-attachment-id="94" data-permalink="https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/gruml_name-4/" data-orig-file="https://i0.wp.com/rosmine.ai/wp-content/uploads/2024/06/gruml_name-4.png?fit=684%2C311&amp;ssl=1" data-orig-size="684,311" data-comments-opened="1" data-image-meta="{&quot;aperture&quot;:&quot;0&quot;,&quot;credit&quot;:&quot;&quot;,&quot;camera&quot;:&quot;&quot;,&quot;caption&quot;:&quot;&quot;,&quot;created_timestamp&quot;:&quot;0&quot;,&quot;copyright&quot;:&quot;&quot;,&quot;focal_length&quot;:&quot;0&quot;,&quot;iso&quot;:&quot;0&quot;,&quot;shutter_speed&quot;:&quot;0&quot;,&quot;title&quot;:&quot;&quot;,&quot;orientation&quot;:&quot;0&quot;}" data-image-title="gruml_name" data-image-description="" data-image-caption="" data-large-file="https://i0.wp.com/rosmine.ai/wp-content/uploads/2024/06/gruml_name-4.png?fit=684%2C311&amp;ssl=1" width="684" height="311" />
</figure>

## GPUs as an investment

This rig cost me \$48K. That sounds expensive, but it’s way less expensive than quitting my job. Because of the loss of income, if more powerful GPUs could help me make my work be successful just 2 months earlier than I would have with a smaller machine, then buying a more powerful server would be worth it. So I decided to buy the most powerful server that I could run in my apartment.

## Choosing the GPUs

I found Tim Dettmers’ [guide to choosing a GPU](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/) helpful. From that I narrowed it down to A100’s, H100’s or RTX 6000 Ada. A100’s don’t support FP8 and have slower inference performance than the newer GPUs, and I’m going to be doing a lot of inference (RL), so narrowed it down to 6000 Ada vs H100. Looking at the [price/throughput](https://lambdalabs.com/gpu-benchmarks) ratios of 6000 Ada vs H100 vs A100, I went with the 6000 Ada GPUs.

## Power Constraints

I live in an apartment and don’t have the option to upgrade my electrical circuits to support standard datacenter servers. 6 GPUs requires too much power for a single apartment circuit to handle, so I had to get 2 power supplies, and plug the power supplies into 2 outlets in separate circuits.

If you google “plugging a PC into multiple outlets”, you get lots of warnings that if you even consider such a setup you will instantly burst into flames. So I hired a professional PC builder make sure it was safe. This is more expensive than doing everything myself, but it’s less expensive than doing something wrong and burning down my apartment.

Ironically, after designing the entire build around apartment power constraints, I ended up moving grumbl to my parents’ basement, where I could upgrade the circuits anyway.

<figure class="wp-block-image size-large">
<img src="https://i0.wp.com/rosmine.ai/wp-content/uploads/2024/06/img_6177.jpg?resize=3024%2C4032&amp;ssl=1" class="wp-image-159" data-recalc-dims="1" decoding="async" data-attachment-id="159" data-permalink="https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/img_6177/" data-orig-file="https://i0.wp.com/rosmine.ai/wp-content/uploads/2024/06/img_6177.jpg?fit=3024%2C4032&amp;ssl=1" data-orig-size="3024,4032" data-comments-opened="1" data-image-meta="{&quot;aperture&quot;:&quot;0&quot;,&quot;credit&quot;:&quot;&quot;,&quot;camera&quot;:&quot;&quot;,&quot;caption&quot;:&quot;&quot;,&quot;created_timestamp&quot;:&quot;0&quot;,&quot;copyright&quot;:&quot;&quot;,&quot;focal_length&quot;:&quot;0&quot;,&quot;iso&quot;:&quot;0&quot;,&quot;shutter_speed&quot;:&quot;0&quot;,&quot;title&quot;:&quot;&quot;,&quot;orientation&quot;:&quot;0&quot;}" data-image-title="IMG_6177" data-image-description="" data-image-caption="" data-large-file="https://i0.wp.com/rosmine.ai/wp-content/uploads/2024/06/img_6177.jpg?fit=768%2C1024&amp;ssl=1" width="3024" height="4032" />
</figure>

## Building my own GPU server vs. using a Cloud Provider

Is it better to buy my own GPUs or should I have rented from a cloud provider? I decided to measure this by calculating how much I used the GPUs, and comparing that to how much it would’ve cost to rent equivalent compute in the cloud.

In 2024 I calculated at the then current GPU rental rates, it would take me about a year of close to 85%+ utilization to match cloud rental rates. That should be easy to do, but for a full analysis, I need to also account for electricity and the fact that as more powerful GPUs become available, the cost to rent equivalent compute will decrease.

To be thorough, I wrote a script that would log the usage of each gpu every minute. I also logged the power usage in watts so I could calculate how much I spent on electricity.

In this analysis, I only compared against on-demand pricing. There are also payment plans where you reserve the instance for 6-12 months, but those seemed not worth it to me, since they were only a little cheaper than buying the server itself, and this way I got to keep the gpus.

<figure class="wp-block-image size-large">
<img src="https://i0.wp.com/rosmine.ai/wp-content/uploads/2024/06/img_6171.jpg?resize=4032%2C3024&amp;ssl=1" class="wp-image-157" data-recalc-dims="1" loading="lazy" decoding="async" data-attachment-id="157" data-permalink="https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/img_6171/" data-orig-file="https://i0.wp.com/rosmine.ai/wp-content/uploads/2024/06/img_6171.jpg?fit=4032%2C3024&amp;ssl=1" data-orig-size="4032,3024" data-comments-opened="1" data-image-meta="{&quot;aperture&quot;:&quot;0&quot;,&quot;credit&quot;:&quot;&quot;,&quot;camera&quot;:&quot;&quot;,&quot;caption&quot;:&quot;&quot;,&quot;created_timestamp&quot;:&quot;0&quot;,&quot;copyright&quot;:&quot;&quot;,&quot;focal_length&quot;:&quot;0&quot;,&quot;iso&quot;:&quot;0&quot;,&quot;shutter_speed&quot;:&quot;0&quot;,&quot;title&quot;:&quot;&quot;,&quot;orientation&quot;:&quot;0&quot;}" data-image-title="IMG_6171" data-image-description="" data-image-caption="" data-large-file="https://i0.wp.com/rosmine.ai/wp-content/uploads/2024/06/img_6171.jpg?fit=1024%2C768&amp;ssl=1" width="4032" height="3024" />
<figcaption>Using grumbl without a monitor is wasting its potential, since it has ports for up to 24 monitors. I could make my own mini vegas sphere<br />
</figcaption>
</figure>

## GPU usage over time graph

To measure GPU usage, for each GPU I counted the number of hours each day where I used that GPU at least once. This seemed a fair comparison against rental since I wouldn’t stop and restart a cloud server if it was only going to be idle for less than an hour.

This comparison is generous to cloud renting, because it assumes I could stop and start each GPU independently. Much of the idle time I had was when I was running multiple experiments in parallel, and one finished/failed but the others kept going, and I wouldn’t have stopped the server if I was renting

Note: This is meant to be a measure of how much I use the gpus, not training efficiency, so a GPU with 10% utilization would still count as active for the hour. (My code would be equally inefficient running in the cloud)

Here is the graph of use over time:

<figure class="wp-block-image size-large">
<img src="https://i0.wp.com/rosmine.ai/wp-content/uploads/2024/06/image-7.png?resize=1024%2C608&amp;ssl=1" class="wp-image-1130" data-recalc-dims="1" loading="lazy" decoding="async" data-attachment-id="1130" data-permalink="https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/image-50/" data-orig-file="https://i0.wp.com/rosmine.ai/wp-content/uploads/2024/06/image-7.png?fit=1263%2C750&amp;ssl=1" data-orig-size="1263,750" data-comments-opened="1" data-image-meta="{&quot;aperture&quot;:&quot;0&quot;,&quot;credit&quot;:&quot;&quot;,&quot;camera&quot;:&quot;&quot;,&quot;caption&quot;:&quot;&quot;,&quot;created_timestamp&quot;:&quot;0&quot;,&quot;copyright&quot;:&quot;&quot;,&quot;focal_length&quot;:&quot;0&quot;,&quot;iso&quot;:&quot;0&quot;,&quot;shutter_speed&quot;:&quot;0&quot;,&quot;title&quot;:&quot;&quot;,&quot;orientation&quot;:&quot;0&quot;}" data-image-title="image" data-image-description="" data-image-caption="" data-large-file="https://i0.wp.com/rosmine.ai/wp-content/uploads/2024/06/image-7.png?fit=1024%2C608&amp;ssl=1" srcset="https://i0.wp.com/rosmine.ai/wp-content/uploads/2024/06/image-7.png?resize=1024%2C608&amp;ssl=1 1024w, https://i0.wp.com/rosmine.ai/wp-content/uploads/2024/06/image-7.png?resize=300%2C178&amp;ssl=1 300w, https://i0.wp.com/rosmine.ai/wp-content/uploads/2024/06/image-7.png?resize=768%2C456&amp;ssl=1 768w, https://i0.wp.com/rosmine.ai/wp-content/uploads/2024/06/image-7.png?resize=1200%2C713&amp;ssl=1 1200w, https://i0.wp.com/rosmine.ai/wp-content/uploads/2024/06/image-7.png?w=1263&amp;ssl=1 1263w" sizes="auto, (max-width: 1000px) 100vw, 1000px" width="1024" height="608" />
</figure>

You can see 3 separate times the server was down for maintenance. This is quite stressful because you don’t know if the server isn’t booting because a single PCIe riser failed, or because something went catastrophically wrong and fried all the GPUs.

In June 2025 you can see a clear increase in usage, before that I was doing smaller experiments where dev time was comparable to experiment time, so there was more down time between experiments when implementing. After June 2025, I had a project that required more compute, so I always had most GPUs continuously running experiments, and only 1-2 GPUs for dev.

From the graph, the total average use was 76%. If you calculate since 1/1/25, utilization is 85%. I have to admit, I’m a little disappointed in that. I’m running experiments 24/7, and always have a queue of more experiments to run once they finish. I thought it would easily be 95+%

## Final Calculation

To calculate money saved, the first step is to use the rental price for each day, and multiply that by the number of GPU hours used for that day, and add it all up. I didn’t have historical provider API logs, so I estimated historical pricing from timestamped references online.

Based on the Wattage records that I had logged, I calculated the electricity cost to be ~\$3000, or about \$125 per month.

Putting this all together, as of 3/13/26, I calculated rental fees for equivalent compute would have cost \$68000 so I saved a total of \$17000 so far.

Now the GPUs have paid for themselves, and based on current market rate I’m saving \$90-\$105 every day after this.

## The Real Final Calculation

The point of buying the server wasn’t to save money, it was to build something cool. I spent a long time trying high risk/high reward experiments and failing. But now I have something good. I’ve solved a major problem with LLMs. And I’m launching next Monday so we will soon see if it’s actually a breakthrough or just LLM psychosis 🙂 (UPDATE: Demo launch was a success! 400K+ views, and multiple companies reached to use my IP. Full product coming very soon)

## Advice/Other notes

- Be very careful about building your own high end server like this, it’s easy to make expensive mistakes. I thought that I could not get a standard datacenter server because my apartment wouldn’t let me upgrade the circuits, so I needed to have 2 power supplies plugged into different circuits. Because of this I got a motherboard with slow GPU interconnect. It’s good for running many small experiments in parallel (which is my main use case) but horrible for any models split across gpus.
- Several of the failures were due to riser issues, and Nathan Odle’s <a href="https://www.mov-axbx.com/wopr/wopr_risers.html" data-type="link" data-id="https://www.mov-axbx.com/wopr/wopr_risers.html">riser investigation</a> was very helpful for debugging
- I have the spending habits of a broke grad student and I’ve been saving up for this for years. I’m very lucky to be in a position where I can take questionable financial risks like this, but I wouldn’t recommend buying this rig to everyone. You can still do great work with just a Google Colab subscription or renting some cheaper cloud GPUs, or smaller personal rigs.
- The mentality shift of renting vs. owning the gpus is huge. When renting, each experiment costs money and I had to ask myself is it worth it. When owning, it feels like \*not\* running experiments is costing me money. Also, it’s so nice to not have the annoyance of constantly starting/stoping cloud instances.
- This analysis doesn’t take into account the cost of my time. Building and maintaining the server took a lot of time.
- I tried to insure it under my renter’s insurance policy. They didn’t like that. I had to get business insurance to cover it.
- If I were to do this again, I wouldn’t do a custom build like this. I would buy a standard datacenter server and rent space in a colocation center. But then I would miss saying Hi to grumbl once in a while.

## 

Questions? Comments? <a href="https://twitter.com/rosmine" data-type="link" data-id="https://twitter.com/rosmine">DM me on X</a> or E-mail me at hello@rosmine.ai

Thanks to @algomancer for sponsoring this and other work

  

<div class="sharedaddy sd-sharing-enabled">

<div class="robots-nocontent sd-block sd-social sd-social-icon-text sd-sharing">

### Share this:

<div class="sd-content">

- <a href="https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/?share=twitter" class="share-twitter sd-button share-icon" rel="nofollow noopener noreferrer" data-shared="sharing-twitter-36" target="_blank" aria-labelledby="sharing-twitter-36"><span id="sharing-twitter-36" hidden="">Share on X (Opens in new window)</span> <span>X</span></a>
- <a href="https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/?share=facebook" class="share-facebook sd-button share-icon" rel="nofollow noopener noreferrer" data-shared="sharing-facebook-36" target="_blank" aria-labelledby="sharing-facebook-36"><span id="sharing-facebook-36" hidden="">Share on Facebook (Opens in new window)</span> <span>Facebook</span></a>
- 

</div>

</div>

</div>

<div id="like-post-wrapper-234301566-36-6a53bde598252" class="sharedaddy sd-block sd-like jetpack-likes-widget-wrapper jetpack-likes-widget-unloaded" data-src="https://widgets.wp.com/likes/?ver=16.0#blog_id=234301566&amp;post_id=36&amp;origin=rosmine.ai&amp;obj_id=234301566-36-6a53bde598252" data-name="like-post-frame-234301566-36-6a53bde598252" data-title="Like or Reblog">

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

### Leave a Reply<span class="small"><a href="/2026/05/13/was-my-48k-gpu-worth-it/#respond" id="cancel-comment-reply-link" rel="nofollow" style="display:none;">Cancel reply</a></span>

</div>

<div class="wp-block-comments">

## 21 responses to “Was my \$48K GPU server worth it?”

1.  <div id="comment-26">

    <div class="wp-block-columns is-layout-flex wp-container-core-columns-is-layout-f56f613f wp-block-columns-is-layout-flex">

    <div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow" style="flex-basis:40px">

    <div class="wp-block-avatar">

    <img src="https://secure.gravatar.com/avatar/?s=40&amp;d=identicon&amp;r=g" class="avatar avatar-40 photo avatar-default wp-block-avatar__image" style="border-radius:20px;" srcset="https://secure.gravatar.com/avatar/?s=80&amp;d=identicon&amp;r=g 2x" loading="lazy" decoding="async" width="40" height="40" alt="mia gu bowie Avatar" />

    </div>

    </div>

    <div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow">

    <div class="wp-block-comment-author-name has-small-font-size">

    mia gu bowie

    </div>

    <div class="wp-block-group is-layout-flex wp-block-group-is-layout-flex" style="margin-top:0px;margin-bottom:0px">

    <div class="wp-block-comment-date has-small-font-size">

    [May 17, 2026](https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/#comment-26)

    </div>

    </div>

    <div class="wp-block-comment-content">

    I think the balenciaga bracelet that looks exactly like scotch tape is a better idea, because with 48k you can get 11 of them. Did you only get 6 GPUs?

    <div id="like-comment-wrapper-234301566-26-6a53bde59acab" class="jetpack-comment-likes-widget-wrapper jetpack-likes-widget-unloaded" data-src="https://widgets.wp.com/likes/#blog_id=234301566&amp;comment_id=26&amp;origin=rosmine.ai&amp;obj_id=234301566-26-6a53bde59acab" data-name="like-comment-frame-234301566-26-6a53bde59acab">

    <div class="likes-widget-placeholder comment-likes-widget-placeholder comment-likes">

    <span class="loading">Loading...</span>

    </div>

    <div class="comment-likes-widget jetpack-likes-widget comment-likes">

    <span class="comment-like-feedback"></span><span class="sd-text-color"></span><span class="sd-link-color"></span>

    </div>

    </div>

    </div>

    <div class="wp-block-comment-reply-link has-small-font-size">

    <a href="https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/?replytocom=26#respond" class="comment-reply-link" rel="nofollow" data-commentid="26" data-postid="36" data-belowelement="comment-26" data-respondelement="respond" data-replyto="Reply to mia gu bowie" aria-label="Reply to mia gu bowie">Reply</a>

    </div>

    </div>

    </div>

    </div>

2.  <div id="comment-27">

    <div class="wp-block-columns is-layout-flex wp-container-core-columns-is-layout-f56f613f wp-block-columns-is-layout-flex">

    <div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow" style="flex-basis:40px">

    <div class="wp-block-avatar">

    </div>

    </div>

    <div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow">

    <div class="wp-block-comment-author-name has-small-font-size">

    <a href="https://rosmine.ai/2026/05/18/fixing-llm-writing-with-distribution-fine-tuning/" rel="external nofollow ugc" target="_self">Fixing LLM writing with Distribution Fine Tuning</a>

    </div>

    <div class="wp-block-group is-layout-flex wp-block-group-is-layout-flex" style="margin-top:0px;margin-bottom:0px">

    <div class="wp-block-comment-date has-small-font-size">

    [May 18, 2026](https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/#comment-27)

    </div>

    </div>

    <div class="wp-block-comment-content">

    \[…\] These results do not require extreme compute; all training was done on my local 6x 6000 Ada server. \[…\]

    <div id="like-comment-wrapper-234301566-27-6a53bde59b709" class="jetpack-comment-likes-widget-wrapper jetpack-likes-widget-unloaded" data-src="https://widgets.wp.com/likes/#blog_id=234301566&amp;comment_id=27&amp;origin=rosmine.ai&amp;obj_id=234301566-27-6a53bde59b709" data-name="like-comment-frame-234301566-27-6a53bde59b709">

    <div class="likes-widget-placeholder comment-likes-widget-placeholder comment-likes">

    <span class="loading">Loading...</span>

    </div>

    <div class="comment-likes-widget jetpack-likes-widget comment-likes">

    <span class="comment-like-feedback"></span><span class="sd-text-color"></span><span class="sd-link-color"></span>

    </div>

    </div>

    </div>

    <div class="wp-block-comment-reply-link has-small-font-size">

    <a href="https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/?replytocom=27#respond" class="comment-reply-link" rel="nofollow" data-commentid="27" data-postid="36" data-belowelement="comment-27" data-respondelement="respond" data-replyto="Reply to Fixing LLM writing with Distribution Fine Tuning" aria-label="Reply to Fixing LLM writing with Distribution Fine Tuning">Reply</a>

    </div>

    </div>

    </div>

    </div>

3.  <div id="comment-31">

    <div class="wp-block-columns is-layout-flex wp-container-core-columns-is-layout-f56f613f wp-block-columns-is-layout-flex">

    <div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow" style="flex-basis:40px">

    <div class="wp-block-avatar">

    </div>

    </div>

    <div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow">

    <div class="wp-block-comment-author-name has-small-font-size">

    <a href="http://virtualnews.com/fixing-llm-writing-with-distribution-fine-tuning/" rel="external nofollow ugc" target="_self">Fixing LLM Writing with Distribution Fine Tuning - Virtual News</a>

    </div>

    <div class="wp-block-group is-layout-flex wp-block-group-is-layout-flex" style="margin-top:0px;margin-bottom:0px">

    <div class="wp-block-comment-date has-small-font-size">

    [May 18, 2026](https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/#comment-31)

    </div>

    </div>

    <div class="wp-block-comment-content">

    \[…\] These results do not require extreme compute; all training was done on my local 6x 6000 Ada server. \[…\]

    <div id="like-comment-wrapper-234301566-31-6a53bde59c0f4" class="jetpack-comment-likes-widget-wrapper jetpack-likes-widget-unloaded" data-src="https://widgets.wp.com/likes/#blog_id=234301566&amp;comment_id=31&amp;origin=rosmine.ai&amp;obj_id=234301566-31-6a53bde59c0f4" data-name="like-comment-frame-234301566-31-6a53bde59c0f4">

    <div class="likes-widget-placeholder comment-likes-widget-placeholder comment-likes">

    <span class="loading">Loading...</span>

    </div>

    <div class="comment-likes-widget jetpack-likes-widget comment-likes">

    <span class="comment-like-feedback"></span><span class="sd-text-color"></span><span class="sd-link-color"></span>

    </div>

    </div>

    </div>

    <div class="wp-block-comment-reply-link has-small-font-size">

    <a href="https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/?replytocom=31#respond" class="comment-reply-link" rel="nofollow" data-commentid="31" data-postid="36" data-belowelement="comment-31" data-respondelement="respond" data-replyto="Reply to Fixing LLM Writing with Distribution Fine Tuning - Virtual News" aria-label="Reply to Fixing LLM Writing with Distribution Fine Tuning - Virtual News">Reply</a>

    </div>

    </div>

    </div>

    </div>

4.  <div id="comment-34">

    <div class="wp-block-columns is-layout-flex wp-container-core-columns-is-layout-f56f613f wp-block-columns-is-layout-flex">

    <div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow" style="flex-basis:40px">

    <div class="wp-block-avatar">

    </div>

    </div>

    <div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow">

    <div class="wp-block-comment-author-name has-small-font-size">

    <a href="https://www.chuhaix.com/hackernews-daily-2026-05-21/" rel="external nofollow ugc" target="_self">Anonymous</a>

    </div>

    <div class="wp-block-group is-layout-flex wp-block-group-is-layout-flex" style="margin-top:0px;margin-bottom:0px">

    <div class="wp-block-comment-date has-small-font-size">

    [May 21, 2026](https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/#comment-34)

    </div>

    </div>

    <div class="wp-block-comment-content">

    \[…\] 网站: rosmine.ai HN评论: \[…\]

    <div id="like-comment-wrapper-234301566-34-6a53bde59ca9b" class="jetpack-comment-likes-widget-wrapper jetpack-likes-widget-unloaded" data-src="https://widgets.wp.com/likes/#blog_id=234301566&amp;comment_id=34&amp;origin=rosmine.ai&amp;obj_id=234301566-34-6a53bde59ca9b" data-name="like-comment-frame-234301566-34-6a53bde59ca9b">

    <div class="likes-widget-placeholder comment-likes-widget-placeholder comment-likes">

    <span class="loading">Loading...</span>

    </div>

    <div class="comment-likes-widget jetpack-likes-widget comment-likes">

    <span class="comment-like-feedback"></span><span class="sd-text-color"></span><span class="sd-link-color"></span>

    </div>

    </div>

    </div>

    <div class="wp-block-comment-reply-link has-small-font-size">

    <a href="https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/?replytocom=34#respond" class="comment-reply-link" rel="nofollow" data-commentid="34" data-postid="36" data-belowelement="comment-34" data-respondelement="respond" data-replyto="Reply to Anonymous" aria-label="Reply to Anonymous">Reply</a>

    </div>

    </div>

    </div>

    </div>

5.  <div id="comment-35">

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

    [May 21, 2026](https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/#comment-35)

    </div>

    </div>

    <div class="wp-block-comment-content">

    So … the more you buy, the more you save? 😉

    <div id="like-comment-wrapper-234301566-35-6a53bde59d3ef" class="jetpack-comment-likes-widget-wrapper jetpack-likes-widget-unloaded" data-src="https://widgets.wp.com/likes/#blog_id=234301566&amp;comment_id=35&amp;origin=rosmine.ai&amp;obj_id=234301566-35-6a53bde59d3ef" data-name="like-comment-frame-234301566-35-6a53bde59d3ef">

    <div class="likes-widget-placeholder comment-likes-widget-placeholder comment-likes">

    <span class="loading">Loading...</span>

    </div>

    <div class="comment-likes-widget jetpack-likes-widget comment-likes">

    <span class="comment-like-feedback"></span><span class="sd-text-color"></span><span class="sd-link-color"></span>

    </div>

    </div>

    </div>

    <div class="wp-block-comment-reply-link has-small-font-size">

    <a href="https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/?replytocom=35#respond" class="comment-reply-link" rel="nofollow" data-commentid="35" data-postid="36" data-belowelement="comment-35" data-respondelement="respond" data-replyto="Reply to Anonymous" aria-label="Reply to Anonymous">Reply</a>

    </div>

    </div>

    </div>

    1.  <div id="comment-36">

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

        [May 21, 2026](https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/#comment-36)

        </div>

        </div>

        <div class="wp-block-comment-content">

        Ha ha please sponsor me Jensen

        <div id="like-comment-wrapper-234301566-36-6a53bde59dc9a" class="jetpack-comment-likes-widget-wrapper jetpack-likes-widget-unloaded" data-src="https://widgets.wp.com/likes/#blog_id=234301566&amp;comment_id=36&amp;origin=rosmine.ai&amp;obj_id=234301566-36-6a53bde59dc9a" data-name="like-comment-frame-234301566-36-6a53bde59dc9a">

        <div class="likes-widget-placeholder comment-likes-widget-placeholder comment-likes">

        <span class="loading">Loading...</span>

        </div>

        <div class="comment-likes-widget jetpack-likes-widget comment-likes">

        <span class="comment-like-feedback"></span><span class="sd-text-color"></span><span class="sd-link-color"></span>

        </div>

        </div>

        </div>

        <div class="wp-block-comment-reply-link has-small-font-size">

        <a href="https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/?replytocom=36#respond" class="comment-reply-link" rel="nofollow" data-commentid="36" data-postid="36" data-belowelement="comment-36" data-respondelement="respond" data-replyto="Reply to Ben Rosmine" aria-label="Reply to Ben Rosmine">Reply</a>

        </div>

        </div>

        </div>

        </div>

    </div>

6.  <div id="comment-40">

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

    [May 21, 2026](https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/#comment-40)

    </div>

    </div>

    <div class="wp-block-comment-content">

    Can you talk a little more about the power consumption including current ratings?

    <div id="like-comment-wrapper-234301566-40-6a53bde59e627" class="jetpack-comment-likes-widget-wrapper jetpack-likes-widget-unloaded" data-src="https://widgets.wp.com/likes/#blog_id=234301566&amp;comment_id=40&amp;origin=rosmine.ai&amp;obj_id=234301566-40-6a53bde59e627" data-name="like-comment-frame-234301566-40-6a53bde59e627">

    <div class="likes-widget-placeholder comment-likes-widget-placeholder comment-likes">

    <span class="loading">Loading...</span>

    </div>

    <div class="comment-likes-widget jetpack-likes-widget comment-likes">

    <span class="comment-like-feedback"></span><span class="sd-text-color"></span><span class="sd-link-color"></span>

    </div>

    </div>

    </div>

    <div class="wp-block-comment-reply-link has-small-font-size">

    <a href="https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/?replytocom=40#respond" class="comment-reply-link" rel="nofollow" data-commentid="40" data-postid="36" data-belowelement="comment-40" data-respondelement="respond" data-replyto="Reply to Anonymous" aria-label="Reply to Anonymous">Reply</a>

    </div>

    </div>

    </div>

    </div>

7.  <div id="comment-41">

    <div class="wp-block-columns is-layout-flex wp-container-core-columns-is-layout-f56f613f wp-block-columns-is-layout-flex">

    <div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow" style="flex-basis:40px">

    <div class="wp-block-avatar">

    </div>

    </div>

    <div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow">

    <div class="wp-block-comment-author-name has-small-font-size">

    <a href="https://cybermediacreations.com/was-my-48k-gpu-server-worth-it/" rel="external nofollow ugc" target="_self">Was my $48K GPU server worth it? - Cyber Media Creations</a>

    </div>

    <div class="wp-block-group is-layout-flex wp-block-group-is-layout-flex" style="margin-top:0px;margin-bottom:0px">

    <div class="wp-block-comment-date has-small-font-size">

    [May 21, 2026](https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/#comment-41)

    </div>

    </div>

    <div class="wp-block-comment-content">

    \[…\] Source: Hacker News \[…\]

    <div id="like-comment-wrapper-234301566-41-6a53bde59ef22" class="jetpack-comment-likes-widget-wrapper jetpack-likes-widget-unloaded" data-src="https://widgets.wp.com/likes/#blog_id=234301566&amp;comment_id=41&amp;origin=rosmine.ai&amp;obj_id=234301566-41-6a53bde59ef22" data-name="like-comment-frame-234301566-41-6a53bde59ef22">

    <div class="likes-widget-placeholder comment-likes-widget-placeholder comment-likes">

    <span class="loading">Loading...</span>

    </div>

    <div class="comment-likes-widget jetpack-likes-widget comment-likes">

    <span class="comment-like-feedback"></span><span class="sd-text-color"></span><span class="sd-link-color"></span>

    </div>

    </div>

    </div>

    <div class="wp-block-comment-reply-link has-small-font-size">

    <a href="https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/?replytocom=41#respond" class="comment-reply-link" rel="nofollow" data-commentid="41" data-postid="36" data-belowelement="comment-41" data-respondelement="respond" data-replyto="Reply to Was my $48K GPU server worth it? - Cyber Media Creations" aria-label="Reply to Was my $48K GPU server worth it? - Cyber Media Creations">Reply</a>

    </div>

    </div>

    </div>

    </div>

8.  <div id="comment-42">

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

    [May 21, 2026](https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/#comment-42)

    </div>

    </div>

    <div class="wp-block-comment-content">

    V cool  
    -Chai

    <div id="like-comment-wrapper-234301566-42-6a53bde59f874" class="jetpack-comment-likes-widget-wrapper jetpack-likes-widget-unloaded" data-src="https://widgets.wp.com/likes/#blog_id=234301566&amp;comment_id=42&amp;origin=rosmine.ai&amp;obj_id=234301566-42-6a53bde59f874" data-name="like-comment-frame-234301566-42-6a53bde59f874">

    <div class="likes-widget-placeholder comment-likes-widget-placeholder comment-likes">

    <span class="loading">Loading...</span>

    </div>

    <div class="comment-likes-widget jetpack-likes-widget comment-likes">

    <span class="comment-like-feedback"></span><span class="sd-text-color"></span><span class="sd-link-color"></span>

    </div>

    </div>

    </div>

    <div class="wp-block-comment-reply-link has-small-font-size">

    <a href="https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/?replytocom=42#respond" class="comment-reply-link" rel="nofollow" data-commentid="42" data-postid="36" data-belowelement="comment-42" data-respondelement="respond" data-replyto="Reply to Anonymous" aria-label="Reply to Anonymous">Reply</a>

    </div>

    </div>

    </div>

    </div>

9.  <div id="comment-43">

    <div class="wp-block-columns is-layout-flex wp-container-core-columns-is-layout-f56f613f wp-block-columns-is-layout-flex">

    <div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow" style="flex-basis:40px">

    <div class="wp-block-avatar">

    </div>

    </div>

    <div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow">

    <div class="wp-block-comment-author-name has-small-font-size">

    <a href="https://www.vccoder.com/2584/" rel="external nofollow ugc" target="_self">2026年5月22日 科技简报 VC程序员，找资源找VC程序员</a>

    </div>

    <div class="wp-block-group is-layout-flex wp-block-group-is-layout-flex" style="margin-top:0px;margin-bottom:0px">

    <div class="wp-block-comment-date has-small-font-size">

    [May 21, 2026](https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/#comment-43)

    </div>

    </div>

    <div class="wp-block-comment-content">

    \[…\] Was my \$48K GPU server worth it? —— 真实复盘：花 4.8 万美元自建 GPU 服务器的算力账本，算力租赁与自建的投入产出比分析。 \[…\]

    <div id="like-comment-wrapper-234301566-43-6a53bde5a01a8" class="jetpack-comment-likes-widget-wrapper jetpack-likes-widget-unloaded" data-src="https://widgets.wp.com/likes/#blog_id=234301566&amp;comment_id=43&amp;origin=rosmine.ai&amp;obj_id=234301566-43-6a53bde5a01a8" data-name="like-comment-frame-234301566-43-6a53bde5a01a8">

    <div class="likes-widget-placeholder comment-likes-widget-placeholder comment-likes">

    <span class="loading">Loading...</span>

    </div>

    <div class="comment-likes-widget jetpack-likes-widget comment-likes">

    <span class="comment-like-feedback"></span><span class="sd-text-color"></span><span class="sd-link-color"></span>

    </div>

    </div>

    </div>

    <div class="wp-block-comment-reply-link has-small-font-size">

    <a href="https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/?replytocom=43#respond" class="comment-reply-link" rel="nofollow" data-commentid="43" data-postid="36" data-belowelement="comment-43" data-respondelement="respond" data-replyto="Reply to 2026年5月22日 科技简报 VC程序员，找资源找VC程序员" aria-label="Reply to 2026年5月22日 科技简报 VC程序员，找资源找VC程序员">Reply</a>

    </div>

    </div>

    </div>

    </div>

10. <div id="comment-44">

    <div class="wp-block-columns is-layout-flex wp-container-core-columns-is-layout-f56f613f wp-block-columns-is-layout-flex">

    <div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow" style="flex-basis:40px">

    <div class="wp-block-avatar">

    </div>

    </div>

    <div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow">

    <div class="wp-block-comment-author-name has-small-font-size">

    <a href="https://www.leesowoon.com/developer-trends-may-22-github-trending-ai-update/" rel="external nofollow ugc" target="_self">개발자 트렌드 — 05월 22일 GitHub Trending &amp; AI 업데이트 - 소운</a>

    </div>

    <div class="wp-block-group is-layout-flex wp-block-group-is-layout-flex" style="margin-top:0px;margin-bottom:0px">

    <div class="wp-block-comment-date has-small-font-size">

    [May 21, 2026](https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/#comment-44)

    </div>

    </div>

    <div class="wp-block-comment-content">

    \[…\] Hail Mary – Stellar Navigation Chart — 685점 · 댓글 163개 Was my \$48K GPU server worth it? — 347점 · 댓글 248개 Python 3.15: features that didn't make the headlines — 343점 \[…\]

    <div id="like-comment-wrapper-234301566-44-6a53bde5a0afa" class="jetpack-comment-likes-widget-wrapper jetpack-likes-widget-unloaded" data-src="https://widgets.wp.com/likes/#blog_id=234301566&amp;comment_id=44&amp;origin=rosmine.ai&amp;obj_id=234301566-44-6a53bde5a0afa" data-name="like-comment-frame-234301566-44-6a53bde5a0afa">

    <div class="likes-widget-placeholder comment-likes-widget-placeholder comment-likes">

    <span class="loading">Loading...</span>

    </div>

    <div class="comment-likes-widget jetpack-likes-widget comment-likes">

    <span class="comment-like-feedback"></span><span class="sd-text-color"></span><span class="sd-link-color"></span>

    </div>

    </div>

    </div>

    <div class="wp-block-comment-reply-link has-small-font-size">

    <a href="https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/?replytocom=44#respond" class="comment-reply-link" rel="nofollow" data-commentid="44" data-postid="36" data-belowelement="comment-44" data-respondelement="respond" data-replyto="Reply to 개발자 트렌드 — 05월 22일 GitHub Trending &amp; AI 업데이트 - 소운" aria-label="Reply to 개발자 트렌드 — 05월 22일 GitHub Trending &amp; AI 업데이트 - 소운">Reply</a>

    </div>

    </div>

    </div>

    </div>

11. <div id="comment-45">

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

    [May 22, 2026](https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/#comment-45)

    </div>

    </div>

    <div class="wp-block-comment-content">

    Regarding renting, vast.ai would have been a more interesting comparison than top inflated cloud prices.

    <div id="like-comment-wrapper-234301566-45-6a53bde5a147c" class="jetpack-comment-likes-widget-wrapper jetpack-likes-widget-unloaded" data-src="https://widgets.wp.com/likes/#blog_id=234301566&amp;comment_id=45&amp;origin=rosmine.ai&amp;obj_id=234301566-45-6a53bde5a147c" data-name="like-comment-frame-234301566-45-6a53bde5a147c">

    <div class="likes-widget-placeholder comment-likes-widget-placeholder comment-likes">

    <span class="loading">Loading...</span>

    </div>

    <div class="comment-likes-widget jetpack-likes-widget comment-likes">

    <span class="comment-like-feedback"></span><span class="sd-text-color"></span><span class="sd-link-color"></span>

    </div>

    </div>

    </div>

    <div class="wp-block-comment-reply-link has-small-font-size">

    <a href="https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/?replytocom=45#respond" class="comment-reply-link" rel="nofollow" data-commentid="45" data-postid="36" data-belowelement="comment-45" data-respondelement="respond" data-replyto="Reply to Anonymous" aria-label="Reply to Anonymous">Reply</a>

    </div>

    </div>

    </div>

    </div>

12. <div id="comment-46">

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

    [May 22, 2026](https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/#comment-46)

    </div>

    </div>

    <div class="wp-block-comment-content">

    Nice. If you have a rooftop or area for balcony solar it would be fun to try and generate some of that electricity yourself too.

    <div id="like-comment-wrapper-234301566-46-6a53bde5a1def" class="jetpack-comment-likes-widget-wrapper jetpack-likes-widget-unloaded" data-src="https://widgets.wp.com/likes/#blog_id=234301566&amp;comment_id=46&amp;origin=rosmine.ai&amp;obj_id=234301566-46-6a53bde5a1def" data-name="like-comment-frame-234301566-46-6a53bde5a1def">

    <div class="likes-widget-placeholder comment-likes-widget-placeholder comment-likes">

    <span class="loading">Loading...</span>

    </div>

    <div class="comment-likes-widget jetpack-likes-widget comment-likes">

    <span class="comment-like-feedback"></span><span class="sd-text-color"></span><span class="sd-link-color"></span>

    </div>

    </div>

    </div>

    <div class="wp-block-comment-reply-link has-small-font-size">

    <a href="https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/?replytocom=46#respond" class="comment-reply-link" rel="nofollow" data-commentid="46" data-postid="36" data-belowelement="comment-46" data-respondelement="respond" data-replyto="Reply to Anonymous" aria-label="Reply to Anonymous">Reply</a>

    </div>

    </div>

    </div>

    </div>

13. <div id="comment-47">

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

    [May 22, 2026](https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/#comment-47)

    </div>

    </div>

    <div class="wp-block-comment-content">

    This is the way 🫡✊

    <div id="like-comment-wrapper-234301566-47-6a53bde5a288f" class="jetpack-comment-likes-widget-wrapper jetpack-likes-widget-unloaded" data-src="https://widgets.wp.com/likes/#blog_id=234301566&amp;comment_id=47&amp;origin=rosmine.ai&amp;obj_id=234301566-47-6a53bde5a288f" data-name="like-comment-frame-234301566-47-6a53bde5a288f">

    <div class="likes-widget-placeholder comment-likes-widget-placeholder comment-likes">

    <span class="loading">Loading...</span>

    </div>

    <div class="comment-likes-widget jetpack-likes-widget comment-likes">

    <span class="comment-like-feedback"></span><span class="sd-text-color"></span><span class="sd-link-color"></span>

    </div>

    </div>

    </div>

    <div class="wp-block-comment-reply-link has-small-font-size">

    <a href="https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/?replytocom=47#respond" class="comment-reply-link" rel="nofollow" data-commentid="47" data-postid="36" data-belowelement="comment-47" data-respondelement="respond" data-replyto="Reply to Anonymous" aria-label="Reply to Anonymous">Reply</a>

    </div>

    </div>

    </div>

    </div>

14. <div id="comment-48">

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

    [May 22, 2026](https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/#comment-48)

    </div>

    </div>

    <div class="wp-block-comment-content">

    test

    <div id="like-comment-wrapper-234301566-48-6a53bde5a348e" class="jetpack-comment-likes-widget-wrapper jetpack-likes-widget-unloaded" data-src="https://widgets.wp.com/likes/#blog_id=234301566&amp;comment_id=48&amp;origin=rosmine.ai&amp;obj_id=234301566-48-6a53bde5a348e" data-name="like-comment-frame-234301566-48-6a53bde5a348e">

    <div class="likes-widget-placeholder comment-likes-widget-placeholder comment-likes">

    <span class="loading">Loading...</span>

    </div>

    <div class="comment-likes-widget jetpack-likes-widget comment-likes">

    <span class="comment-like-feedback"></span><span class="sd-text-color"></span><span class="sd-link-color"></span>

    </div>

    </div>

    </div>

    <div class="wp-block-comment-reply-link has-small-font-size">

    <a href="https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/?replytocom=48#respond" class="comment-reply-link" rel="nofollow" data-commentid="48" data-postid="36" data-belowelement="comment-48" data-respondelement="respond" data-replyto="Reply to Anonymous" aria-label="Reply to Anonymous">Reply</a>

    </div>

    </div>

    </div>

    </div>

15. <div id="comment-49">

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

    [May 22, 2026](https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/#comment-49)

    </div>

    </div>

    <div class="wp-block-comment-content">

    So like, you didn’t provide your use case, and arbitrary runtime information is going to tell me what? Braggart.

    <div id="like-comment-wrapper-234301566-49-6a53bde5a4126" class="jetpack-comment-likes-widget-wrapper jetpack-likes-widget-unloaded" data-src="https://widgets.wp.com/likes/#blog_id=234301566&amp;comment_id=49&amp;origin=rosmine.ai&amp;obj_id=234301566-49-6a53bde5a4126" data-name="like-comment-frame-234301566-49-6a53bde5a4126">

    <div class="likes-widget-placeholder comment-likes-widget-placeholder comment-likes">

    <span class="loading">Loading...</span>

    </div>

    <div class="comment-likes-widget jetpack-likes-widget comment-likes">

    <span class="comment-like-feedback"></span><span class="sd-text-color"></span><span class="sd-link-color"></span>

    </div>

    </div>

    </div>

    <div class="wp-block-comment-reply-link has-small-font-size">

    <a href="https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/?replytocom=49#respond" class="comment-reply-link" rel="nofollow" data-commentid="49" data-postid="36" data-belowelement="comment-49" data-respondelement="respond" data-replyto="Reply to Anonymous" aria-label="Reply to Anonymous">Reply</a>

    </div>

    </div>

    </div>

    </div>

16. <div id="comment-50">

    <div class="wp-block-columns is-layout-flex wp-container-core-columns-is-layout-f56f613f wp-block-columns-is-layout-flex">

    <div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow" style="flex-basis:40px">

    <div class="wp-block-avatar">

    </div>

    </div>

    <div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow">

    <div class="wp-block-comment-author-name has-small-font-size">

    <a href="https://techtrendtrove.com/science-technology/was-my-48k-gpu-server-worth-it/" rel="external nofollow ugc" target="_self">Was my $48K GPU server worth it? - Tech Trend Trove</a>

    </div>

    <div class="wp-block-group is-layout-flex wp-block-group-is-layout-flex" style="margin-top:0px;margin-bottom:0px">

    <div class="wp-block-comment-date has-small-font-size">

    [May 26, 2026](https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/#comment-50)

    </div>

    </div>

    <div class="wp-block-comment-content">

    \[…\] six Ada 6000 GPUs and was designed to maximize performance within apartment power constraints. Choosing the right GPU server for AI workloads can significantly impact cost and performance. The builder, who recently left a FAANG job to pursue \[…\]

    <div id="like-comment-wrapper-234301566-50-6a53bde5a4dc8" class="jetpack-comment-likes-widget-wrapper jetpack-likes-widget-unloaded" data-src="https://widgets.wp.com/likes/#blog_id=234301566&amp;comment_id=50&amp;origin=rosmine.ai&amp;obj_id=234301566-50-6a53bde5a4dc8" data-name="like-comment-frame-234301566-50-6a53bde5a4dc8">

    <div class="likes-widget-placeholder comment-likes-widget-placeholder comment-likes">

    <span class="loading">Loading...</span>

    </div>

    <div class="comment-likes-widget jetpack-likes-widget comment-likes">

    <span class="comment-like-feedback"></span><span class="sd-text-color"></span><span class="sd-link-color"></span>

    </div>

    </div>

    </div>

    <div class="wp-block-comment-reply-link has-small-font-size">

    <a href="https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/?replytocom=50#respond" class="comment-reply-link" rel="nofollow" data-commentid="50" data-postid="36" data-belowelement="comment-50" data-respondelement="respond" data-replyto="Reply to Was my $48K GPU server worth it? - Tech Trend Trove" aria-label="Reply to Was my $48K GPU server worth it? - Tech Trend Trove">Reply</a>

    </div>

    </div>

    </div>

    </div>

17. <div id="comment-51">

    <div class="wp-block-columns is-layout-flex wp-container-core-columns-is-layout-f56f613f wp-block-columns-is-layout-flex">

    <div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow" style="flex-basis:40px">

    <div class="wp-block-avatar">

    </div>

    </div>

    <div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow">

    <div class="wp-block-comment-author-name has-small-font-size">

    <a href="https://geeksalad.org/was-my-48k-gpu-server-worth-it/" rel="external nofollow ugc" target="_self">Was my $48K GPU server worth it? - Geek Salad</a>

    </div>

    <div class="wp-block-group is-layout-flex wp-block-group-is-layout-flex" style="margin-top:0px;margin-bottom:0px">

    <div class="wp-block-comment-date has-small-font-size">

    [May 26, 2026](https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/#comment-51)

    </div>

    </div>

    <div class="wp-block-comment-content">

    \[…\] built a six-GPU server using Nvidia RTX 6000 Ada GPUs to support AI experiments, particularly reinforcement learning inference tasks. The total cost was \$48,000, which included specialized power supplies and professional setup due \[…\]

    <div id="like-comment-wrapper-234301566-51-6a53bde5a5a0b" class="jetpack-comment-likes-widget-wrapper jetpack-likes-widget-unloaded" data-src="https://widgets.wp.com/likes/#blog_id=234301566&amp;comment_id=51&amp;origin=rosmine.ai&amp;obj_id=234301566-51-6a53bde5a5a0b" data-name="like-comment-frame-234301566-51-6a53bde5a5a0b">

    <div class="likes-widget-placeholder comment-likes-widget-placeholder comment-likes">

    <span class="loading">Loading...</span>

    </div>

    <div class="comment-likes-widget jetpack-likes-widget comment-likes">

    <span class="comment-like-feedback"></span><span class="sd-text-color"></span><span class="sd-link-color"></span>

    </div>

    </div>

    </div>

    <div class="wp-block-comment-reply-link has-small-font-size">

    <a href="https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/?replytocom=51#respond" class="comment-reply-link" rel="nofollow" data-commentid="51" data-postid="36" data-belowelement="comment-51" data-respondelement="respond" data-replyto="Reply to Was my $48K GPU server worth it? - Geek Salad" aria-label="Reply to Was my $48K GPU server worth it? - Geek Salad">Reply</a>

    </div>

    </div>

    </div>

    </div>

18. <div id="comment-52">

    <div class="wp-block-columns is-layout-flex wp-container-core-columns-is-layout-f56f613f wp-block-columns-is-layout-flex">

    <div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow" style="flex-basis:40px">

    <div class="wp-block-avatar">

    </div>

    </div>

    <div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow">

    <div class="wp-block-comment-author-name has-small-font-size">

    <a href="https://theideamagazine.com/business-industry/was-my-48k-gpu-server-worth-it/" rel="external nofollow ugc" target="_self">Was my $48K GPU server worth it? - The Idea Magazine</a>

    </div>

    <div class="wp-block-group is-layout-flex wp-block-group-is-layout-flex" style="margin-top:0px;margin-bottom:0px">

    <div class="wp-block-comment-date has-small-font-size">

    [May 26, 2026](https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/#comment-52)

    </div>

    </div>

    <div class="wp-block-comment-content">

    \[…\] Ada 6000 GPUs and was designed to meet the researcher’s AI inference and experimentation needs. Choosing the right GPU server for AI workloads can significantly impact performance and cost-efficiency. The total cost was \$48,000, including \[…\]

    <div id="like-comment-wrapper-234301566-52-6a53bde5a6686" class="jetpack-comment-likes-widget-wrapper jetpack-likes-widget-unloaded" data-src="https://widgets.wp.com/likes/#blog_id=234301566&amp;comment_id=52&amp;origin=rosmine.ai&amp;obj_id=234301566-52-6a53bde5a6686" data-name="like-comment-frame-234301566-52-6a53bde5a6686">

    <div class="likes-widget-placeholder comment-likes-widget-placeholder comment-likes">

    <span class="loading">Loading...</span>

    </div>

    <div class="comment-likes-widget jetpack-likes-widget comment-likes">

    <span class="comment-like-feedback"></span><span class="sd-text-color"></span><span class="sd-link-color"></span>

    </div>

    </div>

    </div>

    <div class="wp-block-comment-reply-link has-small-font-size">

    <a href="https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/?replytocom=52#respond" class="comment-reply-link" rel="nofollow" data-commentid="52" data-postid="36" data-belowelement="comment-52" data-respondelement="respond" data-replyto="Reply to Was my $48K GPU server worth it? - The Idea Magazine" aria-label="Reply to Was my $48K GPU server worth it? - The Idea Magazine">Reply</a>

    </div>

    </div>

    </div>

    </div>

19. <div id="comment-55">

    <div class="wp-block-columns is-layout-flex wp-container-core-columns-is-layout-f56f613f wp-block-columns-is-layout-flex">

    <div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow" style="flex-basis:40px">

    <div class="wp-block-avatar">

    </div>

    </div>

    <div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow">

    <div class="wp-block-comment-author-name has-small-font-size">

    <a href="https://datapipe.app/hn-daily-brief-2026-05-23-cn/" rel="external nofollow ugc" target="_self">HN 热点评论速览 · 2026年5月23日 – DataPipe</a>

    </div>

    <div class="wp-block-group is-layout-flex wp-block-group-is-layout-flex" style="margin-top:0px;margin-bottom:0px">

    <div class="wp-block-comment-date has-small-font-size">

    [June 23, 2026](https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/#comment-55)

    </div>

    </div>

    <div class="wp-block-comment-content">

    \[…\] 原文链接：https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/ \[…\]

    <div id="like-comment-wrapper-234301566-55-6a53bde5a72e0" class="jetpack-comment-likes-widget-wrapper jetpack-likes-widget-unloaded" data-src="https://widgets.wp.com/likes/#blog_id=234301566&amp;comment_id=55&amp;origin=rosmine.ai&amp;obj_id=234301566-55-6a53bde5a72e0" data-name="like-comment-frame-234301566-55-6a53bde5a72e0">

    <div class="likes-widget-placeholder comment-likes-widget-placeholder comment-likes">

    <span class="loading">Loading...</span>

    </div>

    <div class="comment-likes-widget jetpack-likes-widget comment-likes">

    <span class="comment-like-feedback"></span><span class="sd-text-color"></span><span class="sd-link-color"></span>

    </div>

    </div>

    </div>

    <div class="wp-block-comment-reply-link has-small-font-size">

    <a href="https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/?replytocom=55#respond" class="comment-reply-link" rel="nofollow" data-commentid="55" data-postid="36" data-belowelement="comment-55" data-respondelement="respond" data-replyto="Reply to HN 热点评论速览 · 2026年5月23日 – DataPipe" aria-label="Reply to HN 热点评论速览 · 2026年5月23日 – DataPipe">Reply</a>

    </div>

    </div>

    </div>

    </div>

20. <div id="comment-59">

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

    [July 1, 2026](https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/#comment-59)

    </div>

    </div>

    <div class="wp-block-comment-content">

    Would this still be worth it now in mid-2026?

    <div id="like-comment-wrapper-234301566-59-6a53bde5a7fa9" class="jetpack-comment-likes-widget-wrapper jetpack-likes-widget-unloaded" data-src="https://widgets.wp.com/likes/#blog_id=234301566&amp;comment_id=59&amp;origin=rosmine.ai&amp;obj_id=234301566-59-6a53bde5a7fa9" data-name="like-comment-frame-234301566-59-6a53bde5a7fa9">

    <div class="likes-widget-placeholder comment-likes-widget-placeholder comment-likes">

    <span class="loading">Loading...</span>

    </div>

    <div class="comment-likes-widget jetpack-likes-widget comment-likes">

    <span class="comment-like-feedback"></span><span class="sd-text-color"></span><span class="sd-link-color"></span>

    </div>

    </div>

    </div>

    <div class="wp-block-comment-reply-link has-small-font-size">

    <a href="https://rosmine.ai/2026/05/13/was-my-48k-gpu-worth-it/comment-page-1/?replytocom=59#respond" class="comment-reply-link" rel="nofollow" data-commentid="59" data-postid="36" data-belowelement="comment-59" data-respondelement="respond" data-replyto="Reply to Anonymous" aria-label="Reply to Anonymous">Reply</a>

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
