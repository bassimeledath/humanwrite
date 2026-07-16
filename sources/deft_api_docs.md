<!--
Source: https://deftwriting.com/developers
Snapshot retrieved: 2026-07-15
-->

<div hidden="">

</div>

<div class="min-h-screen bg-mist text-ink" role="main">

<div class="flex h-16 shrink-0 items-center justify-between border-b border-hairline bg-surface px-6 lg:px-7" header-surface="app">

<a href="/" aria-label="Deft home"><span class="display-type font-bold uppercase leading-none tracking-[-0.01em] text-[23px]">DEFT</span></a>

<div class="flex items-center" auth-state="signed-out">

Sign in

</div>

</div>

<div class="mx-auto max-w-app px-6 py-10 lg:px-10 lg:py-14">

<div class="flex flex-col gap-3.5">

# API

Build against the Deft server-to-server generation API. Approval, keys, and API billing are managed from Account.

</div>

<div class="mt-9 flex flex-col gap-12 lg:flex-row lg:items-start lg:gap-14">

API

<a href="#api" class="flex items-center gap-3 rounded-control px-3 py-2.5 text-control font-medium text-stone transition-colors hover:bg-surface hover:text-ink"><img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNyIgaGVpZ2h0PSIxNyIgdmlld2JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIxLjYiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgY2xhc3M9Imx1Y2lkZSBsdWNpZGUtYm9vay1vcGVuIHRleHQtZmFpbnQiIGFyaWEtaGlkZGVuPSJ0cnVlIj48cGF0aCBkPSJNMTIgN3YxNCI+PC9wYXRoPjxwYXRoIGQ9Ik0zIDE4YTEgMSAwIDAgMS0xLTFWNGExIDEgMCAwIDEgMS0xaDVhNCA0IDAgMCAxIDQgNCA0IDQgMCAwIDEgNC00aDVhMSAxIDAgMCAxIDEgMXYxM2ExIDEgMCAwIDEtMSAxaC02YTMgMyAwIDAgMC0zIDMgMyAzIDAgMCAwLTMtM3oiPjwvcGF0aD48L3N2Zz4=" class="lucide lucide-book-open text-faint" />Reference</a><a href="#billing" class="flex items-center gap-3 rounded-control px-3 py-2.5 text-control font-medium text-stone transition-colors hover:bg-surface hover:text-ink"><img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNyIgaGVpZ2h0PSIxNyIgdmlld2JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIxLjYiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgY2xhc3M9Imx1Y2lkZSBsdWNpZGUtY2hhcnQtY29sdW1uIHRleHQtZmFpbnQiIGFyaWEtaGlkZGVuPSJ0cnVlIj48cGF0aCBkPSJNMyAzdjE2YTIgMiAwIDAgMCAyIDJoMTYiPjwvcGF0aD48cGF0aCBkPSJNMTggMTdWOSI+PC9wYXRoPjxwYXRoIGQ9Ik0xMyAxN1Y1Ij48L3BhdGg+PHBhdGggZD0iTTggMTd2LTMiPjwvcGF0aD48L3N2Zz4=" class="lucide lucide-chart-column text-faint" />Pricing</a>

<div class="flex min-w-0 flex-1 flex-col gap-6">

<div id="api" class="section scroll-mt-24 flex flex-col gap-6">

<div class="section rounded-panel border border-hairline bg-surface px-7 py-6">

<div class="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">

<div>

<div>

Endpoint

## POST /v1/generate

The API accepts a prompt and returns a complete generated draft plus usage metadata. Keys are created from the API tab in Account after approval.

</div>

</div>

<a href="/account?section=api#api" class="inline-flex items-center justify-center gap-2 rounded-control font-semibold tracking-[0.01em] transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-aubergine disabled:cursor-not-allowed h-10 px-5 text-sm border border-ink text-ink hover:bg-ink hover:text-surface disabled:opacity-50 w-fit shrink-0">Manage access<img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNSIgaGVpZ2h0PSIxNSIgdmlld2JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIxLjgiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgY2xhc3M9Imx1Y2lkZSBsdWNpZGUtYXJyb3ctcmlnaHQiIGFyaWEtaGlkZGVuPSJ0cnVlIj48cGF0aCBkPSJNNSAxMmgxNCI+PC9wYXRoPjxwYXRoIGQ9Im0xMiA1IDcgNy03IDciPjwvcGF0aD48L3N2Zz4=" class="lucide lucide-arrow-right" /></a>

</div>

<div class="mt-5 grid gap-3">

``` overflow-x-auto
curl https://deftwriting.com/v1/generate   -H "Authorization: Bearer $DEFT_API_KEY"   -H "Content-Type: application/json"   -d '{"prompt":"Write a concise launch memo for a new analytics feature."}'
```

``` overflow-x-auto
const response = await fetch("https://deftwriting.com/v1/generate", {
  method: "POST",
  headers: {
    Authorization: "Bearer " + process.env.DEFT_API_KEY,
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    prompt: "Write a concise launch memo for a new analytics feature.",
  }),
});

const result = await response.json();
```

</div>

</div>

<div class="section grid gap-4 md:grid-cols-3">

<div class="rounded-panel border border-hairline bg-surface p-5">

<div class="grid h-10 w-10 place-items-center rounded-control bg-aubergine-wash text-aubergine">

<img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxOCIgaGVpZ2h0PSIxOCIgdmlld2JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIxLjgiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgY2xhc3M9Imx1Y2lkZSBsdWNpZGUta2V5LXJvdW5kIiBhcmlhLWhpZGRlbj0idHJ1ZSI+PHBhdGggZD0iTTIuNTg2IDE3LjQxNEEyIDIgMCAwIDAgMiAxOC44MjhWMjFhMSAxIDAgMCAwIDEgMWgzYTEgMSAwIDAgMCAxLTF2LTFhMSAxIDAgMCAxIDEtMWgxYTEgMSAwIDAgMCAxLTF2LTFhMSAxIDAgMCAxIDEtMWguMTcyYTIgMiAwIDAgMCAxLjQxNC0uNTg2bC44MTQtLjgxNGE2LjUgNi41IDAgMSAwLTQtNHoiPjwvcGF0aD48Y2lyY2xlIGN4PSIxNi41IiBjeT0iNy41IiByPSIuNSIgZmlsbD0iY3VycmVudENvbG9yIj48L2NpcmNsZT48L3N2Zz4=" class="lucide lucide-key-round" />

</div>

## Authentication

Send an approved account's API key as a bearer token. API keys are separate from website sessions and must stay server-side.

</div>

<div class="rounded-panel border border-hairline bg-surface p-5">

<div class="grid h-10 w-10 place-items-center rounded-control bg-aubergine-wash text-aubergine">

<img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxOCIgaGVpZ2h0PSIxOCIgdmlld2JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIxLjgiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgY2xhc3M9Imx1Y2lkZSBsdWNpZGUtYWN0aXZpdHkiIGFyaWEtaGlkZGVuPSJ0cnVlIj48cGF0aCBkPSJNMjIgMTJoLTIuNDhhMiAyIDAgMCAwLTEuOTMgMS40NmwtMi4zNSA4LjM2YS4yNS4yNSAwIDAgMS0uNDggMEw5LjI0IDIuMThhLjI1LjI1IDAgMCAwLS40OCAwbC0yLjM1IDguMzZBMiAyIDAgMCAxIDQuNDkgMTJIMiI+PC9wYXRoPjwvc3ZnPg==" class="lucide lucide-activity" />

</div>

## Rate limits

Keys are limited to 60 requests / 60 seconds. Back off on 429 responses and retry later.

</div>

<div class="rounded-panel border border-hairline bg-surface p-5">

<div class="grid h-10 w-10 place-items-center rounded-control bg-aubergine-wash text-aubergine">

<img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxOCIgaGVpZ2h0PSIxOCIgdmlld2JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIxLjgiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgY2xhc3M9Imx1Y2lkZSBsdWNpZGUtd2FsbGV0LWNhcmRzIiBhcmlhLWhpZGRlbj0idHJ1ZSI+PHJlY3Qgd2lkdGg9IjE4IiBoZWlnaHQ9IjE4IiB4PSIzIiB5PSIzIiByeD0iMiI+PC9yZWN0PjxwYXRoIGQ9Ik0zIDlhMiAyIDAgMCAxIDItMmgxNGEyIDIgMCAwIDEgMiAyIj48L3BhdGg+PHBhdGggZD0iTTMgMTFoM2MuOCAwIDEuNi4zIDIuMS45bDEuMS45YzEuNiAxLjYgNC4xIDEuNiA1LjcgMGwxLjEtLjljLjUtLjUgMS4zLS45IDIuMS0uOUgyMSI+PC9wYXRoPjwvc3ZnPg==" class="lucide lucide-wallet-cards" />

</div>

## Billing

API usage reports preprocessing prompt tokens as input, Qwen/Qwen3-14B final-text tokens as output, and preprocessing completion tokens plus chunk-judge and post-processing tokens as thinking. Failed generations are not charged.

</div>

</div>

<div class="section rounded-panel border border-hairline bg-surface px-7 py-6">

<div>

Best practices

## Ask for complete documents

</div>

Deft was trained to produce whole samples of text from the web. Requests for full documents with the complete context and desired structure generally work best. Requesting fragments, isolated chunks, or partial continuations won't work as well as full documents.

</div>

<div class="section rounded-panel border border-hairline bg-surface px-7 py-6">

<div>

Request

## Prompt-first body

</div>

<div class="mt-4 grid gap-4 lg:grid-cols-[1fr_1fr]">

<div class="text-sm leading-6 text-muted">

Send JSON with a non-empty `prompt`. Optional `generationMode` accepts `simple`, legacy `prompt_optimizer`, or `rewrite`. In rewrite mode, `prompt` is the source text and optional `rewriteInstructions` describes how to transform it.

For best results, keep inputs below an estimated 100,000 tokens. The API can accept longer prompts, but quality may degrade as inputs grow. Prompts above an estimated 150,000 tokens are rejected before preprocessing or billing.

For generate-mode requests, optional `detailMode` accepts `strict` or `creative`. Strict mode keeps outline details grounded in facts and context from your prompt. Creative mode may invent supporting specifics to flesh out sparse, narrative, opinion, or exploratory requests. Omit `detailMode` to let Deft choose the better mode for the prompt.

Optional `style` and `styleKind` fields are accepted as preprocessing guidance. The endpoint returns a complete JSON response. Streaming, user-facing model selection, temperature controls, source-material controls, and token budgeting controls are not exposed.

</div>

``` overflow-x-auto
{
  "prompt": "Write a concise launch memo for a new analytics feature.",
  "detailMode": "strict"
}

{
  "generationMode": "rewrite",
  "prompt": "The source text to rewrite goes here.",
  "rewriteInstructions": "Make it shorter and less formal."
}
```

</div>

</div>

<div class="section rounded-panel border border-hairline bg-surface px-7 py-6">

<div>

Response

## Generated text and usage

</div>

<div class="mt-4 grid gap-4 lg:grid-cols-[1fr_1fr]">

``` overflow-x-auto
{
  "text": "The generated draft text...",
  "usage": {
    "input_tokens": 512,
    "output_tokens": 1024,
    "thinking_tokens": 180
  }
}
```

<div class="flex flex-col gap-3 text-sm leading-6 text-muted">

<img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiIgdmlld2JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIxLjgiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgY2xhc3M9Imx1Y2lkZSBsdWNpZGUtY2lyY2xlLWNoZWNrIG10LTEgc2hyaW5rLTAgdGV4dC1tb3NzIiBhcmlhLWhpZGRlbj0idHJ1ZSI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiPjwvY2lyY2xlPjxwYXRoIGQ9Im05IDEyIDIgMiA0LTQiPjwvcGF0aD48L3N2Zz4=" class="lucide lucide-circle-check mt-1 shrink-0 text-moss" /> **text** is the final Deft output.

<img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiIgdmlld2JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIxLjgiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgY2xhc3M9Imx1Y2lkZSBsdWNpZGUtY2lyY2xlLWNoZWNrIG10LTEgc2hyaW5rLTAgdGV4dC1tb3NzIiBhcmlhLWhpZGRlbj0idHJ1ZSI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiPjwvY2lyY2xlPjxwYXRoIGQ9Im05IDEyIDIgMiA0LTQiPjwvcGF0aD48L3N2Zz4=" class="lucide lucide-circle-check mt-1 shrink-0 text-moss" /> **usage** reports OpenRouter preprocessing prompt tokens, Qwen/Qwen3-14B final-output tokens, and preprocessing completion tokens plus chunk-judge and post-processing tokens reported as thinking tokens.

</div>

</div>

</div>

<div class="section rounded-panel border border-hairline bg-surface px-7 py-6">

<div>

Operational behavior

## Errors, retries, and limits

</div>

<div class="mt-5 overflow-x-auto">

| Status  | Code                                  | Action                                   |
|---------|---------------------------------------|------------------------------------------|
| 400     | invalid_json, invalid_request         | Fix the JSON body.                       |
| 401     | missing_api_key, invalid_api_key      | Send a valid bearer API key.             |
| 402     | insufficient_balance                  | Review billing from Account.             |
| 403     | api_key_revoked, api_account_disabled | Use an active key/account.               |
| 429     | rate_limited                          | Back off for 60 seconds before retrying. |
| 500/502 | generation_failed or upstream errors  | Retry the request when safe.             |

</div>

</div>

</div>

<div id="billing" class="section scroll-mt-24">

<div class="flex flex-col gap-6">

<div class="section rounded-panel border border-hairline bg-surface px-7 py-6" aria-labelledby="api-usage-terminology">

<div>

Terminology

## How Deft measures usage

</div>

Website generation  
One completed draft created in Deft's website or console. It appears in generation history and counts toward website limits.

API token  
A metering unit reported for API input, output, and thinking work. Tokens are not API keys and are not a spendable balance.

API credit  
Prepaid balance used only for server-to-server API charges. API credits do not increase website generation capacity.

Subscription capacity  
The monthly allowance of website generations included in a paid plan. It resets with the billing period and is separate from API credits.

</div>

<div class="section rounded-panel border border-hairline bg-surface px-7 py-6">

<div class="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">

<div>

<div>

API billing

## Token-based API pricing

</div>

Server-to-server API calls are billed at \$2.50 per million preprocessing input tokens and \$12 per million final output and thinking tokens, rounded up to the nearest cent. Failed generations are not charged.

Note: the calculation of input tokens also includes extra added tokens used by our prompt preprocessing script.

</div>

<a href="/account?section=api#api" class="inline-flex items-center justify-center gap-2 rounded-control font-semibold tracking-[0.01em] transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-aubergine disabled:cursor-not-allowed h-10 px-5 text-sm border border-ink text-ink hover:bg-ink hover:text-surface disabled:opacity-50 w-fit shrink-0">Manage billing<img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNSIgaGVpZ2h0PSIxNSIgdmlld2JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIxLjgiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgY2xhc3M9Imx1Y2lkZSBsdWNpZGUtYXJyb3ctcmlnaHQiIGFyaWEtaGlkZGVuPSJ0cnVlIj48cGF0aCBkPSJNNSAxMmgxNCI+PC9wYXRoPjxwYXRoIGQ9Im0xMiA1IDcgNy03IDciPjwvcGF0aD48L3N2Zz4=" class="lucide lucide-arrow-right" /></a>

</div>

<div class="mt-5 grid gap-4 md:grid-cols-3">

<div class="rounded-control border border-hairline bg-mist px-4 py-4">

Rate

\$2.50 per 1M input tokens; \$12 per 1M output and thinking tokens

</div>

<div class="rounded-control border border-hairline bg-mist px-4 py-4">

Rounding

Usage rounds up to the nearest cent.

</div>

<div class="rounded-control border border-hairline bg-mist px-4 py-4">

Failures

Failed generations are not charged.

</div>

</div>

</div>

</div>

</div>

</div>

</div>

</div>

</div>
